#!/usr/bin/env python
"""Build the stratified evaluation set: claim, concept, and identifier queries.

    python scripts/build_strata.py --out strata.json

Run after ingestion. Reads MeSH descriptors from the store and citation queries from the
mined qrels, then reports the shape of each stratum so the difference is visible rather
than assumed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from oncolens.env import load_env, local_data_dir  # noqa: E402
from oncolens.eval.strata import (  # noqa: E402
    StratifiedQuery, assert_no_descriptor_leakage, concept_queries, identifier_queries,
    summarize,
)


def load_doc_descriptors() -> dict[str, list[tuple[str, bool]]]:
    """doc_id -> [(descriptor, is_major)].

    ``documents.descriptors`` is a flat TEXT[]; major-topic flags are carried in the
    stored meta where available. When the flag is absent every descriptor is treated as
    minor, which understates the grade rather than inventing one.
    """
    import psycopg

    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("POSTGRES_URL / DATABASE_URL not set")
    out: dict[str, list[tuple[str, bool]]] = {}
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT doc_id, descriptors, meta FROM documents")
        for doc_id, descs, meta in cur.fetchall():
            majors = set()
            if isinstance(meta, dict):
                for m in (meta.get("mesh") or []):
                    if isinstance(m, dict) and m.get("major"):
                        majors.add(m.get("descriptor"))
            rows = []
            for d in (descs or []):
                clean = d.replace("MESH:", "").strip()
                rows.append((clean, clean in majors))
            if rows:
                out[doc_id] = rows
    return out


def load_claim_queries() -> list[StratifiedQuery]:
    path = local_data_dir() / "qrels_citation.json"
    if not path.exists():
        return []
    d = json.loads(path.read_text(encoding="utf-8"))
    exclude = d.get("exclude", {})
    return [
        StratifiedQuery(query_id=qid, query=q, stratum="claim",
                        judgments=d["qrels"].get(qid, {}), exclude_doc=exclude.get(qid))
        for qid, q in d["queries"].items()
    ]


def sample_texts(n: int = 200) -> list[str]:
    import psycopg

    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT COALESCE(indexed_text, text) FROM chunks LIMIT %s", (n,))
        return [r[0] for r in cur.fetchall()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    load_env()

    claims = load_claim_queries()
    print(f"claim stratum:      {len(claims)} queries (citation contexts)")

    descs = load_doc_descriptors()
    print(f"documents with MeSH: {len(descs)}")
    concepts = concept_queries(descs)
    print(f"concept stratum:    {len(concepts)} queries (MeSH descriptors)")

    idents = identifier_queries(claims)
    print(f"identifier stratum: {len(idents)} queries (literals mined from claims)")

    # Synthesis: R&D questions with expert-curated answer SETS, from review sections.
    # Built separately because it needs the JATS cache, not the store.
    synth_path = local_data_dir() / "qrels_synthesis.json"
    synth: list[StratifiedQuery] = []
    if synth_path.exists():
        sd = json.loads(synth_path.read_text(encoding="utf-8"))
        for qid, q in sd["queries"].items():
            synth.append(StratifiedQuery(
                query_id=qid, query=q, stratum="synthesis",
                judgments=sd["qrels"].get(qid, {}),
                exclude_doc=sd.get("exclude", {}).get(qid),
                note=sd.get("notes", {}).get(qid, "")))
        print(f"synthesis stratum:  {len(synth)} queries (review sections)")
    else:
        print("synthesis stratum:  0 — run scripts/build_synthesis_labels.py first")

    # Guard the artifact, not the accessor.
    try:
        assert_no_descriptor_leakage(sample_texts(), [c.query for c in concepts])
        print("leakage check:      PASS (no MESH: markers in indexed text)")
    except AssertionError as e:
        print(f"leakage check:      FAIL — {e}")
        return 1

    allq = claims + concepts + idents + synth
    s = summarize(allq)
    print("\n" + "=" * 66)
    print(f"{'stratum':<14}{'queries':>9}{'median words':>15}{'median rel docs':>18}")
    print("-" * 66)
    for name in ("synthesis", "concept", "identifier", "claim"):
        if name in s:
            r = s[name]
            print(f"{name:<14}{r['queries']:>9}{r['median_words']:>15}"
                  f"{r['median_relevant_docs']:>18}")
    print("\nThe point of the table: the claim stratum is 27-word sentences and the")
    print("concept stratum is 2-4 word terms. Users type the latter. A system tuned")
    print("only on the former is tuned for a query shape it will never see.")

    print("\nsample concept queries:")
    for c in concepts[:6]:
        print(f"  {c.query:<44} {c.note}")
    print("\nsample identifier queries:")
    for c in idents[:6]:
        print(f"  {c.query:<20} <- {c.note[:70]}")

    if args.out or True:
        out = Path(args.out) if args.out else local_data_dir() / "strata.json"
        out.write_text(json.dumps({
            "queries": {q.query_id: q.query for q in allq},
            "qrels": {q.query_id: q.judgments for q in allq},
            "strata": {q.query_id: q.stratum for q in allq},
            "exclude": {q.query_id: q.exclude_doc for q in allq if q.exclude_doc},
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
