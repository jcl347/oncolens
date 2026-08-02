#!/usr/bin/env python
"""Mine relevance labels from citation contexts in the ingested corpus.

Reads the documents already in Postgres, fetches each one's JATS XML, extracts every
sentence that carries an in-text citation, resolves the cited works to PMIDs, and keeps
those that resolve to a document we also hold. The result is a query set where each query
is a real technical claim written by a domain expert and each judgment is that expert's own
attribution.

Also reports the **snowball candidates**: cited PMIDs that are *not* in the corpus. That
number is the argument for expanding the corpus along its own citation graph — every one of
those is a label we could have had.

    python scripts/build_citation_labels.py --email you@example.com
    python scripts/build_citation_labels.py --snowball-out pmids.txt   # feed ingestion
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from oncolens.env import load_env, local_data_dir  # noqa: E402
from oncolens.eval.citation_labels import CitationQrels, build_labels  # noqa: E402
from oncolens.sources import jats  # noqa: E402


def corpus_documents() -> list[tuple[str, str | None]]:
    """(doc_id, pmcid) for every ingested document."""
    import psycopg

    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("POSTGRES_URL / DATABASE_URL not set")
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT doc_id, meta->>'pmcid' FROM documents ORDER BY doc_id")
        return [(r[0], r[1]) for r in cur.fetchall()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", default="oncolens@example.com")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = all documents")
    ap.add_argument("--out", default=None, help="write qrels JSON here")
    ap.add_argument("--snowball-out", default=None,
                    help="write cited-but-missing PMIDs here, one per line")
    ap.add_argument("--upload", action="store_true", help="also write qrels to Vercel Blob")
    args = ap.parse_args()

    load_env()
    cache = local_data_dir() / "jats_cache"
    cache.mkdir(parents=True, exist_ok=True)

    docs = corpus_documents()
    with_pmc = [(d, p) for d, p in docs if p]
    print(f"{len(docs)} documents in corpus, {len(with_pmc)} with a PMCID (XML available)")
    if args.limit:
        with_pmc = with_pmc[: args.limit]

    corpus_ids = {d for d, _ in docs}
    per_source: dict[str, list[tuple[str, list[str], str]]] = {}
    missing_pmids: Counter[str] = Counter()
    fetched = failed = 0

    for doc_id, pmcid in with_pmc:
        cf = cache / f"{pmcid}.xml"
        try:
            if cf.exists():
                xml = cf.read_text(encoding="utf-8", errors="replace")
            else:
                xml = jats.fetch_xml(pmcid, email=args.email, api_key=args.api_key)
                if not xml:
                    failed += 1
                    continue
                cf.write_text(xml, encoding="utf-8")
                fetched += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {pmcid}: {type(exc).__name__}: {str(exc)[:70]}")
            failed += 1
            continue

        refs = jats.parse_references(xml)
        contexts = jats.extract_citation_contexts(xml)
        rows: list[tuple[str, list[str], str]] = []
        for ctx in contexts:
            cited_doc_ids: list[str] = []
            for rid in ctx.rids:
                ref = refs.get(rid)
                if not ref or not ref.pmid:
                    continue
                target = f"PAPER:PMID{ref.pmid}"
                cited_doc_ids.append(target)
                if target not in corpus_ids:
                    missing_pmids[ref.pmid] += 1
            if cited_doc_ids:
                rows.append((ctx.sentence, cited_doc_ids, ctx.section))
        if rows:
            per_source[doc_id] = rows

    print(f"XML: {fetched} fetched, {len(with_pmc) - fetched - failed} cached, {failed} failed")
    print(f"{sum(len(v) for v in per_source.values())} citation contexts "
          f"across {len(per_source)} citing documents")

    labels, stats = build_labels(per_source, corpus_ids)

    print("\nLABEL YIELD")
    for key in ("contexts_seen", "out_of_corpus", "too_many_cocited", "not_useful_query",
                "self_citation", "duplicate_sentence", "source_cap", "target_cap",
                "labels_kept", "queries", "judgments", "distinct_targets"):
        if key in stats:
            print(f"  {key:<22}{stats[key]:>7}")

    if labels:
        grades = Counter(lb.grade for lb in labels)
        print(f"  grade distribution     {dict(sorted(grades.items(), reverse=True))}")
        print("\nSAMPLE QUERIES (these are real sentences from real papers):")
        for lb in labels[:5]:
            print(f"  [{lb.grade}] {lb.query[:110]}")
            print(f"      -> {lb.target_doc_id}  (cited by {lb.source_doc_id})")

    print(f"\nSNOWBALL: {len(missing_pmids)} distinct cited PMIDs are NOT in the corpus.")
    top = missing_pmids.most_common(10)
    if top:
        print("  most-cited missing papers (PMID x times cited):")
        print("   ", ", ".join(f"{p}x{c}" for p, c in top))
    print("  Ingesting these would both grow the corpus and convert those citations "
          "into labels.")

    if args.snowball_out:
        # Most-cited first: those convert into the most labels per article ingested.
        ordered = [p for p, _ in missing_pmids.most_common()]
        Path(args.snowball_out).write_text("\n".join(ordered), encoding="utf-8")
        print(f"\nwrote {len(ordered)} PMIDs to {args.snowball_out}")

    if labels:
        qrels = CitationQrels.from_labels(labels)
        payload = qrels.as_dict()
        out = Path(args.out) if args.out else local_data_dir() / "qrels_citation.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote qrels to {out}")
        if args.upload and os.environ.get("BLOB_READ_WRITE_TOKEN"):
            from oncolens.serve import vercel_blob
            ref = vercel_blob.put_json("qrels/citation.json", payload)
            print(f"uploaded to Blob: {ref.pathname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
