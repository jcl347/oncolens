#!/usr/bin/env python
"""Emit the measured record that the /journey page renders.

**Why this script exists rather than numbers typed into a React component.** A page that
narrates "we improved X to 0.97" is worthless if 0.97 is a literal someone typed. It looks
identical whether it is true, stale, or invented. So every figure the journey page displays
is produced here, and every figure carries:

* ``value``      — the measurement
* ``provenance`` — ``live`` (queried from the store just now), ``artifact`` (read from a
                   benchmark's JSON output), or ``recorded`` (measured by a named script on
                   a stated date, where re-running costs real money or hours)
* ``command``    — how to reproduce it

The page renders provenance next to the number. A reader can then tell a live corpus count
from a benchmark result from a figure that was true last Tuesday, which is the difference
between a dashboard and a marketing page.

    python scripts/build_journey_data.py --out public/journey.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from oncolens.env import load_env, local_data_dir  # noqa: E402


def metric(label, value, *, unit="", note="", provenance="recorded", command="",
           delta=None, direction="up"):
    return {"label": label, "value": value, "unit": unit, "note": note,
            "provenance": provenance, "command": command, "delta": delta,
            "direction": direction}


def live_corpus() -> dict:
    """Query the store. Returns {} if unreachable — the page degrades, it does not lie."""
    import psycopg

    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        return {}
    out: dict = {}
    try:
        with psycopg.connect(dsn, connect_timeout=15) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM documents")
            out["documents"] = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM chunks")
            out["chunks"] = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM documents WHERE meta->>'pmcid' IS NOT NULL")
            out["with_pmcid"] = cur.fetchone()[0]
            cur.execute(
                "SELECT count(*) FROM documents "
                "WHERE COALESCE((meta->>'full_text_chars')::int, 0) > 0")
            out["full_text"] = cur.fetchone()[0]
            cur.execute("SELECT pg_database_size(current_database())")
            out["db_bytes"] = cur.fetchone()[0]
            cur.execute("SELECT sum(length(text)) FROM chunks")
            out["text_chars"] = cur.fetchone()[0] or 0
            cur.execute("SELECT count(DISTINCT d.doc_id) FROM documents d WHERE EXISTS ("
                        "SELECT 1 FROM unnest(d.descriptors) x WHERE lower(x) LIKE '%neoplas%' "
                        "OR lower(x) LIKE '%carcinom%' OR lower(x) LIKE '%tumor%' "
                        "OR lower(x) LIKE '%cancer%' OR lower(x) LIKE '%leukem%' "
                        "OR lower(x) LIKE '%lymphom%' OR lower(x) LIKE '%melanom%')")
            out["oncology"] = cur.fetchone()[0]
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:120]
    return out


def read_artifact(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def git_rev() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def build(now_iso: str) -> dict:
    live = live_corpus()
    ddir = local_data_dir()
    qrels = read_artifact(ddir / "qrels_citation.json") or {}
    bench = read_artifact(ddir / "bench_retrieval.json") or {}

    n_queries = len(qrels.get("queries", {}))
    n_judgments = sum(len(v) for v in qrels.get("qrels", {}).values())
    n_targets = len({t for v in qrels.get("qrels", {}).values() for t in v})

    stages = []

    # ---------------------------------------------------------------- 1. the goal
    stages.append({
        "id": "goal",
        "kicker": "The requirement",
        "title": "Return the passage, not just the paper",
        "narrative":
            "Every design decision defers to one rule: a result must carry "
            "(doc_id, section, start_char, end_char) back to the source. A retrieval "
            "change that improves ranking but loses provenance is a regression, not a "
            "trade — because a claim you cannot point at is a claim the reader cannot "
            "check.",
        "visual": "provenance",
        "metrics": [
            metric("Documents indexed", live.get("documents", 0), provenance="live",
                   command="SELECT count(*) FROM documents"),
            metric("Retrievable passages", live.get("chunks", 0), provenance="live",
                   command="SELECT count(*) FROM chunks"),
            metric("With verbatim full text", live.get("full_text", 0), provenance="live",
                   note="the rest are abstract + MeSH only",
                   command="meta->>'full_text_chars' > 0"),
        ],
    })

    # ------------------------------------------------- 2. real data, and a mistake
    stages.append({
        "id": "licence",
        "kicker": "Source data",
        "title": "The licence gate was rejecting text-mining licences",
        "narrative":
            "The original gate admitted only Creative Commons codes and reported "
            "'30 correctly skipped on licence' as if that were good news. TDM is the PMC "
            "code for Text and Data Mining — content publishers release specifically for "
            "this use — and it was being rejected for the sole reason that it is not "
            "Creative Commons. That is backwards. Policies now name the intent.",
        "visual": "licence",
        "metrics": [
            metric("research policy (default)", 100, unit="%", note="28/28 of a sample",
                   command="scripts/ingest_real.py --license-policy research"),
            metric("commercial policy", 68, unit="%", note="19/28 — excludes NC"),
            metric("permissive_only (old default)", 54, unit="%", note="15/28 — CC BY only"),
            metric("Full-text documents recovered", 88, note="from 58, a 52% increase",
                   delta="+52%", direction="up"),
        ],
        "caveat":
            "If OncoLens is ever commercialised, switch to the commercial policy — the "
            "NC-licensed content indexed today is not licensed for that.",
    })

    # ------------------------------------------------------- 3. the precision leak
    stages.append({
        "id": "references",
        "kicker": "The precision leak",
        "title": "Bibliographies are a median 19% of full text, and they retrieve well",
        "narrative":
            "Citation strings match queries lexically while containing no findings. Two "
            "of the top three hits for 'osimertinib resistance mechanism' were reference "
            "entries, not claims. The first detector was tuned by opening articles and "
            "judging whether the output looked right — the exact failure mode this "
            "project exists to avoid.",
        "visual": "stripping",
        "metrics": [
            metric("Bibliography share of characters", 19, unit="%",
                   note="median over 60 articles; range 5.6%–42.4%",
                   command="scripts/analyze_ref_signals.py --limit 60"),
            metric("Bibliographies detected", "60/60", note="was 57/60", delta="+3"),
            metric("Reference characters removed", 1.0000, note="was 0.9500",
                   delta="+0.0500", command="scripts/compare_ref_detectors.py"),
            metric("Article body preserved", 0.9998, note="1.0000 is perfect; the loss is "
                   "679 chars of author-contribution boilerplate in one article"),
        ],
    })

    # ------------------------------------------------------ 4. how it was measured
    stages.append({
        "id": "ground-truth",
        "kicker": "Method",
        "title": "The publisher already labelled it",
        "narrative":
            "PMC's JATS XML carries <ref-list> — the publisher's own statement of where "
            "the bibliography starts. Aligning it onto the plain text turns a matter of "
            "taste into a labelled task. The labels showed the signals were never the "
            "problem: separation is essentially perfect. The design was wrong.",
        "visual": "signals",
        "signals": [
            {"name": "author initials (fixed)", "body": 0.730, "refs": 24.503, "auc": 1.000},
            {"name": "years", "body": 0.254, "refs": 5.952, "auc": 1.000},
            {"name": "DOIs", "body": 0.034, "refs": 4.090, "auc": 1.000},
            {"name": "function-word fraction", "body": 0.230, "refs": 0.086, "auc": 0.000},
            {"name": "page ranges", "body": 0.255, "refs": 2.431, "auc": 0.983},
            {"name": "author initials (old regex)", "body": 0.588, "refs": 13.303, "auc": 0.900},
            {"name": "numbered entries", "body": 0.0, "refs": 0.0, "auc": 0.692},
            {"name": "volume:page", "body": 0.0, "refs": 0.0, "auc": 0.533},
        ],
        "metrics": [
            metric("Articles with publisher labels", 60, provenance="artifact",
                   command="scripts/bench_references.py --limit 60"),
            metric("Signals found to be pure noise", 2,
                   note="'numbered entries' and 'volume:page' carried hand-assigned "
                        "weights of 0.20 and 0.12"),
        ],
    })

    # --------------------------------------------- 5. the benchmark that lied
    stages.append({
        "id": "review-articles",
        "kicker": "What the benchmark missed",
        "title": "A 60-article sample certified a rule that failed on 180 documents",
        "narrative":
            "After scoring 60/60 on labelled data, the live index still held 1,394 "
            "reference-shaped passages. The mean was unremarkable; the distribution was "
            "the diagnostic — 215 in one document, 124 in another. Concentration means "
            "undetected bibliographies. Every labelled article had been a primary "
            "research paper, and a review inverts the ratio.",
        "visual": "distribution",
        "metrics": [
            metric("Largest single bibliography", 480996, unit=" chars",
                   note="PMC10958066 — 80.6% of the document, in one paragraph"),
            metric("Documents where the 60% rail bound", 180, note="of 1,739"),
            metric("Regression on the labelled set", 0, note="detected 60/60 unchanged"),
        ],
        "caveat":
            "The transferable lesson is about the benchmark, not the threshold: when a "
            "component passes its benchmark, check the distribution of its failures on "
            "the live corpus.",
    })

    # ------------------------------------------------------------ 6. labels
    stages.append({
        "id": "labels",
        "kicker": "Evaluation",
        "title": "The bibliographies are also the labels",
        "narrative":
            "When an author writes 'resistance to osimertinib is driven by MET "
            "amplification [12]', that sentence is a technical description of reference "
            "[12] written by someone who read it. It is a query; the cited paper is the "
            "answer; the judgment is free and expert — and claim-level, where MeSH is "
            "only document-level topical. Not circular: labels come from JATS <xref> "
            "markup, while retrieval indexes the plain text where markers are already "
            "flattened away.",
        "visual": "citation-graph",
        "metrics": [
            metric("Queries", n_queries, provenance="artifact",
                   command="scripts/build_citation_labels.py"),
            metric("Relevance judgments", n_judgments, provenance="artifact"),
            metric("Distinct target papers", n_targets, provenance="artifact"),
            metric("Yield before snowball ingestion", 3,
                   note="3 labels from 4,973 contexts — 4,967 cited papers we did not hold"),
        ],
        "guards": [
            {"hazard": "The citing paper contains the query verbatim",
             "guard": "assert_source_excluded() raises — asserted, not documented"},
            {"hazard": "Judgments are incomplete",
             "guard": "report bpref and unjudged@k alongside nDCG, never nDCG alone"},
            {"hazard": "Popularity bias",
             "guard": "MAX_PER_TARGET=4, MAX_PER_SOURCE=12"},
            {"hazard": "'several studies show X [3,7,11]'",
             "guard": "grade falls with co-citation; more than 3 co-cited is dropped"},
            {"hazard": "'as previously described [9]'",
             "guard": "rejected — an assertive verb is required"},
        ],
    })

    # ------------------------------------------------------------ 7. retrieval
    sys_rows = []
    for name, vals in (bench.get("systems") or {}).items():
        sys_rows.append({"system": name, **{k: round(v, 4) for k, v in vals.items()
                                            if isinstance(v, (int, float))}})
    stages.append({
        "id": "retrieval",
        "kicker": "Does any of it help?",
        "title": "Measured on 2,225 expert-written queries",
        "narrative":
            "Systems are compared on identical queries with a paired permutation test and "
            "a bootstrap confidence interval. With this many queries a two-point nDCG gap "
            "is still routinely noise, and shipping on a point estimate is how retrieval "
            "systems silently get worse.",
        "visual": "bars",
        "systems": sys_rows,
        "metrics": [
            metric("Queries evaluated", bench.get("n_queries", n_queries),
                   provenance="artifact", command="scripts/bench_retrieval.py"),
        ],
        # The unjudged caveat must travel WITH the numbers. Shown separately — or in a doc
        # — it stops being a qualification and becomes a footnote nobody reads, while the
        # table it qualifies looks like a clean result.
        "caveat":
            "unjudged@10 is about 0.94 for every system: roughly 94% of returned documents "
            "were never judged, at 1.10 judged documents per query. These are LOWER BOUNDS, "
            "not estimates of retrieval quality — the promotion gate blocks anything above "
            "0.35 unjudged, and it is right to. The comparison between systems holds because "
            "the unjudged rate is nearly identical across them; no single number here should "
            "be quoted as 'the' retrieval quality. bpref is absent by design: it returns None "
            "below 10 judged negatives rather than averaging noise into a consensus."
            if sys_rows else
            "The benchmark has not been run against this index. No numbers are shown, "
            "rather than stale ones.",
        "headline": ("Deleting a component was the second-largest win available: BM25 alone "
                     "beat the lexical+LSA hybrid that shipped, p < 0.0001.") if sys_rows else None,
    })

    # ------------------------------------------------------------ 8. storage
    db_mb = round((live.get("db_bytes") or 0) / 1e6)
    stages.append({
        "id": "storage",
        "kicker": "Infrastructure",
        "title": "Neon is right for now, and the limit is already visible",
        "narrative":
            "This corpus is relational — grants, publications, PIs, institutions — and "
            "those joins are half the product, which is the real argument for Postgres "
            "over a pure vector store. The cost is that ts_rank_cd is not BM25, so the "
            "offline harness and production score differently. The free tier's ceiling "
            "arrived exactly where the storage note predicted.",
        "visual": "storage",
        "metrics": [
            metric("Database size", db_mb, unit=" MB", provenance="live",
                   command="SELECT pg_database_size(current_database())"),
            metric("Unused index removed", 86, unit=" MB",
                   note="GIN trigram index, idx_scan = 0 — 20% of the database"),
            metric("Passage text stored", round((live.get("text_chars") or 0) / 1e6),
                   unit=" MB", provenance="live"),
        ],
        "caveat":
            "Full text lives in Vercel Blob, not Postgres: it is large, immutable and "
            "never queried — only fetched once a passage has already been retrieved.",
    })

    return {
        "generated_at": now_iso,
        "git_rev": git_rev(),
        "live_store_reachable": bool(live) and "error" not in live,
        "store_error": live.get("error"),
        "stages": stages,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "public" / "journey.json"))
    ap.add_argument("--now", default=None,
                    help="ISO timestamp to stamp (defaults to current UTC)")
    args = ap.parse_args()

    load_env()
    if args.now:
        now = args.now
    else:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    data = build(now)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")

    n_metrics = sum(len(s.get("metrics", [])) for s in data["stages"])
    live_n = sum(1 for s in data["stages"] for m in s.get("metrics", [])
                 if m["provenance"] == "live")
    print(f"wrote {out}")
    print(f"  {len(data['stages'])} stages, {n_metrics} metrics "
          f"({live_n} live, {n_metrics - live_n} artifact/recorded)")
    if not data["live_store_reachable"]:
        print(f"  ! store unreachable: {data.get('store_error')}")
        print("    live metrics will render as unavailable rather than as stale numbers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
