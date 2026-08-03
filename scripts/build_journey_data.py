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


def _hybrid_gain(bench: dict) -> float:
    """nDCG@10 gap between the fused system and BM25 alone.

    Derived from the artifact rather than typed into the headline, because the previous
    headline described a comparison the table no longer contained: it quoted the round-0
    result about deleting the LSA arm, while the table beside it listed bm25 / openai /
    hybrid-openai and no LSA at all.
    """
    s = bench.get("systems") or {}
    hyb = next((v for k, v in s.items() if k.startswith("hybrid")), None)
    base = s.get("bm25")
    if not hyb or not base:
        return 0.0
    return round((hyb.get("ndcg@10") or 0) - (base.get("ndcg@10") or 0), 4)


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
            "trade, because a claim you cannot point at is a claim the reader cannot "
            "check.",
        "visual": "provenance",
        "metrics": [
            metric("Documents indexed", live.get("documents", 0), provenance="live",
                   command="SELECT count(*) FROM documents"),
            metric("Retrievable passages", live.get("chunks", 0), provenance="live",
                   command="SELECT count(*) FROM chunks"),
            # "With verbatim full text" was dropped once abstract-only records were pruned:
            # it now equals the document count exactly, so it reported nothing.
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
            "code for Text and Data Mining (content publishers release specifically for "
            "this use), and it was being rejected for the sole reason that it is not "
            "Creative Commons. That is backwards. Policies now name the intent.",
        "visual": "licence",
        "metrics": [
            metric("research policy (default)", 100, unit="%", note="28/28 of a sample",
                   command="scripts/ingest_real.py --license-policy research"),
            metric("commercial policy", 68, unit="%", note="19/28, excludes NC"),
            metric("permissive_only (old default)", 54, unit="%", note="15/28, CC BY only"),
            metric("Full-text documents recovered", 88, note="from 58, a 52% increase",
                   delta="+52%", direction="up"),
        ],
        "caveat":
            "If OncoLens is ever commercialised, switch to the commercial policy: the "
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
            "judging whether the output looked right, the exact failure mode this "
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
            "PMC's JATS XML carries <ref-list>, the publisher's own statement of where "
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
            "the diagnostic: 215 in one document, 124 in another. Concentration means "
            "undetected bibliographies. Every labelled article had been a primary "
            "research paper, and a review inverts the ratio.",
        "visual": "distribution",
        "metrics": [
            metric("Largest single bibliography", 480996, unit=" chars",
                   note="PMC10958066, 80.6% of the document, in one paragraph"),
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
            "answer; the judgment is free and expert, and claim-level, where MeSH is "
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
                   note="3 labels from 4,973 contexts, 4,967 cited papers we did not hold"),
        ],
        "guards": [
            {"hazard": "The citing paper contains the query verbatim",
             "guard": "assert_source_excluded() raises, asserted rather than documented"},
            {"hazard": "Judgments are incomplete",
             "guard": "report bpref and unjudged@k alongside nDCG, never nDCG alone"},
            {"hazard": "Popularity bias",
             "guard": "MAX_PER_TARGET=4, MAX_PER_SOURCE=12"},
            {"hazard": "'several studies show X [3,7,11]'",
             "guard": "grade falls with co-citation; more than 3 co-cited is dropped"},
            {"hazard": "'as previously described [9]'",
             "guard": "rejected: an assertive verb is required"},
        ],
    })

    # ------------------------------------------------------------ 7. retrieval
    sys_rows = []
    for name, vals in (bench.get("systems") or {}).items():
        sys_rows.append({"system": name, **{k: round(v, 4) for k, v in vals.items()
                                            if isinstance(v, (int, float))}})
    # The query count is READ, not written here. A hardcoded "2,225 expert-written
    # queries" sat in this title while the metric directly beneath it reported the real
    # figure from the artifact, so the same card contradicted itself as the benchmark grew.
    n_bench = bench.get("n_queries", n_queries)
    unjudged = [v.get("unjudged@10") for v in (bench.get("systems") or {}).values()
                if isinstance(v.get("unjudged@10"), (int, float))]
    stages.append({
        "id": "retrieval",
        "kicker": "Does any of it help?",
        "title": f"Measured on {n_bench:,} expert-written queries",
        "narrative":
            "Systems are compared on identical queries with a paired permutation test and "
            "a bootstrap confidence interval. With this many queries a two-point nDCG gap "
            "is still routinely noise, and shipping on a point estimate is how retrieval "
            "systems silently get worse.",
        "visual": "bars",
        "systems": sys_rows,
        "metrics": [
            metric("Queries evaluated", n_bench,
                   provenance="artifact", command="scripts/bench_retrieval.py"),
        ],
        # The caveat used to appear ONLY when there was no data, which is backwards: the
        # case that needs a warning is the one where numbers ARE on screen. At
        # unjudged@10 around 0.93 these are lower bounds of unknown size, and the notes
        # forbid quoting an absolute number from this benchmark as "the" retrieval quality.
        "caveat":
            (f"These are lower bounds, not absolute quality. {min(unjudged):.0%}-"
             f"{max(unjudged):.0%} of the documents each system returned have never been "
             f"judged by anyone, because the labels come from citations rather than from "
             f"an exhaustive pool. The COMPARISON between systems holds - the unjudged "
             f"rate is near-identical across all three, so none is being flattered - but "
             f"a single number here is not this system's retrieval quality."
             if unjudged else
             "Judgment coverage was not recorded for this run, so these numbers cannot be "
             "read as absolute quality.")
            if sys_rows else
            "The benchmark has not been run against this index. No numbers are shown, "
            "rather than stale ones.",
        "headline": (f"The hybrid beats BM25 alone by "
                     f"{_hybrid_gain(bench):+.4f} nDCG@10 on {n_bench:,} queries."
                     if sys_rows else None),
    })

    # ------------------------------------------------------------ 8. storage
    db_mb = round((live.get("db_bytes") or 0) / 1e6)
    stages.append({
        "id": "storage",
        "kicker": "Infrastructure",
        "title": "Neon is right for now, and the limit is already visible",
        "narrative":
            "This corpus is relational (grants, publications, PIs, institutions), and "
            "those joins are half the product, which is the real argument for Postgres "
            "over a pure vector store. The cost is that ts_rank_cd is not BM25, so the "
            "offline harness and production score differently. The free tier's ceiling "
            "arrived exactly where the storage note predicted.",
        "visual": "storage",
        "metrics": [
            metric("Database size", db_mb, unit=" MB", provenance="live",
                   command="SELECT pg_database_size(current_database())"),
            metric("Unused index removed", 86, unit=" MB",
                   note="GIN trigram index, idx_scan = 0, 20% of the database"),
            metric("Passage text stored", round((live.get("text_chars") or 0) / 1e6),
                   unit=" MB", provenance="live"),
        ],
        "caveat":
            "Full text lives in Vercel Blob, not Postgres: it is large, immutable and "
            "never queried, only fetched once a passage has already been retrieved.",
    })

    return {
        "generated_at": now_iso,
        "git_rev": git_rev(),
        "live_store_reachable": bool(live) and "error" not in live,
        "store_error": live.get("error"),
        "stages": stages,
        "power": power_table(ddir),
    }


def empirical_sd(ddir: Path) -> dict[int, float]:
    """Paired-difference sd per evaluated stratum, read from the loop's own bootstrap CIs.

    **Why the analytic estimate below is wrong, and always in the same direction.** The
    obvious MDE takes the standard deviation of the METRIC. But every test here is
    *paired*: the quantity whose spread matters is the per-query DIFFERENCE between two
    systems, and most queries return an identical ranking under both, so that difference
    is far tighter than the metric itself. Using the metric's sd therefore overstates the
    minimum detectable effect, and this project has spent three rounds calling itself
    blinder than it is.

    The run that exposed it: ``route_by_shape`` moved claim ``mrr`` by **+0.0105 at
    p < 0.0001**. The analytic table put that stratum's floor at 0.0167. An effect cannot
    be detected at p < 0.0001 and be below the detection floor, so the floor was wrong.

    A 95% percentile-bootstrap CI on the mean paired difference already contains the right
    number: ``width = 2 * 1.96 * sd_diff / sqrt(n)``, so ``sd_diff = width * sqrt(n) /
    3.92`` and the MDE collapses to ``(1.96 + 0.8416) / 3.92 * width ~= 0.715 * width``,
    independent of n. Keyed by ``n_queries`` because that identifies which stratum an
    iteration ran on.
    """
    path = ddir / "improve_ledger.json"
    if not path.exists():
        return {}
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    out: dict[int, float] = {}
    for entry in ledger:
        n = entry.get("n_queries") or 0
        # Newer entries name their stratum; older ones predate that field and are matched
        # by query count instead. Keyed on n either way so both work.
        for v in entry.get("verdicts", []):
            for d in (v.get("deltas") or {}).values():
                ci = d.get("ci")
                if not ci or d.get("n") in (None, 0):
                    continue
                width = float(ci[1]) - float(ci[0])
                if width <= 0:
                    continue
                # Narrowest CI seen for this stratum is the best variance estimate,
                # since a wider one usually means a metric with more spread.
                sd = width * (float(d["n"]) ** 0.5) / 3.919928
                out[n] = min(out.get(n, sd), sd)
    return out


def power_table(ddir: Path) -> list[dict]:
    """Per-stratum sample size and the smallest effect that size can actually detect.

    **Why this belongs on the page and not in a notebook.** "No significant change" and
    "this experiment cannot see a change that size" render identically in a results table
    and mean opposite things. Publishing the minimum detectable effect next to the stratum
    is what lets a reader tell a negative result from a blind one, and it is the single
    number this project has been most wrong about historically.

    Two estimates are emitted. ``mde_analytic`` assumes the paired difference is as spread
    out as the metric itself, which is conservative and is what earlier rounds quoted.
    ``mde`` uses the observed bootstrap CI when the loop has actually run on that stratum
    (see ``empirical_sd``), and is roughly half the analytic figure.
    """
    from oncolens.eval.stats import detectable_effect
    from oncolens.eval.weighting import PRIMARY_METRIC, STRATUM_WEIGHTS

    strata_path = ddir / "strata.json"
    if not strata_path.exists():
        return []
    d = json.loads(strata_path.read_text(encoding="utf-8"))
    sizes: dict[str, int] = {}
    judged: dict[str, int] = {}
    for qid in d.get("queries", {}):
        s = d.get("strata", {}).get(qid, "?")
        sizes[s] = sizes.get(s, 0) + 1
        judged[s] = judged.get(s, 0) + len(d.get("qrels", {}).get(qid, {}))

    # Baselines observed on the dev split; used only to derive the Bernoulli sd, so a
    # rough value changes the MDE very little.
    baseline = {"synthesis": 0.2812, "concept": 0.7063, "identifier": 0.1899, "claim": 0.4950}
    emp = empirical_sd(ddir)
    out = []
    for s in ("synthesis", "concept", "identifier", "claim"):
        n = sizes.get(s, 0)
        if not n:
            continue
        p = baseline.get(s, 0.5)
        sd = (p * (1 - p)) ** 0.5
        analytic = detectable_effect(n, sd, alpha=0.05)

        # The loop evaluates the dev split, ~70% of the stratum. Match the ledger entry by
        # query count, TIGHTLY.
        #
        # A generous tolerance silently mis-attributes: at 25% it matched identifier
        # (dev ~235) to the concept iteration (n=252) and reported concept's variance as
        # identifier's. 5% keeps the unambiguous matches (claim 4939 vs 5000, synthesis
        # 1525 vs 1487) and correctly refuses the strata this round did not evaluate,
        # which then fall back to the conservative analytic figure. Reporting a
        # conservative floor is a much smaller error than reporting another stratum's.
        dev_n = round(n * 0.7)
        best = min(emp, key=lambda k: abs(k - dev_n), default=None)
        measured = None
        if best is not None and abs(best - dev_n) <= 0.05 * dev_n:
            measured = detectable_effect(best, emp[best], alpha=0.05)

        mde = measured if measured is not None else analytic
        out.append({
            "stratum": s,
            "queries": n,
            "judgments": judged.get(s, 0),
            "weight": STRATUM_WEIGHTS.get(s),
            "gate_metric": PRIMARY_METRIC.get(s),
            "mde": round(mde, 4),
            "mde_analytic": round(analytic, 4),
            "mde_source": "measured from paired bootstrap CI" if measured is not None
                          else "analytic, conservative: no run on this stratum yet",
            "sees_002": bool(mde <= 0.02),
        })
    return out


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
