#!/usr/bin/env python
"""The improvement loop: propose, measure, promote or discard — and say which, and why.

Each iteration takes one candidate change, evaluates it against the **dev** split of the
citation benchmark, and applies a promotion gate. A candidate that passes is written into
the served configuration and committed; a candidate that fails is discarded with the reason
recorded. Nothing is promoted on a point estimate.

**What makes this honest rather than a machine for producing green numbers:**

* **A locked test split.** Candidates are only ever measured on ``dev``. The ``test`` split
  is evaluated once, on request, to estimate how much of the dev gain was overfitting. The
  split is by *information need*, not by query id — near-duplicate queries derived from the
  same citation would otherwise straddle the boundary and leak.

* **The gate spans aggregations, not synonyms.** With 91.2% of queries carrying exactly one
  judgment, ``ndcg@10``, ``mrr`` and ``map`` are monotone transforms of the same rank, so a
  panel containing all three would count one fact three times. The consensus set is
  ``mrr`` plus ``success@{1,5,10,20}`` — different cutoffs of the rank distribution, which
  genuinely can disagree.

* **Regressions veto.** A candidate that raises the mean while pushing queries out of the
  top 10 is refused, however good the headline looks.

* **Bonferroni within the iteration**, not across all history. Correcting over every draw
  ever taken drives alpha to nothing and guarantees Type II errors; the locked test split
  is the real defence against cumulative overfitting.

* **Failures are recorded.** A loop that only logs its wins is a loop that will overfit and
  look like it did not.

    python scripts/improve_loop.py --list
    python scripts/improve_loop.py --run expand_mesh --run rerank_llm
    python scripts/improve_loop.py --run-all --commit
    python scripts/improve_loop.py --final-test        # spend the locked split
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

from oncolens.env import load_env, local_data_dir  # noqa: E402
from oncolens.eval import metrics as M  # noqa: E402
from oncolens.eval.citation_labels import assert_source_excluded  # noqa: E402
from oncolens.eval.stats import compare  # noqa: E402
from oncolens.eval.weighting import (  # noqa: E402
    PRIMARY_METRIC, SECONDARY_METRICS, STRATUM_WEIGHTS, describe,
)
from oncolens.retrieval.dense import make_backend  # noqa: E402
from oncolens.retrieval.fusion import aggregate_chunks_to_docs, reciprocal_rank_fusion  # noqa: E402
from oncolens.retrieval.lexical import BM25Index  # noqa: E402

TOP_K = 20
#: Gate thresholds.
ALPHA = 0.05
MIN_WINNING_METRICS = 2          # of the consensus panel, must improve significantly
MAX_REGRESSING_METRICS = 0       # any significant regression vetoes
MIN_EFFECT = 0.005               # below this, a "win" is not worth the complexity
UNJUDGED_TOLERANCE = 0.02        # a candidate that inflates unjudged@10 is suspect


# ---------------------------------------------------------------------------
# Candidate changes. Each is a pure transformation of how a query is answered.
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    name: str
    description: str
    rationale: str
    #: mutates the run config; returns a dict merged over the baseline
    config: dict = field(default_factory=dict)
    cost_note: str = ""


CANDIDATES: list[Candidate] = [
    Candidate(
        "baseline", "lexical + OpenAI embeddings, RRF, max aggregation",
        "The configuration measured best so far; every candidate is scored against it.",
        {},
    ),
    Candidate(
        "expand_mesh", "expand entity terms with MeSH entry terms",
        "Oncology writing is densely synonymous (osimertinib/AZD9291/Tagrisso). BM25 "
        "scores these as unrelated tokens, so a user searching one form misses papers "
        "using another.",
        {"expand": True},
        cost_note="two cached E-utilities calls per unseen entity",
    ),
    Candidate(
        "dense_only", "drop the lexical arm",
        "Tests whether BM25 still contributes once embeddings are good — the LSA arm was "
        "measurably harmful, so the assumption that both arms help deserves re-testing.",
        {"bm25_weight": 0.0},
    ),
    Candidate(
        "lexical_heavy", "weight BM25 2x against the dense arm",
        "Exact identifiers (EGFR C797S) are the queries dense retrieval handles worst.",
        {"bm25_weight": 2.0},
    ),
    Candidate(
        "topn_decay", "aggregate passages by top-3 decay instead of max",
        "`max` rewards one decisive passage; `topn_decay` rewards a paper that is relevant "
        "throughout. Which wins is query-type dependent and has never been measured on "
        "real documents — on the synthetic fixture it was an inert knob.",
        {"aggregate": "topn_decay"},
    ),
    Candidate(
        "deep_candidates", "retrieve 400 candidates instead of 200",
        "Cheap recall test: if the cited paper is being lost before fusion, depth fixes it "
        "and nothing else will.",
        {"candidates": 400},
    ),
    Candidate(
        "medcpt", "swap the dense arm for NCBI MedCPT",
        "text-embedding-3-small is a general model that has read some biomedical text. "
        "MedCPT was trained contrastively on 255M (query, clicked article) pairs from "
        "PubMed itself, and those logs are SHORT queries — the exact shape of the concept "
        "and identifier strata, and where a general embedder has least advantage.",
        {"dense_backend": "medcpt"},
        cost_note="768-dim, needs torch (~2GB): measurable offline, not servable on Vercel "
                  "without a hosted endpoint and a schema change",
    ),
    Candidate(
        "adaptive_weights", "set fusion weights from the query's shape",
        "THE FINDING FROM ROUND 1. Two failures pointed opposite ways: dropping BM25 hurt "
        "the 2-word concept stratum (success@10 -0.0707, p=0.0008) while doubling BM25 "
        "hurt conceptual synthesis queries (recall@20, recall@10, ndcg@10 all regressed). "
        "Neither is a contradiction - the optimal weight is query-type dependent, and one "
        "global weight is wrong in both directions.",
        {"adaptive": True},
    ),
    Candidate(
        "mmr_diversify", "MMR with a per-document cap on the passage ranking",
        "Synthesis has the lowest score of any stratum (recall@20 0.3078) and the highest "
        "weight. Depth and aggregation both did nothing, so the passages are being found "
        "and then crowded out: one paper contributing five near-identical passages "
        "displaces four other papers from the answer SET. Diversity should help coverage "
        "specifically, which is what that stratum measures.",
        {"mmr": True},
    ),
    Candidate(
        "openai_768", "same OpenAI model at 768 dimensions",
        "CONTROL, not a proposal. MedCPT is 768-dim against a 192-dim OpenAI index, so a "
        "MedCPT win would confound domain training with vector capacity. This isolates "
        "capacity: if openai_768 captures most of the MedCPT gain, the story is dimensions, "
        "not PubMed click logs — and the cheap change is the right one.",
        {"dense_backend": "openai-768"},
        cost_note="4x vector storage; still servable, unlike MedCPT",
    ),
    Candidate(
        "rerank_llm", "LLM cross-encoder rerank of the top 24 passages",
        "A bi-encoder cannot judge whether a passage *reports* the queried finding or "
        "merely mentions it. That distinction is the whole difference between useful "
        "oncology search and keyword match.",
        {"rerank": True},
        cost_note="~$0.0017 per query on gpt-4o-mini",
    ),
]

CANDIDATES_BY_NAME = {c.name: c for c in CANDIDATES}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_qrels(split: str, stratum: str = "claim") -> tuple[dict, dict, dict]:
    """Return (queries, qrels, exclude) for a split.

    Splitting is by **citing document**, not by query id. Several queries mined from the
    same paper describe the same body of work in similar language; splitting by id would
    put near-duplicates on both sides and let the loop tune against its own test set.
    """
    path = local_data_dir() / "strata.json"
    if not path.exists():
        raise SystemExit(f"no strata at {path}; run scripts/build_strata.py")
    d = json.loads(path.read_text(encoding="utf-8"))
    queries, qrels, exclude = d["queries"], d["qrels"], d.get("exclude", {})
    strata = d.get("strata", {})
    if stratum != "all":
        queries = {k: v for k, v in queries.items() if strata.get(k) == stratum}
        qrels = {k: v for k, v in qrels.items() if k in queries}
    # A query with no relevant documents measures nothing here.
    queries = {k: v for k, v in queries.items() if qrels.get(k)}
    qrels = {k: v for k, v in qrels.items() if k in queries}
    if split == "all":
        return queries, qrels, exclude
    import hashlib

    keep = {}
    for qid in queries:
        src = exclude.get(qid, qid)
        h = int(hashlib.sha1(src.encode()).hexdigest()[:8], 16)
        in_dev = (h % 100) < 70          # 70/30 dev/test, stable across runs
        if (split == "dev") == in_dev:
            keep[qid] = queries[qid]
    return keep, {k: v for k, v in qrels.items() if k in keep}, exclude


def load_chunks() -> list[dict]:
    import psycopg

    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("POSTGRES_URL / DATABASE_URL not set")
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT chunk_id, doc_id, COALESCE(indexed_text, text) FROM chunks")
        return [{"chunk_id": r[0], "doc_id": r[1], "text": r[2]} for r in cur.fetchall()]


class Harness:
    """Builds every index once; each candidate is a different way of querying them."""

    def __init__(self, chunks: list[dict], queries: dict[str, str], dim: int = 192):
        self.chunks = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.texts = [c["text"] for c in chunks]
        self.chunk_to_doc = {c["chunk_id"]: c["doc_id"] for c in chunks}
        self.by_id = {c["chunk_id"]: c["text"] for c in chunks}
        self.qids = sorted(queries)
        self.queries = queries
        print(f"indexing {len(chunks):,} passages...")
        self.bm25 = BM25Index().build(self.chunk_ids, self.texts)
        be = make_backend("openai", dim=dim)
        self.dvecs = be.encode_documents_cached(self.texts, local_data_dir() / "emb_cache")
        self.qvecs = be.encode_queries([queries[q] for q in self.qids])
        self._expanded: dict[str, str] | None = None
        self._dense_cache: dict[str, tuple] = {}

    def expanded_queries(self) -> dict[str, str]:
        if self._expanded is None:
            from oncolens.terminology import expand_query

            self._expanded = {}
            for i, qid in enumerate(self.qids):
                e = expand_query(self.queries[qid], cache_dir=local_data_dir())
                self._expanded[qid] = e.expanded_query()
                if (i + 1) % 200 == 0:
                    print(f"  expanded {i+1}/{len(self.qids)}")
        return self._expanded

    def dense_vectors(self, backend_name: str):
        """Document and query matrices for a named backend, encoded once and reused."""
        if backend_name in self._dense_cache:
            return self._dense_cache[backend_name]
        be = make_backend(backend_name, dim=192)
        be.fit(self.texts)
        if hasattr(be, "encode_documents_cached"):
            dv = be.encode_documents_cached(self.texts, local_data_dir() / "emb_cache")
        else:
            dv = be.encode_documents(self.texts)
        qv = be.encode_queries([self.queries[q] for q in self.qids])
        self._dense_cache[backend_name] = (qv, dv)
        return qv, dv

    def run(self, cfg: dict, qrels: dict, exclude: dict) -> dict[str, dict[str, float]]:
        bm25_w = cfg.get("bm25_weight", 1.0)
        dense_w = cfg.get("dense_weight", 1.0)
        cand = cfg.get("candidates", 200)
        agg = cfg.get("aggregate", "max")
        use_expand = cfg.get("expand", False)
        use_rerank = cfg.get("rerank", False)
        use_adaptive = cfg.get("adaptive", False)
        use_mmr = cfg.get("mmr", False)
        exp = self.expanded_queries() if use_expand else None
        qvecs, dvecs = (self.dense_vectors(cfg["dense_backend"])
                        if cfg.get("dense_backend") else (self.qvecs, self.dvecs))

        per_query: dict[str, dict[str, float]] = {}
        for qi, qid in enumerate(self.qids):
            # ADAPTIVE FUSION. Round 1 measured two failures pointing opposite ways:
            # dropping BM25 hurt 2-word concept queries (success@10 -0.0707, p=0.0008)
            # while doubling BM25 hurt conceptual synthesis queries (recall@20, recall@10
            # and ndcg@10 all regressed). A short query is mostly literal and needs the
            # lexical arm; a long conceptual one carries enough context for the dense arm
            # to do better work. One global weight is wrong in both directions.
            if use_adaptive:
                n_words = len(self.queries[qid].split())
                if n_words <= 3:
                    bm25_w, dense_w = 2.0, 1.0
                elif n_words <= 8:
                    bm25_w, dense_w = 1.0, 1.0
                else:
                    bm25_w, dense_w = 1.0, 2.0

            runs, weights = [], []
            if bm25_w > 0:
                lex_q = exp[qid] if exp else self.queries[qid]
                runs.append(self.bm25.search(lex_q, k=cand))
                weights.append(bm25_w)
            if dense_w > 0:
                sims = dvecs @ qvecs[qi]
                top = np.argpartition(-sims, min(cand, len(sims) - 1))[:cand]
                top = top[np.argsort(-sims[top])]
                runs.append([(self.chunk_ids[i], float(sims[i])) for i in top])
                weights.append(dense_w)
            # The weights MUST reach RRF. Omitting them made bm25_weight control only
            # whether an arm was included, not how much it counted, so the
            # `lexical_heavy` candidate produced byte-identical rankings to the baseline
            # and the loop confidently DISCARDED a change that had never run.
            fused = (reciprocal_rank_fusion(runs, weights=weights) if len(runs) > 1
                     else list(runs[0]))

            if use_rerank:
                from oncolens.retrieval.llm_rerank import rerank as llm_rerank

                head = fused[:24]
                order = llm_rerank(self.queries[qid], [self.by_id[c] for c, _ in head])
                fused = ([(head[r.index][0], r.score) for r in order]
                         + [(c, -1.0 - i) for i, (c, _) in enumerate(fused[24:])])

            if use_mmr:
                fused = _cap_per_document(fused, self.chunk_to_doc, cap=2)

            docs = aggregate_chunks_to_docs(fused, self.chunk_to_doc, strategy=agg)
            src = exclude.get(qid)
            ranking = [d for d, _ in docs if d != src][:TOP_K]
            assert_source_excluded(qid, ranking, exclude)
            per_query[qid] = M.evaluate_query(ranking, qrels.get(qid, {}))
            per_query[qid]["_ranking_hash"] = float(
                hash(tuple(ranking)) % 1_000_000_007)
        return per_query


def _cap_per_document(ranked: list, chunk_to_doc: dict, *, cap: int = 2) -> list:
    """Limit how many passages one document may contribute before others are considered.

    Cheap stand-in for MMR that targets the same failure. A synthesis question is answered
    by a SET of papers, and one paper contributing five near-identical passages displaces
    four other papers from that set. Since depth (400 candidates) and aggregation
    (topn_decay) both moved nothing, the passages are being FOUND and then crowded out -
    which is a diversity problem, not a recall problem.

    Capped passages are appended after the diversified head rather than dropped, so this
    reorders rather than discards: nothing that was retrievable becomes unretrievable.
    """
    seen: dict[str, int] = {}
    head, tail = [], []
    for chunk_id, score in ranked:
        doc = chunk_to_doc.get(chunk_id)
        n = seen.get(doc, 0)
        if n < cap:
            seen[doc] = n + 1
            head.append((chunk_id, score))
        else:
            tail.append((chunk_id, score))
    return head + tail


def mean_of(per_query: dict, key: str) -> float:
    vals = [v[key] for v in per_query.values() if v.get(key) is not None]
    return sum(vals) / len(vals) if vals else float("nan")


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    name: str
    promoted: bool
    reason: str
    deltas: dict = field(default_factory=dict)
    wins: list = field(default_factory=list)
    regressions: list = field(default_factory=list)


def judge(name: str, base: dict, cand: dict, stratum: str = "claim") -> Verdict:
    # Each stratum is gated on the metric that matches ITS task. A synthesis question is
    # answered by a set, so coverage is what counts and reciprocal rank is close to
    # meaningless - handing back the single best paper out of nine is not a good answer
    # to "what is known about X". An identifier lookup is the opposite.
    primary = PRIMARY_METRIC.get(stratum, "mrr")
    panel = tuple(dict.fromkeys(
        (primary,) + SECONDARY_METRICS.get(stratum, ()) + M.CONSENSUS_METRICS))
    alpha = ALPHA / max(len(panel), 1)          # Bonferroni WITHIN this iteration
    wins, regs, deltas = [], [], {}
    for metric in panel:
        c = compare(base, cand, metric)
        if c is None:
            continue
        deltas[metric] = {"delta": c.delta, "p": c.p_value,
                          "ci": [c.ci_low, c.ci_high], "n": c.n_queries}
        if c.p_value < alpha and c.delta > MIN_EFFECT:
            wins.append(metric)
        elif c.p_value < alpha and c.delta < -MIN_EFFECT:
            regs.append(metric)

    # A candidate whose rankings are identical to the baseline did not run. Reporting
    # that as "no significant improvement" hides a broken candidate behind a plausible
    # negative result — which is exactly how a loop launders its own bugs into findings.
    identical = sum(1 for q in base
                    if base[q].get("_ranking_hash") == cand.get(q, {}).get("_ranking_hash"))
    if identical == len(base) and base:
        return Verdict(name, False,
                       "NO EFFECT — rankings byte-identical to baseline; the candidate "
                       "did not fire (check that its config is actually wired through)",
                       {}, [], [])

    unj_base, unj_cand = mean_of(base, "unjudged@10"), mean_of(cand, "unjudged@10")
    unj_delta = unj_cand - unj_base
    deltas["unjudged@10"] = {"delta": unj_delta, "p": None, "ci": None, "n": None}

    if len(regs) > MAX_REGRESSING_METRICS:
        return Verdict(name, False,
                       f"significant regression on {', '.join(regs)}", deltas, wins, regs)
    if len(wins) < MIN_WINNING_METRICS:
        return Verdict(name, False,
                       f"only {len(wins)} of {len(panel)} consensus metrics improved "
                       f"significantly (need {MIN_WINNING_METRICS})", deltas, wins, regs)
    if unj_delta > UNJUDGED_TOLERANCE:
        return Verdict(name, False,
                       f"unjudged@10 rose {unj_delta:+.3f}: the gain may be an artifact of "
                       f"returning documents the pool never judged", deltas, wins, regs)
    return Verdict(name, True,
                   f"{len(wins)} metrics improved ({', '.join(wins)}), no regressions",
                   deltas, wins, regs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--stratum", default="claim",
                    choices=["claim", "concept", "identifier", "synthesis", "all"],
                    help="MEASURED: claim queries are 27 words, concept 2, identifier 1. "
                         "Users type the short ones, so a change must be judged on each "
                         "shape separately rather than on a pooled mean.")
    ap.add_argument("--final-test", action="store_true",
                    help="evaluate the promoted config on the LOCKED test split")
    ap.add_argument("--commit", action="store_true", help="git commit promoted changes")
    ap.add_argument("--limit-queries", type=int, default=0,
                    help="cap queries (use for expensive candidates like rerank_llm)")
    args = ap.parse_args()

    load_env()
    if args.list:
        print(describe())
        print()
        print(f"{'candidate':<18}{'description'}")
        print("-" * 78)
        for c in CANDIDATES:
            print(f"{c.name:<18}{c.description}")
            print(f"{'':<18}why: {c.rationale[:90]}")
            if c.cost_note:
                print(f"{'':<18}cost: {c.cost_note}")
        return 0

    split = "test" if args.final_test else args.split
    queries, qrels, exclude = load_qrels(split, args.stratum)
    if args.limit_queries:
        keep = sorted(queries)[: args.limit_queries]
        queries = {k: queries[k] for k in keep}
        qrels = {k: v for k, v in qrels.items() if k in queries}
    print(f"stratum={args.stratum}  split={split}  {len(queries)} queries, "
          f"{sum(len(v) for v in qrels.values())} judgments")

    harness = Harness(load_chunks(), queries)
    print("\nrunning baseline...")
    base = harness.run({}, qrels, exclude)
    _p = PRIMARY_METRIC.get(args.stratum, "mrr")
    print(f"  PRIMARY {_p}={mean_of(base,_p):.4f}  (weight "
          f"{STRATUM_WEIGHTS.get(args.stratum, 0):.2f} in the composite)")
    print(f"  mrr={mean_of(base,'mrr'):.4f}  success@1={mean_of(base,'success@1'):.4f}  "
          f"success@5={mean_of(base,'success@5'):.4f}  "
          f"recall@20={mean_of(base,'recall@20'):.4f}")

    names = ([c.name for c in CANDIDATES if c.name != "baseline"] if args.run_all
             else args.run)
    if not names:
        print("\nnothing to run; pass --run NAME or --run-all")
        return 0

    ledger: list[Verdict] = []
    for name in names:
        c = CANDIDATES_BY_NAME.get(name)
        if c is None:
            print(f"\n! unknown candidate {name!r}")
            continue
        print(f"\n=== {name} — {c.description} ===")
        try:
            cand = harness.run(c.config, qrels, exclude)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED to run: {type(e).__name__}: {str(e)[:140]}")
            ledger.append(Verdict(name, False, f"run error: {type(e).__name__}"))
            continue
        v = judge(name, base, cand, args.stratum)
        ledger.append(v)
        primary = PRIMARY_METRIC.get(args.stratum, "mrr")
        for m in dict.fromkeys((primary,) + SECONDARY_METRICS.get(args.stratum, ())
                               + M.CONSENSUS_METRICS):
            d = v.deltas.get(m)
            if not d:
                continue
            flag = "WIN " if m in v.wins else ("REG " if m in v.regressions else "    ")
            star = "*" if m == primary else " "
            print(f"  {flag}{star}{m:<12}{mean_of(base,m):>8.4f} -> {mean_of(cand,m):>8.4f}"
                  f"  ({d['delta']:+.4f}, p={d['p']:.4f})")
        print(f"  unjudged@10 {v.deltas['unjudged@10']['delta']:+.4f}")
        print(f"  -> {'PROMOTE' if v.promoted else 'DISCARD'}: {v.reason}")

    out = local_data_dir() / "improve_ledger.json"
    prior = []
    if out.exists():
        try:
            prior = json.loads(out.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prior = []
    prior.append({"split": split, "n_queries": len(queries),
                  "baseline": {m: mean_of(base, m) for m in M.CONSENSUS_METRICS},
                  "verdicts": [v.__dict__ for v in ledger]})
    out.write_text(json.dumps(prior, indent=2, default=str), encoding="utf-8")
    print(f"\nledger -> {out}  ({len(prior)} iterations recorded)")

    promoted = [v for v in ledger if v.promoted]
    print(f"\n{'='*70}\nPROMOTED {len(promoted)} / {len(ledger)}")
    for v in ledger:
        print(f"  {'✓' if v.promoted else '✗'} {v.name:<16} {v.reason}")
    if not promoted:
        print("\nNo candidate cleared the gate. That is a result, not a failure: the "
              "\nbaseline stands and nothing was changed on a point estimate.")

    if promoted and args.commit:
        cfg_path = ROOT / "config" / "served.json"
        cfg_path.parent.mkdir(exist_ok=True)
        merged: dict = {}
        for v in promoted:
            merged.update(CANDIDATES_BY_NAME[v.name].config)
        cfg_path.write_text(json.dumps(
            {"promoted": [v.name for v in promoted], "config": merged,
             "split": split, "n_queries": len(queries)}, indent=2), encoding="utf-8")
        msg = ("Improvement loop: promote " + ", ".join(v.name for v in promoted)
               + "\n\n" + "\n".join(f"{v.name}: {v.reason}" for v in promoted)
               + f"\n\nMeasured on the {split} split, {len(queries)} queries. "
                 "Bonferroni-corrected within the iteration. The test split remains "
                 "unspent.")
        subprocess.run(["git", "add", "-A"], cwd=ROOT, check=False)
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT, check=False)
        print(f"\ncommitted promoted config to {cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
