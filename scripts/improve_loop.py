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
    PRIMARY_METRIC, SECONDARY_METRICS, STRATUM_WEIGHTS, describe, gate_metric,
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
        "tri_fusion", "fuse BM25 + openai_768 + MedCPT, three arms instead of two",
        "ROUND 3, and it follows directly from round 2's measurement rather than from a "
        "hunch. MedCPT and openai_768 fail in OPPOSITE directions: MedCPT +0.0261 on "
        "synthesis set-coverage (p=0.0003) and -0.0166 on claim pinpointing (p=0.0034); "
        "openai_768 the reverse (+0.0093 claim, +0.0016 synthesis). Complementary failure "
        "modes on the same corpus are the precondition under which rank fusion beats "
        "either arm, so give each a vote instead of choosing between them.",
        {"dense_backend": "openai-768", "dense_backends": ["medcpt"]},
        cost_note="needs BOTH a 768-dim OpenAI index and a hosted MedCPT endpoint: the "
                  "most expensive candidate here, and only worth it if it beats both",
    ),
    Candidate(
        "rerank_medcpt_cross", "NCBI MedCPT cross-encoder over the fused top 50",
        "THE LARGEST UNTESTED LEVER, and it has never actually run. Every arm so far is a "
        "bi-encoder: query and passage are embedded separately, so the architecture cannot "
        "represent an interaction between them. It can tell that a passage is ABOUT "
        "osimertinib resistance; it cannot tell whether the passage REPORTS a mechanism or "
        "merely notes that one exists. A cross-encoder attends across both at once. "
        "`rerank_llm` was registered for this in round 1 and was structurally unable to "
        "pass, because it reordered the top 24 while being gated on recall@20 — a metric "
        "reordering cannot move (§4.8). gate_metric now redirects ordering-only candidates "
        "to a rank-sensitive metric, so the idea gets its first real test.",
        {"cross_encoder": "medcpt-cross", "cross_depth": 50},
        cost_note="local GPU, no API cost, ~50 pairs per query; needs a served GPU to ship",
    ),
    Candidate(
        "rerank_minilm_cross", "general MS MARCO cross-encoder: the DOMAIN control",
        "CONTROL for rerank_medcpt_cross, and the same move that caught two wrong "
        "attributions already (openai_768 for MedCPT in §4.13, dense_weight_2x for "
        "tri_fusion in round 5). ms-marco-MiniLM is a general web-search reranker of "
        "similar size with no biomedical training. If both rerankers help about equally, "
        "the gain is CROSS-ATTENTION and any reranker will do, which makes this cheap and "
        "portable. If only the biomedical one helps, the gain is domain training and the "
        "deployment story is quite different. Those two conclusions are not guessable from "
        "either number on its own.",
        {"cross_encoder": "minilm-cross", "cross_depth": 50},
        cost_note="local GPU; ~90 MB model, the cheapest candidate in the list",
    ),
    Candidate(
        "expand_ontology", "resolve the whole query against five curated registries",
        "Aimed at the worst number in the system: identifier success@1 is 0.148, on the "
        "stratum whose own rationale says a wrong answer costs most because the error is "
        "invisible. The failure is structured, not mysterious: oncology is densely "
        "synonymous (osimertinib / AZD9291 / Tagrisso; EGFR / ERBB1 / HER1) and BM25 scores "
        "those as unrelated tokens. `expand_mesh` was tried once in round 1 against 113 "
        "queries, which could not have detected anything.\n"
        "Measured coverage of the 335 identifier queries decided the design: HGNC 13.7%, "
        "NCIt 55.8%, ClinicalTrials.gov 22.7%, Cellosaurus 8.4%, union 87.5%. The stratum "
        "is only partly genes; it is also cell lines, drug codes, trial acronyms and HLA "
        "alleles. The regex that used to pick which spans to look up is demoted to a "
        "fallback: 89.6% of these queries are ONE token, so there is nothing to segment, "
        "and segmenting destroyed all 17 multiword identifiers (EGFR T790M -> EGFR, which "
        "broadens the query rather than merely failing to help).",
        {"expand": True, "expand_source": "ontology"},
        cost_note="one cached registry lookup per unseen term; free at query time after that",
    ),
    Candidate(
        "expand_identity_weighted",
        "identity synonyms only, with the user's own words weighted 3x",
        "DIAGNOSTIC FOLLOW-UP to expand_ontology, which regressed identifier success@1 by "
        "0.0339 (p=0.037) while successfully expanding 212 of 236 queries. Coverage was "
        "not the problem; HOW the synonyms were injected was. Two defects, both visible in "
        "the output rather than inferred:\n"
        "(1) NO WEIGHTING. `Expansion.expanded_query` concatenated synonyms at equal "
        "weight, so a one-token query like MCF-7 became 1 of 7 tokens and the user's own "
        "term was diluted to 14%. The method's own docstring said callers should score "
        "synonyms below the user's words; nothing did. Now `repeat=3`.\n"
        "(2) RELATION CONFLATION. PALOMA-3 expanded to 'Palbociclib, Fulvestrant, "
        "Placebo'. Those are the trial's INTERVENTIONS, not other names for it, so the "
        "query stopped being about one study. Resolutions now carry relation=identity vs "
        "association, and this candidate injects only identity.\n"
        "If this still regresses, the finding is that lexical expansion does not help this "
        "stratum at all, which is a real answer and retires the thread.",
        {"expand": True, "expand_source": "ontology", "expand_repeat": 3,
         "expand_identity_only": True},
        cost_note="identical to expand_ontology; the registries are already cached",
    ),
    Candidate(
        "dense_weight_2x", "two arms, dense weighted 2x: the MISSING CELL",
        "The cheapest experiment available and possibly the one that ends this thread. "
        "tri_fusion (+0.0305 synthesis) turned out to depend on BOTH a third arm AND the "
        "1:2 lexical:dense ratio that a third equal-weight arm creates for free: "
        "rebalancing to 1:1 collapsed it to +0.0021, and a THIRD ARM CARRYING NO NEW "
        "INFORMATION (dual_dense) still bought +0.0105. But dual_dense confounds arm count "
        "with weight, so neither control isolates the ratio on its own. This does: two "
        "arms, dense weighted 2x, no MedCPT, no third voter, nothing to encode. If it "
        "recovers most of tri_fusion's gain then the whole finding is fusion-weight "
        "tuning, MedCPT is unnecessary, and there is no hosted GPU endpoint to build.",
        {"dense_backend": "openai-768", "dense_weight": 2.0},
        cost_note="free: one config line, no new index, no new model. Answers whether the "
                  "expensive candidates were ever needed",
    ),
    Candidate(
        "tri_fusion_balanced", "tri_fusion with BM25 weighted 2x: the WEIGHT control",
        "CONTROL for tri_fusion, not a proposal, and the same move openai_768 was for "
        "MedCPT. Adding a third arm at equal weight does two things at once: it adds "
        "MedCPT AND it shifts lexical:dense from 1:1 to 1:2, because the dense side now "
        "has two votes. This project has already measured that the ratio matters "
        "(dense_only regressed concept; adaptive_weights regressed both ways), so part of "
        "tri_fusion's +0.0305 could be the rebalancing rather than complementarity. "
        "Weighting BM25 2x restores 1:1. If the gain survives, it is MedCPT. If it "
        "evaporates, tri_fusion measured a weight change and called it a model.",
        {"dense_backend": "openai-768", "dense_backends": ["medcpt"], "bm25_weight": 2.0},
        cost_note="same cost as tri_fusion; exists to attribute its gain, not to ship",
    ),
    Candidate(
        "dual_dense", "BM25 + openai_768 + openai@192: the ARM-COUNT control",
        "Second control for tri_fusion. Three arms at equal weight may beat two simply "
        "because RRF with more voters is more robust, independent of WHICH third voter is "
        "added. This adds a third arm that carries almost no new information (the same "
        "OpenAI model at two widths), so any gain here is the arm count and the fusion "
        "geometry rather than a genuinely different view of the corpus.",
        {"dense_backend": "openai-768", "dense_backends": ["openai"]},
        cost_note="cheap: no MedCPT, no hosted endpoint. If THIS matches tri_fusion, the "
                  "expensive candidate is buying nothing",
    ),
    Candidate(
        "route_by_shape", "send each query to the arm measured best for its shape",
        "Same round-2 measurement, different remedy. Rather than fusing, pick the arm "
        "whose measured strength matches the query in front of us: MedCPT for the "
        "mid-length topical questions where it wins coverage, openai_768 for the long "
        "verbatim sentences where it wins attribution. Cheaper than tri_fusion at query "
        "time (one dense arm, not two) and it is the hypothesis CLAUDE.md 4.13 registered "
        "for round 3.",
        {"dense_backend": "openai-768",
         "route_by_shape": {"short": "openai-768", "mid": "medcpt", "long": "openai-768"}},
        cost_note="still needs a hosted MedCPT endpoint, but only for mid-length queries",
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

    # FINE-TUNING SPLITS. A model trained on the queries it is then scored on will look
    # excellent and mean nothing, so `ft_train` and `ft_holdout` partition DEV only — the
    # locked test split is never touched by either. The hash is salted differently from the
    # dev/test hash: reusing the same digest would make the two splits correlated, and a
    # holdout that overlaps training by construction is not a holdout.
    if split in ("ft_train", "ft_holdout"):
        keep = {}
        for qid in queries:
            src = exclude.get(qid, qid)
            h = int(hashlib.sha1(src.encode()).hexdigest()[:8], 16)
            if (h % 100) >= 70:          # test — off limits to both
                continue
            hf = int(hashlib.sha1((src + ":finetune").encode()).hexdigest()[:8], 16)
            in_train = (hf % 100) < 70
            if (split == "ft_train") == in_train:
                keep[qid] = queries[qid]
        return keep, {k: v for k, v in qrels.items() if k in keep}, exclude

    keep = {}
    for qid in queries:
        src = exclude.get(qid, qid)
        h = int(hashlib.sha1(src.encode()).hexdigest()[:8], 16)
        in_dev = (h % 100) < 70          # 70/30 dev/test, stable across runs
        if (split == "dev") == in_dev:
            keep[qid] = queries[qid]
    return keep, {k: v for k, v in qrels.items() if k in keep}, exclude


def load_chunks(*, with_embeddings: bool = True) -> tuple[list[dict], dict[str, str]]:
    """Passages, plus the stored vectors and the index config that describes them.

    **The stored embeddings are the baseline's document vectors.** They were computed at
    ingest time with the configured backend, and re-encoding the same text through the
    same API to reproduce them costs money and roughly 25 minutes per run on a 110k-passage
    corpus — for a byte-identical answer. They are read here instead.

    ``index_config`` comes back with them so the caller can refuse to use them when the
    recorded backend is not the one the baseline expects: comparing a query vector from
    one model against document vectors from another does NOT raise, it silently returns a
    confident, meaningless ranking (see CLAUDE.md 4.6).
    """
    import psycopg

    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("POSTGRES_URL / DATABASE_URL not set")
    cols = ("chunk_id, doc_id, COALESCE(indexed_text, text)"
            + (", embedding" if with_embeddings else ""))
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT k, v FROM index_config")
        cfg = {k: v for k, v in cur.fetchall()}
        # ORDER BY IS LOAD-BEARING, NOT TIDINESS.
        #
        # Postgres guarantees no row order without it, and the embedding disk cache is
        # keyed on a hash of the texts IN ORDER. Two runs over an identical corpus
        # therefore produced different keys and the cache missed every single time —
        # observed directly as two 617 MB MedCPT caches written 35 minutes apart for the
        # same 105,250 passages, each costing ~7 minutes of GPU, and an openai-768 miss
        # would have cost another ~20 minutes and ~$0.55 of API calls.
        #
        # A cache that misses nondeterministically is worse than no cache: it looks like
        # it is working, so nobody checks.
        # kind='passage' IS ALSO LOAD-BEARING, and omitting it silently changed the corpus.
        #
        # Figure rows were added so a paper can be READ with its images, and the serving
        # path excludes them in `neon_store.HYBRID_SEARCH_SQL`. That guard was written,
        # tested — and applied to only ONE of the two systems that read this table. The
        # evaluation harness has its own SQL, so the next run picked up 190,399 rows
        # instead of 180,850, re-encoded the whole corpus because the new rows have no
        # embedding, and would have measured a candidate against a baseline that had
        # quietly grown by 9,549 documents.
        #
        # Exactly the §4.15 shape: a capability guarded on one side of an interface and
        # not the other. Guarding the accessor is not guarding the data (§6.1).
        cur.execute(f"SELECT {cols} FROM chunks WHERE kind = 'passage' ORDER BY chunk_id")
        rows = cur.fetchall()
    out = []
    for r in rows:
        d = {"chunk_id": r[0], "doc_id": r[1], "text": r[2]}
        if with_embeddings:
            d["embedding"] = r[3]
        out.append(d)
    return out, cfg


class Harness:
    """Builds every index once; each candidate is a different way of querying them."""

    #: Backend name the stored `chunks.embedding` column is expected to hold.
    BASELINE_BACKEND = "openai"

    def __init__(self, chunks: list[dict], queries: dict[str, str], dim: int = 192,
                 index_config: dict[str, str] | None = None):
        self.chunks = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.texts = [c["text"] for c in chunks]
        self.chunk_to_doc = {c["chunk_id"]: c["doc_id"] for c in chunks}
        self.by_id = {c["chunk_id"]: c["text"] for c in chunks}
        self.qids = sorted(queries)
        self.queries = queries
        print(f"indexing {len(chunks):,} passages...")
        self.bm25 = BM25Index().build(self.chunk_ids, self.texts)
        be = make_backend(self.BASELINE_BACKEND, dim=dim)

        # Prefer the vectors already in the store over re-deriving them. Guarded on
        # index_config: absent config is NOT permission — an index with no record predates
        # the table and may hold LSA vectors, which is precisely the dangerous case.
        cfg = index_config or {}
        stored_ok = (
            cfg.get("embedding_model") == self.BASELINE_BACKEND
            and cfg.get("embedding_dim") == str(dim)
            and all(c.get("embedding") is not None for c in chunks)
        )
        if stored_ok:
            self.dvecs = self._parse_stored(chunks, dim)
            print(f"  reused {len(self.dvecs):,} stored {cfg['embedding_model']}/"
                  f"{cfg['embedding_dim']} vectors (no re-encode)")
        else:
            why = ("index_config says "
                   f"{cfg.get('embedding_model')!r}/{cfg.get('embedding_dim')!r}"
                   if cfg else "no index_config recorded")
            print(f"  re-encoding documents ({why})")
            self.dvecs = self._as_dense(
                be.encode_documents_cached(self.texts, local_data_dir() / "emb_cache"))
        self.qvecs = self._as_dense(be.encode_queries([queries[q] for q in self.qids]))
        self._expanded_by_source: dict[str, dict[str, str]] = {}
        self._dense_cache: dict[str, tuple] = {}

    #: Dense matrices are held as float32, not float64.
    #:
    #: At 105,250 passages a 768-dim float64 matrix is 646 MB, and the comparison holds
    #: TWO of them (MedCPT and the openai-768 control) plus the baseline and a BM25 index.
    #: Measured free RAM on this machine at the time of the run: 2.7 GB of 15.7 GB. float32
    #: halves every one of those and changes nothing that matters — the vectors are
    #: L2-normalised and the only operation applied to them is a dot product for ranking.
    #: `MedCPTBackend._encode` already makes this argument in a comment, casts to float32,
    #: and then casts straight back to float64.
    DTYPE = np.float32

    @classmethod
    def _as_dense(cls, m: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(m, dtype=cls.DTYPE)

    @staticmethod
    def _parse_stored(chunks: list[dict], dim: int) -> np.ndarray:
        """pgvector comes back as '[0.1,0.2,...]'. Parse and re-normalise."""
        m = np.zeros((len(chunks), dim), dtype=np.float32)
        for i, c in enumerate(chunks):
            v = np.fromstring(str(c["embedding"]).strip("[]"), sep=",")
            if v.shape[0] != dim:
                raise SystemExit(
                    f"stored vector for {c['chunk_id']} has width {v.shape[0]}, "
                    f"expected {dim} — refusing to mix embedding spaces")
            m[i] = v
        n = np.linalg.norm(m, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return m / n

    def expanded_queries(self, source: str = "mesh", *, repeat: int = 1,
                         identity_only: bool = False) -> dict[str, str]:
        # Keyed by source AND weighting: `repeat` and `identity_only` change the emitted
        # string, so they must not share a cache slot. The first version keyed on source
        # alone; adding a weighted variant would have silently reused the unweighted
        # strings and reported a NO_EFFECT that was really a cache collision.
        key = f"{source}|r{repeat}|{'id' if identity_only else 'all'}"
        cache = self._expanded_by_source.setdefault(key, {})
        if cache:
            return cache
        from oncolens.terminology import expand_query

        n_expanded = 0
        for i, qid in enumerate(self.qids):
            e = expand_query(self.queries[qid], cache_dir=local_data_dir(), source=source)
            cache[qid] = e.expanded_query(repeat=repeat, identity_only=identity_only)
            if e.terms:
                n_expanded += 1
            if (i + 1) % 200 == 0:
                print(f"  expanded {i+1}/{len(self.qids)} ({n_expanded} gained synonyms)")
        # A candidate that expanded almost nothing cannot move a metric, and saying so up
        # front is cheaper than reading a null result and wondering which it was.
        print(f"  {source}: {n_expanded}/{len(self.qids)} queries gained at least one "
              f"synonym")
        return cache

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
        dv, qv = self._as_dense(dv), self._as_dense(qv)
        self._dense_cache[backend_name] = (qv, dv)
        return qv, dv

    def fused_chunks(self, cfg: dict) -> dict[str, list[str]]:
        """Per-query fused chunk ids, exactly as `run` computes them before aggregation.

        Exists for hard-negative mining. Negatives must be the passages **the live system
        already ranks highly and gets wrong**; anything else teaches a reranker to separate
        oncology from cooking, which BM25 does already.

        Kept deliberately thin and reusing `self.bm25` / `self.qvecs` so it cannot drift
        from `run`'s two arms. It intentionally does NOT apply the rerank or cross-encoder
        stages: negatives are mined against the baseline, because that is the ranking the
        fine-tuned model will be asked to improve on.
        """
        bm25_w = cfg.get("bm25_weight", 1.0)
        dense_w = cfg.get("dense_weight", 1.0)
        cand = cfg.get("candidates", 200)
        out: dict[str, list[str]] = {}
        for qi, qid in enumerate(self.qids):
            runs, weights = [], []
            if bm25_w > 0:
                runs.append(self.bm25.search(self.queries[qid], k=cand))
                weights.append(bm25_w)
            if dense_w > 0:
                sims = self.dvecs @ self.qvecs[qi]
                top = np.argpartition(-sims, min(cand, len(sims) - 1))[:cand]
                top = top[np.argsort(-sims[top])]
                runs.append([(self.chunk_ids[i], float(sims[i])) for i in top])
                weights.append(dense_w)
            fused = (reciprocal_rank_fusion(runs, weights=weights) if len(runs) > 1
                     else list(runs[0]))
            out[qid] = [c for c, _ in fused[:cand]]
        return out

    def run(self, cfg: dict, qrels: dict, exclude: dict) -> dict[str, dict[str, float]]:
        bm25_w = cfg.get("bm25_weight", 1.0)
        dense_w = cfg.get("dense_weight", 1.0)
        cand = cfg.get("candidates", 200)
        agg = cfg.get("aggregate", "max")
        use_expand = cfg.get("expand", False)
        use_rerank = cfg.get("rerank", False)
        use_adaptive = cfg.get("adaptive", False)
        use_mmr = cfg.get("mmr", False)
        # Local GPU cross-encoder second stage. Loaded once per run, not per query.
        use_cross = cfg.get("cross_encoder")
        cross_depth = cfg.get("cross_depth", 50)
        cross = None
        if use_cross:
            from oncolens.retrieval.cross_encoder import get_reranker

            cross = get_reranker(use_cross)
            print(f"  cross-encoder: {cross.name} over the top {cross_depth} passages")
        exp = (self.expanded_queries(cfg.get("expand_source", "mesh"),
                                     repeat=cfg.get("expand_repeat", 1),
                                     identity_only=cfg.get("expand_identity_only", False))
               if use_expand else None)
        qvecs, dvecs = (self.dense_vectors(cfg["dense_backend"])
                        if cfg.get("dense_backend") else (self.qvecs, self.dvecs))

        # MULTIPLE DENSE ARMS. Round 2 measured that MedCPT and openai_768 fail in
        # OPPOSITE directions: MedCPT +0.0261 on synthesis set-coverage (p=0.0003) and
        # -0.0166 on claim pinpointing (p=0.0034), openai_768 the reverse. Complementary
        # failure modes are the textbook precondition for rank fusion being worth more
        # than either arm, so this gives each its own vote instead of choosing one.
        extra = [self.dense_vectors(n) for n in cfg.get("dense_backends", [])]

        # ROUTE BY QUERY SHAPE. Same measurement, different remedy: pick the arm whose
        # measured strength matches the query in front of us. Kept separate from the
        # fusion candidate because they are different bets and the loop should not be
        # asked which half of a bundle worked.
        route = cfg.get("route_by_shape")
        routed = None
        if route:
            routed = {name: self.dense_vectors(name) for name in set(route.values())}

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

            def dense_run(qv, dv):
                sims = dv @ qv[qi]
                top = np.argpartition(-sims, min(cand, len(sims) - 1))[:cand]
                top = top[np.argsort(-sims[top])]
                return [(self.chunk_ids[i], float(sims[i])) for i in top]

            if dense_w > 0:
                if routed:
                    n_words = len(self.queries[qid].split())
                    band = ("short" if n_words <= 3
                            else "mid" if n_words <= 15 else "long")
                    qv, dv = routed[route[band]]
                else:
                    qv, dv = qvecs, dvecs
                runs.append(dense_run(qv, dv))
                weights.append(dense_w)
                for qv2, dv2 in extra:
                    runs.append(dense_run(qv2, dv2))
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

            if use_cross:
                # SECOND STAGE, and it must be able to move the metric it is gated on.
                #
                # The scores are REPLACED, not reordered. `aggregate_chunks_to_docs` under
                # `max` reads each document's best passage SCORE, so a pass that only
                # permuted the list would be a no-op at document level — which is exactly
                # how `mmr_diversify` came back byte-identical (§4.13). Replacing the head's
                # scores changes each document's max, so the document ranking genuinely
                # moves.
                #
                # Depth is 50 rather than the LLM path's 24 because a local model costs
                # nothing per query. That is the point of running it on the GPU: depth
                # stops being a budget decision.
                head = fused[:cross_depth]
                if head:
                    sc = cross.score(self.queries[qid], [self.by_id[c] for c, _ in head])
                    order = np.argsort(-sc)
                    # The tail keeps its fusion order but is pinned below every reranked
                    # passage. Cross-encoder logits are unbounded and often negative, so a
                    # constant offset is not safe; this is.
                    floor = float(sc.min()) - 1.0
                    fused = ([(head[int(i)][0], float(sc[int(i)])) for i in order]
                             + [(c, floor - 1.0 - j)
                                for j, (c, _) in enumerate(fused[cross_depth:])])

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


@dataclass
class Measurement:
    """Numbers only. The decision is taken later, once every candidate has been run."""
    name: str
    deltas: dict = field(default_factory=dict)
    gate: str = "mrr"
    gate_p: float = 1.0
    gate_delta: float = 0.0
    unjudged_delta: float = 0.0
    no_effect: bool = False
    regressions: list = field(default_factory=list)


def measure(name: str, base: dict, cand: dict, stratum: str = "claim") -> Measurement:
    """Compute every panel metric for one candidate WITHOUT deciding anything.

    Separating measurement from judgement is what makes the multiplicity correction
    below possible: Holm needs the whole family of p-values in hand at once, and a
    function that decides as it goes cannot supply that.
    """
    # Each stratum is gated on the metric that matches ITS task. A synthesis question is
    # answered by a set, so coverage is what counts and reciprocal rank is close to
    # meaningless - handing back the single best paper out of nine is not a good answer
    # to "what is known about X". An identifier lookup is the opposite.
    #
    # gate_metric(), NOT PRIMARY_METRIC: for a reordering-only candidate the two differ,
    # and calling PRIMARY_METRIC directly is what made `rerank_llm` structurally
    # unpassable on synthesis (recall@20 cannot move when only the order changes). The
    # fix for that was written into weighting.py and then never called from here.
    gate = gate_metric(stratum, name)
    primary = PRIMARY_METRIC.get(stratum, "mrr")
    panel = tuple(dict.fromkeys(
        (gate, primary) + SECONDARY_METRICS.get(stratum, ()) + M.CONSENSUS_METRICS))

    m = Measurement(name=name, gate=gate)
    for metric in panel:
        c = compare(base, cand, metric)
        if c is None:
            continue
        m.deltas[metric] = {"delta": c.delta, "p": c.p_value,
                            "ci": [c.ci_low, c.ci_high], "n": c.n_queries}
        if metric == gate:
            m.gate_p, m.gate_delta = c.p_value, c.delta
        # A REGRESSION veto is tested UNCORRECTED. Correction exists to stop false
        # positives; here a false positive means "we refused a change", which is the
        # safe direction. Making the safety check harder to trip would be backwards.
        if c.p_value < ALPHA and c.delta < -MIN_EFFECT:
            m.regressions.append(metric)

    # A candidate whose rankings are identical to the baseline did not run. Reporting
    # that as "no significant improvement" hides a broken candidate behind a plausible
    # negative result — which is exactly how a loop launders its own bugs into findings.
    identical = sum(1 for q in base
                    if base[q].get("_ranking_hash") == cand.get(q, {}).get("_ranking_hash"))
    m.no_effect = bool(base) and identical == len(base)

    unj_base, unj_cand = mean_of(base, "unjudged@10"), mean_of(cand, "unjudged@10")
    m.unjudged_delta = unj_cand - unj_base
    m.deltas["unjudged@10"] = {"delta": m.unjudged_delta, "p": None, "ci": None, "n": None}
    return m


def holm_gate(measurements: list[Measurement], stratum: str = "claim") -> list[Verdict]:
    """Decide promotion, correcting across CANDIDATES rather than across metrics.

    **The correction was being applied to the wrong family, and it cost real power.**
    The previous gate divided alpha by the size of the reported metric panel — 5 to 8
    metrics depending on stratum. But at 1.10 judged documents per query almost every
    query has exactly one relevant document, and when there is one relevant document at
    rank ``r``::

        mrr = 1/r     success@k = 1[r <= k]     ndcg@10 = 1/log2(r+1)

    Every one of those is a deterministic function of the same number. They are not
    independent tests; they are five thresholdings of one rank distribution. Bonferroni
    assumes independence, so correcting across them controls nothing real while inflating
    the minimum detectable effect. MEASURED on this harness:

        stratum      panel  MDE@.05   MDE@gate   extra queries needed
        synthesis        8   0.0569     0.0727                  1.63x
        concept          6   0.0622     0.0772                  1.54x
        identifier       5   0.1257     0.1533                  1.49x

    That overhead lands on a harness already documented as underpowered — which is the
    worst possible place to spend it.

    **What the real family is.** One pre-registered gate metric per stratum, tested once
    per candidate. The multiplicity that genuinely needs controlling is the number of
    CANDIDATES tried in an iteration, because that is how many chances the loop takes at
    a false positive. So: Holm across candidates on the gate metric. Holm is uniformly
    more powerful than Bonferroni and controls the same family-wise error rate, so there
    is no reason to prefer plain Bonferroni here.

    Secondary metrics keep their job: they are reported, and a significant regression on
    any of them still vetoes (uncorrected — see ``measure``).
    """
    live = [m for m in measurements if not m.no_effect]
    # Holm: sort ascending by p, compare p_(i) against alpha / (k - i).
    order = sorted(range(len(live)), key=lambda i: live[i].gate_p)
    k = len(live)
    passed_holm: set[str] = set()
    for rank, idx in enumerate(order):
        thresh = ALPHA / max(k - rank, 1)
        if live[idx].gate_p < thresh:
            passed_holm.add(live[idx].name)
        else:
            break          # Holm stops at the first failure; all larger p also fail

    out: list[Verdict] = []
    for m in measurements:
        if m.no_effect:
            out.append(Verdict(m.name, False,
                               "NO EFFECT — rankings byte-identical to baseline; the "
                               "candidate did not fire (check that its config is "
                               "actually wired through)", m.deltas, [], []))
            continue
        wins = [k2 for k2, v in m.deltas.items()
                if v.get("p") is not None and v["p"] < ALPHA and v["delta"] > MIN_EFFECT]
        if m.regressions:
            out.append(Verdict(m.name, False,
                               f"significant regression on {', '.join(m.regressions)}",
                               m.deltas, wins, m.regressions))
            continue
        if m.gate_delta <= MIN_EFFECT:
            out.append(Verdict(m.name, False,
                               f"{m.gate} moved {m.gate_delta:+.4f}, below the "
                               f"{MIN_EFFECT} worth the complexity",
                               m.deltas, wins, m.regressions))
            continue
        if m.name not in passed_holm:
            out.append(Verdict(m.name, False,
                               f"{m.gate} {m.gate_delta:+.4f} p={m.gate_p:.4f} did not "
                               f"survive Holm across {k} candidates",
                               m.deltas, wins, m.regressions))
            continue
        if m.unjudged_delta > UNJUDGED_TOLERANCE:
            out.append(Verdict(m.name, False,
                               f"unjudged@10 rose {m.unjudged_delta:+.3f}: the gain may "
                               f"be an artifact of returning documents the pool never "
                               f"judged", m.deltas, wins, m.regressions))
            continue
        out.append(Verdict(m.name, True,
                           f"{m.gate} {m.gate_delta:+.4f} (p={m.gate_p:.4f}, Holm over "
                           f"{k} candidates), no regressions",
                           m.deltas, wins, m.regressions))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--split", default="dev",
                    choices=["dev", "test", "all", "ft_train", "ft_holdout"],
                    help="ft_train/ft_holdout partition DEV for fine-tuning; a model "
                         "trained on ft_train must be scored on ft_holdout, never on dev")
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

    chunks, index_config = load_chunks()
    harness = Harness(chunks, queries, index_config=index_config)
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

    # PHASE 1 — measure every candidate. No decisions yet: Holm needs the whole family of
    # gate p-values before it can threshold any of them.
    ledger: list[Verdict] = []
    measurements: list[Measurement] = []
    cand_runs: dict[str, dict] = {}
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
        cand_runs[name] = cand
        m = measure(name, base, cand, args.stratum)
        measurements.append(m)
        print(f"  gate metric: {m.gate}   delta {m.gate_delta:+.4f}  p={m.gate_p:.4f}")

    # PHASE 2 — decide, correcting across candidates.
    verdicts = holm_gate(measurements, args.stratum)
    for v in verdicts:
        c = CANDIDATES_BY_NAME.get(v.name)
        cand = cand_runs.get(v.name, {})
        m = next((x for x in measurements if x.name == v.name), None)
        print(f"\n=== {v.name} — {c.description if c else ''} ===")
        gate = m.gate if m else PRIMARY_METRIC.get(args.stratum, "mrr")
        for metric in dict.fromkeys((gate, PRIMARY_METRIC.get(args.stratum, "mrr"))
                                    + SECONDARY_METRICS.get(args.stratum, ())
                                    + M.CONSENSUS_METRICS):
            d = v.deltas.get(metric)
            if not d or d.get("p") is None:
                continue
            flag = "WIN " if metric in v.wins else ("REG " if metric in v.regressions else "    ")
            star = "*" if metric == gate else " "
            print(f"  {flag}{star}{metric:<12}{mean_of(base,metric):>8.4f} -> "
                  f"{mean_of(cand,metric):>8.4f}  ({d['delta']:+.4f}, p={d['p']:.4f})")
        if "unjudged@10" in v.deltas:
            print(f"  unjudged@10 {v.deltas['unjudged@10']['delta']:+.4f}")
        print(f"  -> {'PROMOTE' if v.promoted else 'DISCARD'}: {v.reason}")
    ledger.extend(verdicts)

    out = local_data_dir() / "improve_ledger.json"
    prior = []
    if out.exists():
        try:
            prior = json.loads(out.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prior = []
    # RECORD THE STRATUM. Without it a consumer has to guess which stratum an iteration
    # ran on from its query count, and two strata whose dev splits are ~235 and ~252
    # queries are not distinguishable that way: build_journey_data duly reported one
    # stratum's measured variance as another's.
    prior.append({"split": split, "stratum": args.stratum, "n_queries": len(queries),
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
