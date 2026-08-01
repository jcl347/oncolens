"""Config-driven retrieval pipeline.

Every knob that could plausibly change quality is a field on ``RetrievalConfig``, so an
"experiment" is exactly a config plus a corpus, and two experiments differing by one field
are a clean A/B. Nothing that affects ranking is allowed to live as a module-level
constant, because a hidden constant is a confound the ledger cannot see.

The pipeline returns document-level rankings (what the qrels judge) *and* the passage
evidence behind each document (what the product must display), so provenance is never
reconstructed after the fact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, asdict, replace

from .chunking import Chunk, chunk_corpus
from .dense import DenseIndex, LsaBackend
from .expansion import Lexicon
from .fusion import aggregate_chunks_to_docs, reciprocal_rank_fusion, score_fusion
from .lexical import BM25Index
from .text import tokenize


@dataclass(frozen=True)
class RetrievalConfig:
    name: str = "baseline"

    # --- indexing ---
    index_unit: str = "chunk"          # "chunk" | "doc"
    include_heading: bool = True       # prepend "<title> — <section>" to indexed text

    # --- lexical arm ---
    use_bm25: bool = True
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    literal_boost: float = 1.0         # IDF multiplier for identifier-looking tokens

    # --- dense arm ---
    use_dense: bool = True
    dense_dim: int = 192
    dense_backend: str = "lsa"         # "lsa" | "voyage"

    # --- fusion ---
    fusion: str = "rrf"                # "rrf" | "score"
    rrf_k: float = 60.0
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    candidates_per_arm: int = 200

    # --- query expansion ---
    expansion: str = "none"            # "none" | "lexicon"
    exp_max_variants: int = 4
    exp_max_total: int = 24
    exp_apply_to: str = "bm25"         # "bm25" | "both"

    # --- chunk -> doc aggregation ---
    chunk_agg: str = "max"             # "max" | "sum" | "mean" | "topn_decay"
    chunk_agg_topn: int = 3
    chunk_agg_decay: float = 0.6

    # --- output / abstention ---
    top_k: int = 50
    abstain: bool = False
    abstain_bm25_floor: float = 0.0    # min raw BM25 evidence required to answer at all
    abstain_rel_floor: float = 0.0     # drop docs scoring below this fraction of the top

    def as_dict(self) -> dict:
        return asdict(self)

    def variant(self, name: str, **kw) -> "RetrievalConfig":
        return replace(self, name=name, **kw)


@dataclass
class Evidence:
    """The passage that justifies a document's presence in the results."""
    chunk_id: str
    doc_id: str
    section: str
    text: str
    start_char: int
    end_char: int
    score: float


@dataclass
class SearchResult:
    ranking: list[str] = field(default_factory=list)          # doc_ids, best first
    scores: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    abstained: bool = False
    debug: dict = field(default_factory=dict)


class Retriever:
    def __init__(self, config: RetrievalConfig, lexicon: Lexicon | None = None) -> None:
        self.config = config
        self.lexicon = lexicon or Lexicon({})
        self.chunks: list[Chunk] = []
        self.unit_ids: list[str] = []
        self.unit_to_doc: dict[str, str] = {}
        self.chunk_by_id: dict[str, Chunk] = {}
        self.bm25: BM25Index | None = None
        self.dense: DenseIndex | None = None

    # -- build ------------------------------------------------------------
    def build(self, docs: Sequence[Mapping], contexts: Mapping[str, str] | None = None) -> "Retriever":
        """``contexts`` maps chunk_id -> Contextual-Retrieval situating sentence."""
        cfg = self.config
        self.chunks = chunk_corpus(list(docs))
        self.chunk_by_id = {c.chunk_id: c for c in self.chunks}
        contexts = contexts or {}

        if cfg.index_unit == "chunk":
            self.unit_ids = [c.chunk_id for c in self.chunks]
            texts = [
                c.indexable_text(include_heading=cfg.include_heading, context=contexts.get(c.chunk_id))
                for c in self.chunks
            ]
            self.unit_to_doc = {c.chunk_id: c.doc_id for c in self.chunks}
        else:
            by_doc: dict[str, list[str]] = {}
            for c in self.chunks:
                by_doc.setdefault(c.doc_id, []).append(
                    c.indexable_text(include_heading=cfg.include_heading, context=contexts.get(c.chunk_id))
                )
            self.unit_ids = list(by_doc)
            texts = ["\n".join(v) for v in by_doc.values()]
            self.unit_to_doc = {d: d for d in self.unit_ids}

        if cfg.use_bm25:
            self.bm25 = BM25Index(
                k1=cfg.bm25_k1, b=cfg.bm25_b, literal_boost=cfg.literal_boost
            ).build(self.unit_ids, texts)
        if cfg.use_dense:
            backend = LsaBackend(dim=cfg.dense_dim) if cfg.dense_backend == "lsa" else _voyage()
            self.dense = DenseIndex(backend).build(self.unit_ids, texts)
        return self

    # -- search -----------------------------------------------------------
    def search(self, query: str) -> SearchResult:
        cfg = self.config
        expanded, added = (query, [])
        if cfg.expansion == "lexicon":
            expanded, added = self.lexicon.expand_query(
                query, max_variants=cfg.exp_max_variants, max_total_added=cfg.exp_max_total
            )

        runs: list[list[tuple[str, float]]] = []
        weights: list[float] = []
        raw_bm25_top = 0.0

        if cfg.use_bm25 and self.bm25 is not None:
            q = expanded if cfg.expansion != "none" else query
            run = self.bm25.search(q, k=cfg.candidates_per_arm)
            raw_bm25_top = run[0][1] if run else 0.0
            runs.append(run); weights.append(cfg.bm25_weight)

        if cfg.use_dense and self.dense is not None:
            q = expanded if (cfg.expansion != "none" and cfg.exp_apply_to == "both") else query
            runs.append(self.dense.search(q, k=cfg.candidates_per_arm))
            weights.append(cfg.dense_weight)

        if not runs:
            return SearchResult(abstained=True, debug={"reason": "no arms enabled"})

        if len(runs) == 1:
            fused = list(runs[0])
        elif cfg.fusion == "rrf":
            fused = reciprocal_rank_fusion(runs, k=cfg.rrf_k, weights=weights)
        else:
            fused = score_fusion(runs, weights=weights)

        doc_ranked = (
            aggregate_chunks_to_docs(
                fused, self.unit_to_doc,
                strategy=cfg.chunk_agg, top_n=cfg.chunk_agg_topn, decay=cfg.chunk_agg_decay,
            )
            if cfg.index_unit == "chunk"
            else fused
        )

        # --- abstention -------------------------------------------------
        abstained = False
        if cfg.abstain:
            if raw_bm25_top < cfg.abstain_bm25_floor:
                abstained = True
                doc_ranked = []
            elif cfg.abstain_rel_floor > 0 and doc_ranked:
                top = doc_ranked[0][1]
                if top > 0:
                    doc_ranked = [(d, s) for d, s in doc_ranked if s >= cfg.abstain_rel_floor * top]

        doc_ranked = doc_ranked[: cfg.top_k]

        # --- passage evidence -------------------------------------------
        evidence: dict[str, Evidence] = {}
        if cfg.index_unit == "chunk":
            best: dict[str, tuple[str, float]] = {}
            for unit_id, score in fused:
                doc = self.unit_to_doc.get(unit_id)
                if doc is None:
                    continue
                if doc not in best or score > best[doc][1]:
                    best[doc] = (unit_id, score)
            for doc, _ in doc_ranked:
                hit = best.get(doc)
                if hit and hit[0] in self.chunk_by_id:
                    c = self.chunk_by_id[hit[0]]
                    evidence[doc] = Evidence(
                        chunk_id=c.chunk_id, doc_id=c.doc_id, section=c.section, text=c.text,
                        start_char=c.start_char, end_char=c.end_char, score=hit[1],
                    )

        return SearchResult(
            ranking=[d for d, _ in doc_ranked],
            scores=dict(doc_ranked),
            evidence=evidence,
            abstained=abstained,
            debug={"expanded_terms": added, "raw_bm25_top": raw_bm25_top,
                   "query_tokens": len(tokenize(query))},
        )


def _voyage():
    from .dense import VoyageBackend
    return VoyageBackend()
