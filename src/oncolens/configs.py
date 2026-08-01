"""The iteration ladder: which hypothesis each round tests, and why.

Each iteration changes **one family of knobs** so an effect can be attributed. Sweeping
everything simultaneously produces a winner nobody can explain and cannot be defended when
it later regresses.

Ordering is deliberate — earlier rounds settle the coarse architecture (which arms exist,
how they fuse) before later rounds spend statistical power on fine tuning. Every dev draw
tightens the Bonferroni-adjusted alpha, so cheap sweeps early are expensive later.
"""

from __future__ import annotations

from .retrieval.pipeline import RetrievalConfig

#: The most defensible starting point: a single well-implemented lexical retriever.
#: Starting from hybrid would hide whether the dense arm earns its keep.
BASELINE = RetrievalConfig(
    name="i0_bm25_only",
    use_bm25=True, use_dense=False,
    index_unit="chunk", chunk_agg="max",
    expansion="none", abstain=False,
)


def iteration_1_arms() -> list[RetrievalConfig]:
    """Does a dense arm help at all, and does fusing beat either arm alone?"""
    return [
        BASELINE.variant("i1_dense_only", use_bm25=False, use_dense=True),
        BASELINE.variant("i1_hybrid_rrf", use_dense=True, fusion="rrf"),
        BASELINE.variant("i1_hybrid_score", use_dense=True, fusion="score"),
    ]


def iteration_2_fusion(champ: RetrievalConfig) -> list[RetrievalConfig]:
    """Fusion shape: how sharply should top ranks dominate, and how to weight arms."""
    return [
        champ.variant("i2_rrf_k20", fusion="rrf", rrf_k=20.0),
        champ.variant("i2_rrf_k10", fusion="rrf", rrf_k=10.0),
        champ.variant("i2_bm25_heavy", fusion="rrf", bm25_weight=1.5, dense_weight=1.0),
        champ.variant("i2_dense_heavy", fusion="rrf", bm25_weight=1.0, dense_weight=1.5),
    ]


def iteration_3_chunk_agg(champ: RetrievalConfig) -> list[RetrievalConfig]:
    """One decisive passage vs. sustained relevance across a document."""
    return [
        champ.variant("i3_agg_topn_decay", chunk_agg="topn_decay", chunk_agg_topn=3, chunk_agg_decay=0.6),
        champ.variant("i3_agg_topn_flat", chunk_agg="topn_decay", chunk_agg_topn=3, chunk_agg_decay=1.0),
        champ.variant("i3_agg_sum", chunk_agg="sum"),
        champ.variant("i3_agg_mean", chunk_agg="mean"),
    ]


def iteration_4_literals(champ: RetrievalConfig) -> list[RetrievalConfig]:
    """Domain hypothesis: identifier-shaped tokens deserve more IDF weight than word tokens."""
    return [
        champ.variant("i4_literal_1p5", literal_boost=1.5),
        champ.variant("i4_literal_2p0", literal_boost=2.0),
        champ.variant("i4_literal_3p0", literal_boost=3.0),
    ]


def iteration_5_expansion(champ: RetrievalConfig) -> list[RetrievalConfig]:
    """Ontology expansion: recall gain vs. precision dilution, and where to apply it."""
    return [
        champ.variant("i5_exp_bm25_narrow", expansion="lexicon", exp_apply_to="bm25",
                      exp_max_variants=2, exp_max_total=8),
        champ.variant("i5_exp_bm25_wide", expansion="lexicon", exp_apply_to="bm25",
                      exp_max_variants=4, exp_max_total=24),
        champ.variant("i5_exp_both", expansion="lexicon", exp_apply_to="both",
                      exp_max_variants=3, exp_max_total=16),
    ]


def iteration_6_indexing(champ: RetrievalConfig) -> list[RetrievalConfig]:
    """Index granularity and whether heading context earns its tokens."""
    return [
        champ.variant("i6_doc_unit", index_unit="doc"),
        champ.variant("i6_no_heading", include_heading=False),
        champ.variant("i6_dense_384", dense_dim=384),
        champ.variant("i6_dense_96", dense_dim=96),
    ]


def iteration_7_bm25_params(champ: RetrievalConfig) -> list[RetrievalConfig]:
    """Length normalisation. Grant sections and paper sections differ a lot in length."""
    return [
        champ.variant("i7_b_0p4", bm25_b=0.4),
        champ.variant("i7_b_0p9", bm25_b=0.9),
        champ.variant("i7_k1_0p9", bm25_k1=0.9),
        champ.variant("i7_k1_1p6", bm25_k1=1.6),
    ]


def iteration_8_abstention(champ: RetrievalConfig) -> list[RetrievalConfig]:
    """no_answer handling — the stratum every recall-only harness ignores."""
    return [
        champ.variant("i8_abstain_lo", abstain=True, abstain_bm25_floor=2.0),
        champ.variant("i8_abstain_mid", abstain=True, abstain_bm25_floor=4.0),
        champ.variant("i8_abstain_hi", abstain=True, abstain_bm25_floor=6.0),
        champ.variant("i8_abstain_rel", abstain=True, abstain_bm25_floor=2.0, abstain_rel_floor=0.35),
    ]


LADDER = [
    ("arms", iteration_1_arms),
    ("fusion", iteration_2_fusion),
    ("chunk_agg", iteration_3_chunk_agg),
    ("literals", iteration_4_literals),
    ("expansion", iteration_5_expansion),
    ("indexing", iteration_6_indexing),
    ("bm25_params", iteration_7_bm25_params),
    ("abstention", iteration_8_abstention),
]
