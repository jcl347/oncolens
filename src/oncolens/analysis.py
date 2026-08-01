"""Failure analysis: turn measurement into the next hypothesis.

A sweep that only reports numbers cannot tell you what to try next. This module answers
the question the numbers raise — *why* did a query fail — and distinguishes failure modes
that call for completely different fixes:

| Diagnosis | Meaning | The fix it argues for |
|---|---|---|
| ``not_retrieved`` | Relevant doc absent even from the deep candidate pool | Recall problem: expansion, a better dense arm, deeper candidates |
| ``retrieved_not_ranked`` | Found deep in the pool but not surfaced | Ranking problem: fusion weights, reranking, chunk aggregation |
| ``vocabulary_gap`` | Query terms have no lexical match anywhere in the corpus | Expansion / semantic problem — BM25 structurally cannot help |
| ``precision_leak`` | Judged-non-relevant documents outrank relevant ones | Precision problem: over-expansion, weak discrimination |

Conflating these is how tuning loops waste iterations — adding synonyms to fix what was
actually a ranking problem, or reweighting fusion to fix what was actually a vocabulary gap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .data import Dataset, Query
from .eval.metrics import ndcg_at_k
from .retrieval.pipeline import Retriever
from .retrieval.text import tokenize


@dataclass
class QueryDiagnosis:
    query_id: str
    query: str
    stratum: str
    score: float
    diagnosis: str
    detail: str
    missed: list[str]
    deep_rank_of_first_relevant: int | None
    unmatched_terms: list[str]


def diagnose(
    query: Query,
    ranking: Sequence[str],
    retriever: Retriever,
    *,
    k: int = 10,
    deep_k: int = 200,
) -> QueryDiagnosis:
    """Classify one query's failure mode."""
    relevant = {d for d, g in query.judgments.items() if g >= 1}
    nonrel = {d for d, g in query.judgments.items() if g == 0}
    score = ndcg_at_k(ranking, query.judgments, k) or 0.0
    top = list(ranking[:k])
    missed = sorted(relevant - set(top))

    # Terms with no posting list at all -> BM25 has nothing to match on.
    unmatched: list[str] = []
    if retriever.bm25 is not None:
        unmatched = sorted({t for t in tokenize(query.query) if retriever.bm25.df.get(t, 0) == 0})

    # Where does the first relevant document appear in a much deeper pool?
    deep_rank: int | None = None
    saved_top_k = retriever.config.top_k
    try:
        object.__setattr__(retriever.config, "top_k", deep_k)
        deep = retriever.search(query.query).ranking
    finally:
        object.__setattr__(retriever.config, "top_k", saved_top_k)
    for i, d in enumerate(deep, start=1):
        if d in relevant:
            deep_rank = i
            break

    if not relevant:
        return QueryDiagnosis(
            query.query_id, query.query, query.stratum, score,
            "no_answer_query", f"returned {len(top)} docs; correct behaviour is to return none",
            [], None, unmatched,
        )

    n_nonrel_above = sum(1 for d in top if d in nonrel)
    first_rel_pos = next((i for i, d in enumerate(top, start=1) if d in relevant), None)

    if deep_rank is None:
        diagnosis = "not_retrieved"
        detail = f"no relevant doc in the top {deep_k}; candidate generation is the bottleneck"
        if unmatched:
            diagnosis = "vocabulary_gap"
            detail = (
                f"{len(unmatched)} query term(s) match nothing in the corpus "
                f"({', '.join(unmatched[:5])}); lexical retrieval cannot help here"
            )
    elif deep_rank > k:
        diagnosis = "retrieved_not_ranked"
        detail = f"first relevant doc sits at deep rank {deep_rank}, outside top {k}; ranking problem"
    elif n_nonrel_above and (first_rel_pos is None or n_nonrel_above >= first_rel_pos):
        diagnosis = "precision_leak"
        detail = f"{n_nonrel_above} judged-non-relevant doc(s) rank inside the top {k}"
    elif score < 0.5:
        diagnosis = "partial_ordering"
        detail = f"relevant docs present but poorly ordered (ndcg@{k}={score:.3f})"
    else:
        diagnosis = "ok"
        detail = f"ndcg@{k}={score:.3f}"

    return QueryDiagnosis(
        query.query_id, query.query, query.stratum, score, diagnosis, detail,
        missed, deep_rank, unmatched,
    )


def analyze_run(
    dataset: Dataset,
    per_query: Mapping[str, Mapping[str, float]],
    runs: Mapping[str, Sequence[str]],
    retriever: Retriever,
    *,
    split: str = "dev",
    worst_n: int = 15,
) -> dict:
    """Diagnose the worst-scoring queries and summarise failure modes by stratum."""
    queries = {q.query_id: q for q in dataset.split(split)}
    scored = sorted(
        ((qid, per_query.get(qid, {}).get("ndcg@10", 0.0)) for qid in runs if qid in queries),
        key=lambda x: x[1],
    )
    diagnoses = [
        diagnose(queries[qid], runs[qid], retriever) for qid, _ in scored[:worst_n]
    ]

    counts: dict[str, int] = {}
    by_stratum: dict[str, dict[str, int]] = {}
    for d in diagnoses:
        counts[d.diagnosis] = counts.get(d.diagnosis, 0) + 1
        by_stratum.setdefault(d.stratum, {}).setdefault(d.diagnosis, 0)
        by_stratum[d.stratum][d.diagnosis] += 1

    return {
        "worst_queries": [vars(d) for d in diagnoses],
        "failure_mode_counts": dict(sorted(counts.items(), key=lambda x: -x[1])),
        "failure_modes_by_stratum": by_stratum,
        "next_hypothesis": _suggest(counts),
    }


def _suggest(counts: Mapping[str, int]) -> list[str]:
    """Map the dominant failure mode to the knob family worth trying next."""
    out: list[str] = []
    ranked = sorted(counts.items(), key=lambda x: -x[1])
    for mode, n in ranked:
        if mode == "vocabulary_gap":
            out.append(f"vocabulary_gap x{n}: try ontology expansion (iteration_5) or a stronger dense arm")
        elif mode == "not_retrieved":
            out.append(f"not_retrieved x{n}: raise candidates_per_arm, or the dense arm is too weak")
        elif mode == "retrieved_not_ranked":
            out.append(f"retrieved_not_ranked x{n}: fusion weights / rrf_k / chunk aggregation (iterations 2-3)")
        elif mode == "precision_leak":
            out.append(f"precision_leak x{n}: narrow expansion, raise literal_boost, or add reranking")
        elif mode == "partial_ordering":
            out.append(f"partial_ordering x{n}: chunk aggregation strategy and BM25 length normalisation")
        elif mode == "no_answer_query":
            out.append(f"no_answer x{n}: tune abstention thresholds (iteration_8)")
    return out


def compare_queries(
    dataset: Dataset,
    a_per_query: Mapping[str, Mapping[str, float]],
    b_per_query: Mapping[str, Mapping[str, float]],
    *,
    split: str = "dev",
    metric: str = "ndcg@10",
    n: int = 10,
) -> dict:
    """Where did B beat A, and where did it lose? The losses are the interesting half.

    A change with a good mean that loses badly on a handful of queries is a trade, not an
    improvement, and the loop should see which queries it traded away.
    """
    queries = {q.query_id: q for q in dataset.split(split)}
    deltas = []
    for qid in set(a_per_query) & set(b_per_query):
        va, vb = a_per_query[qid].get(metric), b_per_query[qid].get(metric)
        if va is None or vb is None:
            continue
        q = queries.get(qid)
        deltas.append({
            "query_id": qid,
            "query": q.query if q else "",
            "stratum": q.stratum if q else "unknown",
            "a": va, "b": vb, "delta": vb - va,
        })
    deltas.sort(key=lambda d: d["delta"])
    return {
        "biggest_losses": deltas[:n],
        "biggest_wins": deltas[-n:][::-1],
        "n_improved": sum(1 for d in deltas if d["delta"] > 1e-9),
        "n_degraded": sum(1 for d in deltas if d["delta"] < -1e-9),
        "n_unchanged": sum(1 for d in deltas if abs(d["delta"]) <= 1e-9),
    }
