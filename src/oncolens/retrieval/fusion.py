"""Rank/score fusion.

RRF is the default because it operates on **ranks**, sidestepping the score-incompatibility
problem: BM25 scores are unbounded and corpus-dependent while cosine similarities live in
[-1, 1], so a naive weighted sum of raw scores is dominated by whichever arm happens to
have the larger numeric range. Score-normalised fusion is provided too, because RRF throws
away margin information and sometimes that margin is the signal (notably when one arm is
extremely confident about an exact identifier match).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def reciprocal_rank_fusion(
    runs: Sequence[Sequence[tuple[str, float]]],
    *,
    k: float = 60.0,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """RRF: score(d) = sum_r w_r / (k + rank_r(d)).

    ``k`` damps the influence of top ranks; the conventional 60 means rank 1 and rank 2
    differ by only ~1.6%, which is intentionally forgiving toward disagreeing arms.
    """
    w = list(weights) if weights is not None else [1.0] * len(runs)
    agg: dict[str, float] = {}
    for run, wi in zip(runs, w):
        for rank, (doc_id, _score) in enumerate(run, start=1):
            agg[doc_id] = agg.get(doc_id, 0.0) + wi / (k + rank)
    return sorted(agg.items(), key=lambda x: (-x[1], x[0]))


def _minmax(run: Sequence[tuple[str, float]]) -> dict[str, float]:
    if not run:
        return {}
    vals = [s for _, s in run]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {d: 1.0 for d, _ in run}
    return {d: (s - lo) / (hi - lo) for d, s in run}


def score_fusion(
    runs: Sequence[Sequence[tuple[str, float]]],
    *,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Min-max normalise each run, then take a weighted sum.

    Documents absent from a run score 0 for that arm rather than being dropped, so a
    document found by only one arm is penalised but not excluded.
    """
    w = list(weights) if weights is not None else [1.0] * len(runs)
    norms = [_minmax(r) for r in runs]
    keys: set[str] = set()
    for n in norms:
        keys |= set(n)
    agg = {d: sum(wi * n.get(d, 0.0) for n, wi in zip(norms, w)) for d in keys}
    return sorted(agg.items(), key=lambda x: (-x[1], x[0]))


FUSIONS = {"rrf": reciprocal_rank_fusion, "score": score_fusion}


def aggregate_chunks_to_docs(
    ranked_chunks: Sequence[tuple[str, float]],
    chunk_to_doc: Mapping[str, str],
    *,
    strategy: str = "max",
    top_n: int = 3,
    decay: float = 0.6,
) -> list[tuple[str, float]]:
    """Collapse a chunk-level ranking to a document-level ranking.

    This is a real quality knob, not plumbing. ``max`` rewards a single decisive passage;
    ``topn_decay`` rewards a document that is relevant throughout. Which one wins is
    query-type dependent, so it is exposed to the experiment loop rather than hard-coded.
    """
    per_doc: dict[str, list[float]] = {}
    for chunk_id, score in ranked_chunks:
        doc = chunk_to_doc.get(chunk_id)
        if doc is None:
            continue
        per_doc.setdefault(doc, []).append(score)

    out: dict[str, float] = {}
    for doc, scores in per_doc.items():
        scores.sort(reverse=True)
        if strategy == "max":
            out[doc] = scores[0]
        elif strategy == "sum":
            out[doc] = sum(scores)
        elif strategy == "mean":
            out[doc] = sum(scores) / len(scores)
        elif strategy == "topn_decay":
            out[doc] = sum(s * (decay**i) for i, s in enumerate(scores[:top_n]))
        else:
            raise ValueError(f"unknown chunk aggregation strategy: {strategy}")
    return sorted(out.items(), key=lambda x: (-x[1], x[0]))
