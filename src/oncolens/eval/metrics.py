"""Retrieval metrics with graded, incomplete judgments.

Design notes (why these metrics, and what each one is defended against):

* Real relevance judgments are *incomplete*. Any metric that treats "unjudged" as
  "non-relevant" will penalise a system precisely when it surfaces a good document the
  pool never saw. That is the failure mode that makes homemade RAG evals punish real
  improvements, so we carry ``bpref`` (which only counts *judged* non-relevant documents)
  alongside the rank metrics, and we report ``unjudged_at_k`` as a first-class diagnostic.

* Relevance is graded (0..3). Binary metrics collapse "the definitive paper" and "mentions
  it once" into the same thing, so nDCG uses exponential gain.

* A judgment of ``0`` means "a human checked this and it is not relevant". That is
  strictly more information than *absent*. ``bpref`` and ``unjudged_at_k`` are the two
  places the distinction actually matters, and both rely on it.

All functions take ``ranking`` (list of doc_id, best first) and ``judgments``
(dict doc_id -> graded relevance). No global state, no numpy requirement.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

# A document counts as "relevant" for set-based metrics at or above this grade.
REL_THRESHOLD = 1

#: bpref needs a usable denominator. Below this many judged negatives it is noise, so it
#: is reported as undefined rather than fed into the consensus panel.
MIN_JUDGED_NEGATIVES_FOR_BPREF = 10


def _relevant_ids(judgments: Mapping[str, int], threshold: int = REL_THRESHOLD) -> set[str]:
    return {d for d, g in judgments.items() if g >= threshold}


def _judged_nonrelevant_ids(judgments: Mapping[str, int], threshold: int = REL_THRESHOLD) -> set[str]:
    """Explicitly judged non-relevant. NOT the same as 'everything else'."""
    return {d for d, g in judgments.items() if g < threshold}


def recall_at_k(ranking: Sequence[str], judgments: Mapping[str, int], k: int) -> float | None:
    rel = _relevant_ids(judgments)
    if not rel:
        return None  # undefined for no-answer queries; caller must not average this away
    return len(rel & set(ranking[:k])) / len(rel)


def precision_at_k(ranking: Sequence[str], judgments: Mapping[str, int], k: int) -> float | None:
    if not judgments:
        return None
    if k == 0:
        return None
    rel = _relevant_ids(judgments)
    return len(rel & set(ranking[:k])) / k


def mrr(ranking: Sequence[str], judgments: Mapping[str, int]) -> float | None:
    rel = _relevant_ids(judgments)
    if not rel:
        return None
    for i, doc in enumerate(ranking, start=1):
        if doc in rel:
            return 1.0 / i
    return 0.0


def average_precision(ranking: Sequence[str], judgments: Mapping[str, int]) -> float | None:
    """MAP component. Unjudged documents occupy rank positions but never score."""
    rel = _relevant_ids(judgments)
    if not rel:
        return None
    hits = 0
    total = 0.0
    for i, doc in enumerate(ranking, start=1):
        if doc in rel:
            hits += 1
            total += hits / i
    return total / len(rel)


def ndcg_at_k(ranking: Sequence[str], judgments: Mapping[str, int], k: int) -> float | None:
    """Exponential-gain nDCG.

    gain = 2**rel - 1, discount = 1/log2(rank+1). The ideal ranking is the judged
    documents sorted by grade descending, truncated at k. Unjudged documents contribute
    zero gain but still consume a rank slot, which is the correct (pessimistic) treatment.
    """
    if not _relevant_ids(judgments):
        return None
    dcg = 0.0
    for i, doc in enumerate(ranking[:k], start=1):
        g = judgments.get(doc, 0)
        if g > 0:
            dcg += (2**g - 1) / math.log2(i + 1)
    ideal_grades = sorted((g for g in judgments.values() if g > 0), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(ideal_grades, start=1))
    if idcg == 0:
        return None
    return dcg / idcg


def bpref(ranking: Sequence[str], judgments: Mapping[str, int]) -> float | None:
    """Binary preference — the incomplete-judgment-robust metric.

    For each relevant document r, count how many *judged non-relevant* documents were
    ranked above it. Unjudged documents are skipped entirely rather than counted against
    the system, which is exactly why this metric does not punish a system for finding
    something the pool missed.

    bpref = (1/R) * sum_r [ 1 - min(|n ranked above r|, R) / min(R, N) ]
    """
    rel = _relevant_ids(judgments)
    nonrel = _judged_nonrelevant_ids(judgments)
    R, N = len(rel), len(nonrel)
    if R == 0:
        return None
    if N < MIN_JUDGED_NEGATIVES_FOR_BPREF:
        # With a denominator of min(R, N), a handful of judged negatives makes bpref
        # swing wildly: at N=2, a single judged-nonrelevant document ranked above a
        # relevant one erases half that document's credit. An audit measured a mean
        # denominator of 2.0 on this qrels, where bpref is noise rather than a
        # robustness signal. Returning None is honest; averaging it into a consensus
        # vote would not be.
        return None
    denom = min(R, N)
    seen_nonrel = 0
    total = 0.0
    for doc in ranking:
        if doc in rel:
            total += 1.0 - (min(seen_nonrel, R) / denom)
        elif doc in nonrel:
            seen_nonrel += 1
        # unjudged: deliberately ignored
    return total / R


def unjudged_at_k(ranking: Sequence[str], judgments: Mapping[str, int], k: int) -> float:
    """Diagnostic, not a quality score.

    Fraction of the top-k carrying no judgment at all. A configuration whose unjudged@k
    jumps relative to the pool baseline is being measured unfairly: its true score is an
    underestimate, and its comparison against other arms is not trustworthy until those
    documents are judged. Every experiment report surfaces this.
    """
    if k == 0:
        return 0.0
    top = ranking[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d not in judgments) / len(top)


# ---------------------------------------------------------------------------
# no_answer stratum: different question, different metrics.
# ---------------------------------------------------------------------------

def abstained(ranking: Sequence[str]) -> float:
    """1.0 if the system correctly returned nothing for an unanswerable query."""
    return 1.0 if len(ranking) == 0 else 0.0


def false_positives_at_k(ranking: Sequence[str], k: int) -> float:
    """Count of confidently-returned documents for a query with no relevant documents."""
    return float(len(ranking[:k]))


# ---------------------------------------------------------------------------
# Metric panel
# ---------------------------------------------------------------------------

#: Metrics computed for every answerable query. Deliberately a panel and not a single
#: number: a change that improves one metric while degrading three is not an improvement,
#: it is a trade the promotion gate should refuse without a human saying otherwise.
PANEL_K = (1, 5, 10, 20, 50)

#: The single metric used for ordering candidates. It is *not* sufficient for promotion.
PRIMARY = "ndcg@10"


def evaluate_query(ranking: Sequence[str], judgments: Mapping[str, int]) -> dict[str, float]:
    """Compute the full metric panel for one query.

    Returns only defined metrics. A missing key means "undefined for this query", which
    is different from 0.0 and must not be imputed as zero when aggregating.
    """
    out: dict[str, float] = {}
    is_no_answer = not _relevant_ids(judgments)

    if is_no_answer:
        out["abstained"] = abstained(ranking)
        for k in (5, 10):
            out[f"false_pos@{k}"] = false_positives_at_k(ranking, k)
        return out

    for k in PANEL_K:
        for name, fn in (("recall", recall_at_k), ("precision", precision_at_k), ("ndcg", ndcg_at_k)):
            v = fn(ranking, judgments, k)
            if v is not None:
                out[f"{name}@{k}"] = v
        out[f"unjudged@{k}"] = unjudged_at_k(ranking, judgments, k)

    for name, fn in (("mrr", mrr), ("map", average_precision), ("bpref", bpref)):
        v = fn(ranking, judgments)
        if v is not None:
            out[name] = v
    return out


#: Metrics that must move *together* for the promotion gate to call a change a real
#: improvement rather than a metric-specific artifact. Chosen to span set-based recall,
#: graded ranking quality, early precision, and incomplete-judgment robustness.
CONSENSUS_METRICS = ("ndcg@10", "recall@10", "recall@20", "map", "mrr", "bpref")
