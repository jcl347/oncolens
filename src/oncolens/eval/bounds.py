"""Bounding metrics under incomplete judgments, instead of guessing.

Standard practice treats an unjudged document as non-relevant. That is a *guess*, and it is
biased in a specific direction: it always understates a system that surfaces documents the
pool has not seen — which is exactly what a genuinely new retrieval approach does.

Rather than guess, compute both extremes:

* **pessimistic** — every unjudged document in the ranking is non-relevant (the usual
  assumption, and a true lower bound);
* **optimistic** — every unjudged document in the ranking is as relevant as the best judged
  document for that query (a true upper bound).

The true score lies between them. The payoff is a **dominance test**: if

    pessimistic(B) > optimistic(A)

then B beats A *no matter how the unjudged documents would be judged*. That conclusion
requires no further labelling effort and cannot be overturned by pooling. When the
intervals overlap, the honest answer is "not yet determined" — and the gap between the
bounds tells you exactly how much judging would be needed to settle it.

This turns "we have incomplete judgments" from an unquantified caveat into a bounded,
reportable quantity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .metrics import REL_THRESHOLD, _relevant_ids


@dataclass(frozen=True)
class Bounded:
    pessimistic: float
    optimistic: float

    @property
    def width(self) -> float:
        return self.optimistic - self.pessimistic

    def __repr__(self) -> str:
        return f"[{self.pessimistic:.4f}, {self.optimistic:.4f}]"


def _max_grade(judgments: Mapping[str, int]) -> int:
    rel = [g for g in judgments.values() if g >= REL_THRESHOLD]
    return max(rel) if rel else 1


def ndcg_bounds(ranking: Sequence[str], judgments: Mapping[str, int], k: int) -> Bounded | None:
    """Lower/upper bound on nDCG@k over all possible labellings of unjudged documents."""
    if not _relevant_ids(judgments):
        return None
    top = list(ranking[:k])
    unjudged_in_top = [d for d in top if d not in judgments]
    g_max = _max_grade(judgments)

    # --- pessimistic: unjudged contribute nothing --------------------------
    dcg_lo = sum(
        (2 ** judgments.get(d, 0) - 1) / math.log2(i + 1)
        for i, d in enumerate(top, start=1)
        if judgments.get(d, 0) > 0
    )
    ideal_lo = sorted((g for g in judgments.values() if g > 0), reverse=True)[:k]
    idcg_lo = sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(ideal_lo, start=1))
    lo = (dcg_lo / idcg_lo) if idcg_lo > 0 else 0.0

    # --- optimistic: unjudged in the top-k are maximally relevant ----------
    # The ideal ranking must also account for those newly-relevant documents, otherwise the
    # "bound" could exceed 1.0 and would not be a bound at all.
    dcg_hi = 0.0
    for i, d in enumerate(top, start=1):
        g = judgments.get(d, g_max if d in unjudged_in_top else 0)
        if g > 0:
            dcg_hi += (2**g - 1) / math.log2(i + 1)
    ideal_hi = sorted(
        [g for g in judgments.values() if g > 0] + [g_max] * len(unjudged_in_top), reverse=True
    )[:k]
    idcg_hi = sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(ideal_hi, start=1))
    hi = (dcg_hi / idcg_hi) if idcg_hi > 0 else 0.0

    return Bounded(pessimistic=min(lo, hi), optimistic=max(lo, hi))


def precision_bounds(ranking: Sequence[str], judgments: Mapping[str, int], k: int) -> Bounded | None:
    """Precision@k bounds — the cleanest case, since the denominator is fixed at k."""
    if k <= 0 or not judgments:
        return None
    top = list(ranking[:k])
    rel = _relevant_ids(judgments)
    hits = sum(1 for d in top if d in rel)
    unjudged = sum(1 for d in top if d not in judgments)
    return Bounded(pessimistic=hits / k, optimistic=min(1.0, (hits + unjudged) / k))


@dataclass
class DominanceReport:
    metric: str
    n_queries: int
    b_dominates_a: int      # queries where B wins regardless of unjudged labelling
    a_dominates_b: int
    undetermined: int
    mean_pess_a: float
    mean_opt_a: float
    mean_pess_b: float
    mean_opt_b: float
    mean_interval_width: float

    @property
    def robust_verdict(self) -> str:
        if self.mean_pess_b > self.mean_opt_a:
            return "B DOMINATES — holds under every possible labelling of unjudged documents"
        if self.mean_pess_a > self.mean_opt_b:
            return "A DOMINATES — holds under every possible labelling of unjudged documents"
        return (
            "UNDETERMINED — the bound intervals overlap, so the ranking of these two "
            "configurations depends on documents nobody has judged. Judge the pool gap "
            "before treating the point estimate as a result."
        )

    def summary(self) -> str:
        return (
            f"{self.metric}: A=[{self.mean_pess_a:.4f}, {self.mean_opt_a:.4f}]  "
            f"B=[{self.mean_pess_b:.4f}, {self.mean_opt_b:.4f}]  "
            f"mean interval width={self.mean_interval_width:.4f}\n"
            f"  per-query: B dominates {self.b_dominates_a}, A dominates {self.a_dominates_b}, "
            f"undetermined {self.undetermined}\n"
            f"  {self.robust_verdict}"
        )


def dominance_report(
    runs_a: Mapping[str, Sequence[str]],
    runs_b: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    k: int = 10,
    metric: str = "ndcg@10",
) -> DominanceReport | None:
    """Compare two runs using bounds rather than point estimates.

    A per-query "dominance" means one system's worst case still beats the other's best
    case. Those queries are settled permanently. The rest are honestly undetermined.
    """
    fn = ndcg_bounds if metric.startswith("ndcg") else precision_bounds
    pa, oa, pb, ob, widths = [], [], [], [], []
    b_dom = a_dom = undet = 0

    for qid in sorted(set(runs_a) & set(runs_b)):
        j = qrels.get(qid)
        if not j:
            continue
        ba, bb = fn(runs_a[qid], j, k), fn(runs_b[qid], j, k)
        if ba is None or bb is None:
            continue
        pa.append(ba.pessimistic); oa.append(ba.optimistic)
        pb.append(bb.pessimistic); ob.append(bb.optimistic)
        widths.extend([ba.width, bb.width])
        if bb.pessimistic > ba.optimistic:
            b_dom += 1
        elif ba.pessimistic > bb.optimistic:
            a_dom += 1
        else:
            undet += 1

    n = len(pa)
    if n == 0:
        return None
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    return DominanceReport(
        metric=metric, n_queries=n,
        b_dominates_a=b_dom, a_dominates_b=a_dom, undetermined=undet,
        mean_pess_a=mean(pa), mean_opt_a=mean(oa),
        mean_pess_b=mean(pb), mean_opt_b=mean(ob),
        mean_interval_width=mean(widths) if widths else 0.0,
    )
