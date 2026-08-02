"""How much each query type counts, and why — grounded in how the tool gets used.

A single pooled mean over all strata would be dishonest twice over: the ``claim`` stratum
is four times larger than any other and would dominate by sample size alone, and the
strata do not measure equally important things.

**The production workflow this is weighted for.** An oncology R&D team using this to
review a target or a mechanism does roughly four things, in descending frequency:

1. *"What's known about X?"* — the literature-review question. Answered by a **set** of
   papers covering a mechanism, not one paper. This is the core of the job and the reason
   the tool exists; it is also the hardest to fake, because returning one excellent paper
   and nothing else scores as incomplete.
2. *"Papers on <concept>"* — a two-word topical query, typed dozens of times a day while
   scoping. High volume, low ceremony.
3. *"<gene> <variant>"* — an exact lookup. Lower volume, but **the highest cost of
   failure**: silently returning papers about a *different* variant is worse than
   returning nothing, because the mistake is invisible in the results.
4. *Pasting a sentence from a paper being read* to find its source. Real, but secondary,
   and the 27-word queries in the claim stratum are the only place this shape occurs.

**Weights are a statement about the product, not about the data.** They are recorded here
so the choice is arguable rather than buried in a script.
"""

from __future__ import annotations

#: Relative importance of each stratum in the composite score.
#:
#: These do NOT let a strong stratum rescue a weak one — per-stratum gating still vetoes
#: any significant regression independently. The weights decide only which improvements
#: are worth *pursuing*, not which are safe to ship.
STRATUM_WEIGHTS: dict[str, float] = {
    "synthesis": 0.35,   # "what's known about X" — the literature-review question
    "concept": 0.30,     # short topical queries, the highest-volume shape
    "identifier": 0.20,  # exact lookup; low volume, highest cost of being wrong
    "claim": 0.15,       # pasting a sentence to find its source; real but secondary
}

#: The metric that best represents each stratum's *task*.
#:
#: Deliberately not uniform. A synthesis question is answered by a SET, so coverage of
#: that set is what matters and reciprocal rank is close to meaningless — being handed the
#: single best paper out of nine is not a good answer to "what is known about X". An
#: identifier lookup is the opposite: there is one right paper and it belongs at rank 1.
PRIMARY_METRIC: dict[str, str] = {
    "synthesis": "recall@20",
    "concept": "success@5",
    "identifier": "success@1",
    "claim": "mrr",
}

#: Which metric family a change is CAPABLE of moving.
#:
#: Learned from a failure of this harness. `rerank_llm` reorders the top 24 chunks; it
#: cannot change WHICH documents are in the top 20, so recall@20 moved by exactly 0.0000
#: on the synthesis stratum. Because synthesis is gated on recall@20, the reranker was
#: structurally unable to pass however good it was, and its significant gains on mrr
#: (+0.0123, p=0.030) and ndcg@10 (+0.0048, p=0.012) were discarded on a technicality.
#:
#: Gating a change on a metric it cannot affect is not a strict standard, it is a broken
#: one: it produces confident negatives that carry no information.
ORDERING_ONLY = frozenset({"rerank_llm", "topn_decay"})
#: For those, judge on a rank-sensitive metric even where the stratum's task metric is
#: coverage. Coverage is still reported, and a coverage REGRESSION still vetoes.
ORDERING_METRIC: dict[str, str] = {
    "synthesis": "ndcg@10",
    "concept": "success@1",
    "identifier": "success@1",
    "claim": "mrr",
}


def gate_metric(stratum: str, candidate: str) -> str:
    """The metric this candidate should be judged on for this stratum."""
    if candidate in ORDERING_ONLY:
        return ORDERING_METRIC.get(stratum, "mrr")
    return PRIMARY_METRIC.get(stratum, "mrr")


#: Reported alongside, never used for promotion on its own.
SECONDARY_METRICS: dict[str, tuple[str, ...]] = {
    "synthesis": ("recall@10", "ndcg@10", "success@5"),
    "concept": ("success@10", "mrr", "ndcg@10"),
    "identifier": ("success@5", "mrr"),
    "claim": ("success@5", "success@10"),
}


def composite(scores_by_stratum: dict[str, dict[str, float]]) -> float | None:
    """Weighted score across strata, using each stratum's own primary metric.

    Returns ``None`` when a stratum is missing rather than silently renormalising over
    the ones present: a composite computed over three strata is not comparable to one
    computed over four, and quietly making it look comparable is how a partial run gets
    mistaken for a full one.
    """
    total = 0.0
    for stratum, weight in STRATUM_WEIGHTS.items():
        metric = PRIMARY_METRIC[stratum]
        got = scores_by_stratum.get(stratum, {}).get(metric)
        if got is None:
            return None
        total += weight * got
    return total


def describe() -> str:
    lines = ["stratum      weight  primary metric  rationale"]
    why = {
        "synthesis": "answer is a SET; coverage is the task",
        "concept": "highest-volume query shape",
        "identifier": "one right paper; wrong is worse than empty",
        "claim": "find-the-source; secondary workflow",
    }
    for s, w in sorted(STRATUM_WEIGHTS.items(), key=lambda kv: -kv[1]):
        lines.append(f"{s:<13}{w:>5.2f}  {PRIMARY_METRIC[s]:<15} {why[s]}")
    return "\n".join(lines)
