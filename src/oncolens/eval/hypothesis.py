"""Pre-registered hypotheses — a change must predict its effect before it is measured.

**The failure this prevents.** Run seven candidates against four strata and twenty
metrics and something will look significant. Pick the winner afterwards, write a
persuasive reason for it, and you have a finding that is indistinguishable from noise —
and it will be *more* convincing than a real result, because the story was written to fit
the data.

So every candidate states, **before** the run:

* which stratum it should help, and which it should not
* which metric should move, and in which direction
* the smallest effect that would count as the prediction coming true
* why — the mechanism, not the hope

The ledger then records the prediction *and* the outcome, and classifies the pair:

===============  ==========================================================
outcome          meaning
===============  ==========================================================
``CONFIRMED``    predicted effect, observed
``REFUTED``      predicted effect, not observed
``NO_EFFECT``    the candidate produced identical output — it never ran
``SURPRISE``     a significant effect somewhere it was NOT predicted
===============  ==========================================================

``SURPRISE`` is deliberately not treated as success. An unpredicted win is the single
most likely thing to be a fluke of multiple comparisons, and promoting on it is how a
loop drifts. It is recorded, and it becomes the *hypothesis* for the next round, where
it gets a fair test on data that has not already been looked at.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Hypothesis:
    """What a change is expected to do, written down first."""

    candidate: str
    stratum: str                 #: where the effect is predicted
    metric: str                  #: which metric should move
    direction: str               #: "up" | "down"
    min_effect: float            #: smallest change that counts as the prediction holding
    mechanism: str               #: WHY, in terms of how retrieval works
    #: Strata where this should make no material difference. Naming these is what turns a
    #: vague "it should help" into something that can be wrong.
    null_strata: tuple[str, ...] = ()

    def describe(self) -> str:
        arrow = "increase" if self.direction == "up" else "decrease"
        s = (f"{self.candidate}: expect {self.metric} on '{self.stratum}' to {arrow} "
             f"by >= {self.min_effect:.3f}")
        if self.null_strata:
            s += f"; no material change on {', '.join(self.null_strata)}"
        return s


@dataclass
class Outcome:
    hypothesis: Hypothesis
    observed: dict = field(default_factory=dict)     #: stratum -> {metric: delta, p}
    verdict: str = "UNTESTED"
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "candidate": self.hypothesis.candidate,
            "predicted": self.hypothesis.describe(),
            "mechanism": self.hypothesis.mechanism,
            "verdict": self.verdict,
            "detail": self.detail,
            "observed": self.observed,
        }


def classify(h: Hypothesis, observed: dict, *, alpha: float = 0.05,
             no_effect: bool = False) -> Outcome:
    """Compare what was predicted against what happened.

    ``observed`` maps stratum -> {metric: {"delta": float, "p": float}}.
    """
    o = Outcome(hypothesis=h, observed=observed)
    if no_effect:
        o.verdict = "NO_EFFECT"
        o.detail = ("output identical to baseline — the candidate did not fire, so this "
                    "is a wiring bug, not evidence about the hypothesis")
        return o

    target = observed.get(h.stratum, {}).get(h.metric)
    if target is None:
        o.verdict = "UNTESTED"
        o.detail = f"{h.metric} was not measured on '{h.stratum}'"
        return o

    delta, p = target.get("delta", 0.0), target.get("p", 1.0)
    want_up = h.direction == "up"
    moved_right_way = (delta >= h.min_effect) if want_up else (delta <= -h.min_effect)
    significant = p < alpha

    if moved_right_way and significant:
        o.verdict = "CONFIRMED"
        o.detail = f"{h.metric} on {h.stratum} moved {delta:+.4f} (p={p:.4f})"
    else:
        o.verdict = "REFUTED"
        why = ("effect too small" if significant else "not significant")
        o.detail = f"{h.metric} on {h.stratum} moved {delta:+.4f} (p={p:.4f}) — {why}"

    # Did it do something notable where it was predicted to do nothing?
    surprises = []
    for stratum in h.null_strata:
        for metric, v in observed.get(stratum, {}).items():
            if v.get("p", 1.0) < alpha and abs(v.get("delta", 0.0)) >= h.min_effect:
                surprises.append(f"{metric} on {stratum} {v['delta']:+.4f}")
    if surprises:
        o.detail += (f" | UNPREDICTED effect: {'; '.join(surprises[:3])} — recorded as a "
                     f"hypothesis for the next round, not counted as success")
        if o.verdict == "REFUTED":
            o.verdict = "SURPRISE"
    return o


#: Predictions for the registered candidates. Written before any of them was run against
#: the synthesis stratum, so they can be wrong.
HYPOTHESES: dict[str, Hypothesis] = {
    "expand_mesh": Hypothesis(
        candidate="expand_mesh", stratum="identifier", metric="success@1",
        direction="up", min_effect=0.02,
        mechanism="Gene and drug symbols have brand/code synonyms (osimertinib/AZD9291/"
                  "Tagrisso) that BM25 scores as unrelated tokens. Expansion should help "
                  "exactly where the query IS such a symbol.",
        null_strata=("claim", "synthesis"),
    ),
    "dense_only": Hypothesis(
        candidate="dense_only", stratum="identifier", metric="success@1",
        direction="down", min_effect=0.02,
        mechanism="Embeddings smooth rare literals toward semantically similar tokens, "
                  "so dropping the lexical arm should hurt exact lookup most.",
        null_strata=(),
    ),
    "lexical_heavy": Hypothesis(
        candidate="lexical_heavy", stratum="identifier", metric="success@1",
        direction="up", min_effect=0.02,
        mechanism="If the dense arm dilutes exact matches, weighting BM25 higher should "
                  "recover them — the mirror of the dense_only prediction.",
        null_strata=("synthesis",),
    ),
    "topn_decay": Hypothesis(
        candidate="topn_decay", stratum="synthesis", metric="recall@20",
        direction="up", min_effect=0.01,
        mechanism="A review section is answered by papers relevant THROUGHOUT, not by one "
                  "decisive passage. Aggregating a document's top-3 passages should suit "
                  "that better than taking its single best.",
        null_strata=("identifier",),
    ),
    "deep_candidates": Hypothesis(
        candidate="deep_candidates", stratum="synthesis", metric="recall@20",
        direction="up", min_effect=0.01,
        mechanism="Answer sets of 3-22 papers need more candidates to surface than a "
                  "single-target lookup does; 200 may truncate the tail of the set.",
        null_strata=("identifier",),
    ),
    "rerank_llm": Hypothesis(
        candidate="rerank_llm", stratum="concept", metric="success@5",
        direction="up", min_effect=0.02,
        mechanism="A cross-encoder reads query and passage together, so it can tell "
                  "'reports this finding' from 'mentions it in passing' — which is the "
                  "distinction a 2-word topical query cannot express on its own.",
        null_strata=(),
    ),
}
