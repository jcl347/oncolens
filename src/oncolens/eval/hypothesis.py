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
    "adaptive_weights": Hypothesis(
        candidate="adaptive_weights", stratum="concept", metric="success@5",
        direction="up", min_effect=0.01,
        mechanism="Round 1 showed dropping BM25 hurts short queries and doubling it hurts "
                  "long conceptual ones. Weighting by query length should capture both "
                  "ends instead of compromising between them. Predicted on concept because "
                  "that stratum is uniformly short, so the rule fires on every query.",
        null_strata=(),
    ),
    "mmr_diversify": Hypothesis(
        candidate="mmr_diversify", stratum="synthesis", metric="recall@20",
        direction="up", min_effect=0.01,
        mechanism="Depth and aggregation both moved nothing on synthesis, so the right "
                  "passages are found and then crowded out by near-duplicates from the "
                  "same paper. Capping per-document contribution should raise SET coverage "
                  "specifically - which is exactly what recall@20 measures.",
        null_strata=("identifier",),
    ),
    "medcpt": Hypothesis(
        candidate="medcpt", stratum="concept", metric="success@5",
        direction="up", min_effect=0.02,
        mechanism="MedCPT is trained on 255M PubMed click logs, which are SHORT queries "
                  "against biomedical articles. The concept stratum is 2-word MeSH terms - "
                  "the same distribution. A general embedder trained mostly on long web "
                  "text has least advantage exactly there.",
        null_strata=("claim",),
    ),
    "openai_768": Hypothesis(
        candidate="openai_768", stratum="concept", metric="success@5",
        direction="up", min_effect=0.02,
        mechanism="Capacity control for MedCPT. If simply widening the SAME model to 768 "
                  "dimensions recovers most of any MedCPT gain, then the gain is capacity "
                  "rather than domain training, and the cheap servable change wins. If it "
                  "does not, the domain-training claim survives a real test.",
        null_strata=(),
    ),
    "tri_fusion": Hypothesis(
        candidate="tri_fusion", stratum="synthesis", metric="recall@20",
        direction="up", min_effect=0.02,
        mechanism="Round 2 measured MedCPT and openai_768 failing in opposite directions "
                  "on the same corpus: MedCPT +0.0261 recall@20 on synthesis and -0.0166 "
                  "mrr on claim, openai_768 the reverse. RRF pays off exactly when arms "
                  "are individually competent and wrong about different queries, so three "
                  "arms should hold MedCPT's coverage gain WITHOUT its attribution cost. "
                  "Predicted on synthesis because that is where the larger of the two "
                  "measured effects lives.",
        # The whole point is that fusion should not inherit MedCPT's regression. If claim
        # drops anyway, the arms are not complementary in the way the deltas suggested.
        null_strata=("claim",),
    ),
    "dense_weight_2x": Hypothesis(
        candidate="dense_weight_2x", stratum="synthesis", metric="recall@20",
        direction="up",
        # Bounded by the two controls that have already run. dual_dense (3 arms AND the
        # 1:2 ratio) bought +0.0105; tri_fusion_balanced (MedCPT at 1:1) bought +0.0021.
        # If the ratio is the active ingredient, the weight-only cell should land near
        # dual_dense's +0.0105, so anything at or above half of that is the prediction
        # holding. Registering 0.02 would be registering tri_fusion's number for a
        # configuration that does not contain tri_fusion's third arm.
        min_effect=0.005,
        mechanism="Isolates the lexical:dense RATIO from the arm COUNT, which no control "
                  "so far separates. Three equal-weight arms silently make the dense side "
                  "outvote BM25 2:1; this reproduces that ratio with two arms. A gain here "
                  "means the third arm was a delivery mechanism for a weight change, and "
                  "the same effect is available for free. A null here means the extra "
                  "VOTER matters independently of the weight, which is the interesting "
                  "case and the one that would justify the fusion machinery.",
        null_strata=(),
    ),
    "tri_fusion_balanced": Hypothesis(
        candidate="tri_fusion_balanced", stratum="synthesis", metric="recall@20",
        direction="up", min_effect=0.02,
        mechanism="WEIGHT control for tri_fusion. tri_fusion runs three arms at equal "
                  "weight, which silently moves lexical:dense from 1:1 to 1:2 as well as "
                  "adding MedCPT. Restoring 1:1 by weighting BM25 2x isolates the two. If "
                  "recall@20 still gains ~0.03 the effect is MedCPT's complementarity; if "
                  "it collapses toward zero the effect was the rebalancing, and "
                  "tri_fusion's headline is a weight change wearing a model's name.",
        null_strata=(),
    ),
    "dual_dense": Hypothesis(
        candidate="dual_dense", stratum="synthesis", metric="recall@20",
        direction="up", min_effect=0.02,
        mechanism="ARM-COUNT control for tri_fusion. Three RRF voters may beat two for "
                  "reasons that have nothing to do with the third voter's content, so "
                  "this adds a third arm carrying almost no new information: the same "
                  "OpenAI model at 192 alongside itself at 768. A gain here would mean "
                  "the fusion geometry is doing the work. PREDICTED TO FAIL, which is the "
                  "point of a control: if it matches tri_fusion, the expensive candidate "
                  "is buying nothing that a second copy of a cheap arm would not.",
        null_strata=(),
    ),
    "route_by_shape": Hypothesis(
        candidate="route_by_shape", stratum="synthesis", metric="recall@20",
        direction="up",
        # DERIVED, not chosen. The routing bands were measured against the live query set
        # first: 870 of 1,272 synthesis queries (68.4%) fall in the 4-15 word band and
        # reach MedCPT, against only 317 of 3,998 claim queries (7.9%). If MedCPT's
        # measured +0.0261 on synthesis is roughly uniform across its queries, routing
        # 68.4% of them to it predicts 0.684 x 0.0261 = +0.0179. Registering 0.02 here
        # would have been registering a number the mechanism cannot produce, and then
        # explaining away the miss.
        min_effect=0.015,
        mechanism="Sends mid-length topical queries to MedCPT and everything else to "
                  "openai_768, which is the per-stratum measurement applied per query. "
                  "Query length separates the two shapes well enough to route on: 68% of "
                  "synthesis lands in the MedCPT band against 8% of claim, so it should "
                  "capture most of the coverage gain while exposing almost none of the "
                  "attribution cost. What this really tests is whether the per-stratum "
                  "effects are properties of the QUERIES or of the strata as pools.",
        null_strata=("claim",),
    ),
    "rerank_medcpt_cross": Hypothesis(
        candidate="rerank_medcpt_cross", stratum="claim", metric="mrr",
        direction="up", min_effect=0.02,
        mechanism="A bi-encoder embeds query and passage separately, so it cannot "
                  "represent an interaction between them: it can tell a passage is ABOUT "
                  "a topic, not whether it REPORTS the specific thing asked. A "
                  "cross-encoder attends across both. Predicted on claim because a claim "
                  "query is a 28-word sentence with exactly one correct source, which is "
                  "the case where reading the pair together should matter most and where "
                  "n=5,000 can actually resolve it. Larger min_effect than usual because "
                  "reranking is the single most-reported win in the RAG literature; if it "
                  "moves less than 0.02 here, this pipeline is not where the headroom is.",
        # Coverage should barely move: the tail is pinned below the reranked head, so
        # which documents reach the top 20 changes only at the margin.
        null_strata=("synthesis",),
    ),
    "rerank_minilm_cross": Hypothesis(
        candidate="rerank_minilm_cross", stratum="claim", metric="mrr",
        direction="up", min_effect=0.02,
        mechanism="DOMAIN control for rerank_medcpt_cross. A general MS MARCO reranker "
                  "with no biomedical training. If it matches the biomedical one, the "
                  "gain is cross-attention and any reranker will do, which is the cheap "
                  "and portable outcome. If it does not, the gain is domain training. "
                  "Registered with the SAME threshold as the treatment on purpose: a "
                  "control held to a lower bar is not a control.",
        null_strata=(),
    ),
    "expand_ontology": Hypothesis(
        candidate="expand_ontology", stratum="identifier", metric="success@1",
        direction="up", min_effect=0.02,
        mechanism="Identifier queries are bare symbols and success@1 is 0.148, the worst "
                  "number in the system, on the stratum where a wrong answer costs most. "
                  "BM25 scores EGFR, ERBB1 and HER1 as unrelated tokens; a curated "
                  "registry states that they are one gene. Expansion should help "
                  "precisely where the query IS such a symbol.\n\n"
                  "REVISED 2026-08-03, BEFORE ANY MEASUREMENT OF THIS CANDIDATE. The "
                  "original registration named HGNC + MeSH, measured end to end at 58.5% "
                  "of the 335 identifier queries. The stratum is not mostly genes - it is "
                  "cell lines (MCF-7, UACC-812), investigational drug codes (CALAA-01, "
                  "HKI-272), trial acronyms (PALOMA-3, x7), HLA alleles, protein variants "
                  "and an EORTC questionnaire. The candidate now runs a five-registry "
                  "cascade (NCIt, Cellosaurus, ClinicalTrials.gov, HGNC, MeSH) reaching "
                  "87.2%, and resolves each term to a TYPE rather than a bare string.\n\n"
                  "WHAT THAT DOES TO THE GATE. A candidate firing on fraction f of "
                  "queries needs a per-affected-query effect of 0.02/f on success@1 to "
                  "move the stratum mean by 0.02: 0.034 at 58.5%, 0.023 at 87.2%. Both "
                  "are attainable, so - unlike 4.8 and 4.13 - this gate was NOT "
                  "unreachable, and an earlier draft of this note claiming it was had "
                  "quoted HGNC's 13.7% as if it described the shipped configuration.\n\n"
                  "The risk being tested is the mirror of the reward: expansion injects "
                  "terms the user did not type. 4.4's ER case is now understood better - "
                  "NCIt resolves ER correctly, but to FOURTEEN concepts including the "
                  "endoplasmic reticulum and Eritrea, so the hazard is silent sense "
                  "selection rather than one vocabulary's error. A regression here would "
                  "say the guards are not tight enough, not that synonymy does not matter.",
        null_strata=("claim",),
    ),
    "expand_identity_weighted": Hypothesis(
        candidate="expand_identity_weighted", stratum="identifier", metric="success@1",
        direction="up", min_effect=0.02,
        mechanism="REGISTERED BEFORE THE RUN, AFTER expand_ontology WAS REFUTED. That "
                  "candidate reached 89.8% of queries (212/236) and still regressed "
                  "success@1 by 0.0339 at p=0.037, so the failure was not coverage. The "
                  "shape of the regression says where it was: success@1 fell 0.0339 and "
                  "success@20 only 0.0254 (p=0.27, not significant). Expansion was pulling "
                  "candidates IN while pushing the right one DOWN - a precision loss at "
                  "the top, which is what injecting undifferentiated extra terms into a "
                  "lexical arm does.\n\n"
                  "Two specific defects are fixed here: synonyms were injected at EQUAL "
                  "weight to the user's own words (MCF-7 became 1 of 7 tokens), and "
                  "association links were treated as synonyms (PALOMA-3 -> Palbociclib, "
                  "Fulvestrant, Placebo, which stops the query being about one trial).\n\n"
                  "PREDICTION: identity-only synonyms with the query repeated 3x recovers "
                  "the baseline and improves on it, because the remaining expansions are "
                  "genuine spelling variants of exactly what was asked for (MCF-7 -> MCF7, "
                  "EGFR T790M -> EGFR p.T790M) and those cannot broaden a query.\n\n"
                  "WHAT A NULL RESULT WOULD MEAN, stated now so it cannot be "
                  "reinterpreted later: if this also fails to beat baseline, the honest "
                  "conclusion is that lexical synonym injection does not help the "
                  "identifier stratum, and the thread is retired rather than tuned "
                  "further. Two registered failures on the same mechanism is evidence "
                  "about the mechanism, not an invitation to a third parameter sweep.",
        null_strata=("claim",),
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
