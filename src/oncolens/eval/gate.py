"""The promotion gate: what "consensus improvement" means, made executable.

A candidate configuration is promoted (and committed) only when it clears *all* of the
following. Each rule exists because of a specific way a naive "the mean went up" check
gets fooled.

| # | Rule | Fooled-by case it blocks |
|---|------|--------------------------|
| 1 | Primary metric improves, significant at the **Bonferroni-adjusted** alpha | Noise winning after many draws against the same dev set |
| 2 | >= ``CONSENSUS_MIN`` of the metric panel move up, and **none** moves significantly down | A change that games one metric (e.g. pads recall@50 while wrecking precision@1) |
| 3 | **No stratum regresses** significantly beyond tolerance | Aggregate gain that hides a collapse — e.g. semantics improve while exact-ID lookup dies |
| 4 | ``no_answer`` behaviour does not degrade | A system that returns more documents everywhere looks better on recall and is worse in production |
| 5 | ``unjudged@10`` does not spike | The candidate is surfacing documents the pool never judged, so its score is an artifact, not a result. Pool and re-judge before deciding |
| 6 | Underpowered strata are reported, never silently passed | Claiming "no regression" on a stratum too small to detect one |

Rule 5 is the one most often missing from homemade harnesses, and it cuts both ways: it
protects against both flattering *and* punishing a genuinely novel candidate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from . import stats as st
from .metrics import CONSENSUS_METRICS, PRIMARY

#: How many of the consensus panel must improve.
CONSENSUS_MIN = 4
#: A stratum may drop by at most this much on the primary metric, even non-significantly.
STRATUM_TOLERANCE = 0.02
#: Rise in unjudged@10 above which the comparison is declared unreliable.
UNJUDGED_SPIKE = 0.15


@dataclass
class GateResult:
    promoted: bool
    reasons: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    comparisons: dict[str, dict] = field(default_factory=dict)
    stratum_deltas: dict[str, dict] = field(default_factory=dict)
    alpha_used: float = 0.05

    def summary(self) -> str:
        head = "PROMOTE" if self.promoted else "HOLD"
        lines = [f"[{head}] alpha={self.alpha_used:.5f}"]
        for r in self.reasons:
            lines.append(f"  + {r}")
        for b in self.blockers:
            lines.append(f"  x {b}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


def mean_defined(values: Sequence[float | None]) -> float | None:
    """Mean over defined values only. Never imputes a missing metric as zero."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def aggregate(
    per_query: Mapping[str, Mapping[str, float]], metric: str
) -> float | None:
    return mean_defined([q.get(metric) for q in per_query.values()])


def by_stratum(
    per_query: Mapping[str, Mapping[str, float]],
    strata: Mapping[str, str],
) -> dict[str, dict[str, Mapping[str, float]]]:
    """Partition per-query results by stratum label."""
    out: dict[str, dict[str, Mapping[str, float]]] = {}
    for qid, m in per_query.items():
        out.setdefault(strata.get(qid, "unknown"), {})[qid] = m
    return out


def _sd(vals: Sequence[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    mu = sum(vals) / n
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1))


def evaluate_gate(
    baseline: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
    strata: Mapping[str, str],
    ledger: st.Ledger,
    *,
    base_alpha: float = 0.05,
) -> GateResult:
    """Run every gate rule and return a decision with a full audit trail."""
    alpha = ledger.adjusted_alpha(base_alpha)
    res = GateResult(promoted=False, alpha_used=alpha)

    # --- Rule 1: primary metric, significance at adjusted alpha -------------
    primary = st.compare(baseline, candidate, PRIMARY)
    if primary is None:
        res.blockers.append(f"{PRIMARY}: too few paired queries to compare")
        return res
    res.comparisons[PRIMARY] = st.comparison_to_dict(primary)
    if primary.delta <= 0:
        res.blockers.append(
            f"{PRIMARY} did not improve (delta={primary.delta:+.4f}, "
            f"{primary.wins}W/{primary.losses}L/{primary.ties}T)"
        )
    elif not primary.significant_at(alpha):
        res.blockers.append(
            f"{PRIMARY} improved (delta={primary.delta:+.4f}) but p={primary.p_value:.4f} "
            f">= adjusted alpha {alpha:.5f} after {ledger.dev_draws} dev draws"
        )
    else:
        res.reasons.append(
            f"{PRIMARY} {primary.mean_a:.4f} -> {primary.mean_b:.4f} "
            f"(delta={primary.delta:+.4f}, p={primary.p_value:.4f}, d={primary.effect_size:+.2f}, "
            f"{primary.wins}W/{primary.losses}L)"
        )

    # --- Rule 2: consensus across the metric panel ---------------------------
    improved = 0
    for m in CONSENSUS_METRICS:
        c = st.compare(baseline, candidate, m)
        if c is None:
            continue
        res.comparisons[m] = st.comparison_to_dict(c)
        if c.delta > 0:
            improved += 1
        if c.delta < 0 and c.significant_at(alpha):
            res.blockers.append(
                f"{m} significantly WORSE (delta={c.delta:+.4f}, p={c.p_value:.4f})"
            )
    if improved >= CONSENSUS_MIN:
        res.reasons.append(f"metric consensus {improved}/{len(CONSENSUS_METRICS)} improved")
    else:
        res.blockers.append(
            f"metric consensus too weak: only {improved}/{len(CONSENSUS_METRICS)} improved "
            f"(need {CONSENSUS_MIN})"
        )

    # --- Rules 3 + 6: per-stratum regression, with power reporting -----------
    b_str, c_str = by_stratum(baseline, strata), by_stratum(candidate, strata)
    for stratum in sorted(set(b_str) | set(c_str)):
        if stratum == "no_answer":
            continue  # handled by rule 4
        bq, cq = b_str.get(stratum, {}), c_str.get(stratum, {})
        comp = st.compare(bq, cq, PRIMARY)
        if comp is None:
            res.warnings.append(f"stratum '{stratum}': not comparable on {PRIMARY}")
            continue
        a_vals, c_vals, _ = st.paired_values(bq, cq, PRIMARY)
        diffs = [y - x for x, y in zip(a_vals, c_vals)]
        mde = st.detectable_effect(len(diffs), _sd(diffs))
        res.stratum_deltas[stratum] = {
            **st.comparison_to_dict(comp),
            "min_detectable_effect": None if math.isinf(mde) else mde,
        }
        if comp.delta < -STRATUM_TOLERANCE:
            if comp.significant_at(alpha):
                res.blockers.append(
                    f"stratum '{stratum}' REGRESSED {PRIMARY} by {comp.delta:+.4f} "
                    f"(p={comp.p_value:.4f}, n={comp.n_queries})"
                )
            else:
                res.warnings.append(
                    f"stratum '{stratum}' dropped {comp.delta:+.4f} but not significant "
                    f"(p={comp.p_value:.4f}, n={comp.n_queries})"
                )
        if not math.isinf(mde) and mde > STRATUM_TOLERANCE:
            res.warnings.append(
                f"stratum '{stratum}' UNDERPOWERED: n={comp.n_queries}, smallest detectable "
                f"effect {mde:.4f} > tolerance {STRATUM_TOLERANCE}. 'No regression' here is weak evidence."
            )

    # --- Rule 4: no_answer behaviour ----------------------------------------
    bna, cna = b_str.get("no_answer", {}), c_str.get("no_answer", {})
    if bna and cna:
        fp = st.compare(bna, cna, "false_pos@10")
        if fp is not None:
            res.comparisons["false_pos@10"] = st.comparison_to_dict(fp)
            if fp.delta > 0 and fp.significant_at(alpha):
                res.blockers.append(
                    f"no_answer: false positives ROSE {fp.delta:+.2f} docs/query (p={fp.p_value:.4f})"
                )
            elif fp.delta < 0:
                res.reasons.append(f"no_answer: false positives fell {fp.delta:+.2f} docs/query")
        ab = st.compare(bna, cna, "abstained")
        if ab is not None and ab.delta < 0 and ab.significant_at(alpha):
            res.blockers.append(f"no_answer: abstention rate fell {ab.delta:+.4f}")
    else:
        res.warnings.append("no_answer stratum absent — abstention behaviour is unmeasured")

    # --- Rule 5: unjudged spike = measurement unreliable ---------------------
    b_unj, c_unj = aggregate(baseline, "unjudged@10"), aggregate(candidate, "unjudged@10")
    if b_unj is not None and c_unj is not None:
        res.comparisons["unjudged@10"] = {"mean_a": b_unj, "mean_b": c_unj, "delta": c_unj - b_unj}
        if c_unj - b_unj > UNJUDGED_SPIKE:
            res.blockers.append(
                f"MEASUREMENT UNRELIABLE: unjudged@10 rose {b_unj:.3f} -> {c_unj:.3f}. The candidate "
                f"is surfacing documents the pool never judged, so its score is an underestimate of "
                f"unknown size. Pool and re-judge before deciding."
            )

    res.promoted = not res.blockers
    return res
