"""Statistical machinery for deciding whether a retrieval change is a real improvement.

The failure this module exists to prevent: running twenty configurations against one
hundred queries, picking the highest mean, and calling it progress. With that many draws
the winner is usually noise, and the loop then optimises toward the noise.

Three defences, all applied:

1. **Paired tests, not unpaired.** Queries vary enormously in difficulty; the variance
   between queries dwarfs the variance between systems. Everything here compares arms
   query-by-query on the same queries.

2. **Randomization (permutation) test as the primary p-value.** It is the standard
   significance test in IR evaluation, makes no distributional assumption, and is exact in
   the limit. A bootstrap CI on the mean difference is reported alongside it.

3. **An explicit multiple-comparisons ledger.** Every dev-set evaluation is recorded. The
   promotion gate uses a Bonferroni-adjusted alpha based on the number of draws actually
   taken, so the significance bar rises as the loop searches harder. Without this, an
   iterative loop is a p-hacking machine by construction.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Fixed seed: significance decisions must be reproducible across reruns of the same data.
_SEED = 0xC0FFEE


@dataclass(frozen=True)
class PairedComparison:
    metric: str
    n_queries: int
    mean_a: float
    mean_b: float
    delta: float
    p_value: float
    ci_low: float
    ci_high: float
    effect_size: float  # Cohen's d_z
    wins: int
    losses: int
    ties: int

    @property
    def direction(self) -> int:
        return (self.delta > 0) - (self.delta < 0)

    def significant_at(self, alpha: float) -> bool:
        return self.p_value < alpha


def paired_values(
    per_query_a: Mapping[str, Mapping[str, float]],
    per_query_b: Mapping[str, Mapping[str, float]],
    metric: str,
) -> tuple[list[float], list[float], list[str]]:
    """Extract aligned per-query values, keeping only queries where BOTH arms define the metric.

    Dropping half-defined pairs is deliberate. Imputing a missing metric as 0.0 would
    manufacture huge fake deltas on exactly the queries where the metric is undefined.
    """
    a_vals, b_vals, qids = [], [], []
    for qid in sorted(set(per_query_a) & set(per_query_b)):
        va, vb = per_query_a[qid].get(metric), per_query_b[qid].get(metric)
        if va is None or vb is None:
            continue
        a_vals.append(float(va))
        b_vals.append(float(vb))
        qids.append(qid)
    return a_vals, b_vals, qids


def permutation_test(a: Sequence[float], b: Sequence[float], iterations: int = 20000) -> float:
    """Two-sided paired randomization test.

    Under the null hypothesis that the two systems are equivalent, the labels on each
    query's pair are exchangeable. Flip each pair with probability 0.5 and rebuild the
    null distribution of the mean difference.
    """
    n = len(a)
    if n == 0:
        return 1.0
    diffs = [bi - ai for ai, bi in zip(a, b)]
    observed = abs(sum(diffs) / n)
    if observed == 0.0:
        return 1.0
    rng = random.Random(_SEED)
    extreme = 0
    for _ in range(iterations):
        total = 0.0
        for d in diffs:
            total += d if rng.random() < 0.5 else -d
        if abs(total / n) >= observed - 1e-12:
            extreme += 1
    # +1 smoothing: never report p == 0 from a finite sample.
    return (extreme + 1) / (iterations + 1)


def bootstrap_ci(
    a: Sequence[float], b: Sequence[float], iterations: int = 10000, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean paired difference (resampling queries)."""
    n = len(a)
    if n == 0:
        return (0.0, 0.0)
    diffs = [bi - ai for ai, bi in zip(a, b)]
    rng = random.Random(_SEED + 1)
    means = []
    for _ in range(iterations):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[max(0, int((alpha / 2) * iterations) - 1)]
    hi = means[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return (lo, hi)


def compare(
    per_query_a: Mapping[str, Mapping[str, float]],
    per_query_b: Mapping[str, Mapping[str, float]],
    metric: str,
    tie_eps: float = 1e-9,
) -> PairedComparison | None:
    """Compare arm B against baseline arm A on one metric. Positive delta favours B."""
    a, b, _ = paired_values(per_query_a, per_query_b, metric)
    n = len(a)
    if n < 2:
        return None
    diffs = [bi - ai for ai, bi in zip(a, b)]
    mean_d = sum(diffs) / n
    var = sum((d - mean_d) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    lo, hi = bootstrap_ci(a, b)
    return PairedComparison(
        metric=metric,
        n_queries=n,
        mean_a=sum(a) / n,
        mean_b=sum(b) / n,
        delta=mean_d,
        p_value=permutation_test(a, b),
        ci_low=lo,
        ci_high=hi,
        effect_size=(mean_d / sd) if sd > 0 else 0.0,
        wins=sum(1 for d in diffs if d > tie_eps),
        losses=sum(1 for d in diffs if d < -tie_eps),
        ties=sum(1 for d in diffs if abs(d) <= tie_eps),
    )


# ---------------------------------------------------------------------------
# Multiple-comparisons ledger
# ---------------------------------------------------------------------------


@dataclass
class Ledger:
    """Records every dev-set evaluation so the significance bar can be adjusted.

    An iterative improvement loop takes many draws against the same dev set. Treating
    each draw as an independent test at alpha=0.05 guarantees false positives. The ledger
    makes the number of draws visible and feeds Bonferroni correction into the gate.
    """

    path: Path
    entries: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls(path=p, entries=data.get("entries", []))
        return cls(path=p)

    def record(self, *, iteration: int, config_name: str, split: str, primary: float) -> None:
        self.entries.append(
            {"iteration": iteration, "config": config_name, "split": split, "primary": primary}
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"entries": self.entries}, indent=2), encoding="utf-8"
        )

    @property
    def dev_draws(self) -> int:
        return sum(1 for e in self.entries if e["split"] == "dev")

    def adjusted_alpha(self, base_alpha: float = 0.05) -> float:
        """Bonferroni over the number of dev draws taken so far (floor of 1 draw)."""
        return base_alpha / max(1, self.dev_draws)


def detectable_effect(n_queries: int, sd: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum mean difference detectable at the given n, sd, alpha and power.

    Used to answer the honest question "is this stratum even large enough to gate on?"
    Normal approximation: delta_min = (z_{1-a/2} + z_{power}) * sd / sqrt(n).
    """
    if n_queries <= 1 or sd <= 0:
        return float("inf")
    z_alpha = 1.959963985 if abs(alpha - 0.05) < 1e-9 else _z(1 - alpha / 2)
    z_power = 0.8416212336 if abs(power - 0.8) < 1e-9 else _z(power)
    return (z_alpha + z_power) * sd / math.sqrt(n_queries)


def _z(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def comparison_to_dict(c: PairedComparison) -> dict:
    return asdict(c)
