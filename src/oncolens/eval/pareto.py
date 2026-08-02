"""Promotion by dominance over a vector, not by a scalar someone chose the weights for.

**Why the weighted composite was the wrong primary object.** It compresses four strata
into one number using weights I invented (0.35/0.30/0.20/0.15). Those weights are a
defensible product judgement, but they are not measurements, and the composite has a
specific failure: it destroys exactly the information that made the first round useful.

Round 1's most valuable finding was that two candidates moved strata in *opposite*
directions — dropping BM25 hurt 2-word concept queries while doubling BM25 hurt
conceptual synthesis queries. A composite would have averaged that into "roughly no
change" and thrown away the insight that the fusion weight should be query-dependent.

So the result of an experiment is the **vector** of per-stratum scores. Promotion is by
*dominance*: a candidate ships if it improves at least one stratum and worsens none.
That is stricter than a composite and, more importantly, it cannot launder a regression
in a small stratum behind a gain in a large one.

The composite survives in exactly one role: **ordering which candidate to try next**. A
heuristic for allocating attention is a legitimate use of invented weights. Deciding what
ships is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StratumResult:
    stratum: str
    metric: str
    base: float
    cand: float
    delta: float
    p_value: float

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


@dataclass
class DominanceVerdict:
    improved: list[str] = field(default_factory=list)
    regressed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    promoted: bool = False
    reason: str = ""

    def summary(self) -> str:
        bits = []
        if self.improved:
            bits.append(f"+{','.join(self.improved)}")
        if self.regressed:
            bits.append(f"-{','.join(self.regressed)}")
        return " ".join(bits) or "no change anywhere"


def judge_dominance(results: list[StratumResult], *, min_effect: float = 0.005,
                    alpha: float = 0.05) -> DominanceVerdict:
    """Promote only a Pareto improvement: better somewhere, worse nowhere.

    ``min_effect`` keeps a statistically significant but operationally meaningless change
    from counting. With thousands of queries, p-values get small long before a user could
    notice the difference, and shipping complexity for an invisible gain is a cost with no
    benefit.
    """
    v = DominanceVerdict()
    for r in results:
        if r.significant and r.delta >= min_effect:
            v.improved.append(f"{r.stratum}/{r.metric}")
        elif r.significant and r.delta <= -min_effect:
            v.regressed.append(f"{r.stratum}/{r.metric}")
        else:
            v.unchanged.append(r.stratum)

    if v.regressed:
        v.promoted = False
        v.reason = (f"dominated: regresses {', '.join(v.regressed)} — a gain elsewhere "
                    f"does not buy back a stratum that got worse")
    elif not v.improved:
        v.promoted = False
        v.reason = "no stratum improved by a margin worth shipping complexity for"
    else:
        v.promoted = True
        v.reason = f"Pareto improvement: better on {', '.join(v.improved)}, worse nowhere"
    return v


def attention_order(candidates: dict[str, float]) -> list[str]:
    """Rank candidates by expected value, to decide what to try next.

    This is where invented weights belong: allocating effort is a judgement call, and
    being wrong costs some wasted compute. Deciding what ships is not a judgement call,
    and being wrong there costs a regression nobody notices.
    """
    return [k for k, _ in sorted(candidates.items(), key=lambda kv: -kv[1])]
