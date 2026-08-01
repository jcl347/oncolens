"""Hand-verified tests for the measurement engine.

The harness is the instrument. If nDCG or bpref is subtly wrong, every downstream
conclusion inherits the error and no amount of iteration recovers. Every expected value
below is computed by hand in the docstring, not captured from a previous run — a
regression test that snapshots buggy output is worse than no test.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oncolens.eval import metrics as M
from oncolens.eval import stats as S

# Shared fixture:
#   judgments: A=3 (definitive), B=0 (judged non-relevant), C=1 (marginal), D=2 (relevant)
#   ranking:   [A, B, C]        (D is relevant but never retrieved)
J = {"A": 3, "B": 0, "C": 1, "D": 2}
R = ["A", "B", "C"]

failures: list[str] = []


def check(name: str, got, want, tol=1e-6):
    ok = (got is None and want is None) or (
        got is not None and want is not None and abs(got - want) < tol
    )
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got={got!r} want={want!r}")
    if not ok:
        failures.append(name)


print("nDCG@3")
# DCG = (2^3-1)/log2(2) + (2^0-1)/log2(3) + (2^1-1)/log2(4) = 7 + 0 + 0.5 = 7.5
# Ideal grades desc = [3,2,1]
# IDCG = 7/log2(2) + 3/log2(3) + 1/log2(4) = 7 + 1.8927892 + 0.5 = 9.3927892
check("ndcg@3", M.ndcg_at_k(R, J, 3), 7.5 / (7 + 3 / math.log2(3) + 0.5))

print("recall / precision")
# relevant (grade>=1) = {A, C, D}; top3 contains A and C
check("recall@3", M.recall_at_k(R, J, 3), 2 / 3)
check("precision@3", M.precision_at_k(R, J, 3), 2 / 3)
check("recall@1", M.recall_at_k(R, J, 1), 1 / 3)

print("MRR / MAP")
check("mrr", M.mrr(R, J), 1.0)  # A is relevant at rank 1
# AP: hit at rank1 -> 1/1; hit at rank3 -> 2/3; sum=1.6667 over R=3 relevant
check("map", M.average_precision(R, J), (1.0 + 2 / 3) / 3)

print("bpref")
# R=3 relevant, N=1 judged non-relevant, denom=min(3,1)=1
#   A (rel), 0 non-rel seen above -> 1 - 0/1 = 1
#   B (non-rel) -> seen=1
#   C (rel), 1 non-rel seen above -> 1 - min(1,3)/1 = 0
#   D never retrieved -> contributes 0
# bpref = (1 + 0 + 0) / 3
check("bpref", M.bpref(R, J), 1 / 3)

print("unjudged@k diagnostic")
check("unjudged@3 (all judged)", M.unjudged_at_k(R, J, 3), 0.0)
check("unjudged@3 (one unknown)", M.unjudged_at_k(["A", "Z", "C"], J, 3), 1 / 3)

print("THE KEY PROPERTY: bpref ignores unjudged docs, nDCG punishes them")
# Insert an UNJUDGED document at rank 1. A system that found something the pool never
# saw must not be penalised by bpref; nDCG necessarily is (it consumed rank 1).
R_unjudged = ["Z", "A", "B", "C"]
b_before, b_after = M.bpref(R, J), M.bpref(R_unjudged, J)
n_before, n_after = M.ndcg_at_k(R, J, 3), M.ndcg_at_k(R_unjudged, J, 3)
check("bpref unchanged by unjudged doc", b_after, b_before)
print(f"  INFO  ndcg@3 {n_before:.4f} -> {n_after:.4f} (drops, as expected)")
if not (n_after < n_before):
    failures.append("ndcg should drop when an unjudged doc takes rank 1")

print("no-answer queries")
check("recall undefined when nothing is relevant", M.recall_at_k(["A"], {"B": 0}, 10), None)
check("abstained=1 for empty ranking", M.abstained([]), 1.0)
check("abstained=0 when docs returned", M.abstained(["A"]), 0.0)
check("false_pos@10", M.false_positives_at_k(["A", "B"], 10), 2.0)
na = M.evaluate_query(["A", "B"], {})
check("no-answer panel has no recall", na.get("recall@10"), None)
check("no-answer panel has false_pos@10", na.get("false_pos@10"), 2.0)

print("bpref returns None without judged negatives (no signal, not a perfect score)")
check("bpref, no negatives", M.bpref(["A"], {"A": 3}), None)

print("statistics")
a = {f"q{i}": {"m": 0.5} for i in range(40)}
b_same = {f"q{i}": {"m": 0.5} for i in range(40)}
c_same = S.compare(a, b_same, "m")
print(f"  INFO  identical arms: delta={c_same.delta:+.4f} p={c_same.p_value:.4f}")
if c_same.p_value < 0.9:
    failures.append("identical arms should not be significant")

b_better = {f"q{i}": {"m": 0.5 + (0.10 if i % 4 else 0.0)} for i in range(40)}
c_better = S.compare(a, b_better, "m")
print(
    f"  INFO  clearly-better arm: delta={c_better.delta:+.4f} p={c_better.p_value:.4f} "
    f"d={c_better.effect_size:+.2f} {c_better.wins}W/{c_better.losses}L/{c_better.ties}T"
)
if not (c_better.p_value < 0.01 and c_better.delta > 0):
    failures.append("clearly-better arm should be significant and positive")

# Noise: half the queries up by 0.02, half down by 0.02 -> no real effect.
b_noise = {f"q{i}": {"m": 0.5 + (0.02 if i % 2 else -0.02)} for i in range(40)}
c_noise = S.compare(a, b_noise, "m")
print(f"  INFO  pure noise: delta={c_noise.delta:+.4f} p={c_noise.p_value:.4f}")
if c_noise.p_value < 0.05:
    failures.append("symmetric noise should not be significant")

print("paired_values drops half-defined pairs instead of imputing zero")
pa = {"q1": {"m": 0.4}, "q2": {"m": 0.6}}
pb = {"q1": {"m": 0.5}, "q2": {}}
va, vb, qids = S.paired_values(pa, pb, "m")
check("only fully-defined pairs kept", float(len(va)), 1.0)
check("kept the right query", 1.0 if qids == ["q1"] else 0.0, 1.0)

print("multiple-comparisons ledger raises the bar as draws accumulate")
import tempfile

with tempfile.TemporaryDirectory() as td:
    led = S.Ledger.load(Path(td) / "ledger.json")
    check("alpha with 0 draws", led.adjusted_alpha(0.05), 0.05)
    for i in range(10):
        led.record(iteration=i, config_name=f"c{i}", split="dev", primary=0.5)
    check("alpha after 10 dev draws", led.adjusted_alpha(0.05), 0.005)
    led.record(iteration=99, config_name="t", split="test", primary=0.5)
    check("test draws do not tighten dev alpha", led.adjusted_alpha(0.05), 0.005)

print("power analysis reports the smallest detectable effect")
mde_small = S.detectable_effect(10, 0.15)
mde_large = S.detectable_effect(200, 0.15)
print(f"  INFO  n=10  -> MDE {mde_small:.4f}")
print(f"  INFO  n=200 -> MDE {mde_large:.4f}")
if not (mde_small > mde_large):
    failures.append("MDE must shrink as n grows")

print()
if failures:
    print(f"FAILED ({len(failures)}): " + ", ".join(failures))
    sys.exit(1)
print("all measurement-engine checks passed")
