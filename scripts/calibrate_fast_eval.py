#!/usr/bin/env python
"""How small can the evaluation set be before it stops being able to see anything?

**Why a fast eval is the missing piece.** Each candidate currently re-runs 3,280 queries
against a 58,306-passage index — minutes per experiment. That is fine for a confirmation
run and fatal for iteration: at five minutes a try, nobody explores, and a loop that is
too slow to run is a loop that does not run.

**Why sampling has to be calibrated rather than guessed.** A small sample does not just
add noise, it silently raises the smallest effect the experiment can detect. Below some
size the loop returns "no significant change" for *everything*, which looks like rigour
and is actually blindness — the most dangerous failure mode available to a measurement
harness, because it is indistinguishable from a well-behaved negative result.

So this computes, per stratum, the **minimum detectable effect** at each candidate sample
size from the observed variance:

    MDE = (z_alpha/2 + z_power) * sd / sqrt(n)

and reports the smallest n whose MDE is below the effect worth acting on. Effects smaller
than that are invisible at that size *by construction*, not by luck.

    python scripts/calibrate_fast_eval.py --target-effect 0.03
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from oncolens.env import load_env, local_data_dir  # noqa: E402
from oncolens.eval.stats import detectable_effect  # noqa: E402
from oncolens.eval.weighting import PRIMARY_METRIC, STRATUM_WEIGHTS  # noqa: E402

SIZES = (50, 100, 200, 300, 500, 800, 1200, 2000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-effect", type=float, default=0.03,
                    help="smallest change worth acting on; anything below this may as "
                         "well be invisible")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    load_env()

    # Variance is estimated from the metric's own distribution. For the binary success@k
    # family that is Bernoulli, whose sd is fully determined by the mean — so no prior run
    # is needed and the calibration cannot be contaminated by a particular candidate.
    strata_path = local_data_dir() / "strata.json"
    if not strata_path.exists():
        raise SystemExit("run scripts/build_strata.py first")
    d = json.loads(strata_path.read_text(encoding="utf-8"))
    sizes_by_stratum: dict[str, int] = {}
    for qid in d["queries"]:
        s = d["strata"].get(qid, "?")
        sizes_by_stratum[s] = sizes_by_stratum.get(s, 0) + 1

    # Observed baselines, from the runs already recorded.
    observed = {
        "synthesis": {"recall@20": 0.3078, "ndcg@10": 0.1844},
        "concept": {"success@5": 0.7003, "success@1": 0.3401},
        "identifier": {"success@1": None},
        "claim": {"mrr": 0.4096},
    }

    print(f"target effect: {args.target_effect:.3f}  "
          f"(alpha 0.05, power 0.80, paired)\n")
    print(f"{'stratum':<13}{'metric':<12}{'available':>10}" +
          "".join(f"{n:>7}" for n in SIZES))
    print("-" * (35 + 7 * len(SIZES)))

    recommend: dict[str, int] = {}
    for stratum in sorted(STRATUM_WEIGHTS, key=lambda s: -STRATUM_WEIGHTS[s]):
        metric = PRIMARY_METRIC[stratum]
        mean = observed.get(stratum, {}).get(metric)
        if mean is None:
            print(f"{stratum:<13}{metric:<12}{sizes_by_stratum.get(stratum,0):>10}"
                  "   (no baseline recorded yet)")
            continue
        # Paired differences have lower variance than the raw metric; assuming they are
        # as variable as the metric itself is deliberately pessimistic, so the recommended
        # sample size errs large rather than small.
        sd = (mean * (1 - mean)) ** 0.5 if metric.startswith("success") else 0.35
        row = f"{stratum:<13}{metric:<12}{sizes_by_stratum.get(stratum,0):>10}"
        best = None
        for n in SIZES:
            n_eff = min(n, sizes_by_stratum.get(stratum, n))
            mde = detectable_effect(n_eff, sd)
            row += f"{mde:>7.3f}"
            if best is None and mde <= args.target_effect:
                best = n_eff
        recommend[stratum] = best or sizes_by_stratum.get(stratum, 0)
        print(row)

    print("\nsmallest sample that can still see a "
          f"{args.target_effect:.3f} effect:")
    total = 0
    for s, n in recommend.items():
        avail = sizes_by_stratum.get(s, 0)
        note = "" if n < avail else "  <- needs the WHOLE stratum; cannot be sampled"
        print(f"  {s:<13}{n:>6} of {avail}{note}")
        total += min(n, avail)
    print(f"\nfast-eval set: {total} queries "
          f"({total / max(sum(sizes_by_stratum.values()), 1):.0%} of the full set)")
    print("\nUse it to EXPLORE. Confirm anything that looks promising on the full set")
    print("before promoting — a fast eval is for deciding what to measure properly,")
    print("never for deciding what ships.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"target_effect": args.target_effect, "recommended": recommend,
             "available": sizes_by_stratum}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
