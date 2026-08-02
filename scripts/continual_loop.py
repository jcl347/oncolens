#!/usr/bin/env python
"""Continual improvement loop: hypothesise, measure across every stratum, promote or record.

One round =

    for each untried candidate:
        state its hypothesis BEFORE measuring
        run it against ALL FOUR strata
        gate holistically:
            - the weighted composite must improve
            - NO stratum may regress significantly, whatever the composite says
        classify the hypothesis: CONFIRMED / REFUTED / NO_EFFECT / SURPRISE
        promote and commit, or record the failure with its reason

**Why holistic gating rather than a single score.** The weighted composite says which
improvements are worth *pursuing*; it must never decide which are safe to *ship*. A change
that lifts the composite by helping the two large strata while quietly destroying exact
identifier lookup is a regression for an oncology R&D team, whatever the arithmetic says —
returning papers about the wrong variant is worse than returning nothing, because the
error is invisible in the results. So any significant per-stratum regression vetoes,
independently of the weights.

**Why the harness is built once.** Indexing 58k passages and encoding queries dominates
runtime; sharing it across candidates and strata is what makes a *continual* loop possible
rather than a thing run once and abandoned.

    python scripts/continual_loop.py --rounds 1
    python scripts/continual_loop.py --rounds 3 --commit --push
    python scripts/continual_loop.py --report          # read the ledger
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from improve_loop import CANDIDATES_BY_NAME, Harness, load_chunks, mean_of  # noqa: E402
from oncolens.env import load_env, local_data_dir  # noqa: E402
from oncolens.eval import metrics as M  # noqa: E402
from oncolens.eval.citation_labels import assert_source_excluded  # noqa: E402
from oncolens.eval.hypothesis import HYPOTHESES, classify  # noqa: E402
from oncolens.eval.stats import compare  # noqa: E402
from oncolens.eval.weighting import (  # noqa: E402
    PRIMARY_METRIC, STRATUM_WEIGHTS, composite, describe,
)

ALPHA = 0.05
LEDGER = "continual_ledger.json"


def load_all_strata() -> tuple[dict, dict, dict, dict]:
    path = local_data_dir() / "strata.json"
    if not path.exists():
        raise SystemExit(f"no strata at {path}; run scripts/build_strata.py")
    d = json.loads(path.read_text(encoding="utf-8"))
    queries, qrels = d["queries"], d["qrels"]
    strata, exclude = d.get("strata", {}), d.get("exclude", {})
    queries = {k: v for k, v in queries.items() if qrels.get(k)}
    qrels = {k: v for k, v in qrels.items() if k in queries}
    return queries, qrels, strata, exclude


def run_all_strata(harness: Harness, cfg: dict, qrels: dict, exclude: dict,
                   strata: dict) -> dict[str, dict]:
    """Evaluate one config, then split the per-query results by stratum."""
    per_query = harness.run(cfg, qrels, exclude)
    out: dict[str, dict] = {}
    for qid, res in per_query.items():
        out.setdefault(strata.get(qid, "unknown"), {})[qid] = res
    return out


def score_table(by_stratum: dict[str, dict]) -> dict[str, dict[str, float]]:
    return {
        s: {m: mean_of(pq, m)
            for m in ("mrr", "success@1", "success@5", "success@10", "success@20",
                      "recall@10", "recall@20", "ndcg@10", "unjudged@10")}
        for s, pq in by_stratum.items()
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--skip", action="append", default=["rerank_llm"],
                    help="candidates to skip (rerank_llm costs money per query)")
    args = ap.parse_args()
    load_env()

    ledger_path = local_data_dir() / LEDGER
    ledger = []
    if ledger_path.exists():
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            ledger = []

    if args.report:
        print(describe(), "\n")
        if not ledger:
            print("ledger empty — no rounds run yet")
            return 0
        print(f"{'round':>6}{'candidate':<18}{'verdict':<12}detail")
        print("-" * 100)
        for r in ledger:
            for o in r.get("outcomes", []):
                print(f"{r['round']:>6}  {o['candidate']:<18}{o['verdict']:<12}"
                      f"{o['detail'][:60]}")
        tried = {o["candidate"] for r in ledger for o in r.get("outcomes", [])}
        promoted = {o["candidate"] for r in ledger for o in r.get("outcomes", [])
                    if o.get("promoted")}
        print(f"\n{len(tried)} candidates tried, {len(promoted)} promoted: "
              f"{sorted(promoted) or '(none)'}")
        return 0

    queries, qrels, strata, exclude = load_all_strata()
    from collections import Counter
    print("stratum sizes:", dict(Counter(strata.get(q, "?") for q in queries)))

    harness = Harness(load_chunks(), queries)
    print("\nbaseline across all strata...")
    base_by = run_all_strata(harness, {}, qrels, exclude, strata)
    base_scores = score_table(base_by)
    base_comp = composite(base_scores)
    print(f"{'stratum':<13}{'primary':<12}{'value':>9}{'weight':>8}")
    for s, w in sorted(STRATUM_WEIGHTS.items(), key=lambda kv: -kv[1]):
        m = PRIMARY_METRIC[s]
        v = base_scores.get(s, {}).get(m)
        print(f"{s:<13}{m:<12}{(v if v is not None else float('nan')):>9.4f}{w:>8.2f}")
    print(f"COMPOSITE = {base_comp:.4f}" if base_comp is not None
          else "COMPOSITE = n/a (a stratum is missing)")

    already = {o["candidate"] for r in ledger for o in r.get("outcomes", [])}
    for rnd in range(args.rounds):
        todo = [n for n in CANDIDATES_BY_NAME
                if n != "baseline" and n not in already and n not in args.skip]
        if not todo:
            print("\nno untried candidates remain; add new hypotheses to continue")
            break
        print(f"\n{'='*78}\nROUND {len(ledger)+1} — {len(todo)} candidates: {todo}")
        outcomes = []
        for name in todo:
            cand = CANDIDATES_BY_NAME[name]
            hyp = HYPOTHESES.get(name)
            print(f"\n--- {name}: {cand.description}")
            if hyp:
                print(f"    HYPOTHESIS: {hyp.describe()}")
                print(f"    mechanism : {hyp.mechanism[:100]}")
            try:
                cand_by = run_all_strata(harness, cand.config, qrels, exclude, strata)
            except Exception as e:  # noqa: BLE001
                print(f"    run error: {type(e).__name__}: {str(e)[:120]}")
                outcomes.append({"candidate": name, "verdict": "ERROR",
                                 "detail": str(e)[:200], "promoted": False})
                already.add(name)
                continue

            observed: dict[str, dict] = {}
            regressions, no_effect_strata = [], []
            for s in sorted(base_by):
                if s not in cand_by:
                    continue
                ident = all(base_by[s][q].get("_ranking_hash")
                            == cand_by[s].get(q, {}).get("_ranking_hash")
                            for q in base_by[s])
                if ident:
                    no_effect_strata.append(s)
                observed[s] = {}
                for m in ("mrr", "success@1", "success@5", "success@10", "success@20",
                          "recall@10", "recall@20"):
                    c = compare(base_by[s], cand_by[s], m)
                    if c is None:
                        continue
                    observed[s][m] = {"delta": c.delta, "p": c.p_value}
                    if c.p_value < ALPHA and c.delta < -0.005:
                        regressions.append(f"{s}/{m} {c.delta:+.4f}")

            cand_scores = score_table(cand_by)
            cand_comp = composite(cand_scores)
            no_effect = len(no_effect_strata) == len(base_by)

            for s in sorted(observed):
                m = PRIMARY_METRIC.get(s)
                d = observed[s].get(m)
                if d:
                    mark = "REG" if f"{s}/{m}" in " ".join(regressions) else "   "
                    print(f"    {mark} {s:<11}{m:<11}{base_scores[s][m]:>7.4f} -> "
                          f"{cand_scores[s][m]:>7.4f} ({d['delta']:+.4f}, p={d['p']:.4f})")
            if cand_comp is not None and base_comp is not None:
                print(f"    composite {base_comp:.4f} -> {cand_comp:.4f} "
                      f"({cand_comp - base_comp:+.4f})")

            out = classify(hyp, observed, no_effect=no_effect) if hyp else None
            promoted = (not no_effect and not regressions and cand_comp is not None
                        and base_comp is not None and cand_comp > base_comp + 0.002
                        and (out is None or out.verdict == "CONFIRMED"))
            reason = ("promoted" if promoted else
                      "no effect — candidate did not fire" if no_effect else
                      f"regression: {'; '.join(regressions[:2])}" if regressions else
                      "composite did not improve materially" if cand_comp is not None
                      and base_comp is not None and cand_comp <= base_comp + 0.002 else
                      "hypothesis not confirmed")
            print(f"    -> {'PROMOTE' if promoted else 'DISCARD'}: {reason}")
            if out:
                print(f"    hypothesis {out.verdict}: {out.detail[:110]}")

            rec = (out.as_dict() if out else
                   {"candidate": name, "verdict": "NO_HYPOTHESIS", "detail": reason})
            rec.update({"promoted": promoted, "reason": reason,
                        "composite_delta": (cand_comp - base_comp)
                        if (cand_comp is not None and base_comp is not None) else None})
            outcomes.append(rec)
            already.add(name)

            if promoted:
                base_by, base_scores, base_comp = cand_by, cand_scores, cand_comp

        ledger.append({"round": len(ledger) + 1, "outcomes": outcomes,
                       "composite": base_comp})
        ledger_path.write_text(json.dumps(ledger, indent=2, default=str), encoding="utf-8")
        print(f"\nledger -> {ledger_path}")

        won = [o for o in outcomes if o.get("promoted")]
        if won and args.commit:
            cfg_path = ROOT / "config" / "served.json"
            cfg_path.parent.mkdir(exist_ok=True)
            merged: dict = {}
            for o in won:
                merged.update(CANDIDATES_BY_NAME[o["candidate"]].config)
            cfg_path.write_text(json.dumps(
                {"promoted": [o["candidate"] for o in won], "config": merged,
                 "composite": base_comp}, indent=2), encoding="utf-8")
            msg = ("Continual loop round %d: promote %s\n\n%s\n\nComposite %.4f. "
                   "Gated per stratum; no regression permitted regardless of weights."
                   % (len(ledger), ", ".join(o["candidate"] for o in won),
                      "\n".join(f"{o['candidate']}: {o['reason']}" for o in won),
                      base_comp or 0.0))
            subprocess.run(["git", "add", "-A"], cwd=ROOT, check=False)
            subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT, check=False)
            if args.push:
                subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=False)
            print(f"committed {len(won)} promotion(s)")
        elif not won:
            print("\nnothing promoted this round. The baseline stands — which is the "
                  "loop working, not the loop failing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
