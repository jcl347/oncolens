#!/usr/bin/env python
"""Generate the evaluation report the website displays.

Run offline, commit the output, deploy it. The site then shows real measured numbers —
including the unflattering ones — instead of asking users to trust an unmeasured system.

    python scripts/build_eval_report.py --out public/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oncolens.configs import BASELINE  # noqa: E402
from oncolens.data import load_dataset  # noqa: E402
from oncolens.eval.metrics import CONSENSUS_METRICS, PRIMARY  # noqa: E402
from oncolens.experiment import run_experiment  # noqa: E402
from oncolens.retrieval.expansion import Lexicon  # noqa: E402
from oncolens.retrieval.pipeline import RetrievalConfig  # noqa: E402
from oncolens.sanity import sanity_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="public/eval_report.json")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    args = ap.parse_args()

    ds = load_dataset(strict=False)
    lex = Lexicon.load(Path(__import__("os").environ.get("ONCOLENS_DATA", REPO / "data")) / "vocab" / "lexicon.json")

    champ_path = REPO / "experiments" / "champion.json"
    cfg = (RetrievalConfig(**json.loads(champ_path.read_text(encoding="utf-8"))["config"])
           if champ_path.exists() else BASELINE.variant("hybrid", use_dense=True))

    res = run_experiment(ds, cfg, split=args.split, lexicon=lex)
    sanity = sanity_report(ds, split=args.split)

    unjudged = res.aggregate.get("unjudged@10", 0.0)
    raw_tf = sanity["primary"].get("raw_tf", 0.0)
    primary = res.aggregate.get(PRIMARY, 0.0)

    caveats: list[str] = []
    if unjudged > 0.35:
        caveats.append(
            f"unjudged@10 is {unjudged:.2f}: most returned documents have never been judged, "
            f"so this score is an underestimate of unknown size and comparisons between "
            f"configurations are not yet interpretable."
        )
    if primary <= raw_tf + 0.02:
        caveats.append(
            f"The system scores {primary:.3f} against a raw term-frequency floor of "
            f"{raw_tf:.3f}. The retrieval machinery is not yet earning its complexity."
        )
    if ds.integrity.get("no_answer_queries", 0) == 0:
        caveats.append(
            "No unanswerable queries are in the evaluation set, so the system's tendency to "
            "return results when it should return nothing is unmeasured."
        )
    if any(v.get("_n", 0) < 15 for v in res.per_stratum.values()):
        caveats.append(
            "Some query strata have fewer than 15 queries; per-stratum numbers there are "
            "too noisy to support a claim of no regression."
        )
    caveats.append(
        "These numbers describe retrieval against this evaluation set only. They are not a "
        "claim about performance on arbitrary oncology literature."
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "config": cfg.name,
        "corpus": {
            "documents": ds.integrity.get("n_docs"),
            "queries": ds.integrity.get("n_queries"),
            "corpus_sha": ds.integrity.get("corpus_sha"),
            "mean_judged_per_query": round(ds.integrity.get("mean_judged_per_query", 0.0), 2),
        },
        "metrics": {m: round(res.aggregate[m], 4) for m in CONSENSUS_METRICS if m in res.aggregate},
        "primary_metric": PRIMARY,
        "floor": {
            "raw_term_frequency": raw_tf,
            "popularity": sanity["primary"].get("popularity"),
            "random": sanity["primary"].get("random"),
        },
        "ceiling": sanity["primary"].get("oracle"),
        "headroom": round((sanity["primary"].get("oracle", 1.0) - primary), 4),
        "per_stratum": {
            k: {"n": int(v.get("_n", 0)),
                PRIMARY: round(v.get(PRIMARY, 0.0), 4),
                "recall@10": round(v.get("recall@10", 0.0), 4)}
            for k, v in sorted(res.per_stratum.items())
        },
        "pool": {
            "unjudged_at_10": round(unjudged, 4),
            "interpretation": (
                "Fraction of the top 10 that carries no relevance judgment. High values mean "
                "the reported score understates true quality by an unknown amount."
            ),
        },
        "benchmark_validity": sanity.get("problems", []),
        "caveats": caveats,
    }

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"  {PRIMARY}={primary:.4f}  floor(raw-TF)={raw_tf:.4f}  ceiling={report['ceiling']}")
    for c in caveats:
        print(f"  caveat: {c[:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
