"""The improvement loop: propose -> measure -> gate -> promote.

One iteration = a champion configuration, a set of challengers, and a decision. A
challenger replaces the champion only if it clears every rule in ``eval.gate``. Everything
is appended to a ledger so the search history — including the failures — is auditable, and
so the significance bar tightens as the number of dev draws grows.

Deliberate properties:
  * The **test split is never read** during iteration. It is opened once, at the end, to
    estimate how much of the dev-set gain was real.
  * Every challenger is recorded even when rejected. A loop that only logs its wins cannot
    tell you it took thirty draws to find them.
  * Retrievers are cached by their index-affecting fields, so an eight-config sweep does
    not rebuild the same index eight times.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .data import Dataset, load_dataset
from .eval.bounds import dominance_report
from .eval.gate import GateResult, evaluate_gate
from .eval.stats import Ledger
from .experiment import ExperimentResult, format_summary, pool_gaps, run_experiment
from .retrieval.expansion import Lexicon
from .retrieval.pipeline import RetrievalConfig, Retriever

REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO / "experiments"
CHAMPION_PATH = EXPERIMENTS / "champion.json"
LEDGER_PATH = EXPERIMENTS / "ledger.json"

#: Config fields that change the *index* (as opposed to only query-time behaviour).
_INDEX_FIELDS = (
    "index_unit", "include_heading", "use_bm25", "bm25_k1", "bm25_b",
    "literal_boost", "use_dense", "dense_dim", "dense_backend",
)


def _index_key(cfg: RetrievalConfig) -> tuple:
    return tuple(getattr(cfg, f) for f in _INDEX_FIELDS)


class RetrieverCache:
    def __init__(self, dataset: Dataset, lexicon: Lexicon, contexts: dict[str, str] | None = None):
        self.dataset = dataset
        self.lexicon = lexicon
        self.contexts = contexts
        self._cache: dict[tuple, Retriever] = {}

    def get(self, cfg: RetrievalConfig) -> Retriever:
        key = _index_key(cfg)
        r = self._cache.get(key)
        if r is None:
            r = Retriever(cfg, lexicon=self.lexicon).build(self.dataset.docs, contexts=self.contexts)
            self._cache[key] = r
        # Reuse the built indexes but honour this config's query-time settings.
        r.config = cfg
        return r


@dataclass
class IterationOutcome:
    iteration: int
    champion_before: str
    champion_after: str
    results: dict[str, dict]
    gates: dict[str, dict]
    promoted: list[str]
    report: str


def load_champion() -> dict | None:
    if CHAMPION_PATH.exists():
        return json.loads(CHAMPION_PATH.read_text(encoding="utf-8"))
    return None


def save_champion(config: RetrievalConfig, result: ExperimentResult) -> None:
    CHAMPION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHAMPION_PATH.write_text(
        json.dumps({"config": config.as_dict(), "aggregate": result.aggregate,
                    "per_stratum": result.per_stratum}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_iteration(
    iteration: int,
    champion_cfg: RetrievalConfig,
    challengers: Sequence[RetrievalConfig],
    *,
    dataset: Dataset | None = None,
    lexicon: Lexicon | None = None,
    contexts: dict[str, str] | None = None,
    split: str = "dev",
) -> IterationOutcome:
    ds = dataset or load_dataset()
    lex = lexicon if lexicon is not None else Lexicon.load(REPO / "data" / "vocab" / "lexicon.json")
    ledger = Ledger.load(LEDGER_PATH)
    cache = RetrieverCache(ds, lex, contexts)
    strata = ds.strata()

    base_res = run_experiment(ds, champion_cfg, split=split, lexicon=lex,
                              contexts=contexts, retriever=cache.get(champion_cfg))
    base_res.save()
    ledger.record(iteration=iteration, config_name=champion_cfg.name, split=split,
                  primary=base_res.primary)

    lines = [
        f"# Iteration {iteration}",
        "",
        f"Champion: `{champion_cfg.name}`",
        "",
        "```",
        format_summary(base_res),
        "```",
        "",
    ]

    results: dict[str, ExperimentResult] = {champion_cfg.name: base_res}
    gates: dict[str, GateResult] = {}
    promoted: list[str] = []

    for cfg in challengers:
        res = run_experiment(ds, cfg, split=split, lexicon=lex,
                             contexts=contexts, retriever=cache.get(cfg))
        res.save()
        results[cfg.name] = res
        ledger.record(iteration=iteration, config_name=cfg.name, split=split, primary=res.primary)
        gate = evaluate_gate(base_res.per_query, res.per_query, strata, ledger,
                             family_size=len(challengers))
        gates[cfg.name] = gate
        if gate.promoted:
            promoted.append(cfg.name)

        # Bounds-based dominance: does the verdict survive every possible labelling of
        # the documents nobody has judged? A gate PASS with an UNDETERMINED dominance
        # verdict is a provisional result, and the report says so rather than implying
        # more certainty than the judgments support.
        qrels = {q.query_id: q.judgments for q in ds.split(split)}
        dom = dominance_report(base_res.runs, res.runs, qrels, k=10)

        lines += [
            f"## `{cfg.name}`",
            "",
            "```",
            format_summary(res),
            "```",
            "",
            "```",
            gate.summary(),
            "```",
            "",
        ]
        if dom is not None:
            lines += ["```", dom.summary(), "```", ""]
            if gate.promoted and dom.robust_verdict.startswith("UNDETERMINED"):
                lines += [
                    "> Note: this challenger cleared the gate on point estimates, but the "
                    "bound intervals overlap. The promotion is provisional until the pool "
                    "gap below is judged.",
                    "",
                ]

    # Promote the passing challenger with the best primary metric.
    new_champion = champion_cfg
    if promoted:
        best = max(promoted, key=lambda n: results[n].primary)
        new_champion = next(c for c in challengers if c.name == best)
        save_champion(new_champion, results[best])
        lines += [f"**PROMOTED: `{best}`**", ""]
    else:
        lines += ["**No challenger cleared the gate — champion unchanged.**", ""]

    # Close the feedback loop: diagnose the *new* champion's remaining failures so the
    # next iteration is chosen from evidence rather than from the ladder's default order.
    from .analysis import analyze_run, compare_queries

    champ_res = results[new_champion.name]
    analysis = analyze_run(ds, champ_res.per_query, champ_res.runs, cache.get(new_champion), split=split)
    lines += ["## Failure analysis (current champion)", "", "```"]
    lines += [f"failure modes: {analysis['failure_mode_counts']}"]
    for stratum, modes in sorted(analysis["failure_modes_by_stratum"].items()):
        lines.append(f"  {stratum}: {modes}")
    lines += ["```", "", "**Next hypotheses, ranked by observed failure mode:**", ""]
    lines += [f"- {h}" for h in analysis["next_hypothesis"]] or ["- (no dominant failure mode)"]
    lines += [""]

    if new_champion.name != champion_cfg.name:
        cmp = compare_queries(ds, base_res.per_query, champ_res.per_query, split=split)
        lines += [
            "### What the promotion traded away", "",
            f"improved {cmp['n_improved']} / degraded {cmp['n_degraded']} / unchanged {cmp['n_unchanged']}",
            "",
        ]
        if cmp["biggest_losses"] and cmp["biggest_losses"][0]["delta"] < -1e-9:
            lines += ["Largest regressions (these are real, even though the gate passed):", ""]
            for d in cmp["biggest_losses"][:5]:
                if d["delta"] < -1e-9:
                    lines.append(
                        f"- `{d['query_id']}` [{d['stratum']}] {d['a']:.3f} -> {d['b']:.3f} "
                        f"({d['delta']:+.3f}) — {d['query'][:70]}"
                    )
            lines += [""]

    gaps = pool_gaps(list(results.values()), ds, depth=10)
    if gaps:
        n = sum(len(v) for v in gaps.values())
        lines += [
            f"> Pool gap: {n} document/query pairs appeared in a top-10 but carry no judgment "
            f"(across {len(gaps)} queries). Scores involving these are underestimates; judge them "
            f"to keep the comparison honest.",
            "",
        ]

    report = "\n".join(lines)
    out = EXPERIMENTS / f"iteration_{iteration:02d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    return IterationOutcome(
        iteration=iteration,
        champion_before=champion_cfg.name,
        champion_after=new_champion.name,
        results={k: v.aggregate for k, v in results.items()},
        gates={k: {"promoted": g.promoted, "blockers": g.blockers, "warnings": g.warnings,
                   "reasons": g.reasons} for k, g in gates.items()},
        promoted=promoted,
        report=report,
    )


def final_test_evaluation(
    champion_cfg: RetrievalConfig,
    baseline_cfg: RetrievalConfig,
    *,
    dataset: Dataset | None = None,
    lexicon: Lexicon | None = None,
    contexts: dict[str, str] | None = None,
) -> dict:
    """Open the locked test split exactly once.

    The gap between the dev-set gain and the test-set gain is the estimate of how much of
    the improvement was fitted to the dev queries. A large gap is the loop's own honesty
    check, and it is reported whether it is flattering or not.
    """
    ds = dataset or load_dataset()
    lex = lexicon if lexicon is not None else Lexicon.load(REPO / "data" / "vocab" / "lexicon.json")
    cache = RetrieverCache(ds, lex, contexts)
    ledger = Ledger.load(LEDGER_PATH)

    base = run_experiment(ds, baseline_cfg, split="test", lexicon=lex,
                          contexts=contexts, retriever=cache.get(baseline_cfg))
    champ = run_experiment(ds, champion_cfg, split="test", lexicon=lex,
                           contexts=contexts, retriever=cache.get(champion_cfg))
    base.save(EXPERIMENTS / "test" / f"{baseline_cfg.name}.json")
    champ.save(EXPERIMENTS / "test" / f"{champion_cfg.name}.json")
    ledger.record(iteration=999, config_name=champion_cfg.name, split="test", primary=champ.primary)

    gate = evaluate_gate(base.per_query, champ.per_query, ds.strata(), ledger)
    return {
        "baseline": base.aggregate,
        "champion": champ.aggregate,
        "baseline_per_stratum": base.per_stratum,
        "champion_per_stratum": champ.per_stratum,
        "gate": {"promoted": gate.promoted, "reasons": gate.reasons,
                 "blockers": gate.blockers, "warnings": gate.warnings},
        "comparisons": gate.comparisons,
    }
