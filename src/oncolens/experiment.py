"""Run a retrieval configuration against a split and record everything needed to judge it.

An experiment record is self-describing on purpose: it carries the config, the data hashes,
the per-query metrics, and the raw runs. The raw runs matter because they are what makes
**pooling** possible after the fact — without them you cannot go back and judge the
documents a new configuration surfaced, and the incomplete-judgment bias becomes permanent.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .data import Dataset, Query
from .eval.gate import aggregate, by_stratum, mean_defined
from .eval.metrics import CONSENSUS_METRICS, PRIMARY, evaluate_query
from .retrieval.expansion import Lexicon
from .retrieval.pipeline import RetrievalConfig, Retriever

def results_dir() -> Path:
    """Resolved at call time so ONCOLENS_EXPERIMENTS works regardless of import order."""
    return Path(os.environ.get("ONCOLENS_EXPERIMENTS",
                               Path(__file__).resolve().parents[2] / "experiments"))

#: How deep each run is recorded for later pooling. Deeper pools cost judging effort but
#: are the only defence against measuring a new system against a stale pool.
POOL_DEPTH = 20


@dataclass
class ExperimentResult:
    config: dict
    split: str
    per_query: dict[str, dict[str, float]] = field(default_factory=dict)
    runs: dict[str, list[str]] = field(default_factory=dict)          # qid -> top POOL_DEPTH doc_ids
    evidence: dict[str, dict] = field(default_factory=dict)            # qid -> top passage
    aggregate: dict[str, float] = field(default_factory=dict)
    per_stratum: dict[str, dict[str, float]] = field(default_factory=dict)
    integrity: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.config.get("name", "unnamed")

    @property
    def primary(self) -> float:
        return self.aggregate.get(PRIMARY, 0.0)

    def save(self, path: Path | None = None) -> Path:
        p = path or (results_dir() / self.split / f"{self.name}.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path) -> "ExperimentResult":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def run_experiment(
    dataset: Dataset,
    config: RetrievalConfig,
    *,
    split: str = "dev",
    lexicon: Lexicon | None = None,
    contexts: dict[str, str] | None = None,
    retriever: Retriever | None = None,
) -> ExperimentResult:
    """Build (or reuse) an index and evaluate every query in the split."""
    queries: Sequence[Query] = dataset.split(split)
    r = retriever or Retriever(config, lexicon=lexicon).build(dataset.docs, contexts=contexts)

    per_query: dict[str, dict[str, float]] = {}
    runs: dict[str, list[str]] = {}
    evidence: dict[str, dict] = {}

    for q in queries:
        res = r.search(q.query)
        per_query[q.query_id] = evaluate_query(res.ranking, q.judgments)
        runs[q.query_id] = res.ranking[:POOL_DEPTH]
        top = res.ranking[0] if res.ranking else None
        if top and top in res.evidence:
            ev = res.evidence[top]
            evidence[q.query_id] = {
                "doc_id": ev.doc_id, "chunk_id": ev.chunk_id, "section": ev.section,
                "start_char": ev.start_char, "end_char": ev.end_char,
                "text": ev.text[:400], "score": ev.score,
            }

    metric_names = sorted({m for v in per_query.values() for m in v})
    agg = {m: v for m in metric_names if (v := aggregate(per_query, m)) is not None}

    strata_map = dataset.strata()
    per_stratum: dict[str, dict[str, float]] = {}
    for stratum, qs in by_stratum(per_query, strata_map).items():
        names = sorted({m for v in qs.values() for m in v})
        per_stratum[stratum] = {
            "_n": float(len(qs)),
            **{m: v for m in names if (v := mean_defined([x.get(m) for x in qs.values()])) is not None},
        }

    return ExperimentResult(
        config=config.as_dict(),
        split=split,
        per_query=per_query,
        runs=runs,
        evidence=evidence,
        aggregate=agg,
        per_stratum=per_stratum,
        integrity=dataset.integrity,
        env={"python": sys.version.split()[0], "platform": platform.system()},
    )


def build_retriever(
    dataset: Dataset, config: RetrievalConfig, *, lexicon: Lexicon | None = None,
    contexts: dict[str, str] | None = None,
) -> Retriever:
    return Retriever(config, lexicon=lexicon).build(dataset.docs, contexts=contexts)


def format_summary(result: ExperimentResult) -> str:
    lines = [f"{result.name}  [{result.split}]"]
    head = [m for m in CONSENSUS_METRICS if m in result.aggregate]
    lines.append("  " + "  ".join(f"{m}={result.aggregate[m]:.4f}" for m in head))
    if "unjudged@10" in result.aggregate:
        lines.append(f"  unjudged@10={result.aggregate['unjudged@10']:.4f}")
    for stratum in sorted(result.per_stratum):
        s = result.per_stratum[stratum]
        n = int(s.get("_n", 0))
        if stratum == "no_answer":
            ab = s.get("abstained", 0.0)
            fp = s.get("false_pos@10", 0.0)
            lines.append(f"    {stratum:<14} n={n:<4} abstained={ab:.3f}  false_pos@10={fp:.2f}")
        else:
            lines.append(
                f"    {stratum:<14} n={n:<4} {PRIMARY}={s.get(PRIMARY, float('nan')):.4f}  "
                f"recall@10={s.get('recall@10', float('nan')):.4f}"
            )
    return "\n".join(lines)


def pool_gaps(results: Sequence[ExperimentResult], dataset: Dataset, depth: int = 10) -> dict[str, list[str]]:
    """Documents surfaced in the top-``depth`` by some run but never judged.

    This is the work queue for keeping the pool honest. Any comparison involving a config
    with a large gap list is provisional: the unjudged documents might be relevant, in
    which case that config is being under-credited.
    """
    judged = {q.query_id: set(q.judgments) for q in dataset.queries}
    gaps: dict[str, set[str]] = {}
    for r in results:
        for qid, run in r.runs.items():
            missing = [d for d in run[:depth] if d not in judged.get(qid, set())]
            if missing:
                gaps.setdefault(qid, set()).update(missing)
    return {q: sorted(v) for q, v in sorted(gaps.items())}
