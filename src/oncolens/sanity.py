"""Degenerate baselines that validate the *benchmark*, not the retriever.

Before believing any measured improvement, establish that the benchmark can tell a real
system apart from a fake one. Three reference points bracket every result:

* **random** — floor. Any real system must clearly beat it.
* **popularity** — the dangerous one. It returns the *same documents for every query*,
  ignoring the query completely. If it scores anywhere near a real retriever, the
  benchmark is exploitable: some documents are relevant to so many queries that ranking is
  unnecessary, and the metric is measuring corpus skew rather than retrieval quality.
* **oracle** — ceiling. Perfect ranking of the judged documents. It answers "how much
  headroom is left", which decides whether an iteration is worth running at all.

A real system scoring 0.55 means something entirely different when popularity scores 0.05
than when popularity scores 0.45. Reporting the real number alone hides that.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from .data import Dataset
from .eval.gate import aggregate
from .eval.metrics import PRIMARY, evaluate_query

_SEED = 1234


def random_run(dataset: Dataset, split: str, *, k: int = 50) -> dict[str, list[str]]:
    rng = random.Random(_SEED)
    ids = sorted(d["doc_id"] for d in dataset.docs)
    out: dict[str, list[str]] = {}
    for q in dataset.split(split):
        shuffled = ids[:]
        rng.shuffle(shuffled)
        out[q.query_id] = shuffled[:k]
    return out


def popularity_run(dataset: Dataset, split: str, *, k: int = 50) -> dict[str, list[str]]:
    """Query-independent ranking: the same documents, every time.

    Ordered by how often each document is judged relevant across the *whole* query set —
    i.e. the strongest possible query-independent prior. If this scores well, the
    benchmark rewards knowing the corpus rather than answering the question.
    """
    freq: dict[str, int] = {}
    for q in dataset.queries:
        for doc, grade in q.judgments.items():
            if grade >= 1:
                freq[doc] = freq.get(doc, 0) + 1
    ranked = [d for d, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))]
    ranked += [d["doc_id"] for d in dataset.docs if d["doc_id"] not in freq]
    fixed = ranked[:k]
    return {q.query_id: list(fixed) for q in dataset.split(split)}


def length_run(dataset: Dataset, split: str, *, k: int = 50) -> dict[str, list[str]]:
    """Another query-independent exploit: rank by document length.

    Longer documents match more query terms by chance, so a benchmark where length
    correlates with relevance will reward verbosity.
    """
    lens = {
        d["doc_id"]: sum(len(s.get("text", "")) for s in d.get("sections", []))
        for d in dataset.docs
    }
    ranked = [d for d, _ in sorted(lens.items(), key=lambda x: (-x[1], x[0]))][:k]
    return {q.query_id: list(ranked) for q in dataset.split(split)}


def oracle_run(dataset: Dataset, split: str, *, k: int = 50) -> dict[str, list[str]]:
    """Ceiling: judged documents sorted by grade. nDCG is 1.0 by construction."""
    out: dict[str, list[str]] = {}
    for q in dataset.split(split):
        ranked = sorted(
            (d for d, g in q.judgments.items() if g >= 1),
            key=lambda d: (-q.judgments[d], d),
        )
        out[q.query_id] = ranked[:k]
    return out


def _score(runs: Mapping[str, Sequence[str]], dataset: Dataset, split: str) -> dict[str, float]:
    qs = {q.query_id: q for q in dataset.split(split)}
    per_query = {
        qid: evaluate_query(run, qs[qid].judgments) for qid, run in runs.items() if qid in qs
    }
    names = sorted({m for v in per_query.values() for m in v})
    return {m: v for m in names if (v := aggregate(per_query, m)) is not None}


def sanity_report(dataset: Dataset, *, split: str = "dev") -> dict:
    """Bracket the benchmark. Returns scores plus an explicit verdict on exploitability."""
    baselines = {
        "random": random_run(dataset, split),
        "popularity": popularity_run(dataset, split),
        "length": length_run(dataset, split),
        "oracle": oracle_run(dataset, split),
    }
    scores = {name: _score(run, dataset, split) for name, run in baselines.items()}

    pop = scores["popularity"].get(PRIMARY, 0.0)
    rnd = scores["random"].get(PRIMARY, 0.0)
    orc = scores["oracle"].get(PRIMARY, 0.0)
    length = scores["length"].get(PRIMARY, 0.0)

    problems: list[str] = []
    # A query-independent baseline scoring above ~0.15 means the benchmark leaks.
    if pop > 0.15:
        problems.append(
            f"EXPLOITABLE: query-independent popularity scores {PRIMARY}={pop:.4f}. Relevance "
            f"is concentrated in a few documents, so a system can score without reading the "
            f"query. Spread judgments across more documents or add discriminating queries."
        )
    if length > 0.15:
        problems.append(
            f"EXPLOITABLE: ranking by document length alone scores {PRIMARY}={length:.4f}. "
            f"Length correlates with relevance; the benchmark rewards verbosity."
        )
    if orc < 0.99:
        problems.append(
            f"Oracle only reaches {PRIMARY}={orc:.4f}, not ~1.0 — the metric or the qrels are "
            f"inconsistent (expected: a perfect ranking of judged docs scores 1.0)."
        )
    if pop - rnd > 0.25:
        problems.append(
            f"popularity beats random by {pop - rnd:.4f} — strong corpus skew; interpret all "
            f"absolute numbers with that prior in mind."
        )

    return {
        "split": split,
        "scores": {k: {m: round(v, 4) for m, v in s.items() if "@" not in m or m.endswith("@10")}
                   for k, s in scores.items()},
        "primary": {k: round(s.get(PRIMARY, 0.0), 4) for k, s in scores.items()},
        "headroom_above_popularity": round(orc - pop, 4),
        "problems": problems,
        "verdict": "BENCHMARK OK" if not problems else f"{len(problems)} VALIDITY PROBLEM(S)",
    }
