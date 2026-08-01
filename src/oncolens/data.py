"""Corpus and qrels loading, with integrity checks that fail loudly.

A judgment pointing at a doc_id that does not exist is the classic silent corruption in a
homemade benchmark: recall is computed against a denominator containing phantom documents,
so every system is scored too low, uniformly — which hides real differences. These loaders
refuse to return a corrupt dataset quietly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name}:{lineno}: malformed JSON — {e}") from e


@dataclass
class Query:
    query_id: str
    query: str
    stratum: str
    source: str
    judgments: dict[str, int]
    notes: str = ""

    @property
    def is_no_answer(self) -> bool:
        return not any(g >= 1 for g in self.judgments.values())


@dataclass
class Dataset:
    docs: list[dict] = field(default_factory=list)
    queries: list[Query] = field(default_factory=list)
    integrity: dict = field(default_factory=dict)

    @property
    def doc_ids(self) -> set[str]:
        return {d["doc_id"] for d in self.docs}

    def split(self, which: str) -> list[Query]:
        """Deterministic dev/test split by hash of query_id.

        Hashing (rather than a stored flag) means the split cannot drift as files are
        edited, and cannot be quietly reshuffled to flatter a result.
        """
        if which == "all":
            return list(self.queries)
        out = []
        for q in self.queries:
            h = int(hashlib.sha256(q.query_id.encode()).hexdigest()[:8], 16)
            bucket = "dev" if (h % 100) < 60 else "test"
            if bucket == which:
                out.append(q)
        return out

    def strata(self) -> dict[str, str]:
        return {q.query_id: q.stratum for q in self.queries}


def load_corpus(corpus_dir: Path | None = None) -> list[dict]:
    d = corpus_dir or (DATA / "corpus")
    docs: list[dict] = []
    seen: set[str] = set()
    for path in sorted(d.glob("*.jsonl")):
        for lineno, obj in _read_jsonl(path):
            did = obj.get("doc_id")
            if not did:
                raise ValueError(f"{path.name}:{lineno}: missing doc_id")
            if did in seen:
                raise ValueError(f"{path.name}:{lineno}: duplicate doc_id {did}")
            seen.add(did)
            obj.setdefault("sections", [])
            obj.setdefault("descriptors", [])
            docs.append(obj)
    return docs


def load_queries(qrels_dir: Path | None = None) -> list[Query]:
    d = qrels_dir or (DATA / "qrels")
    out: list[Query] = []
    seen: set[str] = set()
    for path in sorted(d.glob("*.jsonl")):
        for lineno, obj in _read_jsonl(path):
            qid = obj.get("query_id")
            if not qid:
                raise ValueError(f"{path.name}:{lineno}: missing query_id")
            if qid in seen:
                raise ValueError(f"{path.name}:{lineno}: duplicate query_id {qid}")
            seen.add(qid)
            out.append(
                Query(
                    query_id=qid,
                    query=obj["query"],
                    stratum=obj.get("stratum", "unknown"),
                    source=obj.get("source", "unknown"),
                    judgments={k: int(v) for k, v in (obj.get("judgments") or {}).items()},
                    notes=obj.get("notes", ""),
                )
            )
    return out


def load_dataset(*, strict: bool = True) -> Dataset:
    docs = load_corpus()
    queries = load_queries()
    doc_ids = {d["doc_id"] for d in docs}

    dangling: dict[str, list[str]] = {}
    for q in queries:
        bad = [d for d in q.judgments if d not in doc_ids]
        if bad:
            dangling[q.query_id] = bad

    if dangling and strict:
        n = sum(len(v) for v in dangling.values())
        sample = list(dangling.items())[:5]
        raise ValueError(
            f"{n} judgments across {len(dangling)} queries reference nonexistent doc_ids. "
            f"This silently deflates recall for every system. Examples: {sample}"
        )

    # Drop dangling refs in non-strict mode so a partially-authored dataset is still usable.
    if dangling:
        for q in queries:
            for bad in dangling.get(q.query_id, []):
                q.judgments.pop(bad, None)

    integrity = {
        "n_docs": len(docs),
        "n_queries": len(queries),
        "n_dangling_judgments": sum(len(v) for v in dangling.values()),
        "queries_with_dangling": len(dangling),
        "n_grants": sum(1 for d in docs if d.get("doc_type") == "grant"),
        "n_papers": sum(1 for d in docs if d.get("doc_type") == "paper"),
        "single_relevant_queries": sum(
            1 for q in queries if sum(1 for g in q.judgments.values() if g >= 1) == 1
        ),
        "queries_with_explicit_zeros": sum(
            1 for q in queries if any(g == 0 for g in q.judgments.values())
        ),
        "no_answer_queries": sum(1 for q in queries if q.is_no_answer),
        "mean_judged_per_query": (
            sum(len(q.judgments) for q in queries) / len(queries) if queries else 0.0
        ),
        "corpus_sha": _sha_of(DATA / "corpus"),
        "qrels_sha": _sha_of(DATA / "qrels"),
    }
    return Dataset(docs=docs, queries=queries, integrity=integrity)


def _sha_of(directory: Path) -> str:
    """Content hash of a data directory, so results can be tied to exact inputs.

    If this changes between two experiments, they are not comparable — the gate would be
    measuring a data change, not a code change.
    """
    h = hashlib.sha256()
    if not directory.exists():
        return "missing"
    for path in sorted(directory.glob("*.jsonl")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]
