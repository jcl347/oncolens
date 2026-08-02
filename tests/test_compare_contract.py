"""The /api/compare response shape is an INTERFACE. Pin it.

**The bug this exists to stop happening again.** The served response keys `cells` flat as
``"{doc_id}|{aspect_key}"`` and returns `aspects` as `{key,label,numeric}` objects. The
React client typed `aspects` as `string[]` and read `cells[doc][aspect]` — a nested lookup
that never resolved. Result: every cell in the comparison table rendered "not reported",
and React was handed an object as a child. Half the product was inert and nothing failed,
because no test compared the two sides.

A second variant of the same fault: the "no documents matched" branch returned `aspects`
as a list of plain strings, so the field's *type* depended on whether the query matched
anything — and the empty branch is the one nobody exercises by hand.

These tests do not need a database. They drive `compare()` against a stub index, which is
the point: a contract test that requires production infrastructure is a contract test that
does not get run.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oncolens.serve.live_query import compare  # noqa: E402


class _StubIndex:
    """Minimal stand-in for LiveIndex: no Postgres, no embeddings."""

    def __init__(self, results):
        self._results = results

    def search(self, query, top_k=5):
        return {"results": self._results[:top_k]}

    def conn(self):
        return _StubConn()


class _StubCursor:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a, **k):
        return None

    def fetchall(self):
        return []          # nothing reports any aspect


class _StubConn:
    def cursor(self):
        return _StubCursor()


REQUIRED_TOP_LEVEL = {"query", "aspects", "doc_ids", "titles", "years",
                      "coverage", "cells", "notes"}


def _check_shape(payload, *, expect_docs):
    missing = REQUIRED_TOP_LEVEL - set(payload)
    assert not missing, f"response is missing {sorted(missing)}"

    # aspects: ALWAYS a list of objects, in both the populated and empty branches.
    assert isinstance(payload["aspects"], list), "aspects must be a list"
    for a in payload["aspects"]:
        assert isinstance(a, dict), (
            f"aspects entries must be objects, got {type(a).__name__} — the client "
            f"renders a.label and keys cells by a.key")
        assert {"key", "label", "numeric"} <= set(a), f"aspect missing fields: {a}"

    assert isinstance(payload["doc_ids"], list)
    assert isinstance(payload["titles"], dict), "titles must be doc_id -> title"
    assert isinstance(payload["years"], dict)
    assert isinstance(payload["cells"], dict)

    # cells: ONE FLAT MAP keyed "doc|aspect". Not nested. The client depends on this.
    for k in payload["cells"]:
        assert "|" in k, (
            f"cell key {k!r} is not 'doc_id|aspect_key' — a nested "
            f"cells[doc][aspect] layout silently resolves to undefined in the client")
        doc, _, asp = k.partition("|")
        assert doc in payload["doc_ids"], f"cell {k!r} references an unlisted document"
        assert asp in {a["key"] for a in payload["aspects"]}, f"cell {k!r} unknown aspect"

    if expect_docs:
        # Every (document, aspect) pair must be present — a MISSING cell and a cell
        # marked reported=False mean different things, and the UI shows "not reported"
        # only for the latter.
        for doc in payload["doc_ids"]:
            for a in payload["aspects"]:
                key = f"{doc}|{a['key']}"
                assert key in payload["cells"], f"no cell emitted for {key}"
                assert "reported" in payload["cells"][key]
        for doc in payload["doc_ids"]:
            assert doc in payload["titles"], (
                f"{doc} has no title — the grid falls back to printing the raw doc_id, "
                f"which is not how a researcher recognises a paper")


def test_populated_response_matches_client_contract():
    index = _StubIndex([
        {"doc_id": "PAPER:PMID111", "title": "Osimertinib resistance via MET", "year": 2021},
        {"doc_id": "PAPER:PMID222", "title": "CAR-T persistence in solid tumours", "year": 2023},
    ])
    payload = compare(index, "resistance mechanisms", n_papers=2)
    _check_shape(payload, expect_docs=True)
    assert payload["doc_ids"] == ["PAPER:PMID111", "PAPER:PMID222"]
    assert payload["titles"]["PAPER:PMID111"].startswith("Osimertinib")
    # Nothing matched any aspect cue, so coverage is 0 and every cell is explicit.
    assert payload["coverage"] == 0.0
    assert all(c["reported"] is False for c in payload["cells"].values())


def test_empty_response_has_the_same_shape():
    payload = compare(_StubIndex([]), "no such topic", n_papers=5)
    _check_shape(payload, expect_docs=False)
    assert payload["doc_ids"] == []
    assert payload["cells"] == {}


if __name__ == "__main__":
    test_populated_response_matches_client_contract()
    test_empty_response_has_the_same_shape()
    print("compare contract OK")
