"""The /api/search response shape is an INTERFACE, on BOTH paths. Pin it.

**The bug this exists to stop happening again, which already happened twice.**

§4.11 recorded that `/api/compare` returned a shape the React client did not read, so the
whole comparison table rendered "not reported". The lesson drawn was "a response shape is
an interface", and a contract test was written. For compare. Only.

`/api/search` has two implementations: the bundled artifact in `api/search.py` (preview
deployments, no database) and `live_query.LiveIndex.search` (every deployment with
`POSTGRES_URL` set, i.e. production). They drifted. The live one:

  * never set ``passage.best_clause``, which is the field the UI highlights from
  * hand-rolled clause dicts without ``matched_terms`` or ``spans``
  * dropped ``doc_type`` and ``meta``, hiding the PMID inside a ``source`` object the
    client does not read, so the PubMed link never rendered
  * called ``find_clauses`` with no ``base_offset``, making clause offsets
    passage-relative while ``passage.start_char`` is section-absolute

Net effect: on the only path production uses, no result was ever highlighted and the
provenance line printed a character range that does not exist in the source article. The
product's one rule is to return *the passage where a concept was mentioned* — marked — and
it was inert.

These tests drive the real `_shape` against a stub row. No database, because a contract
test that needs production infrastructure is a contract test nobody runs.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oncolens.serve.live_query import _shape  # noqa: E402

PASSAGE = (
    "Acquired resistance to osimertinib is frequently driven by MET amplification. "
    "In our cohort of 42 patients, MET copy number gain was detected in 9 cases. "
    "No EGFR C797S mutations were observed."
)

ROW = {
    "doc_id": "PAPER:PMID39198425",
    "score": 0.8421,
    "title": "Mechanisms of acquired resistance to osimertinib",
    "doc_type": "paper",
    "year": 2024,
    "meta": {"journal": "Journal of Thoracic Oncology", "pmcid": "PMC11375894",
             "license_code": "CC BY", "blob_url": "https://example/blob.txt"},
    "passage": {
        "chunk_id": "PAPER:PMID39198425:body:7",
        "section": "Body",
        # Deliberately non-zero: this is what catches a missing base_offset.
        "start_char": 4200,
        "end_char": 4200 + len(PASSAGE),
        "text": PASSAGE,
    },
}

#: Read directly by app/search/page.tsx. Changing any of these breaks the UI silently.
REQUIRED_TOP = {"doc_id", "title", "doc_type", "year", "meta", "score", "passage"}
REQUIRED_PASSAGE = {"chunk_id", "section", "start_char", "end_char", "text",
                    "clauses", "best_clause"}


def test_live_shape_matches_what_the_client_reads():
    out = _shape(ROW, "osimertinib MET amplification")

    missing = REQUIRED_TOP - set(out)
    assert not missing, f"response is missing {sorted(missing)}"
    missing_p = REQUIRED_PASSAGE - set(out["passage"])
    assert not missing_p, f"passage is missing {sorted(missing_p)}"

    # ResultCard renders {result.doc_type} as a chip and reads meta.journal / meta.pmid.
    assert out["doc_type"], "doc_type empty: the UI renders it as a blank grey chip"
    assert isinstance(out["meta"], dict)
    assert out["meta"].get("journal") == "Journal of Thoracic Oncology"
    assert out["meta"].get("pmid") == "39198425", (
        "meta.pmid absent: the PubMed link in ResultCard never renders. It must be "
        "derivable from doc_id even when the stored meta has no pmid.")


def test_best_clause_is_present_and_highlightable():
    out = _shape(ROW, "osimertinib MET amplification")
    best = out["passage"]["best_clause"]
    assert best is not None, (
        "best_clause is None: ResultCard falls back to a raw 320-char truncation and "
        "nothing is highlighted, which is the product's one rule not happening")

    # Highlighted() maps over clause.spans and reads span.is_literal; a hand-rolled dict
    # without them throws at render time.
    for field in ("text", "start", "end", "score", "matched_terms", "spans"):
        assert field in best, f"best_clause missing {field!r}, which the UI reads"
    assert isinstance(best["spans"], list)
    for s in best["spans"]:
        assert {"start", "end", "text", "is_literal"} <= set(s), f"bad span {s}"


def test_clause_offsets_are_section_absolute():
    """The clause must sit INSIDE the passage on the same coordinate scale.

    Without ``base_offset`` the clause offsets start at 0 while ``passage.start_char`` is
    4200, so the UI prints "chars 0-77" for text that lives at 4200 in the article, and
    the PaperViewer's rebasing lands outside the string.
    """
    out = _shape(ROW, "osimertinib MET amplification")
    p = out["passage"]
    for c in p["clauses"]:
        assert p["start_char"] <= c["start"] < c["end"] <= p["end_char"], (
            f"clause [{c['start']}, {c['end']}) is outside the passage "
            f"[{p['start_char']}, {p['end_char']}): base_offset was not applied")
        # The slice at the rebased offsets must be the clause text.
        rel_s = c["start"] - p["start_char"]
        rel_e = c["end"] - p["start_char"]
        assert p["text"][rel_s:rel_e] == c["text"], "offsets do not locate the clause text"


def test_degrades_without_meta_rather_than_dropping_the_link():
    bare = {**ROW, "meta": {}, "doc_type": ""}
    out = _shape(bare, "osimertinib")
    assert out["meta"]["pmid"] == "39198425", "pmid must survive an empty meta"
    assert out["doc_type"], "doc_type must default rather than render as an empty chip"


def test_unmatched_query_still_returns_a_well_formed_result():
    """A query matching nothing must not produce a malformed passage."""
    out = _shape(ROW, "zzzznotawordzzzz")
    assert REQUIRED_PASSAGE <= set(out["passage"])
    assert out["passage"]["text"] == PASSAGE
    # best_clause may legitimately be None here; it must be the key, not absent.
    assert "best_clause" in out["passage"]


if __name__ == "__main__":
    test_live_shape_matches_what_the_client_reads()
    test_best_clause_is_present_and_highlightable()
    test_clause_offsets_are_section_absolute()
    test_degrades_without_meta_rather_than_dropping_the_link()
    test_unmatched_query_still_returns_a_well_formed_result()
    print("search contract OK")
