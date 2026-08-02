"""Regression tests for bibliography detection and citation-context labelling.

These pin the two failure modes that were actually observed on real PMC text, so a future
refactor cannot quietly reintroduce them:

1. PMC often emits the **entire bibliography as one paragraph**. The previous detector
   required a trailing *run* of >= 3 reference-shaped paragraphs and therefore never fired,
   missing 3 of 60 publisher-labelled articles.
2. The author-initials pattern required ``[ ,;]`` after the initial, so the ACS/Nature
   style ``Zhou J. Xu Y.`` matched nothing — worth 0.100 of measured AUC.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oncolens.eval.citation_labels import (  # noqa: E402
    CitationQrels, assert_source_excluded, build_labels, grade_for, is_useful_query,
)
from oncolens.retrieval.references import (  # noqa: E402
    find_reference_start, is_reference_like, reference_density, strip_references,
)

# A real-shaped bibliography in the period-after-initial style that used to score zero.
ACS_STYLE = (
    "Zhou J. Xu Y. Liu J. Feng L. Yu J. Chen D. Global burden of lung cancer in 2022 and "
    "projections to 2050 Cancer Epidemiol. 2024 93 102693 10.1016/j.canep.2024.102693 "
    "Siegel R. L. Giaquinto A. N. Jemal A. Cancer statistics 2024 CA Cancer J Clin. 2024 "
    "74 12-49 10.3322/caac.21820 Herbst R. S. Morgensztern D. Boshoff C. The biology and "
    "management of non-small cell lung cancer Nature 2018 553 446-454 10.1038/nature25183 "
    "Rotow J. Bivona T. G. Understanding and targeting resistance mechanisms in NSCLC "
    "Nat Rev Cancer 2017 17 637-658 10.1038/nrc.2017.84 "
)

VANCOUVER_STYLE = (
    "1. Lee EJ, Kim MJ, Cho JS. Acquired resistance to osimertinib in EGFR-mutant lung "
    "cancer. J Clin Oncol. 2022;40(7):709-20. doi:10.1200/JCO.21.01234 PMID: 35123456\n"
    "2. Park SH, Choi YL, Han J. MET amplification as a resistance mechanism. "
    "Lung Cancer. 2021;158:1123-1130. doi:10.1016/j.lungcan.2021.05.011\n"
)

PROSE = (
    "We observed that acquired resistance to osimertinib emerged in 14 of 32 patients "
    "after a median of 9.2 months. In these tumours we detected MET amplification by "
    "fluorescence in situ hybridisation, and the finding was concordant with the "
    "circulating tumour DNA results in 12 of the 14 cases. This is consistent with what "
    "has been described by Chen et al. (2019), who reported a similar frequency in a "
    "larger cohort, although their sequencing depth was lower than ours and they did not "
    "assess concordance with tissue biopsy at the time of progression."
)


def test_density_separates_bibliography_from_prose():
    assert reference_density(ACS_STYLE) > 0.6, "period-after-initial style must score high"
    assert reference_density(VANCOUVER_STYLE) > 0.6
    assert reference_density(PROSE) < 0.35, "prose citing a paper is not a bibliography"


def test_single_paragraph_bibliography_is_found():
    """The exact shape that defeated the previous run-based detector."""
    paragraphs = [PROSE] * 6 + [ACS_STYLE]
    idx = find_reference_start(paragraphs)
    assert idx == 6, f"single trailing block must be detected, got {idx}"


def test_multi_paragraph_bibliography_is_found():
    entries = [e + "." for e in VANCOUVER_STYLE.split("\n") if e.strip()] * 3
    paragraphs = [PROSE] * 6 + entries
    idx = find_reference_start(paragraphs)
    assert idx is not None and idx >= 6, f"must not cut into the body, got {idx}"


def test_body_is_never_truncated_when_there_is_no_bibliography():
    paragraphs = [PROSE] * 8
    assert find_reference_start(paragraphs) is None


def test_explicit_heading_wins():
    paragraphs = [PROSE] * 6 + ["References", VANCOUVER_STYLE]
    assert find_reference_start(paragraphs) == 6


def test_strip_preserves_kept_paragraph_offsets():
    text = "\n\n".join([PROSE] * 4 + [ACS_STYLE])
    res = strip_references(text)
    assert res.start_index == 4
    assert len(res.kept) == 4
    # Every kept paragraph must appear at its original offset, or citation spans break.
    kept_text = "\n\n".join(res.kept)
    assert text.startswith(kept_text)


def test_standalone_filter_does_not_delete_prose():
    """The stricter bar exists because standalone checks have no positional evidence."""
    assert not is_reference_like(PROSE)
    assert is_reference_like(ACS_STYLE)


def test_a_real_body_always_survives():
    """The share guard is tiered, so the invariant is absolute, not proportional.

    A *review* article legitimately is mostly bibliography — PMC10958066 carries a single
    480,996-character reference list that is 80.6% of the document — so capping removal at
    a fixed 60% left 215 bibliography passages of that one article in the index. High share
    is now permitted when the evidence is unambiguous; what is never permitted is reducing
    a document to nothing.
    """
    from oncolens.retrieval.references import MIN_BODY_CHARS

    paragraphs = [PROSE] + [ACS_STYLE] * 12
    idx = find_reference_start(paragraphs)
    if idx is not None:
        kept = sum(len(p) for p in paragraphs[:idx])
        assert kept >= MIN_BODY_CHARS, f"only {kept} chars of body survived"


def test_review_shaped_document_is_stripped_despite_huge_share():
    """The failure class the 60-article labelled set did not contain."""
    body = [PROSE] * 4                      # ~2.5k chars of real article
    refs = [ACS_STYLE * 12]                 # one enormous bibliography block
    idx = find_reference_start(body + refs)
    assert idx == 4, f"review-shaped bibliography must be found, got {idx}"


# --- citation labels ---------------------------------------------------------

def test_grade_falls_with_cocitation():
    assert grade_for(1) == 3 and grade_for(2) == 2 and grade_for(3) == 1


def test_contentless_sentences_are_rejected():
    assert not is_useful_query("As previously described.")
    assert not is_useful_query("This has been reported previously in several studies.")
    assert is_useful_query(
        "MET amplification was identified as the dominant mechanism of acquired "
        "resistance to osimertinib in this cohort.")


def test_self_citation_is_dropped():
    per_source = {"PAPER:PMID1": [(
        "MET amplification was identified as the dominant driver of acquired resistance "
        "to osimertinib in this cohort of patients.", ["PAPER:PMID1"], "Discussion")]}
    labels, stats = build_labels(per_source, {"PAPER:PMID1"})
    assert labels == [] and stats.get("self_citation") == 1


def test_source_exclusion_is_enforced_not_merely_documented():
    per_source = {"PAPER:PMID1": [(
        "MET amplification was identified as the dominant driver of acquired resistance "
        "to osimertinib in this cohort of patients.", ["PAPER:PMID2"], "Discussion")]}
    labels, _ = build_labels(per_source, {"PAPER:PMID1", "PAPER:PMID2"})
    assert len(labels) == 1
    q = CitationQrels.from_labels(labels)
    qid = labels[0].query_id
    assert q.exclude[qid] == "PAPER:PMID1"
    assert_source_excluded(qid, ["PAPER:PMID2"], q.exclude)          # fine
    try:
        assert_source_excluded(qid, ["PAPER:PMID1", "PAPER:PMID2"], q.exclude)
    except AssertionError:
        return
    raise AssertionError("citing document was allowed into its own results")


def test_citation_markers_do_not_leak_into_the_query():
    """The mined query must not carry the rendered citation numerals.

    The first version matched only the ``<xref>`` opening tag, so a query came out as
    "...in cervical cancer cells [ 29 , 34 ]" — the numerals survived as meaningless
    tokens in text that is then handed to a retriever as a search string.
    """
    from oncolens.sources.jats import extract_citation_contexts

    xml = (
        "<article><body><sec><p>MET amplification was identified as the dominant "
        "mechanism of acquired resistance to osimertinib in this cohort "
        '[<xref ref-type="bibr" rid="B29">29</xref>, '
        '<xref ref-type="bibr" rid="B34">34</xref>]. '
        "</p></sec></body></article>"
    )
    ctxs = extract_citation_contexts(xml)
    assert len(ctxs) == 1, f"expected one context, got {len(ctxs)}"
    sent = ctxs[0].sentence
    assert "29" not in sent and "34" not in sent, f"marker leaked: {sent!r}"
    assert "[" not in sent and "]" not in sent, f"bracket survived: {sent!r}"
    assert sent.endswith("cohort."), f"sentence mangled: {sent!r}"
    assert set(ctxs[0].rids) == {"B29", "B34"}


def test_agreement_pointers_are_rejected():
    """Sentences about the CITING paper's own results describe nothing cited."""
    assert not is_useful_query(
        "These results are consistent with those of previous research conducted in "
        "osteosarcoma and cervical cancer cell lines.")
    assert not is_useful_query(
        "Our findings are in agreement with earlier studies of the same pathway in "
        "pancreatic adenocarcinoma models.")
    # ...but a sentence that says WHAT the cited work found must survive.
    assert is_useful_query(
        "Overexpression of AIB1 enhances estrogen receptor transcriptional activity and "
        "can convert tamoxifen from an antagonist into an agonist.")


def test_target_cap_limits_popularity_bias():
    sent = ("MET amplification was identified as the dominant driver of acquired "
            "resistance to osimertinib in cohort {}.")
    per_source = {f"PAPER:PMID{i}": [(sent.format(i), ["PAPER:PMID999"], "Discussion")]
                  for i in range(1, 12)}
    labels, _ = build_labels(per_source, {f"PAPER:PMID{i}" for i in range(1, 12)}
                             | {"PAPER:PMID999"}, max_per_target=4)
    assert sum(1 for lb in labels if lb.target_doc_id == "PAPER:PMID999") == 4



def test_hard_cap_is_an_invariant_not_an_aspiration():
    """Sentence splitting silently fails on tables and citation blocks.

    MEASURED on the live index: 615 chunks exceeded HARD_CAP and the largest was 48,763
    characters — a tab-separated trial table and a reference block, neither of which
    contains a ``. Capital`` boundary to split on. A passage that size cannot be shown as
    'the place a concept was mentioned', and it also exceeds an embedding model's input
    limit.
    """
    from oncolens.retrieval.chunking import HARD_CAP, chunk_document

    table = "\t".join(f"Gene{i}\tProtein{i}\tfunction of {i}" for i in range(2000))
    refs = " ".join(f"Author{i} A Title of work {i} Journal 20{i%20:02d} 10.1000/x{i}"
                    for i in range(600))
    doc = {"doc_id": "T:1", "title": "t", "doc_type": "paper",
           "sections": [{"name": "Body", "text": table + "\n\n" + refs}]}
    chunks = chunk_document(doc)
    assert chunks, "produced no chunks at all"
    worst = max(len(c.text) for c in chunks)
    # Runt-folding may append up to MIN_CHARS to a chunk, so allow modest slack.
    assert worst <= HARD_CAP * 1.5, f"largest chunk {worst} chars exceeds cap {HARD_CAP}"


def test_offsets_stay_valid_after_forced_splitting():
    """Forced splits must still point at real spans, or citation display breaks."""
    from oncolens.retrieval.chunking import chunk_document

    text = "\t".join(f"col{i}\tvalue{i}" for i in range(1500))
    doc = {"doc_id": "T:2", "title": "t", "doc_type": "paper",
           "sections": [{"name": "Body", "text": text}]}
    for c in chunk_document(doc):
        assert c.start_char >= 0 and c.end_char > c.start_char
        assert c.end_char <= len(text) + 2, f"offset {c.end_char} past end {len(text)}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
# NOTE: the runner block above must stay LAST in this file. It collects tests from
# globals(), which is only populated up to the point of execution — functions defined
# after it are silently never run, and the suite reports a green count that omits them.
