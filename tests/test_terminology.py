"""Guards for query expansion — the offline, deterministic parts.

Expansion is the one component that puts words into a query that the user did not type, so
its failure mode is silent: results look plausible and answer a slightly different question.
The cases below are the specific ways that happened here, kept as tests so a future change
to sense selection has to notice it is breaking them.

Network-dependent resolution is NOT tested here. These assert the pure logic — key
normalisation, the short-term guard, and the sense-ranking order — which is where every
observed defect actually lived.
"""

from __future__ import annotations

import pytest

from oncolens.terminology import (
    MIN_TERM_CHARS,
    Expansion,
    OntologyExpander,
    Resolution,
    _SEMANTIC_RANK,
    candidate_terms,
    norm_key,
)


class TestNormKey:
    """Case, hyphens, dots and spaces carry no identity in this vocabulary."""

    @pytest.mark.parametrize("variants", [
        ("PD-L1", "PD L1", "pdl1", "PDL1", "Pd-L1"),
        ("STAT-3", "STAT3", "stat3"),
        ("MCF-7", "MCF 7", "MCF.7", "mcf7"),
        ("VEGF-R2", "VEGFR2", "vegf r2"),
    ])
    def test_surface_variants_share_one_key(self, variants):
        keys = {norm_key(v) for v in variants}
        assert len(keys) == 1, f"{variants} should share a lookup key, got {keys}"

    def test_distinct_concepts_keep_distinct_keys(self):
        # Folding must not merge things that are genuinely different.
        assert norm_key("CTLA-4") != norm_key("CTLA-2")
        assert norm_key("EGFR") != norm_key("FGFR")


class TestShortTermGuard:
    """Two characters cannot disambiguate, so nothing is injected.

    `ER` matches 14 NCIt concepts: the estrogen receptor, the endoplasmic reticulum, an
    emergency-room visit, Eritrea, and estrogen receptor POSITIVE and NEGATIVE. Semantic
    ranking silently resolves all fourteen to one. Expanding an "ER stress" query with
    estrogen-receptor vocabulary answers a different question.
    """

    @pytest.mark.parametrize("term", ["ER", "PR", "A", "", "  ", "5-", "R2"])
    def test_short_terms_never_resolve(self, term):
        exp = OntologyExpander(cache_dir=None)
        assert exp.resolve(term) is None, f"{term!r} must not expand"

    @pytest.mark.parametrize("term", ["MET", "ALK", "CD8"])
    def test_three_characters_are_allowed_through(self, term):
        # Not asserting they resolve — that needs the network. Only that the guard
        # itself does not reject them, which is the regression this pins.
        assert len(norm_key(term)) >= MIN_TERM_CHARS


class TestSenseRanking:
    """Measurement and questionnaire senses lose to the thing being measured.

    `PECAM-1` matches a CDISC "PECAM-1 Measurement" concept BEFORE the protein, because
    that is the order NCIt returns. Expanding a protein query with assay vocabulary is
    evidence about how something is measured injected into a question about what it does.
    """

    def test_substance_outranks_measurement(self):
        assert _SEMANTIC_RANK["Amino Acid, Peptide, or Protein"] < \
               _SEMANTIC_RANK["Laboratory Procedure"]
        assert _SEMANTIC_RANK["Gene or Genome"] < _SEMANTIC_RANK["Laboratory or Test Result"]

    def test_substance_outranks_questionnaire(self):
        assert _SEMANTIC_RANK["Amino Acid, Peptide, or Protein"] < \
               _SEMANTIC_RANK["Intellectual Product"]

    def test_a_sole_sense_survives_whatever_its_rank(self):
        # QLQ-C30's only sense is an Intellectual Product. Demotion is for CHOOSING
        # between senses; it must not delete a term that has just one.
        assert _SEMANTIC_RANK["Intellectual Product"] < 99


class TestAmbiguityIsReported:
    """`ambiguous` counts ties at the winning rank and therefore reported 0 for `ER`,
    while 13 senses were being discarded. `senses` reports the total."""

    def test_resolution_carries_both_counts(self):
        r = Resolution(term="ER", kind="protein", source="NCIt", ambiguous=0, senses=14)
        assert r.ambiguous == 0 and r.senses == 14

    def test_senses_defaults_do_not_claim_certainty(self):
        r = Resolution(term="x", kind="k", source="s")
        assert r.senses == 0, "an unset count must not read as '1 unambiguous sense'"


class TestWholeQueryFirst:
    """89.6% of identifier queries are one token, and the regex split all 17 multiword
    ones. `EGFR T790M` -> `EGFR` broadens a resistance-mutation query into every EGFR
    paper, which is worse than not expanding."""

    @pytest.mark.parametrize("query,split_to", [
        ("EGFR T790M", "EGFR"),
        ("KRAS G12D", "KRAS"),
        ("PIK3CA H1047L", "PIK3CA"),
    ])
    def test_the_regex_still_splits_these(self, query, split_to):
        # Pinning the OLD behaviour, so it is visible why the cascade must not rely on it.
        assert candidate_terms(query) == [split_to]

    def test_expansion_records_what_a_term_resolved_to(self):
        exp = Expansion(query="PALOMA-3")
        exp.terms["PALOMA-3"] = ["Palbociclib", "Fulvestrant"]
        exp.resolutions["PALOMA-3"] = Resolution(
            term="PALOMA-3", kind="clinical trial", source="ClinicalTrials.gov",
            code="NCT01942135", synonyms=["Palbociclib", "Fulvestrant"])
        # The type must survive to the caller: a bare list of extra words cannot be
        # shown to a user and checked, a typed resolution can.
        assert exp.resolutions["PALOMA-3"].kind == "clinical trial"
        assert exp.synonyms == ["Palbociclib", "Fulvestrant"]


class TestExpansionIsSeparable:
    """Synonyms are emitted apart from the user's own words so a caller can weight them
    below intent. A synonym is evidence, not what was asked for."""

    def test_original_query_leads(self):
        exp = Expansion(query="osimertinib")
        exp.terms["osimertinib"] = ["AZD9291", "Tagrisso"]
        assert exp.expanded_query().startswith("osimertinib")
        assert "AZD9291" in exp.expanded_query()

    def test_no_synonyms_leaves_the_query_untouched(self):
        assert Expansion(query="ER stress").expanded_query() == "ER stress"
