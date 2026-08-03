"""Guards for how the evaluation set is built.

A defect here is worse than a defect in retrieval, because it is invisible: the numbers
still come out, they are just measuring something other than what the notes claim. The
cases below are the ones that actually happened.
"""

from __future__ import annotations

import pytest

from oncolens.eval.strata import (
    MAX_MINE_SENSES,
    StratifiedQuery,
    _clean_span,
    _mine_spans,
    gazetteer_identifier_queries,
    merge_duplicate_queries,
)


def q(qid, text, judgments, stratum="identifier", exclude=None):
    return StratifiedQuery(query_id=qid, query=text, stratum=stratum,
                           judgments=dict(judgments), exclude_doc=exclude)


class TestDuplicateQueriesAreUnwinnable:
    """The measured defect: `CTLA-4` was 66 queries, each with ONE different relevant
    paper. Retrieval is a function of the query string, so one ranking serves all 66 and
    at most one can put its judged document at rank 1. Identifier's success@1 ceiling was
    0.4985 while every other stratum's was 1.0."""

    def test_ceiling_below_one_before_merging(self):
        # Three queries, same string, disjoint singleton judgments.
        qs = [q("a", "CTLA-4", {"d1": 3}), q("b", "CTLA-4", {"d2": 3}),
              q("c", "CTLA-4", {"d3": 3})]
        # Whatever single document a system ranks first, at most one query is satisfied.
        best = max(sum(1 for x in qs if doc in x.judgments) for doc in ("d1", "d2", "d3"))
        assert best == 1
        assert best / len(qs) < 1.0, "this is the unwinnable condition"

    def test_merging_restores_a_reachable_ceiling(self):
        merged = merge_duplicate_queries(
            [q("a", "CTLA-4", {"d1": 3}), q("b", "CTLA-4", {"d2": 3}),
             q("c", "CTLA-4", {"d3": 3})])
        assert len(merged) == 1
        # Now any of the three at rank 1 is a success — ceiling 1.0.
        assert set(merged[0].judgments) == {"d1", "d2", "d3"}


class TestMergeSemantics:
    def test_strongest_grade_wins(self):
        merged = merge_duplicate_queries(
            [q("a", "CTLA-4", {"d1": 1}), q("b", "CTLA-4", {"d1": 3})])
        assert merged[0].judgments["d1"] == 3, "a minor grade must not overwrite a major"

    def test_matching_is_case_and_space_insensitive(self):
        merged = merge_duplicate_queries(
            [q("a", "CTLA-4", {"d1": 3}), q("b", " ctla-4 ", {"d2": 3})])
        assert len(merged) == 1

    def test_different_strata_never_merge(self):
        merged = merge_duplicate_queries(
            [q("a", "EGFR", {"d1": 3}, stratum="identifier"),
             q("b", "EGFR", {"d2": 3}, stratum="concept")])
        assert len(merged) == 2

    def test_exclude_doc_is_not_silently_dropped(self):
        # 4.4's leakage guard: the citing paper must never be returnable. Merging two
        # queries with different exclusions must not lose one.
        merged = merge_duplicate_queries(
            [q("a", "CTLA-4", {"d1": 3}, exclude="src1"),
             q("b", "CTLA-4", {"d2": 3}, exclude="src2")])
        assert merged[0].exclude_doc == "src1"
        assert "src2" in merged[0].note

    def test_distinct_queries_pass_through_untouched(self):
        qs = [q("a", "EGFR", {"d1": 3}), q("b", "KRAS", {"d2": 3})]
        assert len(merge_duplicate_queries(qs)) == 2


class TestSpanCleaning:
    """`norm_key` folds punctuation away for LOOKUP, so a raw span still matched and the
    raw span was what got stored. The miner emitted `( fig`, `+ cd45ra +` and `&gt; a` as
    queries, attached to real judgments."""

    @pytest.mark.parametrize("raw,want", [
        ("( fig", "fig"),
        ("+ cd45ra +", "cd45ra"),
        ("&gt; a", "a"),
        ("(CTLA-4)", "CTLA-4"),
        ("EGFR T790M.", "EGFR T790M"),
        ("  MCF-7,  ", "MCF-7"),
    ])
    def test_sentence_punctuation_is_stripped(self, raw, want):
        assert _clean_span(raw) == want

    def test_internal_structure_survives(self):
        assert _clean_span("EGFR NP_005219.2") == "EGFR NP_005219.2"


class TestMinerPrecision:
    GAZ = {
        "brca1": {"code": "C17815", "name": "BRCA1", "sem": ["Gene or Genome"], "senses": 1},
        "osimertinib": {"code": "C116377", "name": "Osimertinib",
                        "sem": ["Pharmacologic Substance"], "senses": 1},
        "ato": {"code": "C1264", "name": "Arsenic Trioxide",
                "sem": ["Pharmacologic Substance"], "senses": 1},
        "response": {"code": "C0000", "name": "Response", "sem": ["Cell"], "senses": 1},
        "er": {"code": "C17069", "name": "Estrogen Receptor", "sem": ["Receptor"],
               "senses": 20},
    }

    def mine(self, text):
        return [s for s, _ in _mine_spans(text, self.GAZ)]

    def test_bare_gene_symbol_is_found(self):
        # The whole point: the regex cannot see this.
        assert "BRCA1" in self.mine("Loss of BRCA1 sensitises cells to PARP inhibition")

    def test_lowercase_drug_name_is_found(self):
        assert "osimertinib" in self.mine("Patients treated with osimertinib progressed")

    def test_a_function_word_span_is_not_an_entity(self):
        # "a to" normalises to "ato" and hits Arsenic Trioxide. Normalisation is what
        # makes the dictionary robust to spelling AND what lets nonsense in.
        assert "a to" not in self.mine("responded a to the same degree")

    def test_ordinary_words_are_rejected_even_when_NCIt_holds_them(self):
        assert "response" not in self.mine("the response was durable")

    def test_high_fanout_surface_forms_are_skipped(self):
        # `ER` matches 20 senses here: organelle, country, emergency room, receptor.
        assert "ER" not in self.mine("ER stress was induced")

    def test_moderate_fanout_is_allowed(self):
        # NCIt routinely holds a gene, its protein and its wt allele as three concepts
        # sharing one surface form. Requiring exactly one sense threw away BCL-2,
        # CALAA-01 and B16-F10 — the same entity in three guises, not rival senses.
        assert MAX_MINE_SENSES > 1


class TestMinedQueriesInheritJudgments:
    def test_judgment_comes_from_the_claim_not_from_us(self):
        gaz = {"brca1": {"code": "C17815", "name": "BRCA1",
                         "sem": ["Gene or Genome"], "senses": 1}}
        claims = [q("cite:1", "BRCA1 loss drives sensitivity", {"docA": 3},
                    stratum="claim")]
        out = gazetteer_identifier_queries(claims, gaz)
        assert out and out[0].query == "BRCA1"
        assert out[0].judgments == {"docA": 3}, "nothing new may be judged here"
        assert out[0].stratum == "identifier"

    def test_one_target_cannot_dominate_the_stratum(self):
        # 4.4's MAX_PER_TARGET: a landmark paper cited 90 times must not contribute 90
        # near-identical queries and turn the stratum into a measurement of that paper.
        gaz = {f"gene{i}": {"code": f"C{i}", "name": f"GENE{i}",
                            "sem": ["Gene or Genome"], "senses": 1} for i in range(10)}
        claims = [q(f"cite:{i}", " ".join(f"GENE{j}" for j in range(10)), {"docA": 3},
                    stratum="claim") for i in range(5)]
        out = gazetteer_identifier_queries(claims, gaz, max_per_target=4)
        assert len(out) <= 4
