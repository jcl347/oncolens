"""Figures: the retrieval-isolation guarantee, and the response shape the reader depends on.

Two separate things are pinned here, and the first matters more.

**1. Figure rows must not reach retrieval.** They were added so a paper can be READ with its
images. Their effect on ranking has never been measured, so the promise made in
docs/PLAN_FIGURES.md §3.9 is that adding them provably cannot move a metric. That promise
lives entirely in two SQL predicates. If someone deletes one, every stored figure silently
joins the corpus, 16k rows appear in the index, and the next candidate's result is measured
against a baseline that quietly changed underneath it — with no error and no test failure.
That is §4.6's shape: a mismatch that raises nothing and returns confident numbers.

**2. The paper response carries what the page renders.** §4.15 is the precedent: `best_clause`
existed on the server for months, was never sent, and every production query lost its
highlight. A field's existence proves nothing; something must assert it is emitted.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pytest  # noqa: E402

from oncolens.serve import neon_store  # noqa: E402


class TestFiguresAreExcludedFromRetrieval:
    """The isolation guarantee, asserted on the SQL itself.

    Checking the text of the query rather than running it is deliberate: a contract test
    that needs a live Postgres is a contract test nobody runs (the reasoning in
    test_search_contract.py), and this is exactly the guard that must never be quietly
    dropped.
    """

    def test_both_arms_filter_to_passages(self):
        sql = neon_store.HYBRID_SEARCH_SQL
        # lexical CTE and dense CTE must each carry the predicate
        lexical = sql.split("dense AS")[0]
        dense = sql.split("dense AS")[1].split("fused AS")[0]
        assert "kind = 'passage'" in lexical, "lexical arm would retrieve figure captions"
        assert "kind = 'passage'" in dense, "dense arm would retrieve figure captions"

    def test_dense_arm_also_guards_null_embeddings(self):
        """Figure rows have no embedding. A NULL in an ORDER BY on a vector distance sorts
        unpredictably, so relying on the kind filter alone would leave a second, subtler
        path for them to leak in."""
        dense = neon_store.HYBRID_SEARCH_SQL.split("dense AS")[1].split("fused AS")[0]
        assert "embedding IS NOT NULL" in dense

    def test_the_predicate_is_defined_once(self):
        """Two copies of a guard is one copy that gets updated."""
        assert hasattr(neon_store, "_RETRIEVABLE")
        assert "passage" in neon_store._RETRIEVABLE

    def test_compare_does_not_answer_aspects_from_captions(self):
        """A caption satisfying an aspect would render as though the paper reported it in
        text — the over-claim §4.15 corrected in the compare grid."""
        src = (REPO / "src" / "oncolens" / "serve" / "live_query.py").read_text(
            encoding="utf-8")
        aspect_q = src.split("SELECT DISTINCT ON (c.doc_id)")[1].split('"""')[0]
        assert "kind = 'passage'" in aspect_q


class TestEveryConsumerOfChunksIsGuarded:
    """The guard that was written for the serving path and NOT for the evaluation path.

    This actually happened, minutes after the serving guard was written and tested. The
    evaluation harness has its own SQL, so the first run after the figure ingest picked up
    **190,399** rows instead of 180,850, began re-encoding the whole corpus because the new
    rows have no embedding, and would have scored a candidate against a baseline that had
    silently grown by 9,549 documents. It was killed before any verdict was recorded.

    A per-file test rather than one assertion, because the failure mode is *a consumer
    nobody remembered*, and eleven places read this table.
    """

    #: Files whose reads must be restricted to body passages. DELETE statements are
    #: deliberately absent: removing a document SHOULD remove its figures too, and an
    #: orphan figure is worse than a deleted one.
    CONSUMERS = [
        ("scripts/improve_loop.py", "the evaluation harness — this is the one that broke"),
        ("scripts/bench_retrieval.py", "the retrieval benchmark corpus"),
        ("scripts/build_clusters.py", "the cluster map; figure rows have no embedding"),
        ("scripts/build_journey_data.py", "passage counts displayed on the site"),
        ("scripts/build_strata.py", "the leakage-check sample"),
        ("scripts/reembed_store.py", "would pay to embed rows nothing retrieves"),
        ("scripts/query_neon.py", "the CLI must match what production serves"),
        ("scripts/prune_abstract_only.py", "per-document counts drive pruning decisions"),
    ]

    @pytest.mark.parametrize("path,why", CONSUMERS)
    def test_reads_are_restricted_to_passages(self, path, why):
        src = (REPO / path).read_text(encoding="utf-8")
        for i, line in enumerate(src.splitlines()):
            if "FROM chunks" not in line:
                continue
            if re.search(r"\bDELETE\b", line, re.I):
                continue
            window = "\n".join(src.splitlines()[max(0, i - 2): i + 4])
            assert "kind" in window, (
                f"{path}:{i+1} reads chunks without a kind filter — {why}")

    def test_the_harness_query_is_explicit(self):
        """Named separately because it is the one that actually failed."""
        src = (REPO / "scripts" / "improve_loop.py").read_text(encoding="utf-8")
        assert "WHERE kind = 'passage' ORDER BY chunk_id" in src, (
            "load_chunks must filter AND order: the filter keeps figures out of the "
            "corpus, the ORDER BY keeps the embedding cache deterministic (§6.6)")


class TestPaperResponseShape:
    """What /api/paper must emit for the reading page to render an image."""

    REQUIRED = ("figure_id", "label", "caption", "image_uri",
                "figure_type", "figure_type_source")

    def test_get_document_selects_every_rendered_field(self):
        src = (REPO / "src" / "oncolens" / "serve" / "live_query.py").read_text(
            encoding="utf-8")
        block = src.split("kind = 'figure'")[1][:900]
        for field in ("figure_id", "label", "caption", "image_uri", "figure_type"):
            assert field in block, f"the reading page reads {field!r} and nothing emits it"

    def test_response_includes_figures_and_a_count(self):
        src = (REPO / "src" / "oncolens" / "serve" / "live_query.py").read_text(
            encoding="utf-8")
        assert '"figures": figures' in src
        assert '"n_figures"' in src

    def test_figures_without_an_image_are_dropped(self):
        """A figure row whose URL failed verification must not reach the page: rendering a
        broken image and calling it provenance is worse than omitting the figure."""
        src = (REPO / "src" / "oncolens" / "serve" / "live_query.py").read_text(
            encoding="utf-8")
        block = src.split("kind = 'figure'")[1][:900]
        assert "if r[3]" in block, "rows with a NULL image_uri are not being filtered out"

    def test_page_distinguishes_stated_from_inferred_type(self):
        """§4.15's 'not reported' lesson. A type read off the caption and a type guessed by
        a model must not render identically."""
        page = (REPO / "app" / "paper" / "page.tsx").read_text(encoding="utf-8")
        assert 'figure_type_source === "caption"' in page
        assert "inferred" in page.lower()


class TestTables:
    """Tables ship as PARSED ROWS, never as publisher markup.

    95.3% of this corpus's tables carry a machine-readable `<table>`, so the numbers are
    already structured — no OCR, no derendering, no vision model. The one hazard is
    rendering: injecting publisher HTML into the page would be an XSS surface for zero
    benefit, so the store holds a JSON grid and the page builds a real `<table>` from it.
    """

    def test_rows_are_stored_as_json_not_markup(self):
        src = (REPO / "scripts" / "ingest_figures.py").read_text(encoding="utf-8")
        assert re.search(r"table_rows\s+JSONB", src), "table_rows must be a JSONB column"

    def test_page_never_injects_publisher_markup(self):
        # Match the JSX ATTRIBUTE, not the word. The first version of this test failed on
        # the doc comment above the Table type, which explains why the property holds —
        # a test literal enough to be tripped by its own documentation.
        page = (REPO / "app" / "paper" / "page.tsx").read_text(encoding="utf-8")
        assert not re.search(r"dangerouslySetInnerHTML\s*=", page)

    def test_only_machine_readable_tables_are_ingested(self):
        """A `<table-wrap>` that is only an image has no grid to show; pretending
        otherwise puts an empty table on the page."""
        src = (REPO / "scripts" / "ingest_figures.py").read_text(encoding="utf-8")
        block = src.split("def tables_from_jats")[1].split("def resolve_image")[0]
        assert 'tbl = tw.find(".//table")' in block
        assert "if tbl is None:" in block and "continue" in block

    def test_truncation_is_disclosed(self):
        """Showing 40 rows of a 300-row table without saying so is the 'not reported'
        over-claim in a new place."""
        src = (REPO / "scripts" / "ingest_figures.py").read_text(encoding="utf-8")
        assert '"truncated"' in src
        page = (REPO / "app" / "paper" / "page.tsx").read_text(encoding="utf-8")
        assert "t.truncated" in page and "of {t.n_rows} rows" in page

    def test_footer_is_carried(self):
        """The abbreviation key and significance markers are where a table's units live."""
        src = (REPO / "scripts" / "ingest_figures.py").read_text(encoding="utf-8")
        assert "table-wrap-foot" in src

    def test_tables_are_excluded_from_retrieval(self):
        """Same guarantee as figures: the predicate is kind='passage', so 'table' rows are
        outside both arms by construction."""
        assert neon_store._RETRIEVABLE.strip() == "c.kind = 'passage'"

    def test_api_emits_what_the_page_reads(self):
        src = (REPO / "src" / "oncolens" / "serve" / "live_query.py").read_text(
            encoding="utf-8")
        assert '"tables": tables' in src and '"n_tables"' in src
        block = src.split("kind = 'table'")[1][:1200]
        for field in ("table_id", "label", "caption", "rows", "n_rows", "truncated", "foot"):
            assert field in block, f"the page reads {field!r} and nothing emits it"


class TestIngestClassifier:
    """The caption-first type cascade. Captions are precise when they fire; silence is
    reported as unknown rather than guessed."""

    @pytest.mark.parametrize("caption,want", [
        ("Kaplan-Meier curves for overall survival by treatment arm", "kaplan-meier"),
        ("Forest plot of hazard ratios across subgroups", "forest-plot"),
        ("Western blot showing PARP cleavage", "western-blot"),
        ("Representative flow cytometry plots with CD8 gating", "flow-cytometry"),
        ("Study flowchart and CONSORT diagram", "schematic"),
        ("Immunohistochemical staining for Ki-67", "microscopy"),
    ])
    def test_types_from_the_publishers_own_words(self, caption, want):
        sys.path.insert(0, str(REPO / "scripts"))
        import ingest_figures as ing
        assert ing.classify(caption) == want

    def test_an_undescribed_figure_returns_none_not_a_guess(self):
        sys.path.insert(0, str(REPO / "scripts"))
        import ingest_figures as ing
        assert ing.classify("Results of the second experiment.") is None
