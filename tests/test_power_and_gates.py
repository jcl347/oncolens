"""The two numbers that decide what gets promoted, and what the site claims about power.

Neither had a test. Both were wrong in the optimistic direction, which is the direction
that produces confident wrong conclusions rather than cautious ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import pytest  # noqa: E402

from oncolens.eval.weighting import (  # noqa: E402
    ORDERING_METRIC, ORDERING_ONLY, PRIMARY_METRIC, gate_metric,
)


class TestGateMetric:
    """The redirect must fire only where the stratum's primary is a COVERAGE metric.

    It previously fired on `concept`, whose primary `success@5` is a rank threshold a
    depth-50 reranker plainly moves. That swapped a movable metric for the panel's noisiest
    one (paired sd_diff 0.375 -> 0.482 on the same stratum) and discarded
    `rerank_medcpt_cross` at success@1 p=0.0315 while its own primary sat at p=0.0160.
    """

    COVERAGE_PREFIXES = ("recall@",)

    def test_redirect_only_where_primary_is_coverage(self):
        for stratum in ORDERING_METRIC:
            primary = PRIMARY_METRIC[stratum]
            assert primary.startswith(self.COVERAGE_PREFIXES), (
                f"{stratum} is redirected but its primary {primary!r} is not a coverage "
                f"metric — the redirect's own justification does not apply")

    @pytest.mark.parametrize("stratum", ["concept", "identifier", "claim"])
    def test_rank_sensitive_strata_keep_their_primary(self, stratum):
        for cand in ORDERING_ONLY:
            assert gate_metric(stratum, cand) == PRIMARY_METRIC[stratum], (
                f"{cand} on {stratum} must be judged on the same bar as every other "
                f"candidate on that stratum")

    def test_synthesis_is_still_redirected(self):
        """The original §4.8 fix must survive the scoping."""
        assert PRIMARY_METRIC["synthesis"].startswith("recall@")
        assert gate_metric("synthesis", "rerank_medcpt_cross") == "ndcg@10"

    def test_non_ordering_candidates_are_never_redirected(self):
        for stratum in PRIMARY_METRIC:
            assert gate_metric(stratum, "tri_fusion") == PRIMARY_METRIC[stratum]


class TestEmpiricalSd:
    """Published MDEs come from here. It reported the tightest metric's variance for every
    stratum, so concept published 0.0154 against a real success@5 floor near 0.054 and the
    site rendered a green 'sees 0.02' badge for a stratum that cannot see 0.02."""

    def _ledger(self, tmp_path, entries):
        import json
        (tmp_path / "improve_ledger.json").write_text(json.dumps(entries), encoding="utf-8")
        return tmp_path

    def test_keys_by_stratum_and_metric(self, tmp_path):
        import build_journey_data as B
        d = self._ledger(tmp_path, [{
            "stratum": "concept", "n_queries": 363,
            "verdicts": [{"name": "x", "deltas": {
                "success@5": {"ci": [-0.04, 0.04], "n": 363},
                "ndcg@10":   {"ci": [-0.01, 0.01], "n": 363},
            }}]}])
        emp = B.empirical_sd(d)
        assert ("concept", "success@5") in emp
        assert ("concept", "ndcg@10") in emp
        # the wide metric must NOT inherit the narrow one's variance
        assert emp[("concept", "success@5")] > emp[("concept", "ndcg@10")] * 3

    def test_entries_without_a_stratum_are_skipped_not_guessed(self, tmp_path):
        """§4.14: matching by query count made identifier inherit concept's variance."""
        import build_journey_data as B
        d = self._ledger(tmp_path, [{
            "n_queries": 363,  # no 'stratum' field — predates it
            "verdicts": [{"name": "x", "deltas": {
                "success@5": {"ci": [-0.04, 0.04], "n": 363}}}]}])
        assert B.empirical_sd(d) == {}

    def test_repeated_estimates_use_the_median_not_the_minimum(self, tmp_path):
        """The minimum of several noisy estimates of one variance is biased low — which is
        how the optimistic floor was produced in the first place."""
        import build_journey_data as B
        mk = lambda w: {"ci": [-w / 2, w / 2], "n": 100}  # noqa: E731
        d = self._ledger(tmp_path, [{
            "stratum": "claim", "n_queries": 100,
            "verdicts": [{"name": "a", "deltas": {"mrr": mk(0.02)}},
                         {"name": "b", "deltas": {"mrr": mk(0.10)}},
                         {"name": "c", "deltas": {"mrr": mk(0.12)}}]}])
        emp = B.empirical_sd(d)
        lo = 0.02 * (100 ** 0.5) / 3.919928
        med = 0.10 * (100 ** 0.5) / 3.919928
        assert emp[("claim", "mrr")] == pytest.approx(med, rel=1e-6)
        assert emp[("claim", "mrr")] > lo


class TestPowerTableHonesty:
    """What the site renders must describe the metric promotion is decided on."""

    def test_mde_is_reported_for_the_gate_metric(self):
        import json
        p = REPO / "public" / "journey.json"
        if not p.exists():
            pytest.skip("journey.json not built")
        rows = json.loads(p.read_text(encoding="utf-8")).get("power") or []
        assert rows, "power table is empty"
        for r in rows:
            assert r["gate_metric"] == PRIMARY_METRIC[r["stratum"]]
            # the analytic and measured figures must be computed at the same n, or the
            # page pairs a full-stratum count with a dev-split floor
            assert r["dev_queries"] <= r["queries"]
            assert r["mde"] > 0

    def test_sees_002_badge_matches_the_number_beside_it(self):
        import json
        p = REPO / "public" / "journey.json"
        if not p.exists():
            pytest.skip("journey.json not built")
        for r in json.loads(p.read_text(encoding="utf-8")).get("power") or []:
            assert r["sees_002"] == (r["mde"] <= 0.02), (
                f"{r['stratum']}: badge says sees_002={r['sees_002']} beside MDE "
                f"{r['mde']}")
