"""The route from a promotion to production, which did not exist for five rounds.

`improve_loop` has written `config/served.json` on every promotion since round 1. Nothing
read it, the directory was never created, and `api/search.py` called `LiveIndex.search`
without weights. Four promoted candidates, zero user-visible change — §4.15 at its most
consequential: not a missing feature, a finished one nothing invoked.

These pin the properties that make shipping safe rather than merely possible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pytest  # noqa: E402

from oncolens.serve import served_config as SC  # noqa: E402


class TestDefaults:
    def test_absent_config_serves_round_zero(self, tmp_path):
        """Absence must mean the measured baseline. Inferring intent from a missing file
        is how §4.6 nearly served LSA vectors to an OpenAI encoder."""
        cfg = SC.load(tmp_path)
        assert cfg["bm25_weight"] == 1.0 and cfg["dense_weight"] == 1.0
        assert cfg["cross_encoder"] is None

    def test_a_broken_config_raises_rather_than_falling_back(self, tmp_path):
        """A promotion that silently fails to apply is worse than one that refuses: the
        next measurement would be against a config that never took effect."""
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "served.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(SC.ServedConfigError):
            SC.load(tmp_path)


class TestEmbeddingSpaceGuard:
    """§4.6: a query vector from one model against document vectors from another raises
    nothing and returns a confident, meaningless ranking."""

    IDX = {"embedding_model": "openai", "embedding_dim": "192"}

    def test_width_mismatch_is_refused(self):
        """The case that actually got through the first version of this guard: comparing
        only the model family let `openai-768` pass against an `openai`/192 index, because
        both share the prefix. `dense_weight_2x` is exactly this config."""
        with pytest.raises(SC.ServedConfigError, match="dim"):
            SC.validate({"dense_backend": "openai-768"}, index_config=self.IDX)

    def test_family_mismatch_is_refused(self):
        with pytest.raises(SC.ServedConfigError):
            SC.validate({"dense_backend": "medcpt"}, index_config=self.IDX)

    def test_a_matching_backend_is_allowed(self):
        assert SC.validate({"dense_backend": "openai-192"}, index_config=self.IDX)
        assert SC.validate({"dense_backend": "openai"}, index_config=self.IDX)

    def test_no_index_config_does_not_block_serving(self):
        """The guard needs the store; its absence must not take the site down."""
        assert SC.validate({"dense_backend": "openai-768"}, index_config=None)


class TestTyposAndBounds:
    def test_unknown_keys_are_rejected(self):
        """A typo'd key that silently does nothing is indistinguishable from a change that
        did not help — which is the failure mode this whole module addresses."""
        with pytest.raises(SC.ServedConfigError, match="unknown key"):
            SC.validate({"bm25_wieght": 2.0})

    @pytest.mark.parametrize("cfg", [
        {"bm25_weight": 50}, {"dense_weight": -1}, {"candidates": 5},
        {"cross_depth": 10_000},
    ])
    def test_absurd_values_are_rejected(self, cfg):
        with pytest.raises(SC.ServedConfigError):
            SC.validate(cfg)

    def test_a_real_promotion_validates(self):
        cfg = SC.validate({"cross_encoder": "medcpt-cross", "cross_depth": 50,
                           "promoted_from": "rerank_medcpt_cross"})
        assert cfg["cross_encoder"] == "medcpt-cross"


class TestObservability:
    def test_the_active_config_is_returned_to_clients(self):
        """A setting that cannot be observed from outside is one nobody can verify is on.
        This module exists because a file sat unread for five rounds."""
        src = (REPO / "src" / "oncolens" / "serve" / "live_query.py").read_text(
            encoding="utf-8")
        assert '"config": served_config.describe(cfg)' in src

    def test_search_defaults_come_from_the_config_not_from_literals(self):
        src = (REPO / "src" / "oncolens" / "serve" / "live_query.py").read_text(
            encoding="utf-8")
        block = src.split("def search(")[1][:900]
        for line in ('cfg["candidates"] if candidates is None',
                     'cfg["bm25_weight"] if bm25_weight is None',
                     'cfg["dense_weight"] if dense_weight is None'):
            assert line in block, f"search() must default from the served config: {line}"

    def test_explicit_arguments_still_win(self):
        """The harness pins configurations; the config must not override an explicit one."""
        src = (REPO / "src" / "oncolens" / "serve" / "live_query.py").read_text(
            encoding="utf-8")
        assert "is None else candidates" in src


class TestWriterReaderAgreement:
    """The loop writes this file. Whatever it writes must be loadable — otherwise the two
    halves drift and the promotion is inert again, silently."""

    def test_the_loop_writes_keys_the_loader_accepts(self, tmp_path):
        loop = (REPO / "scripts" / "improve_loop.py").read_text(encoding="utf-8")
        assert 'config" / "served.json"' in loop, "the writer moved; update this test"
        # Round-trip the shape the loop emits.
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "served.json").write_text(
            json.dumps({"config": {"bm25_weight": 1.0, "dense_weight": 2.0},
                        "promoted_from": "some_candidate"}), encoding="utf-8")
        cfg = SC.load(tmp_path)
        assert cfg["dense_weight"] == 2.0
