"""The configuration production actually serves, and the reason it did not exist.

**The gap this closes.** `improve_loop` has written `config/served.json` on every promotion
since round 1. Nothing has ever read it — no Python, no TypeScript, and the directory was
never created. Five rounds of measurement produced four promoted candidates and zero
user-visible change, because there was no route from a promotion to the served path. It is
the §4.15 pattern at its most consequential: not a missing feature, a finished one that
nothing invokes.

**Design constraints, each from a scar in this repo.**

* **Absent config means round-0, not "whatever is newest".** A missing file must serve the
  measured baseline, because the alternative — inferring intent — is how §4.6's absent
  `index_config` nearly served LSA vectors to an OpenAI encoder.
* **A config that does not match the index is refused, loudly.** §4.6 again: a query vector
  from one model against document vectors from another raises nothing and returns a
  confident, meaningless ranking. Serving `openai-768` against a `vector(192)` column is
  exactly that failure, and `dense_weight_2x` is precisely such a config.
* **The active config is returned in the response.** A setting that cannot be observed from
  outside is a setting nobody can verify is on, and this whole module exists because of a
  file nobody noticed was inert.
* **Unknown keys are rejected rather than ignored.** A typo'd key that silently does
  nothing looks identical to a change that did not help.
"""

from __future__ import annotations

import json
from pathlib import Path

#: The measured round-0 configuration. Every candidate in the ledger is scored against
#: this, so it is the only defensible default.
BASELINE_SERVED: dict = {
    "bm25_weight": 1.0,
    "dense_weight": 1.0,
    "candidates": 200,
    "cross_encoder": None,
    "cross_depth": 50,
    "note": "round-0 baseline; nothing promoted has been shipped",
}

_ALLOWED = set(BASELINE_SERVED) | {"dense_backend", "promoted_from", "promoted_at"}

#: Numeric bounds. A weight of 50 is a typo, not a decision.
_BOUNDS = {"bm25_weight": (0.0, 5.0), "dense_weight": (0.0, 5.0),
           "candidates": (10, 2000), "cross_depth": (1, 500)}


class ServedConfigError(ValueError):
    """Raised rather than falling back, when a config is present but unusable."""


def config_path(root: Path | None = None) -> Path:
    root = root or Path(__file__).resolve().parents[3]
    return root / "config" / "served.json"


def validate(cfg: dict, *, index_config: dict | None = None) -> dict:
    """Merge over the baseline, or refuse. Never silently drop a key.

    ``index_config`` is the store's own record of what the embedding column holds. When
    supplied, a config naming a different dense backend is **refused** — the §4.6 rule that
    absent or mismatched embedding config is the dangerous case, not the harmless one.
    """
    unknown = set(cfg) - _ALLOWED
    if unknown:
        raise ServedConfigError(
            f"unknown key(s) {sorted(unknown)}; a typo that silently does nothing is "
            f"indistinguishable from a change that did not help")

    out = {**BASELINE_SERVED, **cfg}
    for k, (lo, hi) in _BOUNDS.items():
        v = out.get(k)
        if v is None:
            continue
        if not isinstance(v, (int, float)) or not (lo <= v <= hi):
            raise ServedConfigError(f"{k}={v!r} outside [{lo}, {hi}]")

    want = cfg.get("dense_backend")
    if want and index_config:
        # BOTH the family and the width must match. Comparing only the family let
        # `openai-768` pass against an `openai`/192 index — the exact §4.6 failure this
        # guard exists to prevent, and it was caught only because the guard was tested
        # rather than assumed. `dense_weight_2x` is precisely such a config.
        have_model = str(index_config.get("embedding_model") or "")
        have_dim = str(index_config.get("embedding_dim") or "")
        want_family, _, want_dim = want.partition("-")
        if have_model and want_family != have_model.partition("-")[0]:
            raise ServedConfigError(
                f"config asks for dense_backend={want!r} but the index holds "
                f"{have_model!r} vectors. Serving this compares two unrelated embedding "
                f"spaces, which raises nothing and returns a confident meaningless "
                f"ranking (§4.6). Re-embed first.")
        if want_dim and have_dim and want_dim != have_dim:
            raise ServedConfigError(
                f"config asks for dense_backend={want!r} ({want_dim}-dim) but the index "
                f"column holds {have_dim}-dim vectors. Same model, different space — "
                f"cosine distance compares them happily and returns nonsense (§4.6). "
                f"Run scripts/reembed_store.py first.")
    return out


def load(root: Path | None = None, *, index_config: dict | None = None) -> dict:
    """The served configuration.

    Returns the round-0 baseline when no file exists. A file that exists and is broken
    **raises**: a promotion that silently fails to apply is worse than one that refuses,
    because the loop would then measure a change that never took effect.
    """
    p = config_path(root)
    if not p.exists():
        return dict(BASELINE_SERVED)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise ServedConfigError(f"{p} is not readable JSON: {e}") from e
    cfg = raw.get("config", raw) if isinstance(raw, dict) else {}
    if not isinstance(cfg, dict):
        raise ServedConfigError(f"{p} does not contain a config object")
    return validate(cfg, index_config=index_config)


def describe(cfg: dict) -> dict:
    """The subset worth returning to a client, so what is running is observable."""
    return {k: cfg.get(k) for k in
            ("bm25_weight", "dense_weight", "candidates", "cross_encoder", "cross_depth",
             "dense_backend", "promoted_from")
            if cfg.get(k) is not None}
