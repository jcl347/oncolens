"""Biomedical-aware tokenization.

Generic tokenizers quietly destroy the exact strings that matter most in this domain.
``PD-L1`` becomes ``pd`` + ``l1``; ``EGFR C797S`` loses the variant; ``NCT04185883``
survives but ``T790M`` may be split. Since the ``lexical`` query stratum is built almost
entirely from these strings, tokenizer behaviour has a larger measured effect here than
most model choices — which is itself a finding worth keeping visible.

Strategy: emit **both** the joined and the split forms for hyphenated/alphanumeric terms,
so ``PD-L1`` matches queries written ``PD-L1``, ``PDL1`` or ``PD L1`` without needing the
synonym lexicon to cover every surface variant.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Splits on anything that is not alphanumeric or an intra-token joiner.
_SPLIT = re.compile(r"[^\w\-/+]+", re.UNICODE)
# A token that mixes letters and digits: gene variants (T790M, G12C), trial ids, codes.
_ALNUM_MIX = re.compile(r"^(?=.*[a-z])(?=.*\d)[a-z0-9]+$")
_JOINERS = re.compile(r"[\-/+]")

STOPWORDS = frozenset("""
a an and are as at be been but by for from has have how i if in into is it its of on or
that the their then there these they this to was were what when where which who will with
we our us can could would should may might do does did not no nor
""".split())


def _variants(token: str) -> list[str]:
    """Expand one raw token into the surface forms it should be indexed under."""
    out = [token]
    if _JOINERS.search(token):
        parts = [p for p in _JOINERS.split(token) if p]
        out.append("".join(parts))  # pd-l1 -> pdl1
        out.extend(p for p in parts if len(p) > 1 or p.isdigit())  # -> pd, l1
    return out


def tokenize(text: str, *, drop_stopwords: bool = True, keep_short_alnum: bool = True) -> list[str]:
    """Tokenize to lowercase terms, preserving biomedical literals.

    ``keep_short_alnum`` retains 2-character letter+digit tokens (``L1``, ``G1``) that a
    plain ``len > 2`` filter would delete — these carry real signal in this domain.
    """
    tokens: list[str] = []
    for raw in _SPLIT.split(text.lower()):
        if not raw:
            continue
        raw = raw.strip("-/+")
        if not raw:
            continue
        for tok in _variants(raw):
            if not tok:
                continue
            if drop_stopwords and tok in STOPWORDS:
                continue
            if len(tok) < 2:
                continue
            if len(tok) == 2 and not (keep_short_alnum and _ALNUM_MIX.match(tok)):
                continue
            tokens.append(tok)
    return tokens


def is_rare_literal(token: str) -> bool:
    """Heuristic for 'this looks like an identifier, not a word'.

    Used to give exact-match evidence extra weight, and to detect queries in the
    ``lexical`` stratum at runtime without peeking at the stratum label.
    """
    return bool(_ALNUM_MIX.match(token)) and len(token) >= 3


def ngrams(tokens: Iterable[str], n: int) -> list[str]:
    toks = list(tokens)
    if n <= 1:
        return toks
    return ["_".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]
