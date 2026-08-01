"""Okapi BM25 over chunks, with an optional exact-literal boost.

BM25 is the arm that carries the ``lexical`` stratum. Dense retrieval structurally cannot
match a rare identifier it never saw in training, so this index is not a legacy baseline —
it is the only component that can answer "find me the paper about EGFR C797S".
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

from .text import is_rare_literal, tokenize


class BM25Index:
    def __init__(
        self,
        *,
        k1: float = 1.2,
        b: float = 0.75,
        literal_boost: float = 1.0,
    ) -> None:
        self.k1 = k1
        self.b = b
        # Multiplier applied to the IDF of tokens that look like identifiers. 1.0 = off.
        self.literal_boost = literal_boost
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.df: Counter[str] = Counter()
        self.n_docs: int = 0

    def build(self, doc_ids: Sequence[str], texts: Sequence[str]) -> "BM25Index":
        self.doc_ids = list(doc_ids)
        self.n_docs = len(self.doc_ids)
        self.postings = defaultdict(list)
        self.df = Counter()
        self.doc_len = []
        for i, text in enumerate(texts):
            toks = tokenize(text)
            self.doc_len.append(len(toks))
            tf = Counter(toks)
            for term, freq in tf.items():
                self.postings[term].append((i, freq))
                self.df[term] += 1
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        return self

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        if n == 0:
            return 0.0
        # Robertson/Sparck-Jones IDF with +1 to keep it non-negative.
        idf = math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5))
        if self.literal_boost != 1.0 and is_rare_literal(term):
            idf *= self.literal_boost
        return idf

    def score(self, query: str, *, query_tokens: Sequence[str] | None = None) -> dict[str, float]:
        toks = list(query_tokens) if query_tokens is not None else tokenize(query)
        if not toks:
            return {}
        scores = defaultdict(float)
        qtf = Counter(toks)
        for term, qf in qtf.items():
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            if idf <= 0:
                continue
            for doc_idx, freq in postings:
                dl = self.doc_len[doc_idx]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                # qf term lets a repeated/expanded query token contribute more than once
                # but with saturation, so a 40-synonym expansion cannot dominate.
                scores[doc_idx] += idf * (freq * (self.k1 + 1) / denom) * (qf / (qf + 0.5) * 1.5)
        return {self.doc_ids[i]: s for i, s in scores.items()}

    def search(self, query: str, k: int = 100, **kw) -> list[tuple[str, float]]:
        scores = self.score(query, **kw)
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
