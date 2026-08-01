"""Dense semantic retrieval with a pluggable embedding backend.

No network and no API key are available in this environment, so the default backend is
**LSA** (TF-IDF followed by truncated SVD). That is a real dense-retrieval method, not a
placeholder: it learns a latent semantic space from the corpus, so it genuinely matches
paraphrase and genuinely fails on unseen rare identifiers — the same qualitative profile
as a neural embedder, which is what makes the hybrid comparison meaningful.

``VoyageBackend`` is wired and ready; supplying ``VOYAGE_API_KEY`` with network access
swaps it in with no other change. The measurement harness is backend-agnostic by design,
so the same experiment ledger can compare LSA against a real embedder later.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from .text import tokenize


class EmbeddingBackend(Protocol):
    name: str
    def fit(self, texts: Sequence[str]) -> None: ...
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class LsaBackend:
    """TF-IDF + truncated SVD (latent semantic indexing)."""

    name = "lsa"

    def __init__(self, dim: int = 192, sublinear_tf: bool = True, min_df: int = 1) -> None:
        self.dim = dim
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.components: np.ndarray | None = None  # (dim, vocab)

    def _tfidf(self, texts: Sequence[str], *, fit: bool) -> csr_matrix:
        rows, cols, vals = [], [], []
        docs_tokens = [tokenize(t) for t in texts]
        if fit:
            df: dict[str, int] = {}
            for toks in docs_tokens:
                for term in set(toks):
                    df[term] = df.get(term, 0) + 1
            self.vocab = {t: i for i, t in enumerate(sorted(t for t, c in df.items() if c >= self.min_df))}
            n = len(texts)
            idf = np.zeros(len(self.vocab), dtype=np.float64)
            for term, i in self.vocab.items():
                idf[i] = math.log((1 + n) / (1 + df[term])) + 1.0
            self.idf = idf
        assert self.idf is not None
        for r, toks in enumerate(docs_tokens):
            counts: dict[int, int] = {}
            for t in toks:
                j = self.vocab.get(t)
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                tf = (1.0 + math.log(c)) if self.sublinear_tf else float(c)
                rows.append(r); cols.append(j); vals.append(tf * self.idf[j])
        m = csr_matrix((vals, (rows, cols)), shape=(len(texts), max(1, len(self.vocab))))
        return m

    def fit(self, texts: Sequence[str]) -> None:
        X = self._tfidf(texts, fit=True)
        k = min(self.dim, min(X.shape) - 1)
        if k < 2:
            self.components = np.eye(X.shape[1])[: max(2, k)]
            return
        # svds returns singular triplets in ascending order; reverse for descending.
        _, _, vt = svds(X.asfptype(), k=k)
        self.components = vt[::-1].copy()

    def _project(self, texts: Sequence[str]) -> np.ndarray:
        X = self._tfidf(texts, fit=False)
        assert self.components is not None
        return _l2(np.asarray(X @ self.components.T))

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._project(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._project(texts)


class VoyageBackend:
    """Real neural embeddings. Requires VOYAGE_API_KEY and network access.

    ``input_type`` is set asymmetrically ('query' vs 'document') because Voyage prepends
    different instruction prefixes for each, and omitting it measurably degrades retrieval.
    """

    name = "voyage"

    def __init__(self, model: str = "voyage-4", batch: int = 96) -> None:
        self.model = model
        self.batch = batch
        self._client = None

    def _client_or_raise(self):
        if self._client is None:
            if not os.environ.get("VOYAGE_API_KEY"):
                raise RuntimeError("VOYAGE_API_KEY not set — cannot use VoyageBackend")
            import voyageai  # imported lazily; absent in the offline environment
            self._client = voyageai.Client()
        return self._client

    def fit(self, texts: Sequence[str]) -> None:
        return None  # no corpus fitting required

    def _encode(self, texts: Sequence[str], input_type: str) -> np.ndarray:
        client = self._client_or_raise()
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            r = client.embed(list(texts[i : i + self.batch]), model=self.model, input_type=input_type)
            out.extend(r.embeddings)
        return _l2(np.asarray(out, dtype=np.float64))

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "document")

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "query")


class DenseIndex:
    def __init__(self, backend: EmbeddingBackend | None = None) -> None:
        self.backend = backend or LsaBackend()
        self.doc_ids: list[str] = []
        self.matrix: np.ndarray | None = None

    def build(self, doc_ids: Sequence[str], texts: Sequence[str]) -> "DenseIndex":
        self.doc_ids = list(doc_ids)
        self.backend.fit(texts)
        self.matrix = self.backend.encode_documents(texts)
        return self

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        if self.matrix is None:
            return []
        q = self.backend.encode_queries([query])[0]
        sims = self.matrix @ q
        if k >= len(sims):
            idx = np.argsort(-sims)
        else:
            part = np.argpartition(-sims, k)[:k]
            idx = part[np.argsort(-sims[part])]
        return [(self.doc_ids[i], float(sims[i])) for i in idx]
