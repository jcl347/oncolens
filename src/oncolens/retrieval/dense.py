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

from .text import tokenize

# scipy is imported lazily inside LsaBackend, NOT at module scope.
#
# Importing it here made every consumer of this module depend on scipy — including the
# serverless request path, which only ever constructs OpenAIBackend and needs none of it.
# Vercel's function bundle and cold start both pay for that, and the LSA backend it
# supports was measured to be the weakest arm available (nDCG@10 0.3088 against 0.4116 for
# OpenAI embeddings), so making the whole deployment carry scipy for its sake is backwards.


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

    def _tfidf(self, texts: Sequence[str], *, fit: bool):
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
        from scipy.sparse import csr_matrix

        m = csr_matrix((vals, (rows, cols)), shape=(len(texts), max(1, len(self.vocab))))
        return m

    def fit(self, texts: Sequence[str]) -> None:
        X = self._tfidf(texts, fit=True)
        k = min(self.dim, min(X.shape) - 1)
        if k < 2:
            self.components = np.eye(X.shape[1])[: max(2, k)]
            return
        # svds returns singular triplets in ascending order; reverse for descending.
        from scipy.sparse.linalg import svds

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


class OpenAIBackend:
    """OpenAI ``text-embedding-3-*``. Requires OPENAI_API_KEY.

    **Why this exists.** The default :class:`LsaBackend` is TF-IDF + SVD, which is a purely
    *lexical* model wearing a dense coat: it can only relate terms that co-occur in this
    corpus. It cannot know that "EGFR TKI" and "erlotinib" are the same idea unless the
    corpus happens to put them in the same documents. Since the product's whole premise is
    searching by *concept*, that ceiling is the thing worth attacking.

    ``dimensions`` is passed through because ``text-embedding-3-*`` supports Matryoshka
    truncation: shorter vectors cost less to store and index in pgvector, and the models
    are trained so the leading dimensions carry the most information. Whether the shorter
    vector actually costs accuracy on *this* corpus is a measurable question, not an
    assumption — ``scripts/bench_embeddings.py`` answers it.

    Unlike Voyage, OpenAI's embedding models are **symmetric**: there is no separate query
    instruction, so queries and documents are encoded identically. Inventing an asymmetric
    prefix here would be cargo-culting from another provider's API.
    """

    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small", batch: int = 128,
                 dimensions: int | None = None) -> None:
        self.model = model
        self.batch = batch
        self.dimensions = dimensions
        self._client = None

    def _client_or_raise(self):
        if self._client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY not set — cannot use OpenAIBackend")
            from openai import OpenAI  # lazy: absent unless the extra is installed
            self._client = OpenAI()
        return self._client

    def fit(self, texts: Sequence[str]) -> None:
        return None  # no corpus fitting required

    def _embed_batch(self, client, batch: list[str], attempt: int = 0) -> list[list[float]]:
        """One API call, retrying on rate limits.

        A 1M tokens-per-minute ceiling is reached in seconds when embedding a 59k-passage
        corpus, so this is the normal path, not an exceptional one. Backoff is exponential
        with a cap; the request is not split, because a 429 is about *rate*, not size.
        """
        import time

        kw: dict = {"model": self.model, "input": batch}
        if self.dimensions:
            kw["dimensions"] = self.dimensions
        try:
            resp = client.embeddings.create(**kw)
        except Exception as exc:  # noqa: BLE001
            name = type(exc).__name__
            transient = name in ("RateLimitError", "APIConnectionError", "APITimeoutError",
                                 "InternalServerError")
            if not transient or attempt >= 6:
                raise
            time.sleep(min(2.0 * (2 ** attempt), 60.0))
            return self._embed_batch(client, batch, attempt + 1)
        # The API does not guarantee ordering; ``index`` does.
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    #: ``text-embedding-3-*`` reject inputs over 8192 tokens. Leave headroom for the
    #: model's own special tokens.
    MAX_INPUT_TOKENS = 8000
    #: Character fallback used only when tiktoken is unavailable. Deliberately pessimistic:
    #: it assumes ~1 token per character, which no real prose reaches but dense tables
    #: approach.
    MAX_INPUT_CHARS_FALLBACK = 7800

    _encoder = None

    @classmethod
    def _truncate(cls, text: str) -> str:
        """Cut to the model's token limit, counting tokens rather than estimating them.

        **A corpus average is the wrong tool here.** The first version truncated at 24,000
        characters, justified by a measured corpus mean of 0.272 tokens/character. That
        holds for prose and fails badly on the tab-separated tables lifted out of JATS,
        where identifiers and numerals tokenize several times denser — those inputs still
        exceeded 8192 tokens and failed the whole batch. Same mistake as trusting a mean
        instead of a distribution; here it is cheap to just count.
        """
        if cls._encoder is None:
            try:
                import tiktoken
                cls._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:  # noqa: BLE001
                cls._encoder = False
        if cls._encoder is False:
            return text[: cls.MAX_INPUT_CHARS_FALLBACK]
        toks = cls._encoder.encode(text, disallowed_special=())
        if len(toks) <= cls.MAX_INPUT_TOKENS:
            return text
        return cls._encoder.decode(toks[: cls.MAX_INPUT_TOKENS])

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        client = self._client_or_raise()
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            batch = [(self._truncate(t) if t.strip() else " ")
                     for t in texts[i : i + self.batch]]
            out.extend(self._embed_batch(client, batch))
        return _l2(np.asarray(out, dtype=np.float64))

    def encode_documents_cached(self, texts: Sequence[str], cache_dir) -> np.ndarray:
        """Encode with a disk cache keyed by (model, dimensions, corpus content).

        Re-embedding a whole corpus to change one retrieval parameter costs money and
        several minutes, which is enough friction to discourage running the comparison at
        all — and an experiment that is too expensive to repeat stops being an experiment.
        """
        import hashlib
        from pathlib import Path

        h = hashlib.sha1()
        h.update(f"{self.model}|{self.dimensions}|{len(texts)}".encode())
        for t in texts:
            h.update(t[:512].encode("utf-8", "ignore"))
        path = Path(cache_dir) / f"emb_{self.name}_{h.hexdigest()[:16]}.npy"
        if path.exists():
            return np.load(path)
        vecs = self.encode_documents(texts)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
        return vecs

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)


class MedCPTBackend:
    """NCBI's MedCPT — a retriever trained on **255 million real PubMed click logs**.

    **Why this is the strongest baseline available for this corpus.** ``text-embedding-3-
    small`` is a general model that has read some biomedical text. MedCPT was trained
    contrastively on *(query, clicked article)* pairs harvested from PubMed itself, which
    makes it a model of what biomedical researchers actually search for and which article
    they then opened. It reports state-of-the-art on six biomedical IR tasks in BEIR,
    beating models far larger than itself.

    The reason to expect it to matter *here specifically*: those click logs are **short
    queries**. Our concept stratum is 2-word MeSH terms and our identifier stratum is bare
    gene symbols — precisely the distribution MedCPT was fitted to, and precisely where a
    long-document-trained general embedder has least advantage.

    **Asymmetric by construction.** Query and article get different encoders, because a
    2-word query and a 900-character passage are not the same kind of object. Using the
    article encoder for queries would silently degrade retrieval, which is why the two are
    kept distinct here rather than sharing one model as the OpenAI backend does.

    ⚠️ **Deployment consequence, stated up front.** MedCPT is 768-dimensional and needs
    torch + transformers (~2 GB). It does not fit a Vercel function bundle, and the
    ``chunks.embedding`` column is ``vector(192)``. So this is measurable offline today and
    servable only via a hosted inference endpoint plus a schema change. That cost is worth
    paying only if it wins by a margin that justifies it — which is the question, not the
    assumption.
    """

    name = "medcpt"
    QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
    ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"

    def __init__(self, batch: int = 32, max_length: int = 256,
                 dimensions: int | None = None) -> None:
        self.batch = batch
        self.max_length = max_length
        #: Truncation to a shorter width, for a like-for-like comparison against a
        #: 192-dim index. MedCPT is not Matryoshka-trained, so truncating it is NOT
        #: expected to be free — measuring that cost is the point of exposing it.
        self.dimensions = dimensions
        self._q = self._a = None

    def _load(self, which: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        name = self.QUERY_MODEL if which == "q" else self.ARTICLE_MODEL
        tok = AutoTokenizer.from_pretrained(name)
        mod = AutoModel.from_pretrained(name)
        mod.eval()
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        return tok, mod

    def fit(self, texts: Sequence[str]) -> None:
        return None

    def _encode(self, texts: Sequence[str], which: str) -> np.ndarray:
        import torch

        if which == "q":
            if self._q is None:
                self._q = self._load("q")
            tok, mod = self._q
        else:
            if self._a is None:
                self._a = self._load("a")
            tok, mod = self._a

        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch):
                batch = [t if t.strip() else " " for t in texts[i : i + self.batch]]
                enc = tok(batch, truncation=True, padding=True,
                          max_length=self.max_length, return_tensors="pt")
                # MedCPT uses the [CLS] representation, not mean pooling. Substituting
                # mean pooling would quietly change what the model was trained to produce.
                vecs = mod(**enc).last_hidden_state[:, 0, :]
                out.append(vecs.cpu().numpy())
        m = np.vstack(out).astype(np.float64)
        if self.dimensions:
            m = m[:, : self.dimensions]
        return _l2(m)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "a")

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "q")

    def encode_documents_cached(self, texts: Sequence[str], cache_dir) -> np.ndarray:
        """Same disk cache as the OpenAI backend — encoding 58k passages on CPU is slow."""
        import hashlib
        from pathlib import Path

        h = hashlib.sha1()
        h.update(f"{self.name}|{self.dimensions}|{len(texts)}".encode())
        for t in texts:
            h.update(t[:512].encode("utf-8", "ignore"))
        path = Path(cache_dir) / f"emb_{self.name}_{h.hexdigest()[:16]}.npy"
        if path.exists():
            return np.load(path)
        vecs = self.encode_documents(texts)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
        return vecs


def make_backend(name: str, *, dim: int = 192) -> EmbeddingBackend:
    """Resolve a backend by name so scripts can switch with a flag rather than an edit."""
    n = (name or "lsa").lower()
    if n == "lsa":
        return LsaBackend(dim=dim)
    if n in ("openai", "openai-small"):
        return OpenAIBackend("text-embedding-3-small", dimensions=dim)
    if n == "openai-large":
        return OpenAIBackend("text-embedding-3-large", dimensions=dim)
    if n == "openai-768":
        # CONTROL for the MedCPT comparison. MedCPT is 768-dim against our 192-dim
        # OpenAI index, so a MedCPT win would confound "trained on biomedical click
        # logs" with "four times the vector capacity". Matching the width isolates the
        # variable that the hypothesis is actually about.
        return OpenAIBackend("text-embedding-3-small", dimensions=768)
    if n == "voyage":
        return VoyageBackend()
    if n == "medcpt":
        # Full 768 dims: MedCPT is not Matryoshka-trained, so truncating it is a real
        # loss rather than a free saving.
        return MedCPTBackend()
    if n == "medcpt-192":
        return MedCPTBackend(dimensions=192)
    raise ValueError(f"unknown embedding backend {name!r}")


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
