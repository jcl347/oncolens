"""Build a portable, precomputed search index for serverless deployment.

**Why this exists.** Vercel functions are ephemeral, have a read-only filesystem outside
``/tmp``, a bundle-size ceiling, and short execution limits. Fitting a TF-IDF/SVD index at
request time would blow all four. So all fitting happens offline here, and the function
does only scoring.

**Dependency split, which is the thing that makes this deployable:**

* build time (this module) — scipy for the truncated SVD, numpy, the full package
* run time (``api/search.py``) — **numpy only**

scipy is the heavy wheel; keeping it out of the function bundle is what keeps cold starts
and bundle size sane. The artifact stores the already-fitted LSA components, so the
function only needs a sparse dot product and a matmul.

**Scaling boundary, stated honestly.** A bundled artifact is the right answer for a corpus
of this size (hundreds to low tens of thousands of chunks). It is the *wrong* answer at
real NIH/PMC scale — millions of chunks will not fit in a function bundle and should live
in Postgres + pgvector, with the same BM25/dense/RRF logic expressed in SQL. The retrieval
semantics are identical; only the storage moves. See docs/DEPLOYMENT.md.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ..contexts import build_contexts
from ..retrieval.chunking import chunk_corpus
from ..retrieval.dense import LsaBackend
from ..retrieval.pipeline import RetrievalConfig
from ..retrieval.text import tokenize

ARTIFACT_VERSION = 2


def build_artifact(
    docs: Sequence[Mapping],
    config: RetrievalConfig,
    *,
    out_dir: Path,
    context_mode: str = "gist",
    lexicon: Mapping[str, Sequence[str]] | None = None,
) -> dict:
    """Fit every index offline and write a deployable artifact.

    Writes:
      ``index.npz``   dense matrices (float32) — LSA components + chunk vectors + idf
      ``index.json``  postings, vocab, chunk metadata, passages, config, lexicon
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = chunk_corpus(list(docs))
    contexts = build_contexts(docs, chunks, mode=context_mode)
    texts = [
        c.indexable_text(include_heading=config.include_heading, context=contexts.get(c.chunk_id))
        for c in chunks
    ]

    # ---- lexical: postings + idf, stored as plain lists so the runtime needs no scipy ----
    postings: dict[str, list[list[int]]] = {}
    doc_len: list[int] = []
    df: dict[str, int] = {}
    for i, text in enumerate(texts):
        toks = tokenize(text)
        doc_len.append(len(toks))
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        for term, freq in tf.items():
            postings.setdefault(term, []).append([i, freq])
            df[term] = df.get(term, 0) + 1

    n = len(texts)
    avgdl = (sum(doc_len) / n) if n else 0.0

    # ---- dense: fit LSA here (scipy), ship only the fitted matrices ----
    backend = LsaBackend(dim=config.dense_dim)
    backend.fit(texts)
    chunk_vectors = backend.encode_documents(texts).astype(np.float32)
    components = np.asarray(backend.components, dtype=np.float32)
    idf_vec = np.asarray(backend.idf, dtype=np.float32)

    np.savez_compressed(
        out_dir / "index.npz",
        components=components,          # (dim, vocab) LSA projection
        chunk_vectors=chunk_vectors,    # (n_chunks, dim) L2-normalised
        tfidf_idf=idf_vec,              # (vocab,) idf used by the tf-idf stage
    )

    by_doc = {d["doc_id"]: d for d in docs}
    meta = {
        "artifact_version": ARTIFACT_VERSION,
        "config": config.as_dict(),
        "n_chunks": n,
        "avgdl": avgdl,
        "doc_len": doc_len,
        "bm25_df": df,
        "postings": postings,
        "tfidf_vocab": backend.vocab,   # term -> column, for projecting a query
        "chunks": [
            {
                "chunk_id": c.chunk_id, "doc_id": c.doc_id, "section": c.section,
                "start_char": c.start_char, "end_char": c.end_char,
                "text": c.text,          # the verbatim passage shown to the user
            }
            for c in chunks
        ],
        "documents": {
            d["doc_id"]: {
                "title": d.get("title", ""), "doc_type": d.get("doc_type", ""),
                "year": d.get("year"), "meta": d.get("meta", {}),
            }
            for d in by_doc.values()
        },
        "lexicon": {k: list(v) for k, v in (lexicon or {}).items()},
    }
    (out_dir / "index.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    sizes = {
        "index.npz_bytes": (out_dir / "index.npz").stat().st_size,
        "index.json_bytes": (out_dir / "index.json").stat().st_size,
    }
    return {
        "n_chunks": n,
        "n_documents": len(by_doc),
        "vocab_size": len(backend.vocab),
        "dense_dim": int(components.shape[0]),
        **sizes,
        "total_mb": round(sum(sizes.values()) / 1_048_576, 2),
    }


def idf(df_map: Mapping[str, int], term: str, n_docs: int) -> float:
    """Same Robertson/Sparck-Jones IDF the offline BM25 uses.

    Duplicated deliberately: the runtime must not import the full package (and its scipy
    dependency) just to score. Any change here must be mirrored in
    ``retrieval/lexical.py``, and ``tests/test_serve_parity.py`` fails if the two ever
    diverge — a serving path that silently scores differently from the evaluated path
    would make every offline measurement a lie.
    """
    c = df_map.get(term, 0)
    if c == 0:
        return 0.0
    return math.log(1.0 + (n_docs - c + 0.5) / (c + 0.5))
