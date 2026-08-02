"""Serving path that queries the live store, rather than a bundled snapshot.

**Why the bundled artifact is no longer sufficient.** `api/search.py` was written against
an offline-built artifact containing every posting list and chunk vector. That is a good
design for a small fixed corpus — zero infrastructure, numpy-only runtime — and it stops
working at scale: the corpus is now 59,306 passages, whose vectors alone are 45 MB before
any text or postings, against Vercel's function bundle limits. Past roughly 10k chunks the
artifact must be replaced by a query against the store.

**The failure this module is built to prevent.** A query vector produced by one embedding
model, compared against document vectors produced by another, does not raise: cosine
distance happily compares two unrelated 192-dimension spaces and returns a confident,
meaningless ranking. This corpus was first embedded with LSA and later re-embedded with
``text-embedding-3-small`` at the *same* dimensionality, so nothing about the column shape
reveals a mismatch. Every query therefore checks ``index_config`` first and refuses to
answer if the encoders disagree.

**Which configuration this serves, and why.** Measured on 2,225 citation-context queries,
paired permutation test, Bonferroni-corrected within the iteration:

    hybrid-openai   nDCG@10 0.4526   +0.0878 vs shipping   CI [+0.0762, +0.0995]
    openai          nDCG@10 0.4116   +0.0469
    bm25            nDCG@10 0.3888   +0.0241   <- beats the shipping hybrid ALONE
    hybrid-lsa      nDCG@10 0.3647   (what shipped)
    lsa             nDCG@10 0.3088   -0.0560

The LSA dense arm was measurably *harmful*: BM25 on its own outranked the hybrid that
included it, 466 wins to 432 with p < 0.0001. So the default here is lexical + OpenAI,
and ``dense_weight=0`` degrades cleanly to the BM25-only arm that still beats the old
default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import neon_store

DEFAULT_BACKEND = os.environ.get("ONCOLENS_EMBED_BACKEND", "openai")
DEFAULT_DIM = int(os.environ.get("ONCOLENS_EMBED_DIM", "192"))


@dataclass
class LiveIndex:
    """Holds the connection and query encoder for the lifetime of a warm container.

    Both are expensive to create and safe to reuse: a serverless container handles many
    requests, and reconnecting per request would dominate latency on a cold pool.
    """

    dsn: str
    backend_name: str = DEFAULT_BACKEND
    dim: int = DEFAULT_DIM
    _conn: object | None = field(default=None, repr=False)
    _backend: object | None = field(default=None, repr=False)
    _checked: bool = field(default=False, repr=False)

    def conn(self):
        import psycopg

        if self._conn is None or getattr(self._conn, "closed", 1):
            # Neon's pooler endpoint is what makes this viable from serverless; a direct
            # endpoint exhausts connection slots as concurrency rises.
            self._conn = psycopg.connect(self.dsn, connect_timeout=10, autocommit=True)
            self._checked = False
        if not self._checked:
            neon_store.assert_embedding_matches(self._conn, self.backend_name, self.dim)
            self._checked = True
        return self._conn

    def encoder(self):
        if self._backend is None:
            from ..retrieval.dense import make_backend

            self._backend = make_backend(self.backend_name, dim=self.dim)
        return self._backend

    def encode_query(self, query: str) -> list[float]:
        return [float(x) for x in self.encoder().encode_queries([query])[0]]

    def search(self, query: str, *, top_k: int = 10, candidates: int = 200,
               bm25_weight: float = 1.0, dense_weight: float = 1.0) -> dict:
        conn = self.conn()
        vec = self.encode_query(query) if dense_weight > 0 else [0.0] * self.dim
        rows = neon_store.hybrid_search(
            conn, query, vec, candidates=candidates, top_k=top_k,
            bm25_weight=bm25_weight, dense_weight=dense_weight,
        )
        return {
            "query": query,
            "backend": self.backend_name,
            "source": "neon",
            "results": [_shape(r, query) for r in rows],
        }


def _shape(row: dict, query: str) -> dict:
    """Attach clause offsets so the UI can highlight the matched span.

    The offsets are section-relative and computed here rather than stored, because they
    depend on the query. ``start_char``/``end_char`` on the row locate the passage inside
    its section; the clause offsets locate the match inside the passage.
    """
    from ..spans import find_clauses

    # neon_store.hybrid_search already nests the passage fields; reading them from the top
    # level yields None for every offset, which silently removes the provenance that is the
    # entire point of the product rather than raising.
    p = row.get("passage") or {}
    text = p.get("text") or row.get("text") or ""
    try:
        clauses = [
            {"start": c.start, "end": c.end, "score": round(float(c.score), 4),
             "text": text[c.start : c.end]}
            for c in find_clauses(text, query)[:3]
        ]
    except Exception:  # noqa: BLE001 — highlighting must never break a result
        clauses = []
    meta = row.get("meta") or {}
    return {
        "doc_id": row.get("doc_id"),
        "title": row.get("title"),
        "year": row.get("year"),
        "score": round(float(row.get("score") or 0.0), 6),
        "passage": {
            "chunk_id": p.get("chunk_id"),
            "section": p.get("section"),
            "start_char": p.get("start_char"),
            "end_char": p.get("end_char"),
            "text": text,
            "clauses": clauses,
        },
        "source": {
            "pmid": (row.get("doc_id") or "").replace("PAPER:PMID", "") or None,
            "pmcid": meta.get("pmcid"),
            "license": meta.get("license_code"),
            "blob_url": meta.get("blob_url"),
        },
    }


#: A cell must clear this to count as "reports this dimension". Below it the honest answer
#: is NOT REPORTED — a blank cell reads as *no effect* when it means *not measured*, and
#: that misreading is worse than an empty table.
CELL_MIN_SCORE = 0.08


def _cues_to_tsquery(cues: tuple[str, ...]) -> str:
    """Build a valid ``to_tsquery`` string from human-written cue phrases.

    The cue lists are written for readability, so they contain things Postgres rejects:
    trailing spaces (``"hr "``), bare operators (``"p<"``, ``"n ="``), and punctuation
    (``"95% ci"``). Naively substituting ``<->`` for spaces produced
    ``"hr <-> | odds <-> ratio"`` and a syntax error. Cues are therefore reduced to
    alphanumeric tokens, multi-word phrases become adjacency groups, and anything left
    empty is dropped rather than emitted as a dangling operator.
    """
    import re as _re

    parts: list[str] = []
    for cue in cues:
        tokens = [t for t in _re.split(r"[^A-Za-z0-9]+", cue.lower()) if t]
        if not tokens:
            continue
        phrase = " <-> ".join(tokens)
        parts.append(f"({phrase})" if len(tokens) > 1 else phrase)
    # Deduplicate while preserving order; repeated terms only inflate the parse.
    seen: set[str] = set()
    uniq = [p for p in parts if not (p in seen or seen.add(p))]
    return " | ".join(uniq)


def compare(index: LiveIndex, query: str, *, n_papers: int = 5,
            aspect_keys: tuple[str, ...] | None = None) -> dict:
    """Papers x technical dimensions, every cell carrying its own citation.

    **Why this is not just top-k.** Asking "how do these studies measure X" with a plain
    top-k returns passages clustered inside the single most on-topic paper, several of
    which state no method at all. So the query selects *documents* first, then each cell is
    filled by searching **within that document** for a passage that actually reports the
    dimension — full-text ranked against the aspect's cue vocabulary, and for numeric
    aspects (cohort size, effect size) required to contain a digit, because a cohort
    sentence without a number is not a cohort answer.

    Runs entirely in SQL against the live store. The previous implementation imported the
    offline ``Retriever``, which needs a fitted in-process index and scipy — neither
    present in a serverless function, so it returned 500 in production.
    """
    from ..aspects import ASPECTS_BY_KEY, DEFAULT_ASPECT_KEYS

    keys = tuple(aspect_keys or DEFAULT_ASPECT_KEYS)
    aspects = [ASPECTS_BY_KEY[k] for k in keys if k in ASPECTS_BY_KEY]
    if not aspects:
        aspects = [ASPECTS_BY_KEY[k] for k in DEFAULT_ASPECT_KEYS]

    top = index.search(query, top_k=n_papers)
    docs = [r["doc_id"] for r in top["results"]]
    titles = {r["doc_id"]: r["title"] for r in top["results"]}
    years = {r["doc_id"]: r.get("year") for r in top["results"]}
    if not docs:
        return {"query": query, "aspects": list(keys), "doc_ids": [], "coverage": 0.0,
                "cells": {}, "titles": {}, "years": {},
                "notes": ["no documents matched the query"]}

    conn = index.conn()
    cells: dict[str, dict] = {}
    filled = 0
    with conn.cursor() as cur:
        for asp in aspects:
            tsquery = _cues_to_tsquery(asp.cues)
            if not tsquery:
                continue
            cur.execute(
                """
                SELECT DISTINCT ON (c.doc_id)
                       c.doc_id, c.chunk_id, c.section, c.start_char, c.end_char, c.text,
                       ts_rank_cd(c.tsv, to_tsquery('english', %(tsq)s)) AS rank
                FROM chunks c
                WHERE c.doc_id = ANY(%(docs)s)
                  AND c.tsv @@ to_tsquery('english', %(tsq)s)
                  AND (NOT %(numeric)s OR c.text ~ '[0-9]')
                ORDER BY c.doc_id, rank DESC
                """,
                {"tsq": tsquery, "docs": docs, "numeric": asp.numeric},
            )
            found = {r[0]: r for r in cur.fetchall()}
            for doc in docs:
                row = found.get(doc)
                key = f"{doc}|{asp.key}"
                if row is None or float(row[6]) < CELL_MIN_SCORE:
                    cells[key] = {"reported": False, "text": None}
                    continue
                filled += 1
                cells[key] = {
                    "reported": True,
                    "chunk_id": row[1],
                    "section": row[2],
                    "start_char": row[3],
                    "end_char": row[4],
                    "text": (row[5] or "")[:700],
                    "score": round(float(row[6]), 4),
                }

    total = len(docs) * len(aspects)
    return {
        "query": query,
        "aspects": [{"key": a.key, "label": a.label, "numeric": a.numeric} for a in aspects],
        "doc_ids": docs,
        "titles": titles,
        "years": years,
        "coverage": round(filled / total, 4) if total else 0.0,
        "cells": cells,
        "source": "neon",
        "notes": [
            "A cell marked NOT REPORTED means the paper does not report that dimension — "
            "not that the effect was absent.",
        ],
    }


_LIVE: LiveIndex | None = None


def get_live_index() -> LiveIndex | None:
    """Return a warm live index, or None when no database is configured.

    Returning None rather than raising lets the caller fall back to the bundled artifact,
    which is the right behaviour for a preview deployment that has no store attached.
    """
    global _LIVE
    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        return None
    if _LIVE is None:
        _LIVE = LiveIndex(dsn=dsn)
    return _LIVE
