"""Postgres + pgvector store for Vercel deployment (Neon).

**Why Postgres and not a dedicated vector DB.** This corpus is relational: grants link to
publications, publications to PIs and institutions, concepts to documents. Those joins are
half the product ("what has my institution funded on X", "papers from grants studying Y").
A pure vector store makes you reimplement them in application code. pgvector is fine to
roughly 50M vectors, which is far past this corpus.

**Vercel specifics.** Vercel Postgres is sunset; storage now comes from Marketplace
providers. Neon fits best: pgvector on every plan, and database branching that mirrors
Vercel preview deployments so a preview never queries production data. Connect with the
Neon serverless driver over HTTP when running on Edge, where raw TCP is unavailable.

---

## The honest caveat: Postgres full-text search is not BM25

`ts_rank_cd` is a different scoring function. If the offline harness measures BM25 and
production serves `ts_rank_cd`, **the evaluation stops describing the product** — which is
the exact failure this whole project is built to avoid. Three options, in order of
preference:

1. **Install `pg_search` (ParadeDB)** if the provider supports it — real BM25 in Postgres.
2. **Score BM25 in SQL from stored statistics.** `chunk_terms` below stores per-term
   frequencies and the corpus stats needed, so BM25 can be computed exactly as the offline
   index computes it. Heavier, but faithful.
3. **Re-run the evaluation against this backend** and accept `ts_rank_cd` as the lexical
   arm. Legitimate — but then the reported numbers must come from *this* path, not the
   in-process one.

Option 3 is the fallback, and `SearchBackend` exists so the harness can run against
whichever backend is actually deployed. Never mix: measuring one and shipping the other is
how a retrieval system silently regresses in production while the dashboard stays green.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    doc_type    TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    year        INT,
    meta        JSONB NOT NULL DEFAULT '{}'::jsonb,
    descriptors TEXT[] NOT NULL DEFAULT '{}',
    funded_by   TEXT[] NOT NULL DEFAULT '{}'
);

-- One row per retrievable passage. start_char/end_char are the product requirement:
-- they let the UI highlight the exact span inside the source document.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    section     TEXT NOT NULL DEFAULT '',
    ordinal     INT  NOT NULL DEFAULT 0,
    start_char  INT  NOT NULL,
    end_char    INT  NOT NULL,
    text        TEXT NOT NULL,
    -- indexed_text may include the contextual-retrieval prefix; text stays verbatim so
    -- the passage shown to a user is always the real source text.
    indexed_text TEXT NOT NULL,
    tsv         TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', indexed_text)) STORED,
    embedding   VECTOR(%(dim)s)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx   ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc_idx   ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_trgm_idx  ON chunks USING GIN (indexed_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS documents_desc_idx ON documents USING GIN (descriptors);

-- HNSW beats IVFFlat for recall at this scale and needs no training step.
-- vector_cosine_ops because embeddings are L2-normalised.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
"""

#: Optional: exact BM25 in SQL (option 2 above). Populated only when faithful BM25 is
#: required; it roughly triples ingestion size, which is why it is opt-in.
BM25_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chunk_terms (
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    term     TEXT NOT NULL,
    tf       INT  NOT NULL,
    PRIMARY KEY (chunk_id, term)
);
CREATE INDEX IF NOT EXISTS chunk_terms_term_idx ON chunk_terms (term);

CREATE TABLE IF NOT EXISTS corpus_stats (
    term TEXT PRIMARY KEY,
    df   INT NOT NULL
);

CREATE TABLE IF NOT EXISTS corpus_meta (
    k TEXT PRIMARY KEY,
    v DOUBLE PRECISION NOT NULL
);

-- Which embedding backend produced chunks.embedding.
--
-- **Why this table exists.** A query vector from one model compared against document
-- vectors from another returns a ranking: it is not an error, it is nonsense that looks
-- like results. Cosine distance is perfectly happy to compare two unrelated 192-dim
-- spaces. Since the corpus was first embedded with LSA and later re-embedded with
-- text-embedding-3-small at the same dimensionality, nothing about the column's *shape*
-- would reveal the mismatch. Serving therefore reads this and refuses to answer if the
-- query encoder disagrees.
CREATE TABLE IF NOT EXISTS index_config (
    k         TEXT PRIMARY KEY,
    v         TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

#: Keys stored in ``index_config``.
CFG_EMBED_MODEL = "embedding_model"
CFG_EMBED_DIM = "embedding_dim"


def set_index_config(conn, **kv: str) -> None:
    with conn.cursor() as cur:
        for k, v in kv.items():
            cur.execute(
                "INSERT INTO index_config (k, v, updated_at) VALUES (%s, %s, now()) "
                "ON CONFLICT (k) DO UPDATE SET v = EXCLUDED.v, updated_at = now()",
                (k, str(v)),
            )
    conn.commit()


def get_index_config(conn) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.index_config')")
        if cur.fetchone()[0] is None:
            return {}
        cur.execute("SELECT k, v FROM index_config")
        return {k: v for k, v in cur.fetchall()}


class EmbeddingMismatch(RuntimeError):
    """Raised when the query encoder does not match the one that built the index."""


#: What the corpus was embedded with before ``index_config`` existed. Any index missing
#: the config predates it, and therefore holds LSA vectors.
LEGACY_BACKEND = "lsa"


def assert_embedding_matches(conn, backend_name: str, dim: int) -> None:
    """Fail loudly rather than return a plausible-looking wrong ranking.

    **Absent config is not treated as permission.** An index with no recorded backend
    predates this table, which means it holds LSA vectors — so querying it with any other
    encoder is precisely the silent-nonsense case, and it is the *likely* case, because the
    reason a newer encoder is configured is that someone chose it. Tolerating the unset
    state would disable the guard exactly when it is needed.
    """
    cfg = get_index_config(conn)
    stored = cfg.get(CFG_EMBED_MODEL)
    if stored is None:
        if backend_name == LEGACY_BACKEND:
            return          # consistent with what a pre-config index actually contains
        raise EmbeddingMismatch(
            f"the index records no embedding backend, so it predates index_config and "
            f"holds {LEGACY_BACKEND!r} vectors — but the query encoder is {backend_name!r}. "
            f"Comparing vectors across embedding spaces returns a confident, meaningless "
            f"ranking rather than an error. Run "
            f"`python scripts/reembed_store.py --backend {backend_name} --dim {dim}`, or "
            f"set ONCOLENS_EMBED_BACKEND={LEGACY_BACKEND!r} to keep serving the old index."
        )
    stored_dim = cfg.get(CFG_EMBED_DIM)
    if stored != backend_name or (stored_dim and int(stored_dim) != dim):
        raise EmbeddingMismatch(
            f"index was built with {stored!r} (dim {stored_dim}) but the query encoder is "
            f"{backend_name!r} (dim {dim}). Comparing vectors across embedding spaces "
            f"returns a ranking that is nonsense rather than an error. Re-run "
            f"scripts/reembed_store.py, or set ONCOLENS_EMBED_BACKEND to {stored!r}."
        )

# --------------------------------------------------------------------------
# Hybrid search: RRF over a lexical arm and a dense arm, fused in SQL
# --------------------------------------------------------------------------

#: Figure rows (``kind='figure'``) are stored so a paper can be READ with its images, and
#: are excluded from both retrieval arms until their effect is measured. The exclusion is
#: explicit in the SQL rather than implied by a NULL embedding, because a NULL embedding
#: sorts unpredictably in an ORDER BY and would let figures leak into the dense arm silently
#: — the corpus would change without any candidate having been run (docs/PLAN_FIGURES.md
#: §3.9, Path A). Flip this to include them only behind a measured, pre-registered change.
_RETRIEVABLE = "c.kind = 'passage'"

HYBRID_SEARCH_SQL = f"""
WITH lexical AS (
    SELECT c.chunk_id, c.doc_id,
           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.tsv, plainto_tsquery('english', $1)) DESC) AS rank
    FROM chunks c
    WHERE {_RETRIEVABLE} AND c.tsv @@ plainto_tsquery('english', $1)
    LIMIT $3
),
dense AS (
    SELECT c.chunk_id, c.doc_id,
           ROW_NUMBER() OVER (ORDER BY c.embedding <=> $2::vector) AS rank
    FROM chunks c
    WHERE {_RETRIEVABLE} AND c.embedding IS NOT NULL
    ORDER BY c.embedding <=> $2::vector
    LIMIT $3
),
fused AS (
    -- Reciprocal Rank Fusion. Operates on ranks, so the unbounded ts_rank_cd scale and
    -- the [0,2] cosine-distance scale never have to be reconciled.
    --
    -- WHICH ARM FOUND IT IS CARRIED THROUGH, and it is not a diagnostic nicety. The
    -- `lexical` CTE is gated on `tsv @@ plainto_tsquery`, so it returns nothing when the
    -- query's words are absent from the corpus. The `dense` CTE has no WHERE and no
    -- distance threshold: it ALWAYS returns its 200 nearest neighbours. So a misspelling
    -- ("osimertnib") or a term this corpus has never seen still produces ten scored,
    -- titled, offset-bearing results that look exactly like good ones, and a researcher
    -- concludes the literature contains work it does not. Surfacing lex_rank lets the UI
    -- say "no passage contains your terms; these are semantic neighbours" instead.
    SELECT chunk_id, doc_id, SUM(w) AS score,
           MIN(lex_rank)   AS lex_rank,
           MIN(dense_rank) AS dense_rank
    FROM (
        SELECT chunk_id, doc_id, $5::float / ($4 + rank) AS w,
               rank AS lex_rank, NULL::bigint AS dense_rank FROM lexical
        UNION ALL
        SELECT chunk_id, doc_id, $6::float / ($4 + rank) AS w,
               NULL::bigint AS lex_rank, rank AS dense_rank FROM dense
    ) u
    GROUP BY chunk_id, doc_id
),
best_per_doc AS (
    -- Collapse passages to documents, keeping the single best passage as the citation.
    SELECT DISTINCT ON (doc_id) doc_id, chunk_id, score, lex_rank, dense_rank
    FROM fused
    ORDER BY doc_id, score DESC
)
SELECT b.doc_id, b.score, b.lex_rank, b.dense_rank,
       d.title, d.doc_type, d.year, d.meta,
       c.chunk_id, c.section, c.start_char, c.end_char, c.text
FROM best_per_doc b
JOIN documents d ON d.doc_id = b.doc_id
JOIN chunks    c ON c.chunk_id = b.chunk_id
ORDER BY b.score DESC
LIMIT $7;
"""

#: Concept/faceted search — the "search a concept, get associated grants and papers" path.
#: Uses the GIN index on descriptors, so it is a filter rather than a ranking problem.
CONCEPT_SEARCH_SQL = """
SELECT d.doc_id, d.title, d.doc_type, d.year, d.meta
FROM documents d
WHERE d.descriptors && $1::text[]
ORDER BY d.year DESC NULLS LAST
LIMIT $2;
"""


@dataclass
class NeonConfig:
    dsn: str
    dim: int = 192

    @classmethod
    def from_env(cls, dim: int = 192) -> "NeonConfig":
        dsn = (
            os.environ.get("DATABASE_URL")
            or os.environ.get("POSTGRES_URL")
            or os.environ.get("NEON_DATABASE_URL")
        )
        if not dsn:
            raise RuntimeError(
                "No Postgres DSN. Set DATABASE_URL / POSTGRES_URL (the Neon Vercel "
                "integration injects POSTGRES_URL automatically)."
            )
        return cls(dsn=dsn, dim=dim)


def init_schema(conn, dim: int = 192, *, with_bm25: bool = False) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL % {"dim": dim})
        if with_bm25:
            cur.execute(BM25_SCHEMA_SQL)
    conn.commit()


def upsert_documents(conn, docs: Iterable[dict]) -> int:
    rows = [
        (d["doc_id"], d.get("doc_type", "paper"), d.get("title", ""), d.get("year"),
         json.dumps(d.get("meta", {})), d.get("descriptors", []), d.get("funded_by", []))
        for d in docs
    ]
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO documents (doc_id, doc_type, title, year, meta, descriptors, funded_by)
               VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s)
               ON CONFLICT (doc_id) DO UPDATE SET
                 doc_type=EXCLUDED.doc_type, title=EXCLUDED.title, year=EXCLUDED.year,
                 meta=EXCLUDED.meta, descriptors=EXCLUDED.descriptors,
                 funded_by=EXCLUDED.funded_by""",
            rows,
        )
    conn.commit()
    return len(rows)


def replace_document_chunks(conn, doc_ids: Iterable[str]) -> int:
    """Delete every existing chunk for these documents.

    **Why this exists.** ``upsert_chunks`` keys on ``chunk_id``, so re-ingesting a document
    that now yields *fewer* chunks leaves the surplus rows behind. Reference stripping does
    exactly that — it removes ~20% of each article — so after the stripper shipped, the
    index still contained the bibliography passages from the previous run. The retrieval
    numbers then described a corpus that no longer matched the code, which is the specific
    failure this project is built to catch.

    Callers must run this *inside the same transaction* as the insert, so a crash between
    the two cannot leave a document with no passages at all.
    """
    ids = list(doc_ids)
    if not ids:
        return 0
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE doc_id = ANY(%s)", (ids,))
        return cur.rowcount or 0


def upsert_chunks(conn, chunks: Sequence[dict], embeddings: Sequence[Sequence[float]],
                  *, replace: bool = True) -> int:
    """``chunks`` carries chunk_id/doc_id/section/ordinal/offsets/text/indexed_text.

    ``replace=True`` (the default) first clears every existing chunk for the documents
    being written, so the stored passages are exactly what the current chunker produces.
    Pass ``replace=False`` only when appending to a document already written in this run.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"chunk/embedding count mismatch: {len(chunks)} vs {len(embeddings)}")
    if replace and chunks:
        replace_document_chunks(conn, {c["doc_id"] for c in chunks})
    rows = [
        (c["chunk_id"], c["doc_id"], c.get("section", ""), c.get("ordinal", 0),
         c["start_char"], c["end_char"], c["text"], c.get("indexed_text", c["text"]),
         list(map(float, emb)))
        for c, emb in zip(chunks, embeddings)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO chunks (chunk_id, doc_id, section, ordinal, start_char, end_char,
                                   text, indexed_text, embedding)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
               ON CONFLICT (chunk_id) DO UPDATE SET
                 section=EXCLUDED.section, ordinal=EXCLUDED.ordinal,
                 start_char=EXCLUDED.start_char, end_char=EXCLUDED.end_char,
                 text=EXCLUDED.text, indexed_text=EXCLUDED.indexed_text,
                 embedding=EXCLUDED.embedding""",
            rows,
        )
    conn.commit()
    return len(rows)


def hybrid_search(
    conn, query: str, query_embedding: Sequence[float], *,
    candidates: int = 200, rrf_k: int = 60,
    bm25_weight: float = 1.0, dense_weight: float = 1.0, top_k: int = 10,
) -> list[dict]:
    """Run the fused query and return documents with their supporting passage."""
    vec = "[" + ",".join(f"{float(x):.6f}" for x in query_embedding) + "]"
    sql = (HYBRID_SEARCH_SQL
           .replace("$1", "%(q)s").replace("$2", "%(vec)s").replace("$3", "%(cand)s")
           .replace("$4", "%(k)s").replace("$5", "%(bw)s").replace("$6", "%(dw)s")
           .replace("$7", "%(topk)s"))
    with conn.cursor() as cur:
        cur.execute(sql, {"q": query, "vec": vec, "cand": candidates, "k": rrf_k,
                          "bw": bm25_weight, "dw": dense_weight, "topk": top_k})
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        out.append({
            "doc_id": d["doc_id"], "score": float(d["score"]), "title": d["title"],
            "doc_type": d["doc_type"], "year": d["year"], "meta": d["meta"],
            # None means that arm did not retrieve this passage at all. lex_rank is None
            # exactly when the query's words appear nowhere in it.
            "lex_rank": d.get("lex_rank"), "dense_rank": d.get("dense_rank"),
            "passage": {
                "chunk_id": d["chunk_id"], "section": d["section"],
                "start_char": d["start_char"], "end_char": d["end_char"], "text": d["text"],
            },
        })
    return out
