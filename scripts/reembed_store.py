#!/usr/bin/env python
"""Re-embed the stored passages with a named backend, and record which one was used.

**Why this is a separate script from ingestion.** Choosing an embedding backend is a
retrieval decision, not an ingestion decision. Measured on 2,225 citation-context queries,
swapping the dense arm changes nDCG@10 by more than any other single change made to this
system — so it has to be switchable without re-fetching 1,739 articles from PMC, and the
switch has to be recorded so serving cannot silently disagree with the index.

    python scripts/reembed_store.py --backend openai --dim 192
    python scripts/reembed_store.py --backend openai --dry-run   # show the plan only

The document vectors are read from the on-disk cache when the corpus text is unchanged, so
re-running after a serving change costs nothing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

from oncolens.env import load_env, local_data_dir  # noqa: E402
from oncolens.retrieval.dense import make_backend  # noqa: E402
from oncolens.serve import neon_store  # noqa: E402

BATCH = 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="openai",
                    choices=["openai", "openai-large", "lsa", "voyage"])
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_env()
    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("POSTGRES_URL / DATABASE_URL not set")

    import psycopg

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_id, COALESCE(indexed_text, text) FROM chunks "
                        "ORDER BY chunk_id")
            rows = cur.fetchall()
        if not rows:
            raise SystemExit("no chunks in the store")
        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        print(f"{len(ids):,} passages in the store")

        current = neon_store.get_index_config(conn)
        print(f"index currently records: {current or '(nothing — pre-dates index_config)'}")
        print(f"target: backend={args.backend} dim={args.dim}")

        # Confirm the column can hold this width before spending money on encoding.
        with conn.cursor() as cur:
            cur.execute("SELECT atttypmod FROM pg_attribute "
                        "WHERE attrelid='chunks'::regclass AND attname='embedding'")
            r = cur.fetchone()
        col_dim = r[0] if r and r[0] and r[0] > 0 else None
        if col_dim and col_dim != args.dim:
            print(f"\nERROR: chunks.embedding is vector({col_dim}) but --dim is {args.dim}.")
            print("Changing width needs an ALTER plus an index rebuild; do that "
                  "deliberately rather than as a side effect of re-embedding.")
            return 1

        if args.dry_run:
            print("\n--dry-run: nothing written.")
            return 0

        backend = make_backend(args.backend, dim=args.dim)
        backend.fit(texts)
        print(f"encoding with {args.backend} (cache: {local_data_dir() / 'emb_cache'})...")
        if hasattr(backend, "encode_documents_cached"):
            vecs = backend.encode_documents_cached(texts, local_data_dir() / "emb_cache")
        else:
            vecs = backend.encode_documents(texts)
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.shape != (len(ids), args.dim):
            raise SystemExit(f"encoder returned {vecs.shape}, expected {(len(ids), args.dim)}")
        print(f"  {vecs.shape[0]:,} vectors, dim {vecs.shape[1]}")

        # Write in batches: a single 59k-row UPDATE builds one enormous transaction and
        # can exceed a serverless Postgres statement timeout.
        written = 0
        for i in range(0, len(ids), BATCH):
            chunk_ids = ids[i : i + BATCH]
            payload = [
                (cid, "[" + ",".join(f"{float(x):.6f}" for x in vecs[i + j]) + "]")
                for j, cid in enumerate(chunk_ids)
            ]
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE chunks SET embedding = %s::vector WHERE chunk_id = %s",
                    [(v, cid) for cid, v in payload],
                )
            conn.commit()
            written += len(chunk_ids)
            print(f"  {written:,}/{len(ids):,}", end="\r")
        print(f"  {written:,}/{len(ids):,} written")

        neon_store.set_index_config(
            conn,
            **{neon_store.CFG_EMBED_MODEL: args.backend,
               neon_store.CFG_EMBED_DIM: str(args.dim)},
        )
        print(f"\nrecorded index_config: {neon_store.get_index_config(conn)}")
        print("Serving now refuses to answer with a different query encoder "
              "(neon_store.assert_embedding_matches).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
