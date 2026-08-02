#!/usr/bin/env python
"""Backfill the MeSH major-topic flag into ``documents.meta``.

``to_corpus_doc`` returned MeSH-with-major-flag as a sibling key ``mesh_detail``, but
``upsert_documents`` persists only ``meta`` — so the flag was dropped at write time and
every stored judgment collapsed to "minor". ``descriptors`` kept the names without the
grades, which makes the concept stratum ungraded and removes the property that makes NLM's
indexing worth using: the distinction between a paper being *about* a concept and merely
mentioning it.

Ingestion is fixed for future runs. This repairs the rows already written, without
re-fetching full text or touching ``chunks`` — a documents-only UPDATE over ~1,700 small
rows, which avoids the MVCC blow-up that a chunk rewrite causes.

    python scripts/backfill_mesh_major.py --email you@org.edu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from oncolens.env import load_env  # noqa: E402
from oncolens.sources import pubmed  # noqa: E402

BATCH = 180


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", default="oncolens@example.com")
    ap.add_argument("--api-key", default=os.environ.get("NCBI_API_KEY"))
    args = ap.parse_args()
    load_env()

    import psycopg

    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("POSTGRES_URL / DATABASE_URL not set")

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT doc_id FROM documents WHERE meta->'mesh' IS NULL "
                        "AND cardinality(descriptors) > 0")
            doc_ids = [r[0] for r in cur.fetchall()]
        if not doc_ids:
            print("nothing to backfill — every document already carries meta->mesh")
            return 0
        pmids = [d.replace("PAPER:PMID", "") for d in doc_ids if d.startswith("PAPER:PMID")]
        print(f"{len(pmids)} documents missing the major-topic flag")

        updated = majors = 0
        for i in range(0, len(pmids), BATCH):
            chunk = pmids[i : i + BATCH]
            recs = pubmed.efetch(chunk, email=args.email, api_key=args.api_key)
            rows = []
            for rec in recs:
                if not rec.mesh:
                    continue
                rows.append((json.dumps({"mesh": rec.mesh}), f"PAPER:PMID{rec.pmid}"))
                majors += sum(1 for m in rec.mesh if m.get("major"))
            if rows:
                with conn.cursor() as cur:
                    cur.executemany(
                        "UPDATE documents SET meta = meta || %s::jsonb WHERE doc_id = %s",
                        rows)
                conn.commit()
                updated += len(rows)
            print(f"  {min(i+BATCH, len(pmids))}/{len(pmids)}  ({updated} updated)", end="\r")
        print(f"\n{updated} documents updated; {majors} major-topic assignments recovered")

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM documents WHERE meta->'mesh' IS NOT NULL")
            print(f"documents now carrying meta->mesh: {cur.fetchone()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
