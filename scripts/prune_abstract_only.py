#!/usr/bin/env python
"""Remove documents that carry no verbatim full text from the store.

**Why prune rather than keep.** An abstract-only record contributes one or two passages of
publisher summary. That is not what this product returns: the promise is *the passage where
a concept was mentioned*, and an abstract has no passages to locate a concept **in** — the
whole record is the summary. Worse, those documents distort every corpus-level measurement
they take part in: they inflate the document count, drag the mean chunks-per-document from
~34 down toward 1, and sit in the cluster map at positions derived from a single vector.

**Run the backfill first.** ``scripts/ingest_real.py --pmids-file`` re-runs the fixed PMC
fetch path, and on the live corpus that converted 37 "abstract-only" documents into real
full text. Pruning before backfilling throws those away.

    python scripts/prune_abstract_only.py --dry-run     # always do this first
    python scripts/prune_abstract_only.py --apply

A manifest of every deleted PMID is written before the delete, so the set can be
re-ingested. Deletion is by ``meta.full_text_chars``, which is written only when a PMC
object was actually retrieved and parsed — a fact about the fetch, not a heuristic on
chunk counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from oncolens.env import load_env, local_data_dir  # noqa: E402

#: A document must have this many characters of retrieved full text to stay. The floor is
#: deliberately low: the question is "did we get the article body at all", not "is it long".
MIN_FULL_TEXT_CHARS = 2000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="perform the delete")
    ap.add_argument("--dry-run", action="store_true", help="report only (default)")
    ap.add_argument("--min-chars", type=int, default=MIN_FULL_TEXT_CHARS)
    args = ap.parse_args()
    load_env()

    import os

    import psycopg

    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("POSTGRES_URL / DATABASE_URL not set")

    keep_pred = ("(meta ? 'full_text_chars') AND "
                 "COALESCE((meta->>'full_text_chars')::int, 0) >= %(min)s")
    drop_pred = f"NOT ({keep_pred})"

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents")
        total = cur.fetchone()[0]
        cur.execute(f"SELECT count(*) FROM documents WHERE {keep_pred}", {"min": args.min_chars})
        keep = cur.fetchone()[0]
        cur.execute(f"SELECT count(*) FROM documents WHERE {drop_pred}", {"min": args.min_chars})
        drop = cur.fetchone()[0]
        cur.execute(
            f"""SELECT count(*) FROM chunks c
                WHERE c.kind = 'passage' AND EXISTS (SELECT 1 FROM documents d
                              WHERE d.doc_id = c.doc_id AND {drop_pred})""",
            {"min": args.min_chars})
        drop_chunks = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM chunks WHERE kind = 'passage'")
        total_chunks = cur.fetchone()[0]

        print(f"documents      {total:>8,}   keep {keep:>7,}   drop {drop:>7,}")
        print(f"passages       {total_chunks:>8,}   keep {total_chunks-drop_chunks:>7,}   "
              f"drop {drop_chunks:>7,}")
        print(f"\nkeeping documents with >= {args.min_chars:,} chars of retrieved full text")

        if not args.apply:
            print("\n--dry-run (default): nothing deleted. Re-run with --apply.")
            return 0

        # Manifest BEFORE the delete: this is what makes it reversible.
        cur.execute(
            f"""SELECT doc_id, meta->>'pmid', meta->>'pmcid', title
                FROM documents WHERE {drop_pred}""", {"min": args.min_chars})
        rows = cur.fetchall()
        manifest = local_data_dir() / "pruned_abstract_only.json"
        manifest.write_text(json.dumps({
            "pruned_at": datetime.now(timezone.utc).isoformat(),
            "min_full_text_chars": args.min_chars,
            "count": len(rows),
            "documents": [{"doc_id": d, "pmid": p, "pmcid": c, "title": t}
                          for d, p, c, t in rows],
        }, indent=2), encoding="utf-8")
        pmid_file = local_data_dir() / "pruned_pmids.txt"
        pmid_file.write_text("\n".join(p for _, p, _, _ in rows if p), encoding="utf-8")
        print(f"\nmanifest -> {manifest}")
        print(f"pmids    -> {pmid_file}   (re-ingest with ingest_real.py --pmids-file)")

        # chunks first: it has a FK-shaped dependency on documents even where the
        # constraint is not declared, and orphan passages are worse than orphan documents.
        cur.execute(
            f"""DELETE FROM chunks c
                WHERE EXISTS (SELECT 1 FROM documents d
                              WHERE d.doc_id = c.doc_id AND {drop_pred})""",
            {"min": args.min_chars})
        gone_chunks = cur.rowcount
        cur.execute(f"DELETE FROM documents WHERE {drop_pred}", {"min": args.min_chars})
        gone_docs = cur.rowcount
        conn.commit()

        cur.execute("SELECT count(*) FROM documents")
        now_docs = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM chunks WHERE kind = 'passage'")
        now_chunks = cur.fetchone()[0]
        cur.execute("""SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY n) FROM (
                         SELECT doc_id, count(*) n FROM chunks
                          WHERE kind = 'passage' GROUP BY doc_id) t""")
        median = cur.fetchone()[0]

    print(f"\ndeleted {gone_docs:,} documents and {gone_chunks:,} passages")
    print(f"corpus now {now_docs:,} documents / {now_chunks:,} passages "
          f"(median {median:.0f} passages per document)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
