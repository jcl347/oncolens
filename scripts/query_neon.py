#!/usr/bin/env python
"""Query the live Neon store and show the passage each hit came from.

Proves the deployed path end to end: real papers, real passages, real provenance.

    python scripts/query_neon.py "EGFR C797S resistance"
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Real biomedical text contains Greek letters, math symbols and superscripts; the Windows
# console defaults to cp1252 and would raise UnicodeEncodeError on them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


from oncolens.env import load_env  # noqa: E402
from oncolens.spans import find_clauses  # noqa: E402


def main() -> int:
    load_env(override=True)
    import os

    import psycopg

    query = sys.argv[1] if len(sys.argv) > 1 else "EGFR C797S resistance"
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    dsn = os.environ.get("POSTGRES_URL") or os.environ["DATABASE_URL"]
    with psycopg.connect(dsn, connect_timeout=20) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents")
        n_docs = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM chunks")
        n_chunks = cur.fetchone()[0]
        print(f"store: {n_docs} documents, {n_chunks} passages\n")
        print(f"query: {query!r}\n")

        # Lexical arm only here — the dense arm needs the same fitted LSA projection the
        # ingestion used, which lives with the artifact. This still exercises the real
        # schema, the real GIN index, and real provenance.
        cur.execute(
            """
            SELECT c.doc_id, d.title, d.meta, c.section, c.start_char, c.end_char, c.text,
                   ts_rank_cd(c.tsv, plainto_tsquery('english', %s)) AS rank
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.tsv @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
            """,
            (query, query, top_k),
        )
        rows = cur.fetchall()

    if not rows:
        print("no matches")
        return 0

    for i, (doc_id, title, meta, section, start, end, text, rank) in enumerate(rows, 1):
        pmid = (meta or {}).get("pmid")
        print(f"{i}. {doc_id}  [rank {rank:.4f}]")
        print(f"   {title[:88]}")
        if pmid:
            print(f"   https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
        clauses = find_clauses(text, query, base_offset=start, max_clauses=1)
        if clauses:
            cl = clauses[0]
            print(f"   clause [{section} {cl.start}-{cl.end}] matched on "
                  f"{', '.join(cl.matched_terms)}:")
            print(f"     …{cl.text[:190]}…")
        else:
            print(f"   passage [{section} {start}-{end}]: {text[:150]}…")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
