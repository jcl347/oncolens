#!/usr/bin/env python
"""Verify the Vercel storage containers are reachable and correctly provisioned.

Run this before ingestion. It fails loudly and specifically rather than letting a
misconfigured store surface halfway through a long ingest, and it never prints secrets.

    python scripts/check_stores.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oncolens.env import load_env  # noqa: E402


def redact_dsn(dsn: str) -> str:
    """Show enough to identify the host, never the password."""
    try:
        tail = dsn.split("@", 1)[1]
        host = tail.split("/", 1)[0]
        db = tail.split("/", 1)[1].split("?")[0] if "/" in tail else "?"
        return f"{host}/{db}"
    except Exception:
        return "<unparseable>"


def main() -> int:
    loaded = load_env(override=True)
    print(f"loaded {len(loaded)} variables from .env.local\n")
    problems: list[str] = []

    # ---------------------------------------------------------- Postgres
    print("Postgres (Neon + pgvector)")
    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        print("  MISSING  POSTGRES_URL / DATABASE_URL")
        problems.append("Postgres credentials absent")
    else:
        print(f"  host           {redact_dsn(dsn)}")
        try:
            import psycopg

            with psycopg.connect(dsn, connect_timeout=20) as conn, conn.cursor() as cur:
                cur.execute("SELECT version()")
                print(f"  server         {cur.fetchone()[0].split(',')[0]}")

                cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                installed = cur.fetchone() is not None
                cur.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
                available = cur.fetchone() is not None
                print(f"  pgvector       installed={installed} available={available}")
                if not installed and not available:
                    problems.append("pgvector is neither installed nor available on this database")

                cur.execute("SELECT to_regclass('public.documents'), to_regclass('public.chunks')")
                docs, chunks = cur.fetchone()
                print(f"  schema         documents={'yes' if docs else 'no'} chunks={'yes' if chunks else 'no'}")
                if docs and chunks:
                    cur.execute("SELECT count(*) FROM documents")
                    nd = cur.fetchone()[0]
                    cur.execute("SELECT count(*) FROM chunks")
                    nc = cur.fetchone()[0]
                    print(f"  rows           {nd} documents, {nc} passages")
                else:
                    print("  schema         not created yet (ingestion creates it)")
        except Exception as e:
            print(f"  FAILED         {type(e).__name__}: {str(e)[:140]}")
            problems.append(f"Postgres unreachable: {type(e).__name__}")

    # -------------------------------------------------------------- Blob
    print("\nVercel Blob")
    tok = os.environ.get("BLOB_READ_WRITE_TOKEN")
    store = os.environ.get("BLOB_STORE_ID")
    print(f"  BLOB_STORE_ID           {'present' if store else 'MISSING'}")
    print(f"  BLOB_READ_WRITE_TOKEN   {'present' if tok else 'MISSING'}")
    if not tok:
        problems.append(
            "BLOB_READ_WRITE_TOKEN absent. The store exists (BLOB_STORE_ID is set) but the "
            "read/write token was not added to this project's environment"
        )
    else:
        try:
            from oncolens.serve import vercel_blob

            ref = vercel_blob.put_text("_setup/probe.txt", "oncolens store probe")
            body = vercel_blob.get_text(ref.url)
            vercel_blob.delete([ref.url])
            ok = body.startswith("oncolens")
            print(f"  round-trip              {'OK' if ok else 'MISMATCH'} ({ref.size} bytes written, read back, deleted)")
            if not ok:
                problems.append("Blob round-trip returned unexpected content")
        except Exception as e:
            print(f"  round-trip              FAILED {type(e).__name__}: {str(e)[:120]}")
            problems.append(f"Blob unusable: {type(e).__name__}")

    # ------------------------------------------------------------ verdict
    print()
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
        if not tok:
            print(
                "\nTo fix the Blob token:\n"
                "  Vercel dashboard -> Storage -> your Blob store -> Connect Project\n"
                "  (or Project -> Settings -> Environment Variables -> add BLOB_READ_WRITE_TOKEN)\n"
                "  then: vercel env pull .env.local --environment=production"
            )
        return 1

    print("All stores reachable and correctly provisioned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
