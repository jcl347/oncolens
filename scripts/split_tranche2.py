#!/usr/bin/env python
"""Split the second oncology-screened tranche into ingestible batches.

``ingest_real.py`` writes to Postgres only after fetching EVERY paper in the run, so one
long list is all-or-nothing: a transport failure an hour in loses the whole job. Batching
turns that into bounded loss and lets an expansion be stopped at any point with everything
so far already committed.

Also drops PMIDs already held, so a resumed expansion does not re-fetch what it has.
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

from oncolens.env import load_env, local_data_dir  # noqa: E402


def held_pmids() -> set[str]:
    import psycopg

    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        return set()
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT meta->>'pmid' FROM documents WHERE meta->>'pmid' <> ''")
        return {r[0] for r in cur.fetchall() if r[0]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--batch", type=int, default=700)
    args = ap.parse_args()
    load_env()

    src = Path(args.inp) if args.inp else local_data_dir() / "onco_pmids_r2.txt"
    out_dir = Path(args.out_dir) if args.out_dir else local_data_dir() / "batches2"
    pmids = [p for p in src.read_text(encoding="utf-8").split() if p.strip().isdigit()]

    have = held_pmids()
    fresh = [p for p in pmids if p not in have]
    print(f"{len(pmids)} screened, {len(pmids) - len(fresh)} already held, "
          f"{len(fresh)} to ingest")

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("batch_*.txt"):
        f.unlink()
    n = 0
    for i in range(0, len(fresh), args.batch):
        n += 1
        (out_dir / f"batch_{n:02d}.txt").write_text(
            "\n".join(fresh[i:i + args.batch]), encoding="utf-8")
    print(f"wrote {n} batches of {args.batch} to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
