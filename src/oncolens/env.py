"""Load `.env.local` automatically so credentials work the same on every platform.

`vercel env pull` writes a `.env.local` file, and the usual advice is to source it into
the shell. That advice is bash-shaped: on Windows PowerShell it needs a regex loop that is
easy to get subtly wrong (quoted values, `=` inside a connection string, BOM), and getting
it wrong produces a confusing "POSTGRES_URL not set" rather than an obvious error.

So the scripts read the file directly instead. Existing environment variables always win,
so CI — where secrets arrive as real env vars and no file exists — is unaffected.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Searched in order; the first file found wins.
CANDIDATES = (".env.local", ".env")


def load_env(root: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Read `.env.local` (or `.env`) into ``os.environ``. Returns what was loaded.

    Values may be quoted and may contain ``=`` (Postgres DSNs routinely do), so the split
    is on the *first* ``=`` only and surrounding quotes are stripped.
    """
    root = root or Path(__file__).resolve().parents[2]
    loaded: dict[str, str] = {}
    for name in CANDIDATES:
        path = root / name
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = value
            loaded[key] = value
        break
    return loaded


def describe_credentials() -> list[str]:
    """Human-readable report of which credentials are present. Never prints values."""
    checks = [
        ("POSTGRES_URL", os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")),
        ("BLOB_READ_WRITE_TOKEN", os.environ.get("BLOB_READ_WRITE_TOKEN")),
        ("NCBI_API_KEY", os.environ.get("NCBI_API_KEY")),
    ]
    out = []
    for name, val in checks:
        if val:
            out.append(f"  {name:<24} set ({len(val)} chars)")
        else:
            optional = name == "NCBI_API_KEY"
            out.append(f"  {name:<24} MISSING{' (optional)' if optional else ''}")
    return out
