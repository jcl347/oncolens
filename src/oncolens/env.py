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

#: Vercel writes these placeholders for variables marked "Sensitive" — integration-created
#: secrets (Neon, some Blob setups) cannot be read back by the CLI in ANY environment.
#: Treating them as real values produces a baffling driver error ('missing "=" after
#: "[SENSITIVE]"') far from the actual cause, so they are treated as absent.
REDACTED_MARKERS = frozenset({"[SENSITIVE]", "[REDACTED]", "***", "<redacted>"})

_TRUST_STORE_INJECTED = False


def enable_system_trust_store() -> bool:
    """Verify TLS against the **OS** certificate store instead of certifi's bundle.

    **Measured on a machine running Norton (2026-08-02).** Every outbound HTTPS request
    failed — NCBI E-utilities and HuggingFace alike — with::

        [SSL: CERTIFICATE_VERIFY_FAILED] Basic Constraints of CA cert not marked critical

    That error is not a broken server. It is the signature of a TLS-intercepting endpoint
    security product: it terminates the connection, re-signs it with its own root CA, and
    installs that root into the **Windows** certificate store. Python does not read that
    store — ``requests``/``httpx`` verify against the ``certifi`` bundle, which has never
    heard of Norton — so every request fails while every browser on the same machine
    works.

    ⚠️ This corrects an environment fact recorded in CLAUDE.md. "curl is blocked, Python
    ``requests`` is not" held on the previous machine; here ``requests`` fails too, and
    concluding "no network" from that would be the same mistake in a new coat. The network
    is fine — the trust root is the problem.

    ``truststore`` delegates verification to the platform (SChannel on Windows), which
    *does* have the intercepting root, so certificates validate properly. This is not
    ``verify=False``: chain validation still happens, against the store the machine's own
    administrator controls.

    Returns True if injection happened or had already happened, False if ``truststore``
    is not installed — in which case nothing is changed and the caller sees the ordinary
    certifi behaviour.
    """
    global _TRUST_STORE_INJECTED
    if _TRUST_STORE_INJECTED:
        return True
    try:
        import truststore
    except ImportError:
        return False
    truststore.inject_into_ssl()
    _TRUST_STORE_INJECTED = True
    return True


def load_env(root: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Read `.env.local` (or `.env`) into ``os.environ``. Returns what was loaded.

    Values may be quoted and may contain ``=`` (Postgres DSNs routinely do), so the split
    is on the *first* ``=`` only and surrounding quotes are stripped.
    """
    # Done here because every script calls load_env() before its first request, and the
    # alternative is remembering to call it in fifteen entry points. It is a no-op when
    # truststore is absent or the platform store is already in use.
    enable_system_trust_store()

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
            if not key or value in REDACTED_MARKERS:
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
        if val in REDACTED_MARKERS:
            out.append(f"  {name:<24} REDACTED by Vercel (marked Sensitive - cannot be pulled)")
            continue
        if val:
            out.append(f"  {name:<24} set ({len(val)} chars)")
        else:
            optional = name == "NCBI_API_KEY"
            out.append(f"  {name:<24} MISSING{' (optional)' if optional else ''}")
    return out


def local_data_dir() -> Path:
    """Where locally-written artifacts go — deliberately OUTSIDE the repo.

    The repo commonly lives inside a synced folder (OneDrive/Dropbox). Writing a churning
    corpus there causes sync storms and file locks; this project already hit a OneDrive
    lock that blocked a directory delete mid-run. Ingested data belongs in Neon and Blob,
    and anything that must land on disk goes to a local application-data path instead.

    Override with ONCOLENS_LOCAL_DIR.
    """
    import os
    import tempfile

    explicit = os.environ.get("ONCOLENS_LOCAL_DIR")
    if explicit:
        d = Path(explicit)
    elif os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
        d = Path(base) / "oncolens"
    else:
        base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
        d = Path(base) / "oncolens"
    d.mkdir(parents=True, exist_ok=True)
    return d
