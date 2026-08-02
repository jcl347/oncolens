"""Vercel Blob storage — where ingested full text actually lives.

**Why not the repo / not a local folder.** Real PMC full text is ~25KB per article; a few
thousand articles is hundreds of megabytes and it grows. That does not belong in git, and
it especially does not belong inside a synced folder (OneDrive/Dropbox), where large
churning corpora cause sync storms and file locks — this project already hit a OneDrive
lock that blocked a directory delete mid-run.

So the storage split is:

| What | Where | Why |
|---|---|---|
| Raw article full text | **Vercel Blob** | Large, immutable, served by URL, no DB bloat |
| Chunks + embeddings + metadata | **Postgres/pgvector** (Neon) | Queried per request, needs joins and ANN |
| Code, tests, fixtures | git | Small, reviewable |

Blob holds the source of truth for text; Postgres holds only what a query needs. A passage
returned to a user carries its blob URL, so the full article is one fetch away without
storing it twice.

Auth is the ``BLOB_READ_WRITE_TOKEN`` that the Vercel Blob integration injects into the
project environment. This module talks to the REST API directly so ingestion can run
anywhere — laptop, CI, or a Vercel build step — without a Node runtime.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

API = "https://blob.vercel-storage.com"
API_VERSION = "7"

#: Private stores are served from a DIFFERENT host (<store>.private.blob.vercel-storage.com)
#: and reject every REST upload with "Cannot use public access on a private store" — no
#: combination of x-access / x-blob-access / api-version gets past it. The official Node
#: SDK negotiates whatever private stores actually use, so uploads route through a small
#: persistent Node bridge instead of reverse-engineering an undocumented handshake that
#: would break silently the next time Vercel changes it.
_BRIDGE = Path(__file__).resolve().parents[3] / "scripts" / "blob_bridge.mjs"
_bridge_proc = None


@dataclass(frozen=True)
class BlobRef:
    url: str
    pathname: str
    size: int

    def as_meta(self) -> dict:
        return {"blob_url": self.url, "blob_path": self.pathname, "blob_size": self.size}


def _token() -> str:
    tok = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if not tok:
        raise RuntimeError(
            "BLOB_READ_WRITE_TOKEN is not set. Add the Vercel Blob integration to the "
            "project (Vercel dashboard -> Storage -> Blob); it injects this automatically. "
            "For local ingestion, run `vercel env pull` to fetch it into .env.local."
        )
    return tok


def _session():
    import requests
    return requests.Session()


def _bridge_call(req: dict) -> dict:
    """Send one request to the persistent Node bridge and read one response.

    The process is reused across calls: spawning Node per article would dominate runtime
    on a multi-thousand-document ingest.
    """
    global _bridge_proc
    import json as _json
    import subprocess

    if not _BRIDGE.exists():
        raise RuntimeError(f"blob bridge missing at {_BRIDGE}; run `npm install @vercel/blob`")

    if _bridge_proc is None or _bridge_proc.poll() is not None:
        _bridge_proc = subprocess.Popen(
            ["node", str(_BRIDGE)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
            env={**os.environ, "BLOB_READ_WRITE_TOKEN": _token()},
        )

    _bridge_proc.stdin.write(_json.dumps(req) + "\n")
    _bridge_proc.stdin.flush()
    line = _bridge_proc.stdout.readline()
    if not line:
        err = _bridge_proc.stderr.read()[:300] if _bridge_proc.stderr else ""
        raise RuntimeError(f"blob bridge died: {err}")
    resp = _json.loads(line)
    if not resp.get("ok"):
        raise RuntimeError(f"blob: {resp.get('error')}")
    return resp


def put_text(
    pathname: str, content: str, *, content_type: str = "text/plain; charset=utf-8",
    add_random_suffix: bool = False, cache_max_age: int = 31536000,
    access: str = "private",
) -> BlobRef:
    """Upload text and return its public URL.

    ``add_random_suffix=False`` keeps the path deterministic (``pmc/PMC123.1.txt``), so
    re-ingesting the same article overwrites rather than accumulating duplicates — which
    matters when an ingestion job is re-run after a partial failure.
    """
    resp = _bridge_call({
        "op": "put", "pathname": pathname.lstrip("/"),
        "content": content, "contentType": content_type, "access": access,
    })
    return BlobRef(url=resp["url"], pathname=resp.get("pathname", pathname),
                   size=resp.get("size", len(content.encode("utf-8"))))


def put_json(pathname: str, obj: dict, **kw) -> BlobRef:
    return put_text(pathname, json.dumps(obj, ensure_ascii=False), content_type="application/json", **kw)


def get_text(url: str) -> str:
    """Read a blob. Private stores require the token, so it is always sent."""
    s = _session()
    r = s.get(url, headers={"authorization": f"Bearer {_token()}"}, timeout=120)
    r.raise_for_status()
    return r.text


def list_blobs(prefix: str = "", limit: int = 1000) -> list[dict]:
    s = _session()
    out: list[dict] = []
    cursor = None
    while len(out) < limit:
        params = {"prefix": prefix, "limit": str(min(1000, limit - len(out)))}
        if cursor:
            params["cursor"] = cursor
        r = s.get(API, params=params,
                  headers={"authorization": f"Bearer {_token()}", "x-api-version": API_VERSION},
                  timeout=60)
        r.raise_for_status()
        data = r.json()
        out.extend(data.get("blobs", []))
        cursor = data.get("cursor")
        if not cursor or not data.get("hasMore"):
            break
    return out


def delete(urls: list[str]) -> None:
    _bridge_call({"op": "del", "urls": urls})


def blob_path_for(pmcid: str, version: int = 1, kind: str = "txt") -> str:
    """Deterministic, collision-free path. Versioned because PMC revises articles."""
    return f"pmc/{kind}/{pmcid}.{version}.{kind}"
