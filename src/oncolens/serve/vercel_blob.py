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

API = "https://blob.vercel-storage.com"
API_VERSION = "7"


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


def put_text(
    pathname: str, content: str, *, content_type: str = "text/plain; charset=utf-8",
    add_random_suffix: bool = False, cache_max_age: int = 31536000,
) -> BlobRef:
    """Upload text and return its public URL.

    ``add_random_suffix=False`` keeps the path deterministic (``pmc/PMC123.1.txt``), so
    re-ingesting the same article overwrites rather than accumulating duplicates — which
    matters when an ingestion job is re-run after a partial failure.
    """
    s = _session()
    body = content.encode("utf-8")
    r = s.put(
        f"{API}/{pathname.lstrip('/')}",
        data=body,
        headers={
            "authorization": f"Bearer {_token()}",
            "x-api-version": API_VERSION,
            "x-content-type": content_type,
            "x-add-random-suffix": "1" if add_random_suffix else "0",
            "x-cache-control-max-age": str(cache_max_age),
        },
        timeout=180,
    )
    r.raise_for_status()
    data = r.json()
    return BlobRef(url=data["url"], pathname=data.get("pathname", pathname), size=len(body))


def put_json(pathname: str, obj: dict, **kw) -> BlobRef:
    return put_text(pathname, json.dumps(obj, ensure_ascii=False), content_type="application/json", **kw)


def get_text(url: str) -> str:
    """Blob URLs are public by default; no auth needed to read."""
    s = _session()
    r = s.get(url, timeout=120)
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
    s = _session()
    r = s.post(f"{API}/delete", json={"urls": urls},
               headers={"authorization": f"Bearer {_token()}",
                        "x-api-version": API_VERSION,
                        "content-type": "application/json"},
               timeout=120)
    r.raise_for_status()


def blob_path_for(pmcid: str, version: int = 1, kind: str = "txt") -> str:
    """Deterministic, collision-free path. Versioned because PMC revises articles."""
    return f"pmc/{kind}/{pmcid}.{version}.{kind}"
