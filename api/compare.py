"""Comparative retrieval endpoint: papers x technical dimensions, each cell cited.

Runs against the LIVE store. The previous version imported the offline harness
(``load_dataset`` + ``build_retriever``), which needs a fitted in-process index, a local
corpus directory, and scipy — none of which exist in a serverless function. It returned
``No module named 'scipy'`` in production while passing every local test, because locally
all three happened to be present.
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


class handler(BaseHTTPRequestHandler):  # noqa: N801
    def do_GET(self):  # noqa: N802
        p = parse_qs(urlparse(self.path).query)
        q = (p.get("q") or [""])[0].strip()
        if not q:
            return self._send(400, {"error": "missing required query parameter 'q'"})
        try:
            n = max(2, min(10, int((p.get("n") or ["5"])[0])))
        except ValueError:
            n = 5
        aspects = tuple(a for a in (p.get("aspect") or []) if a) or None

        try:
            from oncolens.serve.live_query import compare, get_live_index
            from oncolens.serve.neon_store import EmbeddingMismatch
        except ImportError as e:
            return self._send(503, {
                "error": "server package not available in this deployment",
                "detail": str(e)[:200],
                "fix": 'vercel.json must set "includeFiles": "src/**" for this function, '
                       "and pyproject.toml must list the runtime dependencies — Vercel "
                       "installs from pyproject.toml in preference to requirements.txt",
            })

        live = get_live_index()
        if live is None:
            return self._send(503, {
                "error": "no store configured",
                "fix": "set POSTGRES_URL (Neon pooler endpoint) in the Vercel project's "
                       "Production environment, then redeploy",
            })
        try:
            return self._send(200, compare(live, q, n_papers=n, aspect_keys=aspects))
        except EmbeddingMismatch as e:
            return self._send(503, {"error": "index/query embedding mismatch",
                                    "detail": str(e)[:400]})
        except Exception as e:  # never leak a stack trace to the client
            return self._send(500, {
                "error": "comparison failed",
                "detail": f"{type(e).__name__}: {e}"[:300],
                "config": {
                    "postgres_url_set": bool(os.environ.get("POSTGRES_URL")
                                             or os.environ.get("DATABASE_URL")),
                    "openai_api_key_set": bool(os.environ.get("OPENAI_API_KEY")),
                },
            })

    def _send(self, status: int, body: dict):
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        # Never let a failure be shared-cached. Neon suspends when idle, so the first
        # request after a quiet period can 503 while the very next one succeeds; caching
        # that 503 at the edge turns a one-second wake-up into minutes of "unavailable"
        # and makes the retry that would have worked look futile.
        if status >= 400:
            self.send_header("Cache-Control", "no-store")
        else:
            self.send_header("Cache-Control", "public, max-age=60, s-maxage=300")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        return
