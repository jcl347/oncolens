"""One paper, whole, for reading.

Returns the document's metadata and every passage of it in reading order, each carrying
the offsets retrieval cites. The reading page renders from these rows rather than from a
stored full-text copy so that what a reader scrolls is exactly what was searched: a
highlighted result can never point at text the page does not contain.
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
        doc_id = (p.get("id") or [""])[0].strip()
        if not doc_id:
            return self._send(400, {"error": "missing required query parameter 'id'"})
        # Accept a bare PMID as well as the internal doc_id, because that is what a reader
        # copying from PubMed will have.
        if doc_id.isdigit():
            doc_id = f"PAPER:PMID{doc_id}"
        highlight = (p.get("highlight") or [""])[0][:400]

        try:
            from oncolens.serve.live_query import get_document, get_live_index
        except ImportError as e:
            return self._send(503, {
                "error": "server package not available in this deployment",
                "detail": str(e)[:200],
                "fix": 'vercel.json must set "includeFiles": "src/**" for this function',
            })

        live = get_live_index()
        if live is None:
            return self._send(503, {
                "error": "no store configured",
                "fix": "set POSTGRES_URL in the Vercel project's Production environment",
            })
        try:
            doc = get_document(live, doc_id, highlight=highlight)
        except Exception as e:  # never leak a stack trace to the client
            return self._send(500, {
                "error": "lookup failed",
                "detail": f"{type(e).__name__}: {e}"[:300],
                "config": {"postgres_url_set": bool(os.environ.get("POSTGRES_URL")
                                                    or os.environ.get("DATABASE_URL"))},
            })
        if doc is None:
            return self._send(404, {"error": f"no document {doc_id!r} in this corpus"})
        return self._send(200, doc)

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
            self.send_header("Cache-Control", "public, max-age=300, s-maxage=3600")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        return
