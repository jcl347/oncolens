"""Comparative retrieval endpoint: papers x technical dimensions, each cell cited."""
from __future__ import annotations

import json, os, sys
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
        aspects = [a for a in (p.get("aspect") or []) if a]

        try:
            from oncolens.compare import comparative_search
            from oncolens.configs import BASELINE
            from oncolens.data import load_dataset
            from oncolens.experiment import build_retriever

            ds = load_dataset(strict=False)
            r = build_retriever(ds, BASELINE.variant("cmp", use_dense=True))
            table = comparative_search(r, q, aspects=aspects or None, n_papers=n)
            titles = {d["doc_id"]: d.get("title", "") for d in ds.docs}
            payload = table.as_dict()
            payload["titles"] = {d: titles.get(d, "") for d in table.doc_ids}
            return self._send(200, payload)
        except FileNotFoundError:
            return self._send(503, {"error": "no corpus available", "hint": "run scripts/ingest_real.py"})
        except Exception as e:
            return self._send(500, {"error": "comparison failed", "detail": str(e)[:200]})

    def _send(self, status: int, body: dict):
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "public, max-age=60, s-maxage=300")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        return
