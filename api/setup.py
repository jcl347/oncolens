"""Setup endpoint: check infrastructure status and run the actions that safely fit.

**What can and cannot be a button.** Vercel functions are capped at 60s here. So:

| Action | Runs in a function? | Why |
|---|---|---|
| Status checks (Blob token, Postgres reachable, schema, counts) | yes | milliseconds |
| Create schema + indexes | yes | one DDL round-trip |
| Blob write/read/delete round-trip | yes | one small object |
| **Sample ingest (<= 25 papers)** | yes, just | NCBI asks <= 3 req/s, so ~25 is the ceiling inside 60s |
| **Full ingest (thousands of papers)** | **no** | minutes to hours — the UI emits the command instead |
| **Build eval report on a large corpus** | **no** | fits an SVD in memory; belongs offline |

A button that claimed to run a three-hour job inside a 60-second function would simply
fail halfway and leave the stores half-populated, so those are surfaced as commands to
copy rather than actions to click.

**Auth.** Every mutating action writes to real infrastructure, so they are gated on
``SETUP_TOKEN``. Without it set, mutations are refused and only read-only status is served
— an unauthenticated endpoint that can create tables and spend NCBI quota is not something
to leave open on a public URL.
"""

from __future__ import annotations

import hmac
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

#: Hard ceiling for the in-function sample ingest. NCBI's rate limit makes anything
#: larger a guaranteed timeout rather than a slow success.
SAMPLE_MAX = 25


def _authorized(headers) -> bool:
    expected = os.environ.get("SETUP_TOKEN")
    if not expected:
        return False
    got = (headers.get("x-setup-token") or "").strip()
    return bool(got) and hmac.compare_digest(got, expected)


# --------------------------------------------------------------------------- checks

def check_blob() -> dict:
    if not os.environ.get("BLOB_READ_WRITE_TOKEN"):
        return {"ok": False, "state": "missing",
                "detail": "BLOB_READ_WRITE_TOKEN not set",
                "fix": "Vercel -> Storage -> Create Database -> Blob, then connect it to this project"}
    return {"ok": True, "state": "configured", "detail": "token present"}


def check_postgres() -> dict:
    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        return {"ok": False, "state": "missing",
                "detail": "POSTGRES_URL not set",
                "fix": "Vercel -> Storage -> Create Database -> Neon, then connect it to this project"}
    try:
        import psycopg
    except ImportError:
        return {"ok": False, "state": "driver-missing",
                "detail": "psycopg is not installed in the function bundle",
                "fix": 'add "psycopg[binary]" to requirements.txt and redeploy'}
    try:
        with psycopg.connect(dsn, connect_timeout=8) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            has_vector = cur.fetchone() is not None
            cur.execute("SELECT to_regclass('public.chunks'), to_regclass('public.documents')")
            chunks_tbl, docs_tbl = cur.fetchone()
            counts = {}
            if chunks_tbl and docs_tbl:
                cur.execute("SELECT count(*) FROM documents")
                counts["documents"] = cur.fetchone()[0]
                cur.execute("SELECT count(*) FROM chunks")
                counts["chunks"] = cur.fetchone()[0]
        return {
            "ok": True,
            "state": "ready" if (chunks_tbl and has_vector) else "connected",
            "detail": f"connected; pgvector={'yes' if has_vector else 'no'}; "
                      f"schema={'yes' if chunks_tbl else 'no'}",
            "counts": counts,
            "fix": None if chunks_tbl else "run the Create schema action",
        }
    except Exception as e:
        return {"ok": False, "state": "unreachable", "detail": str(e)[:180],
                "fix": "check POSTGRES_URL and that the Neon database is awake"}


def check_corpus() -> dict:
    try:
        from oncolens.data import load_dataset
        ds = load_dataset(strict=False)
        n = ds.integrity.get("n_docs", 0)
        if n == 0:
            return {"ok": False, "state": "empty", "detail": "no documents ingested",
                    "fix": "run the full ingest command below"}
        return {"ok": True, "state": "populated",
                "detail": f"{n} documents, {ds.integrity.get('n_queries', 0)} eval queries",
                "counts": {"documents": n, "queries": ds.integrity.get("n_queries", 0)}}
    except Exception as e:
        return {"ok": False, "state": "error", "detail": str(e)[:180]}


def check_eval_report() -> dict:
    p = Path(os.environ.get("ONCOLENS_EVAL_REPORT", _ROOT / "public" / "eval_report.json"))
    if not p.exists():
        return {"ok": False, "state": "missing",
                "detail": "no evaluation report published",
                "fix": "run build_eval_report.py and redeploy — until then the site reports itself as unmeasured"}
    try:
        rep = json.loads(p.read_text(encoding="utf-8"))
        primary = rep.get("metrics", {}).get(rep.get("primary_metric", ""), None)
        floor = (rep.get("floor") or {}).get("raw_term_frequency")
        return {"ok": True, "state": "published",
                "detail": f"{rep.get('primary_metric')}={primary} vs raw-TF floor {floor}",
                "generated_at": rep.get("generated_at"),
                "caveats": len(rep.get("caveats", []))}
    except Exception as e:
        return {"ok": False, "state": "unreadable", "detail": str(e)[:150]}


def check_index_artifact() -> dict:
    d = Path(os.environ.get("ONCOLENS_ARTIFACT", _ROOT / "artifact"))
    if not (d / "index.json").exists():
        return {"ok": False, "state": "missing", "detail": "no bundled search index",
                "fix": "run build_artifact.py, or switch the API to the Postgres path"}
    try:
        meta = json.loads((d / "index.json").read_text(encoding="utf-8"))
        return {"ok": True, "state": "built",
                "detail": f"{meta.get('n_chunks')} passages, config {meta.get('config', {}).get('name')}"}
    except Exception as e:
        return {"ok": False, "state": "unreadable", "detail": str(e)[:150]}


def full_status() -> dict:
    checks = {
        "blob": check_blob(),
        "postgres": check_postgres(),
        "corpus": check_corpus(),
        "index": check_index_artifact(),
        "eval_report": check_eval_report(),
    }
    ready = all(c["ok"] for c in checks.values())
    return {
        "ready": ready,
        "auth_configured": bool(os.environ.get("SETUP_TOKEN")),
        "checks": checks,
        # Commands that cannot run in a function, surfaced verbatim so they can be copied.
        "commands": {
            "full_ingest": ("python scripts/ingest_real.py --max-papers 2000 "
                            "--email you@org.edu"),
            "build_eval_report": "python scripts/build_eval_report.py --out public/eval_report.json",
            "build_artifact": "python scripts/build_artifact.py --out artifact",
            "pull_env": "vercel env pull .env.local",
        },
    }


# --------------------------------------------------------------------------- actions

def action_init_schema(dim: int = 192) -> dict:
    import psycopg
    from oncolens.serve import neon_store

    cfg = neon_store.NeonConfig.from_env(dim=dim)
    t0 = time.time()
    with psycopg.connect(cfg.dsn, connect_timeout=15) as conn:
        neon_store.init_schema(conn, dim=dim)
    return {"ok": True, "action": "init_schema",
            "detail": f"tables, HNSW vector index and GIN text index created (dim={dim})",
            "elapsed_ms": int((time.time() - t0) * 1000)}


def action_test_blob() -> dict:
    from oncolens.serve import vercel_blob

    t0 = time.time()
    probe = f"oncolens-setup-probe-{int(time.time())}"
    ref = vercel_blob.put_text(f"_setup/{probe}.txt", "oncolens storage round-trip probe")
    got = vercel_blob.get_text(ref.url)
    vercel_blob.delete([ref.url])
    return {"ok": got.startswith("oncolens"), "action": "test_blob",
            "detail": f"wrote, read back and deleted {ref.size} bytes",
            "elapsed_ms": int((time.time() - t0) * 1000)}


def action_sample_ingest(n: int) -> dict:
    """Small end-to-end proof: real PubMed -> real MeSH -> Blob -> Postgres."""
    import psycopg
    from oncolens.retrieval.chunking import chunk_corpus
    from oncolens.retrieval.dense import LsaBackend
    from oncolens.serve import neon_store, vercel_blob
    from oncolens.sources import pmc_cloud, pubmed

    n = max(1, min(n, SAMPLE_MAX))
    t0 = time.time()
    pmids = pubmed.esearch(
        '"Immune Checkpoint Inhibitors"[MeSH Terms] AND "Drug Resistance, Neoplasm"[MeSH Terms]',
        retmax=n,
    )
    records = pubmed.efetch(pmids)
    docs = [r.to_corpus_doc() for r in records]

    # Attach real full text where the licence permits it.
    full_text, skipped = 0, 0
    for d, rec in zip(docs, records):
        if not rec.pmcid:
            continue
        meta = pmc_cloud.fetch_metadata(rec.pmcid, 1)
        if not meta or meta.get("is_retracted"):
            continue
        if not pmc_cloud.commercial_use_ok(meta):
            skipped += 1
            continue
        text = pmc_cloud.fetch_full_text(rec.pmcid, 1)
        if text:
            d["sections"].append({"name": "Body", "text": text})
            ref = vercel_blob.put_text(vercel_blob.blob_path_for(rec.pmcid, 1, "txt"), text)
            d["meta"].update(ref.as_meta())
            full_text += 1

    chunks = chunk_corpus(docs)
    texts = [c.indexable_text(include_heading=True) for c in chunks]
    backend = LsaBackend(dim=192)
    backend.fit(texts)
    vectors = backend.encode_documents(texts)

    cfg = neon_store.NeonConfig.from_env(dim=192)
    with psycopg.connect(cfg.dsn, connect_timeout=15) as conn:
        neon_store.init_schema(conn, dim=192)
        neon_store.upsert_documents(conn, docs)
        rows = [{
            "chunk_id": c.chunk_id, "doc_id": c.doc_id, "section": c.section,
            "ordinal": c.ordinal, "start_char": c.start_char, "end_char": c.end_char,
            "text": c.text, "indexed_text": t,
        } for c, t in zip(chunks, texts)]
        neon_store.upsert_chunks(conn, rows, vectors.tolist())

    return {
        "ok": True, "action": "sample_ingest",
        "detail": (f"{len(docs)} real papers, {sum(1 for d in docs if d['descriptors'])} with "
                   f"MeSH labels, {full_text} with full text, {skipped} skipped on licence, "
                   f"{len(chunks)} passages indexed"),
        "elapsed_ms": int((time.time() - t0) * 1000),
        "counts": {"documents": len(docs), "chunks": len(chunks),
                   "full_text": full_text, "licence_skipped": skipped},
    }


ACTIONS = {
    "init_schema": lambda p: action_init_schema(int(p.get("dim", ["192"])[0])),
    "test_blob": lambda p: action_test_blob(),
    "sample_ingest": lambda p: action_sample_ingest(int(p.get("n", ["10"])[0])),
}


class handler(BaseHTTPRequestHandler):  # noqa: N801
    def do_GET(self):  # noqa: N802
        return self._send(200, full_status())

    def do_POST(self):  # noqa: N802
        params = parse_qs(urlparse(self.path).query)
        action = (params.get("action") or [""])[0]
        if action not in ACTIONS:
            return self._send(400, {"error": f"unknown action {action!r}",
                                    "available": sorted(ACTIONS)})
        if not _authorized(self.headers):
            return self._send(401, {
                "error": "unauthorized",
                "detail": ("Mutating actions require the x-setup-token header to match "
                           "SETUP_TOKEN. This endpoint can create tables and spend NCBI "
                           "quota, so it is not left open."),
            })
        try:
            return self._send(200, ACTIONS[action](params))
        except Exception as e:
            return self._send(500, {"ok": False, "action": action,
                                    "error": type(e).__name__, "detail": str(e)[:300]})

    def _send(self, status: int, body: dict):
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        return
