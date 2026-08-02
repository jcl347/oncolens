"""Vercel Python serverless function: query -> ranked documents + supporting passages.

Deployment constraints this is written against:

* **Ephemeral, read-only filesystem.** Nothing is fitted or written at request time; the
  index artifact is built offline and bundled.
* **Bundle size / cold start.** Runtime dependency is **numpy only**. scipy is build-time
  only (the SVD lives in the artifact already). The tokenizer is imported from
  ``oncolens.retrieval.text``, which depends on nothing but ``re`` — importing rather than
  duplicating it is deliberate: a serving tokenizer that drifts from the evaluated one
  would silently invalidate every offline measurement.
* **Warm reuse.** The artifact loads at module scope, so a warm container pays the load
  cost once rather than per request.

``tests/test_serve_parity.py`` asserts this path returns the same ranking as the offline
``Retriever`` for the same config. That test is the contract that lets the evaluation
numbers describe production.
"""

from __future__ import annotations

import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from oncolens.retrieval.text import is_rare_literal, tokenize  # noqa: E402
from oncolens.spans import find_clauses  # noqa: E402

ARTIFACT_DIR = Path(os.environ.get("ONCOLENS_ARTIFACT", _ROOT / "artifact"))


class Index:
    """Loaded once per container, reused across warm invocations."""

    def __init__(self, artifact_dir: Path):
        meta = json.loads((artifact_dir / "index.json").read_text(encoding="utf-8"))
        npz = np.load(artifact_dir / "index.npz")
        self.cfg = meta["config"]
        self.postings: dict[str, list[list[int]]] = meta["postings"]
        self.df: dict[str, int] = meta["bm25_df"]
        self.doc_len: list[int] = meta["doc_len"]
        self.avgdl: float = meta["avgdl"] or 1.0
        self.n: int = meta["n_chunks"]
        self.chunks: list[dict] = meta["chunks"]
        self.documents: dict[str, dict] = meta["documents"]
        self.lexicon: dict[str, list[str]] = meta.get("lexicon", {})
        self._lex_keys = sorted(self.lexicon, key=len, reverse=True)
        self.tfidf_vocab: dict[str, int] = meta["tfidf_vocab"]
        self.components = npz["components"]        # (dim, vocab)
        self.chunk_vectors = npz["chunk_vectors"]  # (n_chunks, dim)
        self.tfidf_idf = npz["tfidf_idf"]          # (vocab,)

    # -- lexical ---------------------------------------------------------
    def _idf(self, term: str) -> float:
        c = self.df.get(term, 0)
        if c == 0:
            return 0.0
        v = math.log(1.0 + (self.n - c + 0.5) / (c + 0.5))
        boost = self.cfg.get("literal_boost", 1.0)
        if boost != 1.0 and is_rare_literal(term):
            v *= boost
        return v

    def bm25(self, tokens: list[str]) -> dict[int, float]:
        k1 = self.cfg.get("bm25_k1", 1.2)
        b = self.cfg.get("bm25_b", 0.75)
        qtf: dict[str, int] = {}
        for t in tokens:
            qtf[t] = qtf.get(t, 0) + 1
        scores: dict[int, float] = {}
        for term, qf in qtf.items():
            plist = self.postings.get(term)
            if not plist:
                continue
            w = self._idf(term)
            if w <= 0:
                continue
            qw = (qf / (qf + 0.5)) * 1.5
            for idx, freq in plist:
                dl = self.doc_len[idx]
                denom = freq + k1 * (1 - b + b * dl / self.avgdl)
                scores[idx] = scores.get(idx, 0.0) + w * (freq * (k1 + 1) / denom) * qw
        return scores

    # -- dense -----------------------------------------------------------
    def dense(self, query: str, k: int) -> list[tuple[int, float]]:
        vec = np.zeros(len(self.tfidf_vocab), dtype=np.float32)
        counts: dict[int, int] = {}
        for t in tokenize(query):
            j = self.tfidf_vocab.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        if not counts:
            return []
        for j, c in counts.items():
            vec[j] = (1.0 + math.log(c)) * self.tfidf_idf[j]
        q = self.components @ vec
        nrm = float(np.linalg.norm(q))
        if nrm == 0:
            return []
        q = q / nrm
        sims = self.chunk_vectors @ q
        k = min(k, len(sims))
        part = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
        order = part[np.argsort(-sims[part])]
        return [(int(i), float(sims[i])) for i in order]

    # -- expansion -------------------------------------------------------
    def expand(self, query: str) -> str:
        if self.cfg.get("expansion") != "lexicon" or not self.lexicon:
            return query
        low = query.lower()
        seen = set(tokenize(query))
        added: list[str] = []
        cap = self.cfg.get("exp_max_total", 24)
        per = self.cfg.get("exp_max_variants", 4)
        for key in self._lex_keys:
            if len(added) >= cap:
                break
            if key in low:
                for variant in self.lexicon[key][:per]:
                    for tok in tokenize(variant):
                        if tok not in seen:
                            seen.add(tok)
                            added.append(tok)
        return (query + " " + " ".join(added)).strip()

    # -- search ----------------------------------------------------------
    def search(self, query: str, top_k: int = 10) -> dict:
        cand = self.cfg.get("candidates_per_arm", 200)
        runs: list[list[tuple[int, float]]] = []
        weights: list[float] = []

        if self.cfg.get("use_bm25", True):
            q = self.expand(query) if self.cfg.get("expansion") != "none" else query
            scored = self.bm25(tokenize(q))
            runs.append(sorted(scored.items(), key=lambda x: (-x[1], x[0]))[:cand])
            weights.append(self.cfg.get("bm25_weight", 1.0))

        if self.cfg.get("use_dense", True):
            qd = self.expand(query) if self.cfg.get("exp_apply_to") == "both" else query
            runs.append(self.dense(qd, cand))
            weights.append(self.cfg.get("dense_weight", 1.0))

        if not runs:
            return {"query": query, "results": []}

        if len(runs) == 1:
            fused = list(runs[0])
        elif self.cfg.get("fusion", "rrf") == "rrf":
            kk = self.cfg.get("rrf_k", 60.0)
            agg: dict[int, float] = {}
            for run, w in zip(runs, weights):
                for rank, (idx, _) in enumerate(run, start=1):
                    agg[idx] = agg.get(idx, 0.0) + w / (kk + rank)
            fused = sorted(agg.items(), key=lambda x: (-x[1], x[0]))
        else:
            agg = {}
            for run, w in zip(runs, weights):
                if not run:
                    continue
                vals = [s for _, s in run]
                lo, hi = min(vals), max(vals)
                rng = (hi - lo) or 1.0
                for idx, s in run:
                    agg[idx] = agg.get(idx, 0.0) + w * ((s - lo) / rng)
            fused = sorted(agg.items(), key=lambda x: (-x[1], x[0]))

        # chunk -> document, keeping the best passage as the citation
        best: dict[str, tuple[int, float]] = {}
        for idx, score in fused:
            doc_id = self.chunks[idx]["doc_id"]
            if doc_id not in best or score > best[doc_id][1]:
                best[doc_id] = (idx, score)

        ranked = sorted(best.items(), key=lambda x: (-x[1][1], x[0]))[:top_k]
        results = []
        for doc_id, (idx, score) in ranked:
            c = self.chunks[idx]
            d = self.documents.get(doc_id, {})
            _clauses = find_clauses(c["text"], query, base_offset=c["start_char"], max_clauses=3)
            results.append({
                "doc_id": doc_id,
                "title": d.get("title", ""),
                "doc_type": d.get("doc_type", ""),
                "year": d.get("year"),
                "meta": d.get("meta", {}),
                "score": round(score, 6),
                # The product requirement: the passage, and where it lives.
                "passage": {
                    "chunk_id": c["chunk_id"], "section": c["section"],
                    "start_char": c["start_char"], "end_char": c["end_char"],
                    "text": c["text"],
                    # Clause-level provenance: the UI highlights the exact span that
                    # matched, not the whole paragraph. Offsets are section-absolute so a
                    # viewer can locate them in the original article.
                    "clauses": [cl.as_dict() for cl in _clauses],
                    "best_clause": _clauses[0].as_dict() if _clauses else None,
                },
            })
        return {"query": query, "count": len(results), "results": results}


_INDEX: Index | None = None


def get_index() -> Index:
    global _INDEX
    if _INDEX is None:
        _INDEX = Index(ARTIFACT_DIR)
    return _INDEX


class handler(BaseHTTPRequestHandler):  # noqa: N801 — Vercel expects this exact name
    def do_GET(self):  # noqa: N802
        params = parse_qs(urlparse(self.path).query)
        query = (params.get("q") or [""])[0].strip()
        try:
            top_k = max(1, min(50, int((params.get("k") or ["10"])[0])))
        except ValueError:
            top_k = 10

        if not query:
            return self._send(400, {"error": "missing required query parameter 'q'"})
        try:
            payload = get_index().search(query, top_k=top_k)
        except FileNotFoundError:
            return self._send(503, {
                "error": "search index artifact not found",
                "hint": "run `python scripts/build_artifact.py` and deploy the artifact/ directory",
            })
        except Exception as e:  # never leak a stack trace to the client
            return self._send(500, {"error": "search failed", "detail": str(e)[:200]})
        return self._send(200, payload)

    def _send(self, status: int, body: dict):
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "public, max-age=60, s-maxage=300")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):  # silence default stderr logging
        return
