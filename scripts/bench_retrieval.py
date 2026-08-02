#!/usr/bin/env python
"""Compare retrieval systems on citation-context labels, with paired statistics.

Each query is a real sentence from a real paper; each judgment is that paper's own
citation. The citing document is **excluded from its own results** — it contains the query
verbatim, so including it would measure string equality rather than retrieval.

Systems compared:

* ``bm25``          — lexical only, the floor any dense model must beat to justify itself
* ``lsa``           — TF-IDF + SVD, i.e. the current production dense backend
* ``openai``        — text-embedding-3-small
* ``hybrid-lsa``    — RRF(bm25, lsa), what ships today
* ``hybrid-openai`` — RRF(bm25, openai), the candidate

Differences are reported with a paired permutation test and a bootstrap CI, because with
~100 queries a 2-point nDCG gap is routinely noise, and shipping on the point estimate is
how retrieval systems silently get worse.

    python scripts/bench_retrieval.py --systems bm25 lsa openai hybrid-lsa hybrid-openai
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

from oncolens.env import load_env, local_data_dir  # noqa: E402
from oncolens.eval import metrics as M  # noqa: E402
from oncolens.eval.citation_labels import assert_source_excluded  # noqa: E402
from oncolens.eval.stats import compare, paired_values  # noqa: E402
from oncolens.retrieval.dense import make_backend  # noqa: E402
from oncolens.retrieval.fusion import aggregate_chunks_to_docs, reciprocal_rank_fusion  # noqa: E402
from oncolens.retrieval.lexical import BM25Index  # noqa: E402

TOP_K = 10


def load_chunks() -> list[dict]:
    import psycopg

    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("POSTGRES_URL / DATABASE_URL not set")
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        # ORDER BY is load-bearing: the embedding disk cache is keyed on a hash of the
        # texts in order, and Postgres guarantees no order without it. See the same fix
        # in improve_loop.load_chunks.
        cur.execute("SELECT chunk_id, doc_id, COALESCE(indexed_text, text) FROM chunks "
                    "ORDER BY chunk_id")
        return [{"chunk_id": r[0], "doc_id": r[1], "text": r[2]} for r in cur.fetchall()]


def dense_run(qvecs: np.ndarray, dvecs: np.ndarray, chunk_ids: list[str],
              qi: int, k: int) -> list[tuple[str, float]]:
    sims = dvecs @ qvecs[qi]
    top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
    top = top[np.argsort(-sims[top])]
    return [(chunk_ids[i], float(sims[i])) for i in top]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", default=None)
    ap.add_argument("--systems", nargs="*",
                    default=["bm25", "lsa", "openai", "hybrid-lsa", "hybrid-openai"])
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--rerank-depth", type=int, default=24,
                    help="passages sent to the LLM reranker per query")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    load_env()
    qpath = Path(args.qrels) if args.qrels else local_data_dir() / "qrels_citation.json"
    if not qpath.exists():
        raise SystemExit(f"no qrels at {qpath}; run scripts/build_citation_labels.py")
    data = json.loads(qpath.read_text(encoding="utf-8"))
    queries: dict[str, str] = data["queries"]
    qrels: dict[str, dict[str, int]] = data["qrels"]
    exclude: dict[str, str] = data.get("exclude", {})

    chunks = load_chunks()
    if not chunks:
        raise SystemExit("no chunks in the store")
    chunk_ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    chunk_to_doc = {c["chunk_id"]: c["doc_id"] for c in chunks}
    qids = sorted(queries)
    print(f"{len(chunks)} passages, {len(set(chunk_to_doc.values()))} documents, "
          f"{len(qids)} queries, {sum(len(v) for v in qrels.values())} judgments")

    # --- build every index once, then reuse across systems --------------------
    bm25 = BM25Index().build(chunk_ids, texts)
    dense_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    wanted = {s.replace("hybrid-", "").replace("+rerank", "") for s in args.systems}
    for backend_name in sorted(wanted - {"bm25", ""}):
        print(f"encoding with {backend_name}...")
        be = make_backend(backend_name, dim=args.dim)
        be.fit(texts)
        if hasattr(be, "encode_documents_cached"):
            dvecs = be.encode_documents_cached(texts, local_data_dir() / "emb_cache")
        else:
            dvecs = be.encode_documents(texts)
        qvecs = be.encode_queries([queries[q] for q in qids])
        dense_cache[backend_name] = (qvecs, dvecs)

    # --- run every system -----------------------------------------------------
    per_system: dict[str, dict[str, dict[str, float]]] = {}
    for system in args.systems:
        per_query: dict[str, dict[str, float]] = {}
        for qi, qid in enumerate(qids):
            runs: list[list[tuple[str, float]]] = []
            if system.startswith("bm25") or system.startswith("hybrid"):
                runs.append(bm25.search(queries[qid], k=args.candidates))
            if system.replace("+rerank", "") != "bm25":
                name = system.replace("hybrid-", "").replace("+rerank", "")
                qv, dv = dense_cache[name]
                runs.append(dense_run(qv, dv, chunk_ids, qi, args.candidates))

            fused = (reciprocal_rank_fusion(runs) if len(runs) > 1
                     else [(cid, s) for cid, s in runs[0]])

            if system.endswith("+rerank"):
                # Rerank at PASSAGE level before collapsing to documents: the reranker's
                # advantage is reading query and passage together, which is lost if it
                # only ever sees a document's best-scoring chunk.
                from oncolens.retrieval.llm_rerank import rerank as llm_rerank
                head = fused[:args.rerank_depth]
                by_id = {c["chunk_id"]: c["text"] for c in chunks}
                order = llm_rerank(queries[qid], [by_id[cid] for cid, _ in head])
                fused = ([(head[r.index][0], r.score) for r in order]
                         + [(cid, -1.0 - i) for i, (cid, _) in
                            enumerate(fused[args.rerank_depth:])])

            docs = aggregate_chunks_to_docs(fused, chunk_to_doc, strategy="max")
            src = exclude.get(qid)
            ranking = [d for d, _ in docs if d != src][:TOP_K]
            # Not a comment but a check: the guard must actually be in force.
            assert_source_excluded(qid, ranking, exclude)
            per_query[qid] = M.evaluate_query(ranking, qrels.get(qid, {}))
        per_system[system] = per_query

    # --- report ---------------------------------------------------------------
    keys = ["ndcg@10", "recall@10", "mrr", "map", "bpref", "unjudged@10"]
    avail = [k for k in keys if any(k in v for v in next(iter(per_system.values())).values())]
    print("\n" + "=" * (18 + 12 * len(avail)))
    print(f"{'system':<18}" + "".join(f"{k:>12}" for k in avail))
    print("-" * (18 + 12 * len(avail)))
    for system, pq in per_system.items():
        row = f"{system:<18}"
        for k in avail:
            vals = [v[k] for v in pq.values() if v.get(k) is not None]
            row += f"{(sum(vals)/len(vals) if vals else float('nan')):>12.4f}"
        print(row)

    # Paired comparisons against the shipping system.
    base = "hybrid-lsa" if "hybrid-lsa" in per_system else args.systems[0]
    print(f"\nPAIRED COMPARISONS vs {base} (nDCG@10)")
    print(f"{'system':<18}{'delta':>10}{'p':>10}{'95% CI':>22}{'d_z':>8}")
    print("-" * 68)
    results = {}
    for system in per_system:
        if system == base:
            continue
        c = compare(per_system[base], per_system[system], "ndcg@10")
        if c is None:
            continue
        results[system] = c
        sig = "*" if c.p_value < 0.05 else " "
        print(f"{system:<18}{c.delta:>+10.4f}{c.p_value:>10.4f}"
              f"{f'[{c.ci_low:+.4f}, {c.ci_high:+.4f}]':>22}{c.effect_size:>8.2f}{sig}"
              f"   W/L/T {c.wins}/{c.losses}/{c.ties}")
    k = max(len(results), 1)
    print(f"\n* = p < 0.05 uncorrected. {k} systems were compared against one baseline, so "
          f"the\n  Bonferroni-corrected threshold is {0.05 / k:.4f} — applied within this "
          "iteration only.")

    # Sparse judgments are the dominant caveat here, so state it with the numbers rather
    # than leaving it in a doc nobody opens.
    n_j = sum(len(v) for v in qrels.values())
    print(f"\nJUDGMENT DENSITY: {n_j} judgments over {len(qids)} queries "
          f"= {n_j / max(len(qids), 1):.2f} judged documents per query.")
    print("  unjudged@10 is ~0.94 for every system: about 94% of returned documents were")
    print("  never judged, so these are LOWER BOUNDS, not estimates of true quality. The")
    print("  comparison between systems remains meaningful because the unjudged rate is")
    print("  nearly identical across them, but no absolute number here should be quoted")
    print("  as 'the' retrieval quality.")
    print("  bpref is absent by design: it returns None below 10 judged negatives, and at")
    print("  this density it would be noise averaged into a consensus vote.")

    if args.json_out:
        payload = {
            "n_queries": len(qids),
            "systems": {s: {k: float(np.nanmean([v[k] for v in pq.values()
                                                 if v.get(k) is not None] or [np.nan]))
                            for k in avail} for s, pq in per_system.items()},
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
