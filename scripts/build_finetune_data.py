#!/usr/bin/env python
"""Mine (query, passage, label) triples to fine-tune a reranker on this corpus's own task.

**Why this is the highest-ceiling lever.** Measured (§4.18): the correct paper is in the
retrieved pool **96.2%** of the time and ranked first **39.0%** of the time. That 0.57 gap is
ranking, not recall, and no fusion weight closes it. `rerank_medcpt_cross` — a *general*
biomedical reranker — already bought +0.0441 mrr. This corpus has thousands of labelled
query→answer pairs, so a reranker trained on the exact task is the obvious next step.

**Hard negatives are the whole point.** Training against random passages teaches a model to
separate oncology from cooking, which BM25 already does. The useful signal is the passages
the *current system already ranks highly and gets wrong*, so negatives are sampled from the
top of the live fused ranking, excluding the judged document.

⚠️ **Leakage discipline.** Queries come from `ft_train`, which partitions DEV only. The model
is scored on `ft_holdout`, and the locked `test` split is touched by neither. Verified as an
exact partition with zero overlap.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from oncolens.env import load_env, local_data_dir  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stratum", default="claim")
    ap.add_argument("--negatives", type=int, default=4,
                    help="hard negatives per positive")
    ap.add_argument("--pool", type=int, default=30,
                    help="depth of the fused list negatives are drawn from")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    load_env(REPO)
    import improve_loop as L

    queries, qrels, exclude = L.load_qrels("ft_train", args.stratum)
    print(f"ft_train {args.stratum}: {len(queries):,} queries, "
          f"{sum(len(v) for v in qrels.values()):,} judgments")

    chunks, index_config = L.load_chunks(with_embeddings=True)
    print(f"chunks: {len(chunks):,}")
    h = L.Harness(chunks, queries, dim=192, index_config=index_config)

    # Baseline configuration: negatives must be what the SHIPPED system confuses, not what
    # some other configuration confuses.
    cfg = {"bm25_weight": 1.0, "dense_weight": 1.0, "candidates": 200, "aggregate": "max"}

    # Reach into the fused chunk list rather than the document ranking: a cross-encoder
    # scores passages, so its negatives must be passages.
    print("building the retrieval pool for hard-negative mining ...")
    pools = h.fused_chunks(cfg)

    by_id = {c["chunk_id"]: c["text"] for c in chunks}
    chunk_doc = {c["chunk_id"]: c["doc_id"] for c in chunks}

    rng = random.Random(17)
    rows, n_pos, n_neg, skipped = [], 0, 0, 0
    for qid, qtext in queries.items():
        rel = {d for d, g in (qrels.get(qid) or {}).items() if g > 0}
        if not rel:
            continue
        pool = pools.get(qid) or []
        src = exclude.get(qid)

        pos = [c for c in pool if chunk_doc.get(c) in rel]
        if not pos:
            skipped += 1          # the answer is not in the pool; nothing to learn here
            continue
        # Hard negatives: high-ranked, wrong document, and never the citing paper itself
        # (§4.4 — the source contains the query verbatim and would teach string equality).
        neg = [c for c in pool[: args.pool]
               if chunk_doc.get(c) not in rel and chunk_doc.get(c) != src]
        if not neg:
            skipped += 1
            continue
        rng.shuffle(neg)

        rows.append({"query": qtext, "passage": by_id[pos[0]], "label": 1, "qid": qid})
        n_pos += 1
        for c in neg[: args.negatives]:
            rows.append({"query": qtext, "passage": by_id[c], "label": 0, "qid": qid})
            n_neg += 1

    out = Path(args.out) if args.out else local_data_dir() / "finetune_pairs.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print(f"\n  positives      : {n_pos:,}")
    print(f"  hard negatives : {n_neg:,}")
    print(f"  queries skipped (answer absent from pool): {skipped:,} "
          f"({skipped/max(len(queries),1):.1%})")
    print(f"  wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print("\n  NOTE: skipped queries are the recall failures. A reranker cannot fix those, "
          "which is why 3.8% never-retrieved is the hard ceiling (4.18).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
