"""How deep must you look before the right paper is in the pool?

The benchmark reports recall@10 = 0.6806 and stops. That number alone cannot say where to
spend effort, because two very different systems produce it:

  * recall@200 also ~0.68 -> first-stage retrieval never finds the paper. No reranker helps;
    the fix is candidate generation.
  * recall@200 ~0.95      -> the paper IS retrieved, just ranked badly. Reranking has ~0.27
    of headroom and is the whole answer.

`evaluate_query` already records `first_rel_rank` per query, so the entire recall curve and
the never-found rate fall out of a single baseline run.
"""
import sys, time, collections
from pathlib import Path

ROOT = Path(r"c:\Users\jcl34\OneDrive\Documents\GitHub\oncolens-1")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from oncolens.env import load_env
load_env(ROOT)
import improve_loop as L

# NOTE the order: load_qrels returns (queries, qrels, exclude). Unpacking it the other
# way round silently "worked" and reported 951,623 judgments — which was the total
# CHARACTER COUNT of 5,000 claim sentences. A wrong unpack of a same-typed tuple does not
# raise; it just makes every downstream number nonsense.
queries, qrels, exclude = L.load_qrels("dev", "claim")
print(f"claim dev: {len(queries)} queries, {sum(len(v) for v in qrels.values())} judgments")

# improve_loop truncates every ranking to TOP_K=20 documents before scoring. That is the
# right default for the metric panel (which tops out at k=20) and it makes any deeper
# recall measurement meaningless: the first run of this script reported recall@1000 ==
# recall@20 == 0.7638 and "23.6% never retrieved at any depth", when the truth was
# "23.6% not in the top 20". The tell was a perfectly flat curve past 20 and EXACTLY zero
# answers at rank 51+ — a cliff at a round number is a limit in the code, not in the data.
L.TOP_K = 2000

chunks, index_config = L.load_chunks(with_embeddings=True)
print(f"chunks: {len(chunks):,}  index_config={index_config}")
h = L.Harness(chunks, queries, dim=192, index_config=index_config)

# Deep candidate pool so the rank distribution is not truncated by the pool itself.
cfg = {"bm25_weight": 1.0, "dense_weight": 1.0, "candidates": 1000, "aggregate": "max"}
print(f"config: {cfg}\nrunning...")
t0 = time.perf_counter()
per_q = h.run(cfg, qrels, exclude)
print(f"done in {time.perf_counter()-t0:.0f}s")

ranks = []
never = 0
for qid, m in per_q.items():
    if not (qrels.get(qid) or {}):
        continue
    r = m.get("first_rel_rank")
    if r is None:
        never += 1
    else:
        ranks.append(int(r))
n = len(ranks) + never
print(f"\nqueries with a judgment: {n}")

print(f"\n=== recall@k (claim, 1 relevant doc per query, so recall == success) ===")
prev = 0.0
for d in (1, 3, 5, 10, 20, 50, 100, 200, 500, 1000):
    hit = sum(1 for r in ranks if r <= d)
    rec = hit / n
    print(f"  recall@{d:<5} {rec:.4f}    +{rec-prev:.4f}")
    prev = rec

print(f"\n  never retrieved at any depth : {never}/{n} = {never/n:.1%}")
print(f"  <- this is the HARD ceiling. No reranker, at any depth, can recover these.")

top50 = sum(1 for r in ranks if r <= 50) / n
top10 = sum(1 for r in ranks if r <= 10) / n
top1 = sum(1 for r in ranks if r == 1) / n
print(f"\n  reranking headroom over the top 50 : {top50:.4f} - {top1:.4f} = {top50-top1:.4f}")
print(f"  (the cross-encoder converted +0.0458 of that into success@1)")

print("\n=== where the misses live ===")
buckets = collections.Counter()
for r in ranks:
    for lo, hi, name in ((1, 1, "rank 1"), (2, 5, "2-5"), (6, 10, "6-10"),
                         (11, 50, "11-50"), (51, 200, "51-200"), (201, 10**9, "201+")):
        if lo <= r <= hi:
            buckets[name] += 1
            break
buckets["never found"] = never
for name in ("rank 1", "2-5", "6-10", "11-50", "51-200", "201+", "never found"):
    c = buckets.get(name, 0)
    bar = "#" * int(60 * c / n)
    print(f"  {name:<12} {c:>5}  {c/n:>6.1%}  {bar}")
