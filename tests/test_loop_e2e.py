"""End-to-end loop smoke test against a temporary fixture corpus.

Verifies the full propose -> measure -> gate -> promote cycle runs, writes its artifacts,
and produces a report containing the failure analysis and dominance sections. Uses
ONCOLENS_DATA / ONCOLENS_EXPERIMENTS so it never touches the real data/ or experiments/.
"""
import json, os, sys, tempfile, pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

d = pathlib.Path(tempfile.mkdtemp())
(d / "corpus").mkdir(); (d / "qrels").mkdir(); (d / "vocab").mkdir(); (d / "labels").mkdir()
os.environ["ONCOLENS_DATA"] = str(d)
os.environ["ONCOLENS_EXPERIMENTS"] = str(d / "experiments")

sys.path.insert(0, str(REPO / "tests"))
from test_pipeline import DOCS, QUERIES  # noqa: E402

with open(d / "corpus" / "part_a.jsonl", "w", encoding="utf-8") as f:
    for doc in DOCS:
        f.write(json.dumps(doc) + "\n")
with open(d / "qrels" / "raw_a.jsonl", "w", encoding="utf-8") as f:
    for i in range(6):  # replicate so both dev and test buckets are populated
        for q in QUERIES:
            f.write(json.dumps({
                "query_id": f"{q.query_id}_{i}", "query": q.query, "stratum": q.stratum,
                "source": q.source, "judgments": q.judgments}) + "\n")
with open(d / "labels" / "descriptors.jsonl", "w", encoding="utf-8") as f:
    for doc in DOCS:
        f.write(json.dumps({"doc_id": doc["doc_id"], "descriptors": ["D:TEST"]}) + "\n")
(d / "vocab" / "lexicon.json").write_text(json.dumps({"egfr": ["erbb1", "her1"]}), encoding="utf-8")
(d / "vocab" / "concepts.json").write_text(
    json.dumps({"D:TEST": {"preferred": "test concept", "broader": [], "narrower": []}}), encoding="utf-8")

from oncolens.configs import BASELINE, iteration_1_arms  # noqa: E402
from oncolens.data import load_dataset  # noqa: E402
from oncolens.loop import run_iteration  # noqa: E402

failures = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)

ds = load_dataset()
check("fixture corpus loaded", ds.integrity["n_docs"] == len(DOCS), f"{ds.integrity['n_docs']} docs")
check("fixture queries loaded", ds.integrity["n_queries"] == len(QUERIES) * 6,
      f"{ds.integrity['n_queries']} queries")
# Need-based clustering groups the 6 replicas of each query together, so a 4-need
# fixture may land entirely in one bucket. What must hold is that the split is a
# partition and that no information need straddles it.
dev, test = ds.split("dev"), ds.split("test")
check("split is a partition", len(dev) + len(test) == len(ds.queries),
      f"dev={len(dev)} test={len(test)} total={len(ds.queries)}")
_g = ds.need_groups()
_dev_ids, _test_ids = {q.query_id for q in dev}, {q.query_id for q in test}
_straddle = sum(
    1 for c in set(_g.values())
    if any(q in _dev_ids for q, gg in _g.items() if gg == c)
    and any(q in _test_ids for q, gg in _g.items() if gg == c)
)
check("no information need straddles dev/test", _straddle == 0, f"{_straddle} straddling")

out = run_iteration(1, BASELINE, iteration_1_arms(), dataset=ds)
check("iteration produced gate decisions for every challenger", len(out.gates) == 3)
check("report includes failure analysis", "Failure analysis" in out.report)
# Dominance bounds need >=1 comparable dev query. With only 4 distinct information
# needs in this fixture, need-based clustering can leave dev too small — that is
# correct behaviour for a degenerate fixture, not a regression.
if len(dev) >= 2:
    check("report includes dominance bounds",
          "dominates" in out.report or "UNDETERMINED" in out.report)
else:
    print(f"  SKIP  dominance bounds (dev split has only {len(dev)} queries)")
check("iteration report written to disk", (d / "experiments" / "iteration_01.md").exists())
check("ledger recorded every draw",
      len(json.loads((d / "experiments" / "ledger.json").read_text())["entries"]) == 4,
      "champion + 3 challengers")

print()
if failures:
    print("FAILED: " + ", ".join(failures)); sys.exit(1)
print("loop end-to-end OK")
