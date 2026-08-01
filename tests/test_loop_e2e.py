"""End-to-end loop smoke test against a temporary fixture corpus.

Verifies the full propose -> measure -> gate -> promote cycle runs, writes its artifacts,
and produces a report containing the failure analysis and dominance sections. Uses
ONCOLENS_DATA / ONCOLENS_EXPERIMENTS so it never touches the real data/ or experiments/.
"""
import json, os, sys, tempfile, pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

d = pathlib.Path(tempfile.mkdtemp())
(d / "corpus").mkdir(); (d / "qrels").mkdir(); (d / "vocab").mkdir()
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
check("dev and test splits both populated",
      len(ds.split("dev")) > 0 and len(ds.split("test")) > 0,
      f"dev={len(ds.split('dev'))} test={len(ds.split('test'))}")

out = run_iteration(1, BASELINE, iteration_1_arms(), dataset=ds)
check("iteration produced gate decisions for every challenger", len(out.gates) == 3)
check("report includes failure analysis", "Failure analysis" in out.report)
check("report includes dominance bounds", "dominates" in out.report or "UNDETERMINED" in out.report)
check("iteration report written to disk", (d / "experiments" / "iteration_01.md").exists())
check("ledger recorded every draw",
      len(json.loads((d / "experiments" / "ledger.json").read_text())["entries"]) == 4,
      "champion + 3 challengers")

print()
if failures:
    print("FAILED: " + ", ".join(failures)); sys.exit(1)
print("loop end-to-end OK")
