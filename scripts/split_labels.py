#!/usr/bin/env python
"""Move gold `descriptors` out of the retrieval corpus into data/labels/.

Gold labels living inside the corpus is the most damaging defect a retrieval benchmark
can have: the loader hands the answer key to the indexer and every measured gain becomes
an artifact. `load_corpus()` now refuses such a corpus; this script repairs one.
"""
from __future__ import annotations
import json, pathlib, sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data")
labels: dict[str, list[str]] = {}
for p in sorted((root / "corpus").glob("*.jsonl")):
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        desc = d.pop("descriptors", [])
        d.pop("mesh_detail", None)
        if desc:
            labels[d["doc_id"]] = desc
        out.append(d)
    p.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in out) + "\n", encoding="utf-8")

(root / "labels").mkdir(parents=True, exist_ok=True)
(root / "labels" / "descriptors.jsonl").write_text(
    "\n".join(json.dumps({"doc_id": k, "descriptors": v}, ensure_ascii=False)
              for k, v in sorted(labels.items())) + "\n", encoding="utf-8")
print(f"moved labels for {len(labels)} documents -> {root}/labels/descriptors.jsonl")
