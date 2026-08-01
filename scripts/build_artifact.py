#!/usr/bin/env python
"""Build the deployable search artifact. Run offline, commit or upload the output."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oncolens.data import load_dataset                      # noqa: E402
from oncolens.loop import load_champion                     # noqa: E402
from oncolens.configs import BASELINE                       # noqa: E402
from oncolens.retrieval.pipeline import RetrievalConfig     # noqa: E402
from oncolens.serve.artifact import build_artifact          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="artifact")
ap.add_argument("--context-mode", default="gist", choices=["none", "structural", "gist"])
args = ap.parse_args()

ds = load_dataset(strict=False)
saved = load_champion()
cfg = RetrievalConfig(**saved["config"]) if saved else BASELINE
print(f"building artifact with config: {cfg.name}")

lex_path = REPO / "data" / "vocab" / "lexicon.json"
lexicon = json.loads(lex_path.read_text(encoding="utf-8")) if lex_path.exists() else {}

stats = build_artifact(ds.docs, cfg, out_dir=Path(args.out),
                       context_mode=args.context_mode, lexicon=lexicon)
for k, v in stats.items():
    print(f"  {k:<24} {v}")
print(f"\nartifact -> {args.out}/  (deploy this directory alongside api/)")
