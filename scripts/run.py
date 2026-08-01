#!/usr/bin/env python
"""OncoLens CLI.

    python scripts/run.py validate      # dataset integrity + leakage + contamination audit
    python scripts/run.py iterate       # run the full improvement ladder against dev
    python scripts/run.py test          # open the locked test split, once
    python scripts/run.py demo "query"  # show ranked docs plus the supporting passage
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oncolens import configs as C  # noqa: E402
from oncolens.contexts import assert_no_label_leakage, build_contexts  # noqa: E402
from oncolens.data import load_dataset  # noqa: E402
from oncolens.experiment import build_retriever  # noqa: E402
from oncolens.loop import EXPERIMENTS, final_test_evaluation, load_champion, run_iteration  # noqa: E402
from oncolens.retrieval.chunking import chunk_corpus  # noqa: E402
from oncolens.retrieval.expansion import Lexicon, contamination_report  # noqa: E402
from oncolens.retrieval.pipeline import RetrievalConfig  # noqa: E402
from oncolens.sanity import sanity_report  # noqa: E402

VOCAB = REPO / "data" / "vocab"
RETRIEVAL_SOURCES = [
    str(p) for p in (REPO / "src" / "oncolens" / "retrieval").glob("*.py")
] + [str(REPO / "src" / "oncolens" / "contexts.py")]


def cmd_validate() -> int:
    print("=" * 72)
    print("DATASET INTEGRITY")
    print("=" * 72)
    ds = load_dataset(strict=False)
    for k, v in ds.integrity.items():
        print(f"  {k:<28} {v}")

    problems: list[str] = []
    if ds.integrity["n_dangling_judgments"]:
        problems.append(
            f"{ds.integrity['n_dangling_judgments']} judgments reference nonexistent doc_ids"
        )
    sr = ds.integrity["single_relevant_queries"]
    if sr:
        problems.append(
            f"{sr} queries have exactly ONE relevant doc — these systematically penalise "
            f"better systems and should be pooled/expanded"
        )
    if ds.integrity["queries_with_explicit_zeros"] == 0:
        problems.append("no explicit 0 judgments anywhere — bpref carries no signal")

    print()
    print("=" * 72)
    print("STRATUM BALANCE + STATISTICAL POWER")
    print("=" * 72)
    from collections import Counter
    for split in ("dev", "test"):
        counts = Counter(q.stratum for q in ds.split(split))
        print(f"  {split}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        for stratum, n in sorted(counts.items()):
            if n < 15:
                problems.append(
                    f"stratum '{stratum}' has only {n} {split} queries — underpowered; "
                    f"a per-stratum regression gate here is weak evidence"
                )

    print()
    print("=" * 72)
    print("LABEL LEAKAGE AUDIT (retrieval path must never read gold descriptors)")
    print("=" * 72)
    violations = assert_no_label_leakage(RETRIEVAL_SOURCES)
    if violations:
        for v in violations:
            print(f"  LEAK  {v}")
        problems.append(f"{len(violations)} label-leakage violations in the retrieval path")
    else:
        print("  clean — no retrieval-path source reads 'descriptors'")

    print()
    print("=" * 72)
    print("ONTOLOGY CONTAMINATION (lexicon vs gold concept space)")
    print("=" * 72)
    rep = contamination_report(VOCAB / "lexicon.json", VOCAB / "concepts.json")
    for k, v in rep.items():
        print(f"  {k:<30} {v if not isinstance(v, float) else f'{v:.4f}'}")
    if rep["concept_coverage_by_lexicon"] > 0.6:
        problems.append(
            f"lexicon covers {rep['concept_coverage_by_lexicon']:.1%} of gold-concept vocabulary — "
            f"expansion gains are substantially artifactual and must be discounted"
        )

    print()
    print("=" * 72)
    print("CHUNKING")
    print("=" * 72)
    chunks = chunk_corpus(ds.docs)
    lens = sorted(len(c.text) for c in chunks)
    print(f"  chunks                       {len(chunks)}")
    print(f"  chars  min/median/p95/max    {lens[0]}/{lens[len(lens)//2]}/{lens[int(len(lens)*0.95)]}/{lens[-1]}")
    bad = [c for c in chunks if c.end_char <= c.start_char]
    print(f"  invalid offsets              {len(bad)}")
    if bad:
        problems.append(f"{len(bad)} chunks have invalid offsets — provenance is broken")

    print()
    print("=" * 72)
    print("BENCHMARK EXPLOITABILITY (degenerate baselines bracket every result)")
    print("=" * 72)
    for split in ("dev", "test"):
        rep = sanity_report(ds, split=split)
        line = "  ".join(f"{k}={v}" for k, v in rep["primary"].items())
        print(f"  {split}: {line}")
        print(f"         headroom above popularity = {rep['headroom_above_popularity']}")
        for p in rep["problems"]:
            print(f"         ! {p}")
            problems.append(f"[{split}] {p}")

    print()
    print("=" * 72)
    if problems:
        print(f"VALIDATION FOUND {len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  ! {p}")
        return 1
    print("VALIDATION CLEAN")
    return 0


def _contexts(ds, mode: str):
    return build_contexts(ds.docs, chunk_corpus(ds.docs), mode=mode)


def cmd_iterate(max_iterations: int | None = None) -> int:
    ds = load_dataset()
    lex = Lexicon.load(VOCAB / "lexicon.json")
    ctx = _contexts(ds, "gist")

    saved = load_champion()
    champion = RetrievalConfig(**saved["config"]) if saved else C.BASELINE
    print(f"starting champion: {champion.name}")

    ladder = C.LADDER if max_iterations is None else C.LADDER[:max_iterations]
    for i, (label, builder) in enumerate(ladder, start=1):
        try:
            challengers = builder() if builder is C.iteration_1_arms else builder(champion)
        except TypeError:
            challengers = builder(champion)
        print(f"\n{'='*72}\nITERATION {i}: {label}  ({len(challengers)} challengers)\n{'='*72}")
        out = run_iteration(i, champion, challengers, dataset=ds, lexicon=lex, contexts=ctx)
        print(out.report)
        if out.promoted:
            champion = next(c for c in challengers if c.name == out.champion_after)
            print(f">>> champion is now {champion.name}")
    return 0


def cmd_test() -> int:
    ds = load_dataset()
    lex = Lexicon.load(VOCAB / "lexicon.json")
    ctx = _contexts(ds, "gist")
    saved = load_champion()
    if not saved:
        print("no champion recorded — run `iterate` first")
        return 1
    champion = RetrievalConfig(**saved["config"])
    out = final_test_evaluation(champion, C.BASELINE, dataset=ds, lexicon=lex, contexts=ctx)
    (EXPERIMENTS / "test_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


def cmd_demo(query: str) -> int:
    ds = load_dataset(strict=False)
    lex = Lexicon.load(VOCAB / "lexicon.json")
    saved = load_champion()
    cfg = RetrievalConfig(**saved["config"]) if saved else C.BASELINE
    r = build_retriever(ds, cfg, lexicon=lex, contexts=_contexts(ds, "gist"))
    res = r.search(query)
    print(f"query: {query!r}\nconfig: {cfg.name}\n")
    titles = {d["doc_id"]: d.get("title", "") for d in ds.docs}
    for rank, doc in enumerate(res.ranking[:8], start=1):
        print(f"{rank}. {doc}  [{res.scores[doc]:.4f}]")
        print(f"   {titles.get(doc,'')[:100]}")
        ev = res.evidence.get(doc)
        if ev:
            snippet = ev.text.replace("\n", " ")[:240]
            print(f"   passage [{ev.section} chars {ev.start_char}-{ev.end_char}]: {snippet}...")
        print()
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if cmd == "validate":
        sys.exit(cmd_validate())
    elif cmd == "iterate":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else None
        sys.exit(cmd_iterate(n))
    elif cmd == "test":
        sys.exit(cmd_test())
    elif cmd == "demo":
        sys.exit(cmd_demo(sys.argv[2] if len(sys.argv) > 2 else "EGFR C797S resistance"))
    else:
        print(__doc__)
        sys.exit(2)
