#!/usr/bin/env python
"""Type a figure by MEANING rather than by keyword, and prove it is not just the keyword.

**Why the regex has to go, measured on the 4,783 unlabelled figures.** The caption pattern
list reads only the caption and only matches literal words. Of what it misses:

  7.7%   the sentences CITING the figure name a type the caption never does
  37.3%  the caption paraphrases it ("sections were stained", "quantified as")
  2.3%   the caption is too short to say anything

Only the last needs pixels. The 37.3% is the case that retires the approach: paraphrase is
unbounded, so more patterns is a treadmill, and §4.14 already recorded what happens when a
pattern is asked to decide something its shape cannot carry.

**The method.** Captions the regex DOES match are near-certain ground truth: the publisher
literally wrote "Kaplan-Meier". Embed those, build one centroid per type, and classify the
rest by nearest centroid. No training, no labels invented by us, no model grading its own
homework (§4.4) — the supervision is the publisher's own words.

**The control that makes it honest.** A classifier trained on captions containing the word
"Kaplan-Meier" could simply be detecting that word, in which case it adds nothing over the
regex. So the held-out set is evaluated TWICE: once intact, and once with the matched
keyword DELETED. Accuracy on the masked set is the only number that means anything, because
it is the only one the regex could not also achieve.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

from oncolens.env import load_env  # noqa: E402

#: Below this cosine margin the figure stays NULL. Guessing is worse than admitting
#: ignorance — the page renders an unknown type as nothing, and a wrong type reads as
#: though the paper said it (§4.15).
MIN_MARGIN = 0.02


def mask_keyword(caption: str, patterns) -> str:
    """Delete the literal words the regex would have matched."""
    out = caption
    for _, pat in patterns:
        out = re.sub(pat, " ", out, flags=re.I)
    return " ".join(out.split())


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="write inferred types back (tagged figure_type_src='caption-semantic')")
    ap.add_argument("--min-per-type", type=int, default=40)
    args = ap.parse_args()

    load_env(REPO)
    import psycopg

    import ingest_figures as ing
    from oncolens.retrieval.dense import make_backend

    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    with psycopg.connect(dsn) as cx, cx.cursor() as cur:
        cur.execute("""SELECT chunk_id, doc_id, text, COALESCE(indexed_text, text),
                              figure_type
                       FROM chunks WHERE kind = 'figure'""")
        rows = cur.fetchall()
    print(f"figures: {len(rows):,}")

    # ---- ground truth: the publisher named the type ------------------------------------
    labelled = [(c, d, cap, t) for c, d, cap, _idx, t in rows if t]
    unlabelled = [(c, d, cap, idx) for c, d, cap, idx, t in rows if not t]
    print(f"  publisher-stated (regex fires): {len(labelled):,}")
    print(f"  unlabelled                    : {len(unlabelled):,}")

    keep = {t for t, n in Counter(t for *_x, t in labelled).items()
            if n >= args.min_per_type}
    labelled = [r for r in labelled if r[3] in keep]
    print(f"  types with >= {args.min_per_type} examples: {len(keep)} {sorted(keep)}")

    # ---- split BY DOCUMENT: figures from one paper share house style -------------------
    docs = sorted({d for _c, d, *_ in labelled})
    import hashlib
    test_docs = {d for d in docs
                 if int(hashlib.sha1(d.encode()).hexdigest()[:8], 16) % 100 < 25}
    train = [r for r in labelled if r[1] not in test_docs]
    test = [r for r in labelled if r[1] in test_docs]
    print(f"  train {len(train):,} / test {len(test):,} (split by document, not by figure)")

    be = make_backend("openai", dim=192)

    def embed(texts):
        v = np.asarray(be.encode_documents(list(texts)), dtype=np.float32)
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)

    # ---- prototypes: one centroid per type ---------------------------------------------
    print("\nembedding training captions ...")
    tr_vec = embed([cap for _c, _d, cap, _t in train])
    proto_types = sorted(keep)
    protos = []
    for t in proto_types:
        idx = [i for i, r in enumerate(train) if r[3] == t]
        c = tr_vec[idx].mean(axis=0)
        protos.append(c / np.linalg.norm(c))
    P = np.vstack(protos)

    def classify(vecs):
        sims = vecs @ P.T
        best = sims.argmax(axis=1)
        srt = np.sort(sims, axis=1)
        margin = srt[:, -1] - srt[:, -2]
        return [proto_types[b] for b in best], margin

    # ---- the two evaluations. Only the second one means anything. ----------------------
    print("embedding held-out captions ...")
    te_intact = embed([cap for _c, _d, cap, _t in test])
    te_masked = embed([mask_keyword(cap, ing.FIGURE_TYPES) for _c, _d, cap, _t in test])
    gold = [r[3] for r in test]

    for name, vecs in (("intact caption", te_intact), ("KEYWORD MASKED", te_masked)):
        pred, margin = classify(vecs)
        ok = sum(p == g for p, g in zip(pred, gold))
        conf = margin >= MIN_MARGIN
        ok_conf = sum(p == g for p, g, c in zip(pred, gold, conf) if c)
        n_conf = int(conf.sum())
        print(f"\n=== {name} ===")
        print(f"  accuracy            : {ok}/{len(test)} = {ok/len(test):.1%}")
        print(f"  above margin {MIN_MARGIN}   : {n_conf}/{len(test)} = {n_conf/len(test):.1%}"
              f"  accuracy {ok_conf/max(n_conf,1):.1%}")
        if name.startswith("KEYWORD"):
            per = defaultdict(lambda: [0, 0])
            for p, g in zip(pred, gold):
                per[g][1] += 1
                per[g][0] += (p == g)
            print("  per type:")
            for t in sorted(per, key=lambda k: -per[k][1]):
                h, n = per[t]
                print(f"    {t:<16} {h:>3}/{n:<4} {h/n:>6.0%}")

    # ---- the gold label is not actually single-valued ----------------------------------
    #
    # `classify()` in the ingest returns the FIRST pattern that fires, but 9.8% of these
    # figures match several — a composite panel really is both a western blot and a bar
    # chart. Scoring against one arbitrary member of that set penalises a prediction that
    # is correct, so accuracy against the full matching set is the fair number. Reported
    # alongside, never instead: the single-label figure is what a page would actually show.
    gold_sets = []
    for _c, _d, cap, _t in test:
        s = {name for name, pat in ing.FIGURE_TYPES
             if re.search(pat, cap, re.I) and name in keep}
        gold_sets.append(s or {_t})
    multi = sum(1 for s in gold_sets if len(s) > 1)
    print(f"\n  held-out captions matching MORE THAN ONE type: {multi}/{len(test)} "
          f"({multi/len(test):.1%}) — these have no single right answer")

    pred_m, margin_m = classify(te_masked)
    lenient = sum(p in s for p, s in zip(pred_m, gold_sets)) / len(test)
    print(f"  masked accuracy against ANY valid label: {lenient:.1%} "
          f"(vs {sum(p == g for p, g in zip(pred_m, gold))/len(test):.1%} single-label)")

    masked_acc = sum(p == g for p, g in zip(pred_m, gold)) / len(test)
    print(f"\n  The regex scores 0% on the masked set BY CONSTRUCTION — its keyword is "
          f"gone.\n  Semantic classification recovers {masked_acc:.1%}, which is the "
          f"increment it\n  actually buys over pattern matching.")

    # ---- precision/coverage curve ------------------------------------------------------
    #
    # 63% precision means a third of applied labels are wrong, and a wrong type renders
    # beside a real figure as though it were a property of the paper. The right shape here
    # is HIGH PRECISION, LOW RECALL — the same trade §3.12 argued for chart-to-table:
    # when the failure mode is poisoning a display, an unreliable labeller is only usable
    # with a confidence gate in front of it.
    print("\n=== precision / coverage on the MASKED set (this is what sets the threshold) ===")
    print(f"  {'margin':>8}{'labelled':>10}{'coverage':>10}{'precision':>11}")
    chosen = None
    for thr in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20):
        sel = margin_m >= thr
        n = int(sel.sum())
        if not n:
            continue
        hits = sum(p == g for p, g, s in zip(pred_m, gold, sel) if s)
        prec = hits / n
        print(f"  {thr:>8.2f}{n:>10}{n/len(test):>9.1%}{prec:>11.1%}")
        if chosen is None and prec >= 0.80:
            chosen = (thr, n / len(test), prec)
    if chosen:
        print(f"\n  -> {chosen[0]:.2f} is the lowest margin reaching 80% precision "
              f"({chosen[1]:.1%} coverage). Use it.")
    else:
        print("\n  -> NO threshold reaches 80% precision. Applying this would put a wrong "
              "type\n     on roughly one figure in three, which is worse than leaving them "
              "unknown.")

    if not args.apply:
        print("\n(dry run — pass --apply to write inferred types)")
        return 0

    # ---- apply to the unlabelled, using caption + citing sentences ---------------------
    print(f"\nclassifying {len(unlabelled):,} unlabelled figures ...")
    un_vec = embed([idx[:1500] for _c, _d, _cap, idx in unlabelled])
    pred, margin = classify(un_vec)
    writes = [(p, c) for (c, _d, _cap, _i), p, m in zip(unlabelled, pred, margin)
              if m >= MIN_MARGIN]
    print(f"  confident enough to label: {len(writes):,} "
          f"({len(writes)/max(len(unlabelled),1):.1%}); the rest stay NULL")
    print("  " + ", ".join(f"{t}:{n}" for t, n in Counter(p for p, _ in writes).most_common(8)))

    with psycopg.connect(dsn) as cx, cx.cursor() as cur:
        cur.executemany(
            "UPDATE chunks SET figure_type=%s, figure_type_src='caption-semantic' "
            "WHERE chunk_id=%s", writes)
        cx.commit()
    print(f"\nwrote {len(writes):,} inferred types, tagged 'caption-semantic' so the page "
          f"renders them as inferred rather than stated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
