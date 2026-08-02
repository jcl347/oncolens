#!/usr/bin/env python
"""Paired comparison of the old and new reference detectors on identical articles.

Comparing two detectors on two different samples proves nothing — which is how the old
detector came to look like it worked. This loads the previous implementation straight out
of git and runs both over the *same* cached, publisher-labelled articles, so the difference
is attributable to the code and not to which articles happened to be fetched.

    python scripts/compare_ref_detectors.py --ref HEAD
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bench_references import true_boundary  # noqa: E402
from oncolens.env import load_env, local_data_dir  # noqa: E402
from oncolens.retrieval import references as new_refs  # noqa: E402
from oncolens.sources import jats  # noqa: E402


def load_old(ref: str):
    """Import the committed version of references.py as a separate module."""
    src = subprocess.run(
        ["git", "show", f"{ref}:src/oncolens/retrieval/references.py"],
        capture_output=True, text=True, cwd=ROOT, encoding="utf-8",
    )
    if src.returncode != 0:
        raise SystemExit(f"git show failed: {src.stderr[:200]}")
    tmp = Path(tempfile.mkdtemp()) / "old_references.py"
    tmp.write_text(src.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("old_references", tmp)
    mod = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations via sys.modules[cls.__module__]; registering the
    # module before executing it is required or the decorator raises on a missing entry.
    sys.modules["old_references"] = mod
    spec.loader.exec_module(mod)
    return mod


def boundary(mod, txt: str) -> int | None:
    paras = txt.split("\n\n")
    idx = mod.find_reference_start(paras)
    if idx is None:
        return None
    # Include the separator that precedes the bibliography, so the offset is comparable
    # with the XML-derived boundary rather than 2 characters short of it.
    return len("\n\n".join(paras[:idx])) + (2 if idx else 0)


def score(txt: str, tb: int, pb: int | None) -> tuple[float, float]:
    total = len(txt)
    true_body, true_refs = tb, total - tb
    if pb is None:
        return 1.0, 0.0
    body_kept = min(pb, tb) / true_body if true_body else 1.0
    refs_dropped = min((total - pb) / true_refs, 1.0) if pb >= tb else 1.0
    return body_kept, refs_dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD", help="git ref holding the old detector")
    args = ap.parse_args()

    load_env()
    cache = local_data_dir() / "jats_cache"
    old = load_old(args.ref)

    pairs = []
    for tf in sorted(cache.glob("*.txt")):
        xf = tf.with_suffix(".xml")
        if not xf.exists():
            continue
        txt = tf.read_text(encoding="utf-8", errors="replace")
        ref_text = jats.reference_list_text(xf.read_text(encoding="utf-8", errors="replace"))
        if not ref_text or len(ref_text) < 400:
            continue
        tb = true_boundary(txt, ref_text)
        if tb is None or tb < 1000:
            continue
        pairs.append((tf.stem, txt, tb))

    if not pairs:
        raise SystemExit("no cached labelled articles; run bench_references.py first")

    print(f"paired comparison on {len(pairs)} publisher-labelled articles\n")
    agg = {}
    for name, mod in (("old", old), ("new", new_refs)):
        det = bk = rd = clean = harmed = 0
        for _, txt, tb in pairs:
            pb = boundary(mod, txt)
            b, r = score(txt, tb, pb)
            det += pb is not None
            bk += b
            rd += r
            clean += (r > 0.9 and b > 0.995)
            harmed += (b < 0.995)
        n = len(pairs)
        agg[name] = dict(detected=det, body_kept=bk / n, refs_dropped=rd / n,
                         clean=clean, harmed=harmed, n=n)

    n = len(pairs)
    print(f"{'metric':<26}{'old':>12}{'new':>12}{'delta':>12}")
    print("-" * 62)
    for key, label, pct in (("detected", "bibliography detected", True),
                            ("clean", "fully correct", True),
                            ("harmed", "BODY DAMAGED", True)):
        o, nv = agg["old"][key], agg["new"][key]
        print(f"{label:<26}{o:>7}/{n}{nv:>7}/{n}{nv - o:>+12}")
    for key, label in (("body_kept", "mean body kept"),
                       ("refs_dropped", "mean refs dropped")):
        o, nv = agg["old"][key], agg["new"][key]
        print(f"{label:<26}{o:>12.4f}{nv:>12.4f}{nv - o:>+12.4f}")

    # Per-article regressions matter more than the mean: a change that fixes ten articles
    # and breaks one is still a change that breaks one.
    print("\narticles where the new detector is WORSE:")
    worse = []
    for name, txt, tb in pairs:
        bo, ro = score(txt, tb, boundary(old, txt))
        bn, rn = score(txt, tb, boundary(new_refs, txt))
        if bn < bo - 0.005 or rn < ro - 0.01:
            worse.append((name, bo, ro, bn, rn))
    for name, bo, ro, bn, rn in worse:
        print(f"  {name:>12} body {bo:.3f}->{bn:.3f}  refs {ro:.3f}->{rn:.3f}")
    if not worse:
        print("  (none)")
    print(f"\narticles the old detector missed entirely: "
          f"{sum(1 for _, t, _ in pairs if boundary(old, t) is None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
