#!/usr/bin/env python
"""Measure the reference stripper against publisher ground truth.

The stripper used to be tuned by looking at articles and judging whether the output
"looked right". That is exactly the failure mode this project exists to avoid: a heuristic
validated by the person who wrote it, on the examples they happened to open.

PMC's JATS XML carries ``<ref-list>``, which is the **publisher's own** statement of where
the bibliography begins. Aligning that boundary onto the plain-text rendition gives a real
label, so the stripper can be scored rather than admired.

Two numbers matter, and they trade off against each other:

* **body kept** — fraction of true article-body characters the stripper preserved.
  Losing these deletes findings, which is unrecoverable. Must stay ~1.000.
* **refs dropped** — fraction of true bibliography characters removed. This is the
  precision leak being fixed.

Reported per article and in aggregate, plus the list of articles that fail, because an
average hides exactly the 1-in-6 case that motivated the work.

    python scripts/bench_references.py --limit 40 --email you@example.com
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from oncolens.env import load_env, local_data_dir  # noqa: E402
from oncolens.retrieval.references import find_reference_start  # noqa: E402
from oncolens.sources import jats, pmc_cloud, pubmed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest_real import pmids_to_pmcids  # noqa: E402

_WS = re.compile(r"\s+")


def collapse(text: str) -> tuple[str, list[int]]:
    """Whitespace-collapsed text plus a map from collapsed index -> original index.

    Needed because the XML and the txt rendition agree on *words* but not on line breaks,
    so the boundary can only be aligned after normalising whitespace — and the answer is
    only useful in original coordinates.
    """
    out: list[str] = []
    idx: list[int] = []
    prev_space = True
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            idx.append(i)
            prev_space = True
        else:
            out.append(ch)
            idx.append(i)
            prev_space = False
    return "".join(out), idx


def true_boundary(txt: str, ref_text: str) -> int | None:
    """Character offset in ``txt`` where the bibliography starts, per the XML."""
    flat_txt, imap = collapse(txt)
    flat_ref, _ = collapse(ref_text)
    for probe_len in (120, 80, 50, 32):
        probe = flat_ref[:probe_len]
        if len(probe) < 20:
            break
        pos = flat_txt.find(probe)
        if pos == -1:                      # try from the end: refs are near the tail
            pos = flat_txt.rfind(probe)
        if pos != -1:
            return imap[pos]
    return None


def predicted_boundary(txt: str) -> int | None:
    paras = txt.split("\n\n")
    idx = find_reference_start(paras)
    if idx is None:
        return None
    return len("\n\n".join(paras[:idx]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--email", default="oncolens@example.com")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--query", default="cancer AND therapy")
    ap.add_argument("--cache", default=None, help="dir for XML cache")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    load_env()
    cache = Path(args.cache) if args.cache else local_data_dir() / "jats_cache"
    cache.mkdir(parents=True, exist_ok=True)

    pmids = pubmed.esearch(args.query, retmax=args.limit * 6, email=args.email,
                           api_key=args.api_key)
    pmcids = list(pmids_to_pmcids(pmids).values())
    print(f"candidates: {len(pmcids)} PMCIDs from {len(pmids)} PMIDs")

    rows: list[dict] = []
    skips: dict[str, int] = {}
    for pmcid in pmcids:
        if len(rows) >= args.limit:
            break
        try:
            # Cache the txt as well as the XML: the detector is iterated on many times
            # and refetching 60 articles per change makes the loop too slow to use.
            tf = cache / f"{pmcid}.txt"
            if tf.exists():
                txt = tf.read_text(encoding="utf-8", errors="replace")
            else:
                meta = pmc_cloud.fetch_metadata(pmcid)
                if not meta:
                    skips["no_metadata"] = skips.get("no_metadata", 0) + 1
                    continue
                txt = pmc_cloud.fetch_full_text(pmcid, meta.get("version", 1))
                if not txt or len(txt) < 3000:
                    skips["no_text"] = skips.get("no_text", 0) + 1
                    continue
                tf.write_text(txt, encoding="utf-8")

            cf = cache / f"{pmcid}.xml"
            if cf.exists():
                xml = cf.read_text(encoding="utf-8", errors="replace")
            else:
                xml = jats.fetch_xml(pmcid, email=args.email, api_key=args.api_key)
                if not xml:
                    skips["no_xml"] = skips.get("no_xml", 0) + 1
                    continue
                cf.write_text(xml, encoding="utf-8")

            ref_text = jats.reference_list_text(xml)
            if not ref_text or len(ref_text) < 400:
                skips["no_reflist"] = skips.get("no_reflist", 0) + 1
                continue
            tb = true_boundary(txt, ref_text)
            if tb is None:
                skips["unalignable"] = skips.get("unalignable", 0) + 1
                continue                      # cannot align: not a detector failure
        except Exception as exc:              # noqa: BLE001
            skips[type(exc).__name__] = skips.get(type(exc).__name__, 0) + 1
            continue

        pb = predicted_boundary(txt)
        total = len(txt)
        true_body, true_refs = tb, total - tb
        if pb is None:
            body_kept, refs_dropped = 1.0, 0.0
        else:
            body_kept = min(pb, tb) / true_body if true_body else 1.0
            refs_dropped = max(0, total - max(pb, tb)) / true_refs if true_refs else 1.0
            # If pb < tb we cut into the body; if pb > tb we left refs behind.
            refs_dropped = (total - pb) / true_refs if pb >= tb else 1.0
            refs_dropped = min(refs_dropped, 1.0)
        rows.append({
            "pmcid": pmcid, "chars": total, "true_boundary": tb, "pred_boundary": pb,
            "ref_share": true_refs / total, "body_kept": body_kept,
            "refs_dropped": refs_dropped, "detected": pb is not None,
        })
        flag = "ok " if refs_dropped > 0.9 and body_kept > 0.99 else "FAIL"
        print(f"  {flag} {pmcid:>12} refs={true_refs/total:5.1%} "
              f"body_kept={body_kept:6.3f} refs_dropped={refs_dropped:6.3f}")

    if not rows:
        print("no evaluable articles")
        return 1

    n = len(rows)
    det = sum(r["detected"] for r in rows)
    bk = sum(r["body_kept"] for r in rows) / n
    rd = sum(r["refs_dropped"] for r in rows) / n
    clean = sum(1 for r in rows if r["refs_dropped"] > 0.9 and r["body_kept"] > 0.99)
    harmed = sum(1 for r in rows if r["body_kept"] < 0.99)

    print("\n" + "=" * 62)
    print(f"articles evaluated       {n}")
    print(f"bibliography detected    {det}/{n}  ({det/n:.1%})")
    print(f"mean body kept           {bk:.4f}   <- must be ~1.0000")
    print(f"mean refs dropped        {rd:.4f}")
    print(f"fully correct            {clean}/{n}  ({clean/n:.1%})")
    print(f"BODY DAMAGED             {harmed}/{n}")
    print(f"mean bibliography share  {sum(r['ref_share'] for r in rows)/n:.1%} of characters")
    print(f"skipped (not evaluable)  {skips}")
    worst = sorted(rows, key=lambda r: r["body_kept"])[:5]
    print("\nworst body_kept (body damage is the unrecoverable error):")
    for r in worst:
        print(f"  {r['pmcid']:>12} body_kept={r['body_kept']:.4f} "
              f"lost={(1-r['body_kept'])*r['true_boundary']:.0f} chars "
              f"pred={r['pred_boundary']} true={r['true_boundary']}")
    leftover = sorted((r for r in rows if r["refs_dropped"] < 0.999),
                      key=lambda r: r["refs_dropped"])[:5]
    print("\nworst refs_dropped (bibliography left in the index):")
    for r in leftover or []:
        print(f"  {r['pmcid']:>12} refs_dropped={r['refs_dropped']:.4f} "
              f"detected={r['detected']}")
    if not leftover:
        print("  (none)")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
