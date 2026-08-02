#!/usr/bin/env python
"""Which signals actually separate bibliography from body? Measured, not guessed.

The first version of the reference stripper was assembled from signals that *seemed*
discriminative (DOIs, author initials, years) with weights chosen by intuition. It then
failed on ~1 in 6 articles, and the failure mode was invisible because there was nothing
to measure against.

With ``<ref-list>`` giving a true boundary, every candidate signal can be scored on real
labelled text: compute it over the true body and over the true bibliography of the same
article, and report the separation. A signal that overlaps is noise no matter how sensible
it sounds.

    python scripts/analyze_ref_signals.py --limit 60 --email you@example.com
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bench_references import true_boundary  # noqa: E402
from ingest_real import pmids_to_pmcids  # noqa: E402
from oncolens.env import load_env, local_data_dir  # noqa: E402
from oncolens.sources import jats, pmc_cloud, pubmed  # noqa: E402

# --- candidate signals, all normalised per 1000 characters so blocks compare -----

_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_DOI = re.compile(r"\b10\.\d{4,9}/\S+")
_PMCID = re.compile(r"\bPMC\d{5,}\b")
#: Surname + initials, tolerant of the trailing period ACS/Nature styles use:
#: "Zhou J. Xu Y." as well as "Zhou J, Xu Y".
_AUTHOR_TOLERANT = re.compile(r"\b[A-Z][a-z]{1,20}\s+[A-Z][A-Za-z.\-]{0,3}(?=[\s,;.])")
#: The original, which required [ ,;] and therefore missed "Zhou J."
_AUTHOR_STRICT = re.compile(r"\b[A-Z][a-z]{1,20}\s+[A-Z][A-Za-z\-]{0,3}\b(?=[ ,;])")
_VOL_PAGE = re.compile(r"\b(19|20)\d{2}\s*[;:]\s*\d+\s*[:(]")
#: Page ranges: "709-20", "1123–1130" — pervasive in citations, rare in prose.
_PAGE_RANGE = re.compile(r"\b\d{1,5}\s*[-–]\s*\d{1,5}\b")
_NUMBERED = re.compile(r"^\s*\d{1,3}\.\s+[A-Z]", re.M)

_FUNCTION_WORDS = frozenset("""
the of and to in that we was for is are with as by this on were be have has had not but
which from at an it their our they these those than then when where while can could may
""".split())


def signals(text: str) -> dict[str, float]:
    n_chars = max(len(text), 1)
    per_k = 1000.0 / n_chars
    words = text.split()
    n_words = max(len(words), 1)
    fw = sum(1 for w in words if w.lower().strip(".,;:()") in _FUNCTION_WORDS) / n_words
    return {
        "years_per_1k": len(_YEAR.findall(text)) * per_k,
        "dois_per_1k": len(_DOI.findall(text)) * per_k,
        "pmcids_per_1k": len(_PMCID.findall(text)) * per_k,
        "authors_tolerant_per_1k": len(_AUTHOR_TOLERANT.findall(text)) * per_k,
        "authors_strict_per_1k": len(_AUTHOR_STRICT.findall(text)) * per_k,
        "volpage_per_1k": len(_VOL_PAGE.findall(text)) * per_k,
        "pagerange_per_1k": len(_PAGE_RANGE.findall(text)) * per_k,
        "numbered_per_1k": len(_NUMBERED.findall(text)) * per_k,
        "function_word_frac": fw,
        "mean_word_len": sum(len(w) for w in words) / n_words,
    }


def summarize(name: str, body: list[float], refs: list[float]) -> dict:
    """Separation between the two distributions.

    Reported as a rank statistic (AUC) rather than a mean difference, because the means
    are dominated by a few citation-dense articles while what matters is whether the
    signal orders body below bibliography *consistently*.
    """
    pairs = [(b, r) for b, r in zip(body, refs)]
    wins = sum(1 for b, r in pairs if r > b)
    ties = sum(1 for b, r in pairs if r == b)
    auc = (wins + 0.5 * ties) / max(len(pairs), 1)
    return {
        "signal": name,
        "body_median": statistics.median(body) if body else 0.0,
        "refs_median": statistics.median(refs) if refs else 0.0,
        "pairwise_auc": auc,
        "separated": auc >= 0.95 or auc <= 0.05,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--email", default="oncolens@example.com")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--queries", nargs="*", default=[
        "lung cancer AND resistance",
        "breast cancer AND proteomics",
        "immunotherapy AND melanoma",
        "leukemia AND single-cell",
        "glioblastoma AND radiotherapy",
    ])
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    load_env()
    cache = local_data_dir() / "jats_cache"
    cache.mkdir(parents=True, exist_ok=True)

    pmcids: list[str] = []
    for q in args.queries:
        pmids = pubmed.esearch(q, retmax=60, email=args.email, api_key=args.api_key)
        for p in pmids_to_pmcids(pmids).values():
            if p not in pmcids:
                pmcids.append(p)
    print(f"{len(pmcids)} candidate PMCIDs across {len(args.queries)} queries")

    rows: list[dict] = []
    skips: dict[str, int] = {}
    for pmcid in pmcids:
        if len(rows) >= args.limit:
            break
        try:
            meta = pmc_cloud.fetch_metadata(pmcid)
            if not meta:
                skips["no_metadata"] = skips.get("no_metadata", 0) + 1
                continue
            txt = pmc_cloud.fetch_full_text(pmcid, meta.get("version", 1))
            if not txt or len(txt) < 3000:
                skips["no_text"] = skips.get("no_text", 0) + 1
                continue
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
            if tb is None or tb < 1000:
                skips["unalignable"] = skips.get("unalignable", 0) + 1
                continue
        except Exception as exc:  # noqa: BLE001
            skips[type(exc).__name__] = skips.get(type(exc).__name__, 0) + 1
            continue

        body, refs = txt[:tb], txt[tb:]
        rows.append({"pmcid": pmcid, "chars": len(txt), "boundary": tb,
                     "ref_share": len(refs) / len(txt),
                     "body": signals(body), "refs": signals(refs)})
        if len(rows) % 10 == 0:
            print(f"  {len(rows)} labelled...")

    if not rows:
        print("no labelled articles")
        return 1

    print(f"\nlabelled {len(rows)} articles; skipped: {skips}")
    print("\nSIGNAL SEPARATION (pairwise AUC: 1.0 = refs always higher, 0.5 = useless)")
    print(f"{'signal':<26}{'body med':>10}{'refs med':>10}{'AUC':>8}  verdict")
    print("-" * 66)
    stats = []
    for key in rows[0]["body"]:
        s = summarize(key, [r["body"][key] for r in rows], [r["refs"][key] for r in rows])
        stats.append(s)
    for s in sorted(stats, key=lambda s: -abs(s["pairwise_auc"] - 0.5)):
        verdict = "DECISIVE" if s["separated"] else ("useful" if abs(s["pairwise_auc"] - .5) > .25 else "weak")
        print(f"{s['signal']:<26}{s['body_median']:>10.3f}{s['refs_median']:>10.3f}"
              f"{s['pairwise_auc']:>8.3f}  {verdict}")

    shares = sorted(r["ref_share"] for r in rows)
    print(f"\nbibliography share of characters: median {statistics.median(shares):.1%}, "
          f"min {shares[0]:.1%}, max {shares[-1]:.1%}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
