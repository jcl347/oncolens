"""Build a local oncology gazetteer from the NCI Thesaurus flat file.

**Why local rather than the REST API.** Expansion can afford a cached HTTP round trip per
unseen term. *Mining* cannot: scanning 7,056 claim sentences for every span that might be
an entity is millions of lookups, and at the API's rate limit that is days. A trie over the
whole vocabulary answers in O(span length), independent of dictionary size.

**Why this replaces a regex rather than refining one.** ``eval.strata._IDENTIFIER`` decides
what counts as an identifier using four character-class shapes. It cannot match a bare gene
symbol (``BRCA1``, ``TP53``) or any drug name (``osimertinib``), because those are shaped
like ordinary words — and that is not a tuning problem, it is the same limit described in
§4.14: a pattern sees shape, and identity is not in the shape. A dictionary knows that
``osimertinib`` is a Pharmacologic Substance and ``the`` is not, and it returns the type
along with the match.

Output: ``%LOCALAPPDATA%/oncolens/ncit_gazetteer.json.gz``, a map of
``normalised surface form -> {code, name, semantic types}``.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oncolens.env import load_env, local_data_dir  # noqa: E402
from oncolens.terminology import norm_key  # noqa: E402

FLAT_URL = "https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/Thesaurus.FLAT.zip"

#: Semantic types worth keeping for *mining*. The gazetteer exists to find things a
#: researcher would type into a search box as a literal, so concepts that are only ever
#: administrative or geographic are dropped — otherwise "Eritrea" becomes an identifier
#: query because ``ER`` matched it (§4.14).
MINE_TYPES = {
    "Gene or Genome",
    "Amino Acid, Peptide, or Protein",
    "Enzyme",
    "Receptor",
    "Immunologic Factor",
    "Nucleotide Sequence",
    "Pharmacologic Substance",
    "Organic Chemical",
    "Antibiotic",
    "Cell",
    "Neoplastic Process",
    "Cell or Molecular Dysfunction",
    "Disease or Syndrome",
    "Nucleic Acid, Nucleoside, or Nucleotide",
    "Biologically Active Substance",
    "Indicator, Reagent, or Diagnostic Aid",
}

#: Surface forms this short are ambiguous beyond rescue — see MIN_TERM_CHARS in
#: terminology.py, and the ``ER`` case that motivated it.
MIN_CHARS = 3

#: Common English words that are also NCIt concept names. Mining on these would turn every
#: claim sentence into dozens of spurious identifier queries. This is a stop list against
#: *false positives from the dictionary*, not a hand-built vocabulary of what to look for —
#: the distinction §4.4 draws.
STOPWORDS = {
    "and", "or", "not", "the", "for", "with", "was", "were", "are", "can", "may", "all",
    "this", "that", "these", "those", "from", "into", "than", "then", "when", "where",
    "cell", "cells", "gene", "genes", "protein", "proteins", "study", "studies", "group",
    "groups", "patient", "patients", "result", "results", "method", "methods", "data",
    "level", "levels", "effect", "effects", "type", "types", "case", "cases", "control",
    "controls", "human", "mouse", "rat", "tumor", "tumour", "cancer", "disease", "one",
    "two", "three", "high", "low", "new", "old", "same", "other", "such", "both", "each",
    "more", "most", "less", "only", "also", "well", "very", "much", "many", "some", "any",
}


def download(dest: Path) -> Path:
    """Fetch the flat file, skipping if it is already here."""
    import requests

    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  using cached {dest.name} ({dest.stat().st_size/1e6:.0f} MB)")
        return dest
    print(f"  downloading {FLAT_URL}")
    t0 = time.perf_counter()
    r = requests.get(FLAT_URL, timeout=300, stream=True)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
            n += len(chunk)
    print(f"  {n/1e6:.0f} MB in {time.perf_counter()-t0:.0f}s")
    return dest


def parse(zip_path: Path) -> dict[str, dict]:
    """Read Thesaurus.txt out of the zip into a normalised-key map.

    Column layout of the NCIt flat file (tab separated, no header):
      0 code   1 concept IRI   2 parents   3 synonyms (pipe-sep)
      4 definition   5 display name   6 concept status   7 semantic type (pipe-sep)
    """
    gaz: dict[str, dict] = {}
    collisions = 0
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".txt"))
        print(f"  reading {name}")
        with z.open(name) as fh:
            for line in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 8:
                    continue
                code, syn_field, display, sem_field = parts[0], parts[3], parts[5], parts[7]
                sem = [s.strip() for s in sem_field.split("|") if s.strip()]
                if not (set(sem) & MINE_TYPES):
                    continue
                surfaces = [s.strip() for s in syn_field.split("|") if s.strip()]
                label = display.strip() or (surfaces[0] if surfaces else code)
                for s in surfaces:
                    if len(s) < MIN_CHARS or s.lower() in STOPWORDS:
                        continue
                    k = norm_key(s)
                    if len(k) < MIN_CHARS or k in STOPWORDS:
                        continue
                    if k in gaz:
                        # Keep the first; NCIt's file order puts primary concepts early.
                        # Record that the surface form is not unique, so a caller can
                        # refuse ambiguous mines rather than silently pick one (§4.14).
                        gaz[k]["senses"] += 1
                        collisions += 1
                        continue
                    gaz[k] = {"code": code, "name": label, "sem": sem[:2], "senses": 1}
    print(f"  {len(gaz):,} distinct surface forms  ({collisions:,} collisions recorded)")
    return gaz


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="gazetteer path (default: local data dir)")
    args = ap.parse_args()

    load_env(Path(__file__).resolve().parent.parent)
    ddir = local_data_dir()
    zip_path = download(ddir / "Thesaurus.FLAT.zip")
    gaz = parse(zip_path)

    out = Path(args.out) if args.out else ddir / "ncit_gazetteer.json.gz"
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump(gaz, fh)
    print(f"  wrote {out}  ({out.stat().st_size/1e6:.1f} MB gzipped)")

    for probe in ("BRCA1", "osimertinib", "EGFR T790M", "MCF-7", "pembrolizumab",
                  "TP53", "PD-L1", "trastuzumab"):
        hit = gaz.get(norm_key(probe))
        print(f"    {probe:<16} {'-> ' + hit['code'] + '  ' + hit['name'][:38] if hit else '(absent)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
