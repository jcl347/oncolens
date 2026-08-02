#!/usr/bin/env python
"""Keep only the oncology papers from a candidate PMID list.

**Why this gate exists.** Snowball expansion follows the corpus's own citation graph, and
citation graphs leak: an oncology paper cites the RNAi methods paper, the statistics
paper, the crystallography paper. Measured on the first mining run, the highest-yield
citation targets included Elbashir's siRNA work and generic shRNA toxicity papers — real
science, and not oncology. Ingesting along raw citation counts would gradually turn an
oncology corpus into a molecular-biology-methods corpus, and every retrieval number would
still look fine while the product drifted off its subject.

**The filter is NLM's, not ours.** The obvious implementation — a word list of "cancer",
"tumor", "carcinoma" — is exactly the hand-rolled vocabulary this project avoids
elsewhere, and it would bias the corpus toward the terms we happened to think of. Instead
this asks PubMed to intersect the candidates with a handful of **MeSH branch roots**.
A ``[MeSH Terms]`` search auto-explodes the whole subtree, so naming ``Neoplasms``
captures every one of the ~700 descriptors beneath it, chosen by NLM's indexers rather
than by us.

    python scripts/filter_oncology.py --in snowball_pmids.txt --out onco_pmids.txt

Input order is preserved, so a list sorted by citation count stays sorted — the most
valuable candidates remain first.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from oncolens.env import load_env  # noqa: E402
from oncolens.sources import pubmed  # noqa: E402

#: MeSH branch ROOTS, not leaf terms. PubMed explodes each into its full subtree, so this
#: short list covers the whole of oncology as NLM defines it.
#:
#: **Explosion verified, not assumed** (2026-08-02): 8/8 papers indexed under the child
#: descriptor ``Carcinoma, Non-Small-Cell Lung`` and NOT under ``Neoplasms`` itself still
#: matched ``"Neoplasms"[MeSH Terms]``. Without that, this would be a five-term keyword
#: match wearing a tree's clothes.
#:
#:   Neoplasms                C04, every cancer type, ~700 descriptors
#:   Antineoplastic Agents    D27, the drug classes
#:   Medical Oncology         H02, the discipline itself
#:   Carcinogenesis           G04, mechanism papers that predate a tumour
#:   Oncogenes                G05, the genetics
#:
#: ⚠️ **The first five were not enough, and the gap was immuno-oncology.** Sampling the
#: rejected fraction turned up papers that are unambiguously on-subject and carry no C04
#: descriptor at all: CAR-T antigen-recognition work indexed only as ``Antigens, Neoplasm``
#: + ``Immunotherapy, Adoptive``, and CAR-T neurotoxicity indexed as ``Receptors, Chimeric
#: Antigen``. Those terms live in D23 and E02, outside every branch named above, so the
#: gate was discarding the single largest research area in this corpus.
#:
#: The three additions are deliberately narrow. ``Immunotherapy`` on its own would admit
#: most of general immunology; adoptive transfer, tumour antigens and CARs are oncology by
#: construction.
ONCOLOGY_BRANCHES = (
    '"Neoplasms"[MeSH Terms]',
    '"Antineoplastic Agents"[MeSH Terms]',
    '"Medical Oncology"[MeSH Terms]',
    '"Carcinogenesis"[MeSH Terms]',
    '"Oncogenes"[MeSH Terms]',
    '"Immunotherapy, Adoptive"[MeSH Terms]',
    '"Antigens, Neoplasm"[MeSH Terms]',
    '"Receptors, Chimeric Antigen"[MeSH Terms]',
)

ONCOLOGY_CLAUSE = "(" + " OR ".join(ONCOLOGY_BRANCHES) + ")"


def oncology_subset(pmids: list[str], *, batch: int = 150, email: str | None = None,
                    api_key: str | None = None, sleep: float = 0.34) -> set[str]:
    """PMIDs from ``pmids`` that PubMed indexes under an oncology MeSH branch."""
    keep: set[str] = set()
    total = len(pmids)
    for i in range(0, total, batch):
        chunk = pmids[i : i + batch]
        term = "(" + " OR ".join(f"{p}[uid]" for p in chunk) + ") AND " + ONCOLOGY_CLAUSE
        try:
            # retmax must cover the batch or PubMed silently truncates the answer and the
            # tail of every batch would be dropped as "not oncology".
            got = pubmed.esearch(term, retmax=len(chunk), email=email, api_key=api_key)
        except Exception as e:  # noqa: BLE001
            print(f"  ! batch {i//batch}: {type(e).__name__}: {str(e)[:70]}", file=sys.stderr)
            continue
        keep.update(got)
        done = min(i + batch, total)
        if done % (batch * 10) == 0 or done == total:
            print(f"  {done:>6}/{total}  kept {len(keep):>6} ({len(keep)/max(done,1):.1%})",
                  flush=True)
        time.sleep(sleep)
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="only screen the first N candidates (they are citation-sorted, "
                         "so the first N are the highest-yield)")
    ap.add_argument("--email", default=None)
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()
    load_env()

    raw = Path(args.inp).read_text(encoding="utf-8").split()
    pmids = [p.strip() for p in raw if p.strip().isdigit()]
    if args.limit:
        pmids = pmids[: args.limit]
    print(f"screening {len(pmids)} candidates against {len(ONCOLOGY_BRANCHES)} MeSH branches")

    keep = oncology_subset(pmids, email=args.email, api_key=args.api_key)
    ordered = [p for p in pmids if p in keep]        # preserve citation-count order

    Path(args.out).write_text("\n".join(ordered), encoding="utf-8")
    print(f"\n{len(ordered)}/{len(pmids)} ({len(ordered)/max(len(pmids),1):.1%}) are indexed "
          f"under an oncology MeSH branch")
    print(f"wrote {args.out}")
    print("\nThe rejected fraction is the measured leak rate of the citation graph: those "
          "\npapers are cited BY oncology work without being oncology work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
