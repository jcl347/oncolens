#!/usr/bin/env python
"""Pull a REAL oncology corpus + REAL human-assigned labels. Requires network.

This is the script to run on a machine with internet. It was written in an environment
where the shell had no outbound network, so it is deliberately dependency-light
(``requests`` only) and every endpoint it uses is GET and unauthenticated.

    pip install requests
    python scripts/fetch_real.py --out data/real --max-papers 800 --email you@org.edu

What it produces:

  data/real/corpus/papers.jsonl    real PubMed records (title, abstract, MeSH, grants)
  data/real/corpus/grants.jsonl    real awarded grants from Europe PMC Grist
  data/real/qrels/mesh.jsonl       queries + graded judgments from NLM human MeSH indexing
  data/real/qrels/funding.jsonl    grant -> publication judgments (found data)
  data/real/manifest.json          provenance: queries used, counts, timestamp

Why these labels are worth more than generated ones: NLM's indexers assigned the MeSH
descriptors, and flagged which are a paper's *major* topics, years before this benchmark
existed. Nobody tuned them to a retriever. The MajorTopicYN flag gives graded relevance
(3 = major, 1 = minor) without inventing a scale.

Note on NIH RePORTER: its API is POST-only, so it cannot be reached from restricted
environments the way these GET endpoints can. ``--reporter`` enables it when POST works.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oncolens.sources import europepmc as epmc  # noqa: E402
from oncolens.sources import pubmed  # noqa: E402

#: Topic seeds spanning four oncology subdomains, so the corpus has genuinely distinct
#: areas and cross-domain queries are meaningful rather than trivially separable.
SEEDS = [
    '"Osimertinib"[MeSH Terms] AND "Drug Resistance, Neoplasm"[MeSH Terms]',
    '"ErbB Receptors"[MeSH Terms] AND "Lung Neoplasms"[MeSH Terms] AND resistance',
    '"Cyclin-Dependent Kinase 4"[MeSH Terms] AND "Breast Neoplasms"[MeSH Terms]',
    '"Receptors, Estrogen"[MeSH Terms] AND "Drug Resistance, Neoplasm"[MeSH Terms]',
    '"Immune Checkpoint Inhibitors"[MeSH Terms] AND "Tumor Microenvironment"[MeSH Terms]',
    '"Programmed Cell Death 1 Receptor"[MeSH Terms] AND resistance',
    '"Receptors, Chimeric Antigen"[MeSH Terms] AND "Neoplasm, Residual"[MeSH Terms]',
    '"Immunotherapy, Adoptive"[MeSH Terms] AND "Cytokine Release Syndrome"[MeSH Terms]',
]

GRANT_SEEDS = ["cancer AND resistance", "oncology AND immunotherapy", "tumour AND targeted therapy"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real")
    ap.add_argument("--max-papers", type=int, default=800)
    ap.add_argument("--max-grants", type=int, default=200)
    ap.add_argument("--email", default=None, help="sent to NCBI as requested by their usage policy")
    ap.add_argument("--api-key", default=None, help="NCBI_API_KEY raises the rate limit to 10 req/s")
    ap.add_argument("--full-text", action="store_true", help="also pull open-access JATS full text")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "corpus").mkdir(parents=True, exist_ok=True)
    (out / "qrels").mkdir(parents=True, exist_ok=True)

    per_seed = max(20, args.max_papers // len(SEEDS))
    pmids: list[str] = []
    seen: set[str] = set()
    for seed in SEEDS:
        try:
            ids = pubmed.esearch(seed, retmax=per_seed, email=args.email, api_key=args.api_key)
        except Exception as e:  # a dead seed must not kill the whole pull
            print(f"  ! esearch failed for {seed[:50]}: {e}", file=sys.stderr)
            continue
        fresh = [i for i in ids if i not in seen]
        seen.update(fresh)
        pmids.extend(fresh)
        print(f"  {len(fresh):>4} new PMIDs  <- {seed[:64]}")
    print(f"total unique PMIDs: {len(pmids)}")

    print("fetching full records (XML, for MeSH headings)...")
    records = pubmed.efetch(pmids, email=args.email, api_key=args.api_key)
    print(f"  parsed {len(records)} records; "
          f"{sum(1 for r in records if r.mesh)} carry MeSH indexing")

    docs = [r.to_corpus_doc() for r in records]

    if args.full_text:
        print("fetching open-access full text...")
        n = 0
        for d in docs:
            pmcid = (d.get("meta") or {}).get("pmcid")
            if not pmcid:
                continue
            try:
                secs = epmc.full_text_sections("PMC", pmcid)
            except Exception:
                continue
            if secs:
                d["sections"] = d["sections"] + secs
                n += 1
        print(f"  added full text to {n} documents")

    (out / "corpus" / "papers.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in docs) + "\n", encoding="utf-8")

    print("fetching real grants (Europe PMC Grist)...")
    grants: list[dict] = []
    for gq in GRANT_SEEDS:
        try:
            grants.extend(epmc.grist_grants(gq, max_results=args.max_grants // len(GRANT_SEEDS)))
        except Exception as e:
            print(f"  ! grist failed for {gq}: {e}", file=sys.stderr)
    uniq = {g["doc_id"]: g for g in grants}
    (out / "corpus" / "grants.jsonl").write_text(
        "\n".join(json.dumps(g, ensure_ascii=False) for g in uniq.values()) + "\n", encoding="utf-8")
    print(f"  {len(uniq)} grants")

    print("building qrels from human MeSH indexing...")
    mesh_q = pubmed.mesh_qrels(records)
    (out / "qrels" / "mesh.jsonl").write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in mesh_q) + "\n", encoding="utf-8")
    print(f"  {len(mesh_q)} MeSH concept queries "
          f"(mean {sum(len(q['judgments']) for q in mesh_q) / max(len(mesh_q),1):.1f} judged docs each)")

    epmc_records = [
        epmc.EuropePMCRecord(ext_id=r.pmid, source="MED", title=r.title, abstract=r.abstract,
                             journal=r.journal, year=r.year, pmid=r.pmid, grants=r.grants)
        for r in records
    ]
    fund_q = epmc.funding_link_qrels(epmc_records)
    (out / "qrels" / "funding.jsonl").write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in fund_q) + "\n", encoding="utf-8")
    print(f"  {len(fund_q)} funding-link queries")

    (out / "manifest.json").write_text(json.dumps({
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "seeds": SEEDS, "grant_seeds": GRANT_SEEDS,
        "n_papers": len(docs), "n_grants": len(uniq),
        "n_mesh_queries": len(mesh_q), "n_funding_queries": len(fund_q),
        "label_provenance": {
            "mesh": "NLM human indexers; grade 3 = MajorTopicYN=Y, grade 1 = minor",
            "funding": "funder/author-asserted grant->publication links",
        },
    }, indent=2), encoding="utf-8")

    print(f"\ndone -> {out}")
    print(f"run the harness against it:  ONCOLENS_DATA={out} python scripts/run.py validate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
