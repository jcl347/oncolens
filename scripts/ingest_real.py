#!/usr/bin/env python
"""Ingest REAL oncology literature into Vercel storage. Nothing is written to the repo.

Pipeline:

    PubMed MeSH query  ->  PMIDs
              |
              +-- efetch  ->  real abstracts + NLM human MeSH labels + grant links
              |
              +-- ID convert -> PMCIDs
                        |
                        +-- PMC Cloud (S3, anonymous) -> REAL full text, verbatim
                                  |
                                  +--> Vercel Blob     (article text, by URL)
                                  +--> Postgres/pgvector (chunks + embeddings + metadata)

Why the split: full text is ~25KB/article and grows without bound, so it belongs in object
storage, not in a database and certainly not in a synced folder. Postgres holds only what a
query needs, plus the blob URL so the full article is one fetch away.

    pip install -r requirements-dev.txt
    vercel env pull .env.local          # brings BLOB_READ_WRITE_TOKEN + POSTGRES_URL
    python scripts/ingest_real.py --max-papers 500 --email you@org.edu

Add --dry-run to fetch and report without writing to any store.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Real biomedical text contains Greek letters, math symbols and superscripts; the Windows
# console defaults to cp1252 and would raise UnicodeEncodeError on them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


from oncolens.env import describe_credentials, load_env, local_data_dir  # noqa: E402
from oncolens.retrieval.chunking import chunk_corpus  # noqa: E402
from oncolens.sources import pmc_cloud, pubmed  # noqa: E402

#: MeSH-anchored seeds spanning four oncology subdomains. Using MeSH terms (not free text)
#: means the corpus is defined by the same controlled vocabulary that provides the labels.
SEEDS = [
    '"Osimertinib"[MeSH Terms] AND "Drug Resistance, Neoplasm"[MeSH Terms]',
    '"ErbB Receptors"[MeSH Terms] AND "Lung Neoplasms"[MeSH Terms] AND resistance',
    '"Cyclin-Dependent Kinase 4"[MeSH Terms] AND "Breast Neoplasms"[MeSH Terms]',
    '"Receptors, Estrogen"[MeSH Terms] AND "Drug Resistance, Neoplasm"[MeSH Terms]',
    '"Immune Checkpoint Inhibitors"[MeSH Terms] AND "Tumor Microenvironment"[MeSH Terms]',
    '"Programmed Cell Death 1 Receptor"[MeSH Terms] AND "Drug Resistance, Neoplasm"[MeSH Terms]',
    '"Receptors, Chimeric Antigen"[MeSH Terms] AND "Neoplasm, Residual"[MeSH Terms]',
    '"Immunotherapy, Adoptive"[MeSH Terms] AND "Cytokine Release Syndrome"[MeSH Terms]',
]

ID_CONVERT = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"


def pmids_to_pmcids(pmids: list[str], *, batch: int = 180) -> dict[str, str]:
    """Map PMID -> PMCID via the NCBI ID Converter (GET, unauthenticated)."""
    import requests

    out: dict[str, str] = {}
    s = requests.Session()
    for i in range(0, len(pmids), batch):
        chunk = pmids[i : i + batch]
        r = s.get(ID_CONVERT, params={"ids": ",".join(chunk), "format": "json",
                                      "tool": "oncolens"}, timeout=60)
        if not r.ok:
            continue
        for rec in r.json().get("records", []):
            if rec.get("pmcid") and rec.get("pmid"):
                out[str(rec["pmid"])] = rec["pmcid"]
        time.sleep(0.34)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-papers", type=int, default=500)
    ap.add_argument("--email", default=None, help="NCBI asks for this; avoids throttling")
    ap.add_argument("--api-key", default=os.environ.get("NCBI_API_KEY"))
    ap.add_argument("--dry-run", action="store_true", help="fetch only; write to no store")
    ap.add_argument("--commercial-only", action="store_true", default=True,
                    help="skip articles whose licence forbids commercial use (default on)")
    ap.add_argument("--embed-dim", type=int, default=192)
    ap.add_argument("--no-blob", action="store_true",
                    help="skip Blob upload even if a token is present")
    args = ap.parse_args()

    # Read .env.local directly rather than requiring the caller to source it — the
    # PowerShell incantation for that is the single most error-prone step on Windows.
    loaded = load_env()
    if loaded:
        print(f"loaded {len(loaded)} variables from .env.local")
    print("credentials:")
    for line in describe_credentials():
        print(line)
    if not args.dry_run:
        if not (os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")):
            print("\nERROR: POSTGRES_URL / DATABASE_URL not set - nowhere to write.")
            print("Run:  vercel env pull .env.local --environment=production")
            print("Or add --dry-run to fetch papers without writing to any store.")
            return 1
        # Blob avoids duplicating whole articles and gives each passage a link back to its
        # source, but retrieval never reads it — passage text lives in chunks.text. So a
        # missing or unusable Blob degrades the pipeline rather than blocking it.
        if not os.environ.get("BLOB_READ_WRITE_TOKEN") or args.no_blob:
            print("\nNOTE: skipping Blob upload. Passage text is still stored in Postgres,")
            print("      so search and comparison are fully functional without it.")
    print()

    # ---- 1. discover real PMIDs by MeSH query --------------------------------
    per_seed = max(20, args.max_papers // len(SEEDS))
    pmids, seen = [], set()
    for seed in SEEDS:
        try:
            ids = pubmed.esearch(seed, retmax=per_seed, email=args.email, api_key=args.api_key)
        except Exception as e:
            print(f"  ! esearch failed: {str(e)[:80]}", file=sys.stderr)
            continue
        fresh = [i for i in ids if i not in seen]
        seen.update(fresh)
        pmids.extend(fresh)
        print(f"  {len(fresh):>4} PMIDs  <- {seed[:62]}")
    pmids = pmids[: args.max_papers]
    print(f"total unique PMIDs: {len(pmids)}")

    # ---- 2. real metadata + real human MeSH labels ---------------------------
    print("fetching PubMed records (XML, for MeSH headings)...")
    records = pubmed.efetch(pmids, email=args.email, api_key=args.api_key)
    with_mesh = sum(1 for r in records if r.mesh)
    print(f"  {len(records)} records, {with_mesh} carry NLM MeSH indexing")

    # ---- 3. PMID -> PMCID -> real full text from PMC Cloud -------------------
    print("resolving PMCIDs...")
    pmc_map = pmids_to_pmcids([r.pmid for r in records])
    print(f"  {len(pmc_map)} of {len(records)} have PMC full text available")

    docs, skipped_licence, no_text = [], 0, 0
    for rec in records:
        doc = rec.to_corpus_doc()          # abstract + real MeSH descriptors
        pmcid = pmc_map.get(rec.pmid)
        if pmcid:
            meta = pmc_cloud.fetch_metadata(pmcid, 1)
            if meta:
                if args.commercial_only and not pmc_cloud.commercial_use_ok(meta):
                    skipped_licence += 1
                elif meta.get("is_retracted"):
                    pass                    # never index a retracted article
                else:
                    text = pmc_cloud.fetch_full_text(pmcid, 1)
                    if text:
                        paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 60]
                        doc["sections"].append({"name": "Body", "text": "\n\n".join(paras)})
                        doc["meta"].update({
                            "pmcid": pmcid, "license_code": meta.get("license_code"),
                            "full_text_chars": len(text), "source": "pmc_cloud",
                        })
                    else:
                        no_text += 1
        docs.append(doc)

    full = sum(1 for d in docs if any(s["name"] == "Body" for s in d["sections"]))
    print(f"  {full} documents with REAL full text; {skipped_licence} skipped on licence; "
          f"{no_text} had no text rendition")

    chunks = chunk_corpus(docs)
    print(f"chunked into {len(chunks)} passages "
          f"({len(chunks)/max(len(docs),1):.1f} per document)")

    if args.dry_run:
        print("\n--dry-run: nothing written. Sample:")
        for d in docs[:3]:
            print(f"  {d['doc_id']}  {d['title'][:70]}")
            print(f"    MeSH: {d['descriptors'][:5]}")
        return 0

    # ---- 4. full text -> Vercel Blob (optional) -----------------------------
    use_blob = bool(os.environ.get("BLOB_READ_WRITE_TOKEN")) and not args.no_blob
    uploaded = 0
    if use_blob:
        from oncolens.serve import vercel_blob

        print("uploading full text to Vercel Blob...")
    else:
        print("skipping Blob upload (no token) — text still indexed in Postgres")
    for d in (docs if use_blob else []):
        pmcid = d.get("meta", {}).get("pmcid")
        body = next((s["text"] for s in d["sections"] if s["name"] == "Body"), None)
        if not (pmcid and body):
            continue
        try:
            ref = vercel_blob.put_text(vercel_blob.blob_path_for(pmcid, 1, "txt"), body)
            d["meta"].update(ref.as_meta())
            uploaded += 1
        except Exception as e:
            # A storage failure must not discard an otherwise-good ingestion: everything
            # needed for retrieval is already in hand, and Blob is only a convenience.
            print(f"  ! Blob upload failed ({str(e)[:70]}) — continuing without it")
            use_blob = False
            break
    if use_blob:
        print(f"  {uploaded} articles in Blob")

    # ---- 5. chunks + embeddings -> Postgres/pgvector -------------------------
    import psycopg
    from oncolens.retrieval.dense import LsaBackend
    from oncolens.serve import neon_store

    texts = [c.indexable_text(include_heading=True) for c in chunks]
    print(f"embedding {len(texts)} passages (dim={args.embed_dim})...")
    backend = LsaBackend(dim=args.embed_dim)
    backend.fit(texts)
    vectors = backend.encode_documents(texts)

    cfg = neon_store.NeonConfig.from_env(dim=args.embed_dim)
    with psycopg.connect(cfg.dsn) as conn:
        neon_store.init_schema(conn, dim=args.embed_dim)
        n_docs = neon_store.upsert_documents(conn, docs)
        rows = [{
            "chunk_id": c.chunk_id, "doc_id": c.doc_id, "section": c.section,
            "ordinal": c.ordinal, "start_char": c.start_char, "end_char": c.end_char,
            "text": c.text, "indexed_text": t,
        } for c, t in zip(chunks, texts)]
        n_chunks = neon_store.upsert_chunks(conn, rows, vectors.tolist())
    print(f"  {n_docs} documents, {n_chunks} passages in Postgres")

    # ---- 6. real qrels from NLM human indexing ------------------------------
    qrels = pubmed.mesh_qrels(records)
    if use_blob:
        from oncolens.serve import vercel_blob as vb
        vb.put_json("qrels/mesh.json", {"queries": qrels})
        print(f"  {len(qrels)} MeSH concept queries -> Blob (qrels/mesh.json)")
    else:
        import json as _json
        # NOT REPO/data: the repo may sit inside OneDrive, and ingested data
        # must never land in a synced folder.
        qp = local_data_dir() / "qrels"
        qp.mkdir(parents=True, exist_ok=True)
        with open(qp / "mesh.jsonl", "w", encoding="utf-8") as f:
            for q in qrels:
                f.write(_json.dumps(q, ensure_ascii=False) + "\n")
        print(f"  {len(qrels)} MeSH concept queries -> {qp / 'mesh.jsonl'}")

    print("\nDone. Nothing was written to the repository or to OneDrive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
