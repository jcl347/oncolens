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
import json
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
    ap.add_argument("--license-policy", default="research",
                    choices=["research", "commercial", "permissive_only", "all"],
                    help="which licences to index. 'research' (default) includes "
                         "non-commercial CC and TDM; 'commercial' excludes NC")
    ap.add_argument("--pmids-file", default=None,
                    help="ingest these PMIDs (one per line) instead of running the MeSH "
                         "seed queries. Used for snowball expansion along the citation "
                         "graph: see scripts/build_citation_labels.py --snowball-out")
    ap.add_argument("--embed-dim", type=int, default=192)
    ap.add_argument("--embed-backend", default="openai",
                    choices=["openai", "openai-large", "lsa", "voyage"],
                    help="MEASURED on 2,225 citation queries: openai scores nDCG@10 "
                         "0.4526 hybrid vs 0.3647 for lsa, and the lsa arm was actively "
                         "harmful (bm25 alone beat lsa+bm25). Default changed accordingly.")
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

    # ---- 1. discover real PMIDs ---------------------------------------------
    # Either by MeSH seed query (topical sample) or from an explicit list (snowball
    # expansion along the citation graph). Snowball matters because citation-context
    # labels only exist when BOTH the citing and the cited paper are held: on the first
    # topically-sampled corpus, 4,967 of 4,973 citation contexts pointed outside it.
    pmids: list[str] = []
    seen: set[str] = set()
    if args.pmids_file:
        raw = Path(args.pmids_file).read_text(encoding="utf-8").split()
        pmids = [p.strip() for p in raw if p.strip().isdigit()][: args.max_papers]
        print(f"{len(pmids)} PMIDs from {args.pmids_file} (snowball expansion)")

    per_seed = max(20, args.max_papers // len(SEEDS))
    for seed in ([] if pmids else SEEDS):
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
    # One unreachable article must not discard the whole job. A previous run died 45
    # minutes in on a single transient SSLEOFError from S3, having written nothing,
    # because the store write happens at the end. Transport failures are the expected
    # case across ~3,400 requests, not an exception.
    fetch_errors = 0
    for i, rec in enumerate(records):
        doc = rec.to_corpus_doc()          # abstract + real MeSH descriptors
        pmcid = pmc_map.get(rec.pmid)
        if pmcid:
            try:
                meta = pmc_cloud.fetch_metadata(pmcid, 1)
                if meta:
                    if not pmc_cloud.license_allows(meta, args.license_policy):
                        skipped_licence += 1
                    elif meta.get("is_retracted"):
                        pass                # never index a retracted article
                    else:
                        text = pmc_cloud.fetch_full_text(pmcid, 1)
                        if text:
                            paras = [p.strip() for p in text.split("\n\n")
                                     if len(p.strip()) > 60]
                            doc["sections"].append({"name": "Body",
                                                    "text": "\n\n".join(paras)})
                            doc["meta"].update({
                                "pmcid": pmcid, "license_code": meta.get("license_code"),
                                "full_text_chars": len(text), "source": "pmc_cloud",
                            })
                        else:
                            no_text += 1
            except Exception as e:  # noqa: BLE001
                fetch_errors += 1
                if fetch_errors <= 5:
                    print(f"  ! {pmcid}: {type(e).__name__} — keeping abstract only")
                elif fetch_errors == 6:
                    print("  ! (further fetch errors suppressed)")
        docs.append(doc)
        if (i + 1) % 250 == 0:
            print(f"  fetched {i+1}/{len(records)} "
                  f"({sum(1 for d in docs if len(d['sections']) > 1)} with full text)")

    full = sum(1 for d in docs if any(s["name"] == "Body" for s in d["sections"]))
    print(f"  {full} documents with REAL full text; {skipped_licence} skipped by the "
          f"'{args.license_policy}' licence policy; {no_text} had no text rendition; "
          f"{fetch_errors} fetch errors (abstract retained)")

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
    from oncolens.retrieval.dense import make_backend
    from oncolens.serve import neon_store

    texts = [c.indexable_text(include_heading=True) for c in chunks]
    print(f"embedding {len(texts)} passages "
          f"(backend={args.embed_backend}, dim={args.embed_dim})...")
    # Embedding at ingest time rather than in a second pass matters for storage: a
    # separate re-embed UPDATEs every row, which under MVCC grows the table by its own
    # size and breached Neon's project limit. Inserting the right vectors once avoids it.
    backend = make_backend(args.embed_backend, dim=args.embed_dim)
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
        neon_store.set_index_config(
            conn, **{neon_store.CFG_EMBED_MODEL: args.embed_backend,
                     neon_store.CFG_EMBED_DIM: str(args.embed_dim)})
    print(f"  {n_docs} documents, {n_chunks} passages in Postgres")

    # ---- 6. real qrels from NLM human indexing ------------------------------
    #
    # Everything essential is already committed to Postgres by this point. This step
    # publishes a convenience copy of the qrels, and it MUST NOT be able to fail the run:
    # a suspended Blob store once raised here and returned a non-zero exit after a
    # 40-minute ingest had fully succeeded, which reads as "the ingest failed" when the
    # corpus was in fact complete and correct.
    qrels = pubmed.mesh_qrels(records)
    local_qrels = local_data_dir() / "qrels_mesh.json"
    try:
        local_qrels.parent.mkdir(parents=True, exist_ok=True)
        local_qrels.write_text(json.dumps({"queries": qrels}, ensure_ascii=False),
                               encoding="utf-8")
        print(f"  {len(qrels)} MeSH concept queries -> {local_qrels}")
    except Exception as e:  # noqa: BLE001
        print(f"  ! could not write local qrels ({str(e)[:70]})")
    if use_blob:
        from oncolens.serve import vercel_blob as vb
        try:
            vb.put_json("qrels/mesh.json", {"queries": qrels})
            print("  qrels also published to Blob")
        except Exception as e:  # noqa: BLE001
            print(f"  ! Blob publish skipped: {str(e)[:90]}")
            print("    (the corpus in Postgres is complete; this copy is a convenience)")
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
