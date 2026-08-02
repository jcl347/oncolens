# Vercel setup: storage containers + deployment

Follow top to bottom. Roughly 15 minutes. Nothing is stored in the repo or in OneDrive.

## What goes where, and why

| Data | Store | Reason |
|---|---|---|
| Article full text (~25 KB each) | **Vercel Blob** | Large, immutable, served by URL. Does not belong in git or a database — and definitely not in a synced folder, where a churning corpus causes sync storms and file locks. |
| Chunks + embeddings + metadata | **Neon Postgres + pgvector** | Queried per request; needs ANN search *and* relational joins (grant ↔ publication ↔ PI ↔ institution). |
| Code, tests, fixtures | git | Small and reviewable. |

Postgres stores each passage's blob URL, so the full article is one fetch away without
being stored twice.

---

## Step 1 — Push the repo and import it

```bash
git push origin main
```

Vercel dashboard → **Add New → Project** → import `jcl347/oncolens` → **Deploy**.

The first deploy will succeed but `/api/search` returns `503 index artifact not found`.
That is expected until Step 5.

---

## Step 2 — Create the Blob store (full text)

Vercel dashboard → your project → **Storage** → **Create Database** → **Blob** →
name it `oncolens-text` → **Create**.

Connect it to the project when prompted. This injects:

```
BLOB_READ_WRITE_TOKEN
```

> Blob has no schema and needs no setup beyond creation. Paths are deterministic
> (`pmc/txt/PMC1234567.1.txt`), so re-running ingestion overwrites rather than
> accumulating duplicates — which matters when a long job fails halfway.

---

## Step 3 — Create the Postgres/pgvector store (vectors)

Vercel dashboard → **Storage** → **Create Database** → **Neon** (Marketplace) →
region **US East** (same as the PMC bucket, keeps ingestion fast) → **Create**.

> Vercel Postgres was sunset; storage now comes from Marketplace providers. Neon is the
> right pick here: pgvector on every plan, and **database branching that mirrors Vercel
> preview deployments**, so a preview never queries production data.

This injects `POSTGRES_URL` (and `DATABASE_URL`). No manual schema step — the ingestion
script creates tables, the HNSW vector index and the GIN text index on first run.

---

## Step 4 — Ingest real data

Run this **locally or in CI**, not in a Vercel function (it takes minutes to hours and
exceeds function limits).

```bash
pip install -r requirements-dev.txt
pip install "psycopg[binary]"

npm i -g vercel && vercel link      # once
vercel env pull .env.local          # pulls BLOB_READ_WRITE_TOKEN + POSTGRES_URL

python scripts/ingest_real.py --max-papers 2000 --email you@cornell.edu
```

> The scripts read `.env.local` themselves, so there is nothing to source into your
> shell. That step was the most error-prone part of this on Windows, so it was removed
> rather than documented around.

Verify first without writing anything:

```bash
python scripts/ingest_real.py --max-papers 24 --email you@cornell.edu --dry-run
```

A real dry run on 2026-08-01 produced: 24 PMIDs, **24/24 carrying NLM MeSH indexing**,
14 with PMC full text, **11 ingested verbatim**, 3 correctly skipped on licence,
809 passages (33.7 per document).

### What the ingestion actually does

1. **PubMed MeSH queries → PMIDs.** Seeds are MeSH terms, not free text, so the corpus is
   defined by the same controlled vocabulary that supplies the labels.
2. **efetch → real abstracts + human MeSH descriptors + grant links.** `MajorTopicYN`
   gives *graded* relevance (3 = major topic, 1 = minor) with nothing invented.
3. **ID Converter → PMCIDs**, then **PMC Cloud Service** (S3 `pmc-oa-opendata`,
   anonymous, no AWS account) → **verbatim full text**.
4. **Licence gate.** Only CC0 / CC BY / CC BY-SA / CC BY-ND are indexed; retracted
   articles are skipped entirely. `--commercial-only` is on by default.
5. Text → Blob; chunks + embeddings + metadata → Postgres.

> ⏳ **Deadline that affects this:** NCBI retires the legacy PMC dataset files and the OA
> Web Service API **on or after 24 August 2026**. This code targets the Cloud Service
> (`s3://pmc-oa-opendata`), which is the replacement — anything still using
> `ftp.ncbi.nlm.nih.gov/pub/pmc/...` or `oa.fcgi` breaks then.

---

## Step 5 — Point the API at Postgres

The default `api/search.py` reads a bundled artifact, which suits a small corpus. Once data
is in Postgres, switch to the SQL path so the function stays thin and the corpus can grow:

```python
# api/search.py — swap the index for the store
import psycopg
from oncolens.serve import neon_store

conn = psycopg.connect(neon_store.NeonConfig.from_env().dsn)
results = neon_store.hybrid_search(conn, query, query_embedding, top_k=10)
```

Add `psycopg[binary]` to `requirements.txt` for the function.

---

## Step 6 — Verify

```bash
curl "https://<your-project>.vercel.app/api/search?q=EGFR%20C797S%20resistance&k=5"
```

Each result carries the document **and the passage** with `section`, `start_char`,
`end_char` — the provenance the product requires.

---

## ⚠️ One thing to get right before trusting production numbers

**`ts_rank_cd` is not BM25.** The offline harness measures BM25; Postgres full-text search
scores differently. If you measure one and ship the other, the evaluation stops describing
the product. Options, best first:

1. Install **`pg_search` (ParadeDB)** if Neon supports the extension — real BM25 in SQL.
2. Compute BM25 in SQL from stored statistics (`BM25_SCHEMA_SQL` in `neon_store.py`).
3. Re-run the evaluation against the deployed backend and report *those* numbers.

Option 3 is legitimate. Measuring one and shipping the other is not.

---

## Environment variables (all injected by the integrations)

| Variable | Source | Used by |
|---|---|---|
| `BLOB_READ_WRITE_TOKEN` | Blob integration | `serve/vercel_blob.py` |
| `POSTGRES_URL` / `DATABASE_URL` | Neon integration | `serve/neon_store.py` |
| `NCBI_API_KEY` | optional, [ncbi.nlm.nih.gov](https://ncbi.nlm.nih.gov) | raises E-utilities to 10 req/s |
| `VOYAGE_API_KEY` | optional | swaps LSA for real neural embeddings |

## Cost

Blob and Neon both have free tiers that comfortably hold a few thousand articles. 2,000
articles ≈ 50 MB text in Blob and ≈ 65k passages in Postgres — inside the Neon free tier.
