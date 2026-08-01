# Deploying OncoLens on Vercel, with real data at scale

Two things are separate and should stay separate:

| | Runs where | Needs |
|---|---|---|
| **Evaluation harness** (`src/oncolens/eval`, `loop.py`, `scripts/run.py`) | Offline — your laptop, CI | scipy, numpy, the corpus |
| **Serving path** (`api/search.py` or Postgres) | Vercel | numpy only, or nothing but SQL |

The harness never runs on Vercel. Conflating them is what makes people conclude a
retrieval system "can't be serverless" — the expensive part is fitting, and fitting is a
build-time concern.

---

## 1. Getting real text at scale

**Important:** the inability to pull verbatim full text in the environment where this repo
was written is a property of *that sandbox*, not of the pipeline. A normal machine — your
laptop, a GitHub Action, a Vercel build step — gets raw bytes from `requests.get()`. The
code in `src/oncolens/sources/` is written for that case.

### Papers: PMC Open Access bulk (the scale path)

Per-article API calls do not scale — NCBI asks for ≤3 req/s, so a million articles is
about four days of polite requests. Use the bulk service.

```
https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv   index: PMCID, PMID, path, license
https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/           packaged subsets
https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/08/e0/PMC13900.tar.gz   one article
```

**Pick the licence subset deliberately — this is a product decision:**

| Subset | Licences | Hosted commercial product |
|---|---|---|
| `oa_comm` | CC0, CC BY, CC BY-SA, CC BY-ND | ✅ use this |
| `oa_noncomm` | CC BY-NC, CC BY-NC-SA, CC BY-NC-ND | ❌ non-commercial only |
| `oa_other` | no machine-readable licence / custom | ⚠️ review individually |

`pmc_bulk.DEFAULT_SUBSET` is `oa_comm` so the safe choice is the default.

Prefer the **BioC** distribution where available — it is already segmented into passages
*with character offsets*, which is exactly the provenance this product needs and removes a
whole parsing step.

### Papers: metadata + the labels

`sources/pubmed.py` (E-utilities, all GET). The reason it matters is **MeSH**: NLM's human
indexers assigned those descriptors, and `MajorTopicYN` already distinguishes a paper's
central topics from incidental ones. That yields *graded* relevance judgments (3 = major,
1 = minor) that no LLM invented and no retriever influenced.

`pmc_bulk.attach_mesh()` joins bulk full text to MeSH labels by PMID, so a corpus built
for scale still arrives with human labels attached.

### Grants

* **Europe PMC Grist** (`sources/europepmc.grist_grants`) — GET, unauthenticated, real
  awarded grants with abstracts, PIs and institutions.
* **NIH RePORTER** (`POST /v2/projects/search`) — richer, and the source of
  grant→publication links. POST is only a problem for restricted fetchers; from a normal
  runtime it is fine.

```bash
pip install requests
python scripts/fetch_real.py --out data/real --max-papers 5000 --email you@org.edu --full-text
ONCOLENS_DATA=data/real python scripts/run.py validate
```

---

## 2. Two deployment shapes

### A. Bundled artifact — simplest, right for small/medium corpora

Build offline, ship a precomputed index, serve with numpy only.

```bash
python scripts/build_artifact.py --out artifact
vercel deploy
```

`api/search.py` loads `artifact/` at module scope (so warm containers pay the cost once)
and does BM25 + LSA + RRF in-process. **scipy is build-time only** — it fits the SVD, and
only the fitted matrices ship. That keeps the function bundle and cold start small.

Fits comfortably to roughly the low tens of thousands of chunks. Past that the artifact
stops fitting in a function bundle and you want shape B.

### B. Neon Postgres + pgvector — the real-scale answer

**Vercel Postgres is sunset**; storage now comes from Marketplace providers. Neon fits
best here:

* pgvector on every plan, no paid tier required
* database **branching that mirrors Vercel preview deployments**, so a preview never
  queries production data
* serverless driver over HTTP, which works on Edge where raw TCP does not
* comfortable to roughly 50M vectors — far past this corpus

Why Postgres rather than a dedicated vector DB: **this corpus is relational.** Grants link
to publications, publications to PIs and institutions, concepts to documents. Queries like
"what has my institution funded on X" or "papers from grants studying Y" are joins. A pure
vector store makes you rebuild them in application code.

```bash
# Vercel → Marketplace → Neon; POSTGRES_URL is injected automatically
python scripts/ingest_neon.py --data data/real   # creates schema, embeds, upserts
```

`serve/neon_store.py` has the schema, the HNSW index, and RRF fused in SQL.

`Upstash Vector` is the alternative if you want pure edge/HTTP and don't need the joins.

---

## 3. The caveat that actually matters: measure what you ship

`ts_rank_cd` **is not BM25.** If the harness measures BM25 in-process and production serves
`ts_rank_cd` from Postgres, the evaluation stops describing the product — which is the
precise failure this project exists to prevent.

Three options, in order of preference:

1. **`pg_search` (ParadeDB)** if your provider supports the extension — real BM25 in SQL.
2. **BM25 in SQL from stored statistics.** `BM25_SCHEMA_SQL` stores per-term frequencies
   and corpus stats so BM25 can be computed exactly as the offline index computes it.
   Heavier ingestion, faithful scoring.
3. **Re-run the evaluation against the deployed backend** and report *those* numbers.

Option 3 is legitimate. What is not legitimate is measuring one and shipping the other.
`tests/test_serve_parity.py` enforces this for the bundled-artifact path; add the
equivalent for the SQL path before trusting production numbers.

---

## 4. Vercel constraints this design is written against

| Constraint | How it is handled |
|---|---|
| Read-only FS outside `/tmp` | Nothing is fitted or written at request time |
| Bundle size / cold start | Runtime = numpy only; scipy is build-time |
| Short execution limits | No index construction per request |
| Ephemeral containers | Artifact loads at module scope, reused while warm |
| No TCP on Edge | Neon serverless driver over HTTP, or the bundled artifact |
| Preview isolation | Neon branch per Vercel preview |

## 5. Environment variables

| Variable | Used by | Purpose |
|---|---|---|
| `POSTGRES_URL` / `DATABASE_URL` | `serve/neon_store.py` | Neon connection (auto-injected by the integration) |
| `ONCOLENS_ARTIFACT` | `api/search.py` | Artifact directory, defaults to `./artifact` |
| `NCBI_API_KEY` | `sources/pubmed.py` | Raises E-utilities to 10 req/s |
| `VOYAGE_API_KEY` | `retrieval/dense.py` | Swaps LSA for real neural embeddings |
