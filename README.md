# OncoLens

Retrieval over oncology **papers and grants** that returns the associated documents, **the
exact passage where a concept was mentioned**, and side-by-side **technical comparisons**
between papers.

The passage requirement drives the architecture: every chunk carries
`(doc_id, section, start_char, end_char)` back to its source, and every match narrows
further to the **clause** that actually matched — so a result can always be shown in
context rather than paraphrased.

---

## Status, honestly

| | State |
|---|---|
| Retrieval + evaluation harness | Built, tested, working |
| Real data ingestion (PubMed + PMC full text) | **Verified working** — 24/24 records with real NLM MeSH indexing, verbatim full text |
| Vercel site (search, compare, eval panel, WebGL) | Built; needs `npm install` + deploy |
| Vercel storage (Blob + Neon/pgvector) | Code + setup guide ready; not yet provisioned |
| **A real corpus ingested at scale** | **Not yet run** — `data/` ships empty |
| Retrieval improvement loop | **Deliberately not run** — see *Why the loop hasn't run* below |

`data/` is empty by design. It holds real ingested content, which is never committed.
The only corpus in the repo is `fixtures/synthetic/`, which is **machine-generated test
data, not real papers** — see [`fixtures/README.md`](fixtures/README.md).

---

## Quick start

```bash
pip install -r requirements-dev.txt

# 1. verify the measurement engine itself (hand-checked against computed values)
python tests/test_metrics.py
python tests/test_pipeline.py
python tests/test_loop_e2e.py

# 2. exercise the harness against the synthetic fixture (NOT real data)
ONCOLENS_DATA=fixtures/synthetic python scripts/run.py validate
ONCOLENS_DATA=fixtures/synthetic python scripts/run.py demo "EGFR C797S resistance"

# 3. ingest REAL oncology literature (needs network)
python scripts/ingest_real.py --max-papers 24 --email you@org.edu --dry-run
```

---

## Deploying on Vercel

Full walkthrough: **[`docs/VERCEL_SETUP.md`](docs/VERCEL_SETUP.md)**. Short version:

### 1. Import the repo

Vercel → **Add New → Project** → import `jcl347/oncolens` → **Deploy**.

The Next.js frontend builds immediately. `/api/search` returns `503` until there is an
index — expected.

### 2. Create the two storage containers

Vercel → your project → **Storage** → **Create Database**:

| Store | Choose | Injects | Holds |
|---|---|---|---|
| **Blob** — name it `oncolens-text` | Blob | `BLOB_READ_WRITE_TOKEN` | Article full text (~25 KB each) |
| **Postgres** — Neon, region **US East** | Neon (Marketplace) | `POSTGRES_URL` | Chunks, embeddings, metadata |

Why both: full text is large and immutable, so it belongs in object storage; Postgres holds
only what a query needs, plus each passage's blob URL. Why Neon specifically — Vercel
Postgres is sunset, Neon gives pgvector on every plan and **branches per preview
deployment**, so a preview never queries production data. And why Postgres rather than a
dedicated vector DB: this corpus is *relational* (grants ↔ publications ↔ PIs ↔
institutions), and those joins are half the product.

No schema step — ingestion creates the tables, the HNSW vector index and the GIN text index.

### 3. Ingest real data (run locally or in CI, **not** in a function)

```bash
npm i -g vercel && vercel link
vercel env pull .env.local          # brings both tokens down

pip install -r requirements-dev.txt "psycopg[binary]"
python scripts/ingest_real.py --max-papers 2000 --email you@org.edu
```

This exceeds serverless execution limits by design — it takes minutes to hours.

### 4. Publish the evaluation report and deploy

```bash
python scripts/build_eval_report.py --out public/eval_report.json
git add public/eval_report.json && git commit -m "eval report" && git push
```

The site reads this to populate the metrics panel. Without it, the panel states that the
system is unmeasured rather than quietly showing nothing.

---

## Where the data comes from

| Source | What it provides | Access |
|---|---|---|
| **PubMed E-utilities** | Abstracts, grant links, and **NLM human MeSH indexing** | GET, unauthenticated |
| **PMC Cloud Service** | **Verbatim full text** (`s3://pmc-oa-opendata`) | Anonymous HTTPS, no AWS account |
| **Europe PMC** | JATS full text, and Grist for real grants | GET, unauthenticated |

The labels are the valuable part. NLM's **human indexers** assign MeSH descriptors to every
PubMed record, and `MajorTopicYN` already separates a paper's central topics from its
incidental ones — which yields *graded* relevance judgments (3 = major, 1 = minor) that no
LLM invented and no retriever influenced.

> ⏳ **Deadline:** NCBI retires the legacy PMC dataset files and the OA Web Service API **on
> or after 24 August 2026**. `sources/pmc_cloud.py` targets the replacement Cloud Service.
> `sources/pmc_bulk.py` targets the old FTP paths and is **deprecated** — it will stop
> working then.

---

## Why the loop hasn't run

Three adversarial critics audited the benchmark. Seven defects were fixed (label leakage,
dev/test split leaking across information needs, underpowered per-stratum gating, a
degenerate `bpref`, a missing raw-TF floor, an unjudged-pool blocker, a shadow corpus).
**Three remain open**, and each would make the loop's numbers artifacts:

1. All 34 `conceptual` fixture queries are verbatim the concept's own name — query→answer
   is a dictionary lookup.
2. The `no_answer` stratum is empty (its authoring agent died mid-stream), so the
   abstention gate rule is vacuous.
3. The judgment pool covers ~7% of the corpus; `unjudged@10` is 0.63–0.95.

Ingesting a real corpus with real MeSH labels resolves most of this, which is why that is
the next step rather than more tuning.

---

## What's in the repo

```
src/oncolens/
  retrieval/   text (biomedical tokenizer) · chunking (offset-preserving) · lexical (BM25)
               dense (LSA now, Voyage-ready) · fusion (RRF) · expansion · rerank · pipeline
  eval/        metrics (graded, incomplete-judgment aware) · stats (paired permutation,
               bootstrap, multiple-comparison ledger) · bounds (dominance under unjudged
               docs) · gate (promotion rules)
  sources/     pubmed · pmc_cloud · europepmc · pmc_bulk (deprecated)
  serve/       artifact (bundled index) · neon_store (pgvector) · vercel_blob
  compare.py   cross-paper technical comparison
  spans.py     clause-level match highlighting
  sanity.py    degenerate baselines that validate the benchmark itself
  analysis.py  failure-mode diagnosis
  loop.py      propose → measure → gate → promote
api/           search · compare · evaluate  (Vercel Python functions)
app/ components/  Next.js site + WebGL background
docs/          MEASUREMENT.md · DEPLOYMENT.md · VERCEL_SETUP.md · CORPUS_SCHEMA.md
fixtures/      synthetic test data — NOT real papers
```

---

## How measurement is done (and what it can't claim)

Four things quietly break homemade RAG evaluations. Each has a countermeasure — details in
[`docs/MEASUREMENT.md`](docs/MEASUREMENT.md):

| Trap | Countermeasure |
|---|---|
| Queries generated from their target document | Paraphrase stratum forbids verbatim content-word overlap |
| One relevant doc per query — **penalises better systems** | Graded multi-relevant judgments, pooling, `bpref` |
| A single headline metric | 6-metric consensus panel; ≥4 must agree |
| Aggregate hides a collapsed query type | **Per-stratum gating** blocks any significant regression |

Plus paired permutation tests, a raw-TF floor (a 20-line scorer with no IDF), bounds-based
dominance so a verdict survives every possible labelling of unjudged documents, and an
`unjudged@10` guard that refuses to promote when a comparison is not interpretable.

The site shows these numbers **including the unflattering ones**. On the synthetic fixture
it reports `ndcg@10 = 0.5507` against a floor of `0.4669` — only +0.08 over a scorer with
no IDF — and says so.

⚠️ **One caveat to respect in production:** `ts_rank_cd` is not BM25. If the harness
measures BM25 in-process while Postgres serves `ts_rank_cd`, the evaluation stops
describing the product. `docs/DEPLOYMENT.md` ranks the three fixes.
