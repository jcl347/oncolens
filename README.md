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
| Vercel storage (Blob + Neon/pgvector) | **Provisioned and verified** — `python scripts/check_stores.py` |
| **Real corpus in Neon** | **139 documents, 4,714 passages, 58 with verbatim full text** |
| Vercel Blob (private store) | Working via `scripts/blob_bridge.mjs` — 58 articles uploaded |
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

## Design decisions, and what each one measured

Every choice below is recorded with the number that justified it. `CLAUDE.md` has the full
working notes; `docs/STORAGE_DECISION.md` has the storage evaluation.

### Reference stripping (`retrieval/references.py`)

PMC plain text includes the bibliography, and citation strings retrieve well while
containing no findings. Measured across 60 publisher-labelled articles, bibliographies are
a **median 19% of full-text characters** (range 5.6%–42.4%).

**This is measured against ground truth, not judged by eye.** PMC's JATS XML carries
`<ref-list>` — the publisher's own statement of where the bibliography starts.
`sources/jats.py` aligns it onto the plain text, which turns a matter of taste into a
labelled task:

```bash
python scripts/analyze_ref_signals.py --limit 60   # which signals actually separate?
python scripts/bench_references.py    --limit 60   # score the detector
python scripts/compare_ref_detectors.py --ref HEAD # paired old vs new, same articles
```

The labels showed the original signals were fine and the *design* was wrong in three
specific ways: per-**word** normalisation (a 14k-char block scored like a 200-char one), a
trailing-**run** requirement that could not fire when PMC emits the whole bibliography as
one paragraph, and an author-initials regex requiring `[ ,;]` so the ACS style `Zhou J.
Xu Y.` matched nothing. Two signals carrying hand-assigned weights of 0.20 and 0.12 turned
out to be **pure noise** (AUC 0.692 and 0.533) and were removed.

Replaced by **suffix search** over per-1000-character density — the earliest point from
which everything to the end is reference-dense — which is agnostic to how the rendition
happens to break lines.

| Metric (paired, identical articles) | Old | New |
|---|---|---|
| Bibliographies detected | 57/60 | **60/60** |
| Mean refs dropped | 0.9500 | **1.0000** |
| Mean body kept | 1.0000 | 0.9998 |

The single regression is the article the old detector missed *entirely*: it trades 679 chars
of author-contribution boilerplate for removing the whole bibliography.

**Stale rows fixed.** `upsert_chunks` now defaults to `replace=True`. Stripping removes ~20%
of an article, so re-ingestion previously left surplus rows behind — observed directly as
6,677 rows for a run that produced 6,546 — and the metrics then described a corpus the code
no longer produced.

### Labels from citation contexts (`eval/citation_labels.py`)

When an author writes *"resistance to osimertinib is driven by MET amplification [12]"*,
that sentence is a technical description of reference [12] written by someone who read it.
It is a query; the cited paper is a relevant answer; the judgment is free and expert — and
it is **claim-level**, where MeSH is only document-level topical.

Not circular: labels come from JATS `<xref>` markup, while retrieval indexes the plain-text
rendition where citation markers are already flattened away.

Guards for the ways this could quietly measure nothing:

| Hazard | Guard |
|---|---|
| The citing paper contains the query **verbatim** | `assert_source_excluded()` **raises** if it appears in results |
| Judgments are incomplete | report `bpref` + `unjudged@k`, never nDCG alone |
| Popularity bias | `MAX_PER_TARGET=4`, `MAX_PER_SOURCE=12` |
| *"several studies show X [3,7,11]"* | grade falls with co-citation; >3 dropped |
| *"as previously described [9]"* | rejected — requires an assertive verb |

**The yield number that drives corpus strategy:** on a 139-paper topically-sampled corpus,
mining produced **3 labels from 4,973 citation contexts**, because **4,967 cited papers we
do not hold**. Sampling by topic gives a corpus; sampling *along the citation graph* gives a
corpus with measurable structure:

```bash
python scripts/build_citation_labels.py --snowball-out pmids.txt
python scripts/ingest_real.py --pmids-file pmids.txt --max-papers 1600
```

### Chunk density — why real data was necessary

| Corpus | Chunks/doc |
|---|---|
| Synthetic fixture | **5.0** |
| Real PMC full text | **33–37** |

At 5.0 every section collapsed to a single chunk, making section-aware chunking and the
`max` vs `topn_decay` aggregation knobs **inert**. Tuning them on synthetic data would have
measured nothing.

### Storage: Neon + pgvector

Chosen because **this corpus is relational** — grants ↔ publications ↔ PIs ↔ institutions,
and those joins are half the product. Measured: 4,714 passages = 25 MB including 192-dim
vectors, extrapolating to ~1.8 GB at 10k papers. Correct to ~10k papers; the decision rule
and the alternatives (Upstash, Pinecone, Qdrant, Turbopuffer, bundled artifact) are weighed
in `docs/STORAGE_DECISION.md`.

⚠️ Its real cost: **`ts_rank_cd` is not BM25**, so the harness and production currently
score differently. Three fixes are ranked in that document.

### Private Blob store

Vercel Blob private stores live on a different host and reject **every** REST upload. No
`x-access` / `x-blob-access` / api-version combination works. Uploads route through
`scripts/blob_bridge.mjs`, a persistent Node process using the official SDK — reused across
calls because spawning Node per article would dominate a multi-thousand-document ingest.

### Retrieval quality, stated plainly

`ndcg@10 = 0.5507` against a **raw term-frequency floor of 0.4669** — a scorer with no IDF,
no length normalisation, no chunking, no dense arm. That is **+0.08**, and the site says so
rather than hiding it. Random scores 0.071 and popularity 0.115; those are flattering
floors, and quoting them instead would be misleading.

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
