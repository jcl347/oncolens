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
| **Real corpus in Neon** | **1,754 documents, 105,250 passages — every one with verbatim full text** |
| Vercel Blob (private store) | Working via `scripts/blob_bridge.mjs` |
| Retrieval improvement loop | **Run (round 2).** One candidate promoted on *dev*; nothing shipped — see *What the loop found* |

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

Measured on **2,225 citation-context queries** over the 1,739-document corpus *as it stood
at the time* — the corpus has since been cleaned of abstract-only records and expanded to
1,754 full-text documents, and the label set to 3,998 claim queries, so these are a dated
result and not a current one. Paired permutation test against the previously-shipping
configuration:

| system | nDCG@10 | recall@10 | MRR | Δ vs shipped | 95% CI | W/L |
|---|---|---|---|---|---|---|
| **lexical + OpenAI** | **0.4526** | **0.6082** | **0.4096** | **+0.0878** | [+0.0762, +0.0995] | 705/286 |
| OpenAI alone | 0.4116 | 0.5678 | 0.3681 | +0.0469 | [+0.0331, +0.0609] | 630/456 |
| **BM25 alone** | 0.3888 | 0.5318 | 0.3496 | **+0.0241** | [+0.0133, +0.0349] | 466/432 |
| lexical + LSA *(what shipped)* | 0.3647 | 0.5206 | 0.3218 | — | — | — |
| LSA alone | 0.3088 | 0.4646 | 0.2649 | −0.0560 | [−0.0643, −0.0478] | 220/599 |

**The most useful result was that a component had to be deleted.** BM25 *on its own* beat
the lexical+LSA hybrid that shipped (p < 0.0001). TF-IDF + SVD is a lexical model wearing a
dense coat: it added little BM25 did not already have, while RRF gave it an equal vote. The
second-largest measured win available was removing it.

⚠️ **These are lower bounds, not quality estimates.** `unjudged@10 ≈ 0.94` — about 94% of
returned documents were never judged, at **1.05 judged documents per query on `claim`**
(that figure is a pooled mean and does not describe `concept`, which carries 13.3). The
*comparison* between systems survives because the unjudged rate is near-identical across
them (0.9348–0.9495); no absolute number here should be quoted as "the" retrieval quality.

An earlier version of this note said the gate "blocks anything above 0.35 unjudged, and
that gate is right". At a measured 0.94 such a gate could never be satisfied by anything,
so it was not a strict rule but an inoperative one. What the loop actually enforces is a
bound on the **change** in `unjudged@10`: a candidate must not inflate it.
`bpref` is absent by design — it returns `None` below 10 judged negatives rather than
averaging noise into a consensus vote.

Reproduce:

```bash
python scripts/build_citation_labels.py --email you@org.edu
python scripts/bench_retrieval.py --systems bm25 lsa openai hybrid-lsa hybrid-openai
```

### Embedding-space mismatch — the silent failure this design guards

A query vector from one model compared against document vectors from another **does not
raise**. Cosine distance compares two unrelated 192-dimension spaces perfectly happily and
returns a confident, meaningless ranking. This corpus was embedded with LSA first and
`text-embedding-3-small` later at the *same* dimensionality, so nothing about the column's
shape reveals a mismatch.

`scripts/reembed_store.py` records the backend in an `index_config` table and every query
checks it before running. **An absent record is not treated as permission**: an index with
no recorded backend predates the table and therefore holds LSA vectors, so serving it with a
newer encoder is exactly the silent-nonsense case. On mismatch the API returns `503` and
deliberately does *not* fall back to the bundled artifact, because answering from a
different index would conceal the misconfiguration the check exists to surface.

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

## What the loop found

It was held back for a long time because the benchmark had known defects. That caution was
half right: several more defects surfaced in the running of it (a multiplicity correction
applied to the wrong family, a gate metric written but never wired in, a `SELECT` with no
`ORDER BY` silently defeating the embedding cache). But the loop's first properly-powered
result **reversed the sign** of a conclusion the project had carried since round 1, and no
further inspection would have found that — only data did.

**The headline: MedCPT is a trade, not an improvement — and the composite score would have
shipped it anyway.**

`medcpt` swaps the dense arm for NCBI's MedCPT, trained on 255M PubMed click logs.
`openai_768` is its **control** — the same general embedder widened to the same 768
dimensions — so a MedCPT win could not be confused with having four times the vector
capacity.

| stratum | weight | the task | Δ medcpt | Δ openai_768 |
|---|---|---|---|---|
| synthesis (n=896) | 0.35 | coverage of a paper **set** | **+0.0261** ✓ p=0.0003 | +0.0016 |
| concept (n=252) | 0.30 | 2-word topical lookup | +0.0198 *(MDE 0.066 — blind)* | +0.0079 *(blind)* |
| claim (n=2,887) | 0.15 | find the **one** source of a sentence | **−0.0166** ✗ p=0.0034 | **+0.0093** ✓ p=0.0024 |

MedCPT is a better **topical matcher** and a worse **pinpointer**. Click logs encode "this
article is about what you asked", not "this is the exact sentence you want" — so it wins
literature-review coverage and loses find-the-source, where the same smoothing costs exact
attribution. Its registered hypothesis called `claim` a *null* stratum; it significantly
regressed there instead.

**The composite and the dominance rule disagree, and the dominance rule wins.** Weighting
those deltas gives medcpt **+0.0126** against openai_768's **+0.0043** — a 3× "win". But
promotion here is by Pareto dominance: improve at least one stratum, worsen none. MedCPT
regresses `claim`, so it is refused; `openai_768` is not. A weighted average would have
bought +0.026 on review coverage with −0.017 on find-the-source and called it progress.
Those are different jobs for the same user, and the weights trading them off were chosen by
us, not measured.

MedCPT is not dead — it is a **query-type-conditional** win, which is the hypothesis for
round 3: route synthesis-shaped queries to it, leave the rest on the servable arm. That
avoids moving the whole index to 768 dims, ~2 GB of torch, and inference that does not fit
a Vercel function.

⚠️ `openai_768` is promoted on **dev only**. Shipping still requires the locked `test`
split, which is unspent.

A third candidate, `mmr_diversify`, returned **NO_EFFECT — byte-identical rankings on all
eight metrics**. It reorders passages, but document scores are aggregated by `max`, which
depends on the *set* of a document's passage scores and not their order — so it is a no-op
by construction. That is the third time in this project a candidate has been structurally
incapable of moving the metric it was gated on.

**What made the difference was data, not cleverness.** Expanding the corpus along its own
citation graph took the labelled set from 2,225 to 3,998 claim queries and 0 to 1,272
synthesis questions — converting citations *already mined* into labels, because the cited
paper is now held. That is the only lever that has ever moved the measured bottleneck.

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
