# CLAUDE.md — working notes for this repository

Context for anyone (human or model) picking this up. Everything here was **measured on
this repo's real data**, not assumed. Where a number is quoted, the command that produced
it is nearby.

---

## 1. The one rule

**This project exists to return the passage where a concept was mentioned.** Every design
decision defers to that. A retrieval change that improves ranking but loses
`(doc_id, section, start_char, end_char)` provenance is a regression, not a trade.

## 2. Hard-won environment facts

| Fact | Why it matters |
|---|---|
| `curl` is blocked; **Python `requests` is not** | An early conclusion of "no network" was drawn from curl alone and was **wrong**. Real ingestion works. Always verify with `requests` before declaring the network unavailable. |
| `pip install` works | Same reason. |
| Windows console is cp1252 | Real biomedical titles contain `∆`, `α`, `κ`. Printing them raises `UnicodeEncodeError` and kills long runs. Every user-facing script forces UTF-8 stdout. |
| The repo lives inside **OneDrive** | A churning corpus there causes sync storms and file locks — one already blocked a `rmtree` mid-run. Local artifacts go to `local_data_dir()` (`%LOCALAPPDATA%\oncolens`), never `REPO/data`. |
| `vercel env pull` returns `[SENSITIVE]` | Integration secrets are marked Sensitive and are **unreadable via CLI in every environment**, development included. Not a scoping problem — copy values from the dashboard. `env.py` treats the placeholder as absent so the error names the real cause. |
| Vercel Blob store is **private** | Private stores live on `*.private.blob.vercel-storage.com` and reject **every** REST upload (`Cannot use public access on a private store`). No `x-access` / `x-blob-access` / api-version combination works. Uploads route through `scripts/blob_bridge.mjs`, a persistent Node process using the official SDK. |

## 3. Data sources, and the deadline

| Source | Gives | Access |
|---|---|---|
| PubMed E-utilities | Abstracts, grant links, **NLM human MeSH indexing** | GET, unauthenticated |
| PMC Cloud (`s3://pmc-oa-opendata`) | **Verbatim full text** | Anonymous HTTPS, no AWS account |
| Europe PMC / Grist | JATS full text, real grants | GET |

**MeSH is the asset.** NLM's human indexers assign the descriptors and `MajorTopicYN`
already separates a paper's central topics from incidental ones, giving *graded* relevance
(3 = major, 1 = minor) that no LLM invented and no retriever influenced.

⏳ **NCBI retires the legacy PMC dataset files and the OA Web Service API on or after
24 August 2026.** `sources/pmc_cloud.py` targets the replacement. `sources/pmc_bulk.py`
targets the old FTP paths and is **deprecated**.

**Verified bucket layout** (probed 2026-08-01 — the documented `oa_comm/txt/…` paths 404):

```
metadata/PMC<id>.<ver>.json     licence code, retraction flag, exact object URLs
PMC<id>.<ver>/PMC<id>.<ver>.txt plain text extracted from JATS by NCBI
```

Drive ingestion from the metadata JSON, never from a guessed path convention.

### 3.1 Licence policy — a corrected mistake worth remembering

The original gate admitted only Creative Commons codes and reported "30 correctly skipped on
licence" as if that were good news. It was not. **`TDM` is the PMC code for Text and Data
Mining** — content publishers release *specifically* for this use — and the gate was
rejecting it for the sole reason that it is not Creative Commons. That is backwards.

Measured on a 28-article sample:

| Licence | Count | Old gate |
|---|---|---|
| CC BY | 15 | indexed |
| CC BY-NC-ND | 6 | skipped |
| **TDM** | **4** | **skipped — wrongly** |
| CC BY-NC | 3 | skipped |

`LICENSE_POLICIES` in `sources/pmc_cloud.py` now names the intent rather than hard-coding
one answer:

| Policy | Indexes | Use |
|---|---|---|
| `research` (**default**) | 28/28 (100%) | academic / internal research |
| `commercial` | 19/28 (68%) | a commercial product; excludes NC |
| `permissive_only` | 15/28 (54%) | the old behaviour; CC BY family only |

Effect of the fix: **58 → 88 documents with real full text.** The old default was discarding
46% of available text. ⚠️ If OncoLens is ever commercialised, switch to `commercial` — the
NC-licensed content in the index today is not licensed for that.

## 4. Measured design decisions

### 4.1 Reference stripping — now measured against publisher ground truth

**Problem, measured:** bibliographies are a **median 19% of full-text characters** (range
5.6%–42.4%, n=60). Citation strings match queries lexically while containing no findings —
**two of the top three hits** for *"osimertinib resistance mechanism"* were reference
entries.

**The method change that mattered more than any threshold.** The first version was tuned by
opening articles and judging whether the output looked right — the exact failure mode this
project exists to avoid. PMC's JATS XML carries `<ref-list>`, which is *the publisher's own*
statement of where the bibliography starts. `sources/jats.py` aligns it onto the plain-text
rendition, which turns this from taste into a labelled task.

**What the labels showed** (`scripts/analyze_ref_signals.py`, pairwise AUC over 60 articles;
1.0 = bibliography always scores higher, 0.5 = useless):

| signal (per 1000 chars) | body median | refs median | AUC | |
|---|---|---|---|---|
| years | 0.254 | 5.952 | **1.000** | decisive |
| DOIs | 0.034 | 4.090 | **1.000** | decisive |
| author initials (tolerant) | 0.730 | 24.503 | **1.000** | decisive |
| function-word fraction | 0.230 | 0.086 | **0.000** | decisive, inverted |
| page ranges | 0.255 | 2.431 | 0.983 | decisive |
| author initials (**old, strict**) | 0.588 | 13.303 | 0.900 | degraded by a regex bug |
| numbered entries | 0.000 | 0.000 | 0.692 | **noise — was weighted 0.20** |
| volume:page | 0.000 | 0.000 | 0.533 | **noise — was weighted 0.12** |

So the signals were never the problem. **Three specific defects were:**

1. **Per-word normalisation.** Made a 14,123-char block look like a 200-char one.
   Now normalised **per 1000 characters**.
2. **The trailing-run rule.** Required ≥3 reference-shaped paragraphs, but PMC commonly
   emits the whole bibliography as **one paragraph**, so run length was 1 and it never
   fired. Replaced by **suffix search** — the earliest point from which everything to the
   end is reference-dense — which is agnostic to how the rendition breaks lines.
3. **`_AUTHOR_INITIALS` required `[ ,;]`** after the initial, so the ACS/Nature style
   `Zhou J. Xu Y.` matched *nothing*. Adding `.` to the class moved AUC 0.900 → 1.000.

**Measured outcome — paired on identical articles** (`scripts/compare_ref_detectors.py`):

| Metric | Old | New |
|---|---|---|
| Bibliographies detected | 57/60 | **60/60** |
| Mean refs dropped | 0.9500 | **1.0000** |
| Mean body kept | 1.0000 | 0.9998 |
| Fully correct | 57/60 | **59/60** |

The one regression is **PMC13402827 — the article the old detector missed entirely**: it
trades 679 chars (CRediT author-contribution boilerplate plus figure captions) for removing
the whole bibliography. Net clearly positive, but it is a real loss and the figure/table
captions in it were useful.

Earlier notes here claimed a ~1-in-6 miss rate from an ad-hoc sample; **the measured rate on
labelled data is 3/60 (5%)**. Use the measured number.

#### The benchmark certified a rail that failed in production — read this one twice

After the rewrite scored 60/60 on labelled data, the **live index** still held 1,394
reference-shaped passages (2.35%). The mean was unremarkable; **the distribution was the
diagnostic**. They were concentrated — 215 in one document, 124 in another, 93 in a third —
and concentration means *undetected bibliographies*, not threshold noise.

Cause: `MAX_SHARE = 0.60`, which this file previously justified with "the largest true
bibliography in the labelled set is 42.4%, so it should never bind on real input."
**It bound on 180 of 1,739 documents.** All 60 labelled articles were primary research
papers. A **review article inverts the ratio**: PMC10958066, *"TGF-β signaling in health,
disease, and therapeutics"*, carries a single **480,996-character bibliography that is 80.6%
of the document**, so the suffix search broke on its first iteration.

Share was the wrong evidence. A review legitimately *is* mostly references. What must never
happen is deleting a document to nothing — an **absolute** question, not a proportional one.
The guard is now tiered:

| Condition | Allowed |
|---|---|
| `share ≤ 0.60` | ordinary evidence suffices |
| `share ≤ 0.92` **and** `density ≥ 0.70` | unambiguous evidence required |
| surviving body `< 1500` chars | **always refused** |

| Document | Dropped | Body retained |
|---|---|---|
| PMC10958066 | 80.7% | 115,435 chars |
| PMC11607834 | 61.1% | 118,851 chars |
| PMC11442786 | 63.0% | 80,984 chars |

Zero regression on the 60 labelled articles. **The transferable lesson is about the
benchmark, not the threshold**: a sample drawn from one document class certified a rule that
failed on another. When a component passes its benchmark, check its behaviour on the live
corpus and look at the *distribution* of failures, not the average.

**Stale rows are fixed.** `neon_store.upsert_chunks` now defaults to `replace=True`, which
deletes a document's existing chunks before inserting. Stripping removes ~20% of an article,
so re-ingestion previously left surplus rows behind and the metrics described a corpus the
code no longer produced. This was observed directly: 6,677 rows in `chunks` for a run that
produced 6,546.

`STANDALONE_THRESHOLD = 0.72` stays stricter than `BLOCK_THRESHOLD = 0.55`: the positional
stripper has position as evidence and structurally cannot delete mid-document text, a
standalone check has neither protection, and deleting findings is far worse than retaining
a few reference passages.

### 4.2 Chunk density needs real documents

| Corpus | Chunks/doc |
|---|---|
| Synthetic fixture | **5.0** — every section collapsed to one chunk |
| Real PMC full text | **33–37** |

At 5.0, section-aware chunking and chunk-aggregation strategies (`max` vs `topn_decay`)
were **inert knobs**. Tuning them on synthetic data would have measured nothing.

### 4.3 Comparative retrieval needs different retrieval

Top-k is the wrong shape for *"how do these studies measure X"*: it returns passages
clustered in the most on-topic paper, several not stating a method. `compare.py` adds
aspect conditioning (retrieve passages that *report* a dimension, numeric aspects requiring
an actual number), MMR with a **hard per-document cap**, and marks unreported dimensions
`NOT REPORTED` — a blank cell reads as *no effect* when it means *not measured*.

Verified: 4 papers × 3 aspects at 92% coverage spanning three subdomains. **Known
weakness:** the `effect` aspect often selects the same passage as `cohort`, because cohort
sentences also contain numbers.

### 4.4 Labels: what we have, and why bibliographies are the best source

Three label sources are available here. They are not interchangeable — they measure
different things, and the difference decides what a score means.

| Source | Granularity | Judge | Weakness |
|---|---|---|---|
| **MeSH** (`MajorTopicYN`) | document-level, topical | NLM human indexers | says a paper is *about* a topic; cannot separate measuring a mechanism from mentioning it |
| **Citation contexts** | claim-level, specific | the citing author | incomplete; popularity-biased |
| Hand judgments | anything | us | does not scale, and we would be grading our own homework |

**Citation contexts are found data of unusually good quality.** When an author writes

> "Acquired resistance to osimertinib is frequently driven by MET amplification [12]."

they have written a technical description of reference [12] *after reading it*. That
sentence is a query; the cited paper is a relevant answer; the judgment is free and expert.
It is also exactly the product's shape — a claim-level concept search, not a topic browse.

**It is not circular.** Labels come from JATS `<xref>` markup. We index the plain-text
rendition, where citation markers are already flattened away. Label source and indexed
representation are disjoint.

**Validity hazards, each with a guard in `eval/citation_labels.py`:**

1. **The citing paper contains the query verbatim** — it would rank #1 on string equality
   and the metric would measure nothing. Every query records `source_doc_id`, and
   `assert_source_excluded()` **raises** if it appears in the results. Asserted, not
   documented: a convention you can forget is not a guard.
2. **Judgments are incomplete** — other corpus papers may be equally relevant but simply
   weren't the one cited. Read `bpref` and `unjudged@k` alongside nDCG, never nDCG alone.
3. **Popularity bias** — a few landmark papers are cited constantly. Capped at
   `MAX_PER_TARGET = 4`, and `MAX_PER_SOURCE = 12` stops one review with 300 references
   from becoming the benchmark.
4. **Diffuse attribution** — *"several studies have shown X [3,7,11,14]"* asserts nothing
   specific. Grade falls with co-citation (3 sole → 2 → 1) and >3 co-cited is dropped.
5. **Contentless sentences** — *"as previously described [9]"* describes nothing. Filtered
   on content, requiring an assertive verb rather than a hand-listed vocabulary that would
   bias the benchmark toward terms we thought of.

**The yield number that drives corpus strategy.** On the first topically-sampled corpus
(139 papers), citation mining produced **3 labels from 4,973 contexts** — because **4,967
cited papers we do not hold**. 5,057 distinct cited PMIDs were missing, the most-cited
appearing 12–16 times each.

That is the argument for **snowball ingestion**: expanding the corpus along its own citation
graph both grows it *and* converts existing citations into labels, whereas topical sampling
adds papers whose citations point back out of the corpus. Sampling by topic gives a corpus;
sampling by citation gives a corpus **with a measurable structure**.
`scripts/build_citation_labels.py --snowball-out` → `ingest_real.py --pmids-file`.

### 4.5 The retrieval result: a component had to be deleted

Measured on 2,225 citation-context queries, paired permutation test, Bonferroni threshold
0.0125 within the iteration (`scripts/bench_retrieval.py`):

| system | nDCG@10 | Δ vs shipped | 95% CI | W/L |
|---|---|---|---|---|
| **lexical + OpenAI** | **0.4526** | **+0.0878** | [+0.0762, +0.0995] | 705/286 |
| OpenAI alone | 0.4116 | +0.0469 | [+0.0331, +0.0609] | 630/456 |
| **BM25 alone** | 0.3888 | **+0.0241** | [+0.0133, +0.0349] | 466/432 |
| lexical + LSA (**what shipped**) | 0.3647 | — | — | — |
| LSA alone | 0.3088 | −0.0560 | [−0.0643, −0.0478] | 220/599 |

**The shipping configuration was worse than deleting half of it.** BM25 alone beat the
lexical+LSA hybrid, p < 0.0001. `LsaBackend` is TF-IDF + SVD — a *lexical* model wearing a
dense coat — so it contributed little BM25 did not already have, while RRF gave it an equal
vote and it dragged the fusion down. This went unnoticed for the whole project because
there was no real query set until citation mining produced one.

⚠️ `unjudged@10 ≈ 0.94` at **1.10 judged documents per query**. These are lower bounds.
The gate in `docs/MEASUREMENT.md` blocks promotion above 0.35 unjudged and it is right to;
the *comparison* holds only because the unjudged rate is near-identical across systems.
Never quote an absolute number from this benchmark as "the" retrieval quality.

### 4.6 Embedding-space mismatch — a failure with no error

A query vector from one model compared against document vectors from another **does not
raise**. Cosine distance compares two unrelated 192-dim spaces happily and returns a
confident, meaningless ranking. This corpus was embedded with LSA, then re-embedded with
`text-embedding-3-small` at the *same* dimensionality — the column shape reveals nothing.

`index_config` records the backend; `assert_embedding_matches` runs before every query.
**Absent config is not permission**: an index with no record predates the table and holds
LSA vectors, so serving it with a newer encoder is precisely the dangerous case, and the
likely one. On mismatch: `503`, and **no fallback to the artifact** — answering from a
different index would hide the misconfiguration.

### 4.7 Serving must query the store, not a snapshot

`api/search.py` loaded a bundled artifact and never touched Neon, so the site served a
snapshot of a corpus it no longer had. At 59,306 passages the vectors alone are 45 MB
before any text, well past what belongs in a function bundle. `serve/live_query.py` is the
live path; the artifact is now the preview/offline fallback only.

Bulk writes need `COPY`, not `executemany`: 59,306 individual UPDATEs died mid-run with
*"SSL connection has been closed unexpectedly"*. `reembed_store.py` stages a batch via COPY
and joins it with one set-based UPDATE — two statements per batch instead of four thousand
— committing per batch and reconnecting on failure.

### 4.8 What the improvement loop measured — including about itself

The loop ran. These are its results, and the most useful ones are about the harness.

**Round 1, synthesis stratum (516 queries): 0 of 6 promoted.**

| candidate | outcome |
|---|---|
| `dense_only` (concept) | **regressed** success@10 −0.0707, p=0.0008 |
| `lexical_heavy` (synthesis) | **regressed** recall@20, recall@10, ndcg@10 |
| `expand_mesh` (synthesis) | **regressed** recall@20 — my null prediction was wrong |
| `deep_candidates`, `topn_decay` | no significant change anywhere |
| `rerank_llm` | mrr +0.0123 (p=0.030), ndcg@10 +0.0048 (p=0.012), recall@20 **exactly 0.0000** |

#### The finding: one global fusion weight is wrong in both directions

Dropping BM25 hurt 2-word concept queries; doubling BM25 hurt conceptual synthesis
queries. Not a contradiction — **the optimal weight is query-type dependent**. A short query
is mostly literal and needs the lexical arm; a long conceptual one carries enough context
for the dense arm to do better work. `adaptive_weights` sets the ratio from query length.

This finding only exists because the strata were kept separate. A pooled mean would have
averaged the two opposite effects into "roughly no change" and thrown it away.

#### Three bugs the loop found in itself

1. **RRF weights never reached fusion.** `reciprocal_rank_fusion` accepts `weights`; the
   harness never passed them, so `bm25_weight` controlled only whether an arm was
   *included*. `lexical_heavy` was byte-identical to baseline and the loop confidently
   DISCARDED a change that had never run. `judge()` now reports `NO_EFFECT` when rankings
   match the baseline exactly — a negative result indistinguishable from a broken
   candidate is worse than no result, because it looks like knowledge.
2. **A gate that could not be passed.** Synthesis is gated on `recall@20`, but reranking
   the top 24 chunks reorders them without changing *which* documents are in the top 20 —
   so `recall@20` moved by exactly 0.0000 and the reranker was structurally unable to win,
   however good it was. Gating a change on a metric it *cannot affect* is not a strict
   standard, it is a broken one.
3. **The consensus panel counted one fact six times** (see §4.5).

#### ⚠️ The evaluation is underpowered, and that changes the priority

`scripts/calibrate_fast_eval.py` was written to *shrink* the eval for faster iteration. It
proved the opposite:

| stratum | n | MDE (α=.05, power=.80) | sees a 0.03 effect? |
|---|---|---|---|
| synthesis | 516 | 0.0432 | **no** |
| concept | 426 | 0.0622 | **no** |
| identifier | 113 | 0.1318 | **no** — blind below 13 points |

Every promising effect in round 1 sat *below* its stratum's noise floor: `rerank_llm`'s
mrr +0.0123 against an MDE of 0.0432 is invisible **by construction**.

So **"0 of 6 promoted" is partly a statement about the instrument.** A real +0.02
improvement is currently reported as "no significant change" — which looks like rigour and
is blindness, the most dangerous failure a harness can have, because it is indistinguishable
from a well-behaved negative result.

To detect a 0.02 effect: synthesis needs **4.9×** more queries, concept **10.4×**,
identifier **45×**. **The bottleneck is evaluation power, not retrieval ideas.** Growing the
labelled set compounds — more corpus resolves more citations, which yields more synthesis
questions *and* more concept queries *and* more identifiers from one ingestion. Another
clever candidate measured at MDE 0.043 tells us nothing.

#### Promotion is by dominance, not by a composite

`eval/pareto.py`: a candidate ships if it improves at least one stratum and worsens none.
The weighted composite (§4.7) is retained **only** for deciding which candidate to try
next. Allocating attention with invented weights is legitimate; deciding what ships with
them is not.

### 4.9 Environment facts learned the expensive way

| Fact | Consequence |
|---|---|
| Python 3.14 has **no CUDA wheel on the `cu121` index** | torch silently installed as `+cpu`; MedCPT saturated 24 cores while two loops starved. `cu126` has a `cp314` build at the same version. |
| A first-call GPU benchmark measures **initialisation** | Cold: 19/sec → "50 min, no better than CPU". Warm: **148/sec → 6.6 min**. I nearly concluded the GPU didn't help. |
| Three CPU-bound jobs contending is slower than running them in sequence | Self-inflicted: 485 CPU-seconds across 24 minutes of wall clock for a starved job. |
| Piping a long run through `tail` buffers all progress until exit | Progress reporting exists and is unreadable; state had to be inferred from RSS and CPU counters. |

## 5. Evaluation — the part that is easy to fake

Read `docs/MEASUREMENT.md`. The short version of what protects this:

- **raw-TF floor.** A ~20-line scorer with no IDF and no length normalisation scores
  `ndcg@10 = 0.4669`. The system scores `0.5507` — only **+0.08**. Random (0.071) and
  popularity (0.115) are flattering floors; this is the real one.
- **6-metric consensus**, ≥4 must agree. One metric moving is a trade, not a win.
- **Per-stratum gating** — an aggregate mean rises while exact-identifier lookup collapses.
- **`bpref` returns `None` below 10 judged negatives** — at a denominator of 2 it was noise
  being averaged into a consensus vote.
- **`unjudged@10 > 0.35` blocks promotion outright.** Measured 0.63 — most returned
  documents were never judged, so the score is an underestimate of unknown size.
- **Bonferroni within an iteration**, not cumulatively. Correcting over every draw ever
  taken drives alpha to 0.05/38 and guarantees Type II errors; the locked test split is the
  real defence against cumulative overfitting.

**The improvement loop has never been run,** deliberately — three benchmark defects remain
open (see README). Running it would produce confident numbers that are artifacts.

## 6. Things that bit, so they don't bite again

1. **Guarding code instead of data.** The leakage guard scanned *source* for reads of
   `descriptors` and passed cleanly while all 140 corpus documents carried the labels
   inline. Guard the artifact, not the accessor.
2. **Fabricated identifiers.** The synthetic corpus attached real PMIDs to invented papers
   — PMID `28461425` sat on a fake *Nature Communications* paper while the real record is a
   *Circulation* cardiology correction. Never mint plausible identifiers.
3. **Import-time path binding.** `DATA = Path(os.environ.get(...))` at module scope
   silently ignored `ONCOLENS_DATA`; a run reported "0 docs" while appearing to succeed.
   Resolve config at call time.
4. **Heredocs with `\n` inside Python string literals** (writing files via
   `python - <<'EOF'`) repeatedly produced unterminated-string syntax errors. Use the Write
   or Edit tool for anything containing escape sequences.

## 7. Commands

```bash
python scripts/check_stores.py                 # preflight: Postgres + Blob, never prints secrets
python scripts/ingest_real.py --max-papers N --email you@org.edu [--dry-run] [--no-blob]
python scripts/query_neon.py "EGFR C797S" 5    # live query with clause offsets
python scripts/build_eval_report.py            # publishes the numbers the site displays
ONCOLENS_DATA=fixtures/synthetic python scripts/run.py validate
```

`fixtures/synthetic/` is **machine-generated, not real papers**. It exists to exercise the
harness offline. Never quote a number from it as evidence about real retrieval.
