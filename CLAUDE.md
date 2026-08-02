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

## 4. Measured design decisions

### 4.1 Reference stripping — real, partial

**Problem, measured:** 9.1% of ingested passages (203/2233) were bibliography. Citation
strings match queries lexically while containing no findings — **two of the top three hits**
for *"osimertinib resistance mechanism"* were reference entries.

**What didn't work, and why:**
- *Heading detection.* PMC's txt rendition emits only `JOURNAL INFORMATION` and
  `ARTICLE INFORMATION` as capitalised headings. There is no `REFERENCES` heading in that
  form — it appears as its own short paragraph.
- *Per-paragraph classification alone.* Real prose citing `Chen et al. (2019)` plus a DOI
  scores 0.500 — above the 0.45 paragraph threshold. Classifying independently deletes
  findings.
- *Trailing-run detection alone.* Fired **zero times** on real articles, because PMC emits
  the entire bibliography as **one paragraph** (14,123 chars, 55 entries). The run length
  was 1, below `MIN_RUN=3`.

**What works:** heading paragraph (last 50% only) → single large high-scoring trailing
block → trailing run, in that order. Position is the disambiguator: one reference-shaped
paragraph mid-document is a citation; a large block at the end is a bibliography.

**Measured outcome — report this honestly:**

| Metric | Before | After |
|---|---|---|
| Bibliography share of full-text characters | — | **18.4% removed** |
| Passages reference-shaped (strict ≥0.75) | 9.1% | **5.8%** |
| Bibliographies detected | — | **5 of 6 articles** |

**It is not solved.** ~1 in 6 bibliographies is missed, and a reference block still ranked
first for the test query. Two known causes: detector misses, and **stale rows** — ingestion
upserts rather than replaces, so passages from a pre-stripping run persist. Clear `chunks`
before re-measuring.

`STANDALONE_THRESHOLD = 0.75` is deliberately stricter than `PARA_THRESHOLD = 0.45`: the
positional stripper has position as evidence and structurally cannot delete mid-document
text, a standalone check has neither protection, and deleting findings is far worse than
retaining a few reference passages.

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
