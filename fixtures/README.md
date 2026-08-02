# fixtures/synthetic — NOT REAL DATA

> **Every document in `fixtures/synthetic/` is machine-generated. No real paper, grant,
> author, or institution is represented here. Do not use it for any research claim, do not
> ship it, and do not treat any number computed on it as evidence about real retrieval
> quality.**

## What this is

An LLM-authored corpus (140 documents across four oncology subdomains) plus a stratified,
graded relevance benchmark. It exists for one purpose: exercising the evaluation harness
offline, in an environment with no network access, so the metrics, statistics, gate and
loop can be tested without real data.

It is a **test fixture**, in the same category as `tests/test_pipeline.py`'s inline corpus
— just larger.

## Why it was quarantined out of `data/`

An audit found it was actively misleading, not merely fake:

* **92 documents carried fabricated PMIDs attached to real journal names.** They looked
  verifiable and were not. Spot-checked: PMID `28461425` appeared on a fabricated *Nature
  Communications* breast-cancer genomics paper; the real record is *"Correction to: Severe
  Pulmonary Vein Stenosis Resulting From Ablation for Atrial Fibrillation"* in
  *Circulation* (2017). A user following that citation would land on unrelated cardiology.
* **Invented principal investigators were attached to real institutions** (Dana-Farber,
  MD Anderson, Memorial Sloan Kettering).

All fabricated identifiers have since been stripped: `pmid`, `pmcid` and `doi` removed,
`pi` and `org` replaced with `SYNTHETIC-PI` / `SYNTHETIC-ORG`, and every document tagged
`meta.synthetic = true`.

## Known defects (documented, not fixed)

An adversarial audit of this benchmark found problems that make it unsuitable as an A/B
instrument in its current state:

| Defect | Status |
|---|---|
| All 34 `conceptual` queries are verbatim the concept's `preferred` string — query→answer is a dictionary lookup | **open** |
| `no_answer` stratum is empty (its authoring agent died mid-stream), so the abstention gate rule is vacuous | **open** |
| Judgment pool covers ~7% of the corpus; `unjudged@10` measures 0.63–0.95 | **open** |
| A 20-line raw-TF scorer reaches `ndcg@10` ≈ 0.467, so headroom is small | **open** |
| `doc_id`s encode the topic partition, making a 4-way classifier a free win | **open** |
| Corpus prose is unrealistically uniform (every doc 550–753 words, no statistics, no figure callouts) | **open** |
| Gold labels used to ship inside the corpus documents | fixed — labels now in `labels/`, and `load_corpus()` raises if it sees them |
| dev/test split leaked across information needs | fixed — split now clusters by grade-3 target overlap |

## Using it

```bash
ONCOLENS_DATA=fixtures/synthetic python scripts/run.py validate
```

`validate` will report the open defects above rather than pretending they are absent.

## The real data path

`data/` is reserved for real ingested content. See `docs/DEPLOYMENT.md` and:

```bash
python scripts/fetch_real.py --out data --max-papers 5000 --email you@org.edu --full-text
```

Real labels come from NLM's human MeSH indexing (`MajorTopicYN` gives graded relevance),
and real full text from the PMC Cloud Service — not from anything generated here.
