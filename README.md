# OncoLens

Retrieval over oncology research **grants and papers** that returns the associated
documents **and the exact passage where the concept was mentioned**.

That second half is the product requirement, and it drives the architecture: every chunk
carries `(doc_id, section, start_char, end_char)` back to its source, so a result can
always be shown in context rather than paraphrased.

---

## What is here

```
src/oncolens/
  retrieval/     text (biomedical tokenizer) · chunking (section-aware, offset-preserving)
                 lexical (BM25) · dense (LSA now, Voyage-ready) · fusion (RRF/score)
                 expansion (ontology, with contamination accounting) · pipeline (config-driven)
  eval/          metrics (graded, incomplete-judgment aware) · stats (paired permutation,
                 bootstrap, multiple-comparisons ledger) · gate (the promotion rules)
  data.py        corpus/qrels loading with integrity checks that fail loudly
  experiment.py  run a config, record everything needed to re-judge it later
  loop.py        propose -> measure -> gate -> promote
  configs.py     the iteration ladder
docs/
  MEASUREMENT.md   why the harness measures the way it does, and what it cannot claim
  CORPUS_SCHEMA.md the frozen corpus + label schema
tests/           hand-verified tests for the measurement engine itself
```

## Quick start

```bash
python tests/test_metrics.py        # verify the measurement engine first
python scripts/run.py validate      # dataset integrity, leakage + contamination audit
python scripts/run.py iterate       # run the improvement ladder against dev
python scripts/run.py test          # open the locked test split, once
python scripts/run.py demo "EGFR C797S resistance"
```

## The claim, and its limits

**This harness is a reliable A/B instrument for comparing retrieval configurations. It is
not evidence of absolute real-world retrieval quality.**

The corpus is authored offline (no network or API access in the build environment), so
absolute numbers do not transfer to real NIH/PMC data. Relative comparisons largely do,
because the bias is shared across arms. `docs/MEASUREMENT.md` states the threats to
validity that remain, including the ones that are not fully mitigated.

## Why the measurement is built this way

Four things quietly break homemade RAG evaluations. Each has a specific countermeasure:

| Trap | Countermeasure |
|---|---|
| Queries generated from their target document | `paraphrase` stratum forbids verbatim content-word overlap; `conceptual` labels are lexically silent |
| One relevant document per query — **penalises better systems** | Graded multi-relevant judgments, TREC-style pooling, `bpref` |
| A single headline metric | 6-metric consensus panel; ≥4 must agree |
| Aggregate hides a collapsed query type | **Per-stratum gating** — any significant regression blocks promotion |

Plus: paired permutation tests, Bonferroni over a ledger of every dev draw, per-stratum
minimum-detectable-effect reporting, and an `unjudged@10` guard that refuses to promote
when the comparison is not interpretable.

## Swapping in real data and real models

The harness is corpus- and backend-agnostic on purpose.

* **Real corpus** — replace `data.py`'s loader with NIH RePORTER (`POST /v2/projects/search`)
  + PMC OA (BioC XML already carries passage offsets). Labels then come mostly free: MeSH
  terms are human-assigned on every PubMed record, grant→publication links ship with
  RePORTER, and citation contexts are extractable from PMC full text.
* **Real embeddings** — set `VOYAGE_API_KEY` and `dense_backend="voyage"`. `VoyageBackend`
  is already wired, including the asymmetric `input_type` query/document distinction.
* **Grounded answers** — Claude's Citations feature consumes the passages this pipeline
  already produces; `Evidence` carries the offsets it needs.
