# Is Neon + pgvector the right store? A grounded evaluation

Written after ingesting a real corpus, so the numbers are measured rather than estimated.

## Measured facts about this workload

| Quantity | Measured |
|---|---|
| Documents ingested | 139 |
| Passages | 4,714 |
| Passages per document | **33–37** (real full text) |
| `chunks.text` total | 1.8 MB |
| `chunks` table incl. 192-dim vectors | **25 MB** |
| Bibliography share of raw text | 18.4% (removed at chunk time) |

**Extrapolation.** ~34 passages/document is the load-bearing number:

| Corpus | Passages | Vectors @192-dim float32 | Table size (est.) |
|---|---|---|---|
| 1k papers | ~34k | 26 MB | ~180 MB |
| 10k papers | ~340k | 260 MB | ~1.8 GB |
| 100k papers | ~3.4M | 2.6 GB | ~18 GB |
| All PMC OA (~6M) | ~200M | 154 GB | ~1 TB |

This matters more than any vendor comparison: **the answer changes at 10k papers.**

---

## The decisive property: this corpus is relational

The product's questions are joins, not just similarity:

- "what has my institution funded on X" → `documents ⋈ grants ⋈ institutions`
- "papers from the grants studying CDK4/6 escape" → traverse `funded_by`
- "compare these 5 papers on cohort size" → group passages by document, filter by aspect
- "MeSH concept → all documents carrying it" → array containment on `descriptors`

A pure vector store makes every one of these application code with N+1 fetches. That is
the strongest argument for Postgres and it is **workload-specific**, not a general claim
that Postgres beats vector databases.

---

## Options, honestly weighed

### Neon + pgvector — **current choice**

| For | Against |
|---|---|
| Joins, arrays, full-text and ANN in **one** query | **`ts_rank_cd` is not BM25** — see below, this is the real cost |
| pgvector on every plan, no add-on tier | Free tier is 0.5 GB — holds ~2k papers, not 10k |
| **Branch per Vercel preview** — previews never touch production data | HNSW build slows materially past ~1M vectors |
| Comfortable to ~50M vectors | Vectors and text compete for the same buffer pool |
| One vendor, one connection string, one backup story | Cold starts on scale-to-zero (the pooler endpoint mitigates) |

**Verdict: correct for now and for the realistic near term (≤10k papers).** The relational
requirement is real and immediate; the scale limits are not yet binding.

### Upstash Vector

Serverless, HTTP-native, works on Edge where TCP does not. Genuinely good at what it does.
But it stores vectors + opaque metadata — every join returns to application code, and this
corpus is join-heavy. **Reasonable only if the app becomes pure semantic search.**

### Pinecone

Mature, scales past anything here, strong filtered search. Costs meaningfully more than
Neon at this size, adds a second vendor and a second consistency story, and still no joins.
**Revisit above ~10M vectors** where pgvector's index maintenance becomes the bottleneck.

### Qdrant

Filtering is part of the indexing pipeline rather than a post-filter, so metadata-filtered
ANN is genuinely fast — attractive if per-document access control ever matters. Self-hosted
means infrastructure to run; cloud means another vendor. **Best alternative if filtering
becomes the bottleneck.**

### Turbopuffer / object-storage-backed

Very cheap per vector for large, cold, infrequently-queried corpora — a plausible fit for
"all of PMC OA". Higher tail latency. **Consider only at the 100k+ paper tier.**

### Bundled artifact (already built, `serve/artifact.py`)

Zero infrastructure, numpy-only runtime, 6.95 MB for 700 chunks. **Right for a demo or a
small fixed corpus; wrong past ~10k chunks** — it must fit in the function bundle and is
rebuilt wholesale on every change.

### SQLite + `sqlite-vec`

Excellent locally and for the offline harness. Wrong for serverless: no shared writable
filesystem, and every cold container would need the whole database.

---

## ⚠️ The real cost of choosing Postgres

**`ts_rank_cd` is not BM25.** The offline harness measures BM25; Postgres full-text search
scores differently. Measuring one and shipping the other means the evaluation stops
describing the product — which is precisely the failure this project is built to prevent.

Three fixes, best first:

1. **`pg_search` (ParadeDB)** — real BM25 in SQL, if the provider allows the extension.
2. **BM25 in SQL from stored statistics** — `BM25_SCHEMA_SQL` in `neon_store.py` stores
   per-term frequencies and corpus stats so scoring matches the offline index exactly.
   Roughly triples ingestion size.
3. **Re-run the evaluation against the deployed backend** and publish *those* numbers.

Option 3 is legitimate. Mixing them is not.

---

## Decision rule

| Corpus | Store |
|---|---|
| ≤ 1k papers, demo | Bundled artifact |
| **≤ 10k papers** | **Neon + pgvector** ← today |
| 10k–100k | Neon paid tier, or Qdrant if filtering dominates |
| 100k+ | Turbopuffer / Pinecone for vectors, Postgres retained for the relational half |

Full text stays in **Vercel Blob** at every tier: it is large, immutable, and never
queried — only fetched by URL once a passage has already been retrieved.

## Revisit when

- passages exceed ~5M (pgvector index maintenance becomes the bottleneck), **or**
- p95 query latency exceeds ~300 ms at the target corpus size, **or**
- per-document access control becomes a requirement (favours Qdrant's pre-filtering), **or**
- the relational queries above are dropped from the product — which would remove the main
  reason to prefer Postgres at all.
