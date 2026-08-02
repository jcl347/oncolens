# Is Neon + pgvector the right store? A grounded evaluation

Written after ingesting a real corpus, so the numbers are measured rather than estimated.

## Measured facts about this workload

Re-measured at 1,739 documents. The earlier 139-document figures are kept in the last
column because the *prediction they supported* can now be checked, which is more useful
than quietly replacing them.

| Quantity | Measured @1,739 docs | Was @139 docs |
|---|---|---|
| Documents ingested | **1,739** | 139 |
| Passages | **59,306** | 4,714 |
| Passages per document | **34.1** | 33–37 |
| `chunks.text` total | **53.8 MB** | 1.8 MB |
| Database total | **345 MB** | 25 MB |
| Bibliography share of raw text | 19% median (5.6%–42.4%) | 18.4% |

**The extrapolation was close.** It predicted ~180 MB per 1,000 papers; the measured figure
is **198 MB per 1,000 papers** — 10% optimistic, and the shape of the curve was right.

| Corpus | Passages | Projected DB size (at 198 MB/1k) |
|---|---|---|
| 1k papers | ~34k | ~198 MB |
| **1.7k papers** | **59k** | **345 MB (measured)** |
| 10k papers | ~341k | ~2.0 GB |
| 100k papers | ~3.4M | ~20 GB |
| All PMC OA (~6M) | ~200M | ~1.2 TB |

### Where the 345 MB actually goes — measured, not assumed

| Component | Size | Note |
|---|---|---|
| `chunks` indexes | 186 MB → **100 MB** | after dropping an unused index, below |
| `chunks` TOAST (text columns) | ~135 MB | `text` 53.8 MB + `indexed_text` 60.7 MB |
| `chunks` heap | 99 MB | |
| `documents` | 2.8 MB | negligible |

**An unused index was 20% of the database.** `chunks_trgm_idx`, a GIN trigram index on
`indexed_text`, measured **86 MB with `idx_scan = 0`** — the planner had never once used it.
Dropping it took the database from 431 MB to 345 MB with no functional change. Check
`pg_stat_user_indexes.idx_scan` before adding storage; an index nobody queries is pure cost.

Two further levers exist and are **deliberately not taken yet**, because both trade
retrieval quality for space and there is now a 2,225-query benchmark that can settle them:

| Lever | Saving | Why it is not free |
|---|---|---|
| Drop `indexed_text`, regenerate `tsv` | −61 MB | `tsv` is a *generated column* from it, so this removes the title and section from the searchable text |
| 192 → 128 dims (Matryoshka) | −37 MB | should cost little; "should" is not a measurement |

### ⚠️ The capacity that matters is not the steady-state size

Re-embedding the corpus — a single `UPDATE` of every row's vector — **failed** with

```
psycopg.errors.DiskFull: could not extend file because project size limit (512 MB)
has been exceeded
```

at a steady-state size of 345 MB. Nothing was being added. Under MVCC an `UPDATE` writes a
*new* row version and leaves the old one dead until vacuumed, so rewriting every row grows
the table by roughly its own size, and the HNSW index bloated **66 MB → 115 MB** at the
same time. Measured mid-failure: 40,000 dead tuples, database at 490 MB.

**So the usable capacity for a corpus you intend to maintain is roughly half the quota**,
not the quota. This is the single most important correction to the sizing table above, and
it is invisible if you only measure a freshly-loaded database.

The procedure that works, and why each step is there:

```sql
DROP INDEX chunks_embedding_idx;   -- 115 MB, and re-embedding invalidates it anyway
VACUUM chunks;                     -- reclaim dead tuples IN PLACE
--   ... bulk UPDATE via COPY into a staging table, vacuuming every few batches ...
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
```

* **Drop the vector index first.** Updating 59k vectors in an HNSW index costs far more
  than rebuilding it, and the bloat is what pushes you over the limit.
* **Plain `VACUUM`, not `VACUUM FULL`.** `FULL` rewrites the table and needs the headroom
  you are trying not to consume; plain `VACUUM` makes dead space reusable in place. It
  took 490 MB → 375 MB here purely by dropping the index and vacuuming.
* **`COPY` into a staging table, not `executemany`.** 59,306 individual `UPDATE`s over a
  serverless connection died with *"SSL connection has been closed unexpectedly"*.

**This is a genuine mark against Postgres for this workload.** A dedicated vector store
treats "replace all vectors" as a normal reindex; in Postgres it is a table rewrite with a
transient 2× footprint. It does not overturn the decision — the relational requirement is
still real and immediate — but it means the free tier supports roughly **1,300 maintainable
papers**, not the 2,700 the steady-state arithmetic suggests.

### The free tier's ceiling arrived exactly where predicted

The note below said the 0.5 GB free tier "holds ~2k papers, not 10k". Neon warned at 89%
of 0.54 GB with **1,739 papers** indexed. The prediction was correct and the constraint is
real: **this workload outgrows the free tier at roughly 2,700 papers.**

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
