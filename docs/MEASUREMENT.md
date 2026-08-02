# How OncoLens measures retrieval, and why it measures it that way

This document is the argument for the harness. Its claim is narrow and worth stating
plainly up front:

> **This benchmark is a reliable A/B instrument for comparing retrieval configurations.
> It is not evidence of absolute real-world retrieval quality.**

Everything below either supports that claim or marks its limits.

---

## 1. The default way to do this is wrong

The common recipe — ask an LLM to write one query per chunk, mark that chunk relevant, and
track recall@10 — fails in four independent ways, each of which biases the loop toward the
wrong answer.

### 1.1 Queries derived from their target are not queries

If the query is generated *from* the passage it is supposed to find, it shares that
passage's vocabulary. Lexical retrieval then looks extraordinary, dense retrieval looks
unnecessary, and the conclusion is an artifact of query construction.

**Here:** the `paraphrase` stratum forbids any content word of length ≥ 5 from appearing
verbatim in the top relevant document. The `conceptual` stratum is built on descriptors
that are deliberately **lexically silent** — at least 30% of gold assignments describe a
concept the document never names.

### 1.2 One relevant document per query punishes the systems you want

This is the most damaging error and the least obvious. If qrels mark a single relevant
document, then a *better* system that surfaces a second genuinely relevant document is
scored as if it retrieved junk. The measurement actively fights the improvement.

**Here:** queries carry 3–15 graded judgments spanning all four subdomains.
`scripts/run.py validate` counts single-relevant queries and flags them as a defect.
`experiment.pool_gaps()` lists every document that appeared in some system's top-10 but was
never judged, which is the work queue for keeping the pool current.

### 1.3 Binary relevance destroys ranking information

"The definitive paper" and "mentions the drug once in a Discussion" are not the same
result, and a metric that cannot tell them apart cannot reward putting the right one first.

**Here:** graded 0–3 with exponential-gain nDCG. A judgment of `0` is recorded explicitly
and means *checked, not relevant* — which is strictly more information than absent, and is
what makes `bpref` computable.

### 1.4 Recall-only evaluation rewards returning everything

A system that returns 50 documents for every query, including queries with no answer, will
beat a careful system on recall. In production it is worse.

**Here:** the `no_answer` stratum has empty judgments; the correct behaviour is to return
nothing. It is scored by `abstained` and `false_pos@10`, and the gate blocks any change
that increases false positives.

---

## 2. Metrics, and what each one is protecting against

| Metric | Role | Fails when |
|---|---|---|
| `ndcg@10` | Primary. Graded, rank-sensitive | Judgments are incomplete |
| `recall@{1,5,10,20,50}` | Coverage across the depth curve | Rewards dumping results |
| `precision@k` | Result-set purity | Undefined without negatives |
| `mrr` | Time-to-first-good-result | Ignores everything after rank 1 |
| `map` | Whole-ranking quality | Sensitive to pool depth |
| **`bpref`** | **Robust to incomplete judgments** | Needs judged negatives to exist |
| `unjudged@10` | **Diagnostic, not a score** | — |

No single number is trusted for *reporting*. But see §3 on why the consensus panel is not
what it appears to be: at 1.10 judgments per query its members are deterministic functions
of one rank, so "4 of 6 agree" is one fact counted six times, not six confirmations. The
panel is reported in full; the **decision** rests on one pre-registered metric plus a
regression veto.

### `unjudged@10` is the honesty valve

If a candidate's `unjudged@10` jumps, it is retrieving documents the pool never assessed.
Its score is then an underestimate *of unknown size*, and the comparison is not
interpretable. The gate refuses to promote in that situation rather than guessing.

`tests/test_metrics.py` demonstrates the underlying asymmetry directly: inserting one
unjudged document at rank 1 leaves `bpref` unchanged at 0.3333 while `ndcg@3` falls from
0.7985 to 0.4702.

---

## 3. Statistics: "the mean went up" is not a result

With ~150 queries and effect sizes in the 0.01–0.05 range, run-to-run differences are
routinely larger than real ones.

* **Paired randomization (permutation) test** for the p-value — the IR standard, no
  distributional assumption. Query difficulty varies far more than system quality, so
  everything is paired.
* **Percentile bootstrap CI** on the mean paired difference.
* **Cohen's `d_z`** plus explicit **win/loss/tie** counts. A change that wins on 8 queries
  and loses on 7 is not the same as one that wins on 40 and loses on 2, even at equal mean.
* **Holm across CANDIDATES, on one pre-registered gate metric.** An iterative loop that
  ignores multiplicity is a p-hacking machine by construction: enough challengers and
  something always "wins". But the correction has to be applied to the right family.

  This previously divided alpha by the size of the reported **metric panel** (5–8
  metrics). That was wrong and expensive. At **1.10 judged documents per query** nearly
  every query has exactly one relevant document, and with one relevant document at rank
  `r`, `mrr = 1/r`, `success@k = 1[r ≤ k]` and `ndcg@10 = 1/log2(r+1)` are all
  deterministic functions of the same number. Bonferroni assumes independence; correcting
  across near-perfectly correlated views controls nothing and inflates the minimum
  detectable effect by 1.22–1.28× — **1.5–1.6× more queries** — on a harness already
  short of power. Measured:

  | stratum | panel | MDE @ .05 | MDE @ the old gate |
  |---|---|---|---|
  | synthesis | 8 | 0.0569 | 0.0727 |
  | concept | 6 | 0.0622 | 0.0772 |
  | identifier | 5 | 0.1257 | 0.1533 |

  The family that genuinely needs controlling is the number of **candidates** tried in an
  iteration. So: one pre-registered gate metric per stratum (`weighting.gate_metric`),
  Holm-corrected across candidates. Holm dominates Bonferroni at the same family-wise
  error rate. Secondary metrics remain **regression vetoes tested uncorrected** — a false
  positive there means refusing a change, which is the safe direction.

  Cumulative correction over every draw ever taken is deliberately *not* used: it drives
  alpha to nothing and guarantees Type II errors. The locked `test` split is the real
  defence against cumulative overfitting.
* **Minimum detectable effect** is reported per stratum. Claiming "no regression" on a
  stratum of 12 queries is not a finding — `detectable_effect` says so out loud. Quote it
  at the alpha the gate actually uses, not at 0.05.

---

## 4. Holistic evaluation = per-stratum gating

An aggregate mean can rise while an entire query type collapses. That is the single most
common way a RAG system silently gets worse for real users: semantics improve, exact
identifier lookup dies, and the average looks fine.

The gate therefore evaluates **every stratum independently** and blocks promotion on any
significant regression beyond `STRATUM_TOLERANCE = 0.02`, regardless of the aggregate.

| Stratum | What it protects |
|---|---|
| `lexical` | Exact identifiers (`EGFR C797S`, `NCT…`) — where dense retrieval structurally cannot help |
| `conceptual` | Semantic reach when the words differ from the document's |
| `paraphrase` | Everyday phrasing of a technical idea |
| `multi_hop` | Grant→publication and citation traversal |
| `boolean_scope` | Conjunction/negation — where explicit `0` judgments make precision measurable |
| `no_answer` | Knowing when to return nothing |

---

## 5. Threats this harness does *not* eliminate

Stated because a validity section that only lists strengths is marketing.

1. **The corpus is authored, not collected.** Real grant prose is messier, more repetitive,
   and less internally consistent. Absolute numbers here will not transfer. Relative
   comparisons largely will, because the bias is shared across arms.
2. **Author-side correlation.** The corpus and the queries were produced by the same class
   of model. Some shared vocabulary bias is unavoidable. Mitigations: corpus and labels
   were frozen *before* any retriever existed; query authors were blocked from reading the
   retrieval lexicon; the lexicon was built without reading the gold concept space.
3. **Ontology contamination.** If the retrieval lexicon overlaps the gold concept space,
   expansion wins by construction. `contamination_report()` quantifies the overlap and it
   is printed on every validation run. It is discounted, not ignored.
4. **Pooling is shallow.** Judgments cover the union of what the configurations tried so
   far. A genuinely novel retriever will have inflated `unjudged@10` and be under-credited
   until its results are judged.
5. **Dev-set overfitting persists** despite Bonferroni. The locked `test` split is opened
   exactly once; the dev/test gap is the estimate of how much of the gain was fitted, and
   it is reported whether or not it is flattering.
6. **LSA is not a neural embedder.** It has the right qualitative profile — strong on
   paraphrase, weak on unseen identifiers — but conclusions about *how much* dense
   retrieval helps must be re-derived once a real embedding backend is connected.

---

## 6. Label provenance, and what it would be on real data

The offline label sources deliberately mirror the "found data" that exists in the real
corpus, so the harness ports over when network access exists.

| Harness `source` | Real-world equivalent | Independence |
|---|---|---|
| `descriptor` | NLM MeSH indexing (human indexers) | High — human, pre-existing |
| `funding_link` | NIH RePORTER grant→publication | High — asserted by the PI |
| `citation_ctx` | Citation contexts (SPECTER/SciNCL) | High — author-asserted |
| `pooled` | TREC-style pooled judgments | Highest — but expensive |

**Migration path.** Swap the corpus loader to real NIH RePORTER + PMC OA data and the
labels come almost free: MeSH terms are already attached to every PubMed record,
grant→publication links ship with RePORTER, and citation contexts are extractable from PMC
full text. The metrics, statistics, gate and loop need no changes — only `data.py` does.
That is the main reason the harness is written backend- and corpus-agnostic.

---

## 7. The promotion gate

A challenger is committed only if **all** hold:

1. the stratum's **pre-registered gate metric** improves by more than `MIN_EFFECT`, and
   survives **Holm across the candidates in the iteration**;
2. **no secondary metric regresses significantly** (tested uncorrected — see §3);
3. no stratum regresses significantly beyond 0.02;
4. `no_answer` false positives do not rise and abstention does not fall;
5. `unjudged@10` does not spike by more than 0.02 relative to baseline;
6. underpowered strata are reported rather than silently passed;
7. the candidate's rankings are **not byte-identical to baseline** — a candidate that
   never fired reports as `NO_EFFECT`, not as a well-behaved negative.

Rules 2–5 and 7 are what distinguish this from "the number went up".

⚠️ **On the absolute `unjudged@10 > 0.35` figure quoted elsewhere:** the citation
benchmark measures ≈0.94, so an absolute gate at 0.35 would block every promotion
permanently. The gate that actually runs is on the **delta** (rule 5): a candidate must not
*increase* the unjudged rate. Absolute unjudged remains a reason to distrust any absolute
score from this benchmark — only paired comparisons are interpretable — but it is not a
promotion criterion, because it cannot be met.
