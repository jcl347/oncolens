# Plan: digesting what is inside the image

Every number below is measured on this repo's corpus (`scripts/audit_figures.py`,
`scripts/audit_figure_refs.py`, `scripts/size_figure_gap.py`) or cited. Nothing is asserted
from intuition, because the intuitive ranking of these five approaches turns out to be
wrong for this corpus in two places.

---

## 1. What is actually here

Measured on 3,251 cached JATS documents:

| | measured | consequence |
|---|---|---|
| documents with ≥1 figure | 97.7% | figures are not a niche case |
| total figures | **16,792** (median 5/doc) | one-off VLM pass is a batch job, not a service |
| figures with a caption | 99.0% | captions are near-universal |
| figures with an image reference | **99.9%** | the picture is fetchable for essentially all |
| tables that are machine-readable `<table>`, not pictures | **95.3%** | table extraction is already done |
| median caption length | **934 chars** | these are descriptive paragraphs, not one-liners |
| **caption text already in the index** | **median 100% token coverage; 98.7% of captions ≥90% covered** | ⚠️ see below — this reverses the plan's original premise |

### ⚠️ Correction: the captions are already indexed

An earlier version of this plan claimed **17.4M characters of caption text were absent from
the index** and built "Stage 0 — index the captions, no models, the largest certain gain"
on top of it. **That was wrong.** It was reasoned — *figures are images, so nothing survives
the plain-text rendition* — and never checked.

NCBI's plain-text rendition **inlines figure captions into the body flow**. Measured against
the live index on 250 sampled documents / 1,113 captions:

| | measured |
|---|---|
| median caption token coverage in indexed text | **100.0%** |
| captions ≥90% covered | **98.7%** |
| captions <50% covered | **0.1%** |

Verbatim from `chunks`, showing the caption jammed onto the preceding sentence:

> `...(I2 = 81).Figure 1. Meta-analysis. (A) Rates of all-grade infections and (B) grade ≥3
> infections among patients with multiple myeloma treated with bispecific antibodies...`

The 17.4M figure was double-counting text already inside the ~163M body total. **Stage 0 as
written would have been a candidate incapable of moving its gate** — the fifth instance of
§4.13's fault in this project, and the first caused by asserting a fact rather than
mis-wiring one.

**What this does to the value case.** Caption text is already in both the BM25 and dense
arms, and §1's measurement says **81.1% of the numbers authors cite against a figure are
already in the caption**. So text retrieval already holds most of the figure *signal*. The
figure programme is therefore **not primarily a retrieval-quality project**. What is
genuinely missing is:

1. **the image as a returnable object** — nothing links a chunk to `image_uri`, so a chart
   cannot be shown even when its caption is what matched. This is product value, and it
   will not show up in nDCG;
2. **addressability** — no `figure_id`, no `kind='figure'`, so "show me the survival curves"
   is unanswerable as a query shape;
3. **the pixel-only residue** — the ~18.9% of cited numbers absent from captions, plus
   everything in a plot that no sentence ever mentions;
4. **a real if minor text defect** — `.Figure 1.Meta-analysis.` concatenation corrupts the
   sentence boundary of both the body sentence and the caption, and a ~900-char chunk
   boundary can split a 934-char caption in half.

Evaluate this work as **returning figures as first-class objects**, not as improving
ranking. Registering it as a ranking win would set up a null result that looks like failure
when it is the predicted outcome.

And three numbers that reorder the design space:

| | measured | what it means |
|---|---|---|
| figure queries sharing **<40%** content words with the caption | **49.6%** | **upper** bound on the pixel opportunity |
| figure-cited **numbers** that already appear in the caption | **81.1%** | strong argument *against* pixels for quantitative lookup |
| figures with any **plottable chart** component | **27.5%** | chart-to-table is undefined for ~72% |

Figure composition by what the caption says it is: 22.7% imagery (blots, micrographs,
scans), 14.1% plottable chart alone, 9.8% chart+imagery, 6.5% schematic, 40.5%
unclassified. **A western blot and an H&E slide have no underlying data table to recover.**

---

## 2. The five approaches, ranked for *this* corpus

Ranked by measured value per unit of work here — not by general merit, where the order is
different.

| # | approach | buys | costs | measured verdict |
|---|---|---|---|---|
| **1** | **Table extraction first** | the numbers, exactly, with structure | **≈0** — 95.3% already `<table>` markup | Do immediately. The instinct that "chart RAG is often find-the-table" is right and cheaper here than anywhere |
| **2** | **Layout-aware routing** | figure/table/body separation | **≈0** — JATS provides it at 99.9% | Already free. Not a substrate to build; a substrate that exists |
| **3** | **VLM captioning at index time** | works on **all** figure types incl. the 72% charts-to-table cannot touch | one-off pass over 16,792 figures; local GPU | The main build. Pragmatic answer, agreed |
| **4** | **Chart-to-table** | a checkable table from a chart | model + only **27.5%** applicable | Narrow. Fold into #3 as a sub-case, not a separate track |
| **5** | **Vision-native (ColPali/ColQwen)** | no OCR, retrieves what text misses | index size; **breaks §1 provenance** | Attribution control only. Cannot ship as primary |

### Where I disagree with the proposed ranking, and why

**"Layout-aware routing is the substrate everything else sits on."** True for a PDF corpus,
and it is what the patent family needs because it starts from rendered pages. Here NCBI
already published the segmentation in JATS at 99.9%. Detecting boxes on rendered pages
would re-derive from pixels what the publisher states in markup — and worse. This is §4.1's
lesson: PMC's `<ref-list>` is the publisher's own statement of where the bibliography
starts, and using it turned a matter of taste into a labelled task.

There is an irony worth naming: **PubLayNet's ~360k training annotations were generated by
matching PMC's JATS against rendered PDFs.** JATS was the ground truth the detector learned
to approximate. Using the detector here means consuming a lossy re-derivation of the signal
already on disk.

**"Chart-to-table … degrades badly on dense scientific figures."** Correct, and worse than
that framing suggests. Two independent numbers:

* [CharXiv](https://arxiv.org/abs/2406.18521) (NeurIPS 2024), 2,323 real charts from
  scientific papers: **GPT-4o 47.1%** on reasoning questions, best open model 29.2%,
  **human 80.5%**. A mild distribution shift costs up to **34.5%**. ChartQA's homogeneous
  template charts overstate capability.
* On this corpus, chart-to-table is **not even defined for 72.5% of figures**.

So it is not a track. It is a special case inside #3, applied when the caption says the
figure is plottable.

**Where the proposed ranking is right and I would go further:** "table extraction first" is
the sharpest call in the set. 95.3% of tables here are already structured data, and
**81.1% of the numbers authors cite against figures are already in the caption.** The
cheap text path covers most quantitative need before any vision model runs.

---

## 3. Critical evaluation — the part most likely to produce a wrong answer

### 3.1 The current benchmark cannot measure any of this

Judgments say **which document** is relevant. Adding figures does not change which document
is relevant, so every figure candidate scores ≈0 and the loop records a confident negative
about a subsystem never given a way to matter. This is the **fourth** instance of §4.13's
fault (*a candidate structurally incapable of moving its gate*), now at subsystem scale.

### 3.2 Labels exist, and there are more than the corpus has ever had

JATS marks in-text figure references (`<xref ref-type="fig" rid="F3">`), so the sentence
around one is the **author's own description of what that figure shows**:

* **83,462** in-text references, 97.5% of documents, median 18 each
* **64,371** resolve to exactly one figure after dropping diffuse references
* against the **7,056** citation contexts that carried this corpus through five rounds

### 3.3 ⚠️ The limitation that invalidates the obvious reading

**In-text references are, by construction, the parts of a figure the author DID write down
in prose.** The sentence "prevalence was 56% (Figure 1)" contains the answer. So these
labels measure **retrieval** — can the system surface Figure 1 — and they **cannot** measure
pixel-only question answering, because the answer is in the query's own source sentence.

Anyone building a "figure QA" benchmark from these and reporting high accuracy would be
measuring text copying. Two guards:

1. **Exclude the source passage**, asserted not documented — reuse `assert_source_excluded`.
   Without it a query retrieves its own sentence and scores perfectly on string equality.
2. **Do not claim visual QA from this instrument.** It supports retrieval claims only.

For genuine pixel-only evaluation a second instrument is needed, and the honest version is
small and hand-checked rather than large and generated — see 3.5.

### 3.4 ⚠️ Low caption overlap does NOT mean "needs pixels"

The 49.6% figure is an **upper** bound and will be over-read. Measured example at 0.06
overlap:

> query: *"recent basic researches indicated that radiotherapy has significant synergism
> effect with ICI (hypothetical mechanism summarised in figure 1)"*
> caption: *"Hypothetical mechanism of tumour regression induced by radiotherapy combined
> with PD-1 blockade."*

The caption is **adequate**; the overlap is low because the sentence discusses synergism and
the caption names the mechanism. And the figure is a **schematic** — a VLM reading it
recovers nothing a caption does not already say. Low overlap conflates three cases:

| case | pixels help? |
|---|---|
| value is in the plot, not the caption (`weight, 0.0075`) | **yes** |
| topical mismatch between sentence and caption | no — better text retrieval helps |
| schematic with no extractable data | no |

The defensible estimate of pixel-unique value is the **18.9%** of figure-cited numbers
absent from the caption, not the 49.6%. Stratify on this before running anything.

### 3.5 Stratification, and the number that makes it necessary

Split every figure query into **caption-answerable** and **pixel-only**, assigned by a
checkable rule (do the sentence's numeric literals appear in the caption?) rather than by a
model. The reason is measured: in a
[controlled evaluation](https://arxiv.org/html/2607.16604) using exactly this design, the
text-only baseline scored **0.000** on pixel-only questions and multimodal scored
**0.057–0.114**, while caption-answerable questions scored 0.257–0.371. **Caption-derived
benchmarks substantially overestimate visual capability.** Without the split, caption
indexing gains get reported as chart understanding.

### 3.6 Generated text cannot be both the system and the judge

§4.4 forbids grading our own homework, and
[CHOCOLATE](https://aclanthology.org/2024.findings-acl.41.pdf) documents systematic factual
errors in LVLM chart captions. Therefore:

* VLM output is **retrieval bait only** — it may influence ranking and must never be
  rendered to a user as a finding;
* the user always sees the **real caption and the real image**, at real provenance;
* the description generator must not share a family with any model used to judge relevance;
* a **caption-only control** runs alongside every VLM candidate, so a gain attributable to
  simply having more text in the index is not credited to vision. This is the role
  `openai_768` played for MedCPT and `rerank_minilm_cross` played for the cross-encoder —
  both of which **reversed the conclusion**.

### 3.7 Five approaches is a multiple-comparisons problem

Testing all five against several strata is exactly the setup §4.10 was written about. One
pre-registered gate metric per stratum, **Holm across candidates**, secondaries as
uncorrected regression vetoes. And per §4.16, compute the **achievable ceiling** first: if
one sentence maps to several figures, merge before measuring, or the metric is bounded below
1.0 and every candidate is judged against an unreachable target.

### 3.8 The honest prior

Given 81.1% of cited numbers already in captions, 27.5% of figures plottable, and a best
reported pixel-only accuracy of 0.114 elsewhere — **expect a small effect on a stratum that
did not previously exist, not a headline.** Registered now so a null result reads as
prediction confirmed rather than disappointment.

---

## 3.8b The 18.9% is not a pixel gap, and the real gap cannot be measured from text

The obvious next move after "81.1% of figure-cited numbers are in the caption" is to chase
the other 18.9% with a vision model. Measured on 700 sampled documents, 10,137 such
literals:

| where the value actually is | share |
|---|---|
| elsewhere in the body text | **42.1%** |
| in a `<table>` in the same paper | **21.3%** |
| in another figure's caption | **12.8%** |
| nowhere else in the XML | 23.8% |

**76.2% are recoverable from text that is already indexed** — and the last bucket is mostly
an artifact of the instrument. It required a value to appear **twice** before counting as
text-available, so a number stated once, in the citing sentence, was misfiled as pixel-only.
The examples confirm it:

> *"the LCK metagene had the highest prognostic value … with a univariate hazard ratio of
> **1.81** (95% confidence interval = **1.22** to **2.71**, P = **0.003**)."*

Every one of those four values is in a body sentence, and body sentences are indexed.

### ⚠️ The epistemic limit, which is the actual finding

**Any number an author states is, by construction, in the text.** The instrument here reads
sentences, so it can only ever find values someone wrote down — and those are exactly the
values already retrievable. **A text-derived measurement cannot size the pixel gap, because
the pixel gap is precisely what text does not contain:**

* every point on a survival curve other than the two the abstract quotes;
* the n in each arm when only the total is stated;
* axis ranges, error-bar magnitudes, the shape of a dose-response;
* which of eight panels shows the effect, when the caption lists all eight;
* everything in a western blot, micrograph or flow plot.

That gap is real and probably large. **It is also invisible to every label source this
project has** — citation contexts, MeSH, and figure xrefs are all text. Sizing it requires
an instrument built from images: a small hand-annotated pixel-only set, ~200 questions,
written by someone looking at figures rather than at sentences. That is the only honest way,
and it should be built before, not after, committing to a vision stack.

**Consequence for the roadmap.** Do not justify a vision model by the 18.9%. That number is
a text-retrieval problem, and 21.3% of it is literally "the value is in a table in the same
paper" — the *table extraction first* principle, confirmed on this corpus.

## 3.9 Architecture: how a figure becomes retrievable

### The schema already separates "what is indexed" from "what is shown"

`chunks` carries **both** `text` and `indexed_text`. `tsv` and `embedding` are built from
`indexed_text`; the UI renders `text`. The existing comment states the intent:

> *indexed_text may include the contextual-retrieval prefix; text stays verbatim so the
> passage shown to a user is always the real source text.*

That is exactly §3.6's rule, already enforced at the storage layer. A figure chunk uses it:

| column | figure chunk holds | shown to user? |
|---|---|---|
| `text` | **the real caption, verbatim** | **yes**, at real offsets |
| `indexed_text` | caption + label + the sentences that cite it + VLM description + derived table | **never** |

So the VLM description influences ranking and is structurally incapable of being rendered
as a finding. No new enforcement code — the wrong behaviour requires changing the schema.

### One index, not two — and the reason is attribution, not simplicity

A figure is a row in `chunks`, not a separate `figures` index. Three nullable columns:

```sql
ALTER TABLE chunks ADD COLUMN kind       TEXT NOT NULL DEFAULT 'passage';  -- 'passage'|'figure'|'table'
ALTER TABLE chunks ADD COLUMN figure_id  TEXT;        -- 'PMC10558589:fig1', stable, from JATS
ALTER TABLE chunks ADD COLUMN image_uri  TEXT;        -- blob path to the actual picture
CREATE INDEX IF NOT EXISTS chunks_kind_idx ON chunks (kind) WHERE kind <> 'passage';
```

⚠️ **A separate figure index would add a third retrieval arm, and this project has already
been burned by that.** §4.14 records that `tri_fusion`'s gain is confounded with a
lexical:dense ratio change, because three equal-weight arms move the ratio from 1:1 to 1:2 —
and the controls that would attribute it (`tri_fusion_balanced`, `dual_dense`) still have
not run. Adding a figure arm would re-open exactly that question and make the figure result
uninterpretable. **Same table, same two arms, same RRF, same weights** means a measured
change is attributable to the content and nothing else.

Cost: 16,792 figure rows on 180,850 passages, **+9.3%** index size.

### Retrieval, and the aggregation problem that actually matters

`aggregate_chunks_to_docs(..., strategy="max")` collapses passages to **documents**. If a
figure chunk wins, the current pipeline returns the *document* — the figure disappears.
Pulling back charts needs two paths, and the first must not disturb the measured baseline:

**Path A — figures as evidence on a normal search (default).** Document ranking is computed
exactly as today. Then, for each returned document, the figure chunks that survived fusion
are attached to it. Purely additive: `aggregate_chunks_to_docs` is unchanged, so every
existing metric is unchanged, and the regression veto still means what it meant.

```
result = { doc_id, title, passages: [...],           # unchanged
           figures: [ {figure_id, caption, image_uri, matched_by, rank} ] }   # new
```

**Path B — `kind=figure`, figure-first retrieval.** Skip document aggregation entirely and
rank figures directly. This is the "show me the survival curves for osimertinib" query, and
it is the mode the Stage-A labels actually evaluate, because those labels are
(sentence → figure) pairs.

```sql
-- both CTEs already filter on nothing; add the predicate, keep RRF identical
WHERE c.tsv @@ plainto_tsquery('english', $1) AND ($7::text IS NULL OR c.kind = $7)
```

⚠️ Reuse `_cap_per_document` here. A single review with 12 panels would otherwise fill the
whole result list — the §4.4 popularity-bias problem in a new costume.

### Serving

`live_query._shape` must emit `figure_id`, `caption`, `image_uri`, `matched_by` and the
caption's real offsets. §4.15 is the precedent: `best_clause` existed on one side of that
interface for months and was never sent, so every production query silently lost its
highlight.

**Write the contract test first**, with a **non-zero** `start_char` in the fixture, so a
missing `base_offset` fails loudly instead of printing a character range that does not exist
in the article. `tests/test_search_contract.py` already does this for passages and is the
template.

⚠️ **The gibberish problem returns, harder.** §4.15 found the dense CTE has no distance
threshold, so `zzqqxx flurbotanix` returned five confident ferroptosis papers. With images
this is worse: a returned *picture* reads as much stronger evidence than a returned
paragraph. The `matched_by` labelling (*terms* / *terms + meaning* / *meaning only*) must
carry through to figures, and a figure retrieved on `meaning only` with no lexical hit
should be visibly marked as such.

### Ingestion

`sources/jats.py` already parses the JATS and is where `<ref-list>` alignment happens, so
figures come out of the same pass:

1. `<fig>` → `figure_id`, `<label>`, `<caption>` text, `<graphic xlink:href>`
2. resolve the href against the PMC package, upload the image via `scripts/blob_bridge.mjs`
   (§2: the Blob store is private and rejects every REST upload, so the Node SDK bridge is
   the only path)
3. collect `<xref ref-type="fig" rid=…>` sentences into `indexed_text`
4. `<table-wrap>` with a real `<table>` → serialise the grid to text; **95.3% of tables need
   no vision model at all**
5. Stage 1 only: append the VLM description to `indexed_text`

`upsert_chunks(replace=True)` already deletes a document's chunks before inserting (§4.1),
so re-ingestion will not leave stale figure rows behind.

## 3.10 BiomedCLIP vs ColPali/ColQwen — and what neither of them does

### They are retrievers, not extractors

Both compute similarity. **Neither reads a value off an axis.** If the question is "what was
median OS in the combination arm", a retriever surfaces the right figure and cannot tell you
18.9 months. Extraction needs a *generative* model — VLM description or chart-to-table — and
[CharXiv](https://arxiv.org/abs/2406.18521) puts the ceiling there at **GPT-4o 47.1%** on
real scientific charts against **80.5% human**. Keep the two jobs separate when choosing.

### The comparison, for 16,792 figures

| | **BiomedCLIP** | **ColPali / ColQwen** |
|---|---|---|
| architecture | CLIP-style dual encoder, **one vector per image** | late interaction, **~1,024 patch vectors × 128-d per image** |
| training | **PMC-15M** — 15M biomedical figure-caption pairs, i.e. *this corpus's own distribution* | general documents; no biomedical specialisation |
| storage here | 16,792 × 512 × 4B ≈ **34 MB** | 16,792 × 1024 × 128 × 4B ≈ **8.8 GB** (~260×) |
| fits current infra? | **yes** — a pgvector column, cosine, HNSW, exactly like `chunks.embedding` | **no** — MaxSim is not a pgvector ANN op; needs a separate store or a rerank-only pattern |
| reported retrieval | 77% top-5 / 56% top-1 text→image; >90% top-5 against 700k candidates | SOTA on ViDoRe (page retrieval) |
| known weakness | class-wise P@1 **0.240** vs sample-wise 0.594 — poor at category-level discrimination | storage/latency; recoverable via int8 (4×), Matryoshka 1024→256 (<2% recall loss), or token merging (Light-ColPali holds 98.4% NDCG@5 at 4× merge) |
| provenance | image-level — compatible with §1 | patch grid over a **page** — **no character offsets, conflicts with §1** |

### Why ColPali is the wrong first pick *here* specifically

Three reasons, in descending strength:

1. **Its premise does not apply.** ColPali exists to skip a lossy OCR/parsing pipeline on
   PDFs. This corpus has no PDFs — it has JATS, the publisher's own markup, and figure
   images as separate files. The parsing ColPali avoids is *already not lossy*. Applied to
   individual figure images it degrades to "a very expensive image retriever".
2. **Recall is not the binding constraint.** The guidance for choosing multi-vector is that
   it wins *when recall binds and queries are visually hard*. Measured here (§4.18):
   **recall@1000 = 0.9624**, recall@20 = 0.7638, and `recall@1` = 0.3900. The gap is
   **ranking**, not finding. Multi-vector buys recall at 260× storage in a system whose
   recall is already 0.96.
3. **It breaks §1.** A patch grid has no `(doc_id, section, start_char, end_char)`. It can
   be a fusion arm or a reranker; it cannot be the path that produces a citable result.

### Why BiomedCLIP is the right *experiment*, with one caveat

It is domain-matched the way MedCPT was for text — trained on PMC figure-caption pairs,
which is literally this corpus's distribution — and it costs 34 MB and one pgvector column.
That is cheap enough to test properly.

⚠️ **But run the control, because this project has been wrong here twice.** Round 2:
MedCPT looked better than the capacity control on the underpowered stratum, and the powered
stratum **reversed the sign**. Round 4: the general cross-encoder control **inverted**,
which is what proved domain training was doing the work. So pair BiomedCLIP with a
**general CLIP** control on identical figures. If general CLIP matches it, the gain is
"having an image encoder at all", not biomedical training.

⚠️ **And the caveat that may kill it:** CLIP-family models match *gist*, not text-in-image.
They will match "a Kaplan-Meier curve about lung cancer"; they will not match "median OS
18.9 months", because that string is rendered pixels the encoder was never trained to read.
Since **caption text is already indexed and already carries the gist**, BiomedCLIP's
marginal value over the existing text arm is genuinely uncertain — it may be near zero. That
is the hypothesis to register, not assume.

## 3.11 Integrating BiomedCLIP **additively** — captions keep everything they have

Constraint: nothing the caption currently does may be replaced or degraded. That rules out
the two integration patterns that would have caused trouble, and leaves a clean one.

### Two facts from the model card that shape this before any code

⚠️ **`context_length = 256`, image input `224 × 224`, ViT-B/16.** A six-panel figure
downsampled to 224×224 gives each panel roughly **75×75 pixels**. No axis label, no legend,
no p-value survives that. So BiomedCLIP **structurally cannot read values off a plot** — not
because CLIP is weak at OCR, but because the resolution destroys the text before the encoder
sees it. This is the mechanism behind "matches gist, not text-in-image", and it means
BiomedCLIP is a *topical figure finder*, full stop.

**Licence — resolved, not a blocker.** The model card says "any deployed use case … is
currently out of scope", restricting it to research and reproducibility. **OncoLens is not
being commercialised** (confirmed 2026-08-11), so this is the same position as the
NC-licensed content already in the index under §3.1's `research` policy. It stays recorded
because it is a constraint that would bind if that ever changed, not because it blocks
anything now.

### Storage: one nullable column, nothing else moves

```sql
ALTER TABLE chunks ADD COLUMN image_embedding VECTOR(:img_dim);   -- NULL for all 180,850 passages
CREATE INDEX chunks_img_idx ON chunks USING hnsw (image_embedding vector_cosine_ops)
    WHERE kind = 'figure';        -- partial: indexes only the 16,792 rows that have one
```

`:img_dim` is resolved from the model config at ingest, not hard-coded — §4.6's rule that an
embedding-space mismatch raises no error and returns a confident meaningless ranking. Record
it in `index_config` alongside the text backend so `assert_embedding_matches` covers it too.

Text embeddings, `tsv`, `indexed_text` and `text` are **untouched**. The caption keeps every
property it has today.

### Scoring: two additive patterns, neither able to demote a caption hit

**Pattern R — recall-only union (recommended first).** Run the existing pipeline unchanged.
Separately, ANN the query's BiomedCLIP text vector against figure image vectors, and
**append** any figure the text arm did not already retrieve, below everything it did.

```
final = text_ranked_results ++ [f for f in image_ranked if f not in text_ranked][:n]
```

By construction this cannot reorder, displace or demote anything the caption found. It can
only add. Its measured effect on existing metrics is bounded to be ≥ 0 on recall and exactly
0 on precision-at-1. **That is the strongest possible form of the constraint you asked for.**

**Pattern B — monotone boost (second experiment).** Allow promotion but never demotion:

```
score' = rrf_score + α · max(0, cos(q_img, f_img) − τ)
```

The `max(0, ·)` is load-bearing: a figure with a poor image score keeps its caption-derived
rank exactly. Only `α` and `τ` are new, and both default to "off".

⚠️ **Do not add it as a third RRF arm.** That is the pattern D2 rejects: three equal-weight
arms move lexical:dense from 1:1 to 1:2, and §4.14 records that `tri_fusion`'s gain is
*still* confounded by exactly that, with its attribution controls unrun. A third arm would
also violate the constraint here, because re-weighting demotes caption hits.

### Kill switch and blast radius

One config key, `image_arm: off | recall_only | boost`. At `off` the column is dead weight
(34 MB) and no query path reads it. Nothing about caption behaviour is conditional on it.

### Evaluation: the baseline is always the full caption system

The question is **marginal value over captions**, never "BiomedCLIP vs captions". So:

* baseline = current system **with** caption text indexed (which it already is, §1);
* candidate = baseline + Pattern R;
* **control = general CLIP** on identical figures, because round 2 and round 4 both turned
  on a control and both reversed the reading;
* pre-registered prediction: **figure `success@5` up ≥ 0.02; all text strata NULL.** A
  non-null text stratum means Pattern R is not behaving additively and is a bug, not a win.

### The case against, with the licence objection removed

BiomedCLIP is **not the wrong model** — for an image-retrieval arm over this corpus it is
the right one, beating general CLIP on domain match and ColPali on cost, infra fit and
provenance. The case against is not about the model. It is that **its value is squeezed from
both ends**, and three of the four arguments are structural rather than empirical.

**1. Its headline capability is the one thing JATS already gives exactly.** BiomedCLIP's
reported strength — 77% top-5, >90% top-5 against 700k candidates — is *text→image
retrieval*: given a caption, find its image. That is the task it was trained on, on PMC
figure-caption pairs. **This corpus does not have to infer that mapping.** `<fig>` contains
both `<caption>` and `<graphic xlink:href>`; the linkage is stated by the publisher, at
99.9%, exactly and for free. The benchmark BiomedCLIP wins is the benchmark we do not need
to run. This is the §4.1 / PubLayNet pattern for the third time: a model that learned to
approximate a signal we already hold as markup.

**2. Redundant with captions from above.** Captions are indexed at ~100% token coverage and
already sit in both the BM25 and dense arms. BiomedCLIP at 224×224 contributes *gist*.
Captions **are** gist — median 934 characters of authored description. Two encoders of the
same information is the `dual_dense` situation (§4.14), registered as predicted to fail for
precisely this reason.

**3. Cut off from the pixel gap from below.** The one thing captions genuinely do not cover
is unstated values — points on a curve, per-arm n, axis ranges. At 224×224 a six-panel
figure gives each panel ~75×75 pixels, so BiomedCLIP cannot read any of it. It is excluded
by construction from the only territory captions leave open.

**4. Suggestive, not decisive: weak categorical retrieval.** An out-of-distribution study
reports BiomedCLIP class-wise P@1 of **0.240** against sample-wise 0.594 — much better at
"find this specific image" than "find images of category X". Queries here are categorical
("survival curves for osimertinib"), which is the weaker mode. ⚠️ That study is on
**radiology** images, not figures, so treat it as a reason to measure rather than as a
finding that transfers.

**What would change the verdict:** panel segmentation first (~6× effective resolution per
panel, attacking objection 3), or a corpus where caption↔image linkage must be inferred
(attacking objection 1). Neither is true today.

**The one condition under which I would expect it to win** — and this reorders the stages —
is *after* panel segmentation. Cropping a six-panel figure into six 224×224 inputs is ~6×
the effective resolution per panel, and it is the only change that alters what BiomedCLIP
can see. Registered as a hypothesis, not acted on: reordering on reasoning alone is what
produced the caption error in §1.

## 3.12 Going beyond the caption — critical evaluation

The pixel gap is real (unstated data points, per-arm n, axis ranges, panel-level detail,
everything in a blot) and §3.8b establishes that **no text-derived instrument can size it**.
So the ordering below is by *risk-adjusted* value, and the first item is an instrument
rather than a method, because without it every later number is unfalsifiable.

### What is actually in these figures

| figure kind | count | share | caption contains a number |
|---|---|---|---|
| flow cytometry | 3,159 | 19.1% | 97% |
| western blot / gel | 1,876 | 11.4% | 98% |
| **Kaplan–Meier / survival** | **1,666** | **10.1%** | 85% |
| volcano / heatmap (omics) | 1,504 | 9.1% | 95% |
| forest plot / meta-analysis | 525 | 3.2% | 79% |
| dose-response / IC50 | 261 | 1.6% | 97% |
| ROC / AUC | 193 | 1.2% | 92% |

### 0. Masked-caption evaluation — an instrument, and it is free

Hide the caption, extract from the image, score the extraction against the caption. Ground
truth for **16,517 figures**, of which 79–98% contain a number, at zero annotation cost.
This is the same found-data move as §4.4's citation contexts, one modality over.

⚠️ **It is noisy in a specific, correctable direction.** A caption may state `n = 42` when
42 appears nowhere in the plot, so a correct extractor is scored wrong. That makes it a
**lower bound**. Calibrate by hand-checking ~100 items to estimate the unrecoverable
fraction, then report extraction accuracy *net of it*. Do not skip the calibration — an
uncalibrated lower bound gets quoted as an accuracy figure, which is §6.5's error.

### 1. OCR — the most underrated option, and it should go first

Scientific figures render **text as pixels**: axis labels, tick values, legends, p-values,
`n =`, panel letters, and the at-risk table under a KM curve. Plain OCR (Surya, PaddleOCR,
Tesseract) transcribes those literally.

* **It cannot hallucinate.** It transcribes or it fails; it does not invent a plausible
  number. That property is worth more here than accuracy, for the reason in §3.13.
* Cheap and deterministic — no GPU strictly required, no prompt, no model drift.
* Directly targets the one thing captions structurally lack: content rendered as glyphs
  inside the image.
* **Forest plots (525) are a near-perfect target** — they print the effect size and CI as
  text next to every row.

⚠️ **Its real limit is structure, not accuracy.** OCR yields `0.003` with no knowledge that
it is a p-value belonging to panel B. For *retrieval* that is often enough — you want the
figure findable by "p = 0.003". For *extraction* it is not, and pairing OCR with panel
segmentation is what supplies the missing structure.

### 2. Chart-to-table, made verifiable

Only 27.5% applicable and CharXiv caps quality near 47%, so on its own it is weak. But it
has one property prose description does not: **the output is checkable, and this corpus
makes the check free.** 81.1% of figure-cited numbers appear in the caption, so:

```
derender(figure) -> table
overlap(table values, numbers stated in caption/body) -> confidence
index only the derendered tables that clear a confidence threshold
```

That converts an unreliable model into a **high-precision, low-recall** extractor, which is
the right trade when the failure mode is poisoning an index. It also yields a per-figure
quality score that can be reported rather than assumed.

### 3. ~~Domain-specific extraction: Kaplan–Meier reconstruction~~ — **demoted, see below**

**1,666 figures, 10.1%**, and superficially the highest value density in the corpus: a KM
curve yields median OS, survival at *t*, and with the at-risk table a full IPD
reconstruction — exactly the quantities oncology queries ask for.

⚠️ **An earlier version of this section ranked it third and called the automation
"established". That was wrong on two counts, and the second one demotes it outright.**

**Error 1 — I conflated the algorithm with its automation.** Guyot's iterative method (2012)
*is* established: validated, widely used in HTA and NICE submissions. But it takes
**digitised coordinates as input**, and digitisation is the hard part — historically manual
(WebPlotDigitizer, DigitizeIt, ScanIt). The paper I cited for end-to-end automation,
[KM-GPT](https://arxiv.org/html/2509.18141), is a **September 2025 preprint with no
independent validation**. I attached "established" to the automation because the underlying
mathematics is established. Citing a recent preprint as settled is the §4.14 error —
choosing by reputation and recency rather than by evidence.

**Error 2 — and this is the disqualifying one: reconstruction is the wrong *shape* for a
retrieval index.** Even granting a perfect implementation, ask what would be written into
`indexed_text`. A derived number such as `median OS 18.9 months`. Then §3.13's rule applies
and the argument closes on itself:

* if the derived value **cross-checks against a stated value**, the stated value was already
  in the caption or body — and it is already indexed, so nothing was gained;
* if it **does not cross-check**, it is unverified inference and must not enter the index.

**Either it is redundant or it is inadmissible.** IPD reconstruction is a *meta-analysis*
capability — it produces a dataset for re-analysis. RAG needs retrievable text and returnable
evidence. Those are different products.

And the failure mode is the worst available: **a slightly-wrong reconstruction produces a
survival curve that looks entirely correct** and yields a wrong median or HR with no visible
symptom. That is §3.13 escalated from a wrong number to a wrong *dataset* wearing the
costume of data.

**Revised verdict:** not part of the RAG build. Worth keeping in mind as a separate product
capability if OncoLens ever moves toward evidence synthesis, where a reconstructed dataset is
the deliverable and can be inspected as such. What *is* worth taking from this section is
narrower and safe: **OCR the at-risk table under a KM curve** — it is printed text, so
transcription rather than inference, and it carries the per-arm n that captions frequently
omit.

### ⚠️ A source-quality note that applies to this whole document

The KM-GPT error is systemic, not local. Several works cited here are recent arXiv preprints
without independent replication, and they are not the same kind of evidence as peer-reviewed
and replicated results. Read them accordingly:

| cited as | status |
|---|---|
| CharXiv (47.1% / 80.5%) | **NeurIPS 2024**, peer-reviewed |
| CHOCOLATE chart-caption errors | **ACL Findings 2024**, peer-reviewed |
| ColPali | published, widely replicated |
| PubLayNet, Guyot iKM | long-established, heavily used |
| Open-PMC-18M, Docling "heron", the controlled multimodal/graph-RAG evaluation | **recent preprints** — directional evidence, not settled findings |
| KM-GPT | **recent preprint, no independent validation** |

The numbers this plan leans on hardest — the 0.000 text-only baseline on pixel-only
questions, and 0.057–0.114 for multimodal — come from a **preprint**. They are load-bearing
for the "expect a small effect" prediction, so that prediction should be held with
correspondingly less confidence than the numbers measured on this corpus.

### 4. Panel segmentation, supervised by the caption

~50% of medical figures are multi-panel and PMC ships one image per `<fig>`. The free
supervision is that **the caption enumerates the panels itself**:

> *"(A) Rates of all-grade infections and (B) grade ≥3 infections among patients…"*

So caption sub-sentences align to panel crops without annotation. Panels then improve
everything downstream: ~6× effective resolution for any 224×224 encoder, panel-level
provenance (`Fig 3B` rather than a six-panel composite), and the 2,154 measured panel-level
in-text references become usable labels.

### 5. VLM description — most general, most dangerous, and therefore last

The only option covering the **22.7% pure imagery** (blots, micrographs, scans) that nothing
else touches, and the only one that produces prose a dense encoder can use. Also the one
with CharXiv's 47.1% and CHOCOLATE's documented systematic factual errors.

**Use it for what it cannot get wrong in a costly way:** figure *type*, modality, what is
being compared, which entities appear. Do **not** use it to mint numbers. See below.

## 3.13 The failure mode nobody plans for: hallucinated retrieval bait

The plan says generated text is "retrieval bait only, never shown as a finding". That guard
is necessary and **not sufficient**, and it is worth being precise about why.

Suppose a VLM writes *"median OS 18.9 months"* for a figure whose true value is 22.4:

1. a query for **18.9 months** retrieves the **wrong** figure, confidently;
2. a query for **22.4 months** **misses** the right one;
3. the user sees the real caption and real image, so **the error is invisible** — it lives
   entirely in the ranking layer, where nothing is displayed and nothing is checked.

A wrong number in the index is therefore **worse than no number**, because it actively
misroutes rather than merely failing to help. This is §4.15's pattern — a surface asserting
more than the mechanism underneath knows — relocated to a layer with no surface at all.

**The operating rule that follows:**

> **Prefer transcription over inference for anything that enters the index.**
> OCR'd glyphs and JATS markup are transcription. A derendered table validated against
> stated values is transcription with a receipt. A VLM's free-text number is inference, and
> inference belongs in the *answer*, next to the image, where a reader can see it is wrong.

Practical consequence: index VLM **descriptions** (type, modality, entities, what is
compared) and **withhold VLM numerals** unless they cross-check against a stated value. This
costs little — the numbers are 81.1% in captions anyway — and it removes the only failure
mode in this plan that degrades the system rather than leaving it unchanged.

## 3.14 MEASURED: what BiomedCLIP can actually extract, on this corpus

Rather than continue arguing from architecture, I ran it. `scripts/probe_biomedclip.py`
fetches real figure images via `pmc_cloud` `media_urls`, and zero-shot classifies them
against eight plain type prompts. Reference labels come from captions that match exactly one
type pattern — noisy, so this is *agreement with a noisy reference*, not accuracy.

**n = 69 real PMC figures, 8 classes, chance = 12.5%. Shared embedding dim = 512.**

| | measured |
|---|---|
| **top-1 agreement** | **60.9%** |
| **top-2 agreement** | **75.4%** |

| class | agreement | confused with |
|---|---|---|
| flow cytometry plot | 83% (5/6) | bar chart |
| kaplan-meier survival curve | 73% (8/11) | heatmap ×2, bar chart |
| bar chart | 73% (11/15) | microscopy ×2 |
| schematic diagram | 64% (7/11) | flow cytometry ×3 |
| western blot | 55% (6/11) | **bar chart ×4** |
| heatmap | 50% (4/8) | **flow cytometry ×3** |
| **microscopy image** | **17% (1/6)** | forest plot ×2 |

### What this establishes, and what it does not

**It extracts *type*, at roughly 5× chance.** That is real signal and it is the one thing
CLIP-family models do well at 224×224 — gist. It does **not** extract values; nothing here
contradicts §3.10.

⚠️ **60.9% is too weak to be authoritative.** It cannot be a hard filter ("show only survival
curves" would silently drop 27% of them). It is usable as a *soft* signal, a suggested
facet, or a fallback where captions are silent.

⚠️ **The confusion pattern is the most informative part, and it is evidence for panel
segmentation — measured rather than assumed.** Western blot → bar chart (4/11) and heatmap →
flow cytometry (3/8) are exactly what a **multi-panel composite** produces: the figure
genuinely contains both, and whichever panel dominates the pixels wins. **Type classification
on a whole multi-panel figure is ill-posed — there is no single correct answer.** ~50% of
these figures are multi-panel, so a large share of the 39% "error" may be the reference
being wrong rather than the model.

⚠️ **It is weakest exactly where it would be most useful.** Microscopy — the 22.7% pure-imagery
class that captions describe thinly and no other method touches — scores 17%. Sample is 6, so
treat as a flag to re-measure, not a finding.

⚠️ **Sample caveats, stated plainly:** n=69 (127 sampled, 69 images resolved — 54% of
`media_urls` fetched), one class had n=1 and is uninformative, and the reference is
caption-derived. This is a feasibility probe, not a benchmark.

## 3.15 Implementation plan: type extraction and cross-paper comparison

### The cascade — captions first, model second

Captions are **precise when they fire**: a caption saying "Kaplan-Meier" is near-certain
evidence. The caption regex classified 59.5% of figures and left **40.5% unclassified**.
BiomedCLIP's job is that residue, not the whole task. Same shape as the terminology cascade
in §4.14 — authoritative source first, model where the source is silent.

```
figure_type(fig):
    if caption matches exactly one type pattern  -> that type,  confidence "stated"
    elif BiomedCLIP top-1 margin > threshold     -> that type,  confidence "inferred"
    else                                          -> unknown,  and SAY unknown
```

Three-valued output is load-bearing. §4.15's "not reported" lesson: a cell that renders as
blank when the mechanism merely failed to decide is a claim the system cannot support.

### Storage — unchanged from §3.11, plus two columns

```sql
ALTER TABLE chunks ADD COLUMN figure_type       TEXT;   -- 'kaplan-meier' | ... | NULL
ALTER TABLE chunks ADD COLUMN figure_type_src   TEXT;   -- 'caption' | 'biomedclip' | NULL
```

The 512-dim `image_embedding` from §3.11 is **34 MB** for all 16,792 figures and is what the
zero-shot pass consumes, so typing costs one extra matrix multiply against 8 text vectors —
effectively free once the images are embedded.

### Cross-paper comparison — and why image similarity is the wrong key

The natural fit is the **existing `/api/compare`**, which already builds cross-paper tables
by aspect. A figure row is the same shape: *for these 4 papers, show the survival curves*.

⚠️ **Do not build this on image-to-image similarity.** Near-duplicate detection over
scientific figures is proven at scale (ImageTwin, 160M+ images, western blots and
microscopy), but **near-duplicate is not the same relation as comparable**:

* two Kaplan–Meier curves for *different drugs* look nearly identical and are **not**
  comparable;
* two IHC panels of the *same marker* with different staining protocols look different and
  **are** comparable.

Visual similarity is a poor proxy for scientific comparability, and the failure is silent.
The correct key is **type + topic** — type from the cascade above, topic from the caption
text already indexed. Both are cheap, both are inspectable, and neither requires an image
comparison. Image similarity has one legitimate use here and it is *research integrity*
(same figure reused across papers), which is a different product.

### Registered predictions

| stage | prediction |
|---|---|
| caption-first typing | covers 59.5% at "stated" confidence; no ranking effect |
| + BiomedCLIP on the residue | coverage 59.5% → ~85%, at ~61% precision on the inferred portion; **ranking NULL** |
| type facet in search/compare | product capability; **ranking NULL on all four strata** |
| after panel segmentation | typing agreement **improves** — the multi-panel confusions above are the mechanism, and this is the sharpest test of whether panels are worth building |

**Everything here is registered as ranking-NULL.** The value is faceting and comparison, not
retrieval quality. Per §3.12 the only figure work with a plausible ranking effect is OCR.

## 4. Provenance: the constraint that shapes the schema

§1 says a change that improves ranking and loses provenance is a regression. A figure has no
character offsets, so extend rather than bend:

```
figure_id      PMC<id>:fig:F3         stable, from JATS
panel_bbox     x,y,w,h | null         null until panel segmentation exists
caption_span   (doc_id, start, end)   REAL offsets into caption text
image_uri      blob/PMC<id>/<file>    the picture the reader actually checks
derived_text   VLM output | null      machine-generated; never quoted as a finding
derived_table  chart-to-table | null  only for the 27.5%; marked approximate
```

**The contract rule:** anything rendered as a finding must resolve to either real character
offsets or a real image the reader can look at. `derived_text` is neither. This needs a
contract test on day one — §4.11 and §4.15 are the same failure twice, a served shape
drifting from what the client believes.

---

## 5. Order of work

| stage | what | gate to proceed |
|---|---|---|
| **A** | Figure labels from `<xref>`; exclude source passage; caption-answerable / pixel-only split; report n, judgments/query, measured MDE, achievable ceiling | labels exist and the ceiling is 1.0 |
| **0′** | **Structure text that is already there** — split captions out of body chunks into `kind='figure'` rows, attach `figure_id` and `image_uri`, fix the `.Figure 1.` boundary defect. **No new text, no models.** | figures are *returnable* and `kind=figure` answers; ranking predicted **NULL** |
| **1** | VLM description into `indexed_text` for all 16,792 figures; chart-to-table for the 27.5% plottable | beats 0′ **with the caption-only control also run** |
| **2** | Panel segmentation (~50% of medical figures are multi-panel; PMC ships one image per `<fig>`) | pixel-only queries exist and Stage 1 fails on them specifically |
| **3** | ColPali/ColQwen as an attribution control | — never ships as primary; §1 |

⚠️ **Stage 0′ replaces the deleted Stage 0 and its prediction is inverted.** The old stage
predicted a ranking gain from adding caption text; the text was already there. 0′ adds no
text at all — it re-files existing text as addressable figure rows and attaches the image.
Its gate is a **product capability** (can a chart be returned and shown?), and its ranking
prediction is **NULL on every stratum**. A measurable ranking change from 0′ would be a
warning that chunk boundaries moved, not a win.

Stages 1 and 2 are deliberately inverted against intuition. Panel segmentation is the more
interesting engineering and the less certain payoff; whole-figure descriptions are testable
first, and if those do not help, panels will not either.

---

## 5b. Design decisions, and what each one rules out

Recorded as decisions rather than prose so a later reader can see what was chosen *against*.

| # | decision | chosen | rejected | because |
|---|---|---|---|---|
| D1 | segmentation source | **JATS markup** | page-layout detection | 99.9% figure image refs already; PubLayNet's own labels were *derived from JATS* |
| D2 | index topology | **one `chunks` table, `kind='figure'`** | separate figure index | a third arm re-opens §4.14's unresolved lexical:dense confound |
| D3 | what is embedded | **`indexed_text`** (caption + citing sentences + later VLM text) | embedding the image first | caption text already carries the gist and is already indexed at ~100% coverage |
| D4 | what is displayed | **`text` = verbatim caption + the image** | generated description | §4.4 + CHOCOLATE: generated text is bait, never a finding. Enforced by the schema, not by a code path |
| D5 | table handling | **serialise existing `<table>` markup** | table-structure recognition from pixels | 95.3% already machine-readable; 21.3% of "missing" figure numbers live in these tables |
| D6 | figure result shape | **Path A attach + Path B `kind=figure`** | replacing doc aggregation | Path A leaves every current metric untouched, so a regression veto still means something |
| D7 | image encoder, if any | **BiomedCLIP first, with a general-CLIP control** | ColPali/ColQwen | premise (lossy PDF parsing) does not apply; recall@1000 = 0.9624 so recall does not bind; 260× storage; breaks §1 |
| D8 | value extraction | **VLM/chart-to-table, separate from retrieval** | expecting a retriever to extract | retrievers compute similarity; CharXiv caps extraction at 47.1% |
| D9 | pixel-gap sizing | **~200 hand-written image-only questions** | deriving it from figure xrefs | every author-stated number is already in text; text labels are structurally blind to the pixel gap |

## 5c. Sequence, with the gate that stops each stage

| stage | work | pre-registered prediction | stop if |
|---|---|---|---|
| **A** | figure labels from `<xref>`; exclude source passage; report n, MDE, ceiling | — instrument, not a candidate | ceiling < 1.0 after merging |
| **A2** | **masked-caption extraction benchmark** (§3.12.0) + ~100 hand-checked items to calibrate its noise floor | — instrument | calibration shows the unrecoverable fraction dominates |
| **A3** | **OCR every figure**, index the transcribed glyphs (§3.12.1) | figure `success@5` up ≥0.02; text strata NULL | OCR yield per figure is negligible |
| **0′** | split captions into `kind='figure'` rows; attach `figure_id`, `image_uri`; fix `.Figure 1.` boundary | **NULL on every ranking stratum**; success = a chart can be returned and shown | any significant ranking change (means chunk boundaries moved) |
| **1** | serialise `<table>` markup into retrievable rows | synthesis `recall@20` up ≥0.005 | regression on `claim` |
| **2** | VLM description into `indexed_text`; chart-to-table for the 27.5% plottable | figure `success@5` up ≥0.02 **over a caption-only control** | control matches it — then it was just more text |
| **3** | BiomedCLIP image vectors, **plus general-CLIP control** | figure `success@5` up ≥0.02 over Stage 2 | general CLIP matches — then domain training bought nothing |
| **4** | panel segmentation | pixel-only set improves; caption-answerable **NULL** | Stage 2 already saturates pixel-only |
| **5** | ColPali/ColQwen | — attribution control only | never ships as primary (§1) |

**Cheapest-first is deliberate and it is also the ordering most likely to end early.** If
0′ and 1 deliver the product capability and Stage 2's caption-only control matches the VLM,
the correct outcome is to stop — and that would be a real result, not a failure.

## 6. Registered as predicted to fail

* **Knowledge-graph augmentation.** Measured elsewhere at **+0.028 / −0.017 / 0.000** on
  text / multi-hop / figure questions, attributed to unrestricted entity matching importing
  unrelated facts while provenance-restricted matching merely restates retrieved passages.
* **Page-layout detection on this corpus.** JATS supplies it at 99.9%.
* **Chart-to-table as a primary track.** Undefined for 72.5% of these figures.
* **Large absolute pixel-only gains.** Best reported is 0.114; CharXiv puts GPT-4o at 47.1%
  against 80.5% human on real scientific charts.
* **A stronger generator rescuing weak retrieval.** Measured there at 0.086 → 0.143 while
  retrieval Recall@1 was 0.229 — consistent with this project's own finding
  (§4.18) that ranking, not generation, is the binding constraint.

---

## Sources

[PubLayNet](https://arxiv.org/abs/1908.07836) ·
[DocLayout-YOLO](https://arxiv.org/abs/2410.12628) ·
[Docling advanced layout](https://arxiv.org/abs/2509.11720) ·
[ColPali](https://arxiv.org/abs/2407.01449) ·
[CharXiv](https://arxiv.org/abs/2406.18521) ·
[Open-PMC-18M compound figures](https://arxiv.org/abs/2506.02738) ·
[controlled multimodal/graph-RAG evaluation](https://arxiv.org/html/2607.16604) ·
[CHOCOLATE chart-caption errors](https://aclanthology.org/2024.findings-acl.41.pdf) ·
[olmOCR](https://arxiv.org/pdf/2502.18443)
