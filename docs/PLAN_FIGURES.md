# Plan: figures, charts and tables

Everything below is measured on this repo's own corpus or cited to a source. Where a number
appears, the script that produced it is named.

---

## 1. The framing, and why this corpus changes it

Document layout analysis was reframed as object detection: render the page, detect
`title / paragraph / table / figure / list / caption` instead of COCO classes, keep the
detector. [PubLayNet](https://arxiv.org/abs/1908.07836) (IBM, 2019) made it practical by
auto-generating ~360k annotated pages from PMC's structured XML matched against rendered
PDFs — no human labelling — and its Faster/Mask R-CNN baselines became the default weights,
later packaged in LayoutParser and the Detectron2 zoo. DocBank (2020) did the same from
LaTeX.

That stack is the standard one, and it is also **behind the current field**. Since 2022 the
work moved to models that encode text, position and image *jointly* rather than bolting a
text extractor onto a box detector:

| approach | note |
|---|---|
| LayoutLMv3 / DiT | ~95% mAP on PubLayNet with a Cascade R-CNN head |
| [DocLayout-YOLO](https://arxiv.org/abs/2410.12628) (2024) | synthetic diverse data + global-to-local perception; the practical speed/accuracy pick |
| [Docling advanced layout](https://arxiv.org/abs/2509.11720) (2025) | "heron" model, reported +23.5% mAP over its predecessor |
| [ColPali](https://arxiv.org/abs/2407.01449) | skips detection entirely — 1,024 patch embeddings per page, ColBERT-style MaxSim late interaction |

⚠️ **Generalisation is the open problem, not accuracy.** The same architectures that score
95–97% mAP on PubLayNet drop to **80–82% on DocLayNet**, a 10–20 point fall, because
PubLayNet is one document class (PMC biomedical articles). This is §4.1's lesson in the
literature's own voice: *a benchmark drawn from one document class certifies a rule that
fails on another.*

### The detection problem this project has is not the one in the patent

A three-model architecture with a rule-based merge exists because the vision model and the
text models are separate systems needing reconciliation after the fact. **OncoLens has no
such problem, because it never starts from a PDF.** It ingests JATS XML, where NCBI has
already published the segmentation.

Measured on the 3,251 cached JATS documents (`scripts/` audit):

| | measured |
|---|---|
| documents with ≥1 figure | **97.7%** |
| total figures | **16,792** (median 5/doc) |
| figures carrying a caption | **99.0%** |
| figures carrying an image reference (`<graphic xlink:href>`) | **99.9%** |
| tables with machine-readable `<table>` markup, **not** a picture | **95.3%** |
| caption text **currently absent from the index** | **17.4M characters** |

For scale, the body index is ~163M characters over 180,850 passages, so **captions alone
are ~10.7% more text than the index currently holds**, and 3,861 tables are already
structured data needing no vision model at all.

Running a page-layout detector here would re-derive from pixels what the publisher already
states in markup, and do it worse. That is exactly §4.1: PMC's `<ref-list>` is the
publisher's own statement of where the bibliography starts, and using it turned a matter of
taste into a labelled task.

**Where detection genuinely is required: panel segmentation.** Roughly **50% of medical
literature figures are multi-panel**, PMC ships one image per `<fig>`, and a compound figure
is several experiments in one file. That is a real object-detection task on the *figure*,
not the page. [Open-PMC-18M](https://arxiv.org/abs/2506.02738) (2025) is the current
reference: transformer-based subfigure detection trained on 500k programmatically composed
compound figures, SOTA on ImageCLEF 2016. It also reports that at PMC scale **16.3% of
compound figures have no caption** and 1.8% have captions under ten words — so panel-level
text cannot simply be read off.

---

## 2. The blocking problem: the current benchmark cannot measure any of this

The labels are citation contexts. A judgment says **which document** is correct. Adding
figures to the index does not change which document is correct, so every figure candidate
would score a delta of approximately zero — and the loop would record a confident negative
about a subsystem that was never given a way to matter.

That is §4.13's fault (*a candidate structurally incapable of moving its gate*) at the scale
of an entire subsystem, and it is the **fourth** instance in this project. Before any model
is built, the evaluation has to exist.

### Labels are available, found rather than written, and there are more of them

JATS marks in-text figure references as `<xref ref-type="fig" rid="F3">`. The sentence
around one is the **author's own description of what that figure shows**, written by the
person who made it. Measured on the same 3,251 documents:

| | measured |
|---|---|
| documents with ≥1 in-text figure reference | **97.5%** |
| total in-text figure references | **83,462** |
| median per document | **18** |
| sentences naming a specific panel (`Fig 3B`) | 2,154 |

**83,462 against the 7,056 citation-context labels that carried this corpus through five
rounds — 11.8×.** And the shape is right:

> "grade ≥3 infections were significantly higher among BCMA-targeting bispecifics (25%;
> 95% CI, 0.17-0.32) than with non-BCMA bispecifics (20%; 95% CI, 0.16-0.23; P < .01;
> **Figure 2**)."

Query = the sentence. Answer = Figure 2. The numbers in it exist **only in the figure**.

⚠️ **Three hazards, each needing a guard before this is trustworthy** — same discipline as
§4.4:

1. **The sentence is body text and body text is indexed.** A query built from it would
   retrieve its own source passage and score perfectly on string equality. The passage
   containing the reference must be excluded, asserted not documented (`assert_source_excluded`
   already exists and should be reused).
2. **Caption leakage.** Many such sentences paraphrase the caption. If the caption is
   indexed, the task collapses to caption matching. This is not a defect to remove — it is
   the thing to *stratify on* (below).
3. **Diffuse references.** "(figures 2 and 3)" asserts nothing specific about either; the
   samples above contain several. Grade down with co-reference exactly as §4.4 does for
   co-citation, and drop >2.

### The stratification that decides everything

The one design that separates real visual understanding from caption recovery, taken from a
[controlled evaluation](https://arxiv.org/html/2607.16604) that ran exactly this comparison:

* **caption-answerable** — the answer appears in the caption text;
* **pixel-only** — the answer requires reading the image (axis values, panel-specific
  numbers, direction of an effect).

Their measured result, and the reason this matters: on pixel-only questions the **text-only
baseline scored 0.000** across all four generators, while multimodal scored **0.057–0.114**.
On caption-answerable questions multimodal scored 0.257–0.371. **Caption-derived benchmarks
substantially overestimate visual capability**, and without this split we would measure
caption recovery and report it as chart understanding.

Assign the split automatically and conservatively: if the numeric literals in the reference
sentence appear in the caption, it is caption-answerable; if not, it is a pixel-only
candidate. That rule is checkable and does not require a model to arbitrate.

---

## 3. Staged plan, each stage gated on the previous

### Stage 0 — index what is already there. No models.

Add caption text and table markup as retrievable passages with full provenance. Zero new
dependencies, ~10.7% more indexed text, 3,861 machine-readable tables.

**This is the baseline every later stage must beat**, and skipping it is how a VLM gets
credited with a gain that plain caption indexing would have delivered. It is the direct
analogue of §4.5, where BM25 alone beat the shipped hybrid.

* Pre-register: `synthesis recall@20` **up ≥ 0.01**; `identifier` and `claim` **NULL** — a
  caption is topical prose, and claim queries are answered by body sentences.
* Risk to watch: captions are dense with terms and could crowd out body passages the way
  reference strings did in §4.1. The regression veto on `claim` is what catches that.

### Stage A — build the figure benchmark (blocking; nothing after this works without it)

`scripts/build_figure_labels.py`: mine `<xref ref-type="fig">`, exclude the source passage,
grade by co-reference, split caption-answerable / pixel-only, hold out a locked test split
on the same document-level hash as the existing splits.

* Report `n`, judgments per query, and the **measured** MDE from the paired bootstrap
  (§4.14) before running any candidate.
* Report the **achievable ceiling** (§4.16): if the same sentence maps to several figures,
  the metric is bounded and must be merged first.

### Stage 1 — panel segmentation

Only if Stage 0 shows figures are retrieved at all. Detect subfigure panels so a query can
return *panel 3B* rather than a 6-panel composite. Open-PMC-18M-style detector, or the
synthetic-composition trick to generate training data from single-panel PMC figures.

* Pre-register: improves **pixel-only** figure retrieval; **NULL on caption-answerable**,
  because a caption describes the whole figure and panels do not change that.
* This is the only place a detector earns its cost, and the prediction says exactly where
  it should show up.

### Stage 2 — VLM labelling of figures ("LLM labelling of charts")

For each figure or panel, generate structured text offline and index it: chart type,
axis labels and ranges, series names, extracted data table where the chart is a plottable
type, and a factual description. Chart→table derendering (DePlot/MatCha lineage,
ChartGemma) is the higher-value output because a table is *checkable*; free-text captions
are not.

⚠️ **Generated text is not evidence, and this is where the project's own rule bites.**
§4.4 forbids grading our own homework, and
[CHOCOLATE](https://aclanthology.org/2024.findings-acl.41.pdf) documents that LVLM chart
captions contain systematic factual errors. Therefore:

* the generated description is **retrieval bait only** — never shown to the user as a
  finding, and never a source for a judgment;
* what the user sees is the **image and the real caption**, at their real provenance;
* the generator must not be from the same family as any model used to judge relevance.

Offline cost is the deciding factor: at 16,792 figures this is a one-off batch, not a
request-path cost. Budget it against the measured alternative — olmOCR reports
<$190/million pages, ~1/32 of GPT-4o API pricing, so a local VLM is the default.

### Stage 3 — visual retrieval arm (ColPali), as a *control*, not a proposal

ColPali retrieves over page-image patches with no OCR at all. Run it to answer one
question: **how much of Stage 2's gain is the VLM's description versus simply having pixels
in the index?** Same role `openai_768` played for MedCPT and `rerank_minilm_cross` played
for the cross-encoder — both of which changed the conclusion.

⚠️ **ColPali conflicts with §1 and cannot ship as the primary path.** It returns a *page*,
and the one rule is returning the passage with `(doc_id, section, start_char, end_char)`. A
patch grid has no character offsets. It is legitimate as an extra fusion arm or a
diagnostic; it is a regression as a replacement.

---

## 4. Provenance: the constraint that shapes the schema

§1 says a retrieval change that improves ranking and loses provenance is a regression. A
figure has no character offsets, so the model must extend rather than bend:

```
figure_id       PMC<id>:fig:F3          stable, from JATS
panel_bbox      x,y,w,h | null          null until Stage 1
caption_span    (doc_id, start, end)    REAL offsets into the caption text
image_uri       blob/PMC<id>/<graphic>  the actual picture the user checks
derived_text    VLM output | null       marked machine-generated, never quoted as finding
```

The rule to enforce in the response contract: **anything shown as a finding must resolve to
either real character offsets or a real image the reader can look at.** `derived_text` is
neither, so it may influence ranking and must never be rendered as evidence. This wants a
contract test on day one — §4.11 and §4.15 are both the same failure of a served shape
drifting from what the client believes.

---

## 5. What I predict will not work, written down first

* **Knowledge-graph augmentation.** Directly measured in the controlled evaluation above:
  **+0.028 on text questions, −0.017 on multi-hop, 0.000 on figure questions.** The authors
  attribute it to unrestricted entity matching importing facts from unrelated documents,
  while provenance-restricted matching merely restates retrieved passages. If "detection
  over graphs" is ever read as *knowledge* graphs rather than charts, this is the evidence
  against starting there.
* **Page-layout detection on this corpus.** JATS already provides it at 99.9% for figures.
* **Big absolute gains on pixel-only questions.** The best multimodal number in the
  controlled evaluation was **0.114**. Expect a small effect on a stratum that did not
  previously exist, not a headline.
* **A stronger generator rescuing weak retrieval.** Measured there too: GPT-4o moved
  accuracy 0.086 → 0.143 while CLIP Recall@1 was 0.229. Consistent with this project's own
  finding that ranking, not generation, is the binding constraint.

---

## 6. Order of work

1. **Stage A labels** (blocking — without it nothing downstream is measurable)
2. **Stage 0 caption + table indexing** (no models, largest certain gain, the baseline)
3. Stage 2 VLM labelling on figures already shown to be retrieved
4. Stage 1 panel segmentation, gated on pixel-only queries existing and failing
5. Stage 3 ColPali as an attribution control

Stages 1 and 2 are deliberately out of intuitive order. Panel segmentation is the more
interesting engineering and the less certain payoff; the VLM description is testable on
whole figures first, and if whole-figure descriptions do not help, panels will not either.
