"use client";

import { Fragment, useState } from "react";

import ClusterMap from "../components/ClusterMap";
import WebGLBackground from "../components/WebGLBackground";

type Metric = {
  label: string; value: number | string; unit?: string; note?: string;
  provenance: "live" | "artifact" | "recorded"; command?: string; delta?: string | null;
};
type Stage = {
  id: string; kicker: string; title: string; narrative: string;
  metrics?: Metric[]; caveat?: string | null; headline?: string | null;
  systems?: Record<string, string | number>[];
};
/** Emitted by build_journey_data.power_table: never typed into this component. */
type Power = {
  stratum: string; queries: number; judgments: number; weight: number | null;
  gate_metric: string; mde: number; sees_002: boolean;
  /** "measured from paired bootstrap CI" vs the conservative analytic fallback. */
  mde_source?: string; mde_analytic?: number;
};
type Journey = {
  generated_at: string; git_rev: string; live_store_reachable: boolean;
  stages: Stage[]; power?: Power[];
};

const PROVENANCE: Record<string, { label: string; cls: string; title: string }> = {
  live: { label: "live", cls: "border-emerald-400/25 text-emerald-300/90",
    title: "Queried from the store when this page was built" },
  artifact: { label: "benchmark", cls: "border-cyan-400/25 text-cyan-300/90",
    title: "Read from a benchmark script's JSON output" },
  recorded: { label: "measured", cls: "border-slate-400/20 text-slate-400",
    title: "Measured by the named script" },
};

function fmt(v: number | string) {
  if (typeof v === "number") {
    if (Number.isInteger(v) && Math.abs(v) >= 1000) return v.toLocaleString();
    if (!Number.isInteger(v)) return v.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }
  return String(v);
}

/**
 * A headline figure that lights up under the pointer.
 *
 * The accent sits behind the number rather than around it, and the `command` that produced
 * the figure is revealed on hover: the point of animating a number is to invite scrutiny of
 * it, so the invitation had better lead somewhere.
 *
 * CSS, not WebGL. An earlier version gave this element (and every number in the benchmark
 * and power tables) its own canvas and its own GL context, which reached ~29 contexts on
 * this page. Browsers cap them near 8-16, then return null from getContext and evict the
 * oldest live ones — so the page lost these very numbers AND the cluster map. See
 * globals.css.
 */
function LiveMetric({ m }: { m: Metric }) {
  const p = PROVENANCE[m.provenance];
  return (
    <div className="group relative">
      <dt className="text-[11px] uppercase tracking-wider text-slate-500">{m.label}</dt>
      <dd className="relative mt-1 overflow-hidden rounded">
        <span className="accent-glow pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
        <span className="relative font-mono text-xl tabular-nums text-white">
          {fmt(m.value)}
          {m.unit ? <span className="ml-0.5 text-sm text-slate-400">{m.unit}</span> : null}
        </span>
      </dd>
      <span className="accent-rule mt-1 block h-[2px] w-full opacity-60 transition-opacity group-hover:opacity-100" />
      {p ? (
        <span className={`mt-1.5 inline-block rounded border px-1 py-px text-[9px] uppercase tracking-wider ${p.cls}`}
              title={p.title}>
          {p.label}
        </span>
      ) : null}
      {m.command ? (
        <code className="mt-1 block truncate font-mono text-[9px] text-slate-600 opacity-0 transition-opacity group-hover:opacity-100"
              title={m.command}>
          {m.command}
        </code>
      ) : null}
    </div>
  );
}

/**
 * A number that lights from within on hover.
 *
 * Used for the figures a reader is meant to interrogate rather than skim: benchmark
 * scores, detectable effects, sample sizes. Pure CSS, and deliberately so — a table of
 * these is 15 to 20 elements, and one GL context per number is what broke the page.
 */
function LiveNumber({
  children, tone = "neutral", title,
}: { children: React.ReactNode; tone?: "neutral" | "good" | "bad"; title?: string }) {
  const text =
    tone === "good" ? "text-emerald-300/90"
    : tone === "bad" ? "text-rose-300/80"
    : "text-slate-200";
  return (
    <span
      title={title}
      className="group relative inline-block overflow-hidden rounded px-1 py-0.5"
    >
      <span className="accent-glow-teal pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
      <span className={`relative font-mono tabular-nums ${text}`}>{children}</span>
    </span>
  );
}

/** Section heading with a live hairline under it, so each section reads as one unit. */
function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <div>
      <h2 className="text-xs uppercase tracking-[0.16em] text-cyan-300/70">{children}</h2>
      <span className="accent-rule mt-2 block h-[2px] w-28" />
    </div>
  );
}

/**
 * A link whose accent responds to the pointer.
 *
 * On a solid button the accent is white, because a cyan glow over a cyan fill is
 * invisible.
 */
function GlowLink({
  href, children, tone = "solid",
}: { href: string; children: React.ReactNode; tone?: "solid" | "outline" }) {
  const solid = tone === "solid";
  return (
    <a
      href={href}
      className={`group relative overflow-hidden rounded-md px-4 py-2 text-sm transition-colors ${
        solid
          ? "bg-cyan-400/90 font-medium text-[#04121a] hover:bg-cyan-300"
          : "border border-white/12 text-slate-300 hover:border-white/25 hover:text-white"
      }`}
    >
      {/* White over a solid cyan fill; teal over the dark outline variant. A cyan glow on
          a cyan button is invisible. */}
      <span
        className={`pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100 group-focus:opacity-100 ${
          solid ? "accent-glow" : "accent-glow-teal"
        }`}
      />
      <span className="relative">{children}</span>
    </a>
  );
}

/**
 * The four query types, each with a REAL example.
 *
 * Every `example` below is verbatim from `strata.json`, the query set the benchmark
 * actually scores against. None was written for this page. Inventing a plausible-looking
 * query here would be the same class of mistake as minting a plausible identifier: it
 * reads as evidence while being decoration.
 */
const QUERY_TYPES = [
  {
    name: "synthesis",
    shape: "“what is known about X”",
    judge: "review authors",
    scored: "recall@20 · coverage of the set",
    example: "Antigen-positive relapse. in Mechanisms of resistance to CAR T cell therapy",
  },
  {
    name: "concept",
    shape: "2-word MeSH term",
    judge: "NLM indexers",
    scored: "success@5 · was it on screen",
    example: "Cytokine Release Syndrome",
  },
  {
    // NOT only genes, which is a measured correction rather than a wording change. The
    // stratum is 103 protein, 70 clinical trial, 39 gene, 28 variant, 22 cell line and
    // 5 drug concepts; a component built on the assumption that it was genes covered an
    // eighth of it.
    name: "identifier",
    shape: "bare gene, drug, cell line or trial",
    judge: "citing author",
    scored: "success@1 · exact lookup",
    example: "EGFR T790M",
  },
  {
    name: "claim",
    shape: "28-word sentence",
    judge: "citing author",
    scored: "MRR · find the source",
    example:
      "Blocking antibodies against PD-1 or PD-L1 have demonstrated substantial clinical "
      + "activity in patients with metastatic melanoma, renal cell carcinoma, non-small "
      + "cell lung cancer, and other tumors.",
  },
];

/** The retrieval pipeline, in the order a query passes through it. */
const PIPELINE = [
  {
    n: "01",
    title: "Ingest",
    sub: "PubMed + PMC Open Access",
    body: "Papers arrive as verbatim full text, never abstracts. Bibliographies are "
        + "stripped before indexing: they are a median 19% of an article's characters and "
        + "they match queries lexically while containing no findings.",
  },
  {
    n: "02",
    title: "Chunk",
    sub: "~900 characters, offsets preserved",
    body: "Each passage keeps (doc_id, section, start_char, end_char) back to its source. "
        + "That provenance is the product: a retrieval change that improves ranking but "
        + "loses it is a regression, not a trade.",
  },
  {
    n: "03",
    title: "Index",
    sub: "BM25 + dense vectors in Postgres",
    body: "A lexical index over the passage text and a pgvector column of embeddings, in "
        + "the same database as the documents, so grants and authors can be joined "
        + "against retrieval results.",
  },
  {
    n: "04",
    title: "Retrieve",
    sub: "two arms, fused by rank",
    body: "The query runs against BM25 and the dense index independently. Reciprocal Rank "
        + "Fusion combines the two ranked lists rather than their scores, because BM25 "
        + "scores are unbounded while cosine similarity lives in [-1, 1] and a naive sum "
        + "is dominated by whichever has the larger numeric range.",
  },
  {
    n: "05",
    title: "Aggregate",
    sub: "passages to documents",
    body: "A document scores as its single best passage. That is a real choice: rewarding "
        + "one decisive passage rather than a paper that is vaguely relevant throughout.",
  },
  {
    n: "06",
    title: "Serve",
    sub: "the passage, with its offsets",
    body: "Results carry the matched clause and the characters it sits at, so a claim can "
        + "be checked against the paper instead of trusted.",
  },
  {
    // Deliberately numbered outside the retrieval path, because it IS outside it. Figures
    // are stored and displayed and are excluded from both retrieval arms by an explicit
    // SQL predicate, so adding them provably cannot move a ranking metric.
    n: "R",
    title: "Read",
    sub: "the paper, with its figures",
    body: "Opening a result shows the whole article in reading order, plus its figures: "
        + "9,549 images served from NCBI's own Open Access store, with the publisher's "
        + "verbatim caption. Figures are NOT part of retrieval — captions are already "
        + "indexed as body text, so their effect on ranking is unmeasured and they are "
        + "held out of both arms until it is.",
  },
];

/**
 * Every metric in the panel, and what it answers FOR A RAG SYSTEM specifically.
 *
 * Retrieval quality is the ceiling on generation quality: a model can only be as accurate
 * as the passages handed to it, and it cannot know what it was not shown. So each metric
 * here is described by the generation failure it predicts, not by its textbook definition.
 */
const METRICS = [
  {
    name: "recall@20",
    role: "gate: synthesis",
    asks: "Of the papers that belong in the answer, how many are in the top 20?",
    rag: "A literature-review answer is bounded by the set the generator sees. A paper "
       + "missing from the context is a claim the model cannot make and will not know it "
       + "failed to make. This is the only metric here that measures an absence.",
  },
  {
    name: "success@5",
    role: "gate: concept",
    asks: "Did at least one right paper reach the first screen?",
    rag: "Short topical queries are typed dozens of times a day while scoping. If the "
       + "right paper is at rank 12 it is not in the context window and, for the reader, "
       + "does not exist.",
  },
  {
    name: "success@1",
    role: "gate: identifier",
    asks: "Was the very first result the right one?",
    rag: "For an exact lookup the top hit is often the only one read. Returning a paper "
       + "about a DIFFERENT variant is worse than returning nothing, because the mistake "
       + "is invisible in a fluent answer.",
  },
  {
    name: "mrr",
    role: "gate: claim",
    asks: "How far down the list is the first correct answer?",
    rag: "Proxy for how much irrelevant context the generator wades through before "
       + "reaching signal. Rank 1 and rank 8 both count as a hit at k=10, and they are "
       + "not the same prompt.",
  },
  {
    name: "ndcg@10",
    role: "secondary",
    asks: "Graded, position-weighted quality of the whole top 10.",
    rag: "The top-k IS the context window, so scoring it as a unit rather than as a hit "
       + "or miss is the closest single number to what the generator receives.",
  },
  {
    name: "success@10 / @20",
    role: "secondary",
    asks: "Does the answer survive at deeper cutoffs?",
    rag: "Catches a change that sharpens the head while pushing answers out of the tail. "
       + "A reranker can look excellent at k=1 and lose documents at k=20.",
  },
  {
    name: "unjudged@10",
    role: "diagnostic, never a target",
    asks: "What fraction of returned documents has nobody judged?",
    rag: "The honesty valve. If a change raises this, its apparent gain may only be that "
       + "it returns documents the pool never assessed. Optimising it directly would mean "
       + "preferring documents that happen to have been judged.",
  },
  {
    name: "bpref",
    role: "reported when defined",
    asks: "Rank quality counting only JUDGED non-relevant documents.",
    rag: "Robust to incomplete judgments, which is the normal condition here. It returns "
       + "nothing below 10 judged negatives rather than averaging noise into a verdict.",
  },
];

/**
 * The guards. Each exists because the corresponding failure actually happened in this
 * project, which is why they are stated as rules rather than as aspirations.
 */
const GUARDS = [
  {
    name: "Per-stratum gating",
    what: "Four query types scored separately; no pooled mean.",
    why: "An aggregate rises while exact-identifier lookup collapses. Round 1 found the "
       + "optimal lexical weight points in OPPOSITE directions for short and long "
       + "queries; a pooled mean would have averaged that into 'no change' and discarded it.",
  },
  {
    name: "Pareto dominance, not a weighted score",
    what: "Ships only if better on at least one stratum and worse on none.",
    why: "Measured, not hypothetical: MedCPT wins the weighted composite by 3x and is "
       + "refused, because its coverage gain is paid for with a significant regression on "
       + "find-the-source. Those are different jobs for the same user, and the weights "
       + "trading them off were chosen by us rather than measured.",
  },
  {
    name: "Pre-registered predictions",
    what: "Each candidate states its stratum, metric, direction and minimum effect before it runs.",
    why: "Run enough candidates against enough metrics and something looks significant. "
       + "Writing the prediction down first is what makes a result capable of being wrong, "
       + "and an unpredicted win is recorded as a hypothesis for next round, not as success.",
  },
  {
    name: "Holm across candidates",
    what: "One gate metric per stratum, corrected over the candidates tried.",
    why: "The earlier gate corrected across 5 to 8 metrics that are deterministic "
       + "functions of a single rank. That controls nothing and inflated the detectable "
       + "effect by about 25%, on a harness already short of power.",
  },
  {
    name: "Regression veto, uncorrected",
    what: "Any significant drop on any secondary metric refuses the change.",
    why: "A false positive here means refusing a change, which is the safe direction. "
       + "Making a safety check harder to trip is backwards.",
  },
  {
    name: "NO_EFFECT detection",
    what: "Byte-identical rankings are reported as a wiring bug, not a negative result.",
    why: "Three candidates in this project were structurally incapable of moving the "
       + "metric they were judged on. Each time the loop was one step from recording a "
       + "confident negative about an idea that never ran.",
  },
  {
    name: "Minimum detectable effect",
    what: "Published per stratum, next to the result.",
    why: "'No significant change' and 'this experiment cannot see a change that size' "
       + "look identical in a table and mean opposite things. Blindness that reads as "
       + "rigour is the most dangerous failure a harness has.",
  },
  {
    name: "Locked test split",
    what: "Candidates only ever touch dev; test is spent once.",
    why: "The real defence against fitting the benchmark across many rounds. Nothing on "
       + "this page has spent it.",
  },
  {
    name: "The citing paper is excluded from its own results",
    what: "assert_source_excluded() raises rather than warns.",
    why: "The source contains the query verbatim, so leaving it in would measure string "
       + "equality and score near-perfectly. A convention you can forget is not a guard.",
  },
  {
    name: "A raw term-frequency floor",
    what: "A ~20-line scorer with no IDF and no length normalisation.",
    why: "Random and popularity baselines are flattering. The honest question is how much "
       + "the system beats the simplest thing that could work.",
  },
];

/** How the tool is used. Placed above the design narrative because a reader who cannot
 *  work the thing has no reason to care how it was built. */
/**
 * The infrastructure, and what each piece is FOR.
 *
 * A reader evaluating a retrieval system wants to know what it runs on and why those
 * choices, not a logo wall. Each row names the decision, the alternative that was rejected,
 * and the measurement or constraint that decided it — so the list reads as reasoning rather
 * than as a stack.
 */
const INFRASTRUCTURE = [
  {
    layer: "Store",
    what: "Neon Postgres + pgvector",
    why: "Passages, embeddings, documents, grants and MeSH live in ONE database, so a "
       + "retrieval result can be joined against funding and indexing without a second "
       + "system. HNSW over cosine, because the vectors are L2-normalised and IVFFlat "
       + "needs a training step this corpus does not justify.",
    instead: "a dedicated vector DB — which would put the metadata a query needs on the "
           + "other side of a network hop",
  },
  {
    layer: "Retrieval",
    what: "BM25 + dense, fused by Reciprocal Rank Fusion",
    why: "RRF combines RANKS, not scores. BM25 is unbounded and cosine lives in [-1,1], so "
       + "a naive weighted sum is dominated by whichever has the larger numeric range. "
       + "Both arms run in SQL in one round trip.",
    instead: "score normalisation, which needs a calibration set that does not exist here",
  },
  {
    layer: "Embeddings",
    what: "text-embedding-3-small at 192 dimensions",
    why: "Matryoshka truncation: the same model serves 192 or 768 dims, so vector width is "
       + "a measurable knob rather than a model migration. 192 keeps the whole index inside "
       + "a serverless function's reach.",
    instead: "MedCPT as the dense arm — measured WORSE on find-the-source, and it needs "
           + "torch (~2 GB), which does not fit a Vercel function",
  },
  {
    layer: "Reranking",
    what: "MedCPT cross-encoder over the fused top 50",
    why: "A bi-encoder embeds query and passage separately and cannot represent an "
       + "interaction between them: it knows a passage is ABOUT a topic, not whether it "
       + "REPORTS the finding. Largest measured gain in the project.",
    instead: "an LLM reranker at ~$0.0017/query, which made depth a budget decision and a "
           + "full evaluation sweep cost real money",
  },
  {
    layer: "Figures & tables",
    what: "JATS markup; images served from NCBI's own S3",
    why: "PMC publishes structured XML, so figure captions, image URLs and table GRIDS are "
       + "already stated by the publisher. Nothing is OCR'd and no image is re-hosted. "
       + "Tables keep their row and column headers, so a hazard ratio does not become a "
       + "decontextualised number.",
    instead: "page-layout detection — which would re-derive from pixels what the markup "
           + "already states, and PubLayNet's own training labels came from this same XML",
  },
  {
    layer: "Serving",
    what: "Next.js on Vercel, Python functions for the API",
    why: "The retrieval path is Python because the harness is; the same code answers a "
       + "query in production and in the evaluation loop, so the thing measured is the "
       + "thing served.",
    instead: "a bundled snapshot artifact — which served a corpus the site no longer had",
  },
  {
    layer: "Evaluation",
    what: "Citation-mined labels, four strata, a locked test split",
    why: "Labels are FOUND, not written: a sentence citing a paper is a domain expert "
       + "describing it after reading it. Nothing was annotated for this project, so the "
       + "benchmark cannot be tuned toward the answers.",
    instead: "hand-judged relevance — which does not scale and grades our own homework",
  },
];

/**
 * Design decisions, stated as the decision plus the thing it rules out.
 *
 * Written this way because a decision without its rejected alternative is not a decision,
 * it is a description — and several of these exist only because the obvious choice was
 * tried first and measured worse.
 */
const DECISIONS = [
  {
    q: "Why are figures shown but not searched?",
    a: "Captions are ALREADY indexed — NCBI inlines them into the plain text, at a measured "
     + "median of 100% token coverage. So the text a figure carries is already retrievable, "
     + "and adding figure rows to the index would change the corpus every candidate has "
     + "been measured against. They are stored, displayed, and excluded from both retrieval "
     + "arms by an explicit SQL predicate until that effect is measured.",
  },
  {
    q: "Why isn't a vision model reading the plots?",
    a: "Because the failure is silent. A model that writes \"median OS 18.9 months\" where "
     + "the truth is 22.4 means a query for 18.9 retrieves the WRONG figure confidently and "
     + "a query for 22.4 misses the right one — and the reader sees the real caption and "
     + "image, so the error is invisible. It lives in the ranking layer, where nothing is "
     + "displayed and nothing is checked. The rule is: prefer transcription over inference "
     + "for anything that enters the index.",
  },
  {
    q: "Why a keyword list for figure types, when the notes argue against regexes?",
    a: "Because the publisher already named the type: a caption saying \"Kaplan-Meier\" is "
     + "the paper stating a fact, and matching that word is recognition, not inference. A "
     + "semantic classifier was built for the half it misses and MEASURED AS UNUSABLE — "
     + "80% precision only at 3.3% coverage. It is not applied. What did work was noticing "
     + "36.5% of figures match SEVERAL types, so the fix was to stop discarding what the "
     + "keyword list already computed.",
  },
  {
    q: "Why does nothing here ship on a good-looking number?",
    a: "One pre-registered gate metric per query type, Holm-corrected across the candidates "
     + "tried in a round, with every other metric acting as an uncorrected veto. A change "
     + "ships only if it improves at least one query type and worsens none. Five rounds "
     + "have produced far more refutations than promotions, which is the point.",
  },
];

const HOW_TO_USE = [
  {
    n: "01",
    title: "Search a concept, get the passage",
    body: "Type what you would say to a colleague: a mechanism, a drug, a variant. "
        + "Results are papers, each carrying the exact passage that matched and the "
        + "character offsets it sits at, so you can check the claim rather than trust it.",
    example: "EGFR C797S resistance to osimertinib",
  },
  {
    n: "02",
    title: "Compare papers on the same dimensions",
    body: "Switch to Compare and the same query returns a table: papers down the side, "
        + "technical dimensions across the top: cohort, assay, endpoint, effect size. "
        + "Each filled cell cites the passage it came from.",
    example: "CAR-T persistence in solid tumors",
  },
  {
    n: "03",
    title: "Open any cell to read the passage it came from",
    body: "A comparison table you cannot audit is one you should not trust. Every filled "
        + "cell opens the verbatim passage at the character offsets it was drawn from, so "
        + "the claim in the grid can be checked against the paper rather than believed.",
    example: null,
  },
];

export default function OverviewClient({ journey, clusters }: { journey: Journey | null; clusters: any }) {
  const stages = journey?.stages ?? [];
  const corpus = stages.find((s) => s.id === "goal");
  const retrieval = stages.find((s) => s.id === "retrieval");
  /** Live figures come from the journey artifact, never from a literal typed here. */
  const liveMetric = (label: string) =>
    corpus?.metrics?.find((m) => m.label === label)?.value;
  const passages = liveMetric("Retrievable passages");
  const power = journey?.power ?? [];

  return (
    // NO opaque background on this wrapper: `html, body` in globals.css already paint
    // #060B14, and an opaque colour here would sit in the root stacking context ABOVE the
    // fixed -z-10 canvas and hide it completely — the canvas would render, cost GPU time,
    // and never be seen. This mirrors how /search mounts it.
    <div className="relative min-h-screen text-slate-200">
      {/* Held at 0.55 intensity: this page is long-form reading, and the field must sit
          under the prose rather than beside it. Parallax gives the scroll depth without
          adding motion in the reader's focal area. */}
      <WebGLBackground intensity={0.55} parallax />

      <main className="relative mx-auto max-w-4xl px-6 pb-32">
        {/* ---------- what this is ---------- */}
        <header className="border-b border-white/8 py-16">
          <h1 className="text-3xl font-medium leading-tight text-white sm:text-4xl">
            A RAG system over the oncology literature
          </h1>
          <p className="mt-5 max-w-2xl text-[15px] leading-relaxed text-slate-400">
            Search by concept and get back the paper <em className="not-italic text-slate-200">
            and the passage where the concept appears</em>, with character offsets into the
            source. Built for the question an R&amp;D team actually asks, “what is known
            about X?”, which is answered by a set of papers, not one.
          </p>

          {corpus?.metrics?.length ? (
            <dl className="mt-10 grid grid-cols-2 gap-x-8 gap-y-5 sm:grid-cols-4">
              {corpus.metrics.map((m) => (
                <LiveMetric key={m.label} m={m} />
              ))}
            </dl>
          ) : null}

          <div className="mt-10 flex flex-wrap gap-3">
            <GlowLink href="/search">Open search</GlowLink>
            <GlowLink href="#measurement" tone="outline">How it is measured</GlowLink>
          </div>
        </header>

        {/* ---------- the corpus, as a map ----------
            FIRST on the page, before the instructions. A reader's first question about a
            literature tool is "what literature?", and an answer they can see and rotate
            settles it faster than a paragraph claiming coverage. */}
        {clusters ? (
          <section className="border-b border-white/8 py-14">
            <SectionHeading>The literature this searches</SectionHeading>
            <div className="mt-3 max-w-2xl space-y-3 text-sm leading-relaxed text-slate-400">
              <p>
                {clusters.n_documents?.toLocaleString?.() ?? ""} peer-reviewed oncology
                papers from PubMed and PMC, every one held as{" "}
                <span className="text-slate-200">verbatim full text</span> rather than an
                abstract, split into{" "}
                {typeof passages === "number" ? fmt(passages) : "over 100,000"}{" "}
                individually retrievable passages. Abstract-only records were removed: an
                abstract is already a summary, so there is no passage inside it to point at.
              </p>
              <p>
                The set was grown along its own citation graph. Starting from MeSH-seeded
                oncology searches, the papers those papers cite were ingested too, screened
                against NLM&apos;s MeSH tree so the corpus stays oncology rather than
                drifting into general molecular biology. Screening the top 9,000 candidates
                kept 5,389 of them, so <span className="text-slate-200">two in five fail
                </span>: they are cited by oncology work without being oncology work, and
                would have drifted the corpus off its subject while every retrieval metric
                continued to look healthy.
              </p>
              <p>
                Below, each point is one paper, positioned by the same embedding retrieval
                uses, so two papers sit together for the reason a query would return both.
                Regions are named by the MeSH major topics most distinctive to each,
                measured by log-odds against the whole corpus, which is why they read
                &ldquo;Cytokine Release Syndrome&rdquo; rather than &ldquo;Humans&rdquo;.
                Drag to rotate, click a region to list its papers.
              </p>
            </div>
            {/* Deliberately breaks the max-w-4xl reading column on wide screens. This is
                the one element on the page that is a picture rather than prose, and a
                3,166-point cloud is unreadable squeezed into a text measure. Negative
                margins only engage above lg, so narrow viewports are untouched. */}
            <div className="mt-7 lg:-mx-24 xl:-mx-48 2xl:-mx-72">
              <ClusterMap data={clusters} />
            </div>
          </section>
        ) : null}

        {/* ---------- how the system is built ---------- */}
        <section className="border-b border-white/8 py-14">
          <SectionHeading>How it works</SectionHeading>
          <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-400">
            A query takes six steps from typed text to a cited passage. The one design rule
            everything else defers to is that a result must be able to point back at its
            source, because a claim you cannot check is a claim a generator can invent.
          </p>

          <ol className="mt-7 grid gap-px overflow-hidden rounded-lg bg-white/8 sm:grid-cols-2 lg:grid-cols-3">
            {PIPELINE.map((s) => (
              <li key={s.n} className="bg-[#080d16] p-4">
                <div className="flex items-baseline gap-2">
                  <span className="font-mono text-[11px] text-cyan-300/70">{s.n}</span>
                  <h3 className="text-sm font-medium text-white">{s.title}</h3>
                </div>
                <p className="mt-0.5 font-mono text-[10px] uppercase tracking-wider text-slate-600">
                  {s.sub}
                </p>
                <p className="mt-2 text-xs leading-relaxed text-slate-400">{s.body}</p>
              </li>
            ))}
          </ol>

          <h3 className="mt-12 text-sm font-medium text-white">
            The dense arm, and why the model choice was not obvious
          </h3>
          <div className="mt-3 max-w-2xl space-y-3 text-sm leading-relaxed text-slate-400">
            <p>
              The lexical arm matches words. The dense arm is supposed to match{" "}
              <span className="text-slate-200">meaning</span>, so that a search for
              &ldquo;EGFR TKI resistance&rdquo; finds a paper that only ever says
              &ldquo;osimertinib&rdquo;. Which embedding model does that best for oncology
              is a measurable question, and the answer was not the one we expected.
            </p>
            <p>
              <span className="text-slate-200">MedCPT</span> is NCBI&apos;s biomedical
              retriever, trained contrastively on 255 million real
              (query, clicked-article) pairs harvested from PubMed itself. It is a model of
              what biomedical researchers actually search for and which paper they then
              opened. It uses two separate encoders, one for queries and one for articles,
              because a two-word query and a 900-character passage are not the same kind of
              object. Against it we ran{" "}
              <span className="text-slate-200">text-embedding-3-small</span>, a general
              model, at the same 768 dimensions specifically so that a MedCPT win could not
              be confused with simply having four times the vector capacity.
            </p>
            <p>
              <span className="text-slate-200">MedCPT turned out to be a trade, not an
              upgrade.</span> It is a better topical matcher and a worse pinpointer: it won
              literature-review coverage decisively and significantly regressed
              find-the-source. Click data encodes &ldquo;this article is about what you
              asked&rdquo;, not &ldquo;this is the exact sentence you want&rdquo;, and the
              smoothing that lifts topical recall costs exact attribution.
            </p>
            <p>
              What worked was refusing to choose. Running{" "}
              <span className="text-slate-200">both</span> dense models alongside BM25 and
              fusing three ranked lists beat either model alone on both query types, and
              cancelled the regression entirely. Arms that are individually competent and
              wrong about <em className="not-italic text-slate-200">different</em> queries
              are exactly when rank fusion pays.
            </p>
          </div>

          <div className="mt-5 overflow-x-auto">
            <table className="w-full min-w-[520px] text-left text-xs">
              <thead className="text-[10px] uppercase tracking-wider text-slate-500">
                <tr className="border-b border-white/10">
                  <th className="py-2 pr-4 font-normal">dense arm</th>
                  <th className="py-2 pr-4 text-right font-normal">review coverage</th>
                  <th className="py-2 pr-4 text-right font-normal">find the source</th>
                  <th className="py-2 pr-4 font-normal">deployable</th>
                </tr>
              </thead>
              <tbody className="text-slate-300">
                {[
                  ["MedCPT alone", "+0.0261", "−0.0166", true, false, "needs a hosted GPU endpoint"],
                  ["general model, 768-dim", "+0.0016", "+0.0128", false, true, "one API call"],
                  ["all three, fused", "+0.0305", "+0.0321", true, true, "both of the above"],
                ].map(([name, syn, clm, synGood, clmGood, deploy]) => (
                  <tr key={String(name)} className="border-b border-white/5">
                    <td className="py-2.5 pr-4 font-medium text-white">{name}</td>
                    <td className={`py-2.5 pr-4 text-right font-mono tabular-nums ${
                      synGood ? "text-emerald-300/90" : "text-slate-500"}`}>{syn}</td>
                    <td className={`py-2.5 pr-4 text-right font-mono tabular-nums ${
                      clmGood ? "text-emerald-300/90" : "text-rose-300/80"}`}>{clm}</td>
                    <td className="py-2.5 pr-4 text-[11px] text-slate-500">{deploy}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 max-w-2xl text-xs leading-relaxed text-slate-500">
            Measured on the dev split, both gains significant at p &lt; 0.0001. The fused
            configuration is <span className="text-slate-300">not shipped</span>: two
            controls that would attribute its gain have not run, two of the four query
            types were not evaluated, and the locked test split is unspent.
          </p>
        </section>

        {/* ---------- infrastructure ---------- */}
        <section className="border-b border-white/8 py-14">
          <SectionHeading>What it runs on, and why</SectionHeading>
          <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-400">
            Each row names the choice, what it was chosen{" "}
            <em className="not-italic text-slate-300">instead of</em>, and the constraint or
            measurement that decided it. Several of these exist only because the obvious
            option was tried first and measured worse.
          </p>
          <div className="mt-5 space-y-px overflow-hidden rounded-lg bg-white/8">
            {INFRASTRUCTURE.map((r) => (
              <div key={r.layer} className="bg-[#080d16] p-4 sm:grid sm:grid-cols-[9rem_1fr] sm:gap-5">
                <div>
                  <h3 className="text-xs font-medium text-white">{r.layer}</h3>
                  <p className="mt-1 text-[11px] leading-relaxed text-teal/80">{r.what}</p>
                </div>
                <div className="mt-2 sm:mt-0">
                  <p className="text-xs leading-relaxed text-slate-400">{r.why}</p>
                  <p className="mt-1.5 text-[11px] leading-relaxed text-slate-600">
                    <span className="uppercase tracking-wider text-slate-700">instead of</span>{" "}
                    {r.instead}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ---------- figures and tables ---------- */}
        <section className="border-b border-white/8 py-14">
          <SectionHeading>Figures and tables</SectionHeading>
          <div className="mt-3 max-w-2xl space-y-3 text-sm leading-relaxed text-slate-400">
            <p>
              Opening any result shows the whole paper with its{" "}
              <span className="text-slate-200">9,549 figures</span> and{" "}
              <span className="text-slate-200">3,682 tables</span>. Images are served from
              NCBI&apos;s own Open Access store rather than copied, and captions are the
              publisher&apos;s verbatim text. Tables keep their row and column headers, so a
              hazard ratio arrives with the label that gives it meaning instead of as a loose
              number in a stream of cells.
            </p>
            <p>
              <span className="text-slate-200">What is deliberately missing is the reading
              of the plots.</span> A Kaplan-Meier curve, a dose-response curve and a forest
              plot all encode their result <em className="not-italic text-slate-200">
              positionally</em> — the answer is where the lines sit relative to each other,
              and nothing in the file says which arm did better. So the honest question is
              whether a vision model can recover that, and it was measured rather than
              assumed.
            </p>
          </div>

          <div className="mt-5 overflow-x-auto">
            <table className="w-full min-w-[480px] text-left text-xs">
              <thead className="text-[10px] uppercase tracking-wider text-slate-500">
                <tr className="border-b border-white/10">
                  <th className="py-2 pr-4 font-normal">figure type</th>
                  <th className="py-2 pr-4 text-right font-normal">n</th>
                  <th className="py-2 pr-4 text-right font-normal">sees the image</th>
                  <th className="py-2 pr-4 text-right font-normal">caption only</th>
                  <th className="py-2 pr-4 text-right font-normal">gain from looking</th>
                </tr>
              </thead>
              <tbody className="text-slate-300">
                {[["Kaplan-Meier", 58, "9%", "0%", "+9%"],
                  ["ROC curve", 15, "0%", "0%", "0%"],
                  ["Forest plot", 11, "0%", "0%", "0%"],
                  ["Dose-response", 6, "0%", "0%", "0%"]].map(([t, n, v, b, g]) => (
                  <tr key={String(t)} className="border-b border-white/5">
                    <td className="py-2.5 pr-4">{t}</td>
                    <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-slate-400">{n}</td>
                    <td className="py-2.5 pr-4 text-right font-mono tabular-nums">{v}</td>
                    <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-slate-500">{b}</td>
                    <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-amber-300/80">{g}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-4 max-w-2xl border-l-2 border-amber-400/40 bg-amber-400/[0.03] py-3 pl-4 pr-4 text-sm leading-relaxed text-amber-200/70">
            A vision model was asked to recover the exact value the paper&apos;s own authors
            quoted from each figure, scored twice: once seeing the image, once seeing only the
            caption. <span className="text-amber-100">6% against 0%.</span> The gap is what
            looking buys, and it is small enough that indexing those numbers would be indexing
            a guess. Forest plots scored zero despite printing their effect sizes as text,
            which points at resolution rather than reasoning. So figures are shown, and their
            values are not asserted.
          </p>
        </section>

        {/* ---------- design decisions ---------- */}
        <section className="border-b border-white/8 py-14">
          <SectionHeading>Design decisions</SectionHeading>
          <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-400">
            The four questions a reader is most likely to think are oversights.
          </p>
          <div className="mt-5 space-y-px overflow-hidden rounded-lg bg-white/8">
            {DECISIONS.map((d) => (
              <div key={d.q} className="bg-[#080d16] p-4">
                <h3 className="text-sm font-medium text-white">{d.q}</h3>
                <p className="mt-1.5 max-w-3xl text-xs leading-relaxed text-slate-400">{d.a}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ---------- how to use ---------- */}
        <section className="border-b border-white/8 py-14">
          <SectionHeading>Using it</SectionHeading>
          <div className="mt-7 space-y-7">
            {HOW_TO_USE.map((s) => (
              <div key={s.n} className="grid gap-3 sm:grid-cols-[2.5rem_1fr]">
                <span className="font-mono text-xs text-slate-600">{s.n}</span>
                <div>
                  <h3 className="text-sm font-medium text-white">{s.title}</h3>
                  <p className="mt-1.5 max-w-2xl text-sm leading-relaxed text-slate-400">{s.body}</p>
                  {s.example ? (
                    <a href={`/search?q=${encodeURIComponent(s.example)}`}
                       className="mt-2.5 inline-block rounded border border-white/10 bg-white/[0.03] px-2.5 py-1 font-mono text-[11px] text-cyan-300/80 transition-colors hover:border-cyan-400/30 hover:text-cyan-200">
                      {s.example} →
                    </a>
                  ) : null}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ---------- measurement ---------- */}
        <section id="measurement" className="border-b border-white/8 py-14">
          <SectionHeading>How it is measured</SectionHeading>
          <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-400">
            Four query types, because they are different jobs and a single average hides a
            regression in the one that matters least often and costs most when wrong.
          </p>

          <div className="mt-6 overflow-x-auto">
            <table className="w-full min-w-[600px] text-left text-xs">
              <thead className="text-[10px] uppercase tracking-wider text-slate-500">
                <tr className="border-b border-white/10">
                  <th className="py-2 pr-4 font-normal">query type</th>
                  <th className="py-2 pr-4 font-normal">shape</th>
                  <th className="py-2 pr-4 font-normal">judged by</th>
                  <th className="py-2 pr-4 font-normal">scored on</th>
                </tr>
              </thead>
              <tbody className="text-slate-300">
                {QUERY_TYPES.map((t) => (
                  <Fragment key={t.name}>
                    <tr className="border-t border-white/5">
                      <td className="pt-3 pr-4 font-medium text-white">{t.name}</td>
                      <td className="pt-3 pr-4 text-slate-400">{t.shape}</td>
                      <td className="pt-3 pr-4 text-slate-400">{t.judge}</td>
                      <td className="pt-3 pr-4 font-mono text-[11px] text-slate-400">{t.scored}</td>
                    </tr>
                    <tr>
                      <td colSpan={4} className="pb-3.5 pt-1.5">
                        {/* Every example below is a query the benchmark actually contains,
                            lifted from strata.json. None was written for the page. */}
                        <a
                          href={`/search?q=${encodeURIComponent(t.example)}`}
                          className="group flex items-start gap-2 rounded border border-white/8 bg-white/[0.02] px-2.5 py-2 transition-colors hover:border-cyan-400/30 hover:bg-cyan-400/[0.04]"
                        >
                          <span className="mt-px shrink-0 text-[10px] uppercase tracking-wider text-slate-600">
                            e.g.
                          </span>
                          <span className="min-w-0 flex-1 text-[12px] leading-snug text-slate-300 group-hover:text-slate-100">
                            {t.example}
                          </span>
                          <span className="shrink-0 text-[11px] text-cyan-300/60 group-hover:text-cyan-300">
                            try it
                          </span>
                        </a>
                      </td>
                    </tr>
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-6 space-y-3 text-sm leading-relaxed text-slate-400">
            <p>
              <span className="text-slate-200">Labels are found, not written.</span> A
              citing sentence is a domain expert&apos;s description of the work it cites; a
              review section heading is an R&amp;D question whose cited papers are the
              answer set; MeSH major topics are NLM&apos;s human indexing. Nothing was
              annotated for this project, and no model chose its own training signal.
            </p>
          </div>

          {/* ---------- every metric, and what it predicts about generation ---------- */}
          <h3 className="mt-12 text-sm font-medium text-white">
            What each number means for a RAG answer
          </h3>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">
            Retrieval quality is the ceiling on generation quality. A model can only be as
            accurate as the passages handed to it, and it cannot know what it was never
            shown, so a retrieval metric that flatters itself produces a system that is
            confidently wrong. Each metric below is described by the failure it predicts.
          </p>

          <div className="mt-5 grid gap-px overflow-hidden rounded-lg bg-white/8 sm:grid-cols-2">
            {METRICS.map((m) => (
              <div key={m.name} className="bg-[#080d16] p-4">
                <div className="flex items-baseline justify-between gap-3">
                  <code className="font-mono text-[13px] text-cyan-300">{m.name}</code>
                  <span className="shrink-0 text-[10px] uppercase tracking-wider text-slate-600">
                    {m.role}
                  </span>
                </div>
                <p className="mt-1.5 text-xs leading-relaxed text-slate-300">{m.asks}</p>
                <p className="mt-1.5 text-xs leading-relaxed text-slate-500">{m.rag}</p>
              </div>
            ))}
          </div>

          {/* ---------- what each stratum can actually resolve ---------- */}
          {power.length > 0 && (
            <>
              <h3 className="mt-12 text-sm font-medium text-white">
                What the benchmark can and cannot see
              </h3>
              <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">
                The minimum detectable effect is the smallest change each stratum could
                distinguish from noise, at 80% power. Anything below it is invisible by
                construction, so a result there says nothing about the idea being tested.
                Publishing it next to the sample size is what separates a negative result
                from a blind one.
              </p>
              <div className="mt-5 overflow-x-auto">
                <table className="w-full min-w-[560px] text-left text-xs">
                  <thead className="text-[10px] uppercase tracking-wider text-slate-500">
                    <tr className="border-b border-white/10">
                      <th className="py-2 pr-4 font-normal">stratum</th>
                      <th className="py-2 pr-4 text-right font-normal">queries</th>
                      <th className="py-2 pr-4 text-right font-normal">judgments</th>
                      <th className="py-2 pr-4 text-right font-normal">weight</th>
                      <th className="py-2 pr-4 font-normal">gate metric</th>
                      <th className="py-2 pr-4 text-right font-normal">smallest visible effect</th>
                    </tr>
                  </thead>
                  <tbody className="text-slate-300">
                    {power.map((r) => (
                      <tr key={r.stratum} className="border-b border-white/5">
                        <td className="py-2.5 pr-4 font-medium text-white">{r.stratum}</td>
                        <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                          {r.queries.toLocaleString()}
                        </td>
                        <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-slate-400">
                          {r.judgments.toLocaleString()}
                        </td>
                        <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-slate-400">
                          {r.weight != null ? r.weight.toFixed(2) : "n/a"}
                        </td>
                        <td className="py-2.5 pr-4 font-mono text-[11px] text-slate-400">
                          {r.gate_metric}
                        </td>
                        <td className="py-2.5 pr-4 text-right">
                          <LiveNumber
                            tone={r.sees_002 ? "good" : "bad"}
                            title={r.mde_source ?? "minimum detectable effect"}
                          >
                            {r.mde.toFixed(4)}
                          </LiveNumber>
                          <span className={`ml-1.5 text-[10px] uppercase tracking-wider ${
                            r.sees_002 ? "text-emerald-300/70" : "text-amber-300/70"}`}>
                            {r.sees_002 ? "sees 0.02" : "blind below"}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

          {/* ---------- the guards ---------- */}
          <h3 className="mt-12 text-sm font-medium text-white">
            What stops these numbers from flattering themselves
          </h3>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">
            Each of these exists because the corresponding failure actually happened here,
            which is why they are rules rather than intentions.
          </p>
          <div className="mt-5 space-y-px overflow-hidden rounded-lg bg-white/8">
            {GUARDS.map((g) => (
              <div key={g.name} className="bg-[#080d16] p-4 sm:grid sm:grid-cols-[15rem_1fr] sm:gap-5">
                <div>
                  <h4 className="text-xs font-medium text-white">{g.name}</h4>
                  <p className="mt-1 text-[11px] leading-relaxed text-slate-500">{g.what}</p>
                </div>
                <p className="mt-2 text-xs leading-relaxed text-slate-400 sm:mt-0">{g.why}</p>
              </div>
            ))}
          </div>

          <p className="mt-6 max-w-2xl border-l-2 border-white/15 pl-4 text-sm leading-relaxed text-slate-400">
            <span className="text-slate-200">A change ships only if it is a Pareto
            improvement</span>: better on at least one query type and worse on none.
            A weighted average would let a gain on common queries pay for a regression on
            exact lookup, where returning the wrong variant is worse than returning nothing
            because the error is invisible in the results.
          </p>

          {retrieval?.systems?.length ? (
            <div className="mt-7 overflow-x-auto">
              <table className="w-full min-w-[520px] text-left text-xs">
                <thead className="text-[10px] uppercase tracking-wider text-slate-500">
                  <tr className="border-b border-white/10">
                    {Object.keys(retrieval.systems[0]).map((c) => (
                      <th key={c} className={`py-2 pr-4 font-normal ${c === "system" ? "" : "text-right"}`}>{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="text-slate-400">
                  {retrieval.systems.map((row) => (
                    <tr key={String(row.system)} className="border-b border-white/5">
                      {Object.entries(row).map(([k, v]) => (
                        <td key={k} className={`py-2 pr-4 ${
                          k === "system" ? "font-sans text-slate-300" : "text-right"}`}>
                          {typeof v === "number" ? (
                            <LiveNumber
                              tone={k === "unjudged@10" ? "neutral" : "good"}
                              title={k === "unjudged@10"
                                ? "diagnostic, not a score: the fraction of returned documents nobody judged"
                                : `${k} on ${row.system}`}
                            >
                              {v.toFixed(4)}
                            </LiveNumber>
                          ) : (
                            String(v)
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}

          {retrieval?.caveat ? (
            <p className="mt-6 border-l-2 border-amber-400/40 bg-amber-400/[0.03] py-3 pl-4 pr-4 text-sm leading-relaxed text-amber-200/70">
              {retrieval.caveat}
            </p>
          ) : null}
        </section>


        <footer className="py-12 text-xs leading-relaxed text-slate-500">
          {journey ? (
            <p>
              Built {journey.generated_at} at rev {journey.git_rev}.
            </p>
          ) : (
            <p>
              Measurement artifacts have not been generated. Run{" "}
              <code className="text-slate-400">python scripts/build_journey_data.py</code>.
              This page shows nothing rather than placeholder numbers.
            </p>
          )}
        </footer>
      </main>
    </div>
  );
}
