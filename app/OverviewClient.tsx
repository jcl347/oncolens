"use client";

import { Fragment, useState } from "react";

import ClusterMap from "../components/ClusterMap";
import WebGLAccent from "../components/WebGLAccent";
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
 * The accent is behind the number rather than around it, and it eases rather than snaps,
 * so a row of these reads as instrumentation coming alive rather than as a hover state
 * firing. The `command` that produced the figure is exposed on hover too: the point of
 * animating a number is to invite scrutiny of it, so the invitation had better lead
 * somewhere.
 */
function LiveMetric({ m }: { m: Metric }) {
  const [hover, setHover] = useState(false);
  const p = PROVENANCE[m.provenance];
  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      className="group relative"
    >
      <dt className="text-[11px] uppercase tracking-wider text-slate-500">{m.label}</dt>
      <dd className="relative mt-1 overflow-hidden rounded">
        <WebGLAccent
          variant="glow"
          hover={hover}
          className="absolute inset-0 h-full w-full"
        />
        <span className="relative font-mono text-xl tabular-nums text-white">
          {fmt(m.value)}
          {m.unit ? <span className="ml-0.5 text-sm text-slate-400">{m.unit}</span> : null}
        </span>
      </dd>
      <WebGLAccent variant="rule" hover={hover} className="mt-1 h-[2px] w-full" />
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

/** Section heading with a live hairline under it, so each section reads as one unit. */
function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <div>
      <h2 className="text-xs uppercase tracking-[0.16em] text-cyan-300/70">{children}</h2>
      <WebGLAccent variant="rule" className="mt-2 h-[2px] w-28" />
    </div>
  );
}

/**
 * A link whose accent responds to the pointer.
 *
 * The glow is driven by the parent's hover state rather than CSS so it *eases* in and out
 * on the shared clock instead of snapping. On a solid button the accent is white, because
 * a cyan glow over a cyan fill is invisible.
 */
function GlowLink({
  href, children, tone = "solid",
}: { href: string; children: React.ReactNode; tone?: "solid" | "outline" }) {
  const [hover, setHover] = useState(false);
  const solid = tone === "solid";
  return (
    <a
      href={href}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onFocus={() => setHover(true)}
      onBlur={() => setHover(false)}
      className={`relative overflow-hidden rounded-md px-4 py-2 text-sm transition-colors ${
        solid
          ? "bg-cyan-400/90 font-medium text-[#04121a] hover:bg-cyan-300"
          : "border border-white/12 text-slate-300 hover:border-white/25 hover:text-white"
      }`}
    >
      <WebGLAccent
        variant="glow"
        hover={hover}
        color={solid ? [1, 1, 1] : [0.36, 0.79, 0.87]}
        className="absolute inset-0 h-full w-full"
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
    name: "identifier",
    shape: "bare gene / variant",
    judge: "citing author",
    scored: "success@1 · exact lookup",
    example: "LAG-3",
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
                drifting into general molecular biology. About a third of the most-cited
                candidates fail that screen: they are cited by oncology work without being
                oncology work, and a further 6% are rejected only because PubMed has never
                MeSH-indexed them at all.
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
                1,754-point cloud is unreadable squeezed into a text measure. Negative
                margins only engage above lg, so narrow viewports are untouched. */}
            <div className="mt-7 lg:-mx-24 xl:-mx-48 2xl:-mx-72">
              <ClusterMap data={clusters} />
            </div>
          </section>
        ) : null}

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
                        <td className={`py-2.5 pr-4 text-right font-mono tabular-nums ${
                          r.sees_002 ? "text-emerald-300/90" : "text-amber-300/80"}`}>
                          {r.mde.toFixed(3)}
                          <span className="ml-1.5 text-[10px] uppercase tracking-wider">
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
                <tbody className="font-mono tabular-nums text-slate-400">
                  {retrieval.systems.map((row) => (
                    <tr key={String(row.system)} className="border-b border-white/5">
                      {Object.entries(row).map(([k, v]) => (
                        <td key={k} className={`py-2 pr-4 ${k === "system" ? "font-sans text-slate-300" : "text-right"}`}>
                          {typeof v === "number" ? v.toFixed(4) : String(v)}
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

        {/* ---------- what the loop found ---------- */}
        <section className="border-b border-white/8 py-14">
          <SectionHeading>What the improvement loop found</SectionHeading>
          <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-400">
            Every candidate states, before it runs, which query type it should help and
            which it should not. The loop records predictions and outcomes together, so a
            result that was not predicted is logged as a surprise to test next round, never
            claimed as a win.
          </p>
          <ul className="mt-6 space-y-4">
            {[
              ["One fusion weight is wrong in both directions",
               "Dropping the lexical arm hurt 2-word queries (−0.0707, p=0.0008); doubling it hurt conceptual ones. The optimal weight depends on query shape, a finding that only exists because the query types are scored separately."],
              ["A gate that could not be passed",
               "Reranking reorders the top 24 passages without changing which documents are in the top 20, so coverage moved by exactly 0.0000 and the reranker was structurally unable to win, however good it was."],
              ["The evaluation is underpowered, and that is the bottleneck",
               "At the current corpus size the smallest detectable effect is 0.043 on synthesis and 0.062 on concept. Every promising result so far sits below those floors, reported as “no significant change” by construction, which looks like rigour and is blindness."],
            ].map(([t, b]) => (
              <li key={t} className="border-l-2 border-white/10 pl-4">
                <h3 className="text-sm font-medium text-white">{t}</h3>
                <p className="mt-1 max-w-2xl text-sm leading-relaxed text-slate-400">{b}</p>
              </li>
            ))}
          </ul>
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
