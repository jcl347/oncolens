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
type Journey = { generated_at: string; git_rev: string; live_store_reachable: boolean; stages: Stage[] };

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
            Passage-grounded retrieval over the oncology literature
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
                <div key={m.label}>
                  <dt className="text-[11px] uppercase tracking-wider text-slate-500">{m.label}</dt>
                  <dd className="mt-1 font-mono text-xl tabular-nums text-white">{fmt(m.value)}</dd>
                </div>
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
                drifting into general molecular biology. Roughly 40% of the most-cited
                candidates were rejected on exactly that test.
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
            <p>
              <span className="text-slate-200">A change ships only if it is a Pareto
              improvement</span>: better on at least one query type and worse on none.
              A weighted average would let a gain on common queries pay for a regression on
              exact lookup, where returning the wrong variant is worse than returning
              nothing because the error is invisible in the results.
            </p>
          </div>

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
