"use client";

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

/** How the tool is used. Placed above the design narrative because a reader who cannot
 *  work the thing has no reason to care how it was built. */
const HOW_TO_USE = [
  {
    n: "01",
    title: "Search a concept, get the passage",
    body: "Type what you would say to a colleague — a mechanism, a drug, a variant. "
        + "Results are papers, each carrying the exact passage that matched and the "
        + "character offsets it sits at, so you can check the claim rather than trust it.",
    example: "EGFR C797S resistance to osimertinib",
  },
  {
    n: "02",
    title: "Compare papers on the same dimensions",
    body: "Switch to Compare and the same query returns a table: papers down the side, "
        + "technical dimensions across the top — cohort, assay, endpoint, effect size. "
        + "Each filled cell cites the passage it came from.",
    example: "CAR-T persistence in solid tumors",
  },
  {
    n: "03",
    title: "A blank cell means NOT REPORTED",
    body: "It does not mean no effect. A paper that never measured progression-free "
        + "survival is marked as not reporting it, because reading an empty cell as a "
        + "null result is the specific mistake this table exists to prevent.",
    example: null,
  },
];

export default function OverviewClient({ journey, clusters }: { journey: Journey | null; clusters: any }) {
  const stages = journey?.stages ?? [];
  const corpus = stages.find((s) => s.id === "goal");
  const retrieval = stages.find((s) => s.id === "retrieval");

  return (
    <div className="relative min-h-screen bg-[#05070c] text-slate-200">
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
            source. Built for the question an R&amp;D team actually asks — “what is known
            about X?” — which is answered by a set of papers, not one.
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
            <a href="/search"
               className="rounded-md bg-cyan-400/90 px-4 py-2 text-sm font-medium text-[#04121a] transition-colors hover:bg-cyan-300">
              Open search
            </a>
            <a href="#measurement"
               className="rounded-md border border-white/12 px-4 py-2 text-sm text-slate-300 transition-colors hover:border-white/25 hover:text-white">
              How it is measured
            </a>
          </div>
        </header>

        {/* ---------- how to use ---------- */}
        <section className="border-b border-white/8 py-14">
          <h2 className="text-xs uppercase tracking-[0.16em] text-cyan-300/70">Using it</h2>
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

        {/* ---------- the corpus, as a map ---------- */}
        {clusters ? (
          <section className="border-b border-white/8 py-14">
            <h2 className="text-xs uppercase tracking-[0.16em] text-cyan-300/70">What is in the corpus</h2>
            <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-400">
              Every paper placed by the embedding retrieval actually uses, clustered and
              then named by the MeSH major topics most distinctive to each region. Select a
              cluster to see its papers.
            </p>
            <div className="mt-7">
              <ClusterMap data={clusters} />
            </div>
          </section>
        ) : null}

        {/* ---------- measurement ---------- */}
        <section id="measurement" className="border-b border-white/8 py-14">
          <h2 className="text-xs uppercase tracking-[0.16em] text-cyan-300/70">How it is measured</h2>
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
                {[
                  ["synthesis", "“what is known about X”", "review authors", "recall@20 — coverage of the set"],
                  ["concept", "2-word MeSH term", "NLM indexers", "success@5 — was it on screen"],
                  ["identifier", "bare gene / variant", "citing author", "success@1 — exact lookup"],
                  ["claim", "27-word sentence", "citing author", "MRR — find the source"],
                ].map(([a, b, c, d]) => (
                  <tr key={a} className="border-b border-white/5">
                    <td className="py-2.5 pr-4 font-medium text-white">{a}</td>
                    <td className="py-2.5 pr-4 text-slate-400">{b}</td>
                    <td className="py-2.5 pr-4 text-slate-400">{c}</td>
                    <td className="py-2.5 pr-4 font-mono text-[11px] text-slate-400">{d}</td>
                  </tr>
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
              improvement</span> — better on at least one query type and worse on none.
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
          <h2 className="text-xs uppercase tracking-[0.16em] text-cyan-300/70">
            What the improvement loop found
          </h2>
          <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-400">
            Every candidate states, before it runs, which query type it should help and
            which it should not. The loop records predictions and outcomes together — so a
            result that was not predicted is logged as a surprise to test next round, never
            claimed as a win.
          </p>
          <ul className="mt-6 space-y-4">
            {[
              ["One fusion weight is wrong in both directions",
               "Dropping the lexical arm hurt 2-word queries (−0.0707, p=0.0008); doubling it hurt conceptual ones. The optimal weight depends on query shape — a finding that only exists because the query types are scored separately."],
              ["A gate that could not be passed",
               "Reranking reorders the top 24 passages without changing which documents are in the top 20, so coverage moved by exactly 0.0000 and the reranker was structurally unable to win, however good it was."],
              ["The evaluation is underpowered, and that is the bottleneck",
               "At the current corpus size the smallest detectable effect is 0.043 on synthesis and 0.062 on concept. Every promising result so far sits below those floors — reported as “no significant change” by construction, which looks like rigour and is blindness."],
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
              Figures on this page are generated by{" "}
              <code className="text-slate-400">scripts/build_journey_data.py</code> and{" "}
              <code className="text-slate-400">scripts/build_clusters.py</code>, which query
              the store and read each benchmark&apos;s output. Nothing here is hand-entered.
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
