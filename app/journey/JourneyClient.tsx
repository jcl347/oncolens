"use client";

import { Fragment, useEffect, useRef, useState } from "react";
import JourneyCanvas from "../../components/JourneyCanvas";

type Metric = {
  label: string;
  value: number | string;
  unit?: string;
  note?: string;
  provenance: "live" | "artifact" | "recorded";
  command?: string;
  delta?: string | null;
  direction?: string;
};

type Signal = { name: string; body: number; refs: number; auc: number };
type Guard = { hazard: string; guard: string };
type SystemRow = Record<string, string | number>;

type Stage = {
  id: string;
  kicker: string;
  title: string;
  narrative: string;
  visual: string;
  metrics?: Metric[];
  signals?: Signal[];
  guards?: Guard[];
  systems?: SystemRow[];
  caveat?: string | null;
};

type Journey = {
  generated_at: string;
  git_rev: string;
  live_store_reachable: boolean;
  store_error?: string | null;
  stages: Stage[];
};

/** Provenance is shown next to every number, because a reader cannot otherwise tell a
 *  live corpus count from a benchmark result from a figure that was true last week. */
const PROVENANCE: Record<string, { label: string; cls: string; title: string }> = {
  live: {
    label: "live",
    cls: "border-emerald-400/30 bg-emerald-400/10 text-emerald-300",
    title: "Queried from the store when this page was built",
  },
  artifact: {
    label: "benchmark",
    cls: "border-cyan-400/30 bg-cyan-400/10 text-cyan-300",
    title: "Read from a benchmark script's JSON output",
  },
  recorded: {
    label: "measured",
    cls: "border-slate-400/25 bg-slate-400/10 text-slate-300",
    title: "Measured by the named script; re-running costs real time or money",
  },
};

function fmt(v: number | string) {
  if (typeof v === "number") {
    if (Number.isInteger(v) && Math.abs(v) >= 1000) return v.toLocaleString();
    if (!Number.isInteger(v)) return v.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }
  return String(v);
}

function MetricCard({ m }: { m: Metric }) {
  const p = PROVENANCE[m.provenance] ?? PROVENANCE.recorded;
  return (
    <div className="rounded-lg border border-white/8 bg-white/[0.025] p-4 backdrop-blur-sm">
      <div className="flex items-start justify-between gap-3">
        <span className="text-[11px] uppercase tracking-wider text-slate-400">
          {m.label}
        </span>
        <span
          title={p.title}
          className={`shrink-0 rounded border px-1.5 py-px text-[9px] uppercase tracking-wider ${p.cls}`}
        >
          {p.label}
        </span>
      </div>
      <div className="mt-2 flex items-baseline gap-2">
        <span className="font-mono text-2xl tabular-nums text-white">{fmt(m.value)}</span>
        {m.unit ? <span className="text-sm text-slate-400">{m.unit}</span> : null}
        {m.delta ? (
          <span className="rounded bg-emerald-400/10 px-1.5 py-px font-mono text-[11px] text-emerald-300">
            {m.delta}
          </span>
        ) : null}
      </div>
      {m.note ? (
        <p className="mt-1.5 text-xs leading-relaxed text-slate-400">{m.note}</p>
      ) : null}
      {m.command ? (
        <code className="mt-2 block truncate text-[10px] text-slate-500" title={m.command}>
          {m.command}
        </code>
      ) : null}
    </div>
  );
}

/** Signal separation. Bars are log-scaled because the contrast is 100x on some signals
 *  and a linear axis would flatten every other row to invisibility. */
function SignalChart({ signals }: { signals: Signal[] }) {
  const scale = (v: number) => Math.min(1, Math.log10(1 + v * 40) / Math.log10(1 + 25 * 40));
  return (
    <div className="mt-6 space-y-2.5">
      <div className="flex items-center gap-4 text-[10px] uppercase tracking-wider text-slate-500">
        <span className="flex items-center gap-1.5">
          <i className="h-2 w-2 rounded-full bg-sky-400" /> body text
        </span>
        <span className="flex items-center gap-1.5">
          <i className="h-2 w-2 rounded-full bg-rose-400" /> bibliography
        </span>
        <span className="ml-auto">AUC 1.0 = perfect separation, 0.5 = useless</span>
      </div>
      {signals.map((s) => {
        const decisive = s.auc >= 0.95 || s.auc <= 0.05;
        const noise = Math.abs(s.auc - 0.5) < 0.25;
        return (
          <div key={s.name} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3">
            <div>
              <div className="flex items-baseline justify-between">
                <span className={`text-xs ${noise ? "text-slate-500 line-through" : "text-slate-300"}`}>
                  {s.name}
                </span>
                <span className="font-mono text-[10px] text-slate-500">
                  {s.body.toFixed(3)} → {s.refs.toFixed(3)}
                </span>
              </div>
              <div className="mt-1 flex h-1.5 gap-px overflow-hidden rounded-full bg-white/5">
                <div className="bg-sky-400/70" style={{ width: `${scale(s.body) * 100}%` }} />
                <div className="bg-rose-400/70" style={{ width: `${scale(s.refs) * 100}%` }} />
              </div>
            </div>
            <span
              className={`w-24 shrink-0 rounded px-2 py-0.5 text-center font-mono text-[11px] ${
                noise
                  ? "bg-amber-400/10 text-amber-300"
                  : decisive
                    ? "bg-emerald-400/10 text-emerald-300"
                    : "bg-white/5 text-slate-400"
              }`}
            >
              {s.auc.toFixed(3)}
            </span>
          </div>
        );
      })}
      <p className="pt-1 text-xs leading-relaxed text-slate-400">
        Struck-through rows are the two signals that carried hand-assigned weights of 0.20
        and 0.12 and turned out to be noise. The signals were never the problem.
      </p>
    </div>
  );
}

function SystemTable({ systems }: { systems: SystemRow[] }) {
  if (!systems?.length) return null;
  const cols = Object.keys(systems[0]).filter((k) => k !== "system");
  const best: Record<string, number> = {};
  cols.forEach((c) => {
    best[c] = Math.max(...systems.map((s) => Number(s[c]) || 0));
  });
  return (
    <div className="mt-6 overflow-x-auto">
      <table className="w-full min-w-[520px] text-left text-xs">
        <thead>
          <tr className="border-b border-white/10 text-[10px] uppercase tracking-wider text-slate-500">
            <th className="py-2 pr-4 font-normal">system</th>
            {cols.map((c) => (
              <th key={c} className="py-2 pr-4 text-right font-normal">{c}</th>
            ))}
          </tr>
        </thead>
        <tbody className="font-mono tabular-nums">
          {systems.map((s) => (
            <tr key={String(s.system)} className="border-b border-white/5">
              <td className="py-2 pr-4 font-sans text-slate-300">{String(s.system)}</td>
              {cols.map((c) => {
                const v = Number(s[c]);
                const isBest = v === best[c] && c !== "unjudged@10";
                return (
                  <td
                    key={c}
                    className={`py-2 pr-4 text-right ${isBest ? "text-emerald-300" : "text-slate-400"}`}
                  >
                    {Number.isFinite(v) ? v.toFixed(4) : "—"}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function JourneyClient({ data }: { data: Journey }) {
  const [active, setActive] = useState(0);
  const sectionRefs = useRef<(HTMLElement | null)[]>([]);

  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            const i = sectionRefs.current.indexOf(e.target as HTMLElement);
            if (i >= 0) setActive(i);
          }
        });
      },
      { rootMargin: "-45% 0px -45% 0px", threshold: 0 },
    );
    sectionRefs.current.forEach((el) => el && obs.observe(el));
    return () => obs.disconnect();
  }, []);

  return (
    <div className="relative min-h-screen bg-[#05070c] text-slate-200">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(ellipse_at_50%_0%,rgba(56,189,248,0.10),transparent_60%)]" />
      <JourneyCanvas stage={active} />

      {/* Progress rail. Also a table of contents — a long scroll page without one is a
          maze, and the reader should be able to jump to the measurement they care about. */}
      <nav
        aria-label="Journey stages"
        className="fixed left-4 top-1/2 z-20 hidden -translate-y-1/2 flex-col gap-2 md:flex"
      >
        {data.stages.map((s, i) => (
          <a
            key={s.id}
            href={`#${s.id}`}
            title={s.title}
            className="group flex items-center gap-2"
          >
            <span
              className={`h-px transition-all duration-300 ${
                i === active ? "w-7 bg-cyan-300" : "w-3 bg-white/25 group-hover:w-5"
              }`}
            />
            <span
              className={`text-[10px] uppercase tracking-wider transition-opacity ${
                i === active ? "text-cyan-300 opacity-100" : "text-slate-500 opacity-0 group-hover:opacity-100"
              }`}
            >
              {s.kicker}
            </span>
          </a>
        ))}
      </nav>

      <main className="relative z-10 mx-auto max-w-3xl px-6 pb-40 md:pl-28">
        <header className="flex min-h-[80vh] flex-col justify-center py-24">
          <p className="text-[11px] uppercase tracking-[0.2em] text-cyan-300/80">
            OncoLens — technical journey
          </p>
          <h1 className="mt-5 text-4xl font-medium leading-tight text-white sm:text-5xl">
            Every number here has a command that produces it.
          </h1>
          <p className="mt-6 max-w-xl text-[15px] leading-relaxed text-slate-400">
            This page is the record of what was tried, what failed, and the measurement
            that decided each choice — including the times the benchmark said a component
            was fine and the live corpus disagreed.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-x-5 gap-y-2 font-mono text-[11px] text-slate-500">
            <span>generated {data.generated_at}</span>
            <span>rev {data.git_rev}</span>
            <span
              className={
                data.live_store_reachable ? "text-emerald-400/80" : "text-amber-400/80"
              }
            >
              {data.live_store_reachable
                ? "store reachable — live metrics current"
                : `store unreachable — live metrics omitted${data.store_error ? ` (${data.store_error})` : ""}`}
            </span>
          </div>
        </header>

        {data.stages.map((s, i) => (
          <section
            key={s.id}
            id={s.id}
            ref={(el) => {
              sectionRefs.current[i] = el;
            }}
            className="min-h-[85vh] scroll-mt-24 border-t border-white/8 py-24"
          >
            <p className="text-[11px] uppercase tracking-[0.18em] text-cyan-300/70">
              {String(i + 1).padStart(2, "0")} — {s.kicker}
            </p>
            <h2 className="mt-4 text-2xl font-medium leading-snug text-white sm:text-3xl">
              {s.title}
            </h2>
            <p className="mt-5 max-w-2xl text-[15px] leading-relaxed text-slate-400">
              {s.narrative}
            </p>

            {s.metrics?.length ? (
              <div className="mt-8 grid gap-3 sm:grid-cols-2">
                {s.metrics.map((m) => (
                  <MetricCard key={m.label} m={m} />
                ))}
              </div>
            ) : null}

            {s.signals?.length ? <SignalChart signals={s.signals} /> : null}

            {s.guards?.length ? (
              <div className="mt-8 overflow-hidden rounded-lg border border-white/8">
                <div className="grid grid-cols-[1fr_1.3fr] gap-px bg-white/8 text-xs">
                  <div className="bg-[#080b12] px-4 py-2 text-[10px] uppercase tracking-wider text-slate-500">
                    validity hazard
                  </div>
                  <div className="bg-[#080b12] px-4 py-2 text-[10px] uppercase tracking-wider text-slate-500">
                    guard in code
                  </div>
                  {s.guards.map((g) => (
                    // Fragment needs an explicit key; the shorthand <> cannot take one,
                    // and a grid needs the two cells as direct siblings, not wrapped.
                    <Fragment key={g.hazard}>
                      <div className="bg-[#05070c] px-4 py-3 text-slate-400">{g.hazard}</div>
                      <div className="bg-[#05070c] px-4 py-3 text-slate-300">{g.guard}</div>
                    </Fragment>
                  ))}
                </div>
              </div>
            ) : null}

            {s.systems?.length ? <SystemTable systems={s.systems} /> : null}

            {s.caveat ? (
              <p className="mt-8 border-l-2 border-amber-400/40 bg-amber-400/[0.04] py-3 pl-4 pr-4 text-sm leading-relaxed text-amber-200/70">
                {s.caveat}
              </p>
            ) : null}
          </section>
        ))}

        <footer className="border-t border-white/8 py-16 text-sm leading-relaxed text-slate-500">
          <p>
            Nothing on this page is hand-entered.{" "}
            <code className="text-slate-400">scripts/build_journey_data.py</code> queries
            the store and reads each benchmark&apos;s JSON output, and the page renders
            whatever it finds — including nothing, when a benchmark has not been run against
            the current index.
          </p>
          <p className="mt-4">
            <a href="/" className="text-cyan-300/80 underline-offset-4 hover:underline">
              ← back to search
            </a>
          </p>
        </footer>
      </main>
    </div>
  );
}
