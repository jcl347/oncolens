"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import PipelineCanvas, { Stage, StageState } from "@/components/PipelineCanvas";

/* ---------------------------------------------------------------- types */

type Check = {
  ok: boolean; state: string; detail: string; fix?: string | null;
  counts?: Record<string, number>; generated_at?: string; caveats?: number;
};
type Status = {
  ready: boolean;
  auth_configured: boolean;
  checks: Record<string, Check>;
  commands: Record<string, string>;
};

/* ------------------------------------------------------------ primitives */

function StateDot({ ok, busy }: { ok: boolean; busy?: boolean }) {
  return (
    <span
      className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full ${
        busy ? "animate-pulse bg-accent" : ok ? "bg-teal" : "bg-slate-600"
      }`}
    />
  );
}

/** Copy-to-clipboard command block. Long jobs are commands, not buttons — on purpose. */
function Command({ cmd }: { cmd: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={() => {
        navigator.clipboard?.writeText(cmd);
        setCopied(true);
        setTimeout(() => setCopied(false), 1400);
      }}
      className="group flex w-full items-center gap-3 rounded border border-edge bg-black/30 px-3 py-2 text-left font-mono text-[11px] text-slate-400 transition hover:border-teal/40 hover:text-slate-200"
    >
      <span className="select-none text-slate-600">$</span>
      <span className="flex-1 overflow-x-auto whitespace-nowrap">{cmd}</span>
      <span className={`shrink-0 text-[10px] ${copied ? "text-teal" : "text-slate-600 group-hover:text-slate-400"}`}>
        {copied ? "copied" : "copy"}
      </span>
    </button>
  );
}

/** Numbered section — the structural device borrowed from the reference design. */
function Section({ n, title, blurb, children }: {
  n: string; title: string; blurb: string; children: React.ReactNode;
}) {
  return (
    <section className="border-t border-edge py-9">
      <div className="grid gap-7 md:grid-cols-[minmax(0,15rem)_1fr]">
        <div>
          <div className="font-mono text-[11px] text-slate-600">{n}</div>
          <h2 className="mt-1.5 text-[15px] font-medium text-slate-100">{title}</h2>
          <p className="mt-2 max-w-xs text-[12px] leading-relaxed text-slate-500">{blurb}</p>
        </div>
        <div className="min-w-0">{children}</div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ page */

export default function SetupPage() {
  const [status, setStatus] = useState<Status | null>(null);
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [log, setLog] = useState<{ t: string; ok: boolean; msg: string }[]>([]);

  const refresh = useCallback(async () => {
    try {
      const r = await fetch("/api/setup", { cache: "no-store" });
      setStatus(await r.json());
    } catch {
      /* leave prior status visible rather than blanking the page */
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 15000);
    return () => clearInterval(id);
  }, [refresh]);

  const push = (ok: boolean, msg: string) =>
    setLog((l) => [{ t: new Date().toLocaleTimeString(), ok, msg }, ...l].slice(0, 40));

  const run = useCallback(
    async (action: string, qs = "") => {
      if (!token) {
        push(false, "SETUP_TOKEN required — paste it above before running actions.");
        return;
      }
      setBusy(action);
      push(true, `${action} started…`);
      try {
        const r = await fetch(`/api/setup?action=${action}${qs}`, {
          method: "POST",
          headers: { "x-setup-token": token },
        });
        const d = await r.json();
        push(!!d.ok, d.ok ? `${action}: ${d.detail} (${d.elapsed_ms}ms)`
                          : `${action} failed — ${d.detail || d.error}`);
        await refresh();
      } catch (e: any) {
        push(false, `${action} failed — ${e.message}`);
      } finally {
        setBusy(null);
      }
    },
    [token, refresh]
  );

  const stages: Stage[] = useMemo(() => {
    const c = status?.checks;
    const st = (k: string): StageState =>
      busy && k === "corpus" ? "running" : c?.[k]?.ok ? "ok" : "pending";
    return [
      { id: "src", label: "PubMed / PMC", state: "ok" },
      { id: "blob", label: "Blob", state: c ? st("blob") : "pending" },
      { id: "pg", label: "pgvector", state: c ? st("postgres") : "pending" },
      { id: "corpus", label: "Corpus", state: c ? st("corpus") : "pending" },
      { id: "eval", label: "Metrics", state: c ? st("eval_report") : "pending" },
    ];
  }, [status, busy]);

  const c = status?.checks;

  return (
    <main className="min-h-screen bg-ink text-slate-200">
      <div className="mx-auto max-w-4xl px-6 py-16">
        {/* ------------------------------------------------------- header */}
        <header className="pb-9">
          <a href="/" className="font-mono text-[11px] text-slate-600 hover:text-teal">← oncolens</a>
          <h1 className="mt-4 text-2xl font-medium tracking-tight text-white">Setup</h1>
          <p className="mt-2 max-w-lg text-[13px] leading-relaxed text-slate-500">
            Provision storage, ingest real oncology literature, and publish the evaluation
            report the site displays. Steps that fit inside a serverless function are
            buttons; the ones that take minutes to hours are commands, because a button
            that timed out halfway would leave the stores partly populated.
          </p>
        </header>

        {/* --------------------------------------------- live pipeline viz */}
        <div className="rounded-lg border border-edge bg-white/[0.015] px-4 pb-6 pt-3">
          <PipelineCanvas stages={stages} />
          <p className="mt-3 text-center text-[10px] text-slate-600">
            Flow appears only between stages that are actually configured — a gap is a
            missing link, not an animation choice.
          </p>
        </div>

        {/* ------------------------------------------------------ 01 auth */}
        <Section
          n="01"
          title="Authorise"
          blurb="Actions here create tables and spend NCBI quota, so they are gated. Set SETUP_TOKEN in your Vercel project environment, then paste the same value here."
        >
          <input
            type="password"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            placeholder="SETUP_TOKEN"
            className="w-full rounded border border-edge bg-black/30 px-3 py-2 font-mono text-xs text-slate-200 placeholder:text-slate-600 focus:border-teal/50 focus:outline-none"
          />
          <p className="mt-2 text-[11px] text-slate-600">
            {status?.auth_configured
              ? "SETUP_TOKEN is set on the server."
              : "SETUP_TOKEN is not set on the server — mutating actions will be refused."}
            {" "}The token is sent per-request and never stored.
          </p>
        </Section>

        {/* --------------------------------------------------- 02 storage */}
        <Section
          n="02"
          title="Storage"
          blurb="Blob holds article full text; Postgres holds chunks, embeddings and metadata plus each passage's blob URL. Create both in Vercel → Storage, then verify here."
        >
          <div className="space-y-2">
            {[
              { k: "blob", label: "Vercel Blob", action: "test_blob", cta: "Test round-trip" },
              { k: "postgres", label: "Neon Postgres + pgvector", action: "init_schema", cta: "Create schema" },
            ].map((row) => {
              const chk = c?.[row.k];
              return (
                <div key={row.k} className="flex items-center gap-3 rounded border border-edge bg-white/[0.015] px-3 py-2.5">
                  <StateDot ok={!!chk?.ok} busy={busy === row.action} />
                  <div className="min-w-0 flex-1">
                    <div className="text-[13px] text-slate-200">{row.label}</div>
                    <div className="truncate text-[11px] text-slate-500">
                      {chk?.detail ?? "checking…"}
                      {chk?.fix && <span className="text-accent"> · {chk.fix}</span>}
                    </div>
                  </div>
                  <button
                    onClick={() => run(row.action)}
                    disabled={busy !== null}
                    className="shrink-0 rounded border border-teal/40 px-3 py-1.5 text-[11px] text-teal transition hover:bg-teal/10 disabled:opacity-40"
                  >
                    {busy === row.action ? "running…" : row.cta}
                  </button>
                </div>
              );
            })}
          </div>
        </Section>

        {/* ---------------------------------------------------- 03 ingest */}
        <Section
          n="03"
          title="Ingest"
          blurb="Real papers from PubMed with human MeSH labels, plus verbatim full text from the PMC Cloud Service. The licence gate skips anything not cleared for commercial use."
        >
          <div className="flex items-center gap-3 rounded border border-edge bg-white/[0.015] px-3 py-2.5">
            <StateDot ok={!!c?.corpus?.ok} busy={busy === "sample_ingest"} />
            <div className="min-w-0 flex-1">
              <div className="text-[13px] text-slate-200">Sample ingest — 10 papers</div>
              <div className="truncate text-[11px] text-slate-500">
                {c?.corpus?.detail ?? "checking…"}
              </div>
            </div>
            <button
              onClick={() => run("sample_ingest", "&n=10")}
              disabled={busy !== null}
              className="shrink-0 rounded border border-teal/40 px-3 py-1.5 text-[11px] text-teal transition hover:bg-teal/10 disabled:opacity-40"
            >
              {busy === "sample_ingest" ? "ingesting…" : "Run sample"}
            </button>
          </div>

          <p className="mt-4 mb-2 text-[11px] text-slate-500">
            A real corpus is thousands of papers and takes minutes to hours — well past the
            function limit. Run it locally or in CI:
          </p>
          <div className="space-y-1.5">
            {status?.commands?.pull_env && <Command cmd={status.commands.pull_env} />}
            {status?.commands?.full_ingest && <Command cmd={status.commands.full_ingest} />}
          </div>
        </Section>

        {/* ---------------------------------------------------- 04 metrics */}
        <Section
          n="04"
          title="Metrics"
          blurb="The site shows measured retrieval quality, including a raw term-frequency floor and an auto-generated list of what the numbers do not establish. Without this report it declares itself unmeasured."
        >
          <div className="mb-4 flex items-center gap-3 rounded border border-edge bg-white/[0.015] px-3 py-2.5">
            <StateDot ok={!!c?.eval_report?.ok} />
            <div className="min-w-0 flex-1">
              <div className="text-[13px] text-slate-200">Evaluation report</div>
              <div className="truncate text-[11px] text-slate-500">
                {c?.eval_report?.detail ?? "checking…"}
                {c?.eval_report?.caveats ? ` · ${c.eval_report.caveats} caveats` : ""}
              </div>
            </div>
            <a href="/#eval" className="shrink-0 text-[11px] text-slate-500 hover:text-teal">view →</a>
          </div>
          <div className="space-y-1.5">
            {status?.commands?.build_eval_report && <Command cmd={status.commands.build_eval_report} />}
            {status?.commands?.build_artifact && <Command cmd={status.commands.build_artifact} />}
          </div>
        </Section>

        {/* -------------------------------------------------------- 05 log */}
        <Section n="05" title="Activity" blurb="Everything this page has done, newest first. Failures are shown in full rather than summarised.">
          {log.length === 0 ? (
            <p className="text-[12px] text-slate-600">No actions run yet.</p>
          ) : (
            <div className="max-h-64 space-y-1 overflow-y-auto font-mono text-[11px]">
              {log.map((l, i) => (
                <div key={i} className="flex gap-3">
                  <span className="shrink-0 text-slate-600">{l.t}</span>
                  <span className={l.ok ? "text-slate-400" : "text-red-400"}>{l.msg}</span>
                </div>
              ))}
            </div>
          )}
        </Section>

        <footer className="border-t border-edge pt-6 text-[11px] text-slate-600">
          {status?.ready
            ? "All components configured."
            : "Some components are not yet configured — the site will report itself as unmeasured until they are."}
          {" · "}Next.js on Vercel · storage: Vercel Blob + Neon pgvector
        </footer>
      </div>
    </main>
  );
}
