"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import WebGLAccent from "@/components/WebGLAccent";
import WebGLBackground from "@/components/WebGLBackground";

/* ---------------------------------------------------------------- types */

type Span = { start: number; end: number; text: string; is_literal: boolean };
type Clause = { text: string; start: number; end: number; score: number; matched_terms: string[]; spans: Span[] };
type Passage = {
  chunk_id: string; section: string; start_char: number; end_char: number; text: string;
  clauses?: Clause[]; best_clause?: Clause | null;
};
type Result = {
  doc_id: string; title: string; doc_type: string; year: number | null;
  meta: Record<string, any>; score: number; passage: Passage;
  /** Which arm retrieved it. "semantic" means the query's words are NOT in the passage. */
  matched_by?: "lexical+semantic" | "lexical" | "semantic";
};
/**
 * These mirror `oncolens.serve.live_query.compare` EXACTLY.
 *
 * They previously did not, and the whole comparison view was dead: `aspects` was typed as
 * `string[]` when the API sends `{key,label,numeric}` objects (React throws "Objects are
 * not valid as a React child" on render), and `cells` was typed as a nested
 * `Record<doc, Record<aspect, Cell>>` when the API sends ONE flat map keyed `doc|aspect`,
 * so every lookup returned undefined and every cell rendered "not reported". A response
 * shape is an interface; drifting from it silently produces a UI that is confidently wrong.
 */
type Aspect = { key: string; label: string; numeric: boolean };
type Cell = {
  reported: boolean; chunk_id?: string; section?: string;
  start_char?: number; end_char?: number; text?: string | null; score?: number;
  /** Why an unfilled cell is unfilled: no cue word present, or a near miss on threshold. */
  reason?: "no_cue_match" | "below_threshold"; threshold?: number;
};
type Comparison = {
  query: string; aspects: Aspect[]; doc_ids: string[];
  titles: Record<string, string>; years: Record<string, number | null>;
  coverage: number; notes: string[]; cells: Record<string, Cell>;
};
type EvalReport = any;

/* ------------------------------------------------------- highlighting */

/**
 * Render a clause with its matched terms highlighted.
 *
 * The spans carry section-absolute offsets, so they are rebased onto the clause before
 * slicing. Getting this wrong produces highlights that drift a few characters — which
 * looks like a rendering glitch but is actually a provenance bug, so it is done once here
 * rather than in each caller.
 */
function Highlighted({ clause }: { clause: Clause }) {
  const parts: React.ReactNode[] = [];
  let cursor = 0;
  const rel = clause.spans
    .map((s) => ({ ...s, s: s.start - clause.start, e: s.end - clause.start }))
    .filter((s) => s.s >= 0 && s.e <= clause.text.length)
    .sort((a, b) => a.s - b.s);

  rel.forEach((s, i) => {
    if (s.s > cursor) parts.push(<span key={`t${i}`}>{clause.text.slice(cursor, s.s)}</span>);
    parts.push(
      <mark
        key={`m${i}`}
        className={
          s.is_literal
            ? "rounded bg-accent/25 px-0.5 font-mono text-accent ring-1 ring-accent/40"
            : "rounded bg-teal/20 px-0.5 text-teal"
        }
        title={s.is_literal ? "exact identifier match" : "term match"}
      >
        {clause.text.slice(s.s, s.e)}
      </mark>
    );
    cursor = s.e;
  });
  if (cursor < clause.text.length) parts.push(<span key="tail">{clause.text.slice(cursor)}</span>);
  return <>{parts}</>;
}

/* --------------------------------------------------------------- page */

export default function Page() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"search" | "compare">("search");
  const [results, setResults] = useState<Result[]>([]);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [openDoc, setOpenDoc] = useState<Result | null>(null);
  const [evalReport, setEvalReport] = useState<EvalReport | null>(null);
  const [showEval, setShowEval] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [runHover, setRunHover] = useState(false);
  /** Set when a query arrives from the URL, cleared once it has been dispatched. */
  const [pending, setPending] = useState<string | null>(null);
  /** Monotonic request id: only the newest in-flight request may write state. */
  const seqRef = useRef(0);
  /** The query the visible results actually answer, which can lag the input box. */
  const [ranQuery, setRanQuery] = useState("");
  /** Server-side notes, e.g. "no passage contains your search terms". */
  const [noLexical, setNoLexical] = useState<string[]>([]);

  useEffect(() => {
    fetch("/api/evaluate").then((r) => r.json()).then(setEvalReport).catch(() => {});
  }, []);

  /**
   * Accept `?q=` and `?mode=` from the URL and run immediately.
   *
   * Without this every link from the Overview landed on an empty search box: the page
   * advertised "try it" against a specific query, navigated, and then asked the reader to
   * type it themselves. The links were not broken, the receiving end simply ignored them.
   *
   * Read from `window.location` rather than `useSearchParams` so the component does not
   * need a Suspense boundary, which is what that hook forces on a statically rendered
   * route.
   */
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const q = params.get("q");
    if (!q) return;
    const m = params.get("mode");
    if (m === "compare" || m === "search") setMode(m);
    setQuery(q);
    setPending(q);
  }, []);

  // Run once state has actually landed; calling run() straight from the effect above
  // would close over the empty initial query.
  useEffect(() => {
    if (pending === null) return;
    setPending(null);
    void run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pending]);

  const run = useCallback(async () => {
    if (!query.trim()) return;
    // SEQUENCE THE REQUESTS. Without this, typing "EGFR", hitting Enter, refining to
    // "EGFR C797S" and hitting Enter again leaves whichever response lands LAST on
    // screen, and nothing on the page names the query the list came from. For a tool
    // whose output gets cited, showing passages that answer a question the user already
    // replaced is the worst failure available.
    const mine = ++seqRef.current;
    const forMode = mode;
    setLoading(true); setError(null); setComparison(null); setResults([]);
    setNoLexical([]);
    setRanQuery(query);
    try {
      const url = forMode === "search"
        ? `/api/search?q=${encodeURIComponent(query)}&k=10`
        : `/api/compare?q=${encodeURIComponent(query)}&n=5`;
      const r = await fetch(url);
      // Read as TEXT first. A gateway timeout or a cold-start failure returns an HTML
      // error page, and calling r.json() on it throws "Unexpected token '<'", which reads
      // to a researcher as a browser bug and hides the one useful fact: retrying against
      // a warm container usually works.
      const raw = await r.text();
      let data: any = null;
      try { data = JSON.parse(raw); } catch { /* not JSON: handled below */ }
      if (!r.ok || data === null) {
        throw new Error(
          data?.error
            ?? (r.status === 504 || r.status === 502
                ? "The server took too long to answer. This usually clears on a retry: the database sleeps when idle and the first query wakes it."
                : `The server returned ${r.status} and not a result. Try again in a moment.`)
        );
      }
      if (mine !== seqRef.current) return;   // a newer query superseded this one
      if (forMode === "search") {
        setResults(data.results || []);
        setNoLexical(data.lexical_match === false ? (data.notes || []) : []);
      } else {
        setComparison(data);
      }
      // WRITE THE SEARCH TO THE URL. The read side already accepted ?q= and ?mode=, so
      // inbound links worked and outbound ones did not exist: a search could not be
      // bookmarked, shared, or survive a refresh, and the core loop (open result 3, come
      // back, open result 4) cost a retyped and re-run query every time. replaceState
      // rather than pushState so refining a query does not bury the previous page under
      // a dozen history entries; the back button should leave the search, not step
      // through its drafts.
      try {
        const qs = `?q=${encodeURIComponent(query)}${forMode === "compare" ? "&mode=compare" : ""}`;
        window.history.replaceState(null, "", `/search${qs}`);
      } catch { /* history is unavailable in some embedded contexts; not worth failing */ }
    } catch (e: any) {
      if (mine !== seqRef.current) return;
      setError(e.message || "request failed");
    } finally {
      if (mine === seqRef.current) setLoading(false);
    }
  }, [query, mode]);

  const floorGap = useMemo(() => {
    if (!evalReport?.metrics || !evalReport?.floor) return null;
    const p = evalReport.metrics[evalReport.primary_metric];
    const f = evalReport.floor.raw_term_frequency;
    return p != null && f != null ? p - f : null;
  }, [evalReport]);

  return (
    <main className="relative min-h-screen text-slate-200">
      <WebGLBackground />

      <div className="mx-auto max-w-6xl px-6 py-14">
        <header className="mb-10">
          <h1 className="text-4xl font-semibold tracking-tight text-white">
            Onco<span className="text-teal">Lens</span>
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">
            Retrieval over oncology papers and grants that returns{" "}
            <span className="text-slate-200">the passage where a concept was mentioned</span>,
            and sets papers side by side on the same technical dimensions.
          </p>
        </header>

        {/* ---------------------------------------------------- query bar */}
        <div className="rounded-xl border border-edge bg-surface p-4 backdrop-blur-md">
          <div className="mb-3 flex gap-1.5">
            {(["search", "compare"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`rounded-md px-3 py-1.5 text-xs font-medium transition ${
                  mode === m ? "bg-teal/20 text-teal ring-1 ring-teal/40" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {m === "search" ? "Find passages" : "Compare papers"}
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && run()}
              placeholder={
                mode === "search"
                  ? "EGFR C797S resistance…"
                  : "how do these studies measure ctDNA clearance, and in what cohorts?"
              }
              className="flex-1 rounded-lg border border-edge bg-ink/60 px-4 py-3 text-sm text-slate-100 placeholder:text-slate-500 focus:border-teal/50 focus:outline-none"
            />
            <button
              onClick={run}
              disabled={loading}
              onMouseEnter={() => setRunHover(true)}
              onMouseLeave={() => setRunHover(false)}
              onFocus={() => setRunHover(true)}
              onBlur={() => setRunHover(false)}
              className="relative overflow-hidden rounded-lg bg-teal px-5 py-3 text-sm font-medium text-ink transition hover:bg-teal/85 disabled:opacity-50"
            >
              <WebGLAccent
                variant="glow"
                hover={runHover && !loading}
                color={[1, 1, 1]}
                className="absolute inset-0 h-full w-full"
              />
              <span className="relative">
                {loading ? "…" : mode === "search" ? "Search" : "Compare"}
              </span>
            </button>
          </div>
          {error && <p className="mt-3 text-xs text-accent">{error}</p>}
        </div>

        {/* ------------------------------------------- transparency strip
            The headline score is shown ONLY when the server confirms the report describes
            the index being served. It previously rendered ndcg@10 from the 140-document
            synthetic fixture directly above real results, which is the exact thing
            fixtures/README.md and CLAUDE.md forbid: a number machine-generated data
            produced, presented as this system's measured quality. On a mismatch the strip
            says so instead of quoting it. */}
        {evalReport?.metrics && evalReport.applies_to_served_index !== false && (
          <button
            onClick={() => setShowEval((v) => !v)}
            className="mt-4 flex w-full items-center gap-4 rounded-lg border border-edge bg-surface/70 px-4 py-2.5 text-left text-xs backdrop-blur-md transition hover:border-teal/30"
          >
            <span className="font-mono text-teal">
              {evalReport.primary_metric} {evalReport.metrics[evalReport.primary_metric]?.toFixed(3)}
            </span>
            <span className="text-slate-500">
              floor {evalReport.floor?.raw_term_frequency?.toFixed(3)} · ceiling {evalReport.ceiling}
            </span>
            {floorGap != null && (
              <span className={floorGap < 0.05 ? "text-accent" : "text-slate-400"}>
                {floorGap > 0 ? "+" : ""}{floorGap.toFixed(3)} over a no-IDF baseline
              </span>
            )}
            <span className="ml-auto text-slate-500">{showEval ? "hide" : "how is this measured?"}</span>
          </button>
        )}
        {evalReport?.applies_to_served_index === false && (
          <p className="mt-4 rounded-lg border border-amber-400/25 bg-amber-400/[0.04] px-4 py-2.5 text-xs leading-relaxed text-amber-200/70">
            {evalReport.applicability_note} See{" "}
            <a href="/#measurement" className="underline hover:text-amber-100">
              how this index is measured
            </a>{" "}
            for figures that do describe it.
          </p>
        )}

        {showEval && evalReport && <EvalPanel report={evalReport} />}

        {/* --------------------------------------------------- results */}
        {results.length > 0 && (
          <section className="mt-8 space-y-3">
            {/* Name the query these results answer. The input box can already hold
                something else by the time they land, and a passage read against the wrong
                question is the failure this product can least afford. */}
            <p className="text-xs text-slate-500">
              {results.length} passages for{" "}
              <span className="text-slate-300">&ldquo;{ranQuery}&rdquo;</span>
            </p>
            {/* The dense arm has no relevance threshold, so a query whose words appear
                nowhere still returns ten ranked results. Saying nothing here lets a
                misspelling read as coverage. */}
            {noLexical.map((n, i) => (
              <p key={i} className="rounded border border-amber-400/25 bg-amber-400/[0.04] px-3 py-2 text-xs leading-relaxed text-amber-200/80">
                {n}
              </p>
            ))}
            {results.map((r, i) => (
              <ResultCard key={r.doc_id} rank={i + 1} result={r} onOpen={() => setOpenDoc(r)} />
            ))}
          </section>
        )}

        {comparison && <ComparisonGrid data={comparison} />}
      </div>

      {openDoc && <PaperViewer result={openDoc} onClose={() => setOpenDoc(null)} />}
    </main>
  );
}

/* ------------------------------------------------------------ result */

function ResultCard({ rank, result, onOpen }: { rank: number; result: Result; onOpen: () => void }) {
  const clause = result.passage.best_clause;
  return (
    <article className="rounded-xl border border-edge bg-surface p-5 backdrop-blur-md transition hover:border-teal/30">
      {/* The TITLE is the primary object on this card. A researcher scans a result list by
          title, not by rank or score, so it gets the type scale and the score is demoted to
          a quiet monospace tag rather than competing with it at the same weight. */}
      <div className="flex items-start gap-3">
        <span className="mt-1 font-mono text-xs text-slate-600">{String(rank).padStart(2, "0")}</span>
        <h3 className="flex-1 text-[17px] font-semibold leading-snug tracking-[-0.01em] text-white">
          {result.title}
        </h3>
        {/* HOW it matched, not a bare score. The raw RRF sum spans about 0.008 to 0.033,
            so adjacent ranks tie at three decimals and the badge said strictly less than
            the rank printed beside it. Which arm found the passage is the thing a reader
            can actually act on: "semantic" means their words are not in this text. */}
        <span
          className={`mt-0.5 shrink-0 rounded px-1.5 py-0.5 text-[10px] ${
            result.matched_by === "semantic"
              ? "bg-amber-400/10 text-amber-300/80"
              : "bg-teal/10 text-teal/80"
          }`}
          title={
            result.matched_by === "semantic"
              ? "Your search terms do not appear in this passage. It was retrieved by meaning alone."
              : result.matched_by === "lexical"
                ? "Contains your search terms. The embedding did not rank it."
                : "Contains your search terms and ranks highly by meaning."
          }
        >
          {result.matched_by === "semantic" ? "meaning only"
            : result.matched_by === "lexical" ? "terms"
            : "terms + meaning"}
        </span>
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-2 pl-8 text-[11px] text-slate-500">
        <span className="rounded bg-white/5 px-1.5 py-0.5">{result.doc_type}</span>
        {result.year && <span>{result.year}</span>}
        {result.meta?.journal && <span className="italic text-slate-400">{result.meta.journal}</span>}
        {/* Derive from doc_id when meta is absent. The server now always sends
            meta.pmid, but a card that silently drops its only outbound citation link
            because one field went missing is not a failure mode worth keeping. */}
        {(result.meta?.pmid || result.doc_id?.startsWith("PAPER:PMID")) && (
          <a
            href={`https://pubmed.ncbi.nlm.nih.gov/${
              result.meta?.pmid ?? result.doc_id.replace(/^PAPER:PMID/, "")}/`}
            target="_blank" rel="noreferrer"
            className="text-teal hover:underline"
          >
            PMID {result.meta?.pmid ?? result.doc_id.replace(/^PAPER:PMID/, "")}
          </a>
        )}
      </div>

      {/* The clause is the product: not the document, not the paragraph. */}
      <blockquote className="mt-3 border-l-2 border-teal/40 pl-4 text-sm leading-relaxed text-slate-300">
        {clause ? <Highlighted clause={clause} /> : result.passage.text.slice(0, 320)}
      </blockquote>

      <div className="mt-3 flex items-center gap-3 pl-4 text-[11px] text-slate-500">
        <span className="font-mono">
          {result.passage.section} · chars {clause?.start ?? result.passage.start_char}–
          {clause?.end ?? result.passage.end_char}
        </span>
        <button onClick={onOpen} className="text-teal hover:underline">
          quick look
        </button>
        {/* Deep link: the reading page scrolls to this passage rather than the top. */}
        <a
          href={`/paper?id=${encodeURIComponent(result.doc_id)}&highlight=${encodeURIComponent(
            (clause?.text ?? result.passage.text).slice(0, 200))}&from=${encodeURIComponent(
            typeof window !== "undefined" ? window.location.search.replace(/^\?/, "") : "")}`}
          className="text-teal hover:underline"
        >
          read the paper →
        </a>
      </div>
    </article>
  );
}

/* ------------------------------------------------------ paper viewer */

/** Full passage with the matched clause highlighted and scrolled into view. */
function PaperViewer({ result, onClose }: { result: Result; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  const { passage } = result;
  const clause = passage.best_clause;
  const rel = clause ? { s: clause.start - passage.start_char, e: clause.end - passage.start_char } : null;
  // meta.pmid is authoritative; fall back to the doc_id, which encodes it.
  const pmid = result.meta?.pmid || result.doc_id.replace(/^PAPER:PMID/, "");

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-ink/80 p-6 backdrop-blur-sm" onClick={onClose}>
      <div
        className="mt-10 w-full max-w-3xl rounded-xl border border-edge bg-ink/95 p-7 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-4">
          <div className="flex-1">
            <h2 className="text-lg font-medium leading-snug text-white">{result.title}</h2>
            {/* Was printing the raw internal doc_id ("PAPER:PMID39198425"), which is not
                an identifier a researcher can act on. The PMID is the one they recognise,
                so it links out to PubMed, and the internal reading view is one click away
                rather than something to reconstruct by hand. */}
            <p className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-slate-500">
              {pmid && (
                <a
                  href={`https://pubmed.ncbi.nlm.nih.gov/${pmid}/`}
                  target="_blank" rel="noreferrer"
                  className="font-mono text-teal hover:underline"
                >
                  PMID {pmid}
                </a>
              )}
              <span>·</span>
              <span>{passage.section}</span>
              <span>·</span>
              <a
                href={`/paper?id=${encodeURIComponent(result.doc_id)}&highlight=${encodeURIComponent(
                  (clause?.text ?? passage.text).slice(0, 200))}`}
                className="text-teal hover:underline"
              >
                read the paper
              </a>
              {result.meta?.blob_url && (
                <>
                  <span>·</span>
                  <a href={result.meta.blob_url} target="_blank" rel="noreferrer" className="text-teal hover:underline">
                    source text
                  </a>
                </>
              )}
            </p>
          </div>
          <button onClick={onClose} className="rounded px-2 py-1 text-slate-500 hover:text-slate-200">✕</button>
        </div>

        <div className="mt-5 max-h-[60vh] overflow-y-auto rounded-lg bg-black/30 p-5 text-sm leading-7 text-slate-300">
          {rel && rel.s >= 0 ? (
            <>
              {passage.text.slice(0, rel.s)}
              <span
                ref={(el) => el?.scrollIntoView({ block: "center", behavior: "smooth" })}
                className="rounded bg-teal/15 px-1 ring-1 ring-teal/40"
              >
                {clause && <Highlighted clause={clause} />}
              </span>
              {passage.text.slice(rel.e)}
            </>
          ) : (
            passage.text
          )}
        </div>

        {clause && (
          <p className="mt-3 text-[11px] text-slate-500">
            Matched on{" "}
            <span className="font-mono text-teal">{clause.matched_terms.join(", ")}</span> · characters{" "}
            {clause.start}–{clause.end} of the {passage.section} section
          </p>
        )}
      </div>
    </div>
  );
}

/* --------------------------------------------------- comparison grid */

/** A comparison cell blown up to its full passage, with provenance. */
function CellViewer({
  cell, title, aspect, docId, onClose,
}: { cell: Cell; title: string; aspect: Aspect; docId: string; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  const pmid = docId.replace(/^PAPER:PMID/, "");
  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-ink/80 p-6 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="mt-16 w-full max-w-2xl rounded-xl border border-edge bg-ink/95 p-6 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-4">
          <div className="min-w-0 flex-1">
            <span className="rounded bg-teal/15 px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-teal">
              {aspect.label}
            </span>
            <h3 className="mt-2 text-[15px] font-semibold leading-snug text-white">{title}</h3>
            <p className="mt-1 text-[11px] text-slate-500">
              <a
                href={`https://pubmed.ncbi.nlm.nih.gov/${pmid}/`}
                target="_blank" rel="noreferrer"
                className="font-mono text-teal hover:underline"
              >
                PMID {pmid}
              </a>
              {cell.section ? ` · ${cell.section}` : ""}
              {cell.start_char != null ? ` · chars ${cell.start_char}–${cell.end_char}` : ""}
            </p>
          </div>
          <button onClick={onClose} className="rounded px-2 py-1 text-slate-500 hover:text-slate-200">
            ✕
          </button>
        </div>

        <div className="mt-4 max-h-[55vh] overflow-y-auto rounded-lg bg-black/30 p-4 text-sm leading-7 text-slate-300">
          {cell.text}
        </div>

        <div className="mt-3 flex items-center justify-between gap-4">
          <p className="text-[11px] leading-relaxed text-slate-500">
            The passage the cell was filled from, verbatim, at the offsets above. It was
            selected because it matched this dimension&apos;s cue vocabulary
            {cell.score != null && (
              <> (retrieval score <span className="font-mono text-slate-400">{cell.score}</span>)</>
            )}
            , which is not the same as the paper answering the question. The cell is a
            claim about this paper only if this passage supports it.
          </p>
          <a
            href={`/paper?id=${encodeURIComponent(docId)}&highlight=${encodeURIComponent(
              (cell.text || "").slice(0, 200))}`}
            className="shrink-0 rounded border border-teal/30 px-2.5 py-1 text-[11px] text-teal transition-colors hover:bg-teal/10"
          >
            read the paper →
          </a>
        </div>
      </div>
    </div>
  );
}

function ComparisonGrid({ data }: { data: Comparison }) {
  const pmidOf = (doc: string) => doc.replace(/^PAPER:PMID/, "");
  const [open, setOpen] = useState<{ cell: Cell; aspect: Aspect; doc: string } | null>(null);
  return (
    <section className="mt-8 rounded-xl border border-edge bg-surface p-5 backdrop-blur-md">
      <div className="mb-4 flex items-baseline gap-3">
        <h2 className="text-sm font-medium text-slate-200">Technical comparison</h2>
        <span className="text-xs text-slate-500">
          {data.doc_ids.length} papers × {data.aspects.length} dimensions ·{" "}
          {(data.coverage * 100).toFixed(0)}% of cells evidenced
        </span>
      </div>

      {data.notes.map((n, i) => (
        <p key={i} className="mb-3 rounded border border-accent/30 bg-accent/10 px-3 py-2 text-xs text-accent">{n}</p>
      ))}

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-xs">
          <thead>
            <tr className="border-b border-edge">
              <th className="w-[22rem] p-2 text-left font-medium text-slate-400">Paper</th>
              {data.aspects.map((a) => (
                <th key={a.key} className="p-2 text-left font-medium text-slate-400">
                  {a.label}
                  {a.numeric && (
                    <span className="ml-1 font-normal text-slate-600" title="requires a number in the passage">
                      #
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.doc_ids.map((doc) => (
              <tr key={doc} className="border-b border-edge/50 align-top">
                {/* The TITLE is what a researcher recognises a paper by. The doc_id is an
                    internal key and was previously all this column showed. */}
                <td className="w-[22rem] p-2 pr-4">
                  <a
                    href={`/paper?id=${encodeURIComponent(doc)}`}
                    className="block text-[13px] font-medium leading-snug text-slate-100 transition-colors hover:text-teal"
                  >
                    {data.titles?.[doc] || doc}
                  </a>
                  <span className="mt-1 block text-[10px] text-slate-500">
                    {data.years?.[doc] ?? "n/a"} ·{" "}
                    <a
                      href={`https://pubmed.ncbi.nlm.nih.gov/${pmidOf(doc)}/`}
                      target="_blank" rel="noreferrer"
                      className="font-mono text-teal hover:underline"
                    >
                      PMID {pmidOf(doc)}
                    </a>
                  </span>
                </td>
                {data.aspects.map((a) => {
                  // ONE flat map keyed `doc|aspect`, matching the API. The nested lookup
                  // this replaced silently missed every cell.
                  const cell = data.cells[`${doc}|${a.key}`];
                  return (
                    <td key={a.key} className="max-w-xs p-1 align-top">
                      {cell?.reported && cell.text ? (
                        // The truncated cell is a preview, not the evidence. Clicking
                        // opens the passage it came from, because a comparison table a
                        // researcher cannot audit is a comparison table they should not
                        // trust.
                        <button
                          onClick={() => setOpen({ cell, aspect: a, doc })}
                          className="group w-full rounded p-1.5 text-left transition-colors hover:bg-teal/10"
                        >
                          <span className="text-slate-300 group-hover:text-slate-100">
                            {cell.text.slice(0, 170)}…
                          </span>
                          <span className="mt-1 flex items-center gap-1.5 font-mono text-[10px] text-slate-600">
                            {cell.section} {cell.start_char}–{cell.end_char}
                            <span className="text-teal/0 transition-colors group-hover:text-teal/80">
                              open passage →
                            </span>
                          </span>
                        </button>
                      ) : (
                        // "no passage matched" is a fact about THIS SEARCH. "not reported"
                        // was a claim about the paper that the mechanism cannot support:
                        // a cell fills when some chunk clears ts_rank_cd >= 0.08 against
                        // an OR of cue words, so an empty one can be different wording or
                        // a 0.079. Contrast raised from slate-600 (about 2.5:1 here, which
                        // read as a blank cell) to slate-400, because a cell that looks
                        // empty invites exactly the "no effect" reading it should prevent.
                        <span
                          className="block p-1.5 text-[11px] italic leading-snug text-slate-400"
                          title={
                            cell?.reason === "below_threshold"
                              ? `A passage matched but scored ${cell.score} against a ${cell.threshold} threshold: a near miss, not an absence.`
                              : "No passage in this paper contains this dimension's cue words. The paper may still report it in other wording."
                          }
                        >
                          no passage matched
                          {cell?.reason === "below_threshold" && (
                            <span className="mt-0.5 block font-mono text-[10px] not-italic text-amber-300/60">
                              near miss {cell.score}
                            </span>
                          )}
                        </span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {open && (
        <CellViewer
          cell={open.cell}
          aspect={open.aspect}
          docId={open.doc}
          title={data.titles?.[open.doc] || open.doc}
          onClose={() => setOpen(null)}
        />
      )}
    </section>
  );
}

/* ------------------------------------------------------- eval panel */

function EvalPanel({ report }: { report: EvalReport }) {
  if (report.status === "unavailable") {
    return (
      <div className="mt-3 rounded-lg border border-accent/30 bg-accent/10 p-4 text-xs text-accent">
        {report.message}
      </div>
    );
  }
  return (
    <section className="mt-3 rounded-xl border border-edge bg-surface p-5 text-xs backdrop-blur-md">
      <h3 className="mb-3 text-sm font-medium text-slate-200">How this system is measured</h3>

      <div className="grid gap-5 sm:grid-cols-2">
        <div>
          <p className="mb-2 font-medium text-slate-400">Metric panel</p>
          {Object.entries(report.metrics || {}).map(([k, v]) => (
            <div key={k} className="flex justify-between border-b border-edge/40 py-1">
              <span className="font-mono text-slate-500">{k}</span>
              <span className="font-mono text-slate-200">{(v as number).toFixed(4)}</span>
            </div>
          ))}
          <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
            A panel, not one number: a change that improves one metric while degrading
            three is a trade, not an improvement.
          </p>
        </div>

        <div>
          <p className="mb-2 font-medium text-slate-400">By query type</p>
          {Object.entries(report.per_stratum || {}).map(([k, v]: any) => (
            <div key={k} className="flex justify-between border-b border-edge/40 py-1">
              <span className="text-slate-500">
                {k} <span className="text-slate-600">n={v.n}</span>
              </span>
              <span className="font-mono text-slate-200">{v["ndcg@10"]?.toFixed(3)}</span>
            </div>
          ))}
          <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
            Reported separately because an aggregate mean routinely rises while one query
            class collapses.
          </p>
        </div>
      </div>

      {report.caveats?.length > 0 && (
        <div className="mt-5 border-t border-edge pt-4">
          <p className="mb-2 font-medium text-accent">What these numbers do not establish</p>
          <ul className="space-y-1.5 text-[11px] leading-relaxed text-slate-400">
            {report.caveats.map((c: string, i: number) => (
              <li key={i} className="flex gap-2">
                <span className="text-accent">·</span>
                <span>{c}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <p className="mt-4 text-[10px] text-slate-600">
        Generated {report.generated_at} · corpus {report.corpus?.documents} docs /{" "}
        {report.corpus?.queries} queries · sha {report.corpus?.corpus_sha}
      </p>
    </section>
  );
}
