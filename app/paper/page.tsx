"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import WebGLBackground from "@/components/WebGLBackground";

/**
 * A paper, rendered for reading.
 *
 * **Why this page exists.** Every link out of the corpus map, the result list and the
 * comparison grid used to go to `/search`, which meant clicking a specific paper ran a
 * keyword query for its title and returned a list that might not even contain it. The
 * corpus was browsable in aggregate and unreadable in particular.
 *
 * **What it renders.** The passage rows, in order, which ARE the article as this system
 * holds it: reference-stripped and chunked at the offsets retrieval cites. Rendering from
 * the same rows that were searched is the point. A separate full-text copy could drift,
 * and then a highlighted result would point at text the reader cannot find.
 *
 * `?highlight=` scrolls to and marks the passage a result came from, so arriving here from
 * a search lands on the sentence rather than at the top of a long article.
 */

type Passage = {
  chunk_id: string; section: string; ordinal: number;
  start_char: number; end_char: number; text: string;
};
type Paper = {
  doc_id: string; title: string; year: number | null; pmid: string; pmcid: string | null;
  journal: string; license_code: string | null; full_text_chars: number | null;
  blob_url?: string | null;
  mesh_major: string[]; mesh_minor: string[]; descriptors: string[];
  grants: { grant_id?: string; agency?: string }[];
  n_passages: number; passages: Passage[]; highlight: string;
};

export default function PaperPage() {
  const [paper, setPaper] = useState<Paper | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [highlight, setHighlight] = useState("");
  const [backTo, setBackTo] = useState<{ href: string; query: string } | null>(null);
  const hitRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const id = params.get("id");
    const hl = params.get("highlight") || "";
    setHighlight(hl);
    // Carry the originating search so this page is a round trip rather than a dead end.
    // Its only exit used to be "back to the corpus" -> /, which is not where the reader
    // came from, and the nav's Search link lands on an empty box: checking one result and
    // returning for the next cost a retyped query every time.
    const from = params.get("from");
    if (from) {
      const q = new URLSearchParams(from).get("q");
      setBackTo({ href: `/search?${from}`, query: q || "" });
    }
    if (!id) { setError("No paper specified."); setLoading(false); return; }
    fetch(`/api/paper?id=${encodeURIComponent(id)}&highlight=${encodeURIComponent(hl)}`)
      .then(async (r) => {
        const data = await r.json();
        if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
        setPaper(data);
      })
      .catch((e) => setError(e.message || "could not load this paper"))
      .finally(() => setLoading(false));
  }, []);

  // Group passages by section so the article reads as an article, not a list of chunks.
  const sections = useMemo(() => {
    if (!paper) return [];
    const out: { name: string; passages: Passage[] }[] = [];
    for (const p of paper.passages) {
      const last = out[out.length - 1];
      if (last && last.name === p.section) last.passages.push(p);
      else out.push({ name: p.section, passages: [p] });
    }
    return out;
  }, [paper]);

  /**
   * The abstract is pulled OUT of the section list and given its own panel.
   *
   * It arrives as just another section, which is correct for retrieval and wrong for
   * reading: it is the one part of a paper that exists to orient someone before they
   * commit to the rest, so burying it as the first of forty passages wastes it. It stays
   * collapsible because a reader who arrived from a search result is here for a specific
   * passage further down, not for the summary.
   */
  const abstract = useMemo(
    () => sections.find((s) => /^abstract$/i.test(s.name)) ?? null,
    [sections],
  );
  const body = useMemo(
    () => sections.filter((s) => !/^abstract$/i.test(s.name)),
    [sections],
  );
  const abstractText = useMemo(
    () => (abstract ? abstract.passages.map((p) => p.text).join("\n\n") : ""),
    [abstract],
  );
  // Collapsed by default when a highlight sent the reader to a passage further down.
  const [showAbstract, setShowAbstract] = useState(true);
  useEffect(() => { if (highlight) setShowAbstract(false); }, [highlight]);

  useEffect(() => {
    if (paper && highlight) {
      hitRef.current?.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }, [paper, highlight]);

  const norm = (s: string) => s.replace(/\s+/g, " ").trim().toLowerCase();
  const hlNorm = norm(highlight);
  const isHit = (p: Passage) =>
    hlNorm.length > 24 && norm(p.text).includes(hlNorm.slice(0, 120));

  return (
    <main className="relative min-h-screen text-slate-200">
      <WebGLBackground intensity={0.4} parallax />

      <div className="relative mx-auto max-w-3xl px-6 py-14">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
          {backTo ? (
            <a href={backTo.href} className="text-teal hover:underline">
              ← back to results{backTo.query && <> for &ldquo;{backTo.query}&rdquo;</>}
            </a>
          ) : null}
          <a href="/" className="text-slate-500 hover:text-slate-300">the corpus</a>
        </div>

        {loading && <p className="mt-8 text-sm text-slate-500">Loading the paper…</p>}

        {error && (
          <div className="mt-8 rounded-lg border border-accent/30 bg-accent/10 p-4 text-sm text-accent">
            {error}
          </div>
        )}

        {paper && (
          <>
            <header className="mt-6 border-b border-white/10 pb-6">
              <h1 className="text-2xl font-semibold leading-tight tracking-[-0.01em] text-white">
                {paper.title}
              </h1>
              <p className="mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500">
                {paper.journal && <span className="italic text-slate-400">{paper.journal}</span>}
                {paper.year && <span>{paper.year}</span>}
                <a
                  href={`https://pubmed.ncbi.nlm.nih.gov/${paper.pmid}/`}
                  target="_blank" rel="noreferrer"
                  className="font-mono text-teal hover:underline"
                >
                  PMID {paper.pmid}
                </a>
                {paper.pmcid && (
                  <a
                    href={`https://www.ncbi.nlm.nih.gov/pmc/articles/${paper.pmcid}/`}
                    target="_blank" rel="noreferrer"
                    className="font-mono text-teal hover:underline"
                  >
                    {paper.pmcid}
                  </a>
                )}
                {paper.license_code && (
                  <span className="rounded bg-white/5 px-1.5 py-0.5">{paper.license_code}</span>
                )}
                <span>{paper.n_passages} passages</span>
              </p>

              {paper.mesh_major.length > 0 && (
                <div className="mt-4">
                  <p className="text-[10px] uppercase tracking-wider text-slate-600">
                    NLM major topics
                  </p>
                  <div className="mt-1.5 flex flex-wrap gap-1.5">
                    {paper.mesh_major.map((m) => (
                      <a
                        key={m}
                        href={`/search?q=${encodeURIComponent(m)}`}
                        className="rounded border border-teal/25 bg-teal/10 px-1.5 py-0.5 text-[11px] text-teal/90 transition-colors hover:border-teal/50 hover:text-teal"
                      >
                        {m}
                      </a>
                    ))}
                  </div>
                </div>
              )}
            </header>

            {/* ---------- abstract ---------- */}
            {abstract && (
              <section className="mt-6">
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => setShowAbstract((v) => !v)}
                    aria-expanded={showAbstract}
                    className="flex items-center gap-2 rounded-md border border-teal/30 bg-teal/10 px-3 py-1.5 text-xs font-medium text-teal transition-colors hover:border-teal/60 hover:bg-teal/15"
                  >
                    <span className={`inline-block transition-transform ${showAbstract ? "rotate-90" : ""}`}>
                      ▸
                    </span>
                    {showAbstract ? "Hide abstract" : "Show abstract"}
                  </button>
                  <span className="font-mono text-[10px] text-slate-600">
                    {abstractText.length.toLocaleString()} chars
                  </span>
                  <button
                    onClick={() => navigator.clipboard?.writeText(abstractText)}
                    className="text-[11px] text-slate-500 transition-colors hover:text-slate-300"
                  >
                    copy
                  </button>
                </div>
                {showAbstract && (
                  <div className="mt-3 rounded-lg border border-teal/20 bg-teal/[0.04] p-4">
                    <p className="text-[10px] uppercase tracking-[0.16em] text-teal/60">
                      Abstract
                    </p>
                    <div className="mt-2 space-y-3 text-[15px] leading-7 text-slate-300">
                      {abstract.passages.map((p) => (
                        <p key={p.chunk_id}>{p.text}</p>
                      ))}
                    </div>
                  </div>
                )}
              </section>
            )}

            {/* The article. Reference lists were stripped at ingest, so what is here is
                body text: see CLAUDE.md 4.1 for why that matters to precision. */}
            <article className="mt-8 space-y-8">
              {body.map((s) => (
                <section key={s.name}>
                  <h2 className="sticky top-0 -mx-2 bg-[#060B14]/90 px-2 py-1.5 text-[10px] uppercase tracking-[0.16em] text-cyan-300/60 backdrop-blur-sm">
                    {s.name}
                  </h2>
                  <div className="mt-3 space-y-4">
                    {s.passages.map((p) => {
                      const hit = isHit(p);
                      return (
                        <div
                          key={p.chunk_id}
                          ref={hit ? hitRef : undefined}
                          className={`group relative rounded-lg px-3 py-2 text-[15px] leading-7 transition-colors ${
                            hit
                              ? "bg-teal/10 ring-1 ring-teal/40"
                              : "hover:bg-white/[0.03]"
                          }`}
                        >
                          <p className="text-slate-300">{p.text}</p>
                          {/* Offsets on every passage, not just the matched one: this is
                              the provenance the whole product is built around. */}
                          <span className="mt-1 block font-mono text-[10px] text-slate-600 opacity-0 transition-opacity group-hover:opacity-100">
                            chars {p.start_char}–{p.end_char}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </section>
              ))}
            </article>

            {paper.mesh_minor.length > 0 && (
              <footer className="mt-12 border-t border-white/10 pt-6">
                <p className="text-[10px] uppercase tracking-wider text-slate-600">
                  Other NLM descriptors
                </p>
                <p className="mt-1.5 text-xs leading-relaxed text-slate-500">
                  {paper.mesh_minor.join(" · ")}
                </p>
              </footer>
            )}
          </>
        )}
      </div>
    </main>
  );
}
