"use client";

import { useEffect, useRef, useState } from "react";

/**
 * The corpus as a map you can interrogate.
 *
 * **Why this is not decoration.** Point positions are the document embeddings that
 * retrieval actually uses, projected linearly. Two papers sit near each other here for
 * exactly the reason the same query would return both. Cluster labels are MeSH major
 * topics chosen by log-odds against the corpus background, so a region is named by what
 * makes it *distinct* rather than by "Humans".
 *
 * **The projection is honest about what it hides.** Two dimensions capture ~14% of the
 * variance in a 192-dimensional space. t-SNE or UMAP would look far cleaner and would
 * invent that cleanliness — their distances and cluster sizes carry no meaning. The
 * explained-variance figure is displayed rather than buried, because a map that overstates
 * its own fidelity is worse than a plain list.
 *
 * WebGL because 1,710 points with per-frame hover testing is more than the DOM should be
 * asked to do; gl.POINTS with a single buffer keeps it at one draw call.
 */

type Paper = { doc_id: string; title: string; year: number | null; pmid: string; pmcid: string | null };
type Cluster = { id: number; label: string; terms: string[]; size: number; x: number; y: number; papers: Paper[] };
type Data = {
  k: number; n_documents: number; explained_variance: number; projection: string;
  clusters: Cluster[]; points: { x: number; y: number; c: number }[];
};

/** Distinguishable at small size, and legible against a dark field in both themes. */
const PALETTE: [number, number, number][] = [
  [0.38, 0.72, 0.93], [0.98, 0.62, 0.35], [0.46, 0.83, 0.60], [0.90, 0.47, 0.55],
  [0.66, 0.60, 0.95], [0.95, 0.82, 0.40], [0.40, 0.85, 0.83], [0.85, 0.55, 0.80],
  [0.60, 0.78, 0.42], [0.95, 0.70, 0.62], [0.52, 0.66, 0.88], [0.80, 0.72, 0.50],
  [0.55, 0.85, 0.70], [0.88, 0.60, 0.45],
];

const VERT = `
attribute vec2 aPos;
attribute float aCluster;
uniform float uActive;     // -1 = none focused
uniform vec2  uScale;
varying vec3 vColor;
varying float vDim;
uniform vec3 uPalette[14];
void main() {
  gl_Position = vec4(aPos * uScale, 0.0, 1.0);
  int idx = int(aCluster);
  vec3 c = uPalette[0];
  for (int i = 0; i < 14; i++) { if (i == idx) c = uPalette[i]; }
  vColor = c;
  // Focusing a cluster dims the rest rather than hiding it: the surrounding density is
  // context, and removing it would misrepresent how isolated the cluster is.
  vDim = (uActive < 0.0 || abs(uActive - aCluster) < 0.5) ? 1.0 : 0.16;
  gl_PointSize = (uActive >= 0.0 && abs(uActive - aCluster) < 0.5) ? 4.5 : 3.0;
}
`;

const FRAG = `
precision mediump float;
varying vec3 vColor;
varying float vDim;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  float r = dot(d, d);
  if (r > 0.25) discard;
  gl_FragColor = vec4(vColor, vDim * smoothstep(0.25, 0.05, r) * 0.85);
}
`;

function compile(gl: WebGLRenderingContext, type: number, src: string) {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    console.warn(gl.getShaderInfoLog(sh));
    return null;
  }
  return sh;
}

export default function ClusterMap({ data }: { data: Data }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [active, setActive] = useState<number | null>(null);
  const activeRef = useRef<number | null>(null);
  const drawRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    activeRef.current = active;
    drawRef.current?.();
  }, [active]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const gl = canvas.getContext("webgl", { antialias: true, alpha: true });
    if (!gl) return;

    const vs = compile(gl, gl.VERTEX_SHADER, VERT);
    const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
    if (!vs || !fs) return;
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return;
    gl.useProgram(prog);

    const n = data.points.length;
    const pos = new Float32Array(n * 2);
    const cl = new Float32Array(n);
    data.points.forEach((p, i) => { pos[i * 2] = p.x; pos[i * 2 + 1] = p.y; cl[i] = p.c; });

    const pb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, pb);
    gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
    const aPos = gl.getAttribLocation(prog, "aPos");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    const cb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, cb);
    gl.bufferData(gl.ARRAY_BUFFER, cl, gl.STATIC_DRAW);
    const aCluster = gl.getAttribLocation(prog, "aCluster");
    gl.enableVertexAttribArray(aCluster);
    gl.vertexAttribPointer(aCluster, 1, gl.FLOAT, false, 0, 0);

    const flat = new Float32Array(14 * 3);
    PALETTE.forEach((c, i) => { flat[i * 3] = c[0]; flat[i * 3 + 1] = c[1]; flat[i * 3 + 2] = c[2]; });
    gl.uniform3fv(gl.getUniformLocation(prog, "uPalette"), flat);
    const uActive = gl.getUniformLocation(prog, "uActive");
    const uScale = gl.getUniformLocation(prog, "uScale");

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    const draw = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = canvas.clientWidth * dpr, h = canvas.clientHeight * dpr;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w; canvas.height = h; gl.viewport(0, 0, w, h);
      }
      // Preserve aspect so the projection is not stretched — a stretched map would
      // misrepresent the distances it exists to show.
      const a = w / h;
      gl.uniform2f(uScale, a > 1 ? 0.88 / a : 0.88, a > 1 ? 0.88 : 0.88 * a);
      gl.uniform1f(uActive, activeRef.current ?? -1);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.POINTS, 0, n);
    };
    drawRef.current = draw;
    draw();
    window.addEventListener("resize", draw);
    return () => {
      window.removeEventListener("resize", draw);
      gl.deleteBuffer(pb); gl.deleteBuffer(cb); gl.deleteProgram(prog);
    };
  }, [data]);

  const activeCluster = active === null ? null : data.clusters.find((c) => c.id === active) ?? null;
  const rgb = (i: number) => {
    const c = PALETTE[i % PALETTE.length];
    return `rgb(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)})`;
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[1.25fr_1fr]">
      <div>
        <div className="relative aspect-[4/3] w-full overflow-hidden rounded-lg border border-white/10 bg-[#070a10]">
          <canvas ref={canvasRef} className="h-full w-full" />
          {data.clusters.map((c) => (
            <button
              key={c.id}
              onClick={() => setActive(active === c.id ? null : c.id)}
              className={`absolute -translate-x-1/2 -translate-y-1/2 whitespace-nowrap rounded px-1.5 py-0.5 text-[10px] transition-all ${
                active === c.id
                  ? "z-10 bg-white/15 text-white"
                  : active === null
                    ? "bg-black/40 text-slate-300 hover:bg-white/10 hover:text-white"
                    : "bg-black/30 text-slate-600"
              }`}
              style={{
                left: `${50 + c.x * 44}%`,
                top: `${50 - c.y * 44}%`,
                borderLeft: `2px solid ${rgb(c.id)}`,
              }}
            >
              {c.terms[0] ?? c.label}
            </button>
          ))}
        </div>
        <p className="mt-3 text-xs leading-relaxed text-slate-500">
          {data.n_documents.toLocaleString()} papers positioned by their retrieval
          embeddings, {data.k} clusters labelled by the MeSH major topics most distinctive
          to each. <span className="text-amber-300/70">
            These two dimensions carry {(data.explained_variance * 100).toFixed(0)}% of the
            variance in a 192-dimensional space
          </span>{" "}
          — a linear projection, so distances are comparable. t-SNE would separate the
          clusters far more cleanly and the separation would not mean anything.
        </p>
      </div>

      <div className="min-w-0">
        {activeCluster ? (
          <div>
            <div className="flex items-baseline justify-between gap-3">
              <h3 className="text-sm font-medium text-white">
                {activeCluster.terms[0] ?? activeCluster.label}
              </h3>
              <button onClick={() => setActive(null)} className="text-[11px] text-slate-500 hover:text-slate-300">
                clear
              </button>
            </div>
            <p className="mt-1 text-[11px] text-slate-500">
              {activeCluster.size} papers · distinctive terms:{" "}
              {activeCluster.terms.slice(1).join(", ") || "—"}
            </p>
            <ul className="mt-4 space-y-2">
              {activeCluster.papers.map((p) => (
                <li key={p.doc_id}>
                  <a
                    href={`/search?q=${encodeURIComponent(p.title.slice(0, 80))}`}
                    className="block rounded border border-white/8 bg-white/[0.02] p-2.5 transition-colors hover:border-white/20 hover:bg-white/[0.05]"
                  >
                    <span className="block text-xs leading-snug text-slate-200">{p.title}</span>
                    <span className="mt-1 block font-mono text-[10px] text-slate-500">
                      {p.year ?? "—"} · PMID {p.pmid}
                      {p.pmcid ? ` · ${p.pmcid}` : ""}
                    </span>
                  </a>
                </li>
              ))}
            </ul>
            <p className="mt-3 text-[11px] text-slate-500">
              Papers nearest this cluster&apos;s centre in the full 192-dimensional space —
              not in the projection, which is for display only.
            </p>
          </div>
        ) : (
          <div>
            <h3 className="text-sm font-medium text-white">Research areas in the corpus</h3>
            <p className="mt-1 text-xs text-slate-400">
              Select a region to see its papers. Clusters are found in the embedding space,
              then named by NLM&apos;s human indexing.
            </p>
            <ul className="mt-4 max-h-[26rem] space-y-1 overflow-y-auto pr-1">
              {data.clusters.map((c) => (
                <li key={c.id}>
                  <button
                    onClick={() => setActive(c.id)}
                    className="flex w-full items-center gap-2.5 rounded px-2 py-1.5 text-left transition-colors hover:bg-white/5"
                  >
                    <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: rgb(c.id) }} />
                    <span className="min-w-0 flex-1 truncate text-xs text-slate-300">
                      {c.terms[0] ?? c.label}
                    </span>
                    <span className="shrink-0 font-mono text-[10px] text-slate-500">{c.size}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
