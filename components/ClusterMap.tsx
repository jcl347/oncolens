"use client";

import { useEffect, useMemo, useRef, useState } from "react";

/**
 * The corpus as a map you can interrogate — and rotate.
 *
 * **Why this is not decoration.** Point positions are the document embeddings that
 * retrieval actually uses, projected linearly. Two papers sit near each other here for
 * exactly the reason the same query would return both. Cluster labels are MeSH major
 * topics chosen by log-odds against the corpus background, so a region is named by what
 * makes it *distinct* rather than by "Humans".
 *
 * **Three dimensions, two projections, and honest numbers for both.**
 *
 * A flat PCA map of a 192-dimensional space carried ~14% of the variance, and most of
 * what it did carry was *within*-cluster spread — so distinct research areas piled into
 * one blob. Two changes fix that without lying:
 *
 * 1. **A third axis.** Free structure: one more float per point, and the variance a flat
 *    map simply throws away becomes visible as depth.
 * 2. **A discriminant projection.** PCA on the cluster centroids picks the directions in
 *    which the regions are furthest apart, instead of the directions of greatest total
 *    spread. Still a *linear* map — a rotation of the same space, so distances stay a true
 *    shadow. That is the difference from t-SNE, which reshapes neighbourhoods until
 *    distance and cluster size mean nothing.
 *
 * The discriminant view is chosen *after* seeing the labels, so it shows the partition at
 * its most flattering. That is a camera angle, not an invention — and it is why the view
 * is named on screen and both variance figures are reported next to it.
 *
 * WebGL because 1,710 points with per-frame rotation and depth sorting is more than the
 * DOM should be asked to do; one buffer, one draw call.
 */

type Paper = { doc_id: string; title: string; year: number | null; pmid: string; pmcid: string | null };
type Cluster = {
  id: number; label: string; terms: string[]; size: number;
  x: number; y: number; z?: number; dx?: number; dy?: number; dz?: number;
  papers: Paper[];
};
type Variance = {
  pca_2d: number; pca_3d: number; discriminant_3d: number; cluster_bss: number;
  between_cluster_kept_pca: number; between_cluster_kept_discriminant: number;
  source_dim: number;
};
type Point = { x: number; y: number; z?: number; dx?: number; dy?: number; dz?: number; c: number };
type Data = {
  k: number; n_documents: number; explained_variance: number; projection: string;
  variance?: Variance; clusters: Cluster[]; points: Point[];
};

/** Distinguishable at small size, and legible against a dark field in both themes. */
const PALETTE: [number, number, number][] = [
  [0.38, 0.72, 0.93], [0.98, 0.62, 0.35], [0.46, 0.83, 0.60], [0.90, 0.47, 0.55],
  [0.66, 0.60, 0.95], [0.95, 0.82, 0.40], [0.40, 0.85, 0.83], [0.85, 0.55, 0.80],
  [0.60, 0.78, 0.42], [0.95, 0.70, 0.62], [0.52, 0.66, 0.88], [0.80, 0.72, 0.50],
  [0.55, 0.85, 0.70], [0.88, 0.60, 0.45],
];

/** Shared by the shader and the label placement, so tags track the points exactly. */
function rotatePoint(x: number, y: number, z: number, yaw: number, pitch: number) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const ax = cy * x + sy * z, ay = y, az = -sy * x + cy * z;
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  return { x: ax, y: cp * ay - sp * az, z: sp * ay + cp * az };
}

const VERT = `
attribute vec3 aPos;
attribute float aCluster;
uniform float uActive;      // -1 = none focused
uniform float uYaw;
uniform float uPitch;
uniform vec2  uScale;
uniform float uDpr;
uniform vec3  uPalette[14];
varying vec3  vColor;
varying float vAlpha;

vec3 rotate(vec3 p) {
  float cy = cos(uYaw), sy = sin(uYaw);
  vec3 a = vec3(cy * p.x + sy * p.z, p.y, -sy * p.x + cy * p.z);
  float cp = cos(uPitch), sp = sin(uPitch);
  return vec3(a.x, cp * a.y - sp * a.z, sp * a.y + cp * a.z);
}

void main() {
  vec3 r = rotate(aPos);
  // Gentle perspective: enough to read depth, not enough to distort the linear
  // projection the map exists to show.
  float persp = 1.0 / (1.0 + 0.22 * r.z);
  gl_Position = vec4(r.xy * uScale * persp, 0.0, 1.0);

  int idx = int(aCluster);
  vec3 c = uPalette[0];
  for (int i = 0; i < 14; i++) { if (i == idx) c = uPalette[i]; }
  vColor = c;

  bool focused = (uActive < 0.0 || abs(uActive - aCluster) < 0.5);
  // Focusing dims the rest rather than hiding it: surrounding density is context, and
  // removing it would misrepresent how isolated the cluster is.
  float base = focused ? 1.0 : 0.10;
  // Depth cue: nearer points are brighter and larger.
  float depth = clamp(0.5 - r.z * 0.5, 0.0, 1.0);
  vAlpha = base * (0.45 + 0.55 * depth);
  float size = (focused && uActive >= 0.0) ? 5.6 : 3.4;
  gl_PointSize = size * (0.7 + 0.6 * depth) * uDpr;
}
`;

const FRAG = `
precision mediump float;
varying vec3 vColor;
varying float vAlpha;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  float r2 = dot(d, d);
  if (r2 > 0.25) discard;
  // Bright core with a soft halo — reads as luminous rather than as a flat dot.
  float core = smoothstep(0.25, 0.0, r2);
  float halo = smoothstep(0.25, 0.12, r2);
  vec3 col = mix(vColor, vec3(1.0), core * 0.35);
  gl_FragColor = vec4(col, vAlpha * (0.30 * halo + 0.70 * core));
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
  const labelWrapRef = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState<number | null>(null);
  const [mode, setMode] = useState<"discriminant" | "pca">("discriminant");

  const activeRef = useRef<number | null>(null);
  const yawRef = useRef(0.6);
  const pitchRef = useRef(-0.25);
  const spinRef = useRef(true);
  const dragRef = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => { activeRef.current = active; }, [active]);

  /** Does the payload carry the 3-D / discriminant fields, or is it an older build? */
  const has3d = useMemo(() => data.points.some((p) => p.z !== undefined), [data]);
  const hasDisc = useMemo(() => data.points.some((p) => p.dx !== undefined), [data]);
  const useDisc = mode === "discriminant" && hasDisc;

  const coordsOf = useMemo(() => {
    return data.points.map((p) =>
      useDisc
        ? { x: p.dx ?? p.x, y: p.dy ?? p.y, z: p.dz ?? 0 }
        : { x: p.x, y: p.y, z: p.z ?? 0 }
    );
  }, [data, useDisc]);

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

    const n = coordsOf.length;
    const pos = new Float32Array(n * 3);
    const cl = new Float32Array(n);
    coordsOf.forEach((p, i) => {
      pos[i * 3] = p.x; pos[i * 3 + 1] = p.y; pos[i * 3 + 2] = p.z;
      cl[i] = data.points[i].c;
    });

    const pb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, pb);
    gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
    const aPos = gl.getAttribLocation(prog, "aPos");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);

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
    const uYaw = gl.getUniformLocation(prog, "uYaw");
    const uPitch = gl.getUniformLocation(prog, "uPitch");
    const uDpr = gl.getUniformLocation(prog, "uDpr");

    gl.enable(gl.BLEND);
    // Additive-ish: overlapping regions glow where the corpus is dense, which is real
    // information — density here means many papers on one topic.
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    let raf = 0;
    let last = performance.now();

    const labelEls = () =>
      Array.from(labelWrapRef.current?.children ?? []) as HTMLElement[];

    const frame = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      if (spinRef.current && !dragRef.current) yawRef.current += dt * 0.12;

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = canvas.clientWidth * dpr, h = canvas.clientHeight * dpr;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w; canvas.height = h; gl.viewport(0, 0, w, h);
      }
      // Preserve aspect so the projection is not stretched — a stretched map would
      // misrepresent the distances it exists to show.
      const a = w / h;
      const sx = a > 1 ? 0.82 / a : 0.82;
      const sy = a > 1 ? 0.82 : 0.82 * a;
      gl.uniform2f(uScale, sx, sy);
      gl.uniform1f(uActive, activeRef.current ?? -1);
      gl.uniform1f(uYaw, yawRef.current);
      gl.uniform1f(uPitch, pitchRef.current);
      gl.uniform1f(uDpr, dpr);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.POINTS, 0, n);

      // Labels follow the same rotation, positioned in CSS percentages.
      const els = labelEls();
      data.clusters.forEach((c, i) => {
        const el = els[i];
        if (!el) return;
        const cx = useDisc ? (c.dx ?? c.x) : c.x;
        const cyy = useDisc ? (c.dy ?? c.y) : c.y;
        const cz = useDisc ? (c.dz ?? 0) : (c.z ?? 0);
        const r = rotatePoint(cx, cyy, cz, yawRef.current, pitchRef.current);
        const persp = 1 / (1 + 0.22 * r.z);
        el.style.left = `${50 + r.x * sx * persp * 50}%`;
        el.style.top = `${50 - r.y * sy * persp * 50}%`;
        // Behind the cloud reads dimmer, so the label ordering matches the depth cue.
        const depth = Math.min(Math.max(0.5 - r.z * 0.5, 0), 1);
        el.style.opacity = String(
          activeRef.current === null ? 0.45 + 0.55 * depth
            : activeRef.current === c.id ? 1 : 0.18
        );
        el.style.zIndex = String(10 + Math.round(depth * 20));
      });

      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      gl.deleteBuffer(pb); gl.deleteBuffer(cb); gl.deleteProgram(prog);
    };
  }, [data, coordsOf, useDisc]);

  /* ------------------------------------------------------------- pointer */
  const onDown = (e: React.PointerEvent) => {
    dragRef.current = { x: e.clientX, y: e.clientY };
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
  };
  const onMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    if (!d) return;
    yawRef.current += (e.clientX - d.x) * 0.006;
    pitchRef.current = Math.max(-1.2, Math.min(1.2, pitchRef.current + (e.clientY - d.y) * 0.006));
    dragRef.current = { x: e.clientX, y: e.clientY };
  };
  const onUp = () => { dragRef.current = null; };

  const activeCluster = active === null ? null : data.clusters.find((c) => c.id === active) ?? null;
  const rgb = (i: number) => {
    const c = PALETTE[i % PALETTE.length];
    return `rgb(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)})`;
  };

  const v = data.variance;
  const shownVar = v ? (useDisc ? v.discriminant_3d : v.pca_3d) : data.explained_variance;
  const shownSep = v ? (useDisc ? v.between_cluster_kept_discriminant : v.between_cluster_kept_pca) : null;

  return (
    <div className="grid gap-6 lg:grid-cols-[1.25fr_1fr]">
      <div>
        <div className="relative aspect-[4/3] w-full overflow-hidden rounded-lg border border-white/10 bg-[radial-gradient(ellipse_at_50%_40%,#0d1420_0%,#05070c_70%)]">
          <canvas
            ref={canvasRef}
            className="h-full w-full cursor-grab touch-none active:cursor-grabbing"
            onPointerDown={onDown}
            onPointerMove={onMove}
            onPointerUp={onUp}
            onPointerLeave={onUp}
          />
          <div ref={labelWrapRef} className="pointer-events-none absolute inset-0">
            {data.clusters.map((c) => (
              <button
                key={c.id}
                onClick={() => setActive(active === c.id ? null : c.id)}
                className={`pointer-events-auto absolute -translate-x-1/2 -translate-y-1/2 whitespace-nowrap rounded px-1.5 py-0.5 text-[10px] backdrop-blur-sm transition-colors ${
                  active === c.id
                    ? "bg-white/15 text-white"
                    : "bg-black/45 text-slate-300 hover:bg-white/10 hover:text-white"
                }`}
                style={{ borderLeft: `2px solid ${rgb(c.id)}` }}
              >
                {c.terms[0] ?? c.label}
              </button>
            ))}
          </div>

          {/* controls */}
          <div className="absolute bottom-2 left-2 flex items-center gap-1.5">
            {hasDisc && (
              <div className="flex overflow-hidden rounded border border-white/10 bg-black/50 text-[10px] backdrop-blur-sm">
                {(["discriminant", "pca"] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    className={`px-2 py-1 transition-colors ${
                      mode === m ? "bg-white/15 text-white" : "text-slate-400 hover:text-slate-200"
                    }`}
                    title={
                      m === "discriminant"
                        ? "Linear axes chosen to separate the discovered clusters"
                        : "Linear axes of greatest total variance"
                    }
                  >
                    {m === "discriminant" ? "separation" : "variance"}
                  </button>
                ))}
              </div>
            )}
            <button
              onClick={() => { spinRef.current = !spinRef.current; }}
              className="rounded border border-white/10 bg-black/50 px-2 py-1 text-[10px] text-slate-400 backdrop-blur-sm transition-colors hover:text-slate-200"
            >
              pause / spin
            </button>
          </div>
          <span className="pointer-events-none absolute bottom-2 right-2 text-[10px] text-slate-600">
            drag to rotate
          </span>
        </div>

        <p className="mt-3 text-xs leading-relaxed text-slate-500">
          {data.n_documents.toLocaleString()} papers positioned by their retrieval
          embeddings, {data.k} clusters labelled by the MeSH major topics most distinctive
          to each.{" "}
          <span className="text-amber-300/70">
            These three axes carry {(shownVar * 100).toFixed(0)}% of the variance in a{" "}
            {v?.source_dim ?? 192}-dimensional space
            {shownSep != null && <> and {(shownSep * 100).toFixed(0)}% of the separation between clusters</>}
          </span>
          {v && (
            <>
              , and the clustering itself accounts for {(v.cluster_bss * 100).toFixed(0)}% of
              total variance
            </>
          )}
          . Both views are linear projections, so distances stay comparable —{" "}
          {useDisc
            ? "the “separation” view rotates to the plane where the clusters are furthest apart, which is a camera angle chosen after seeing the labels, not invented structure."
            : "the “variance” view takes the directions of greatest total spread."}{" "}
          t-SNE would separate the clusters far more cleanly and the separation would not
          mean anything.
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
              Papers nearest this cluster&apos;s centre in the full{" "}
              {v?.source_dim ?? 192}-dimensional space — not in the projection, which is for
              display only.
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
