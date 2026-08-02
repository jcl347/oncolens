"use client";

import { useEffect, useRef } from "react";

/**
 * The corpus, rendered as one continuous object that changes state as you scroll.
 *
 * **Why one morphing visual rather than eight separate effects.** Each stage of the
 * journey is a transformation of the *same* corpus — filtering it by licence, stripping
 * bibliographies, linking it by citation, ranking it. Eight unrelated animations would
 * imply eight unrelated systems. A single point cloud that reorganises makes the actual
 * claim: this is one pipeline, and each stage changes the shape of one thing.
 *
 * Every configuration is meaningful, not decorative:
 *   0 provenance  — points converge toward a single highlighted span
 *   1 licence     — the cloud splits; rejected licences fall away, then most return
 *   2 references  — each document sheds its tail (the bibliography is ~19% of it)
 *   3 signals     — two distributions separate cleanly, which is what AUC 1.000 looks like
 *   4 distribution— failures concentrate in a few columns rather than spreading
 *   5 citations   — points link into a graph; edges are the labels
 *   6 retrieval   — the cloud sorts into ranked columns
 *   7 storage     — collapses into stacked layers, one oversized band removed
 *
 * Implementation notes:
 *   - gl.POINTS with a single index attribute; all positions are computed in the vertex
 *     shader, so there is no per-frame CPU work and no geometry upload.
 *   - Raw WebGL, no three.js: a few KB instead of ~600 KB on a Vercel bundle.
 *   - Honours prefers-reduced-motion (renders one settled frame), pauses when the tab is
 *     hidden, and caps devicePixelRatio at 2 so 4K displays don't quadruple fragment work.
 */

const POINTS = 2600;

const VERT = `
precision highp float;
attribute float aIndex;
uniform float uStage;      // continuous 0..7, fractional during transitions
uniform float uTime;
uniform vec2  uRes;
varying float vAlpha;
varying float vKind;       // 0 = body, 1 = reference/rejected, 2 = highlighted

float hash(float n) { return fract(sin(n * 127.1) * 43758.5453); }
vec2  hash2(float n) { return vec2(hash(n), hash(n + 71.3)); }

// Documents are grouped: index -> (document, position within document).
const float DOCS = 130.0;

vec2 posScatter(float i, float t) {
  vec2 h = hash2(i);
  float a = h.x * 6.2831853 + t * 0.04;
  float r = 0.22 + 0.72 * sqrt(h.y);
  return vec2(cos(a) * r * 0.92, sin(a) * r);
}

vec2 posConverge(float i, float t) {
  vec2 s = posScatter(i, t);
  float pull = 0.86;
  vec2 target = vec2(0.0, 0.02);
  return mix(s, target, pull * (0.35 + 0.65 * hash(i + 9.0)));
}

vec2 posGate(float i, float t, out float rejected) {
  vec2 h = hash2(i * 1.7);
  // 46% was being discarded by the old licence gate; that is the falling group.
  rejected = step(h.x, 0.46);
  float lane = floor(h.y * 5.0);
  float x = -0.78 + lane * 0.39 + (hash(i + 3.0) - 0.5) * 0.16;
  float fall = rejected * (0.55 + 0.45 * sin(t * 0.6 + i * 0.01));
  float y = 0.55 - fract(hash(i + 5.0) + t * 0.05) * 1.1 - fall * 0.55;
  return vec2(x, y);
}

vec2 posShed(float i, float t, out float isRef) {
  float doc = floor(i / (${POINTS}.0 / DOCS));
  float within = fract(i / (${POINTS}.0 / DOCS));
  // Bibliography is a median 19% of characters: the tail of each document.
  isRef = step(0.81, within);
  float col = mod(doc, 26.0);
  float row = floor(doc / 26.0);
  float x = -0.86 + col * 0.069;
  float y = 0.42 - row * 0.2 - within * 0.15;
  // The shed tail drifts down and away.
  float shed = isRef * clamp((sin(t * 0.5) * 0.5 + 0.5), 0.0, 1.0);
  return vec2(x + shed * 0.12 * (hash(i) - 0.5), y - shed * 0.42);
}

vec2 posSeparate(float i, float t, out float grp) {
  vec2 h = hash2(i * 2.3);
  grp = step(0.62, h.x);                       // 0 = body, 1 = references
  // Two well-separated gaussians: this is what perfect rank separation looks like.
  float centre = mix(-0.42, 0.44, grp);
  float spread = mix(0.13, 0.15, grp);
  float g = (hash(i + 11.0) + hash(i + 23.0) + hash(i + 37.0) - 1.5) * 0.9;
  float x = centre + g * spread;
  float y = -0.52 + hash(i + 41.0) * 0.94 + sin(t * 0.4 + i * 0.02) * 0.012;
  return vec2(x, y);
}

vec2 posConcentrate(float i, float t, out float hot) {
  vec2 h = hash2(i * 3.1);
  // Most documents are clean; failures pile into three columns.
  hot = step(0.90, h.x);
  float x, y;
  if (hot > 0.5) {
    float which = floor(hash(i + 13.0) * 3.0);
    x = -0.42 + which * 0.42 + (hash(i + 17.0) - 0.5) * 0.05;
    y = -0.55 + hash(i + 19.0) * 0.92;
  } else {
    x = (h.y - 0.5) * 1.7;
    y = -0.55 + hash(i + 29.0) * 0.30;
  }
  return vec2(x, y + sin(t * 0.35 + i * 0.03) * 0.008);
}

vec2 posGraph(float i, float t) {
  float doc = floor(i / (${POINTS}.0 / DOCS));
  vec2 h = hash2(doc * 5.7);
  float a = h.x * 6.2831853 + t * 0.06;
  float r = 0.30 + 0.60 * h.y;
  vec2 hub = vec2(cos(a) * r * 0.92, sin(a) * r);
  vec2 j = (hash2(i * 1.3) - 0.5) * 0.05;
  return hub + j;
}

vec2 posRank(float i, float t, out float top) {
  vec2 h = hash2(i * 4.9);
  float lane = floor(h.x * 5.0);                 // five systems compared
  // Column heights differ per system, which is the comparison itself.
  float height = 0.34 + lane * 0.085 + sin(t * 0.3 + lane) * 0.012;
  float fill = hash(i + 7.0);
  top = step(0.82, fill);
  float x = -0.72 + lane * 0.36 + (h.y - 0.5) * 0.20;
  float y = -0.60 + fill * height * 2.0;
  return vec2(x, y);
}

vec2 posLayers(float i, float t, out float dropped) {
  vec2 h = hash2(i * 6.3);
  float band = floor(h.x * 4.0);
  dropped = step(3.0, band);                     // the removed 86 MB index
  float y = -0.46 + band * 0.27;
  float x = (h.y - 0.5) * 1.62;
  float push = dropped * (0.5 + 0.5 * sin(t * 0.7));
  return vec2(x + push * 0.5, y + push * 0.25);
}

void main() {
  float i = aIndex;
  float t = uTime;
  float s = clamp(uStage, 0.0, 7.0);
  float seg = floor(s);
  float f = smoothstep(0.0, 1.0, fract(s));

  vec2 pA, pB;
  float kA = 0.0, kB = 0.0, tmp;

  // --- position for the current segment and the next, then blend ---
  if (seg < 0.5)       { pA = posConverge(i, t);    kA = step(0.88, hash(i)) * 2.0; }
  else if (seg < 1.5)  { pA = posGate(i, t, tmp);   kA = tmp; }
  else if (seg < 2.5)  { pA = posShed(i, t, tmp);   kA = tmp; }
  else if (seg < 3.5)  { pA = posSeparate(i, t, tmp); kA = tmp; }
  else if (seg < 4.5)  { pA = posConcentrate(i, t, tmp); kA = tmp; }
  else if (seg < 5.5)  { pA = posGraph(i, t);       kA = step(0.93, hash(i + 2.0)) * 2.0; }
  else if (seg < 6.5)  { pA = posRank(i, t, tmp);   kA = tmp * 2.0; }
  else                 { pA = posLayers(i, t, tmp); kA = tmp; }

  float nseg = min(seg + 1.0, 7.0);
  if (nseg < 0.5)      { pB = posConverge(i, t);    kB = step(0.88, hash(i)) * 2.0; }
  else if (nseg < 1.5) { pB = posGate(i, t, tmp);   kB = tmp; }
  else if (nseg < 2.5) { pB = posShed(i, t, tmp);   kB = tmp; }
  else if (nseg < 3.5) { pB = posSeparate(i, t, tmp); kB = tmp; }
  else if (nseg < 4.5) { pB = posConcentrate(i, t, tmp); kB = tmp; }
  else if (nseg < 5.5) { pB = posGraph(i, t);       kB = step(0.93, hash(i + 2.0)) * 2.0; }
  else if (nseg < 6.5) { pB = posRank(i, t, tmp);   kB = tmp * 2.0; }
  else                 { pB = posLayers(i, t, tmp); kB = tmp; }

  vec2 p = mix(pA, pB, f);
  vKind = mix(kA, kB, f);

  // Aspect-correct so the cloud is not stretched on wide viewports.
  float aspect = uRes.x / max(uRes.y, 1.0);
  vec2 q = vec2(p.x / max(aspect / 1.6, 1.0), p.y);

  gl_Position = vec4(q, 0.0, 1.0);
  float dpr = clamp(uRes.y / 800.0, 0.75, 2.0);
  gl_PointSize = (1.7 + 2.3 * hash(i + 53.0)) * dpr;
  vAlpha = 0.30 + 0.55 * hash(i + 67.0);
}
`;

const FRAG = `
precision highp float;
varying float vAlpha;
varying float vKind;

void main() {
  vec2 c = gl_PointCoord - 0.5;
  float d = dot(c, c);
  if (d > 0.25) discard;
  float soft = smoothstep(0.25, 0.02, d);

  // Body text, discarded/reference material, and highlighted spans read differently.
  vec3 body      = vec3(0.42, 0.72, 0.86);
  vec3 discarded = vec3(0.85, 0.36, 0.34);
  vec3 highlight = vec3(0.99, 0.80, 0.36);

  vec3 col = body;
  col = mix(col, discarded, clamp(vKind, 0.0, 1.0));
  col = mix(col, highlight, clamp(vKind - 1.0, 0.0, 1.0));

  gl_FragColor = vec4(col, vAlpha * soft);
}
`;

function compile(gl: WebGLRenderingContext, type: number, src: string) {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    // Fail silently in production: a broken shader must not take the page down.
    console.warn("shader:", gl.getShaderInfoLog(sh));
    return null;
  }
  return sh;
}

export default function JourneyCanvas({ stage }: { stage: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef(stage);
  const shownRef = useRef(stage);

  useEffect(() => {
    stageRef.current = stage;
  }, [stage]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const gl = canvas.getContext("webgl", {
      antialias: true,
      alpha: true,
      powerPreference: "low-power",
    });
    if (!gl) return; // CSS gradient underneath is the fallback

    const vs = compile(gl, gl.VERTEX_SHADER, VERT);
    const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
    if (!vs || !fs) return;
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return;
    gl.useProgram(prog);

    const idx = new Float32Array(POINTS);
    for (let i = 0; i < POINTS; i++) idx[i] = i;
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, idx, gl.STATIC_DRAW);
    const aIndex = gl.getAttribLocation(prog, "aIndex");
    gl.enableVertexAttribArray(aIndex);
    gl.vertexAttribPointer(aIndex, 1, gl.FLOAT, false, 0, 0);

    const uStage = gl.getUniformLocation(prog, "uStage");
    const uTime = gl.getUniformLocation(prog, "uTime");
    const uRes = gl.getUniformLocation(prog, "uRes");

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = canvas.clientWidth * dpr;
      const h = canvas.clientHeight * dpr;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
        gl.viewport(0, 0, w, h);
      }
    };

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    let running = true;
    const start = performance.now();

    const draw = () => {
      if (!running) return;
      resize();
      // Ease toward the target stage so scrolling feels like motion, not teleporting.
      shownRef.current += (stageRef.current - shownRef.current) * (reduced ? 1 : 0.055);
      const t = reduced ? 12 : (performance.now() - start) / 1000;
      gl.uniform1f(uStage, shownRef.current);
      gl.uniform1f(uTime, t);
      gl.uniform2f(uRes, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.POINTS, 0, POINTS);
      if (!reduced) raf = requestAnimationFrame(draw);
    };
    draw();

    const onVis = () => {
      // No background GPU burn on a hidden tab.
      if (document.hidden) {
        running = false;
        cancelAnimationFrame(raf);
      } else if (!running) {
        running = true;
        draw();
      }
    };
    document.addEventListener("visibilitychange", onVis);
    window.addEventListener("resize", resize);

    return () => {
      running = false;
      cancelAnimationFrame(raf);
      document.removeEventListener("visibilitychange", onVis);
      window.removeEventListener("resize", resize);
      gl.deleteBuffer(buf);
      gl.deleteProgram(prog);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 h-full w-full"
    />
  );
}
