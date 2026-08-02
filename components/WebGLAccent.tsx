"use client";

import { useEffect, useRef } from "react";

/**
 * Small WebGL accents for the things a reader is meant to act on.
 *
 * **Why WebGL for something this small.** A CSS gradient animation can only interpolate
 * between stops on a fixed track; it reads as a loop within a second or two. These are
 * driven by value noise, so the motion never repeats exactly and sits below the threshold
 * where the eye starts predicting it. That is the difference between "alive" and
 * "animated", and on a research tool the first is the only acceptable one.
 *
 * **Cost control is the whole design.** Decorative canvases are how a page ends up
 * burning a GPU core to render six pixels of shimmer:
 *
 *  - one shared `requestAnimationFrame` loop drives every instance, not one per canvas
 *  - an `IntersectionObserver` unsubscribes anything scrolled out of view
 *  - `prefers-reduced-motion` renders a single static frame and stops
 *  - the whole tab pauses on `visibilitychange`
 *  - backing stores are capped at 2x DPR and are tiny by construction (a rule is 2px tall)
 *
 * `variant="rule"` is a flowing hairline for section headings. `variant="glow"` is a soft
 * pulse that sits behind a button label. Both take the accent colour so a call to action
 * and its heading read as the same system.
 */

type Variant = "rule" | "glow";

const VERT = `
attribute vec2 aPos;
varying vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

const FRAG = `
precision mediump float;
varying vec2 vUv;
uniform float uTime;
uniform vec3  uColor;
uniform float uGlow;    // 0 = rule, 1 = glow
uniform float uHover;   // eased 0..1

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
             mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

void main() {
  float t = uTime * 0.35;

  if (uGlow < 0.5) {
    // RULE: a travelling crest along x, thickened and brightened on hover. Two noise
    // octaves at different speeds so the crest drifts rather than sweeps on a timer.
    float n = noise(vec2(vUv.x * 3.0 - t, t * 0.5))
            + 0.5 * noise(vec2(vUv.x * 7.0 + t * 1.3, 0.0));
    float crest = smoothstep(0.55, 1.35, n);
    float body = 0.18 + 0.55 * crest;
    float a = body * (0.55 + 0.45 * uHover);
    gl_FragColor = vec4(uColor * (0.7 + 0.6 * crest), a);
  } else {
    // GLOW: a soft field behind a label. Brightest at the leading edge so the control
    // reads as directional, which is what a call to action should feel like.
    vec2 p = vUv;
    float n = noise(vec2(p.x * 2.5 - t * 0.8, p.y * 2.0 + t * 0.4));
    float edge = smoothstep(1.0, 0.15, p.x);
    float vert = smoothstep(0.0, 0.45, p.y) * smoothstep(1.0, 0.55, p.y);
    float a = (0.10 + 0.30 * n) * edge * (0.35 + 0.65 * vert);
    a *= 0.35 + 0.85 * uHover;
    gl_FragColor = vec4(uColor, a);
  }
}
`;

/* ------------------------------------------------------------------ shared clock */

type Sub = () => void;
const subscribers = new Set<Sub>();
let rafId = 0;
let running = false;

function tick() {
  subscribers.forEach((fn) => fn());
  rafId = requestAnimationFrame(tick);
}
function ensureClock() {
  if (running || subscribers.size === 0) return;
  running = true;
  rafId = requestAnimationFrame(tick);
}
function stopClock() {
  running = false;
  cancelAnimationFrame(rafId);
}
function subscribe(fn: Sub) {
  subscribers.add(fn);
  ensureClock();
  return () => {
    subscribers.delete(fn);
    if (subscribers.size === 0) stopClock();
  };
}
if (typeof document !== "undefined") {
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") stopClock();
    else ensureClock();
  });
}

function compile(gl: WebGLRenderingContext, type: number, src: string) {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) return null;
  return sh;
}

export default function WebGLAccent({
  variant = "rule",
  color = [0.36, 0.79, 0.87],
  hover = false,
  className = "",
}: {
  variant?: Variant;
  /** Linear RGB 0..1. Defaults to the cyan used for interactive affordances. */
  color?: [number, number, number];
  /** Drive from the parent's hover/focus state so the accent responds to the control. */
  hover?: boolean;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hoverRef = useRef(0);
  const targetRef = useRef(0);

  useEffect(() => { targetRef.current = hover ? 1 : 0; }, [hover]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const gl = canvas.getContext("webgl", { antialias: false, alpha: true,
                                            premultipliedAlpha: false,
                                            powerPreference: "low-power" });
    if (!gl) return;

    const vs = compile(gl, gl.VERTEX_SHADER, VERT);
    const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
    if (!vs || !fs) return;
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return;
    gl.useProgram(prog);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "aPos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    const uTime = gl.getUniformLocation(prog, "uTime");
    const uColor = gl.getUniformLocation(prog, "uColor");
    const uGlow = gl.getUniformLocation(prog, "uGlow");
    const uHover = gl.getUniformLocation(prog, "uHover");
    gl.uniform3f(uColor, color[0], color[1], color[2]);
    gl.uniform1f(uGlow, variant === "glow" ? 1 : 0);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    const start = performance.now();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    const draw = () => {
      const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
      const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w; canvas.height = h; gl.viewport(0, 0, w, h);
      }
      hoverRef.current += (targetRef.current - hoverRef.current) * 0.12;
      gl.uniform1f(uHover, hoverRef.current);
      gl.uniform1f(uTime, (performance.now() - start) / 1000);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) {
      hoverRef.current = targetRef.current;
      draw();
      return () => { gl.deleteProgram(prog); gl.deleteBuffer(buf); };
    }

    // Only animate what is on screen. A long page has many of these and none of the
    // ones below the fold are worth a frame.
    let unsubscribe: (() => void) | null = null;
    const io = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !unsubscribe) unsubscribe = subscribe(draw);
      else if (!entry.isIntersecting && unsubscribe) { unsubscribe(); unsubscribe = null; }
    }, { rootMargin: "100px" });
    io.observe(canvas);
    draw();

    return () => {
      io.disconnect();
      unsubscribe?.();
      gl.deleteProgram(prog);
      gl.deleteBuffer(buf);
    };
  }, [variant, color, className]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className={`pointer-events-none ${className}`}
    />
  );
}
