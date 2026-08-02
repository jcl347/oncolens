"use client";

import { useEffect, useRef } from "react";

export type StageState = "ok" | "pending" | "error" | "running";

export interface Stage {
  id: string;
  label: string;
  state: StageState;
}

/**
 * WebGL data-pipeline visualisation, driven by real infrastructure status.
 *
 * Design note: the reference aesthetic (jordan-limperis.dev) is deliberately minimal and
 * uses no WebGL at all — it trusts content over effects. That is the right instinct, so
 * the effect budget is spent on exactly one element that carries information rather than
 * decorating the page: particles flow between stages **only where the upstream stage is
 * actually configured**, so a broken link is visible as a gap in the flow before you read
 * a single status label.
 *
 * Everything else on the page stays flat, typographic and monochrome.
 *
 * Cost controls: pauses when hidden, honours prefers-reduced-motion (renders a static
 * frame with stage colours intact, so no information is lost), caps DPR at 2.
 */
export default function PipelineCanvas({ stages }: { stages: Stage[] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const stagesRef = useRef(stages);
  stagesRef.current = stages;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const gl = canvas.getContext("webgl", { antialias: true, alpha: true });
    if (!gl) return; // graceful: the DOM stage labels below remain fully readable

    const vsrc = `
      attribute vec2 aPos;
      attribute vec3 aColor;
      attribute float aSize;
      uniform vec2 uRes;
      varying vec3 vColor;
      varying float vSize;
      void main() {
        vec2 clip = (aPos / uRes) * 2.0 - 1.0;
        gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
        gl_PointSize = aSize;
        vColor = aColor;
        vSize = aSize;
      }
    `;
    const fsrc = `
      precision mediump float;
      varying vec3 vColor;
      varying float vSize;
      void main() {
        vec2 c = gl_PointCoord - 0.5;
        float d = length(c);
        // Soft round sprite with a bright core; alpha falloff avoids square artefacts.
        float a = smoothstep(0.5, 0.08, d);
        float core = smoothstep(0.22, 0.0, d) * 0.7;
        gl_FragColor = vec4(vColor + core, a);
      }
    `;

    const mk = (t: number, src: string) => {
      const s = gl.createShader(t)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        console.error(gl.getShaderInfoLog(s));
        return null;
      }
      return s;
    };
    const vs = mk(gl.VERTEX_SHADER, vsrc);
    const fs = mk(gl.FRAGMENT_SHADER, fsrc);
    if (!vs || !fs) return;

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.useProgram(prog);

    const aPos = gl.getAttribLocation(prog, "aPos");
    const aColor = gl.getAttribLocation(prog, "aColor");
    const aSize = gl.getAttribLocation(prog, "aSize");
    const uRes = gl.getUniformLocation(prog, "uRes");

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // additive: overlapping flow reads as density

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let W = 0, H = 0;
    const resize = () => {
      W = canvas.clientWidth * dpr;
      H = canvas.clientHeight * dpr;
      canvas.width = W;
      canvas.height = H;
      gl.viewport(0, 0, W, H);
      gl.uniform2f(uRes, W, H);
    };
    resize();
    window.addEventListener("resize", resize);

    const COLORS: Record<StageState, [number, number, number]> = {
      ok:      [0.24, 0.68, 0.75],
      running: [0.93, 0.51, 0.29],
      pending: [0.28, 0.33, 0.40],
      error:   [0.85, 0.30, 0.32],
    };

    // Particles travel a normalised 0..1 track; each is assigned to a segment between
    // consecutive stages. Segment k only emits when stage k is configured.
    const N = 220;
    const t = new Float32Array(N);
    const seg = new Int32Array(N);
    const jitter = new Float32Array(N);
    const speed = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      t[i] = Math.random();
      seg[i] = i % Math.max(1, stagesRef.current.length - 1);
      jitter[i] = (Math.random() - 0.5) * 2;
      speed[i] = 0.10 + Math.random() * 0.16;
    }

    const data = new Float32Array(N * 6);
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    let running = true;
    let last = performance.now();

    const frame = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;

      const st = stagesRef.current;
      const n = st.length;
      const padX = W * 0.09;
      const spanX = W - padX * 2;
      const midY = H * 0.5;
      const stepX = n > 1 ? spanX / (n - 1) : 0;

      let w = 0;

      // --- flow particles -------------------------------------------------
      for (let i = 0; i < N; i++) {
        const s = seg[i];
        const from = st[s];
        const to = st[s + 1];
        // A segment carries data only if its upstream stage is genuinely configured.
        // A gap in the flow therefore *means* something.
        const live = from && to && (from.state === "ok" || from.state === "running");
        if (!live) continue;

        if (!reduced) {
          t[i] += speed[i] * dt;
          if (t[i] > 1) t[i] -= 1;
        }

        const x0 = padX + s * stepX;
        const x1 = padX + (s + 1) * stepX;
        const x = x0 + (x1 - x0) * t[i];
        const wobble = Math.sin(t[i] * Math.PI * 2 + i) * 5 * dpr * jitter[i];
        const y = midY + wobble;

        const c = COLORS[from.state === "running" ? "running" : "ok"];
        const fade = Math.sin(t[i] * Math.PI); // fade in/out at the endpoints
        data[w++] = x; data[w++] = y;
        data[w++] = c[0] * fade; data[w++] = c[1] * fade; data[w++] = c[2] * fade;
        data[w++] = (1.6 + 1.7 * fade) * dpr;
      }

      // --- stage nodes ----------------------------------------------------
      for (let i = 0; i < n; i++) {
        const c = COLORS[st[i].state];
        const x = padX + i * stepX;
        const pulse = st[i].state === "running" && !reduced
          ? 1 + 0.28 * Math.sin(now / 180)
          : 1;
        data[w++] = x; data[w++] = midY;
        data[w++] = c[0]; data[w++] = c[1]; data[w++] = c[2];
        data[w++] = 13 * dpr * pulse;
      }

      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.bufferData(gl.ARRAY_BUFFER, data.subarray(0, w), gl.DYNAMIC_DRAW);
      const stride = 6 * 4;
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, stride, 0);
      gl.enableVertexAttribArray(aColor);
      gl.vertexAttribPointer(aColor, 3, gl.FLOAT, false, stride, 8);
      gl.enableVertexAttribArray(aSize);
      gl.vertexAttribPointer(aSize, 1, gl.FLOAT, false, stride, 20);
      gl.drawArrays(gl.POINTS, 0, w / 6);

      if (running) raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);

    const onVis = () => {
      running = document.visibilityState === "visible";
      if (running) {
        last = performance.now();
        raf = requestAnimationFrame(frame);
      } else {
        cancelAnimationFrame(raf);
      }
    };
    document.addEventListener("visibilitychange", onVis);

    return () => {
      running = false;
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      document.removeEventListener("visibilitychange", onVis);
      gl.deleteProgram(prog);
      gl.deleteBuffer(buf);
    };
  }, []);

  return (
    <div className="relative">
      <canvas ref={ref} className="h-[104px] w-full" aria-hidden="true" />
      {/* Labels live in the DOM, not the shader: they must remain readable and
          selectable if WebGL is unavailable or motion is reduced. */}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 flex justify-between px-[7%]">
        {stages.map((s) => (
          <span
            key={s.id}
            className={`translate-y-1 font-mono text-[10px] uppercase tracking-wider ${
              s.state === "ok" ? "text-teal"
                : s.state === "error" ? "text-red-400"
                : s.state === "running" ? "text-accent"
                : "text-slate-600"
            }`}
          >
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}
