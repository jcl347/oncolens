"use client";

import { useEffect, useRef } from "react";

/**
 * Animated embedding-space background.
 *
 * Deliberately *meaningful* rather than decorative: the field depicts drifting points in a
 * projected vector space with soft density ridges, which is what the retrieval layer
 * actually operates over. It sits behind content at low contrast so it never competes with
 * the text — this is a research tool, and unreadable body copy is not a fair trade for
 * motion.
 *
 * Raw WebGL, no three.js: the whole effect is one fragment shader, so it costs a few KB
 * instead of ~600 KB and keeps the Vercel bundle small.
 *
 * Accessibility and cost controls:
 *  - honours `prefers-reduced-motion` by rendering a single static frame
 *  - pauses entirely when the tab is hidden (no background GPU burn)
 *  - caps devicePixelRatio at 2 so 4K displays don't quadruple fragment work
 */
export default function WebGLBackground({
  intensity = 1,
  parallax = false,
}: {
  /** Scales the whole field's brightness. Below 1 for pages with dense body copy. */
  intensity?: number;
  /** Drift the field with scroll and cursor. Gives a long page depth without motion
   *  that competes with reading — the shift is a fraction of the scroll distance. */
  parallax?: boolean;
} = {}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl", {
      antialias: false,
      alpha: true,
      powerPreference: "low-power",
    });
    // No WebGL (older browsers, some VMs): the CSS gradient underneath is the fallback.
    if (!gl) return;

    const vert = `
      attribute vec2 aPos;
      void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
    `;

    const frag = `
      precision highp float;
      uniform vec2  uRes;
      uniform float uTime;
      uniform float uIntensity;
      uniform vec2  uShift;    // scroll + pointer parallax, in field units

      // Hash/noise/fbm: standard value-noise stack, cheap enough for a full-screen pass.
      float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

      float noise(vec2 p) {
        vec2 i = floor(p), f = fract(p);
        vec2 u = f * f * (3.0 - 2.0 * f);
        return mix(mix(hash(i + vec2(0,0)), hash(i + vec2(1,0)), u.x),
                   mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), u.x), u.y);
      }

      float fbm(vec2 p) {
        float v = 0.0, a = 0.5;
        for (int i = 0; i < 5; i++) { v += a * noise(p); p *= 2.02; a *= 0.5; }
        return v;
      }

      void main() {
        vec2 uv = gl_FragCoord.xy / uRes.xy;
        vec2 p  = (gl_FragCoord.xy - 0.5 * uRes.xy) / uRes.y + uShift;

        float t = uTime * 0.020;

        // Two counter-drifting fields: reads as slow structural motion rather than
        // a looping texture, which is what makes it feel like a space and not a GIF.
        float f1 = fbm(p * 2.4 + vec2(t, -t * 0.6));
        float f2 = fbm(p * 4.1 - vec2(t * 0.8, t * 0.4) + f1);

        // Density ridges - the "clusters" of an embedding space.
        float ridge = abs(f2 - 0.5);
        float glow  = smoothstep(0.16, 0.0, ridge) * 0.55;

        // A second, finer ridge set at a different scale and phase. Two scales of
        // structure is what separates "a space" from "a texture" - the eye reads the
        // interference between them as depth rather than as a repeating pattern.
        float f3    = fbm(p * 7.3 + vec2(-t * 1.4, t * 1.1));
        float fine  = smoothstep(0.085, 0.0, abs(f3 - 0.5)) * 0.22;

        // Sparse bright points, suggesting individual documents. Three parallax layers:
        // each is a different depth, so scrolling separates them.
        float pt = 0.0;
        for (int L = 0; L < 3; L++) {
          float fl    = float(L);
          float scale = 9.0 + fl * 7.0;
          float depth = 1.0 - fl * 0.30;
          vec2  gp    = p * scale + uShift * fl * 1.6;
          vec2  cell  = floor(gp);
          float rnd   = hash(cell + fl * 37.0);
          vec2  cpos  = fract(gp) - 0.5
                      - 0.30 * vec2(sin(t * 6.0 + rnd * 30.0), cos(t * 5.0 + rnd * 21.0));
          pt += smoothstep(0.055 * depth, 0.0, length(cpos)) * step(0.955, rnd) * depth;
        }

        // Deep navy base -> teal ridges -> a warm accent only on the brightest points,
        // so the accent colour stays rare and therefore meaningful.
        vec3 base   = vec3(0.024, 0.043, 0.078);
        vec3 teal   = vec3(0.106, 0.427, 0.478);
        vec3 cyan   = vec3(0.180, 0.560, 0.620);
        vec3 accent = vec3(0.925, 0.510, 0.290);

        vec3 col = base;
        col = mix(col, teal, glow * (0.55 + 0.45 * f1));
        col += cyan * fine * (0.5 + 0.5 * f1);
        col += accent * pt * 0.85;

        // Vignette keeps the centre calm so overlaid text stays legible.
        float vig = smoothstep(1.25, 0.25, length(uv - 0.5) * 1.6);
        col *= vig;

        col = base + (col - base) * uIntensity;
        gl_FragColor = vec4(col, 1.0);
      }
    `;

    const compile = (type: number, src: string) => {
      const sh = gl.createShader(type)!;
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        console.error(gl.getShaderInfoLog(sh));
        return null;
      }
      return sh;
    };

    const vs = compile(gl.VERTEX_SHADER, vert);
    const fs = compile(gl.FRAGMENT_SHADER, frag);
    if (!vs || !fs) return;

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.useProgram(prog);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "aPos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    const uRes = gl.getUniformLocation(prog, "uRes");
    const uTime = gl.getUniformLocation(prog, "uTime");
    const uIntensity = gl.getUniformLocation(prog, "uIntensity");
    const uShift = gl.getUniformLocation(prog, "uShift");
    gl.uniform1f(uIntensity, intensity);

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const resize = () => {
      canvas.width = Math.floor(canvas.clientWidth * dpr);
      canvas.height = Math.floor(canvas.clientHeight * dpr);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.uniform2f(uRes, canvas.width, canvas.height);
    };
    resize();
    window.addEventListener("resize", resize);

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    let running = true;
    const start = performance.now();

    // Parallax targets, eased toward each frame so neither scroll nor cursor snaps.
    let targetX = 0, targetY = 0, curX = 0, curY = 0;
    const onScroll = () => { targetY = -(window.scrollY / Math.max(window.innerHeight, 1)) * 0.22; };
    const onPointer = (e: PointerEvent) => {
      targetX = (e.clientX / window.innerWidth - 0.5) * 0.06;
    };
    if (parallax && !reduced) {
      window.addEventListener("scroll", onScroll, { passive: true });
      window.addEventListener("pointermove", onPointer, { passive: true });
      onScroll();
    }

    const draw = () => {
      curX += (targetX - curX) * 0.05;
      curY += (targetY - curY) * 0.05;
      gl.uniform2f(uShift, curX, curY);
      gl.uniform1f(uTime, (performance.now() - start) / 1000);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      if (running && !reduced) raf = requestAnimationFrame(draw);
    };
    draw();

    // Stop work entirely when the tab is backgrounded.
    const onVisibility = () => {
      running = document.visibilityState === "visible";
      if (running && !reduced) {
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(draw);
      }
    };
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      running = false;
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("pointermove", onPointer);
      document.removeEventListener("visibilitychange", onVisibility);
      gl.deleteProgram(prog);
      gl.deleteBuffer(buf);
    };
  }, [intensity, parallax]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 -z-10 h-full w-full"
    />
  );
}
