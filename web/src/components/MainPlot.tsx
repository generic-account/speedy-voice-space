import { useEffect, useRef, useState } from "react";
import { C } from "../theme";

export interface TrailPoint {
  pitch: number;
  resonance: number;
}

interface Props {
  trail: TrailPoint[];
  xRange: [number, number];
  yRange: [number, number];
}

// Plot insets (px): left/bottom hold axis labels + titles; right/top are margins.
// Shared by the renderer and the zoom/pan math so they can't drift apart.
const PAD = { l: 58, r: 28, t: 24, b: 32 };

// Pitch (x) vs Resonance (y) scatter with a fading trail. The configured ranges
// are the "home" view; the wheel zooms, drag pans, and double-click resets.
export function MainPlot({ trail, xRange, yRange }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [view, setView] = useState<{ x: [number, number]; y: [number, number] }>({
    x: xRange,
    y: yRange,
  });
  const [resizeTick, setResizeTick] = useState(0);

  // Re-home when the configured range VALUES change (not on every render — the
  // parent passes fresh array literals, so depend on the numbers themselves).
  useEffect(() => {
    setView({ x: [xRange[0], xRange[1]], y: [yRange[0], yRange[1]] });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [xRange[0], xRange[1], yRange[0], yRange[1]]);

  // Redraw when the canvas box changes size (window/layout resize).
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setResizeTick((t) => t + 1));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const { l: padL, r: padR, t: padT, b: padB } = PAD;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    const [x0, x1] = view.x;
    const [y0, y1] = view.y;
    const sx = (p: number) => padL + ((p - x0) / (x1 - x0)) * plotW;
    const sy = (r: number) => padT + (1 - (r - y0) / (y1 - y0)) * plotH;

    ctx.fillStyle = C.canvas;
    ctx.fillRect(padL, padT, plotW, plotH);
    ctx.strokeStyle = C.grid;
    ctx.fillStyle = C.muted;
    ctx.font = "10px system-ui, sans-serif"; // axis numbers

    // X gridlines at "nice" pitch intervals.
    const xStep = niceStep(x1 - x0, 6);
    ctx.textAlign = "center";
    for (let p = Math.ceil(x0 / xStep) * xStep; p <= x1; p += xStep) {
      const X = sx(p);
      ctx.beginPath();
      ctx.moveTo(X, padT);
      ctx.lineTo(X, padT + plotH);
      ctx.stroke();
      ctx.fillText(String(Math.round(p)), X, padT + plotH + 16);
    }
    // Y gridlines.
    const yStep = niceStep(y1 - y0, 5);
    ctx.textAlign = "right";
    for (let r = Math.ceil(y0 / yStep) * yStep; r <= y1 + 1e-9; r += yStep) {
      const Y = sy(r);
      ctx.beginPath();
      ctx.moveTo(padL, Y);
      ctx.lineTo(padL + plotW, Y);
      ctx.stroke();
      ctx.fillText(r.toFixed(2), padL - 6, Y + 4);
    }

    ctx.fillStyle = C.text;
    ctx.font = "11px system-ui, sans-serif"; // axis titles
    ctx.textAlign = "center";
    ctx.fillText("Pitch (Hz)", padL + plotW / 2, h - 6);
    ctx.save();
    ctx.translate(12, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Resonance", 0, 0);
    ctx.restore();

    // Trail (older = lighter), then current point. Clip to the plot rect.
    ctx.save();
    ctx.beginPath();
    ctx.rect(padL, padT, plotW, plotH);
    ctx.clip();
    const n = trail.length;
    for (let i = 0; i < n; i++) {
      const a = (40 + 180 * ((i + 1) / n)) / 255;
      ctx.fillStyle = `rgba(${C.trailRGB},${a})`;
      ctx.beginPath();
      ctx.arc(sx(trail[i].pitch), sy(trail[i].resonance), 3, 0, Math.PI * 2);
      ctx.fill();
    }
    if (n > 0) {
      const last = trail[n - 1];
      ctx.fillStyle = C.accent;
      ctx.beginPath();
      ctx.arc(sx(last.pitch), sy(last.resonance), 6, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();

    ctx.fillStyle = C.text;
    ctx.font = "bold 13px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Pitch vs Resonance", padL + plotW / 2, 15);
  }, [trail, view, resizeTick]);

  // Wheel zoom about the cursor; drag to pan; double-click to reset.
  function onWheel(e: React.WheelEvent) {
    const rect = ref.current!.getBoundingClientRect();
    const fx = (e.clientX - rect.left - PAD.l) / (rect.width - PAD.l - PAD.r);
    const fy = 1 - (e.clientY - rect.top - PAD.t) / (rect.height - PAD.t - PAD.b);
    const k = Math.exp(e.deltaY * 0.0015); // <1 zoom in, >1 zoom out
    setView((v) => ({
      x: zoomAxis(v.x, clamp01(fx), k),
      y: zoomAxis(v.y, clamp01(fy), k),
    }));
  }

  const drag = useRef<{ x: number; y: number } | null>(null);
  function onPointerDown(e: React.PointerEvent) {
    drag.current = { x: e.clientX, y: e.clientY };
    (e.target as Element).setPointerCapture(e.pointerId);
  }
  function onPointerMove(e: React.PointerEvent) {
    if (!drag.current) return;
    const rect = ref.current!.getBoundingClientRect();
    const dx = ((e.clientX - drag.current.x) / (rect.width - PAD.l - PAD.r)) * (view.x[1] - view.x[0]);
    const dy = ((e.clientY - drag.current.y) / (rect.height - PAD.t - PAD.b)) * (view.y[1] - view.y[0]);
    drag.current = { x: e.clientX, y: e.clientY };
    setView((v) => ({
      x: [v.x[0] - dx, v.x[1] - dx],
      y: [v.y[0] + dy, v.y[1] + dy],
    }));
  }
  function onPointerUp() {
    drag.current = null;
  }

  return (
    <canvas
      ref={ref}
      data-testid="main-plot"
      onWheel={onWheel}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onDoubleClick={() => setView({ x: xRange, y: yRange })}
      title="scroll to zoom · drag to pan · double-click to reset"
      style={{ width: "100%", height: "100%", display: "block", cursor: "crosshair" }}
    />
  );
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

function zoomAxis([lo, hi]: [number, number], frac: number, k: number): [number, number] {
  const center = lo + (hi - lo) * frac;
  const nlo = center - (center - lo) * k;
  const nhi = center + (hi - center) * k;
  return nhi - nlo < 1e-6 ? [lo, hi] : [nlo, nhi];
}

// A "nice" axis step (~1/2/5 × 10ⁿ) giving roughly `target` divisions.
function niceStep(span: number, target: number): number {
  const raw = span / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
  return step * mag;
}
