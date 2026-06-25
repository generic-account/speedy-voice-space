import { useEffect, useRef, type MutableRefObject } from "react";
import { C } from "../theme";

export type Sample = { t: number; f2: number; f3: number; pitch: number };
type SamplesRef = MutableRefObject<Sample[]>;
type ClockRef = MutableRefObject<{ audioT: number; wall: number }>;
type Series = { pick: (s: Sample) => number; color: string; label: string };

// Stable module-level series so the draw effect mounts once (not per render).
const FORMANT_SERIES: Series[] = [
  { pick: (s) => s.f2, color: C.f2, label: "F2" },
  { pick: (s) => s.f3, color: C.f3, label: "F3" },
];
const PITCH_SERIES: Series[] = [{ pick: (s) => s.pitch, color: C.accent, label: "Pitch" }];

interface StripProps {
  title: string;
  samplesRef: SamplesRef;
  clockRef: ClockRef;
  series: Series[];
  yMax: number;
  windowSec: number;
  instrument?: boolean; // push __stripDraw for the smoothness/backlog tests
}

// Small legend chip in the top-right, on an opaque background so it reads cleanly
// over the scrolling lines without hiding more than its own corner.
function drawLegend(ctx: CanvasRenderingContext2D, w: number, series: Series[]) {
  ctx.font = "10px system-ui, sans-serif";
  ctx.textBaseline = "alphabetic";
  const itemW = (s: Series) => 12 + 3 + ctx.measureText(s.label).width + 10;
  const total = series.reduce((a, s) => a + itemW(s), 0);
  let x = w - total - 4;
  const y = 4;
  ctx.fillStyle = C.canvas;
  ctx.fillRect(x - 4, y, total + 6, 15);
  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, y + 8);
    ctx.lineTo(x + 12, y + 8);
    ctx.stroke();
    ctx.fillStyle = C.label;
    ctx.textAlign = "left";
    ctx.fillText(s.label, x + 15, y + 11);
    x += itemW(s);
  }
}

// Clock-driven scrolling line plot. Right edge = the audio-clock "now"; samples
// are placed at their own timestamp (see App's clock anchor), so bursty arrival
// never lurches the line.
function Strip({ title, samplesRef, clockRef, series, yMax, windowSec, instrument }: StripProps) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;

    const draw = () => {
      raf = requestAnimationFrame(draw);
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (w === 0 || h === 0) return;
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = C.canvas;
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = C.grid;
      ctx.fillStyle = C.muted;
      ctx.font = "10px system-ui, sans-serif";
      ctx.textAlign = "left";
      for (let i = 1; i < 4; i++) {
        const y = (i / 4) * h;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
        ctx.fillText(String(Math.round(yMax * (1 - i / 4))), 2, y - 1);
      }

      const clk = clockRef.current;
      const now = clk.wall === 0 ? 0 : clk.audioT + (performance.now() - clk.wall) / 1000;
      const left = now - windowSec;
      const data = samplesRef.current;

      ctx.lineWidth = 1.5;
      for (const s of series) {
        ctx.strokeStyle = s.color;
        let started = false;
        ctx.beginPath();
        for (let i = 0; i < data.length; i++) {
          const d = data[i];
          if (d.t < left || d.t > now) {
            started = false;
            continue;
          }
          const v = s.pick(d);
          if (!Number.isFinite(v)) {
            started = false;
            continue;
          }
          const x = ((d.t - left) / windowSec) * w;
          const y = h - (Math.min(v, yMax) / yMax) * h;
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      }

      ctx.fillStyle = C.text;
      ctx.font = "bold 13px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(title, w / 2, 13);

      if (series.length > 1) drawLegend(ctx, w, series);

      if (instrument) {
        const win = window as unknown as { __instrument?: boolean; __stripDraw?: { t: number; now: number }[] };
        if (win.__instrument) (win.__stripDraw ??= []).push({ t: performance.now(), now });
      }
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [title, yMax, windowSec, series, samplesRef, clockRef, instrument]);

  return (
    <canvas
      ref={ref}
      data-testid={`strip-${title}`}
      style={{ width: "100%", height: 90, display: "block", border: `1px solid ${C.border}`, borderRadius: 4 }}
    />
  );
}

export function Strips({
  samplesRef,
  clockRef,
  windowSec,
}: {
  samplesRef: SamplesRef;
  clockRef: ClockRef;
  windowSec: number;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <Strip
        title="Pitch (Hz) / time"
        samplesRef={samplesRef}
        clockRef={clockRef}
        series={PITCH_SERIES}
        yMax={500}
        windowSec={windowSec}
      />
      <Strip
        title="Formants (Hz) / time"
        samplesRef={samplesRef}
        clockRef={clockRef}
        series={FORMANT_SERIES}
        yMax={5000}
        windowSec={windowSec}
        instrument
      />
    </div>
  );
}
