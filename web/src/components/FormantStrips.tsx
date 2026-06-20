import { useEffect, useRef, type MutableRefObject } from "react";
import { C } from "../theme";

export type Sample = { t: number; f2: number; f3: number };
type SamplesRef = MutableRefObject<Sample[]>;
type ClockRef = MutableRefObject<{ audioT: number; wall: number }>;

interface StripProps {
  title: string;
  samplesRef: SamplesRef;
  clockRef: ClockRef;
  pick: (s: Sample) => number;
  yMax: number;
  color: string;
  windowSec: number; // visible time window (right edge = now, left = now - windowSec)
}

// Imperative 60fps render loop driven by the audio clock: the right edge is the
// current audio time (extrapolated from the last result by wall-clock), and each
// sample is placed at its own timestamp. Bursty arrival just fills samples at
// their true positions, so the line never lurches sideways.
function Strip({ title, samplesRef, clockRef, pick, yMax, color, windowSec }: StripProps) {
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

      // gridlines + axis numbers
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

      // Right edge = audio anchor + wall-clock elapsed. The anchor (set once in
      // App) only ever moves forward, so `now` is monotonic by construction — no
      // backlog burst can drag it. The window is [now - windowSec, now].
      const clk = clockRef.current;
      const now = clk.wall === 0 ? 0 : clk.audioT + (performance.now() - clk.wall) / 1000;
      const left = now - windowSec;

      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      const data = samplesRef.current;
      let started = false;
      ctx.beginPath();
      for (let i = 0; i < data.length; i++) {
        const s = data[i];
        if (s.t < left || s.t > now) {
          started = false;
          continue;
        }
        const v = pick(s);
        if (!Number.isFinite(v)) {
          started = false;
          continue;
        }
        const x = ((s.t - left) / windowSec) * w;
        const y = h - (Math.min(v, yMax) / yMax) * h;
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      ctx.fillStyle = C.text;
      ctx.font = "bold 13px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(title, w / 2, 13);

      // Instrumentation for smoothness tests (off unless a test enables it).
      const inst = (window as unknown as { __instrument?: boolean }).__instrument;
      if (inst && title === "F2 (Hz) / time") {
        const log = ((window as unknown as { __stripDraw?: { t: number; now: number }[] }).__stripDraw ??= []);
        log.push({ t: performance.now(), now });
      }
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [title, yMax, color, windowSec, pick, samplesRef, clockRef]);

  return (
    <canvas
      ref={ref}
      data-testid={`strip-${title}`}
      style={{ width: "100%", height: 90, display: "block", border: `1px solid ${C.border}`, borderRadius: 4 }}
    />
  );
}

export function FormantStrips({
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
        title="F2 (Hz) / time"
        samplesRef={samplesRef}
        clockRef={clockRef}
        pick={(s) => s.f2}
        yMax={5000}
        color={C.f2}
        windowSec={windowSec}
      />
      <Strip
        title="F3 (Hz) / time"
        samplesRef={samplesRef}
        clockRef={clockRef}
        pick={(s) => s.f3}
        yMax={7000}
        color={C.f3}
        windowSec={windowSec}
      />
    </div>
  );
}
