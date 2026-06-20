import { test, expect, type Page } from "@playwright/test";

// Regression for the "leftward snap when I stop a sustained sound" bug.
//
// Root cause: the worker stamps results as it *processes* them. A sustained vowel
// is expensive (LPC) so the worker falls behind real time, building a backlog;
// result.t advances slower than the wall clock. When voicing stops, analysis goes
// cheap and the worker drains the backlog in a burst — result.t lurches forward by
// the accumulated backlog L. If the strip's right edge is snapped to the latest
// result.t, the whole window jumps left by L (bigger the longer you held).
//
// The old smoothness test never caught this because continuous file playback keeps
// the worker caught up, so result.t never had that discontinuity. Here we inject
// synthetic results that reproduce the backlog-then-flush timeline directly and
// assert the strip's right-edge clock keeps advancing at ~real time with no jump.

type Draw = { t: number; now: number };

async function runBacklogTimeline(page: Page): Promise<Draw[]> {
  await page.goto("/");
  await page.waitForFunction(() => !!(window as unknown as { __engine?: unknown }).__engine);
  await page.evaluate(() => {
    (window as unknown as { __instrument: boolean }).__instrument = true;
  });

  return page.evaluate(async () => {
    type Res = {
      voiced: boolean;
      rms: number;
      pitchHz: number | null;
      formantsHz: number[];
      t: number;
    };
    const eng = (window as unknown as { __engine: { onResult: ((r: Res) => void) | null } }).__engine;
    const emit = (t: number, voiced: boolean) =>
      eng.onResult?.({
        voiced,
        rms: voiced ? 0.05 : 0.0001,
        pitchHz: voiced ? 150 : null,
        formantsHz: voiced ? [700, 1200, 2600] : [],
        t,
      });
    const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

    (window as unknown as { __stripDraw: Draw[] }).__stripDraw = [];
    const t0 = performance.now();
    const real = () => (performance.now() - t0) / 1000;

    // Phase A — steady, real-time voiced frames (~21 ms blocks). Anchors the clock.
    while (real() < 1.2) {
      emit(real(), true);
      await sleep(21);
    }
    // Phase B — sustained vowel: worker falls behind, result.t advances at ~0.4x
    // real time, so a backlog (real - t) accumulates. t never exceeds real capture.
    let t = real();
    while (real() < 2.6) {
      t = Math.min(t + 0.021 * 0.4, real());
      emit(t, true);
      await sleep(21);
    }
    // Phase C — stop: analysis goes cheap, the backlog drains as a fast burst that
    // catches result.t up to real time within a few ms.
    const target = real();
    while (t < target - 0.001) {
      t = Math.min(t + 0.05, target);
      emit(t, t < target - 0.05);
      await sleep(2);
    }
    await sleep(300); // let the strips keep scrolling after the flush

    return (window as unknown as { __stripDraw: Draw[] }).__stripDraw;
  });
}

test("strip right edge never lurches when a sustained-sound backlog drains", async ({ page }) => {
  const draws = await runBacklogTimeline(page);
  expect(draws.length).toBeGreaterThan(120);

  let backsteps = 0;
  let worstBack = 0; // most-negative Δnow, seconds
  let maxJump = 0; // largest forward Δnow in one frame, seconds
  const vels: number[] = [];
  for (let i = 1; i < draws.length; i++) {
    const dNow = draws[i].now - draws[i - 1].now;
    const dWall = (draws[i].t - draws[i - 1].t) / 1000;
    if (dNow < 0) {
      backsteps++;
      worstBack = Math.min(worstBack, dNow);
    }
    if (dNow > maxJump) maxJump = dNow;
    if (dWall > 0) vels.push(dNow / dWall);
  }
  const mean = vels.reduce((a, b) => a + b, 0) / Math.max(1, vels.length);
  const std = Math.sqrt(vels.reduce((a, v) => a + (v - mean) ** 2, 0) / Math.max(1, vels.length));

  console.log(
    `[backlog] draws=${draws.length} vel mean=${mean.toFixed(3)} std=${std.toFixed(3)} | ` +
      `backsteps=${backsteps} worst=${(worstBack * 1000).toFixed(1)}ms | maxFrameJump=${(maxJump * 1000).toFixed(1)}ms`,
  );

  // The right edge advances at ~real time, the whole way through the flush.
  expect(mean).toBeGreaterThan(0.85);
  expect(mean).toBeLessThan(1.15);
  // It never steps backwards (no leftward snap).
  expect(backsteps).toBe(0);
  // No single frame jumps far — the backlog drain doesn't drag the window. Even at
  // 60 fps a real-time frame is ~16 ms; allow generous headroom but well under the
  // ~1 s backlog the old code would have snapped.
  expect(maxJump).toBeLessThan(0.1);
});
