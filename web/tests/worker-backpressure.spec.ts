import { test, expect } from "@playwright/test";

// Replicates the "UI gets laggier over time on slower machines" bug. Capture runs
// in real time (~21 ms blocks); if the DSP worker is slower than that, naive
// per-block posting queues blocks without bound, so analysis falls further and
// further behind real time (strips drift left, readouts go stale). Backpressure
// keeps at most one block in flight and processes the freshest audio, so the
// backlog stays bounded no matter how long it runs or how slow the device is.

type Eng = {
  startFromUrl: (u: string) => Promise<void>;
  stop: () => Promise<void>;
  setProcDelay: (ms: number) => void;
  backlogS: number;
};
declare global {
  interface Window {
    __engine: Eng;
  }
}

test("analysis backlog stays bounded when the DSP is slower than real time", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => !!(window as unknown as { __engine?: unknown }).__engine);

  await page.evaluate(() => window.__engine.startFromUrl("samples/real_speech_48k.wav"));
  // Emulate a weak device: ~45 ms/block DSP, roughly 2x the ~21 ms block period.
  await page.evaluate(() => window.__engine.setProcDelay(45));

  const series: number[] = [];
  for (let i = 0; i < 6; i++) {
    await page.waitForTimeout(500);
    series.push(await page.evaluate(() => window.__engine.backlogS));
  }
  await page.evaluate(() => window.__engine.setProcDelay(0));
  await page.evaluate(() => window.__engine.stop());

  console.log(`[backpressure] backlog (s) over 3s: ${series.map((x) => x.toFixed(2)).join(", ")}`);

  // Bounded to a couple block-periods of in-flight work...
  expect(Math.max(...series)).toBeLessThan(0.4);
  // ...and not trending upward over runtime (the hallmark of the unbounded-queue bug).
  expect(series[series.length - 1]).toBeLessThan(series[0] + 0.25);
});
