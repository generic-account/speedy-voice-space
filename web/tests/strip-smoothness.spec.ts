import { test, expect, type Page } from "@playwright/test";

// Steady-state strip health on real playback: the RAF render loop runs near
// 60fps with no stalls, and the displayed F2/F3 are steady (low vertical jitter)
// on a sustained vowel. The scroll clock's smoothness under the hard case (a
// drained backlog) is covered separately by strip-backlog.spec.ts.

interface Stats {
  count: number;
  mean: number;
  p95: number;
  max: number;
}

function stats(xs: number[]): Stats {
  const s = [...xs].sort((a, b) => a - b);
  const at = (q: number) => s[Math.min(s.length - 1, Math.floor(q * s.length))];
  return {
    count: s.length,
    mean: s.reduce((a, b) => a + b, 0) / s.length,
    p95: at(0.95),
    max: s[s.length - 1],
  };
}

function intervals(ts: number[]): number[] {
  const d: number[] = [];
  for (let i = 1; i < ts.length; i++) d.push(ts[i] - ts[i - 1]);
  return d;
}

// Median of |Δ| between consecutive finite values — frame-to-frame "jitter".
function medianDelta(vals: (number | null)[]): number {
  const ds: number[] = [];
  for (let i = 1; i < vals.length; i++) {
    const a = vals[i - 1];
    const b = vals[i];
    if (a != null && b != null) ds.push(Math.abs(b - a));
  }
  if (!ds.length) return 0;
  ds.sort((x, y) => x - y);
  return ds[Math.floor(ds.length / 2)];
}

async function measure(page: Page, fixture: string, ms = 3000) {
  await page.goto("/");
  await page.waitForFunction(() => !!(window as unknown as { __engine?: unknown }).__engine);
  await page.evaluate(() => {
    (window as unknown as { __instrument: boolean }).__instrument = true;
  });
  await page.evaluate(
    (f) =>
      (window as unknown as { __engine: { startFromUrl: (u: string) => Promise<void> } }).__engine.startFromUrl(`samples/${f}`),
    fixture,
  );
  await page.waitForTimeout(600); // settle
  await page.evaluate(() => {
    (window as unknown as { __stripDraw: unknown[]; __resultLog: unknown[] }).__stripDraw = [];
    (window as unknown as { __stripDraw: unknown[]; __resultLog: unknown[] }).__resultLog = [];
  });
  await page.waitForTimeout(ms);
  return page.evaluate(() => {
    const w = window as unknown as {
      __stripDraw: { t: number; now: number }[];
      __resultLog: { t: number; voiced: boolean; f2: number | null; f3: number | null }[];
    };
    return { draws: w.__stripDraw.map((d) => d.t), results: w.__resultLog };
  });
}

test("strip render loop is healthy and displayed F2/F3 are steady on a vowel", async ({ page }) => {
  const { draws, results } = await measure(page, "vowel_a_150hz.wav");

  const renderInt = stats(intervals(draws.length ? draws : [0]));
  const voiced = results.filter((r) => r.voiced);
  const jitterF2 = medianDelta(voiced.map((r) => r.f2));
  const jitterF3 = medianDelta(voiced.map((r) => r.f3));

  console.log(
    `[smoothness] renders=${renderInt.count} interval mean=${renderInt.mean.toFixed(1)} p95=${renderInt.p95.toFixed(1)} max=${renderInt.max.toFixed(1)}ms | ` +
      `jitter F2=${jitterF2.toFixed(0)}Hz F3=${jitterF3.toFixed(0)}Hz (n=${voiced.length})`,
  );

  // Renders run at ~60fps with no big stalls.
  expect(renderInt.count).toBeGreaterThan(120); // ~3s * ~50fps headless floor
  expect(renderInt.p95).toBeLessThan(40);
  expect(renderInt.max).toBeLessThan(120);

  // On a sustained vowel the displayed F2/F3 should be steady (low jitter).
  expect(jitterF2).toBeLessThan(60);
  expect(jitterF3).toBeLessThan(90);
});
