import { test, expect, type Page } from "@playwright/test";
import type { TrackDebugFrame } from "../src/audio/engine";

// Guards the F3 continuity fix: the tracker must FOLLOW a real F3 pole rather
// than abandon the slot. With transCap above the slot-drop cost, F3 was left
// empty ~30% of the time on connected speech even when an F3-band pole existed;
// capping it below that cost makes following always win.

type Eng = {
  setTrackDebug: (b: boolean) => void;
  startFromUrl: (u: string) => Promise<void>;
  stop: () => Promise<void>;
  trackDebugLog: TrackDebugFrame[];
};

async function collect(page: Page, fixture: string, ms: number): Promise<TrackDebugFrame[]> {
  await page.evaluate(() => (window as unknown as { __engine: Eng }).__engine.setTrackDebug(true));
  await page.evaluate((f) => (window as unknown as { __engine: Eng }).__engine.startFromUrl(`samples/${f}`), fixture);
  await page.waitForTimeout(ms);
  await page.evaluate(() => (window as unknown as { __engine: Eng }).__engine.stop());
  return page.evaluate(() => (window as unknown as { __engine: Eng }).__engine.trackDebugLog);
}

test("formant tracker follows F3 instead of dropping it", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => !!(window as unknown as { __engine?: unknown }).__engine);
  // Pre-warm the worker so debug applies to the first measured fixture.
  await page.evaluate(() => (window as unknown as { __engine: Eng }).__engine.startFromUrl("samples/silence.wav"));
  await page.waitForTimeout(150);
  await page.evaluate(() => (window as unknown as { __engine: Eng }).__engine.stop());

  const frames = await collect(page, "real_speech_48k.wav", 2500);
  expect(frames.length).toBeGreaterThan(60);

  const inBand = (v: number) => v >= 2000 && v <= 3200;
  const emptyPct = (100 * frames.filter((f) => f.committed[2] == null).length) / frames.length;
  // F3 dropped *while an F3-band pole was available* is the real bug.
  const emptyWithPole = frames.filter((f) => f.committed[2] == null && f.cand.some((p) => inBand(p[0]))).length;

  console.log(`[tracker] F3 empty=${emptyPct.toFixed(0)}% emptyWithPole=${emptyWithPole}/${frames.length}`);
  expect(emptyPct).toBeLessThan(15); // was ~30% before the fix
  expect(emptyWithPole).toBeLessThan(12);
});
