import { test, expect, type Page } from "@playwright/test";

// End-to-end WASM link test: drives a fixture through the real chain
//   file source → AudioWorklet → DSP Web Worker (wasm pitch) → processor → UI.
// This guards against the integration breaking as we build up — i.e. the Rust
// compiled to wasm must actually load in the worker and produce results.

interface Display {
  voiced: boolean;
  filteredPitchHz: number | null;
  rawPitchHz: number | null;
  rawF2Hz: number | null;
  rawF3Hz: number | null;
  formantsHz: number[];
  filteredResonance: number | null;
}

function lastDisplay(page: Page): Promise<Display | undefined> {
  return page.evaluate(
    () => (window as unknown as { __lastDisplay?: Display }).__lastDisplay,
  );
}

async function play(page: Page, file: string) {
  await page.evaluate(
    (f) =>
      (
        window as unknown as {
          __engine: { startFromUrl: (u: string) => Promise<void> };
        }
      ).__engine.startFromUrl(`samples/${f}`),
    file,
  );
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(
    () => !!(window as unknown as { __engine?: unknown }).__engine,
  );
});

async function collectPitch(page: Page, file: string, n = 25): Promise<number[]> {
  await play(page, file);
  const samples: number[] = [];
  // Poll the smoothed pitch as the fixture streams through the wasm worker.
  for (let i = 0; i < 80 && samples.length < n; i++) {
    await page.waitForTimeout(50);
    const d = await lastDisplay(page);
    if (d && d.filteredPitchHz && Number.isFinite(d.filteredPitchHz)) {
      samples.push(d.filteredPitchHz);
    }
  }
  return samples;
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

test("wasm pitch in worker tracks a 220 Hz tone", async ({ page }) => {
  const samples = await collectPitch(page, "tone_220hz.wav");
  expect(samples.length).toBeGreaterThan(10);
  const m = median(samples);
  expect(m).toBeGreaterThan(212);
  expect(m).toBeLessThan(228);
});

test("wasm pitch in worker voices a sung vowel", async ({ page }) => {
  const samples = await collectPitch(page, "vowel_a_150hz.wav");
  expect(samples.length).toBeGreaterThan(10);
  const m = median(samples);
  // Synthetic /a/ at 150 Hz f0 (Praat reads ~172 due to harmonics); just assert
  // it's a plausible voiced pitch, proving the wasm path runs end to end.
  expect(m).toBeGreaterThan(120);
  expect(m).toBeLessThan(260);
});

test("wasm worker produces formants + resonance for a vowel", async ({ page }) => {
  await play(page, "vowel_a_150hz.wav");
  // Wait until formants + a resonance score have been computed.
  await page.waitForFunction(
    () => {
      const d = (window as unknown as { __lastDisplay?: Display }).__lastDisplay;
      return !!d && d.formantsHz.length >= 3 && d.filteredResonance !== null;
    },
    { timeout: 10_000 },
  );
  const d = (await lastDisplay(page))!;
  // Synthetic /a/: F1~700, F2~1200 (the oracle/Praat targets). Allow slack.
  expect(d.rawF2Hz).not.toBeNull();
  expect(d.rawF2Hz!).toBeGreaterThan(900);
  expect(d.rawF2Hz!).toBeLessThan(1600);
  expect(d.filteredResonance!).toBeGreaterThanOrEqual(0);
  expect(d.filteredResonance!).toBeLessThanOrEqual(1);
});

test("noise suppression toggle keeps the pipeline producing voiced speech", async ({
  page,
}) => {
  // Enable noise suppression via the engine config, then play speech and
  // confirm the wasm denoiser path runs end-to-end without breaking analysis.
  await page.evaluate(() => {
    const eng = (
      window as unknown as {
        __engine: {
          updateConfig: (c: unknown) => void;
          config: Record<string, unknown>;
        };
      }
    ).__engine;
    eng.updateConfig({
      ...eng.config,
      noiseSuppressionEnabled: true,
      noiseSuppressionMix: 1.0,
    });
  });
  await play(page, "real_speech_48k.wav");
  await page.waitForFunction(
    () => {
      const d = (window as unknown as { __lastDisplay?: Display }).__lastDisplay;
      return !!d && d.rawPitchHz !== null;
    },
    { timeout: 10_000 },
  );
  const d = (await lastDisplay(page))!;
  expect(d.voiced).toBe(true);
  expect(d.rawPitchHz!).toBeGreaterThan(60);
});

test("wasm worker reports silence as unvoiced", async ({ page }) => {
  await play(page, "silence.wav");
  await page.waitForTimeout(600);
  const d = await lastDisplay(page);
  expect(d).toBeTruthy();
  expect(d!.voiced).toBe(false);
  expect(d!.filteredPitchHz).toBeNull();
});
