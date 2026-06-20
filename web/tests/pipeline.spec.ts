import { test, expect, type Page } from "@playwright/test";

interface AudioStats {
  running: boolean;
  source: string;
  blocks: number;
  lastRms: number;
  peakRms: number;
  sampleRate: number;
}

function readStats(page: Page): Promise<AudioStats> {
  return page.evaluate(
    () => (window as unknown as { __audioStats?: AudioStats }).__audioStats!,
  );
}

async function playAndSettle(page: Page, file: string, minBlocks = 40) {
  await page.evaluate(
    (f) =>
      (
        window as unknown as {
          __engine: { startFromUrl: (u: string) => Promise<void> };
        }
      ).__engine.startFromUrl(`samples/${f}`),
    file,
  );
  await page.waitForFunction(
    (n) => {
      const s = (window as unknown as { __audioStats?: AudioStats })
        .__audioStats;
      return !!s && s.running && s.blocks > n;
    },
    minBlocks,
    { timeout: 15_000 },
  );
}

// Energetic fixtures should register signal; silence should not.
const SIGNAL_FILES = [
  "tone_220hz.wav",
  "vowel_a_150hz.wav",
  "real_speech_48k.wav",
];

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(
    () => !!(window as unknown as { __engine?: unknown }).__engine,
  );
});

for (const file of SIGNAL_FILES) {
  test(`capture pipeline: ${file} delivers blocks at 48kHz with signal`, async ({
    page,
  }) => {
    await playAndSettle(page, file);
    const s = await readStats(page);
    expect(s.running).toBe(true);
    expect(s.source).toBe("file");
    expect(s.sampleRate).toBe(48000);
    expect(s.blocks).toBeGreaterThan(40);
    expect(s.peakRms).toBeGreaterThan(0.01);
  });
}

test("capture pipeline: silence delivers blocks but ~zero energy", async ({
  page,
}) => {
  await playAndSettle(page, "silence.wav");
  const s = await readStats(page);
  expect(s.blocks).toBeGreaterThan(40);
  expect(s.peakRms).toBeLessThan(0.005);
});

test("level meter reacts in the DOM to a played tone", async ({ page }) => {
  await playAndSettle(page, "tone_220hz.wav");
  // The rms readout in the DOM should be clearly non-zero.
  // The meter's data-rms is fed by engine stats via requestAnimationFrame,
  // proving blocks flow all the way to the React render layer.
  const rmsAttr = await page.getByTestId("level-meter").getAttribute("data-rms");
  expect(Number(rmsAttr)).toBeGreaterThan(0.05);
  expect((await readStats(page)).running).toBe(true);
});
