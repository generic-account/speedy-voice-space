import { test, expect, type Page } from "@playwright/test";

// Real mic-path test: clicks the actual Start button (a trusted user gesture),
// which runs getUserMedia → AudioContext.resume() → AudioWorklet on Chromium's
// fake microphone. Guards the regression where the context stayed suspended
// after the permission dialog and no audio ever flowed.

interface AudioStats {
  running: boolean;
  source: string;
  blocks: number;
  peakRms: number;
  sampleRate: number;
  ctxState: string;
}

function readStats(page: Page): Promise<AudioStats> {
  return page.evaluate(
    () => (window as unknown as { __audioStats?: AudioStats }).__audioStats!,
  );
}

test("clicking Start starts the live mic feed (context running, blocks flow)", async ({
  page,
}) => {
  await page.goto("/");
  await page.waitForFunction(
    () => !!(window as unknown as { __engine?: unknown }).__engine,
  );

  await page.getByTestId("start").click();
  await expect(page.getByTestId("status")).toContainText("running");

  // The context must end up running (not suspended) and blocks must arrive.
  await page.waitForFunction(
    () => {
      const s = (window as unknown as { __audioStats?: AudioStats })
        .__audioStats;
      return !!s && s.ctxState === "running" && s.blocks > 20;
    },
    { timeout: 10_000 },
  );

  const s = await readStats(page);
  expect(s.source).toBe("mic");
  expect(s.ctxState).toBe("running");
  expect(s.blocks).toBeGreaterThan(20);
  expect(s.peakRms).toBeGreaterThan(0.01); // real speech fixture has energy

  await expect(page.getByTestId("ctx-state")).toHaveText("running");
});
