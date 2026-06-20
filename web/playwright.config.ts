import { defineConfig } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// Committed, privacy-safe samples (the user voice recordings are gitignored).
const SAMPLES = path.resolve(__dirname, "public/samples");

// Most tests drive a deterministic file source (decoded WAV → AudioWorklet).
// The `mic` project additionally feeds a WAV into Chromium's fake microphone so
// we can exercise the real getUserMedia → AudioContext.resume() → worklet path
// (clicking the actual Start button = a trusted user gesture).
export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: "http://localhost:5173" },
  projects: [
    {
      name: "default",
      testIgnore: /mic\.spec\.ts/,
    },
    {
      name: "mic",
      testMatch: /mic\.spec\.ts/,
      use: {
        permissions: ["microphone"],
        launchOptions: {
          args: [
            "--use-fake-device-for-media-stream",
            "--use-fake-ui-for-media-stream",
            `--use-file-for-fake-audio-capture=${path.join(SAMPLES, "real_speech_48k.wav")}`,
          ],
        },
      },
    },
  ],
  webServer: {
    command: "npm run dev",
    url: "http://localhost:5173",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
