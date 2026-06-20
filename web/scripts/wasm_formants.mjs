// Dump OUR (built wasm) formants per analysis window for a WAV → CSV on stdout.
// Mirrors the worker's analysis window (buffer_duration 0.06s, hop 0.02s) so it
// reflects exactly what the app computes. Used by the formant experiments.
//   node scripts/wasm_formants.mjs <wav>
import { initSync, FormantAnalyzer, PitchAnalyzer } from "../src/dsp/dsp.js";
import fs from "node:fs";

const wavPath = process.argv[2];
initSync({ module: fs.readFileSync(new URL("../src/dsp/dsp_bg.wasm", import.meta.url)) });

function readWav(path) {
  const buf = fs.readFileSync(path);
  const sr = buf.readUInt32LE(24);
  let off = 12;
  while (off < buf.length) {
    const id = buf.toString("ascii", off, off + 4);
    const size = buf.readUInt32LE(off + 4);
    if (id === "data") {
      const n = size / 2;
      const x = new Float32Array(n);
      for (let i = 0; i < n; i++) x[i] = buf.readInt16LE(off + 8 + i * 2) / 32768;
      return { x, sr };
    }
    off += 8 + size;
  }
  throw new Error("no data chunk");
}

const winLen = process.argv[3] ? Number(process.argv[3]) : 0.025;
const bufDur = process.argv[4] ? Number(process.argv[4]) : 0.06;

const { x, sr } = readWav(wavPath);
const fa = new FormantAnalyzer(sr);
fa.setMaxNumberOfFormants(5);
fa.setMaximumFormant(5500);
fa.setWindowLength(winLen);
fa.setPreEmphasisFrom(50);
const pitch = new PitchAnalyzer(sr);
pitch.setRange(75, 400);

const win = Math.round(bufDur * sr);
const hop = Math.round(0.02 * sr);
// Wide CSV: f0, then up to 6 candidate poles as freq/bw pairs (cand{i}f,cand{i}b).
const NC = 6;
const header = ["t", "f0"];
for (let i = 1; i <= NC; i++) header.push(`c${i}f`, `c${i}b`);
const out = [header.join(",")];
for (let i = 0; i + win <= x.length; i += hop) {
  const frame = x.subarray(i, i + win);
  const det = Array.from(fa.analyzeDetailed(frame)); // [f,bw,f,bw,...]
  const f0 = pitch.analyze(frame);
  const t = (i + win / 2) / sr;
  const row = [t.toFixed(4), Number.isFinite(f0) ? f0.toFixed(1) : ""];
  for (let c = 0; c < NC; c++) {
    row.push(det[c * 2] ?? "", det[c * 2 + 1] ?? "");
  }
  out.push(row.join(","));
}
process.stdout.write(out.join("\n") + "\n");
