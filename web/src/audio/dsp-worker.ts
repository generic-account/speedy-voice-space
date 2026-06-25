// DSP worker: owns the wasm analyzers, keeps a rolling buffer of
// bufferDurationS, gates on RMS, and emits an AnalysisResult per block.
import init, { PitchAnalyzer, FormantAnalyzer, Denoiser } from "../dsp/dsp.js";
import wasmUrl from "../dsp/dsp_bg.wasm?url";
import type { AnalysisConfig, AnalysisResult } from "../processing/defaults";
import { FormantTracker, type Pole } from "../processing/formantTracker";

let pitch: PitchAnalyzer | null = null;
let formant: FormantAnalyzer | null = null;
let denoiser: Denoiser | null = null;
const tracker = new FormantTracker();
let lastFormants: number[] = []; // coasted F1..F3 across momentarily-empty slots
let ready = false;

let config: AnalysisConfig | null = null;
let buffer = new Float32Array(2880);
let filled = 0;
let captureFrame = 0; // capture position of the latest block (from the audio clock)
let trackDebug = false; // when on, emit per-frame formant-tracker diagnostics
let procDelayMs = 0; // test/diagnostic: simulate slow per-block processing

const post = (msg: unknown) => (self as unknown as Worker).postMessage(msg);

function applyConfig(cfg: AnalysisConfig) {
  config = cfg;
  buffer = new Float32Array(Math.max(32, Math.round(cfg.bufferDurationS * cfg.samplerate)));
  filled = 0;
  captureFrame = 0;
  pitch?.setRange(cfg.pitchFloorHz, cfg.pitchCeilingHz);
  pitch?.setThresholds(cfg.pitchSilenceThreshold, cfg.pitchVoicingThreshold);
  pitch?.setVeryAccurate(cfg.pitchVeryAccurate);
  pitch?.resetTracking();
  formant?.setMaxNumberOfFormants(cfg.maxNumberOfFormants);
  formant?.setMaximumFormant(cfg.maximumFormantHz);
  formant?.setWindowLength(cfg.windowLengthS);
  formant?.setPreEmphasisFrom(cfg.preEmphasisFromHz);
  denoiser?.reset();
  tracker.reset();
  lastFormants = [];
}

async function ensureReady(cfg: AnalysisConfig) {
  if (!ready) {
    await init(wasmUrl);
    pitch = new PitchAnalyzer(cfg.samplerate);
    formant = new FormantAnalyzer(cfg.samplerate);
    denoiser = new Denoiser();
    ready = true;
  }
  applyConfig(cfg);
}

function appendBlock(block: Float32Array) {
  const n = block.length;
  if (n >= buffer.length) {
    buffer.set(block.subarray(n - buffer.length));
    filled = buffer.length;
  } else {
    buffer.copyWithin(0, n);
    buffer.set(block, buffer.length - n);
    filled = Math.min(filled + n, buffer.length);
  }
}

function rms(view: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < view.length; i++) sum += view[i] * view[i];
  return Math.sqrt(sum / view.length + 1e-12);
}

function trackedFormants(view: Float32Array): number[] {
  const det = formant!.analyzeDetailed(view); // [f,bw,f,bw,...]
  const cand: Pole[] = [];
  for (let i = 0; i + 1 < det.length; i += 2) cand.push([det[i], det[i + 1]]);
  const tracked = tracker.push(cand);
  const out: number[] = [];
  for (let s = 0; s < 3; s++) {
    const v = tracked[s] ?? lastFormants[s];
    if (v !== undefined) out[s] = v;
  }
  lastFormants = out.slice();

  // Diagnostics: raw candidate poles, the tracker's committed slots (pre-coast),
  // and the coasted output that's actually displayed. Off unless toggled.
  if (trackDebug && config) {
    const r = (v: number | null | undefined) => (v == null ? null : Math.round(v));
    post({
      type: "trackDebug",
      t: captureFrame / config.samplerate,
      cand: cand.map((p) => [Math.round(p[0]), Math.round(p[1])]),
      committed: [tracked[0], tracked[1], tracked[2]].map(r),
      coasted: [out[0], out[1], out[2]].map(r),
    });
  }
  return out.filter((v) => v !== undefined);
}

function analyze(): AnalysisResult | null {
  if (!pitch || !config || filled < 32) return null;
  const view = buffer.subarray(buffer.length - filled);
  const r = rms(view);
  const t = captureFrame / config.samplerate;

  if (r < config.rmsThreshold) {
    // Stay silent for the strips, but DON'T reset the formant tracker/lastFormants:
    // F3 (a wide/jumpy pole at a vowel's onset) should coast across the gap. Do
    // drop pitch history so a new utterance isn't biased toward the old f0.
    pitch.resetTracking();
    return { voiced: false, pitchHz: null, formantsHz: [], t };
  }

  const f0 = pitch.analyze(view);
  const voiced = Number.isFinite(f0);
  const formantsHz = !formant
    ? []
    : config.formantTrackingEnabled
      ? trackedFormants(view)
      : Array.from(formant.analyze(view));

  return { voiced, pitchHz: voiced ? f0 : null, formantsHz, t };
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;
  switch (msg?.type) {
    case "init":
    case "config":
      await ensureReady(msg.config as AnalysisConfig);
      post({ type: "ready" });
      break;
    case "trackDebug":
      trackDebug = !!msg.on;
      break;
    case "procDelay":
      procDelayMs = Math.max(0, msg.ms | 0);
      break;
    case "block": {
      if (!ready) break;
      let samples = msg.samples as Float32Array;
      if (config?.noiseSuppressionEnabled && denoiser) {
        samples = denoiser.process(samples, config.noiseSuppressionMix);
      }
      if (typeof msg.frame === "number") captureFrame = msg.frame;
      appendBlock(samples);
      if (procDelayMs > 0) {
        const t0 = performance.now();
        while (performance.now() - t0 < procDelayMs) {
          /* busy-wait to emulate slow DSP on a weaker device */
        }
      }
      // Always reply (result may be null) so the main thread can pace work to
      // the worker's real throughput, see AudioEngine backpressure.
      post({ type: "result", result: analyze() });
      break;
    }
  }
};
