// AudioEngine: audio source → capture AudioWorklet → DSP worker, plus stats.
// The source is pluggable, getUserMedia for the mic, a decoded WAV for
// deterministic tests, both feeding the same worklet.
import {
  defaultAnalysisConfig,
  type AnalysisConfig,
  type AnalysisResult,
} from "../processing/defaults";

const TARGET_SAMPLE_RATE = 48000;
const BLOCK_SIZE = 1024;

export interface AudioStats {
  running: boolean;
  source: "mic" | "file" | "none";
  blocks: number;
  lastRms: number;
  peakRms: number;
  sampleRate: number;
  ctxState: string;
  deviceLabel: string;
}

export interface AudioInputDevice {
  deviceId: string;
  label: string;
}

export interface TrackDebugFrame {
  t: number; // audio time (s)
  cand: [number, number][]; // candidate LPC poles [freq, bandwidth]
  committed: (number | null)[]; // tracker's F1/F2/F3 this frame (null = unfilled)
  coasted: (number | null)[]; // displayed F1/F2/F3 after coasting
}

function rms(block: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < block.length; i++) sum += block[i] * block[i];
  return Math.sqrt(sum / block.length + 1e-12);
}

export class AudioEngine {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private srcNode: AudioNode | null = null;
  private node: AudioWorkletNode | null = null;
  private worker: Worker | null = null;
  private resumeFallbackInstalled = false;

  config: AnalysisConfig = defaultAnalysisConfig();
  onResult: ((result: AnalysisResult) => void) | null = null;

  // Formant-tracker diagnostics: enable with setTrackDebug(true), then read the
  // collected per-frame {candidates, committed, coasted} frames from trackDebugLog.
  trackDebugLog: TrackDebugFrame[] = [];

  // Backpressure: keep one block in flight; while the worker is busy, accumulate
  // and send it only the freshest audio on reply (drops stale blocks under load).
  private busy = false;
  private pending: Float32Array[] = [];
  private pendingFrame = 0;
  private lastCaptureT = 0; // newest captured audio time (s)
  private lastProcessedT = 0; // newest analyzed audio time (s)

  /** How far analysis lags real-time capture (seconds). Bounded by backpressure. */
  get backlogS(): number {
    return Math.max(0, this.lastCaptureT - this.lastProcessedT);
  }

  readonly stats: AudioStats = {
    running: false,
    source: "none",
    blocks: 0,
    lastRms: 0,
    peakRms: 0,
    sampleRate: TARGET_SAMPLE_RATE,
    ctxState: "none",
    deviceLabel: "",
  };

  async listInputDevices(): Promise<AudioInputDevice[]> {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices
      .filter((d) => d.kind === "audioinput")
      .map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Microphone ${i + 1}` }));
  }

  async start(deviceId?: string): Promise<void> {
    await this.stop();
    const ctx = await this.createContext();

    // Resume *before* getUserMedia, while the click's gesture is still active,
    // after the permission dialog the gesture has expired and resume() no-ops,
    // leaving the context suspended (no audio). The listener below is a backup.
    try {
      await ctx.resume();
    } catch {
      /* retried by the gesture listener */
    }
    this.installResumeFallback();

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
      video: false,
    });
    this.stats.deviceLabel = this.stream.getAudioTracks()[0]?.label ?? "";
    this.srcNode = ctx.createMediaStreamSource(this.stream);
    await this.wireGraph(ctx, this.srcNode, "mic");
  }

  /** Deterministic capture from a decoded audio file (tests + manual A/B). */
  async startFromUrl(url: string, loop = true): Promise<void> {
    await this.stop();
    const ctx = await this.createContext();
    const buffer = await ctx.decodeAudioData(await (await fetch(url)).arrayBuffer());
    const node = ctx.createBufferSource();
    node.buffer = buffer;
    node.loop = loop;
    this.srcNode = node;
    await this.wireGraph(ctx, this.srcNode, "file");
    node.start();
  }

  private installResumeFallback(): void {
    if (this.resumeFallbackInstalled) return;
    this.resumeFallbackInstalled = true;
    const handler = () => {
      if (this.ctx?.state === "suspended") void this.ctx.resume();
    };
    window.addEventListener("pointerdown", handler);
    window.addEventListener("keydown", handler);
  }

  private async createContext(): Promise<AudioContext> {
    // Use the hardware's native rate; forcing 48 kHz can make Chrome deliver no
    // audio (silently) on devices that don't open at it. The DSP adapts to the
    // reported rate (only the off-by-default denoiser prefers 48 kHz).
    let ctx: AudioContext;
    try {
      ctx = new AudioContext({ latencyHint: "interactive" });
    } catch {
      ctx = new AudioContext();
    }
    this.ctx = ctx;
    this.stats.ctxState = ctx.state;
    ctx.onstatechange = () => {
      this.stats.ctxState = ctx.state;
    };
    await ctx.audioWorklet.addModule("capture-worklet.js");
    this.ensureWorker(ctx.sampleRate);
    return ctx;
  }

  private ensureWorker(sampleRate: number): void {
    if (!this.worker) {
      this.worker = new Worker(new URL("./dsp-worker.ts", import.meta.url), { type: "module" });
      this.worker.onmessage = (e: MessageEvent) => {
        if (e.data?.type === "result") this.onWorkerReply(e.data.result as AnalysisResult | null);
        else if (e.data?.type === "trackDebug") {
          this.trackDebugLog.push(e.data as TrackDebugFrame);
          if (this.trackDebugLog.length > 5000) this.trackDebugLog.shift();
        }
      };
    }
    this.resetBackpressure();
    this.config = { ...this.config, samplerate: sampleRate };
    this.worker.postMessage({ type: "init", config: this.config });
  }

  private resetBackpressure(): void {
    this.busy = false;
    this.pending = [];
    this.lastCaptureT = 0;
    this.lastProcessedT = 0;
  }

  // Worker finished a block: surface the result, then feed it the freshest audio
  // that piled up while it was busy (merged into one block), or go idle.
  private onWorkerReply(result: AnalysisResult | null): void {
    if (result) {
      this.lastProcessedT = result.t;
      this.onResult?.(result);
    }
    if (this.pending.length > 0) {
      let total = 0;
      for (const b of this.pending) total += b.length;
      const merged = new Float32Array(total);
      let off = 0;
      for (const b of this.pending) {
        merged.set(b, off);
        off += b.length;
      }
      this.pending = [];
      this.sendBlock(merged, this.pendingFrame);
    } else {
      this.busy = false;
    }
  }

  private sendBlock(samples: Float32Array, frame: number): void {
    this.busy = true;
    this.worker?.postMessage({ type: "block", samples, frame }, [samples.buffer]);
  }

  /** Toggle per-frame formant-tracker diagnostics (clears the log when enabling). */
  setTrackDebug(on: boolean): void {
    if (on) this.trackDebugLog = [];
    this.worker?.postMessage({ type: "trackDebug", on });
  }

  /** Diagnostic: simulate slow per-block DSP (ms) to emulate a weaker device. */
  setProcDelay(ms: number): void {
    this.worker?.postMessage({ type: "procDelay", ms });
  }

  /** Push updated analysis config to the worker (resets its rolling buffer). */
  updateConfig(config: AnalysisConfig): void {
    this.config = config;
    this.worker?.postMessage({ type: "config", config });
  }

  private async wireGraph(ctx: AudioContext, source: AudioNode, kind: "mic" | "file"): Promise<void> {
    this.node = new AudioWorkletNode(ctx, "capture-processor", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      channelCount: 1,
      processorOptions: { blockSize: BLOCK_SIZE },
    });
    this.node.port.onmessage = (e: MessageEvent) => {
      if (e.data?.type === "block") this.handleBlock(e.data.samples as Float32Array, e.data.frame as number);
    };
    source.connect(this.node);
    // Connect to destination so the node is pulled; it writes no output, so
    // nothing is played back (no mic feedback).
    this.node.connect(ctx.destination);
    if (ctx.state === "suspended") await ctx.resume();

    Object.assign(this.stats, {
      running: true,
      source: kind,
      blocks: 0,
      lastRms: 0,
      peakRms: 0,
      sampleRate: ctx.sampleRate,
    });
    (window as unknown as { __audioStats?: AudioStats }).__audioStats = this.stats;
  }

  private handleBlock(block: Float32Array, frame: number): void {
    this.stats.blocks++;
    const r = rms(block);
    this.stats.lastRms = r;
    if (r > this.stats.peakRms) this.stats.peakRms = r;
    this.lastCaptureT = frame / this.stats.sampleRate;
    // If the worker is still chewing on a block, hold this one (and any others)
    // and let onWorkerReply send the freshest merged audio. Otherwise send now.
    if (this.busy) {
      this.pending.push(block);
      this.pendingFrame = frame;
    } else {
      this.pendingFrame = frame;
      this.sendBlock(block, frame);
    }
  }

  async stop(): Promise<void> {
    if (this.node) {
      this.node.port.onmessage = null;
      this.node.disconnect();
      this.node = null;
    }
    if (this.srcNode) {
      try {
        (this.srcNode as AudioBufferSourceNode).stop?.();
      } catch {
        /* mic source has no stop() */
      }
      this.srcNode.disconnect();
      this.srcNode = null;
    }
    if (this.stream) {
      for (const t of this.stream.getTracks()) t.stop();
      this.stream = null;
    }
    if (this.ctx) {
      await this.ctx.close();
      this.ctx = null;
    }
    this.stats.running = false;
    this.stats.source = "none";
  }
}
