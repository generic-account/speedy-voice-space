// Median filtering + EMA smoothing + mel-normalized resonance score, run on the
// main thread for each AnalysisResult (port of processing.py).

import type { AnalysisResult, ProcessingSettings } from "./defaults";

export interface DisplayState {
  voiced: boolean;
  rawPitchHz: number | null;
  filteredPitchHz: number | null;
  filteredResonance: number | null;
  resonanceConfidence: number;
  rawF2Hz: number | null;
  rawF3Hz: number | null;
  // Median-filtered F2/F3 for DISPLAY (strips + readouts) — suppresses intrinsic
  // per-frame LPC jitter. Resonance is still computed from the raw values.
  filteredF2Hz: number | null;
  filteredF3Hz: number | null;
  formantsHz: number[];
}

function clamp(lo: number, hi: number, x: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function hzToMel(hz: number): number {
  return 2595.0 * Math.log10(1.0 + hz / 700.0);
}

function normMel01(hz: number, loHz: number, hiHz: number): number {
  const mel = hzToMel(hz);
  const melLo = hzToMel(loHz);
  const melHi = hzToMel(hiHz);
  if (melHi <= melLo) return 0.0;
  return clamp(0.0, 1.0, (mel - melLo) / (melHi - melLo));
}

function smoothValue(
  previous: number | null,
  newValue: number | null,
  alpha: number,
): number | null {
  if (newValue === null) return previous;
  if (previous === null) return newValue;
  return alpha * newValue + (1.0 - alpha) * previous;
}

/** Bounded median window (mirrors collections.deque(maxlen)). */
class MedianWindow {
  private buf: number[] = [];
  constructor(public maxlen: number) {}
  setMaxlen(n: number) {
    this.maxlen = Math.max(1, n);
    while (this.buf.length > this.maxlen) this.buf.shift();
  }
  push(v: number | null): number | null {
    if (v !== null) {
      this.buf.push(v);
      while (this.buf.length > this.maxlen) this.buf.shift();
    }
    if (this.buf.length === 0) return null;
    const s = [...this.buf].sort((a, b) => a - b);
    const mid = Math.floor(s.length / 2);
    // statistics.median: average of two middle values for even counts.
    return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
  }
  clear() {
    this.buf = [];
  }
}

export class VoiceProcessor {
  private pitchWindow: MedianWindow;
  private resonanceWindow: MedianWindow;
  private f2Window: MedianWindow;
  private f3Window: MedianWindow;
  private smoothedPitch: number | null = null;
  private smoothedResonance: number | null = null;

  constructor(private settings: ProcessingSettings) {
    this.pitchWindow = new MedianWindow(settings.pitchMedianWindow);
    this.resonanceWindow = new MedianWindow(settings.resonanceMedianWindow);
    this.f2Window = new MedianWindow(settings.formantMedianWindow);
    this.f3Window = new MedianWindow(settings.formantMedianWindow);
  }

  updateSettings(settings: ProcessingSettings) {
    this.settings = settings;
    this.pitchWindow.setMaxlen(settings.pitchMedianWindow);
    this.resonanceWindow.setMaxlen(settings.resonanceMedianWindow);
    this.f2Window.setMaxlen(settings.formantMedianWindow);
    this.f3Window.setMaxlen(settings.formantMedianWindow);
    this.smoothedPitch = null;
    this.smoothedResonance = null;
  }

  reset() {
    this.pitchWindow.clear();
    this.resonanceWindow.clear();
    this.f2Window.clear();
    this.f3Window.clear();
    this.smoothedPitch = null;
    this.smoothedResonance = null;
  }

  private computeResonance(formants: number[]): {
    score: number | null;
    confidence: number;
    f2: number | null;
    f3: number | null;
  } {
    const f2 = formants.length > 1 ? formants[1] : null;
    const f3 = formants.length > 2 ? formants[2] : null;
    const values: number[] = [];
    const weights: number[] = [];

    if (f2 !== null) {
      values.push(normMel01(f2, this.settings.f2LowHz, this.settings.f2HighHz));
      weights.push(this.settings.f2Weight);
    }
    if (f3 !== null) {
      values.push(normMel01(f3, this.settings.f3LowHz, this.settings.f3HighHz));
      weights.push(this.settings.f3Weight);
    }
    const totalWeight = weights.reduce((a, b) => a + b, 0);
    if (values.length === 0 || totalWeight <= 0) {
      return { score: null, confidence: 0.0, f2, f3 };
    }
    let score = 0;
    for (let i = 0; i < values.length; i++) score += values[i] * (weights[i] / totalWeight);
    return { score, confidence: clamp(0.0, 1.0, totalWeight), f2, f3 };
  }

  process(result: AnalysisResult): DisplayState {
    const { score, confidence, f2, f3 } = this.computeResonance(result.formantsHz);

    const medianPitch = this.pitchWindow.push(result.pitchHz);
    const medianResonance = this.resonanceWindow.push(score);

    // During unvoiced/silent frames there are no formants — clear the windows so
    // the displayed F2/F3 read null (a gap in the strips), not a stale held line.
    let medianF2: number | null = null;
    let medianF3: number | null = null;
    if (result.voiced) {
      medianF2 = this.f2Window.push(f2);
      medianF3 = this.f3Window.push(f3);
    } else {
      this.f2Window.clear();
      this.f3Window.clear();
    }

    this.smoothedPitch = smoothValue(
      this.smoothedPitch,
      medianPitch,
      this.settings.pitchAlpha,
    );
    this.smoothedResonance = smoothValue(
      this.smoothedResonance,
      medianResonance,
      this.settings.resonanceAlpha,
    );

    return {
      voiced: result.voiced,
      rawPitchHz: result.pitchHz,
      filteredPitchHz: this.smoothedPitch,
      filteredResonance: this.smoothedResonance,
      resonanceConfidence: confidence,
      rawF2Hz: f2,
      rawF3Hz: f3,
      filteredF2Hz: medianF2,
      filteredF3Hz: medianF3,
      formantsHz: [...result.formantsHz],
    };
  }
}
