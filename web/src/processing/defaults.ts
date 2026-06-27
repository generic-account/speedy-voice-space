// Analysis + processing defaults, ported from settings_defaults.py. Field names
// match the Python config so parity comparisons stay 1:1.

export interface AnalysisConfig {
  samplerate: number; // runtime value (the AudioContext rate), not user-editable
  bufferDurationS: number;
  rmsThreshold: number;
  pitchFloorHz: number;
  pitchCeilingHz: number;
  pitchSilenceThreshold: number;
  pitchVoicingThreshold: number;
  pitchVeryAccurate: boolean;
  maxNumberOfFormants: number;
  maximumFormantHz: number;
  preEmphasisFromHz: number;
  noiseSuppressionEnabled: boolean;
  noiseSuppressionMix: number;
  // Assign LPC poles to F1/F2/F3 by continuity (smooth) vs. raw frequency rank.
  formantTrackingEnabled: boolean;
}

export interface ProcessingSettings {
  pitchMedianWindow: number;
  resonanceMedianWindow: number;
  // Causal median on the DISPLAYED formants only (display, not the score).
  formantMedianWindow: number;
  stripWindowSec: number; // seconds of history shown in the F2/F3/pitch strips
  // EMA smoothing amount in [0,1): higher = smoother + laggier (alpha = 1 - this).
  pitchSmoothing: number;
  resonanceSmoothing: number;
  f2LowHz: number;
  f2HighHz: number;
  f3LowHz: number;
  f3HighHz: number;
  f2Weight: number;
  f3Weight: number;
}

export interface AnalysisResult {
  voiced: boolean;
  pitchHz: number | null;
  formantsHz: number[];
  t: number; // audio time (seconds) of this frame, for clock-driven plotting
}

export function defaultAnalysisConfig(): AnalysisConfig {
  return {
    samplerate: 48000,
    bufferDurationS: 0.06,
    rmsThreshold: 0.0005,
    pitchFloorHz: 75,
    pitchCeilingHz: 400,
    pitchSilenceThreshold: 0.03,
    pitchVoicingThreshold: 0.45,
    pitchVeryAccurate: false,
    maxNumberOfFormants: 5,
    maximumFormantHz: 5500,
    preEmphasisFromHz: 50,
    noiseSuppressionEnabled: false,
    noiseSuppressionMix: 0.05,
    formantTrackingEnabled: true,
  };
}

export function defaultProcessingSettings(): ProcessingSettings {
  return {
    pitchMedianWindow: 5,
    resonanceMedianWindow: 7,
    formantMedianWindow: 3,
    stripWindowSec: 10,
    pitchSmoothing: 0.75,
    resonanceSmoothing: 0.85,
    f2LowHz: 600,
    f2HighHz: 3000,
    f3LowHz: 1500,
    f3HighHz: 4500,
    f2Weight: 0.6,
    f3Weight: 0.4,
  };
}
