import { useState } from "react";
import { createPortal } from "react-dom";
import { C } from "../theme";
import type { AnalysisConfig, ProcessingSettings } from "../processing/defaults";

// A field targets either the analysis config (a) or processing settings (p);
// `boolA` is a checkbox on the analysis config.
export type Field =
  | { kind: "boolA"; key: keyof AnalysisConfig; label: string; tip: string }
  | { kind: "numA"; key: keyof AnalysisConfig; label: string; tip: string; step?: number }
  | { kind: "numP"; key: keyof ProcessingSettings; label: string; tip: string; step?: number }
  // Compact 2×3 grid for the F2/F3 Low/High/Weight resonance-mapping fields.
  | { kind: "resonGrid" };

// Core: input, levels, and resonance smoothing.
export const CORE_FIELDS: Field[] = [
  { kind: "boolA", key: "formantTrackingEnabled", label: "Continuity Tracking", tip: "Keeps F1/F2/F3 assignment smooth. Best left on." },
  { kind: "boolA", key: "noiseSuppressionEnabled", label: "Noise Suppression", tip: "Toggles background-noise removal." },
  { kind: "numA", key: "noiseSuppressionMix", label: "Suppression Strength", step: 0.01, tip: "How strongly noise is removed (0 to 1)." },
  { kind: "numA", key: "rmsThreshold", label: "RMS Threshold", step: 0.0001, tip: "Loudness gate. Set between background and speaking levels." },
  { kind: "numA", key: "bufferDurationS", label: "Buffer (s)", step: 0.005, tip: "Audio window length per frame." },
  { kind: "numP", key: "resonanceMedianWindow", label: "Resonance Median", step: 1, tip: "Median filter on resonance. Rejects spikes." },
  { kind: "numP", key: "resonanceSmoothing", label: "Resonance Smoothing", step: 0.05, tip: "Resonance smoothing. Higher is smoother (more lag)." },
];

// Pitch: detection and smoothing.
export const PITCH_FIELDS: Field[] = [
  { kind: "boolA", key: "pitchVeryAccurate", label: "Very Accurate", tip: "Slower, more precise pitch analysis." },
  { kind: "numA", key: "pitchFloorHz", label: "Pitch Floor", step: 1, tip: "Lowest pitch (Hz) considered." },
  { kind: "numA", key: "pitchCeilingHz", label: "Pitch Ceiling", step: 1, tip: "Highest pitch (Hz) considered." },
  { kind: "numA", key: "pitchSilenceThreshold", label: "Silence Threshold", step: 0.01, tip: "Below this, frames count as unvoiced." },
  { kind: "numA", key: "pitchVoicingThreshold", label: "Voicing Threshold", step: 0.01, tip: "Strength needed to call a frame voiced. Higher is stricter." },
  { kind: "numP", key: "pitchMedianWindow", label: "Pitch Median", step: 1, tip: "Median filter on pitch." },
  { kind: "numP", key: "pitchSmoothing", label: "Pitch Smoothing", step: 0.05, tip: "Pitch smoothing. Higher is smoother (more lag)." },
];

// Formant: extraction, display median, and resonance (F2/F3) balance.
export const FORMANT_FIELDS: Field[] = [
  { kind: "numA", key: "maxNumberOfFormants", label: "Max Formants", step: 0.5, tip: "How many formant peaks to find." },
  { kind: "numA", key: "maximumFormantHz", label: "Max Formant (Hz)", step: 100, tip: "Upper frequency bound for the search." },
  { kind: "numA", key: "windowLengthS", label: "Window (s)", step: 0.001, tip: "Audio window length for formants." },
  { kind: "numA", key: "preEmphasisFromHz", label: "Pre-emphasis (Hz)", step: 10, tip: "Boosts higher formants before analysis." },
  { kind: "numP", key: "formantMedianWindow", label: "Formant Median", step: 1, tip: "Median filter on displayed F2/F3." },
  { kind: "resonGrid" },
];

// The resonance-mapping grid: rows F2/F3 × columns Low/High/Weight.
const RESON_GRID: { row: "F2" | "F3"; low: keyof ProcessingSettings; high: keyof ProcessingSettings; weight: keyof ProcessingSettings }[] = [
  { row: "F2", low: "f2LowHz", high: "f2HighHz", weight: "f2Weight" },
  { row: "F3", low: "f3LowHz", high: "f3HighHz", weight: "f3Weight" },
];

interface Props {
  title: string;
  fields: Field[];
  analysis: AnalysisConfig;
  processing: ProcessingSettings;
  onAnalysis: (a: AnalysisConfig) => void;
  onProcessing: (p: ProcessingSettings) => void;
}

const rowStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  fontSize: 11,
  gap: 6,
  minHeight: 18,
};

const labelStyle: React.CSSProperties = { color: C.label, whiteSpace: "nowrap" };
// width is the resting size; minWidth:0 lets it shrink instead of overflowing
// when the panel column gets narrow.
const numInputStyle: React.CSSProperties = { width: 60, minWidth: 0, fontSize: 11 };
const gridInputStyle: React.CSSProperties = { flex: 1, minWidth: 0, width: 0, fontSize: 11 };

interface TipState {
  text: string;
  x: number;
  y: number;
}

export function SettingsPanel({ title, fields, analysis, processing, onAnalysis, onProcessing }: Props) {
  // Lightweight hover tooltip rendered to <body> so the scrollable column can't
  // clip it (and so it shows instantly, unlike the native `title` delay).
  const [tip, setTip] = useState<TipState | null>(null);
  const tipProps = (text: string) => ({
    onMouseEnter: (e: React.MouseEvent) => setTip({ text, x: e.clientX, y: e.clientY }),
    onMouseMove: (e: React.MouseEvent) => setTip({ text, x: e.clientX, y: e.clientY }),
    onMouseLeave: () => setTip(null),
  });

  const gridCell = (key: keyof ProcessingSettings, step: number) => (
    <input
      type="number"
      step={step}
      value={processing[key] as number}
      onChange={(e) => onProcessing({ ...processing, [key]: Number(e.target.value) })}
      style={gridInputStyle}
    />
  );

  return (
    <div
      style={{
        background: C.surface,
        border: `1px solid ${C.border}`,
        borderRadius: 4,
        padding: "6px 8px",
        height: "100%",
        boxSizing: "border-box",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div style={{ fontWeight: 700, fontSize: 13, textAlign: "center", color: C.text, marginBottom: 4 }}>
        {title}
      </div>
      {/* flex:1 + space-between distributes rows to fill the box, so panels of
          different row counts still end at the same bottom. */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
      {fields.map((f, i) => {
        if (f.kind === "resonGrid") {
          return (
            <div
              key={i}
              style={{ marginTop: 2 }}
              {...tipProps("Per formant: Hz range to normalize over (Low, High) and weight in the resonance score.")}
            >
              <div style={{ display: "flex", gap: 4, fontSize: 10, color: C.muted }}>
                <span style={{ width: 22 }} />
                <span style={{ flex: 1, textAlign: "center" }}>Low</span>
                <span style={{ flex: 1, textAlign: "center" }}>High</span>
                <span style={{ flex: 1, textAlign: "center" }}>Weight</span>
              </div>
              {RESON_GRID.map((g) => (
                <div key={g.row} style={{ display: "flex", gap: 4, alignItems: "center", marginTop: 2 }}>
                  <span style={{ width: 22, color: C.label, fontSize: 11 }}>{g.row}</span>
                  {gridCell(g.low, 50)}
                  {gridCell(g.high, 50)}
                  {gridCell(g.weight, 0.05)}
                </div>
              ))}
            </div>
          );
        }
        return (
          <label key={i} style={rowStyle} {...tipProps(f.tip)}>
            <span style={labelStyle}>{f.label}</span>
            {f.kind === "boolA" ? (
              <input
                type="checkbox"
                checked={analysis[f.key] as boolean}
                onChange={(e) => onAnalysis({ ...analysis, [f.key]: e.target.checked })}
              />
            ) : f.kind === "numA" ? (
              <input
                type="number"
                step={f.step}
                value={analysis[f.key] as number}
                onChange={(e) => onAnalysis({ ...analysis, [f.key]: Number(e.target.value) })}
                style={numInputStyle}
              />
            ) : (
              <input
                type="number"
                step={f.step}
                value={processing[f.key] as number}
                onChange={(e) => onProcessing({ ...processing, [f.key]: Number(e.target.value) })}
                style={numInputStyle}
              />
            )}
          </label>
        );
      })}
      </div>
      {tip &&
        createPortal(
          <div
            style={{
              position: "fixed",
              left: Math.min(tip.x + 8, window.innerWidth - 222),
              top: tip.y + 8,
              maxWidth: 210,
              background: "#2a2a2a",
              color: C.text,
              border: `1px solid ${C.border}`,
              fontFamily: "system-ui, sans-serif",
              fontSize: 11,
              lineHeight: 1.4,
              padding: "5px 7px",
              borderRadius: 4,
              boxShadow: "0 2px 8px rgba(0,0,0,0.5)",
              pointerEvents: "none",
              zIndex: 1000,
            }}
          >
            {tip.text}
          </div>,
          document.body,
        )}
    </div>
  );
}
