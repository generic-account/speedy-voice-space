import { C } from "../theme";

interface Props {
  rms: number;
  peak: number;
  threshold: number; // RMS voicing-gate cutoff (from settings)
}

// Map RMS to a 0..1 bar via a dB curve. Floor at -80 dB so the voicing cutoff
// (often ~-66 dB) sits visibly in range, not pinned to the left edge.
function toBar(rms: number): number {
  if (rms <= 1e-6) return 0;
  const db = 20 * Math.log10(rms);
  return Math.max(0, Math.min(1, (db + 80) / 80));
}

export function LevelMeter({ rms, peak, threshold }: Props) {
  const bar = toBar(rms);
  const peakBar = toBar(peak);
  const cutBar = toBar(threshold);
  return (
    <div
      data-testid="level-meter"
      data-rms={rms}
      title={`RMS ${rms.toFixed(4)} · peak ${peak.toFixed(4)} · cutoff ${threshold}`}
      style={{
        position: "relative",
        height: 22,
        background: "#0a0a0a",
        border: `1px solid ${C.border}`,
        borderRadius: 4,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          width: `${bar * 100}%`,
          height: "100%",
          background: "#52c97a",
        }}
      />
      {/* RMS cutoff (voicing gate): amber line. */}
      <div
        title={`RMS cutoff ${threshold}`}
        style={{ position: "absolute", top: 0, left: `${cutBar * 100}%`, width: 2, height: "100%", background: "#ffb300" }}
      />
      {/* Peak hold (max since start): white line. */}
      <div
        title="peak"
        style={{ position: "absolute", top: 0, left: `${peakBar * 100}%`, width: 2, height: "100%", background: "#fff" }}
      />
    </div>
  );
}
