import { C } from "../theme";
import type { DisplayState } from "../processing/voiceProcessor";

function fmtHz(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : `${v.toFixed(1)} Hz`;
}

function Row({ label, value, testid }: { label: string; value: string; testid?: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, padding: "1px 0" }}>
      <span style={{ color: C.label }}>{label}</span>
      <span data-testid={testid} style={{ color: C.text, fontVariantNumeric: "tabular-nums" }}>{value}</span>
    </div>
  );
}

export function Readouts({ display }: { display: DisplayState | null }) {
  const d = display;
  return (
    <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 4, padding: "6px 8px" }}>
      <div style={{ fontWeight: 700, textAlign: "center", fontSize: 13, color: C.text }}>Voice</div>
      <Row label="Pitch" testid="ro-pitch" value={fmtHz(d?.filteredPitchHz)} />
      <Row label="Resonance" testid="ro-resonance" value={d?.filteredResonance == null ? "—" : d.filteredResonance.toFixed(3)} />
      <Row label="Confidence" value={d ? d.resonanceConfidence.toFixed(2) : "—"} />
      <Row label="Voiced" testid="ro-voiced" value={d ? (d.voiced ? "Yes" : "No") : "—"} />
      <div style={{ fontWeight: 700, textAlign: "center", fontSize: 13, color: C.text, marginTop: 4 }}>Formants</div>
      <Row label="F2" value={fmtHz(d?.filteredF2Hz)} />
      <Row label="F3" value={fmtHz(d?.filteredF3Hz)} />
      <Row label="All" value={d && d.formantsHz.length ? d.formantsHz.slice(0, 5).map((f) => f.toFixed(0)).join(", ") : "—"} />
    </div>
  );
}
