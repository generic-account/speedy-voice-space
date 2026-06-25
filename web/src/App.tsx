import { useEffect, useRef, useState } from "react";
import { C } from "./theme";
import { AudioEngine, type AudioInputDevice } from "./audio/engine";
import { LevelMeter } from "./components/LevelMeter";
import { MainPlot, type TrailPoint } from "./components/MainPlot";
import { FormantStrips } from "./components/FormantStrips";
import { Readouts } from "./components/Readouts";
import { SettingsPanel, CORE_FIELDS, PITCH_FIELDS, FORMANT_FIELDS } from "./components/SettingsPanel";
import { VoiceProcessor, type DisplayState } from "./processing/voiceProcessor";
import {
  defaultAnalysisConfig,
  defaultProcessingSettings,
  type AnalysisConfig,
  type ProcessingSettings,
} from "./processing/defaults";

const TRAIL_LEN = 120;
const STRIP_WINDOW_SEC = 10; // formant strips show the last N seconds (clock-driven)

const ABOUT_CARD: React.CSSProperties = {
  background: C.surface,
  border: `1px solid ${C.border}`,
  borderRadius: 4,
  padding: "10px 12px",
  boxSizing: "border-box",
  fontSize: 13,
  color: C.label,
  lineHeight: 1.5,
};
const ABOUT_P: React.CSSProperties = { margin: "0 0 8px" };
const ABOUT_LEAD: React.CSSProperties = { margin: "0 0 2px" };
const ABOUT_UL: React.CSSProperties = { margin: "0 0 8px", paddingLeft: 18 };
const ABOUT_TABS: React.CSSProperties = {
  display: "flex",
  gap: 16,
  marginBottom: 8,
  borderBottom: `1px solid ${C.border}`,
};
function aboutTabStyle(active: boolean): React.CSSProperties {
  return {
    background: "transparent",
    border: "none",
    borderRadius: 0,
    padding: "0 0 5px",
    marginBottom: -1, // sit the underline on the row's border
    cursor: "pointer",
    fontSize: 13,
    fontWeight: 700,
    color: active ? C.text : C.muted,
    borderBottom: `2px solid ${active ? C.accent : "transparent"}`,
  };
}

// About content as switchable pages, append an entry to add a tab.
const ABOUT_PAGES: { label: string; body: React.ReactNode }[] = [
  {
    label: "About",
    body: (
      <>
        <p style={ABOUT_P}>
          This site provides live pitch and resonance tracking, based on Praat
          algorithms ported to Rust and compiled to wasm. It is hosted statically
          on GitHub Pages and runs entirely in your browser. Your data never
          leaves your device.
        </p>
        <p style={ABOUT_P}>
          The site <em>should</em> work in every modern desktop and mobile
          browser. If you notice any problems or would like to see any new
          features, please open a GitHub issue{" "}
          <a
            href="https://github.com/generic-account/speedy-voice-space/issues"
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: C.accent }}
          >
            here
          </a>
          .
        </p>
        <p style={ABOUT_P}>
          Our voices are composed of a distribution of frequencies, with pitch
          being the base. Resonance is generally the process by which the higher
          frequencies of the voice’s audio spectrum are transformed by the shape
          of our vocal tract.
        </p>
        <p style={ABOUT_P}>
          By making your vocal tract smaller (e.g. by pushing your tongue to the
          front of your mouth), you can raise the resonance of your voice, and by
          moving your tongue further down and back to enlarge the vocal tract you
          can lower the resonance of your voice.
        </p>
        <p style={ABOUT_P}>
          Formants represent “peaks” in the frequency spectrum of voice, with F0
          representing pitch, F1 being associated with vowel height, F2 associated
          with vowel advancement, and F3 associated with vocal tract length. The
          weighted average of normalized F2 and F3 values provides a pretty good
          heuristic for our perception of resonance, which is what preexisting
          projects, literature, and I use.
        </p>
        <p style={{ ...ABOUT_P, marginBottom: 0 }}>
          Thank you so much to{" "}
          <a
            href="https://acousticgender.space/"
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: C.accent }}
          >
            Acoustic Genderspace
          </a>{" "}
          for the inspiration! It’s an amazing site and I wanted something real time
          to shorten the voice training feedback loop. I hope it helps!
        </p>
      </>
    ),
  },
  {
    label: "How to use",
    body: (
      <>
        <p style={ABOUT_P}>
          Press “Start” and give the site access to your microphone, selecting the
          input you intend to use. Observe your baseline level of background noise
          (RMS), and the RMS measurement when you are speaking, and set the RMS
          threshold in “Core Settings” in between.
        </p>
        <p style={ABOUT_P}>
          Hold out a vowel sound, and watch your pitch and resonance on the 2D
          graph, or your raw extracted F2 and F3 values on the time-series charts
          above.
        </p>
        <p style={ABOUT_P}>
          If your environment has a lot of background noise, turning on noise
          suppression and tuning the suppression strength could improve results.
          Active noise suppression is heavier computationally and can slightly
          affect output quality, so it is not on by default.
        </p>
        <p style={ABOUT_P}>
          If the output data remains noisy or spiky (which can be the case
          depending on voice / what is being spoken), adjust the median or smoothing
          parameters for resonance or pitch, corresponding to the noisy output
          parameter you observe. A larger median outright rejects outlier input
          data points, while exponential smoothing blends each median-filtered data
          point into a moving average. A larger median more aggressively eliminates
          outliers, and a larger smoothing value leans more on historical points for
          a smoother line. Raising either parameter leads to slightly laggier output.
        </p>
        <p style={{ ...ABOUT_P, marginBottom: 0 }}>Hover over any setting for more information.</p>
      </>
    ),
  },
  {
    label: "Shaping",
    body: (
      <>
        <p style={ABOUT_P}>
          <b>Darker resonance:</b> higher F1, lower F2, lower F3.
          <br />
          <b>Brighter resonance:</b> lower F1, higher F2, higher F3.
        </p>
        <p style={ABOUT_LEAD}>
          <b>F1</b> (tongue up vs. down):
        </p>
        <ul style={ABOUT_UL}>
          <li>High F1: low vowel, low tongue body.</li>
          <li>Low F1: high vowel, high tongue body.</li>
        </ul>
        <p style={ABOUT_LEAD}>
          <b>F2</b> (tongue forward vs. back):
        </p>
        <ul style={ABOUT_UL}>
          <li>High F2: front vowel, tongue forward.</li>
          <li>Low F2: back vowel, tongue back / retracted.</li>
        </ul>
        <p style={ABOUT_LEAD}>
          <b>F3</b> (partly lip rounding vs. spreading):
        </p>
        <ul style={{ ...ABOUT_UL, marginBottom: 0 }}>
          <li>High F3: spread lips.</li>
          <li>Low F3: rounded and/or protruded lips.</li>
        </ul>
      </>
    ),
  },
];
// Plot "home" range (double-click reset / initial view). Wide enough for any
// voice; users scroll/drag to zoom into their own range.
const PLOT_PITCH_RANGE: [number, number] = [65, 500];
const PLOT_RESONANCE_RANGE: [number, number] = [0, 1];

// Merge persisted settings over defaults (tolerant of added/removed fields).
function loadSettings<T extends object>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    return { ...fallback, ...JSON.parse(raw) };
  } catch {
    return fallback;
  }
}

function saveSettings(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable (private mode), ignore */
  }
}

function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const onChange = () => setMatches(mq.matches);
    onChange();
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [query]);
  return matches;
}

export default function App() {
  const engineRef = useRef<AudioEngine | null>(null);
  if (engineRef.current === null) engineRef.current = new AudioEngine();
  const engine = engineRef.current;

  const processorRef = useRef<VoiceProcessor | null>(null);
  if (processorRef.current === null) {
    processorRef.current = new VoiceProcessor(defaultProcessingSettings());
  }
  const processor = processorRef.current;

  const [running, setRunning] = useState(false);
  const [devices, setDevices] = useState<AudioInputDevice[]>([]);
  const [deviceId, setDeviceId] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [source, setSource] = useState<string>("none");

  // Config drives the live pipeline + plot ranges; settings apply on change.
  const [analysisCfg, setAnalysisCfg] = useState<AnalysisConfig>(() =>
    loadSettings("svs.analysis", defaultAnalysisConfig()),
  );
  const [procCfg, setProcCfg] = useState<ProcessingSettings>(() =>
    loadSettings("svs.processing", defaultProcessingSettings()),
  );

  // Apply persisted processing settings to the processor on first mount.
  useEffect(() => {
    processor.updateSettings(procCfg);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const [rms, setRms] = useState(0);
  const [peak, setPeak] = useState(0);
  const [blocks, setBlocks] = useState(0);
  const [ctxState, setCtxState] = useState("none");
  const [deviceLabel, setDeviceLabel] = useState("");
  const [display, setDisplay] = useState<DisplayState | null>(null);

  // When the window is too horizontally short (portrait-ish or just narrow), the
  // height-sized square plot crowds out the sidebar, stack everything instead.
  const narrow = useMediaQuery("(max-aspect-ratio: 7/5), (max-width: 760px)");

  const [aboutPage, setAboutPage] = useState(0); // selected About tab

  // Plot histories (mutable refs; React re-renders pull snapshots each frame).
  const trailRef = useRef<TrailPoint[]>([]);
  // Timestamped formant samples (audio seconds) + an audio-clock anchor, so the
  // strips scroll on the clock and bursty arrival never lurches the line.
  const samplesRef = useRef<{ t: number; f2: number; f3: number }[]>([]);
  const clockRef = useRef<{ audioT: number; wall: number }>({ audioT: 0, wall: 0 });
  const [trail, setTrail] = useState<TrailPoint[]>([]);

  // Wire engine results → processor → display state (+ test hook).
  useEffect(() => {
    (window as unknown as { __engine?: AudioEngine }).__engine = engine;
    engine.onResult = (result) => {
      const state = processor.process(result);
      setDisplay(state);
      (window as unknown as { __lastDisplay?: DisplayState }).__lastDisplay =
        state;

      // Append the timestamped sample (NaN when unvoiced), anchor the audio
      // clock, and drop samples older than the window.
      samplesRef.current.push({
        t: result.t,
        f2: state.filteredF2Hz ?? NaN,
        f3: state.filteredF3Hz ?? NaN,
      });
      // Phase-lock the wall-clock edge to the audio capture clock (they drift):
      // catch up fast when capture runs ahead, ease back slowly when it falls
      // behind. Capped both ways so it stays smooth.
      const wallNow = performance.now();
      const c = clockRef.current;
      if (c.wall === 0) {
        clockRef.current = { audioT: result.t, wall: wallNow };
      } else {
        const ahead = result.t - (c.audioT + (wallNow - c.wall) / 1000);
        const nudge = ahead > 0 ? Math.min(ahead, 0.05) : Math.max(ahead, -0.003);
        clockRef.current = { audioT: c.audioT + nudge, wall: c.wall };
      }
      const cutoff = result.t - (STRIP_WINDOW_SEC + 1);
      const s = samplesRef.current;
      let drop = 0;
      while (drop < s.length && s[drop].t < cutoff) drop++;
      if (drop > 0) s.splice(0, drop);

      const w = window as unknown as {
        __instrument?: boolean;
        __resultLog?: { t: number; voiced: boolean; f2: number | null; f3: number | null }[];
      };
      if (w.__instrument)
        (w.__resultLog ??= []).push({
          t: performance.now(),
          voiced: state.voiced,
          f2: state.filteredF2Hz,
          f3: state.filteredF3Hz,
        });

      if (state.filteredPitchHz !== null && state.filteredResonance !== null) {
        trailRef.current = [
          ...trailRef.current,
          { pitch: state.filteredPitchHz, resonance: state.filteredResonance },
        ].slice(-TRAIL_LEN);
      }
    };
    return () => {
      engine.onResult = null;
    };
  }, [engine, processor]);

  // Pull engine stats + plot snapshots each animation frame.
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const s = engine.stats;
      setRms(s.lastRms);
      setPeak(s.peakRms);
      setBlocks(s.blocks);
      setSource(s.source);
      setCtxState(s.ctxState);
      setDeviceLabel(s.deviceLabel);
      setTrail(trailRef.current);
      // Strips render imperatively from samplesRef/clockRef (see FormantStrips).
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [engine]);

  function clearHistories() {
    trailRef.current = [];
    samplesRef.current = [];
    clockRef.current = { audioT: 0, wall: 0 };
    processor.reset();
  }

  async function refreshDevices() {
    try {
      setDevices(await engine.listInputDevices());
    } catch (e) {
      setError(String(e));
    }
  }

  async function start() {
    setError("");
    clearHistories();
    try {
      engine.updateConfig(analysisCfg);
      await engine.start(deviceId || undefined);
      setRunning(true);
      await refreshDevices();
    } catch (e) {
      setError(String(e));
      setRunning(false);
    }
  }

  async function stop() {
    await engine.stop();
    setRunning(false);
  }

  // Settings apply live. Analysis changes go to the worker (debounced so typing
  // a multi-digit value doesn't thrash the rolling buffer); processing changes
  // are cheap main-thread and applied instantly.
  const cfgTimer = useRef<number | undefined>(undefined);
  function liveAnalysis(cfg: AnalysisConfig) {
    setAnalysisCfg(cfg);
    saveSettings("svs.analysis", cfg);
    clearTimeout(cfgTimer.current);
    cfgTimer.current = window.setTimeout(() => engine.updateConfig(cfg), 200);
  }
  function liveProcessing(p: ProcessingSettings) {
    setProcCfg(p);
    saveSettings("svs.processing", p);
    processor.updateSettings(p);
  }

  return (
    <div
      style={{
        fontFamily: "system-ui, sans-serif",
        color: C.text,
        background: C.bg,
        height: "100vh",
        padding: 10,
        boxSizing: "border-box",
        display: "flex",
        flexDirection: narrow ? "column" : "row",
        gap: 8,
        overflowX: "hidden",
        overflowY: narrow ? "auto" : "hidden",
      }}
    >
      {/* Square scatter plot, sized to height when side-by-side, to width when stacked */}
      <div
        style={{
          flex: "none",
          height: narrow ? "auto" : "100%",
          width: narrow ? "100%" : "auto",
          aspectRatio: "1 / 1",
          minWidth: 0,
          minHeight: 0,
          background: C.canvas,
          border: `1px solid ${C.border}`,
          borderRadius: 4,
          overflow: "hidden",
          boxSizing: "border-box",
        }}
      >
        <MainPlot
          trail={trail}
          xRange={PLOT_PITCH_RANGE}
          yRange={PLOT_RESONANCE_RANGE}
        />
      </div>

      {/* Right: strips, readouts, controls, beside the plot, or below it when stacked */}
      <div
        style={{
          flex: narrow ? "none" : "1 1 0",
          minWidth: 0,
          width: narrow ? "100%" : undefined,
          height: narrow ? "auto" : "100%",
          minHeight: 0,
          overflowY: narrow ? "visible" : "auto",
          display: "flex",
          flexDirection: "column",
          gap: 8,
        }}
      >
        {/* Controls + live status (rms lives here, it's a settings/levels cue) */}
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          {!running ? (
            <button data-testid="start" onClick={start}>Start</button>
          ) : (
            <button data-testid="stop" onClick={stop}>Stop</button>
          )}
          <select
            value={deviceId}
            onChange={(e) => setDeviceId(e.target.value)}
            data-testid="device-select"
            style={{ maxWidth: 160 }}
          >
            <option value="">Default input</option>
            {devices.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>{d.label}</option>
            ))}
          </select>
          <div style={{ flex: 1, minWidth: 120 }}>
            <LevelMeter rms={rms} peak={peak} threshold={analysisCfg.rmsThreshold} />
          </div>
        </div>

        <div style={{ fontSize: 11, color: C.muted }}>
          <span data-testid="status">{running ? "running" : "stopped"}</span> ({source})
          {" · context: "}
          <span data-testid="ctx-state">{ctxState}</span>
          {deviceLabel ? ` · ${deviceLabel}` : ""}
          {" · RMS "}
          {rms.toFixed(4)}
        </div>
        {running && source === "mic" && ctxState === "suspended" && (
          <div style={{ fontSize: 11, color: C.error }}>
            Audio is suspended. Click anywhere on the page to resume.
          </div>
        )}
        {running && source === "mic" && ctxState === "running" && blocks === 0 && (
          <div style={{ fontSize: 11, color: C.error }}>
            No audio frames arriving from the mic. Check the selected input
            device and OS mic permissions for your browser.
          </div>
        )}

        {/* Left (strips → Core + Pitch settings) | right (readout → Formant
            settings, which fills the space under the shorter readout box). */}
        <div style={{ display: "flex", gap: 8, alignItems: "stretch" }}>
          <div style={{ flex: 2, minWidth: 0, display: "flex", flexDirection: "column", gap: 8 }}>
            <FormantStrips samplesRef={samplesRef} clockRef={clockRef} windowSec={STRIP_WINDOW_SEC} />
            <div style={{ display: "flex", gap: 8, alignItems: "stretch", flex: 1 }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <SettingsPanel
                  title="Core Settings"
                  fields={CORE_FIELDS}
                  analysis={analysisCfg}
                  processing={procCfg}
                  onAnalysis={liveAnalysis}
                  onProcessing={liveProcessing}
                />
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <SettingsPanel
                  title="Pitch Settings"
                  fields={PITCH_FIELDS}
                  analysis={analysisCfg}
                  processing={procCfg}
                  onAnalysis={liveAnalysis}
                  onProcessing={liveProcessing}
                />
              </div>
            </div>
          </div>
          <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 8 }}>
            <Readouts display={display} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <SettingsPanel
                title="Formant Settings"
                fields={FORMANT_FIELDS}
                analysis={analysisCfg}
                processing={procCfg}
                onAnalysis={liveAnalysis}
                onProcessing={liveProcessing}
              />
            </div>
          </div>
        </div>

        {error && (
          <pre style={{ color: C.error, whiteSpace: "pre-wrap", fontSize: 11, margin: 0 }}>
            {error}
          </pre>
        )}

        <div style={ABOUT_CARD}>
          {ABOUT_PAGES.length > 1 && (
            <div style={ABOUT_TABS}>
              {ABOUT_PAGES.map((p, i) => (
                <button
                  key={p.label}
                  type="button"
                  onClick={() => setAboutPage(i)}
                  style={aboutTabStyle(i === aboutPage)}
                >
                  {p.label}
                </button>
              ))}
            </div>
          )}
          {ABOUT_PAGES[aboutPage].body}
        </div>
      </div>
    </div>
  );
}
