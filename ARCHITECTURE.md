# Web Rewrite Architecture — speedy-voice-space

A fully client-side, offline-capable rewrite of the Python/PyQt real-time voice
trainer. No server, no upload of audio, statically hostable (GitHub Pages /
Netlify / any CDN). Heavy DSP runs in WebAssembly off the main thread.

---

## 1. Goals & constraints

- **100% user-side.** Audio never leaves the device. App is a static bundle.
- **Real-time.** End-to-end latency comparable to the desktop app (block ≈ 21 ms
  at 1024/48 kHz; analysis cadence ~30 Hz).
- **Faithful to Praat.** Pitch/formant numbers must track the current app's
  output closely — the resonance score, ranges, and defaults are tuned to it.
- **Minimal dependencies.** Self-contained WASM + a thin React UI. No audio
  data services, no native add-ons, no dependency hell (the very problem we're
  escaping: `av`/`audiolab`/`pyrnnoise` broke on Python 3.13).

---

## 2. Source app → web mapping

| Python module | Responsibility | Web replacement |
|---|---|---|
| `audio.py` (`sounddevice`) | Mic capture, device list, 1024-sample mono blocks | `getUserMedia` + `AudioWorklet`; `enumerateDevices` for the picker |
| `denoise.py` (`pyrnnoise`) | RNNoise 48 kHz / 480-frame suppression + dry/wet mix | `nnnoiseless` (pure-Rust RNNoise) in WASM — same 48 kHz / 480 frame model |
| `analysis.py` (`parselmouth`) | RMS gate → `to_pitch_ac` + `to_formant_burg` | **Ported** Boersma AC pitch + Burg LPC formants in Rust→WASM |
| `processing.py` | Median filter, EMA smoothing, mel-normalized resonance | Pure TypeScript (no deps) |
| `ui.py` (`PyQt6` + `pyqtgraph`) | Scatter trail, F2/F3 strip charts, ~30 settings | React + a canvas/WebGL plot lib |
| `settings_defaults.py` | All default constants | `defaults.ts` — single source of truth, ported verbatim |

The desktop app only calls **two** Praat functions. That's the entire DSP port
surface — not "compile Praat," just two well-specified algorithms.

---

## 3. Threading model

```
┌──────────────────────────────────────────────────────────────────────┐
│ AudioWorklet (audio render thread)                                     │
│   getUserMedia → mono frames (128-sample renders, coalesced to blocks) │
│   posts Float32Array block (transferable) ─────────────┐              │
└─────────────────────────────────────────────────────────┼────────────┘
                                                           │ postMessage (zero-copy transfer)
┌──────────────────────────────────────────────────────────▼────────────┐
│ DSP Web Worker  (owns the WASM module)                                  │
│   1. RNNoise denoise (optional) + dry/wet mix   [nnnoiseless]           │
│   2. push into rolling buffer (buffer_duration_s * sr)                  │
│   3. RMS gate                                                           │
│   4. pitch  = ac_pitch(buffer, cfg)             [ported]                │
│   5. formants = burg_formants(buffer, cfg)      [ported]                │
│   6. postMessage(AnalysisResult) ───────────────────────┐              │
└──────────────────────────────────────────────────────────┼────────────┘
                                                           │ postMessage (small struct, ~30 Hz)
┌──────────────────────────────────────────────────────────▼────────────┐
│ Main thread (React)                                                     │
│   VoiceProcessor: median + EMA + resonance  [pure TS]                  │
│   render: scatter trail, F2/F3 strips, readouts (requestAnimationFrame)│
│   settings panel → posts config to Worker                              │
└────────────────────────────────────────────────────────────────────────┘
```

**Why a Worker, not the AudioWorklet, for analysis:** Burg LPC + autocorrelation
over a ~3000-sample window is too heavy for the audio render quantum (128
samples / 2.7 ms). Doing it in the Worker keeps the audio thread glitch-free.

**SharedArrayBuffer:** *Not required for v1.* We transfer `Float32Array` blocks
(zero-copy via `postMessage` transfer list) at ~47 blocks/s — cheap. This avoids
the COOP/COEP cross-origin-isolation headers that GitHub Pages can't set. If we
later want a lock-free ring buffer (SAB), document the `coi-serviceworker` shim
as the upgrade path. **Decision for v1: no SAB.**

---

## 4. The WASM core (Rust)

Single crate `dsp/`, built with `wasm-pack` → `wasm32-unknown-unknown`. One
`.wasm` + JS glue, loaded by the Worker. Pure Rust — **no Emscripten, no C
toolchain.**

```
dsp/
  Cargo.toml          # deps: nnnoiseless, (no_std-friendly math), wasm-bindgen
  src/
    lib.rs            # #[wasm_bindgen] facade: Engine { push_block, set_config }
    denoise.rs        # nnnoiseless wrapper: 48k resample, 480-frame, dry/wet mix
    pitch.rs          # Boersma autocorrelation (to_pitch_ac port)
    formant.rs        # Burg LPC + polynomial root-finding (to_formant_burg port)
    resample.rs       # polyphase resample (mirrors scipy.signal.resample_poly)
    buffer.rs         # rolling deque buffer
```

### Algorithm ports (the hard, high-value part)

**Pitch — `to_pitch_ac` (Boersma 1993, autocorrelation method):**
1. Window the frame (Hanning/Gaussian per Praat).
2. Autocorrelation of the windowed signal ÷ autocorrelation of the window
   (sinc-interpolated normalization — this is what makes it Praat, not naïve AC).
3. Find peaks in the lag domain within `[1/ceiling, 1/floor]`.
4. Parabolic interpolation for sub-sample peak refinement.
5. Voicing decision via `silence_threshold` / `voicing_threshold`.
   - Inputs we must honor: `pitch_floor_hz`, `pitch_ceiling_hz`,
     `pitch_silence_threshold`, `pitch_voicing_threshold`, `pitch_very_accurate`,
     `pitch_time_step`.
   - For realtime we evaluate at the buffer midpoint (matches current
     `get_value_at_time(total_duration/2)`).

**Formants — `to_formant_burg`:**
1. Pre-emphasis from `pre_emphasis_from_hz`.
2. Resample to `2 * maximum_formant_hz` (Praat's formant ceiling convention).
3. Gaussian-like window of `window_length_s`.
4. Burg method → LPC coefficients (order ≈ `2 * max_number_of_formants`).
5. Solve LPC polynomial roots → formant freqs/bandwidths from root angles.
6. Return F1..Fn at buffer midpoint.

### Validation strategy — Python as golden oracle

The existing Python still runs (core verified). Build a fixture harness:

```
tools/oracle/
  gen_fixtures.py     # feed WAVs/synthetic vowels through parselmouth,
                      # dump (frame, cfg) → expected pitch/F1..F5 as JSON
  fixtures/*.json
dsp/tests/parity.rs   # run ported Rust over same frames, assert within tolerance
```

Tolerance target: pitch within ~1 Hz / 1%, formants within ~3–5%. This is the
acceptance gate for the port — we don't ship until parity passes on a corpus of
sustained vowels across pitch/resonance ranges.

---

## 5. Frontend (React)

```
web/
  index.html
  vite.config.ts          # static build; base path for GH Pages
  src/
    main.tsx
    App.tsx
    audio/
      engine.ts           # orchestrates AudioWorklet ↔ Worker ↔ React
      capture-worklet.ts  # AudioWorklet processor (block coalescing)
      dsp-worker.ts       # loads WASM, runs denoise+pitch+formant
    dsp/                  # wasm-pack output (generated)
    processing/
      voiceProcessor.ts   # port of processing.py (median, EMA, mel resonance)
      defaults.ts         # port of settings_defaults.py
    components/
      MainPlot.tsx        # pitch × resonance scatter trail (fading alpha)
      FormantStrips.tsx   # F2/time and F3/time strip charts
      SettingsPanel.tsx   # ~30 controls, grouped (audio/pitch/formant/processing)
      Readouts.tsx        # pitch, resonance, confidence, RMS, formants
    state/
      useSettings.ts      # config store; debounced push to Worker
```

- **Plot library:** `uPlot` (tiny, canvas, built for high-frequency streaming) for
  the F2/F3 strip charts; the pitch×resonance scatter trail is a small custom
  canvas component (fading-alpha dots — `uPlot` is overkill for a 120-point
  trail). React owns layout/settings; canvas owns the hot render path via
  `requestAnimationFrame`, decoupled from React re-renders.
- **Settings parity:** every control in `ui.py` maps 1:1 (same ranges, steps,
  defaults). "Apply Settings" semantics → push config to Worker, reset buffers.
- **Device picker:** `enumerateDevices` (labels appear only after mic permission,
  mirror the current dropdown). Sample-rate fallback logic (`audio.py`'s
  candidate-rate list) → AudioContext `sampleRate` handling.

---

## 6. Repo layout (target)

```
speedy-voice-space/
  dsp/            # Rust → WASM DSP core
  web/            # React app (Vite)
  tools/oracle/   # Python parity-fixture generator (uses existing parselmouth)
  legacy-python/  # current app, kept as reference + oracle (moved, not deleted)
  ARCHITECTURE.md # this file
  README.md
```

Build: `wasm-pack build dsp --target web` → `web/src/dsp/`; `npm run build` in
`web/` → static `dist/`. CI: run `parity.rs` against committed fixtures.

---

## 7. Phased plan

1. **Scaffold + capture.** Vite+React app, AudioWorklet mic capture, level meter.
   Proves the audio path end-to-end with zero DSP. *(node toolchain)*
2. **Oracle fixtures.** `gen_fixtures.py` over synthetic vowels + a few WAVs.
   Locks the numerical target before porting.
3. **Pitch port.** `pitch.rs` + `parity.rs`; wire Worker→UI; live pitch readout +
   trail X-axis. Gate on parity.
4. **Formant port.** `formant.rs`; F2/F3 strips; resonance score (port
   `processing.py`). Gate on parity.
5. **Denoise.** `nnnoiseless` + dry/wet mix + the suppression toggle/mix controls.
6. **Full UI.** All ~30 settings, readouts, polish, persistence (localStorage).
7. **Ship.** Static build, COOP/COEP-free; document SAB upgrade path.

---

## 8. Key risks

- **Praat parity (highest).** The AC/Burg ports must match Praat's specific
  normalization and windowing, not textbook variants. Mitigation: oracle
  fixtures + tolerance gate from day one.
- **AudioContext sample rate.** Browsers often force 48 kHz; the desktop app
  negotiates rates. We standardize on 48 kHz internally (also RNNoise's native
  rate) and resample only where an algorithm needs a different ceiling.
- **Latency/jitter.** postMessage cadence + GC. Mitigation: transferables, reuse
  buffers, keep the Worker hot; revisit SAB only if measured jitter demands it.
- **Mobile/Safari quirks.** AudioWorklet + WASM are supported but
  permission/autoplay flows differ; test early.
```
