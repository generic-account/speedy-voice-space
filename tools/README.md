# Validation & test infrastructure

Tooling that lets the web rewrite be validated **without a human and a
microphone**, and that keeps the WASM DSP port numerically honest against the
working Python implementation.

## `audio/` — deterministic fixtures

`gen_fixtures.py` writes 48 kHz / 16-bit / mono WAVs to `audio/fixtures/`:

| Fixture | Purpose |
|---|---|
| `silence.wav` | gate / noise-floor checks (should be unvoiced, ~0 RMS) |
| `tone_150hz.wav`, `tone_220hz.wav` | exact known pitch |
| `vowel_a_150hz.wav`, `vowel_i_220hz.wav` | known f0 + formant targets |
| `sweep_120_300hz.wav` | pitch glide → smoothing/median behavior |
| `real_speech_*.wav` | real voiced/unvoiced speech (OSR Harvard sentences, public domain) |

These feed two consumers: the browser pipeline tests (decoded → AudioWorklet)
and the parity oracle below.

```
python tools/audio/gen_fixtures.py
```

## `oracle/` — Python golden data (parity gate)

`gen_oracle.py` runs the Python `analyze_window(frame, config)` (the stateless
contract extracted into `analysis.py`) over each fixture and dumps expected
`rms / voiced / pitch_hz / formants_hz` per analysis window to
`oracle/expected/*.json`.

The **equivalency contract** (both Python and the Rust port must follow):
- read the same 16-bit PCM mono 48 kHz WAV,
- scale samples `i16 / 32768` → float,
- take `window_samples`-long windows at each `start_sample`,
- run `analyze_window`, compare at the window midpoint.

When the Rust pitch/formant port lands, `dsp/tests/parity.rs` reads these JSON
files, re-extracts the same windows from the same WAVs, and asserts agreement
within tolerance (pitch ~1 Hz / 1%, formants ~3–5%). This is the acceptance gate
for the algorithm port.

```
python tools/oracle/gen_oracle.py     # regenerate after any analysis change
```

## Browser pipeline tests

`web/tests/pipeline.spec.ts` (Playwright) drives the real `AudioWorklet` path via
a **file source** (decoded WAV → same worklet), avoiding Chromium's flaky
headless mic emulation. Asserts blocks flow at 48 kHz and the level meter
reflects signal vs. silence. The live `getUserMedia` mic path is validated
manually in a real browser.

```
cd web && npm run test:e2e
```
