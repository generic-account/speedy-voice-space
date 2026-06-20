# Voice Space — web (client-side rewrite)

100% client-side rewrite of the Python/PyQt voice trainer. Mic audio never
leaves the browser; DSP runs in WebAssembly off the main thread.

## Architecture (see ../ARCHITECTURE.md for the full picture)

```
AudioWorklet (capture-worklet.js)   mic OR decoded WAV → 1024-sample blocks
        │ transfer
   AudioEngine (src/audio/engine.ts) pluggable source + stats
        │ postMessage(block)
   DSP Worker (src/audio/dsp-worker.ts)  loads ../dsp wasm; rolling buffer,
        │ postMessage(result)            RMS gate, pitch, formants, denoise
   VoiceProcessor (src/processing/)  median + EMA + mel resonance  (port of processing.py)
        │
   React UI (src/components/)  scatter trail, F2/F3 strips, settings, readouts
```

The DSP core is the Rust crate in `../dsp` compiled to wasm; the algorithms are
validated against the Python reference (see `../tools/oracle`).

## Develop

```bash
npm install
npm run dev            # http://localhost:5173
```

The wasm package is built from the Rust crate into `src/dsp/`:

```bash
# from repo root, with rust + wasm-pack installed:
cd dsp && wasm-pack build --target web --out-dir ../web/src/dsp
```

## Test

```bash
npm run test:e2e       # Playwright
```

- `tests/pipeline.spec.ts` — capture path (file source → AudioWorklet → blocks).
- `tests/wasm-pipeline.spec.ts` — **end-to-end wasm link test**: a fixture driven
  through the worker's wasm analyzer must produce correct pitch. This guards
  against the Rust/wasm/worker integration silently breaking as the app grows.

Tests drive a deterministic **file source** (decoded WAV) rather than Chromium's
unreliable headless mic emulation; the live `getUserMedia` mic path is validated
manually in a real browser.

## Build

```bash
npm run build          # static bundle in dist/ (relative base, CDN/Pages-ready)
```
