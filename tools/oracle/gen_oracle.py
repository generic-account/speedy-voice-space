"""
Parity oracle generator.

Runs the Python (Praat/Parselmouth) analysis over each audio fixture and dumps
the expected pitch + formant values per analysis window to JSON. The Rust/WASM
port's test suite re-extracts the same windows from the same WAVs and asserts it
matches these numbers within tolerance — this is the acceptance gate for the
algorithm port (see ARCHITECTURE.md §4).

Equivalency contract:
  - Both sides read the SAME 16-bit PCM mono 48 kHz WAV.
  - Samples are scaled i16 -> float by dividing by `sample_scale` (32768).
  - Each window is `window_samples` long starting at `start_sample`.
  - Each side runs its `analyze_window(frame, config)` and compares
    rms / voiced / pitch_hz / formants_hz at the window midpoint.

Run:  python tools/oracle/gen_oracle.py
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys

import numpy as np
import soundfile as sf

# Import the shared analysis contract from the repo root.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import parselmouth  # noqa: E402
from analysis import analyze_window, AnalysisConfig  # noqa: E402

FIXTURES = os.path.join(ROOT, "tools", "audio", "fixtures")
OUT = os.path.join(os.path.dirname(__file__), "expected")
SAMPLE_SCALE = 32768  # i16 -> float divisor (must match the Rust reader)

# Fixtures to characterize (48 kHz PCM16 mono).
FILES = [
    "tone_150hz.wav",
    "tone_220hz.wav",
    "vowel_a_150hz.wav",
    "vowel_i_220hz.wav",
    "vowel_u_130hz.wav",
    "vowel_a_res_130hz.wav",
    "vowel_u_nasal_130hz.wav",
    "vowel_a_nasal_130hz.wav",
    "sweep_120_300hz.wav",
    "real_speech_48k.wav",
    "silence.wav",
]

HOP_S = 0.05          # window stride
MAX_WINDOWS = 120     # cap per file to keep JSON lean (esp. long speech)


def formants_with_bandwidths(frame, config):
    """Praat formant freqs + bandwidths at the window midpoint, all slots.

    Mirrors analysis.analyze_window's to_formant_burg call exactly, but also
    pulls get_bandwidth_at_time so diagnostics can compare bandwidths too.
    """
    snd = parselmouth.Sound(
        np.asarray(frame, dtype=np.float64),
        sampling_frequency=float(config.samplerate),
    )
    fo = snd.to_formant_burg(
        time_step=config.formant_time_step,
        max_number_of_formants=config.max_number_of_formants,
        maximum_formant=config.maximum_formant_hz,
        window_length=config.window_length_s,
        pre_emphasis_from=config.pre_emphasis_from_hz,
    )
    t = snd.get_total_duration() / 2.0
    freqs, bws = [], []
    for i in range(1, int(config.max_number_of_formants) + 1):
        f = fo.get_value_at_time(i, t)
        if f is None or (isinstance(f, float) and np.isnan(f)):
            continue
        b = fo.get_bandwidth_at_time(i, t)
        freqs.append(float(f))
        bws.append(0.0 if b is None or np.isnan(b) else float(b))
    return freqs, bws


def process_file(name: str, config: AnalysisConfig) -> dict | None:
    path = os.path.join(FIXTURES, name)
    if not os.path.exists(path):
        print(f"  skip {name} (missing)")
        return None

    info = sf.info(path)
    if info.samplerate != config.samplerate:
        print(f"  WARN {name}: sr={info.samplerate} != {config.samplerate}")

    # Read exactly as the Rust side will: i16 PCM -> float / 32768.
    raw, sr = sf.read(path, dtype="int16", always_2d=False)
    if raw.ndim > 1:
        raw = raw[:, 0]
    x = raw.astype(np.float64) / SAMPLE_SCALE

    win = int(round(config.buffer_duration_s * sr))
    hop = int(round(HOP_S * sr))

    windows = []
    start = 0
    while start + win <= len(x) and len(windows) < MAX_WINDOWS:
        frame = x[start : start + win]
        r = analyze_window(frame, config)
        freqs, bws = formants_with_bandwidths(frame, config) if r.voiced else ([], [])
        windows.append(
            {
                "start_sample": int(start),
                "t_mid_s": round((start + win / 2) / sr, 6),
                "rms": round(r.rms, 8),
                "voiced": bool(r.voiced),
                "pitch_hz": None if r.pitch_hz is None else round(r.pitch_hz, 4),
                "formants_hz": [round(f, 4) for f in r.formants_hz],
                # All Praat formant slots with bandwidths (not NaN-compacted), so
                # the Rust diagnostics can align pole-for-pole incl. bandwidth.
                "formant_freq_hz": [round(f, 4) for f in freqs],
                "formant_bw_hz": [round(b, 4) for b in bws],
            }
        )
        start += hop

    voiced_n = sum(1 for w in windows if w["voiced"])
    print(f"  {name:22s} {len(windows):4d} windows  ({voiced_n} voiced)")

    return {
        "file": name,
        "samplerate": int(sr),
        "sample_scale": SAMPLE_SCALE,
        "window_samples": win,
        "hop_samples": hop,
        "config": dataclasses.asdict(config),
        "parselmouth_version": parselmouth.VERSION,
        "windows": windows,
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    config = AnalysisConfig()  # defaults = the app's defaults
    print(f"Oracle → {OUT}")
    print(f"window={config.buffer_duration_s}s hop={HOP_S}s cap={MAX_WINDOWS}")

    index = []
    for name in FILES:
        data = process_file(name, config)
        if data is None:
            continue
        out_path = os.path.join(OUT, name.replace(".wav", ".json"))
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        index.append(os.path.basename(out_path))

    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump({"files": index}, f, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
