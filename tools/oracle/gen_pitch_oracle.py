"""
Whole-file pitch oracle ("best Praat").

The main oracle (gen_oracle.py) runs Praat per analysis window, which has no
cross-frame continuity. This one runs Praat's `to_pitch_ac` over the WHOLE file
once (its Viterbi path penalizes octave jumps) and samples the resulting contour
at each window midpoint. That contour is the best-case reference the live
real-time (per-window) algorithm is measured against.

Reuses the window layout (start_sample, t_mid_s) from expected/<fixture>.json so
the Rust test analyzes exactly the same frames.

Run:  .venv/bin/python tools/oracle/gen_pitch_oracle.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import soundfile as sf
import parselmouth

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIXTURES = os.path.join(ROOT, "tools", "audio", "fixtures")
OUT = os.path.join(os.path.dirname(__file__), "expected_pitch")

PITCH_FLOOR = 75.0
PITCH_CEILING = 400.0
TIME_STEP = 0.01
SAMPLE_SCALE = 32768
WINDOW_S = 0.06  # app analysis buffer
HOP_S = 0.02     # ~live block cadence (compare at the rate the app actually runs)
MAX_WINDOWS = 300

FILES = [
    "tone_150hz.wav",
    "tone_220hz.wav",
    "vowel_a_150hz.wav",
    "vowel_i_220hz.wav",
    "vowel_u_130hz.wav",
    "real_speech_48k.wav",
]


def process(name: str) -> dict | None:
    wav_path = os.path.join(FIXTURES, name)
    if not os.path.exists(wav_path):
        print(f"  skip {name} (missing wav)")
        return None

    raw, sr = sf.read(wav_path, dtype="int16", always_2d=False)
    if raw.ndim > 1:
        raw = raw[:, 0]
    x = raw.astype(np.float64) / SAMPLE_SCALE

    snd = parselmouth.Sound(x, sampling_frequency=float(sr))
    pitch = snd.to_pitch_ac(
        time_step=TIME_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING
    )

    win = int(round(WINDOW_S * sr))
    hop = int(round(HOP_S * sr))
    windows = []
    voiced = 0
    start = 0
    while start + win <= len(x) and len(windows) < MAX_WINDOWS:
        t_mid = round((start + win / 2) / sr, 6)
        f = pitch.get_value_at_time(t_mid)
        hz = None if f is None or np.isnan(f) else round(float(f), 4)
        if hz is not None:
            voiced += 1
        windows.append({"start_sample": int(start), "t_mid_s": t_mid, "whole_pitch_hz": hz})
        start += hop

    print(f"  {name:22s} {len(windows):4d} windows  ({voiced} voiced)")
    return {
        "file": name,
        "samplerate": int(sr),
        "sample_scale": SAMPLE_SCALE,
        "window_samples": win,
        "pitch_floor_hz": PITCH_FLOOR,
        "pitch_ceiling_hz": PITCH_CEILING,
        "parselmouth_version": parselmouth.VERSION,
        "windows": windows,
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    print(f"Whole-file pitch oracle -> {OUT}")
    index = []
    for name in FILES:
        data = process(name)
        if data is None:
            continue
        with open(os.path.join(OUT, name.replace(".wav", ".json")), "w") as f:
            json.dump(data, f, indent=2)
        index.append(name.replace(".wav", ".json"))
    json.dump({"files": index}, open(os.path.join(OUT, "index.json"), "w"), indent=2)
    print("done.")


if __name__ == "__main__":
    main()
