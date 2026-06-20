"""
Robust formant parity oracle generator.

Mirrors tools/oracle/gen_oracle.py exactly (same windowing, sample reading, hop),
but dumps Praat's *robust* formants (IRLS-reweighted LPC, Lee 1988) via
parselmouth's "To Formant (robust)..." instead of to_formant_burg. The Rust port
of the robust method is validated against these numbers in
dsp/tests/formant_robust_parity.rs.

Robust parameters (Praat defaults): number_of_std_dev = 1.5, max iterations = 5,
tolerance = 1e-6.

Run:  python tools/oracle/gen_robust_oracle.py
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import parselmouth  # noqa: E402
from parselmouth.praat import call  # noqa: E402
from analysis import analyze_window, AnalysisConfig  # noqa: E402

FIXTURES = os.path.join(ROOT, "tools", "audio", "fixtures")
OUT = os.path.join(os.path.dirname(__file__), "expected_robust")
SAMPLE_SCALE = 32768  # i16 -> float divisor (must match the Rust reader)

# COMMITTED fixtures only (synthetic + OSR public-domain). No rec_* recordings.
FILES = [
    "vowel_a_150hz.wav",
    "vowel_i_220hz.wav",
    "vowel_u_130hz.wav",
    "vowel_a_res_130hz.wav",
    "vowel_u_nasal_130hz.wav",
    "vowel_a_nasal_130hz.wav",
    "real_speech_48k.wav",
    "silence.wav",
]

HOP_S = 0.05
MAX_WINDOWS = 120

# Praat robust knobs.
NUM_STD_DEV = 1.5
MAX_ITER = 5
TOLERANCE = 1e-6


def robust_formants(frame, config):
    """Praat robust formant freqs + bandwidths at the window midpoint, all slots."""
    snd = parselmouth.Sound(
        np.asarray(frame, dtype=np.float64),
        sampling_frequency=float(config.samplerate),
    )
    rob = call(
        snd,
        "To Formant (robust)...",
        config.formant_time_step,
        float(config.max_number_of_formants),
        config.maximum_formant_hz,
        config.window_length_s,
        config.pre_emphasis_from_hz,
        NUM_STD_DEV,
        MAX_ITER,
        TOLERANCE,
    )
    t = snd.get_total_duration() / 2.0
    freqs, bws = [], []
    for i in range(1, int(config.max_number_of_formants) + 1):
        f = call(rob, "Get value at time", i, t, "hertz", "linear")
        if f is None or (isinstance(f, float) and np.isnan(f)):
            continue
        b = call(rob, "Get bandwidth at time", i, t, "hertz", "linear")
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
        freqs, bws = robust_formants(frame, config) if r.voiced else ([], [])
        windows.append(
            {
                "start_sample": int(start),
                "t_mid_s": round((start + win / 2) / sr, 6),
                "rms": round(r.rms, 8),
                "voiced": bool(r.voiced),
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
        "robust": {
            "number_of_std_dev": NUM_STD_DEV,
            "max_iterations": MAX_ITER,
            "tolerance": TOLERANCE,
        },
        "parselmouth_version": parselmouth.VERSION,
        "windows": windows,
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    config = AnalysisConfig()
    print(f"Robust oracle -> {OUT}")
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
