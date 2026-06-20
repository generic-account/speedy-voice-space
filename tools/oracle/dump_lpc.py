"""Dump Praat's LPC coefficients for fixture windows.

Localizes whether our formant divergence lives in Burg/preprocessing (the LPC
coefficients) or in root extraction. Replicates Praat's Sound_to_Formant_burg
preprocessing as closely as parselmouth allows:
  1. Sound_resample(me, maximumFormant*2, 50)     -> fs = 2*maximumFormant
  2. To LPC (burg) order=2*nformants, width=window_length, preemph -> per-frame
     LPC. (To LPC burg internally pre-emphasizes + Gaussian-windows like
     Sound_to_Formant_burg.)
We read the frame nearest the resampled-window midpoint.

IMPORTANT: run with the repo venv python (system python's parselmouth hangs):
    .venv/bin/python tools/oracle/dump_lpc.py [stem] [window_index]
With no args, dumps a JSON for the four synthetic vowels to expected/lpc_dump.json
for the Rust comparator to read.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import soundfile as sf
import parselmouth
from parselmouth.praat import call

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIXTURES = os.path.join(ROOT, "tools", "audio", "fixtures")
EXPECTED = os.path.join(os.path.dirname(__file__), "expected")
SAMPLE_SCALE = 32768

VOWELS = [
    "vowel_u_130hz",
    "vowel_u_nasal_130hz",
    "vowel_a_res_130hz",
    "vowel_a_nasal_130hz",
]


def load_windows(stem):
    with open(os.path.join(EXPECTED, f"{stem}.json")) as f:
        o = json.load(f)
    raw, sr = sf.read(os.path.join(FIXTURES, f"{stem}.wav"), dtype="int16", always_2d=False)
    if raw.ndim > 1:
        raw = raw[:, 0]
    x = raw.astype(np.float64) / SAMPLE_SCALE
    voiced = [w for w in o["windows"] if w.get("voiced") and w.get("formant_freq_hz")]
    return o, x, voiced


def praat_lpc_coeffs(frame, cfg):
    sr = float(cfg["samplerate"])
    maxf = float(cfg["maximum_formant_hz"])
    nform = float(cfg["max_number_of_formants"])
    wlen = float(cfg["window_length_s"])
    preemph = float(cfg["pre_emphasis_from_hz"])
    order = int(round(2 * nform))

    snd = parselmouth.Sound(np.asarray(frame, dtype=np.float64), sampling_frequency=sr)
    rs = call(snd, "Resample", 2 * maxf, 50)
    lpc = call(rs, "To LPC (burg)", order, wlen, 0.005, preemph)
    nframes = int(call(lpc, "Get number of frames"))
    tmid = rs.get_total_duration() / 2.0
    fi = int(max(1, min(nframes, round(call(lpc, "Get frame number from time", tmid)))))

    path = tempfile.mktemp(suffix=".LPC")
    lpc.save_as_short_text_file(path)
    with open(path) as f:
        nums = [ln.strip() for ln in f
                if ln.strip() and '"' not in ln and '=' not in ln]
    os.unlink(path)
    # numeric layout: xmin xmax nx dx x1 samplingPeriod maxnCoefficients,
    # then per frame: nCoefficients, coeff_1..coeff_n, gain.
    body = nums[7:]
    frames = []
    i = 0
    while i < len(body):
        nc = int(round(float(body[i]))); i += 1
        if nc <= 0 or i + nc > len(body):
            raise ValueError(f"bad LPC frame parse nc={nc} at i={i}")
        cs = [float(body[i + k]) for k in range(nc)]; i += nc
        i += 1  # gain
        frames.append(cs)
    return order, frames[fi - 1], 2 * maxf


def main():
    if len(sys.argv) > 1 and sys.argv[1] not in ("--dump",):
        stem = sys.argv[1]
        widx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        o, x, voiced = load_windows(stem)
        w = voiced[widx]
        frame = x[w["start_sample"]: w["start_sample"] + o["window_samples"]]
        order, cs, fs = praat_lpc_coeffs(frame, o["config"])
        print(json.dumps({
            "stem": stem, "window_index": widx, "start_sample": w["start_sample"],
            "new_fs": fs, "order": order, "praat_lpc": cs,
            "praat_formant_freq_hz": w.get("formant_freq_hz"),
            "praat_formant_bw_hz": w.get("formant_bw_hz"),
        }, indent=2))
        return

    # default: dump a few windows per vowel for the Rust comparator
    out = {}
    for stem in VOWELS:
        print("processing", stem, flush=True)
        o, x, voiced = load_windows(stem)
        entries = []
        for widx in (0, len(voiced) // 2):
            print("  window", widx, flush=True)
            w = voiced[widx]
            frame = x[w["start_sample"]: w["start_sample"] + o["window_samples"]]
            order, cs, fs = praat_lpc_coeffs(frame, o["config"])
            entries.append({
                "window_index": widx, "start_sample": w["start_sample"],
                "window_samples": o["window_samples"], "new_fs": fs, "order": order,
                "praat_lpc": cs,
                "praat_formant_freq_hz": w.get("formant_freq_hz"),
                "praat_formant_bw_hz": w.get("formant_bw_hz"),
            })
        out[stem] = {"config": o["config"], "samplerate": o["samplerate"],
                     "sample_scale": o["sample_scale"], "entries": entries}
    dest = os.path.join(EXPECTED, "lpc_dump.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", dest)


if __name__ == "__main__":
    main()
