"""
Formant smoothing/robustness experiments.

For each recording: ground-truth Praat formants vs our (wasm) raw formants, then
apply candidate CAUSAL post-filters (usable in the realtime app) and measure:
  - acc   : median |our-praat|/praat % over voiced frames (accuracy vs Praat)
  - glitch: % frames with |our-praat|/praat > 30% (the visible noise)
  - jitter: median |Δ| Hz frame-to-frame (lower = smoother)

Goal: cut glitch+jitter WITHOUT raising acc. Sustained vowels (rec_u5, rec_a7)
are the app's use case; rec_4 (connected speech) is the do-no-harm control.

Prereq: CSVs from web/scripts/wasm_formants.mjs at /tmp/ours_<tag>.csv
Run:    python tools/experiments/formant_smoothing.py
"""
import sys, csv
sys.path.insert(0, "/Users/glichtstein/Documents/speedy-voice-space")
import numpy as np, soundfile as sf
from analysis import analyze_window, AnalysisConfig

cfg = AnalysisConfig()
FIX = "/Users/glichtstein/Documents/speedy-voice-space/tools/audio/fixtures"
CASES = [("rec_u5", "/tmp/ours_u5.csv"), ("rec_a7", "/tmp/ours_a7.csv"),
         ("rec_user_4", "/tmp/ours_rec4.csv")]


def praat_formants(stem):
    raw, sr = sf.read(f"{FIX}/{stem}.wav", dtype="int16")
    if raw.ndim > 1:
        raw = raw[:, 0]
    x = raw.astype(np.float64) / 32768.0
    win = int(round(cfg.buffer_duration_s * sr)); hop = int(round(0.02 * sr))
    out = {}
    i = 0
    while i + win <= len(x):
        r = analyze_window(x[i:i+win], cfg); t = round((i+win/2)/sr, 4)
        if r.voiced and len(r.formants_hz) >= 3:
            out[t] = (r.formants_hz[0], r.formants_hz[1], r.formants_hz[2])
        i += hop
    return out


def load_ours(path):
    out = {}
    for row in csv.DictReader(open(path)):
        t = round(float(row["t"]), 4)
        def g(k):
            return float(row[k]) if row[k] else np.nan
        out[t] = (g("f1"), g("f2"), g("f3"))
    return out


# ---- causal filters ----
def f_identity(a):
    return a.copy()

def f_median(a, w):
    out = a.copy()
    for k in range(len(a)):
        seg = a[max(0, k-w+1):k+1]; seg = seg[~np.isnan(seg)]
        if len(seg):
            out[k] = np.median(seg)
    return out

def f_hampel(a, w=7, n=3.0):
    """Causal Hampel: replace only statistical outliers w/ trailing median."""
    out = a.copy()
    for k in range(len(a)):
        seg = a[max(0, k-w+1):k+1]; seg = seg[~np.isnan(seg)]
        if len(seg) < 3 or np.isnan(a[k]):
            continue
        med = np.median(seg); mad = np.median(np.abs(seg - med))
        if mad > 0 and abs(a[k] - med) > n * 1.4826 * mad:
            out[k] = med
    return out

def f_hampel_ema(a, w=7, n=3.0, alpha=0.4):
    h = f_hampel(a, w, n)
    out = h.copy(); prev = np.nan
    for k in range(len(h)):
        if np.isnan(h[k]):
            continue
        prev = h[k] if np.isnan(prev) else alpha*h[k] + (1-alpha)*prev
        out[k] = prev
    return out


FILTERS = {
    "raw": lambda a: f_identity(a),
    "median3": lambda a: f_median(a, 3),
    "median5": lambda a: f_median(a, 5),
    "hampel7n3": lambda a: f_hampel(a, 7, 3.0),
    "hampel5n2.5": lambda a: f_hampel(a, 5, 2.5),
    "hampel7n3+ema.4": lambda a: f_hampel_ema(a, 7, 3.0, 0.4),
}


def metrics(ours_series, praat_series):
    o = np.array(ours_series); p = np.array(praat_series)
    m = ~np.isnan(o) & ~np.isnan(p)
    o, p = o[m], p[m]
    if len(o) == 0:
        return None
    err = np.abs(o - p) / p * 100
    jit = np.median(np.abs(np.diff(o))) if len(o) > 1 else 0.0
    return np.median(err), 100*np.mean(err > 30), jit


def run():
    for stem, csvpath in CASES:
        praat = praat_formants(stem)
        ours = load_ours(csvpath)
        ts = [t for t in sorted(praat) if t in ours]
        for slot, name in [(2, "F3"), (1, "F2")]:
            o = np.array([ours[t][slot] for t in ts])
            p = np.array([praat[t][slot] for t in ts])
            print(f"\n[{stem} {name}] {len(ts)} voiced frames")
            print(f"  {'filter':16s} {'acc%':>6} {'glitch%':>8} {'jitterHz':>9}")
            for fname, fn in FILTERS.items():
                r = metrics(fn(o), p)
                if r:
                    print(f"  {fname:16s} {r[0]:6.1f} {r[1]:8.0f} {r[2]:9.0f}")


if __name__ == "__main__":
    run()
