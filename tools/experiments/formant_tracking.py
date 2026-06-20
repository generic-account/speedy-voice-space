"""
Experiment: formant continuity TRACKING vs raw slot-by-frequency assignment.

Our wasm returns candidate poles F1..F4 sorted by frequency. On hard frames the
true F2 lands in the F3 slot (a "collapse"). A causal tracker that assigns the
3 lowest tracks to the candidates by continuity (nearest to a robust running
estimate, keeping ascending order) should recover the real F2/F3 — fixing the
collapses WITHOUT the lag/smearing of a blanket median.

Compares, per recording, raw vs tracked F2/F3 against Praat:
  acc% (median err), glitch% (>30% off), jitterHz (median |Δ|).
"""
import sys, csv, itertools
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
            out[t] = r.formants_hz[:3]
        i += hop
    return out


def load_candidates(path):
    """t -> sorted list of candidate pole freqs (f1..f4)."""
    out = {}
    for row in csv.DictReader(open(path)):
        t = round(float(row["t"]), 4)
        cands = []
        for k in ("f1", "f2", "f3", "f4"):
            if row.get(k):
                cands.append(float(row[k]))
        out[t] = sorted(cands)
    return out


def track(cands_by_t, ts, alpha=0.3):
    """Causal continuity tracker for F1,F2,F3.

    Running estimate `est` per track (EMA). Each frame, choose 3 ascending
    candidates minimizing sum |cand-est| (+ small freq-order prior); update est.
    Falls back to raw sorted slots until initialized.
    """
    est = None
    out = {}
    for t in ts:
        c = cands_by_t.get(t, [])
        if len(c) < 3:
            out[t] = (c + [np.nan, np.nan, np.nan])[:3]
            continue
        if est is None:
            est = [c[0], c[1], c[2]]
            out[t] = (c[0], c[1], c[2])
            continue
        # Choose best ascending triple of candidates vs current estimate.
        best, bestcost = None, 1e18
        for combo in itertools.combinations(range(len(c)), 3):
            f = [c[combo[0]], c[combo[1]], c[combo[2]]]  # already ascending
            cost = sum(abs(f[i] - est[i]) for i in range(3))
            if cost < bestcost:
                bestcost, best = cost, f
        out[t] = tuple(best)
        est = [alpha * best[i] + (1 - alpha) * est[i] for i in range(3)]
    return out


def metrics(o, p):
    o = np.array(o); p = np.array(p)
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
        cands = load_candidates(csvpath)
        ts = [t for t in sorted(praat) if t in cands]
        tracked = track(cands, ts)
        print(f"\n=== {stem} ({len(ts)} voiced frames) ===")
        for slot, name in [(1, "F2"), (2, "F3")]:
            raw = [cands[t][slot] if len(cands[t]) > slot else np.nan for t in ts]
            trk = [tracked[t][slot] for t in ts]
            ref = [praat[t][slot] for t in ts]
            r = metrics(raw, ref); k = metrics(trk, ref)
            print(f"  {name} raw    : acc={r[0]:5.1f}% glitch={r[1]:3.0f}% jitter={r[2]:4.0f}Hz")
            print(f"  {name} tracked: acc={k[0]:5.1f}% glitch={k[1]:3.0f}% jitter={k[2]:4.0f}Hz")


if __name__ == "__main__":
    run()
