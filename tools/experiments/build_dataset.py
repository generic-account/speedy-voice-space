"""Merge our candidate poles (/tmp/cand_<stem>.csv from web/scripts/wasm_formants.mjs)
with Praat ground-truth formants into a single self-contained CSV per stem under
tools/experiments/data/. Columns:
  t, voiced, praat_f1, praat_f2, praat_f3,
  c1f,c1b, ... c6f,c6b   (our candidate poles freq/bandwidth)
This is the input for the tracking experiments (no node/wasm needed afterwards).
"""
import csv, os, glob
import parselmouth
import numpy as np

HERE = os.path.dirname(__file__)
FIX = os.path.join(HERE, "..", "audio", "fixtures")
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

STEMS = ["rec_9", "rec_10", "rec_u5", "rec_a7", "rec_user_4", "rec_switch"]

for stem in STEMS:
    candpath = f"/tmp/cand_{stem}.csv"
    if not os.path.exists(candpath):
        print("skip (no candidates):", stem); continue
    snd = parselmouth.Sound(os.path.join(FIX, f"{stem}.wav"))
    fo = snd.to_formant_burg(time_step=0.005, max_number_of_formants=5,
                             maximum_formant=5500, window_length=0.025, pre_emphasis_from=50)
    rows_in = list(csv.DictReader(open(candpath)))
    out_rows = []
    for r in rows_in:
        t = float(r["t"])
        voiced = 1 if r["f0"] else 0
        def pf(i):
            v = fo.get_value_at_time(i, t)
            return "" if v is None or np.isnan(v) else round(v, 2)
        row = {"t": r["t"], "voiced": voiced,
               "praat_f1": pf(1), "praat_f2": pf(2), "praat_f3": pf(3)}
        for c in range(1, 7):
            row[f"c{c}f"] = r[f"c{c}f"]; row[f"c{c}b"] = r[f"c{c}b"]
        out_rows.append(row)
    cols = ["t","voiced","praat_f1","praat_f2","praat_f3"] + \
           [f"c{c}{k}" for c in range(1,7) for k in ("f","b")]
    with open(os.path.join(DATA, f"{stem}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(out_rows)
    print(f"{stem}: {len(out_rows)} frames -> data/{stem}.csv")
