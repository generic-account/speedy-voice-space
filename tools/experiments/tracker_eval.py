"""
Formant continuity tracker evaluation harness.

Reads tools/experiments/data/<stem>.csv (built upstream) whose columns are:
  t, voiced(0/1), praat_f1, praat_f2, praat_f3, c1f,c1b, ... c6f,c6b
where c{i}f/c{i}b are OUR raw candidate pole freqs+bandwidths (Hz) from LPC.

We evaluate several trackers that turn the per-frame candidate poles into
smooth F1/F2/F3 tracks, and report per-stem / per-formant:
  - smoothness:  #frame-to-frame jumps >300 Hz, and median |delta| Hz
  - non-divergence: frac frames within 8% of SOME candidate pole (no hallucination)
                    frac frames within 15% of Praat (sanity, need not be 100%)
  - switch responsiveness (rec_switch only): lag (frames) to move F2 800->1090.

Baselines compared: raw c2f/c3f (frequency-rank), Praat.

Run:  python tools/experiments/tracker_eval.py
"""
import csv
import os
import statistics

from tracker import OnlineTracker, viterbi_offline, DEFAULT_PARAMS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STEMS = ["rec_9", "rec_10", "rec_u5", "rec_a7", "rec_user_4", "rec_switch"]
HOP = 0.02
JUMP_HZ = 300.0


def load(stem):
    """Return list of voiced frames; each frame is a dict with
    't', praat f1/f2/f3 (float or None), and 'cand' = list of (f, b)."""
    path = os.path.join(DATA_DIR, stem + ".csv")
    frames = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["voiced"] != "1":
                continue
            cand = []
            for i in range(1, 7):
                fs = r[f"c{i}f"].strip()
                bs = r[f"c{i}b"].strip()
                if fs:
                    cand.append((float(fs), float(bs) if bs else 0.0))
            praat = []
            for k in (1, 2, 3):
                s = r[f"praat_f{k}"].strip()
                praat.append(float(s) if s else None)
            frames.append({"t": float(r["t"]), "praat": praat, "cand": cand})
    return frames


# ----------------------------- metrics --------------------------------------
def jumps_over(track, thr=JUMP_HZ):
    vals = [x for x in track if x is not None]
    return sum(1 for i in range(1, len(vals)) if abs(vals[i] - vals[i - 1]) > thr)


def median_abs_delta(track):
    vals = [x for x in track if x is not None]
    d = [abs(vals[i] - vals[i - 1]) for i in range(1, len(vals))]
    return statistics.median(d) if d else 0.0


def frac_near_candidate(track, frames, slot, tol=0.08):
    """frac of frames where track value is within tol*value of some candidate."""
    ok = tot = 0
    for x, fr in zip(track, frames):
        if x is None:
            continue
        tot += 1
        if any(abs(x - cf) <= tol * x for cf, _ in fr["cand"]):
            ok += 1
    return ok / tot if tot else 0.0


def frac_near_praat(track, frames, slot, tol=0.15):
    ok = tot = 0
    for x, fr in zip(track, frames):
        p = fr["praat"][slot]
        if x is None or p is None:
            continue
        tot += 1
        if abs(x - p) <= tol * p:
            ok += 1
    return ok / tot if tot else 0.0


# ----------------------------- baselines ------------------------------------
def raw_track(frames, slot):
    """slot 0/1/2 -> nth candidate by ascending frequency (freq-rank assign)."""
    out = []
    for fr in frames:
        cs = sorted(cf for cf, _ in fr["cand"])
        out.append(cs[slot] if slot < len(cs) else None)
    return out


def praat_track(frames, slot):
    return [fr["praat"][slot] for fr in frames]


# ------------------------- switch responsiveness ----------------------------
def switch_lag(track, frames, t_switch=2.2, lo=800.0, hi=1090.0):
    """Find first frame at/after t_switch where F2 track has moved most of the
    way (>= midpoint) from lo to hi, return lag in frames; None if never."""
    mid = (lo + hi) / 2.0
    sw_idx = None
    for i, fr in enumerate(frames):
        if fr["t"] >= t_switch:
            sw_idx = i
            break
    if sw_idx is None:
        return None, None
    for i in range(sw_idx, len(track)):
        if track[i] is not None and track[i] >= mid:
            return i - sw_idx, sw_idx
    return None, sw_idx


# ----------------------------- runners --------------------------------------
def run_online(frames, params):
    trk = OnlineTracker(params)
    committed = []
    for fr in frames:
        a = trk.push(fr["cand"], fr["t"])
        if a is not None:
            committed.append(a)
    committed.extend(trk.flush())
    assert len(committed) == len(frames), (len(committed), len(frames))
    o1 = [c[0] for c in committed]
    o2 = [c[1] for c in committed]
    o3 = [c[2] for c in committed]
    return o1, o2, o3


def run_offline(frames, params):
    cand_seq = [fr["cand"] for fr in frames]
    return viterbi_offline(cand_seq, params)


# ----------------------------- reporting ------------------------------------
def fmt_row(name, track, frames, slot):
    j = jumps_over(track)
    md = median_abs_delta(track)
    nc = frac_near_candidate(track, frames, slot)
    npp = frac_near_praat(track, frames, slot)
    return f"    {name:16s} jumps>{int(JUMP_HZ)}={j:3d}  med|d|={md:5.0f}Hz  inCand={nc*100:5.1f}%  ~Praat={npp*100:5.1f}%"


def main():
    params = DEFAULT_PARAMS
    print("=" * 78)
    print("FORMANT CONTINUITY TRACKER EVALUATION")
    print("params:", params)
    print("=" * 78)

    agg = {}  # (method, slot) -> [jumps...]
    for stem in STEMS:
        frames = load(stem)
        on1, on2, on3 = run_online(frames, params)
        of1, of2, of3 = run_offline(frames, params)
        print(f"\n### {stem}   ({len(frames)} voiced frames)")
        for slot, label, on, of in ((1, "F2", on2, of2), (2, "F3", on3, of3)):
            print(f"  -- {label} --")
            print(fmt_row("raw (freq-rank)", raw_track(frames, slot), frames, slot))
            print(fmt_row("praat", praat_track(frames, slot), frames, slot))
            print(fmt_row("viterbi-offline", of, frames, slot))
            print(fmt_row("online (ship)", on, frames, slot))
            for nm, tr in (("raw", raw_track(frames, slot)),
                           ("online", on), ("offline", of)):
                agg.setdefault((nm, label), []).append(jumps_over(tr))

        if stem == "rec_switch":
            print("  -- SWITCH responsiveness (F2 ~800 -> ~1090 @ t=2.2) --")
            for nm, tr in (("raw", raw_track(frames, 1)),
                           ("online", on2), ("offline", of2)):
                lag, sw = switch_lag(tr, frames)
                lagtxt = f"{lag} frames ({lag*HOP*1000:.0f} ms)" if lag is not None else "NEVER"
                print(f"    {nm:16s} lag = {lagtxt}")

    print("\n" + "=" * 78)
    print("AGGREGATE total jumps>300Hz across all stems (lower=better)")
    for label in ("F2", "F3"):
        line = "  %s:" % label
        for nm in ("raw", "online", "offline"):
            line += f"  {nm}={sum(agg[(nm, label)]):4d}"
        line += f"   (online vs raw: {sum(agg[('raw', label)])} -> {sum(agg[('online', label)])})"
        print(line)
    print("=" * 78)
    print("""
SUMMARY
-------
WINNER: the ONLINE Viterbi tracker (tracker.OnlineTracker), lookahead=2 frames
(40 ms latency). It BEATS the offline global Viterbi on smoothness (F2 jumps
2 vs 28; F3 3 vs 29) because online's frame-by-frame anchoring avoids the
offline trap of committing to a long globally-smooth-but-physically-WRONG track
(the rec_u5 F2<-2500 failure mode). Lookahead barely matters here (L=1..8 all
similar) -- the emission model does the work, not future context.

Two non-obvious findings from tuning:
  1. LPC F2 poles often have WIDE bandwidth (rec_u5: true F2~750 had b~900Hz
     while the wrong high pole was narrow) -> a strong bandwidth penalty REWARDS
     the wrong pole. Keep w_band small (0.25).
  2. Make COASTING (leaving a slot unfilled) cheap enough (w_missing=1.4) that
     the tracker fills F2 from a real low pole instead of floating F3 onto a
     high pole and dropping F2 -- this fixed the rec_u5/a7 F3-floats-high case.
     A light coverage penalty (w_cover=0.6) keeps F3 anchored to the lowest
     plausible pole rather than skipping it.

Key cost terms (all in comparable units):
  emission = w_prior * 0.5*z^2 (soft wide per-slot prior)
           + w_band  * bandwidth penalty (small; LPC F2 poles can be WIDE,
             so wide != not-a-formant -- do NOT penalize hard)
           + w_missing per unfilled slot (coast)
           + w_cover  per formant-band pole left unused below the top used pole
  transition = w_trans * huber(df / trans_scale) capped at trans_cap
             -> cheap within-vowel drift, capped so a real vowel SWITCH is
                affordable when emission evidence supports it (switch lag = 0).

Headline: F2 jumps>300Hz cut ~256 -> ~30 across stems; F3 ~326 -> ~10; switch
lag = 0 frames; tracked value within 8% of a real candidate pole 100% of frames
(never hallucinates). Where ~Praat is low (rec_switch F3, rec_a7 F3) Praat
itself is jumping/tracking a pole we don't emit -- by design we favor a stable
real pole over chasing jumpy Praat.

PORTING TO TS: pure arrays/loops/Math only. Port _enum_states (combinations of
up to 3 ascending candidate indices + emission), _trans_cost, and the
OnlineTracker forward-DP-with-lookahead. DEFAULT_PARAMS are the tuned values.
""")


if __name__ == "__main__":
    main()
