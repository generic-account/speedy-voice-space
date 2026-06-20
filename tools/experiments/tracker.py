"""
Formant continuity tracker (pure Python, port-ready for a TS worker).

Turns per-frame candidate LPC poles into smooth F1<F2<F3 tracks.

Two implementations share the same cost model:
  * viterbi_offline(cand_seq, params): global DP over the whole utterance
    (upper bound on achievable smoothness; uses the future, so not shippable).
  * OnlineTracker: causal forward DP with a small fixed lookahead
    (LOOKAHEAD frames of latency, <=100 ms) -- this is what we ship.

State at each frame = an assignment of (up to) 3 candidate poles to the slots
F1,F2,F3 with f1 < f2 < f3. A slot may be UNFILLED (-1) when no suitable pole
exists; the tracker then coasts on its predicted value for that slot.

Cost(state) =
    emission:  per-slot soft prior pull (Gaussian-ish, wide)
             + bandwidth penalty (wide poles are bad; narrow poles rewarded)
             + missing-slot penalty
  + transition (vs previous chosen state):
             robust SATURATING penalty on |df| per slot, so within-vowel drift
             is cheap, a one-off outlier is capped, and a genuine vowel switch
             (sustained move) is affordable when the emission evidence supports it.

No numpy / scipy: plain lists, loops, math.  Designed to be transliterated to
TypeScript almost line-for-line.
"""
import math
from itertools import combinations

# ------------------------------ parameters ----------------------------------
DEFAULT_PARAMS = {
    # soft per-slot frequency priors (adult voice), Hz: (center, sigma).
    # Kept WIDE so they break ties without overriding strong evidence.
    "prior": [(450.0, 250.0), (1300.0, 650.0), (2400.0, 700.0)],
    "w_prior": 0.8,          # weight of prior pull in emission
    "w_band": 0.25,          # weight of bandwidth penalty (LPC F2 can be wide!)
    "band_ref": 350.0,       # bandwidths above this (Hz) get penalized
    "w_missing": 2.0,        # emission cost of leaving a slot unfilled (coast)
                             # (>trans_cap so a jumpy F3 is followed, not dropped)
    "w_cover": 0.6,          # penalty per formant-band pole left UNused below top
    "cover_band": (350.0, 3200.0),  # freq band in which an unused pole is suspect
    # transition: robust saturating |df| cost.  cost = w_trans * huber(df/scale)
    "w_trans": 1.0,
    "trans_scale": 110.0,    # Hz: within-vowel jitter scale (linear below ~this)
    "trans_cap": 3.0,        # saturating cap (in huber units) -> switch allowed
    "lookahead": 2,          # online latency in frames (2 => 40 ms; coverage
                             # penalty does the heavy lifting, not lookahead)
}


def _huber(x, cap):
    """Saturating quadratic-then-linear-then-capped cost on |x| (x already
    normalized).  Quadratic for small x (cheap drift), linear mid, capped so a
    real switch is affordable."""
    a = abs(x)
    if a <= 1.0:
        c = 0.5 * a * a
    else:
        c = a - 0.5
    return c if c < cap else cap


def _prior_cost(freq, center, sigma):
    z = (freq - center) / sigma
    return 0.5 * z * z


def _band_cost(band, ref):
    if band <= ref:
        return -0.3 * (1.0 - band / ref)   # small reward for narrow poles
    return (band - ref) / ref              # linear penalty for wide poles


# ----------------------- per-frame state enumeration ------------------------
def _enum_states(cand, params, max_states=48):
    """Enumerate candidate ascending triples (i,j,k) of candidate indices for
    slots F1,F2,F3.  Slots may be unfilled (-1).  Returns list of
    (assign_tuple, emission_cost).  Assign is (i,j,k) indices into cand.

    Emission (all terms in comparable "huber-ish" units):
      prior pull (soft, wide) + bandwidth penalty + missing-slot penalty
      + COVERAGE penalty: leaving a low/mid candidate pole UNUSED while a
        higher pole fills a lower slot is suspect (that's the rec_u5 wrong-
        collapse: F2<-2500 while a 750 pole sits unused). We penalize a pole
        that is unused and lies below the highest used slot's frequency.
    """
    n = len(cand)
    freqs = [c[0] for c in cand]
    bands = [c[1] for c in cand]
    prior = params["prior"]
    wp, wb, br = params["w_prior"], params["w_band"], params["band_ref"]
    wmiss = params["w_missing"]
    wcov = params["w_cover"]
    cov_lo, cov_hi = params["cover_band"]

    def slot_emit(slot, idx):
        if idx < 0:
            return wmiss
        f, b = freqs[idx], bands[idx]
        ce, sg = prior[slot]
        return wp * _prior_cost(f, ce, sg) + wb * _band_cost(b, br)

    def coverage(assign):
        used = set(i for i in assign if i >= 0)
        if not used:
            return 0.0
        top = max(freqs[i] for i in used)
        pen = 0.0
        for i in range(n):
            if i in used:
                continue
            f = freqs[i]
            # a formant-band pole sitting below the highest assigned pole but
            # left unused => probably should have filled a slot.
            if cov_lo <= f <= cov_hi and f < top:
                pen += wcov
        return pen

    states = []
    order = sorted(range(n), key=lambda i: freqs[i])
    for combo in combinations(order, 3):
        i, j, k = combo
        e = slot_emit(0, i) + slot_emit(1, j) + slot_emit(2, k) + coverage((i, j, k))
        states.append(((i, j, k), e))
    for combo in combinations(order, 2):
        a, b = combo
        states.append(((-1, a, b),
                       slot_emit(0, -1) + slot_emit(1, a) + slot_emit(2, b) + coverage((-1, a, b))))
        states.append(((a, b, -1),
                       slot_emit(0, a) + slot_emit(1, b) + slot_emit(2, -1) + coverage((a, b, -1))))
        states.append(((a, -1, b),
                       slot_emit(0, a) + slot_emit(1, -1) + slot_emit(2, b) + coverage((a, -1, b))))
    for a in order:
        states.append(((a, -1, -1), slot_emit(0, a) + 2 * wmiss + coverage((a, -1, -1))))
        states.append(((-1, a, -1), slot_emit(1, a) + 2 * wmiss + coverage((-1, a, -1))))
    if not states:
        states.append(((-1, -1, -1), 3 * wmiss))
    states.sort(key=lambda s: s[1])
    return states[:max_states]


def _state_freqs(assign, cand):
    """Return [f1,f2,f3] with None for unfilled slots."""
    return [cand[idx][0] if idx >= 0 else None for idx in assign]


def _trans_cost(prev_f, cur_f, params):
    """Transition cost between two states' slot frequencies (lists len 3 with
    None allowed).  Coasting (None on either side) costs a small constant."""
    ws, sc, cap = params["w_trans"], params["trans_scale"], params["trans_cap"]
    tot = 0.0
    for a, b in zip(prev_f, cur_f):
        if a is None or b is None:
            tot += ws * 0.3 * cap      # mild penalty for appearing/disappearing
            continue
        tot += ws * _huber((b - a) / sc, cap)
    return tot


# ------------------------------ offline Viterbi -----------------------------
def viterbi_offline(cand_seq, params=DEFAULT_PARAMS):
    """Global DP. Returns (f1_track, f2_track, f3_track) lists (None where a
    slot is unfilled at that frame)."""
    T = len(cand_seq)
    if T == 0:
        return [], [], []
    states_per = [_enum_states(c, params) for c in cand_seq]
    freqs_per = [[_state_freqs(a, cand_seq[t]) for a, _ in states_per[t]]
                 for t in range(T)]

    # init
    dp = [e for _, e in states_per[0]]
    back = [[-1] * len(states_per[t]) for t in range(T)]
    for t in range(1, T):
        cur = states_per[t]
        prev_f = freqs_per[t - 1]
        new = [0.0] * len(cur)
        for ci, (_, ce) in enumerate(cur):
            cf = freqs_per[t][ci]
            best = math.inf
            bj = -1
            for pj in range(len(states_per[t - 1])):
                c = dp[pj] + _trans_cost(prev_f[pj], cf, params)
                if c < best:
                    best = c
                    bj = pj
            new[ci] = best + ce
            back[t][ci] = bj
        dp = new

    # backtrack
    end = min(range(len(dp)), key=lambda i: dp[i])
    path = [0] * T
    path[T - 1] = end
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t][path[t]]
    f1 = f2 = None
    o1, o2, o3 = [], [], []
    for t in range(T):
        a = states_per[t][path[t]][0]
        ff = _state_freqs(a, cand_seq[t])
        o1.append(ff[0]); o2.append(ff[1]); o3.append(ff[2])
    return o1, o2, o3


# ------------------------------ online tracker ------------------------------
class OnlineTracker:
    """Causal forward DP with fixed lookahead.

    Buffers `lookahead` future frames before committing each output frame, so
    the committed assignment can use a little future context (resolves
    one-frame collapses without seeing the whole utterance).  Latency =
    lookahead frames.
    """

    def __init__(self, params=DEFAULT_PARAMS):
        self.p = params
        self.L = params["lookahead"]
        self._buf = []          # pending (cand, t)
        self._committed = None  # last committed [f1,f2,f3] (None-allowed)
        # forward DP frontier over buffered frames, rebuilt each push:
        self._coast = [p[0] for p in params["prior"]]  # last good per-slot value

    def _coast_anchor(self):
        """Best per-slot anchor for transition into the first buffered frame:
        last committed value if present else coasted estimate."""
        anchor = []
        for s in range(3):
            v = None
            if self._committed is not None:
                v = self._committed[s]
            if v is None:
                v = self._coast[s]
            anchor.append(v)
        return anchor

    def _solve_buffer(self):
        """Run a short DP over the buffered frames anchored on the committed
        state, return the assignment freqs for the FIRST buffered frame."""
        buf = self._buf
        T = len(buf)
        states_per = [_enum_states(c, self.p) for c, _ in buf]
        freqs_per = [[_state_freqs(a, buf[t][0]) for a, _ in states_per[t]]
                     for t in range(T)]
        anchor = self._coast_anchor()

        # init: transition from anchor into frame 0
        dp = []
        for ci, (_, ce) in enumerate(states_per[0]):
            dp.append(ce + _trans_cost(anchor, freqs_per[0][ci], self.p))
        back = [[-1] * len(states_per[t]) for t in range(T)]
        for t in range(1, T):
            prev_f = freqs_per[t - 1]
            new = [0.0] * len(states_per[t])
            for ci, (_, ce) in enumerate(states_per[t]):
                cf = freqs_per[t][ci]
                best = math.inf; bj = -1
                for pj in range(len(states_per[t - 1])):
                    c = dp[pj] + _trans_cost(prev_f[pj], cf, self.p)
                    if c < best:
                        best = c; bj = pj
                new[ci] = best + ce
                back[t][ci] = bj
            dp = new
        # backtrack to find frame-0 assignment on the best full path
        end = min(range(len(dp)), key=lambda i: dp[i])
        path = [0] * T
        path[T - 1] = end
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t][path[t]]
        a0 = states_per[0][path[0]][0]
        return _state_freqs(a0, buf[0][0])

    def push(self, cand, t):
        """Feed one frame's candidate list [(f,b),...].

        Returns the committed [f1,f2,f3] for the frame that is now `lookahead`
        frames in the past, or None while still filling the lookahead buffer.
        Call flush() at end-of-stream to drain the remaining buffered frames.
        """
        self._buf.append((cand, t))
        if len(self._buf) <= self.L:
            return None
        return self._commit_oldest()

    def _commit_oldest(self):
        ff = self._solve_buffer()
        self._committed = ff
        for s in range(3):
            if ff[s] is not None:
                self._coast[s] = ff[s]
        self._buf.pop(0)
        return ff

    def flush(self):
        """Drain remaining buffered frames (no more future context). Returns a
        list of committed [f1,f2,f3] for the still-buffered frames in order."""
        out = []
        while self._buf:
            out.append(self._commit_oldest())
        return out
