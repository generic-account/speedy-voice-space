// Formant continuity tracker — assigns each frame's candidate LPC poles to
// smooth F1<F2<F3 tracks, fixing the slot-jumps that frequency-rank assignment
// causes. Greedy per-frame (zero latency): pick the candidate→slot assignment
// minimizing emission (soft prior + bandwidth + missing/coverage) + transition
// from the last committed frame (saturating |Δf|: cheap within-vowel drift, a
// real vowel switch still affordable). Ported from tools/experiments/tracker.py.

export interface TrackerParams {
  prior: [number, number][]; // per-slot (center, sigma)
  wPrior: number;
  wBand: number;
  bandRef: number;
  wMissing: number;
  wCover: number;
  coverBand: [number, number];
  wTrans: number;
  transScale: number;
  transCap: number;
}

export const DEFAULT_TRACKER_PARAMS: TrackerParams = {
  prior: [
    [450, 250],
    [1300, 650],
    [2400, 700],
  ],
  wPrior: 0.8,
  wBand: 0.25,
  bandRef: 350,
  // Cost of leaving a slot empty. Must exceed the capped transition cost
  // (trans_cap·w_trans = 3.0) enough that following a moving formant beats
  // dropping it — at 1.4 the jumpy F3 was abandoned on ~⅔ of sustained /a/.
  wMissing: 2.0,
  wCover: 0.6,
  coverBand: [350, 3200],
  wTrans: 1.0,
  transScale: 110,
  // Cap on the per-slot transition penalty. Kept below the cost of abandoning a
  // slot (wMissing + null-transition ≈ 2.5) so following a real but fast-moving
  // formant always beats dropping it — otherwise F3 (variable, esp. on /a/) was
  // left empty whenever it strayed >~1 transScale from the coasted anchor.
  transCap: 1.8,
};

export type Pole = [number, number]; // [freq, bandwidth]
type Assign = [number, number, number]; // candidate indices into cand (or -1)
type SlotFreqs = [number | null, number | null, number | null];

function huber(x: number, cap: number): number {
  const a = Math.abs(x);
  const c = a <= 1 ? 0.5 * a * a : a - 0.5;
  return c < cap ? c : cap;
}

function priorCost(freq: number, center: number, sigma: number): number {
  const z = (freq - center) / sigma;
  return 0.5 * z * z;
}

function bandCost(band: number, ref: number): number {
  if (band <= ref) return -0.3 * (1 - band / ref);
  return (band - ref) / ref;
}

// ascending r-combinations of the given index array
function combinations(arr: number[], r: number): number[][] {
  const res: number[][] = [];
  const n = arr.length;
  if (r > n) return res;
  const idx = Array.from({ length: r }, (_, i) => i);
  for (;;) {
    res.push(idx.map((i) => arr[i]));
    let i = r - 1;
    while (i >= 0 && idx[i] === i + n - r) i--;
    if (i < 0) break;
    idx[i]++;
    for (let j = i + 1; j < r; j++) idx[j] = idx[j - 1] + 1;
  }
  return res;
}

function enumStates(
  cand: Pole[],
  p: TrackerParams,
  maxStates = 48,
): [Assign, number][] {
  const n = cand.length;
  const freqs = cand.map((c) => c[0]);
  const bands = cand.map((c) => c[1]);
  const [covLo, covHi] = p.coverBand;

  const slotEmit = (slot: number, idx: number): number => {
    if (idx < 0) return p.wMissing;
    const [ce, sg] = p.prior[slot];
    return p.wPrior * priorCost(freqs[idx], ce, sg) + p.wBand * bandCost(bands[idx], p.bandRef);
  };

  const coverage = (assign: Assign): number => {
    const used = new Set(assign.filter((i) => i >= 0));
    if (used.size === 0) return 0;
    let top = -Infinity;
    for (const i of used) top = Math.max(top, freqs[i]);
    let pen = 0;
    for (let i = 0; i < n; i++) {
      if (used.has(i)) continue;
      const f = freqs[i];
      if (f >= covLo && f <= covHi && f < top) pen += p.wCover;
    }
    return pen;
  };

  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => freqs[a] - freqs[b]);
  const states: [Assign, number][] = [];

  for (const [i, j, k] of combinations(order, 3)) {
    const a: Assign = [i, j, k];
    states.push([a, slotEmit(0, i) + slotEmit(1, j) + slotEmit(2, k) + coverage(a)]);
  }
  for (const [a, b] of combinations(order, 2)) {
    const s1: Assign = [-1, a, b];
    const s2: Assign = [a, b, -1];
    const s3: Assign = [a, -1, b];
    states.push([s1, slotEmit(0, -1) + slotEmit(1, a) + slotEmit(2, b) + coverage(s1)]);
    states.push([s2, slotEmit(0, a) + slotEmit(1, b) + slotEmit(2, -1) + coverage(s2)]);
    states.push([s3, slotEmit(0, a) + slotEmit(1, -1) + slotEmit(2, b) + coverage(s3)]);
  }
  for (const a of order) {
    states.push([[a, -1, -1], slotEmit(0, a) + 2 * p.wMissing + coverage([a, -1, -1])]);
    states.push([[-1, a, -1], slotEmit(1, a) + 2 * p.wMissing + coverage([-1, a, -1])]);
  }
  if (states.length === 0) states.push([[-1, -1, -1], 3 * p.wMissing]);

  states.sort((x, y) => x[1] - y[1]);
  return states.slice(0, maxStates);
}

function stateFreqs(a: Assign, cand: Pole[]): SlotFreqs {
  return [
    a[0] >= 0 ? cand[a[0]][0] : null,
    a[1] >= 0 ? cand[a[1]][0] : null,
    a[2] >= 0 ? cand[a[2]][0] : null,
  ];
}

function transCost(prev: SlotFreqs, cur: SlotFreqs, p: TrackerParams): number {
  let tot = 0;
  for (let s = 0; s < 3; s++) {
    const a = prev[s];
    const b = cur[s];
    if (a === null || b === null) {
      tot += p.wTrans * 0.3 * p.transCap;
      continue;
    }
    tot += p.wTrans * huber((b - a) / p.transScale, p.transCap);
  }
  return tot;
}

export class FormantTracker {
  private committed: SlotFreqs | null = null;
  private coast: number[];

  constructor(private p: TrackerParams = DEFAULT_TRACKER_PARAMS) {
    this.coast = p.prior.map((pr) => pr[0]);
  }

  reset(): void {
    this.committed = null;
    this.coast = this.p.prior.map((pr) => pr[0]);
  }

  // Anchor for the transition cost: last committed frequencies, falling back to
  // the coasted last-good value (then the prior center) for any empty slot.
  private coastAnchor(): SlotFreqs {
    return [0, 1, 2].map((s) => this.committed?.[s] ?? this.coast[s]) as SlotFreqs;
  }

  /** Feed one frame's candidate poles; returns the committed [f1,f2,f3] (null
   * where a slot is unfilled). */
  push(cand: Pole[]): SlotFreqs {
    const anchor = this.coastAnchor();
    let bestAssign: Assign = [-1, -1, -1];
    let bestCost = Infinity;
    for (const [a, emit] of enumStates(cand, this.p)) {
      const cost = emit + transCost(anchor, stateFreqs(a, cand), this.p);
      if (cost < bestCost) {
        bestCost = cost;
        bestAssign = a;
      }
    }
    const ff = stateFreqs(bestAssign, cand);
    this.committed = ff;
    for (let s = 0; s < 3; s++) if (ff[s] !== null) this.coast[s] = ff[s]!;
    return ff;
  }
}
