//! Pitch estimation — a port of Praat's `Sound_to_Pitch` (ac) method
//! (Boersma 1993), evaluated at a single analysis window's midpoint.
//!
//! Notable choices vs. full Praat: direct time-domain autocorrelation (only
//! lags up to sr/floor are needed, avoiding FFT normalization subtleties), the
//! Boersma window-function autocorrelation correction, and a per-frame best
//! candidate with no cross-frame Viterbi (the app median-filters f0 downstream).
//! Validated against the Parselmouth oracle in `tests/parity.rs`.

#[derive(Clone, Copy, Debug)]
pub struct PitchParams {
    pub samplerate: f64,
    pub floor: f64,
    pub ceiling: f64,
    pub silence_threshold: f64,
    pub voicing_threshold: f64,
    /// Praat default 0.01 (favours higher f0 to suppress sub-harmonics).
    pub octave_cost: f64,
    pub very_accurate: bool,
}

impl Default for PitchParams {
    fn default() -> Self {
        PitchParams {
            samplerate: 48000.0,
            floor: 75.0,
            ceiling: 400.0,
            silence_threshold: 0.03,
            voicing_threshold: 0.45,
            octave_cost: 0.01,
            very_accurate: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PitchResult {
    pub voiced: bool,
    /// f0 in Hz when voiced, else None.
    pub f0: Option<f64>,
    /// Strength of the chosen candidate (debug/inspection).
    pub strength: f64,
}

/// Praat-style Hanning window (1-indexed cosine, length n).
fn hanning(n: usize) -> Vec<f64> {
    let mut w = vec![0.0; n];
    let denom = (n + 1) as f64;
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * (i as f64 + 1.0) / denom).cos();
    }
    w
}

/// Unnormalized autocorrelation of `x` for lags `0..=max_lag`.
fn autocorr(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    let mut r = vec![0.0; max_lag + 1];
    for (lag, rl) in r.iter_mut().enumerate() {
        let mut s = 0.0;
        for i in 0..(n - lag) {
            s += x[i] * x[i + lag];
        }
        *rl = s;
    }
    r
}

/// Parabolic sub-sample refinement of a local maximum at index `i` over `y`.
/// Returns (offset in [-1,1], interpolated peak value).
fn parabolic_peak(y: &[f64], i: usize) -> (f64, f64) {
    if i == 0 || i + 1 >= y.len() {
        return (0.0, y[i]);
    }
    let a = y[i - 1];
    let b = y[i];
    let c = y[i + 1];
    let denom = a - 2.0 * b + c;
    if denom.abs() < 1e-12 {
        return (0.0, b);
    }
    let offset = 0.5 * (a - c) / denom;
    let peak = b - 0.25 * (a - c) * offset;
    (offset, peak)
}

pub fn analyze_pitch(frame: &[f64], p: &PitchParams) -> PitchResult {
    let sr = p.samplerate;
    let n_total = frame.len();
    if n_total < 16 {
        return PitchResult { voiced: false, f0: None, strength: 0.0 };
    }

    // Silence reference: global peak of the mean-subtracted sound.
    let global_mean = frame.iter().sum::<f64>() / n_total as f64;
    let global_peak = frame
        .iter()
        .fold(0.0_f64, |m, &v| m.max((v - global_mean).abs()));
    if global_peak <= 0.0 {
        return PitchResult { voiced: false, f0: None, strength: 0.0 };
    }

    // Window spans periodsPerWindow / floor (3 periods, or 6 if very accurate).
    let periods_per_window = if p.very_accurate { 6.0 } else { 3.0 };
    let mut nwin = (periods_per_window / p.floor * sr).round() as usize;
    if nwin > n_total {
        nwin = n_total;
    }
    if nwin < 16 {
        return PitchResult { voiced: false, f0: None, strength: 0.0 };
    }
    let start = (n_total - nwin) / 2; // centered window (the midpoint frame)
    let seg = &frame[start..start + nwin];

    // Center the segment; local peak is the voicing intensity reference.
    let local_mean = seg.iter().sum::<f64>() / nwin as f64;
    let mut local_peak = 0.0_f64;
    let mut centered = vec![0.0; nwin];
    for i in 0..nwin {
        let v = seg[i] - local_mean;
        centered[i] = v;
        local_peak = local_peak.max(v.abs());
    }

    let w = hanning(nwin);
    let windowed: Vec<f64> = centered.iter().zip(&w).map(|(a, b)| a * b).collect();

    let min_lag = (sr / p.ceiling).floor() as usize;
    let max_lag = ((sr / p.floor).ceil() as usize).min(nwin - 2);
    if max_lag <= min_lag || min_lag < 1 {
        return unvoiced(p, local_peak, global_peak);
    }

    // Window-normalized autocorrelation (Boersma): r = (acx/acx0) / (acw/acw0).
    let acx = autocorr(&windowed, max_lag);
    let acw = autocorr(&w, max_lag);
    if acx[0] <= 0.0 || acw[0] <= 0.0 {
        return unvoiced(p, local_peak, global_peak);
    }
    let mut r = vec![0.0; max_lag + 1];
    for lag in 0..=max_lag {
        let rw = acw[lag] / acw[0];
        r[lag] = if rw.abs() > 1e-9 {
            (acx[lag] / acx[0]) / rw
        } else {
            0.0
        };
    }

    // Best voiced candidate among local maxima in [min_lag, max_lag].
    let mut best_f = 0.0;
    let mut best_strength = -1e9;
    for lag in min_lag..max_lag {
        if r[lag] > r[lag - 1] && r[lag] >= r[lag + 1] {
            let (offset, peak) = parabolic_peak(&r, lag);
            let lag_interp = lag as f64 + offset;
            if lag_interp <= 0.0 {
                continue;
            }
            let f = sr / lag_interp;
            if f < p.floor || f > p.ceiling {
                continue;
            }
            // Octave cost favours the true f0 over its subharmonics.
            let strength = peak - p.octave_cost * (p.ceiling / f).log2();
            if strength > best_strength {
                best_strength = strength;
                best_f = f;
            }
        }
    }

    let unvoiced_strength = unvoiced_strength(p, local_peak, global_peak);

    if best_strength > unvoiced_strength && best_f > 0.0 {
        PitchResult { voiced: true, f0: Some(best_f), strength: best_strength }
    } else {
        PitchResult { voiced: false, f0: None, strength: unvoiced_strength }
    }
}

fn unvoiced_strength(p: &PitchParams, local_peak: f64, global_peak: f64) -> f64 {
    let ratio = local_peak / global_peak;
    let denom = p.silence_threshold / (1.0 + p.voicing_threshold);
    p.voicing_threshold + (2.0 - ratio / denom).max(0.0)
}

fn unvoiced(p: &PitchParams, local_peak: f64, global_peak: f64) -> PitchResult {
    PitchResult {
        voiced: false,
        f0: None,
        strength: unvoiced_strength(p, local_peak, global_peak),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_tone(freq: f64, sr: f64, n: usize, amp: f64) -> Vec<f64> {
        (0..n)
            .map(|i| amp * (2.0 * std::f64::consts::PI * freq * i as f64 / sr).sin())
            .collect()
    }

    #[test]
    fn detects_pure_tone() {
        let p = PitchParams::default();
        let x = synth_tone(150.0, 48000.0, 2880, 0.3);
        let r = analyze_pitch(&x, &p);
        assert!(r.voiced, "150 Hz tone should be voiced");
        let f0 = r.f0.unwrap();
        assert!((f0 - 150.0).abs() < 1.5, "got {f0}");
    }

    #[test]
    fn silence_is_unvoiced() {
        let p = PitchParams::default();
        let x = vec![0.0; 2880];
        assert!(!analyze_pitch(&x, &p).voiced);
    }
}
