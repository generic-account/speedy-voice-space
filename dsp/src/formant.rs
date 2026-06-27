//! Formant estimation — a port of Praat's `Sound_to_Formant_burg`, evaluated at
//! a single analysis window's midpoint.
//!
//! Pipeline per frame: resample to `2 * maximum_formant` (so the new Nyquist is
//! the formant ceiling) → DC removal → pre-emphasis high-pass → Gaussian window
//! → Burg LPC of order `2 * max_number_of_formants` → LPC polynomial roots via
//! Aberth → poles in the upper half plane become formants. See `resample`,
//! `preprocess`, and `burg` for the parity-critical details.
//!
//! References: Boersma & Weenink, Praat `Sound_to_Formant.cpp` / `NUMburg`;
//! Press et al., Numerical Recipes (memcof). Validated against the Parselmouth
//! oracle in `tests/formant_parity.rs`.

use aberth::aberth;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct FormantParams {
    pub samplerate: f64,
    /// Number of formants to return; LPC order = 2 * this.
    pub max_number_of_formants: usize,
    /// Formant ceiling (Hz). The window is resampled so Nyquist == this.
    pub maximum_formant_hz: f64,
    /// Effective analysis window length (s). Praat's "Gaussian" window.
    pub window_length_s: f64,
    /// Pre-emphasis corner frequency (Hz).
    pub pre_emphasis_from_hz: f64,
    /// Use Praat's robust (IRLS / Huber-reweighted) LPC instead of plain Burg.
    /// On by default — it produces narrower (less spuriously damped) formant
    /// poles at vowel onsets. The Burg path is kept for A/B and parity tests.
    pub robust: bool,
}

impl Default for FormantParams {
    fn default() -> Self {
        FormantParams {
            samplerate: 48000.0,
            max_number_of_formants: 5,
            maximum_formant_hz: 5500.0,
            window_length_s: 0.025,
            pre_emphasis_from_hz: 50.0,
            robust: true,
        }
    }
}

/// Praat's robust-LPC knobs (`Sound_to_Formant_robust` defaults).
const ROBUST_K_STDEV: f64 = 1.5; // number_of_std_dev (Huber cutoff)
const ROBUST_ITERMAX: usize = 5; // outer IRLS iterations
const ROBUST_TOL: f64 = 1e-6; // convergence tolerance on the scale estimate
const HUBER_ITERMAX: usize = 5; // inner location/scale M-estimate iterations

#[derive(Clone, Debug)]
pub struct FormantResult {
    /// Formant frequencies F1..Fn (Hz), ascending.
    pub formants: Vec<f64>,
    /// Bandwidths aligned with `formants` (Hz), for inspection/filtering.
    pub bandwidths: Vec<f64>,
}

/// Fourier-domain resampler from `sr_in` to `sr_out`, equivalent to
/// `scipy.signal.resample` (rFFT → spectral truncation → irFFT, scaled by
/// `n_out/n_in`).
///
/// A Fourier resample (rather than a time-domain windowed sinc) is load-bearing
/// for parity: the sinc leaves a ~1e-6 residual at the new Nyquist bin, and the
/// ill-conditioned order-`2N` Burg on nasal vowels turns that into a spurious
/// split pole. The Fourier resample yields a cleanly band-limited frame whose
/// LPC lands on Praat's poles. A direct DFT is used (no FFT crate) — a single
/// analysis frame is small, so the O(n_in · n_out/2) cost is modest and wasm-safe.
fn resample(x: &[f64], sr_in: f64, sr_out: f64) -> Vec<f64> {
    if (sr_in - sr_out).abs() < 1e-6 {
        return x.to_vec();
    }
    let n_in = x.len();
    let n_out = (n_in as f64 * sr_out / sr_in).round() as usize;
    if n_out == 0 || n_in == 0 {
        return Vec::new();
    }

    // Forward real DFT, keeping bins 0..=keep where keep is the new Nyquist
    // index. Downsampling discards the high band (anti-aliasing); upsampling
    // leaves the extra bins implicitly zero.
    let keep = n_out / 2;
    let n_bins = keep + 1;
    let mut re = vec![0.0f64; n_bins];
    let mut im = vec![0.0f64; n_bins];
    let two_pi_over_n = 2.0 * PI / n_in as f64;
    for (k, (rk, ik)) in re.iter_mut().zip(im.iter_mut()).enumerate() {
        if k > n_in / 2 {
            break; // bins beyond the input Nyquist are zero
        }
        let mut sr = 0.0;
        let mut si = 0.0;
        let wk = two_pi_over_n * k as f64;
        for (n, &xn) in x.iter().enumerate() {
            let ang = wk * n as f64;
            sr += xn * ang.cos();
            si -= xn * ang.sin();
        }
        *rk = sr;
        *ik = si;
    }

    // Inverse real DFT, scaled by n_out/n_in (scipy convention), reconstructed
    // from the half spectrum via Hermitian symmetry: bin k and its mirror both
    // contribute, so count each twice except DC and (for even n_out) Nyquist.
    let scale = (n_out as f64 / n_in as f64) / n_in as f64;
    let mut out = vec![0.0f64; n_out];
    let two_pi_over_out = 2.0 * PI / n_out as f64;
    for (m, om) in out.iter_mut().enumerate() {
        let mut acc = re[0];
        let base = two_pi_over_out * m as f64;
        for k in 1..n_bins {
            let ang = base * k as f64;
            let term = re[k] * ang.cos() - im[k] * ang.sin();
            if k == keep && n_out % 2 == 0 {
                acc += term;
            } else {
                acc += 2.0 * term;
            }
        }
        *om = acc * scale;
    }
    out
}

/// Praat's Gaussian analysis window (see `Sound_to_Formant.cpp`):
/// `w[i] = (exp(-48 (i/N - 0.5)^2) - edge) / (1 - edge)`,
/// `edge = exp(-12)`. Length `n`.
/// Post-resample sample rate: frames are resampled so Nyquist == the formant ceiling.
fn resample_rate(p: &FormantParams) -> f64 {
    2.0 * p.maximum_formant_hz
}

fn gaussian_window(n: usize) -> Vec<f64> {
    let edge = (-12.0_f64).exp();
    let mut w = vec![0.0; n];
    let nm1 = (n - 1).max(1) as f64;
    for (i, wi) in w.iter_mut().enumerate() {
        let imid = i as f64 / nm1 - 0.5;
        *wi = ((-48.0 * imid * imid).exp() - edge) / (1.0 - edge);
    }
    w
}

/// Burg's method, a faithful port of Praat's `NUMburg` (the Numerical Recipes
/// `memcof` formulation). Returns the analysis-filter coefficients `a[1..=order]`
/// where `A(z) = 1 + a1 z^-1 + ... + a_p z^-p`.
///
/// Load-bearing vs a textbook Marple recursion: the reflection-coefficient
/// denominator is recomputed *fresh* each order from the current forward/backward
/// error arrays (`sum b1^2 + b2^2`) rather than carried in a running recurrence.
/// Both agree on well-conditioned vowels, but on near-singular nasal spectra the
/// running recurrence accumulates error and throws off the high-order poles.
fn burg(x: &[f64], order: usize) -> Vec<f64> {
    let n = x.len();
    let mut coeff = vec![0.0; order + 1]; // coeff[1..=order]
    if n <= order || order == 0 {
        return coeff;
    }

    // b1/b2 = forward/backward error arrays (NR memcof working arrays).
    let mut b1 = vec![0.0; n];
    let mut b2 = vec![0.0; n];
    let mut aa = vec![0.0; order + 1]; // scratch for the order-update

    b1[0] = x[0];
    b2[n - 2] = x[n - 1];
    for j in 1..n - 1 {
        b1[j] = x[j];
        b2[j - 1] = x[j];
    }

    for k in 1..=order {
        // Reflection coefficient: numerator and a FRESH denominator.
        let mut num = 0.0;
        let mut den = 0.0;
        for j in 0..n - k {
            num += b1[j] * b2[j];
            den += b1[j] * b1[j] + b2[j] * b2[j];
        }
        let kc = if den != 0.0 { 2.0 * num / den } else { 0.0 };
        coeff[k] = kc;

        // Order update of the AR coefficients.
        for i in 1..k {
            coeff[i] = aa[i] - kc * aa[k - i];
        }
        if k == order {
            break;
        }
        for i in 1..=k {
            aa[i] = coeff[i];
        }

        // Update the forward/backward error arrays for the next order.
        for j in 0..n - k - 1 {
            b1[j] -= aa[k] * b2[j];
            b2[j] = b2[j + 1] - aa[k] * b1[j + 1];
        }
    }

    // NR's predictor x[n] = sum coeff[k] x[n-k] → analysis filter a[k] = -coeff[k].
    let mut a = vec![0.0; order + 1];
    for k in 1..=order {
        a[k] = -coeff[k];
    }
    a
}

// ---------------------------------------------------------------------------
// Robust LPC — a port of Praat's `LPC_Sound_to_LPC_robust` / `huber_struct`
// (Sound_and_LPC_robust.cpp) + `NUMstatistics_huber` (NUMhuber.cpp), Lee 1988.
//
// Pipeline difference vs Burg: the initial coefficients come from the
// autocorrelation (Levinson) method, then we IRLS-reweight the samples with a
// Huber weight (down-weighting large-residual samples such as glottal pulses)
// and re-solve the *weighted covariance* normal equations each iteration until
// the robust scale estimate converges. The preprocessing (resample → DC removal
// → pre-emphasis → Gaussian window) and the root→formant extraction are shared
// with the Burg path; only the AR-coefficient estimation changes.
// ---------------------------------------------------------------------------

/// Autocorrelation LPC via Levinson-Durbin. Returns `a[1..=order]` of the
/// analysis filter `A(z) = 1 + a1 z^-1 + ... + ap z^-p` (the robust initial
/// guess, matching Praat's `Sound_to_LPC_autocorrelation`).
fn autocorrelation_lpc(x: &[f64], order: usize) -> Vec<f64> {
    let n = x.len();
    let mut a = vec![0.0; order + 1];
    if n <= order || order == 0 {
        return a;
    }
    // Autocorrelation r[0..=order].
    let mut r = vec![0.0; order + 1];
    for (lag, rl) in r.iter_mut().enumerate() {
        let mut s = 0.0;
        for i in lag..n {
            s += x[i] * x[i - lag];
        }
        *rl = s;
    }
    if r[0] <= 0.0 {
        return a;
    }
    // Levinson-Durbin recursion. `coef[1..=order]` are the predictor coeffs
    // (x[n] ≈ sum coef[k] x[n-k]); the analysis filter is a[k] = -coef[k].
    let mut coef = vec![0.0; order + 1];
    let mut err = r[0];
    let mut tmp = vec![0.0; order + 1];
    for i in 1..=order {
        let mut acc = r[i];
        for j in 1..i {
            acc -= coef[j] * r[i - j];
        }
        let k = if err.abs() > 1e-30 { acc / err } else { 0.0 };
        coef[i] = k;
        for j in 1..i {
            tmp[j] = coef[j] - k * coef[i - j];
        }
        for j in 1..i {
            coef[j] = tmp[j];
        }
        err *= 1.0 - k * k;
        if err <= 0.0 {
            break;
        }
    }
    for k in 1..=order {
        a[k] = -coef[k];
    }
    a
}

/// 0.5-quantile (Praat `NUMquantile` convention) of a *sorted* slice.
fn quantile_sorted(sorted: &[f64], f: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    // Praat: place = f*n + 0.5; interpolate between samples (1-based).
    let place = f * n as f64 + 0.5;
    if place <= 1.0 {
        return sorted[0];
    }
    if place >= n as f64 {
        return sorted[n - 1];
    }
    let left = place.floor() as usize; // 1-based index of lower sample
    let frac = place - left as f64;
    sorted[left - 1] + frac * (sorted[left] - sorted[left - 1])
}

/// Standard normal CDF Φ(x) and pdf φ(x), for the Huber scale bias correction.
fn gauss_p(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}
fn gauss_pdf(x: f64) -> f64 {
    (1.0 / (2.0 * PI).sqrt()) * (-0.5 * x * x).exp()
}

/// Complementary error function (Abramowitz & Stegun 7.1.26-style rational
/// approximation; |error| < 1.2e-7 — ample for the Huber bias constant).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587
                                        + t * (-0.82215223 + t * 0.17087277)))))))))
        .exp();
    if x >= 0.0 {
        ans
    } else {
        2.0 - ans
    }
}

/// Robust location + scale via Praat's `NUMstatistics_huber` (winsorized
/// M-estimate). Seeds from the median / MAD, then iterates.
fn huber_location_scale(
    x: &[f64],
    location: &mut f64,
    scale: &mut f64,
    k_stdev: f64,
    tol: f64,
    max_iter: usize,
    work: &mut [f64],
) {
    let n = x.len();
    if n == 0 {
        return;
    }
    let theta = 2.0 * gauss_p(k_stdev) - 1.0;
    let beta = theta + k_stdev * k_stdev * (1.0 - theta) - 2.0 * k_stdev * gauss_pdf(k_stdev);

    // MAD seed (median absolute deviation), location = median.
    let w = &mut work[..n];
    w.copy_from_slice(x);
    w.sort_by(|a, b| a.partial_cmp(b).unwrap());
    *location = quantile_sorted(w, 0.5);
    for (wi, &xi) in w.iter_mut().zip(x) {
        *wi = (xi - *location).abs();
    }
    w.sort_by(|a, b| a.partial_cmp(b).unwrap());
    *scale = 1.4826 * quantile_sorted(w, 0.5);

    if *scale <= 0.0 {
        return;
    }
    let dof = (n - 1) as f64; // wantlocation = true
    let mut iter = 0;
    loop {
        let prev_loc = *location;
        let prev_scale = *scale;
        let lo = *location - k_stdev * *scale;
        let hi = *location + k_stdev * *scale;
        // Winsorize into the work buffer.
        for (wi, &xi) in w.iter_mut().zip(x) {
            *wi = xi.clamp(lo, hi);
        }
        // Location = mean of winsorized.
        let mean = w.iter().sum::<f64>() / n as f64;
        *location = mean;
        let far_loc = (*location - prev_loc).abs() > (tol * location.abs()).max(f64::EPSILON);
        // Scale from sum of squared deviations / (dof * beta).
        let mut sumsq = 0.0;
        for &wi in w.iter() {
            let d = wi - *location;
            sumsq += d * d;
        }
        *scale = (sumsq / (dof * beta)).sqrt();
        let far_scale = (*scale - prev_scale).abs() > (tol * *scale).max(f64::EPSILON);
        iter += 1;
        if iter >= max_iter || !(far_scale || far_loc) {
            break;
        }
    }
}

/// Prediction residual e = A(z) x via the direct-form inverse filter, matching
/// Praat's `VECfilterInverse_inplace`: `e[i] = x[i] + sum_j a[j] * x[i-j]`.
fn inverse_filter(x: &[f64], a: &[f64], out: &mut [f64]) {
    let p = a.len() - 1;
    for i in 0..x.len() {
        let mut acc = x[i];
        for j in 1..=p {
            if i >= j {
                acc += a[j] * x[i - j];
            }
        }
        out[i] = acc;
    }
}

/// Solve the symmetric linear system `C a = c` (size p×p) by Gaussian
/// elimination with partial pivoting. Returns false if singular. `c` is
/// overwritten with the solution. Praat uses an SVD with a tolerance; for the
/// well-posed 10×10 weighted-covariance systems here LU with pivoting matches it
/// and is far cheaper. On (near-)singularity we report failure and the caller
/// keeps the previous coefficients (as Praat does on a failed frame).
fn solve_linear(c_mat: &mut [f64], p: usize, c: &mut [f64]) -> bool {
    // c_mat is row-major p×p.
    for col in 0..p {
        // Partial pivot.
        let mut piv = col;
        let mut best = c_mat[col * p + col].abs();
        for r in (col + 1)..p {
            let v = c_mat[r * p + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-30 {
            return false;
        }
        if piv != col {
            for k in 0..p {
                c_mat.swap(col * p + k, piv * p + k);
            }
            c.swap(col, piv);
        }
        let diag = c_mat[col * p + col];
        for r in (col + 1)..p {
            let factor = c_mat[r * p + col] / diag;
            if factor != 0.0 {
                for k in col..p {
                    c_mat[r * p + k] -= factor * c_mat[col * p + k];
                }
                c[r] -= factor * c[col];
            }
        }
    }
    // Back substitution.
    for col in (0..p).rev() {
        let mut acc = c[col];
        for k in (col + 1)..p {
            acc -= c_mat[col * p + k] * c[k];
        }
        c[col] = acc / c_mat[col * p + col];
    }
    true
}

/// Scratch buffers for the robust IRLS loop, allocated once per `analyze` call
/// (not per iteration) so the hot loop is allocation-free.
struct RobustScratch {
    error: Vec<f64>,
    weights: Vec<f64>,
    work: Vec<f64>,
    covar: Vec<f64>,      // p*p row-major
    covar_copy: Vec<f64>, // p*p scratch for the (destructive) solve
    rhs: Vec<f64>,        // p
}

/// Robust LPC for one preprocessed (resampled/windowed) frame `s`. `a_init` is
/// the autocorrelation seed (length order+1). Returns the robust analysis-filter
/// coefficients and the iteration count actually used (for bounded-cost checks).
fn robust_lpc(
    s: &[f64],
    a_init: &[f64],
    order: usize,
    k_stdev: f64,
    itermax: usize,
    tol: f64,
    scratch: &mut RobustScratch,
) -> (Vec<f64>, usize) {
    let n = s.len();
    let mut a = a_init.to_vec(); // current coeffs, a[1..=order]
    let mut location = 0.0;
    let mut scale = 1e308;
    let mut iter = 0;

    let RobustScratch { error, weights, work, covar, covar_copy, rhs } = scratch;

    loop {
        let prev_scale = scale;
        // Residual e = A(z) s.
        inverse_filter(s, &a, &mut error[..n]);
        // Robust location + scale of the residual.
        huber_location_scale(
            &error[..n],
            &mut location,
            &mut scale,
            k_stdev,
            tol,
            HUBER_ITERMAX,
            &mut work[..n],
        );
        // Huber weights: w = 1 inside ±k*scale, else (k*scale)/|e-loc|.
        let kstdev = k_stdev * scale;
        for (wt, &e) in weights[..n].iter_mut().zip(&error[..n]) {
            let ad = (e - location).abs();
            *wt = if kstdev > 0.0 && ad >= kstdev { kstdev / ad } else { 1.0 };
        }
        // Weighted covariance: covar[i][j] = sum_{k=p+1..N} s[k-j] s[k-i] w[k];
        // rhs[i] = - sum s[k-i] s[k] w[k].   (1-based i,j in Praat; 0-based here)
        for i in 1..=order {
            for j in i..=order {
                let mut cv1 = 0.0;
                for k in order..n {
                    cv1 += s[k - j] * s[k - i] * weights[k];
                }
                covar[(i - 1) * order + (j - 1)] = cv1;
                covar[(j - 1) * order + (i - 1)] = cv1;
            }
            let mut cv2 = 0.0;
            for k in order..n {
                cv2 += s[k - i] * s[k] * weights[k];
            }
            rhs[i - 1] = -cv2;
        }
        // Solve C a = rhs (solve is destructive, so copy into scratch). On
        // failure keep previous coeffs and stop.
        covar_copy.copy_from_slice(covar);
        if solve_linear(covar_copy, order, rhs) {
            for k in 1..=order {
                a[k] = rhs[k - 1];
            }
        } else {
            break;
        }
        iter += 1;
        let far = (scale - prev_scale).abs() > (tol * scale.abs()).max(f64::EPSILON);
        if iter >= itermax || !far {
            break;
        }
    }
    (a, iter)
}

/// Robust analogue of `analyze_lpc`: preprocess → autocorrelation seed → robust
/// IRLS → coefficients. Returns coeffs, post-resample sample rate, iteration
/// count.
fn analyze_lpc_robust(frame: &[f64], p: &FormantParams) -> (Vec<f64>, f64, usize) {
    let order = 2 * p.max_number_of_formants;
    let new_fs = resample_rate(p);
    if frame.len() < 32 || order == 0 {
        return (vec![0.0; order + 1], new_fs, 0);
    }
    let sig = preprocess(frame, p);
    if sig.len() <= order + 2 {
        return (vec![0.0; order + 1], new_fs, 0);
    }
    let n = sig.len();
    let a_init = autocorrelation_lpc(&sig, order);
    let mut scratch = RobustScratch {
        error: vec![0.0; n],
        weights: vec![0.0; n],
        work: vec![0.0; n],
        covar: vec![0.0; order * order],
        covar_copy: vec![0.0; order * order],
        rhs: vec![0.0; order],
    };
    let (a, iters) = robust_lpc(
        &sig,
        &a_init,
        order,
        ROBUST_K_STDEV,
        ROBUST_ITERMAX,
        ROBUST_TOL,
        &mut scratch,
    );
    (a, new_fs, iters)
}

/// Run the preprocessing pipeline and return the Burg LPC coefficients
/// `a[1..=order]` plus the post-resample sample rate. Exposed for the
/// us-vs-Praat LPC coefficient diagnostic (`tests/lpc_compare.rs`).
pub fn analyze_lpc(frame: &[f64], p: &FormantParams) -> (Vec<f64>, f64) {
    let order = 2 * p.max_number_of_formants;
    let new_fs = resample_rate(p);
    if frame.len() < 32 || order == 0 {
        return (vec![0.0; order + 1], new_fs);
    }
    let sig = preprocess(frame, p);
    if sig.len() <= order + 2 {
        return (vec![0.0; order + 1], new_fs);
    }
    (burg(&sig, order), new_fs)
}

/// Resample to `2*maxformant` → DC removal → pre-emphasis → Gaussian window,
/// returning the windowed signal ready for Burg.
///
/// The window spans the **whole resampled frame**, not a `window_length`
/// sub-window: the caller hands us one analysis buffer per estimate, and the
/// full frame both matches the oracle at the buffer midpoint and conditions the
/// order-`2N` Burg far better (short windows make nasal formants ill-conditioned
/// and spawn spurious poles). `window_length_s` is kept for API compatibility.
fn preprocess(frame: &[f64], p: &FormantParams) -> Vec<f64> {
    let sr_in = p.samplerate;
    let new_fs = resample_rate(p);

    let mut sig = resample(frame, sr_in, new_fs);
    if sig.len() < 16 {
        return Vec::new();
    }

    let mean = sig.iter().sum::<f64>() / sig.len() as f64;
    for v in sig.iter_mut() {
        *v -= mean;
    }

    if p.pre_emphasis_from_hz > 0.0 {
        let a = (-2.0 * PI * p.pre_emphasis_from_hz / new_fs).exp();
        for i in (1..sig.len()).rev() {
            sig[i] -= a * sig[i - 1];
        }
    }

    let w = gaussian_window(sig.len());
    for (s, wi) in sig.iter_mut().zip(&w) {
        *s *= wi;
    }
    sig
}

pub fn analyze_formants(frame: &[f64], p: &FormantParams) -> FormantResult {
    if p.robust {
        return analyze_formants_robust(frame, p);
    }
    // Burg path (kept for A/B and the Burg parity gate).
    let order = 2 * p.max_number_of_formants;
    let empty = FormantResult { formants: Vec::new(), bandwidths: Vec::new() };
    if frame.len() < 32 || order == 0 {
        return empty;
    }
    let sig = preprocess(frame, p);
    if sig.len() <= order + 2 {
        return empty;
    }
    extract_formants(&burg(&sig, order), order, resample_rate(p), p)
}

/// Robust formant extraction (Praat's IRLS/Huber LPC). Same signature as
/// `analyze_formants` but always takes the robust path regardless of
/// `p.robust`; the burg path is reachable by clearing `p.robust`.
pub fn analyze_formants_robust(frame: &[f64], p: &FormantParams) -> FormantResult {
    let order = 2 * p.max_number_of_formants;
    if frame.len() < 32 || order == 0 {
        return FormantResult { formants: Vec::new(), bandwidths: Vec::new() };
    }
    let (a, fs, _iters) = analyze_lpc_robust(frame, p);
    extract_formants(&a, order, fs, p)
}

/// Robust LPC iteration count for one frame (bounded-cost diagnostic).
pub fn robust_iterations(frame: &[f64], p: &FormantParams) -> usize {
    analyze_lpc_robust(frame, p).2
}

/// Solve for the LPC polynomial roots and convert to formants. Monomorphised per
/// `order` via a match so `aberth`'s const-generic TERMS is satisfied.
///
/// `A(z) = 1 + a1 z^-1 + ... + ap z^-p`; substituting `x = z^-1` gives the
/// ascending polynomial `1 + a1 x + ... + ap x^p`, whose roots `x_k` map back to
/// poles `z_k = 1/x_k`. Formant frequency from `arg(z)`, bandwidth from `|z|`.
fn extract_formants(a: &[f64], order: usize, fs: f64, p: &FormantParams) -> FormantResult {
    let mut coeffs = vec![0.0; order + 1];
    coeffs[0] = 1.0;
    for j in 1..=order {
        coeffs[j] = a[j];
    }

    let roots = solve_roots(&coeffs);

    let mut found: Vec<(f64, f64)> = Vec::new();
    for x in roots {
        let denom = x.re * x.re + x.im * x.im;
        if denom < 1e-30 {
            continue;
        }
        let zre = x.re / denom; // z = 1/x
        let zim = x.im / denom;
        // Take each conjugate pair once, from the upper half plane.
        if zim < 0.0 {
            continue;
        }
        let r = (zre * zre + zim * zim).sqrt();
        if r <= 0.0 {
            continue;
        }
        let freq = zim.atan2(zre).abs() * fs / (2.0 * PI);
        let bw = -r.ln() * fs / PI;
        // Praat's acceptance test: keep poles in (safety, Nyquist - safety) with
        // safety = 50 Hz and NO upper bandwidth cap (it keeps wide damped poles,
        // e.g. nasal /a/ F3 ~1920 Hz, bw ~1445 Hz).
        let nyq = fs / 2.0;
        let safety = 50.0;
        if freq > safety && freq < nyq - safety {
            found.push((freq, bw.abs()));
        }
    }

    found.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    found.truncate(p.max_number_of_formants);
    let (formants, bandwidths) = found.into_iter().unzip();
    FormantResult { formants, bandwidths }
}

/// Find roots of `c0 + c1 x + ... + cp x^p` (ascending coeffs) via Aberth.
/// Dispatches on degree to satisfy aberth's const-generic array length.
fn solve_roots(coeffs: &[f64]) -> Vec<aberth::Complex<f64>> {
    macro_rules! solve_n {
        ($n:literal) => {{
            let mut arr = [0.0f64; $n];
            for (i, c) in coeffs.iter().enumerate().take($n) {
                arr[i] = *c;
            }
            let roots = aberth(&arr, 100, 1e-12);
            roots.iter().copied().collect()
        }};
    }
    match coeffs.len() {
        1..=3 => solve_n!(3),
        4..=5 => solve_n!(5),
        6..=7 => solve_n!(7),
        8..=9 => solve_n!(9),
        10..=11 => solve_n!(11),
        12..=13 => solve_n!(13),
        _ => solve_n!(15),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthesize a vowel: a pulse-train glottal source through two formant
    /// resonators, so F1/F2 are at known frequencies.
    fn synth_vowel(f0: f64, formants: &[(f64, f64)], sr: f64, n: usize) -> Vec<f64> {
        // Impulse train.
        let mut src = vec![0.0; n];
        let period = (sr / f0).round() as usize;
        let mut i = 0;
        while i < n {
            src[i] = 1.0;
            i += period;
        }
        // Cascade of 2-pole resonators.
        let mut x = src;
        for &(fc, bw) in formants {
            let r = (-PI * bw / sr).exp();
            let theta = 2.0 * PI * fc / sr;
            let a1 = -2.0 * r * theta.cos();
            let a2 = r * r;
            let gain = 1.0 - 2.0 * r * theta.cos() + r * r; // normalize DC-ish
            let mut y = vec![0.0; n];
            for k in 0..n {
                let in_k = gain * x[k];
                let y1 = if k >= 1 { y[k - 1] } else { 0.0 };
                let y2 = if k >= 2 { y[k - 2] } else { 0.0 };
                y[k] = in_k - a1 * y1 - a2 * y2;
            }
            x = y;
        }
        x
    }

    #[test]
    fn detects_synthetic_vowel_formants() {
        let sr = 48000.0;
        let n = 4800;
        // Vowel /a/-ish: F1=700, F2=1200.
        let sig = synth_vowel(150.0, &[(700.0, 80.0), (1200.0, 90.0)], sr, n);
        let p = FormantParams { samplerate: sr, ..Default::default() };
        let r = analyze_formants(&sig, &p);
        assert!(r.formants.len() >= 2, "expected >=2 formants, got {:?}", r.formants);
        let f1 = r.formants[0];
        let f2 = r.formants[1];
        assert!((f1 - 700.0).abs() / 700.0 < 0.15, "F1 {f1} off");
        assert!((f2 - 1200.0).abs() / 1200.0 < 0.15, "F2 {f2} off");
    }

    #[test]
    fn empty_on_short_frame() {
        let p = FormantParams::default();
        assert!(analyze_formants(&[0.0; 8], &p).formants.is_empty());
    }
}
