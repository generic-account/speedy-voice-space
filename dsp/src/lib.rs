//! WebAssembly DSP core for speedy-voice-space.
//!
//! Pure-Rust signal processing (pitch, formants, noise suppression), exposed to
//! JS via wasm-bindgen. The algorithm modules are plain Rust so they can be
//! unit/parity-tested natively against the Python oracle.

pub mod denoise;
pub mod formant;
pub mod pitch;

use wasm_bindgen::prelude::*;

/// Wasm handle around a configured pitch analyzer. JS passes a mono
/// `Float32Array` window; `analyze` returns f0 (Hz) or NaN when unvoiced.
#[wasm_bindgen]
pub struct PitchAnalyzer {
    tracker: pitch::PitchTracker,
}

#[wasm_bindgen]
impl PitchAnalyzer {
    #[wasm_bindgen(constructor)]
    pub fn new(samplerate: f64) -> PitchAnalyzer {
        PitchAnalyzer {
            tracker: pitch::PitchTracker::new(pitch::PitchParams { samplerate, ..Default::default() }),
        }
    }

    #[wasm_bindgen(js_name = setRange)]
    pub fn set_range(&mut self, floor: f64, ceiling: f64) {
        self.tracker.params.floor = floor;
        self.tracker.params.ceiling = ceiling;
    }

    #[wasm_bindgen(js_name = setThresholds)]
    pub fn set_thresholds(&mut self, silence: f64, voicing: f64) {
        self.tracker.params.silence_threshold = silence;
        self.tracker.params.voicing_threshold = voicing;
    }

    #[wasm_bindgen(js_name = setVeryAccurate)]
    pub fn set_very_accurate(&mut self, very_accurate: bool) {
        self.tracker.params.very_accurate = very_accurate;
    }

    /// Drop cross-frame pitch history (call on a voicing gap or config change).
    #[wasm_bindgen(js_name = resetTracking)]
    pub fn reset_tracking(&mut self) {
        self.tracker.reset();
    }

    /// Returns f0 in Hz, or NaN if the window is unvoiced.
    #[wasm_bindgen]
    pub fn analyze(&mut self, frame: &[f32]) -> f64 {
        let buf: Vec<f64> = frame.iter().map(|&v| v as f64).collect();
        match self.tracker.analyze(&buf).f0 {
            Some(f) => f,
            None => f64::NAN,
        }
    }
}

/// Wasm handle around a configured formant analyzer (Praat Burg LPC). JS passes
/// a mono `Float32Array` window; `analyze` returns F1..Fn (Hz, ascending).
#[wasm_bindgen]
pub struct FormantAnalyzer {
    params: formant::FormantParams,
}

#[wasm_bindgen]
impl FormantAnalyzer {
    #[wasm_bindgen(constructor)]
    pub fn new(samplerate: f64) -> FormantAnalyzer {
        FormantAnalyzer {
            params: formant::FormantParams { samplerate, ..Default::default() },
        }
    }

    #[wasm_bindgen(js_name = setMaxNumberOfFormants)]
    pub fn set_max_number_of_formants(&mut self, n: usize) {
        self.params.max_number_of_formants = n;
    }

    #[wasm_bindgen(js_name = setMaximumFormant)]
    pub fn set_maximum_formant(&mut self, hz: f64) {
        self.params.maximum_formant_hz = hz;
    }

    #[wasm_bindgen(js_name = setWindowLength)]
    pub fn set_window_length(&mut self, seconds: f64) {
        self.params.window_length_s = seconds;
    }

    #[wasm_bindgen(js_name = setPreEmphasisFrom)]
    pub fn set_pre_emphasis_from(&mut self, hz: f64) {
        self.params.pre_emphasis_from_hz = hz;
    }

    /// Toggle the robust (IRLS/Huber-reweighted) LPC path. On by default; clear
    /// for the legacy Burg estimator (A/B). The worker keeps working unchanged —
    /// `analyze`/`analyzeDetailed` honour this flag.
    #[wasm_bindgen(js_name = setRobust)]
    pub fn set_robust(&mut self, robust: bool) {
        self.params.robust = robust;
    }

    /// Returns formant frequencies F1..Fn in Hz (ascending). Empty if the
    /// window is too short or no formants are found.
    #[wasm_bindgen]
    pub fn analyze(&self, frame: &[f32]) -> Vec<f64> {
        let buf: Vec<f64> = frame.iter().map(|&v| v as f64).collect();
        formant::analyze_formants(&buf, &self.params).formants
    }

    /// Returns interleaved [f1, bw1, f2, bw2, ...] (Hz) for all candidate poles,
    /// so a stateful tracker (in JS) can assign poles to formant slots by
    /// continuity and down-weight wide (spurious) poles.
    #[wasm_bindgen(js_name = analyzeDetailed)]
    pub fn analyze_detailed(&self, frame: &[f32]) -> Vec<f64> {
        let buf: Vec<f64> = frame.iter().map(|&v| v as f64).collect();
        let r = formant::analyze_formants(&buf, &self.params);
        let mut out = Vec::with_capacity(r.formants.len() * 2);
        for (f, b) in r.formants.iter().zip(r.bandwidths.iter()) {
            out.push(*f);
            out.push(*b);
        }
        out
    }
}

/// Streaming noise suppressor (RNNoise via nnnoiseless). JS pushes mono blocks
/// at 48 kHz; we return a same-length denoised/dry-mixed block.
#[wasm_bindgen]
pub struct Denoiser {
    inner: denoise::Denoiser,
}

#[wasm_bindgen]
impl Denoiser {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Denoiser {
        Denoiser { inner: denoise::Denoiser::new() }
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Process a block; `mix` in [0,1] (1 = fully denoised, 0 = passthrough).
    pub fn process(&mut self, block: &[f32], mix: f64) -> Vec<f32> {
        self.inner.process_block(block, mix as f32)
    }

    #[wasm_bindgen(js_name = lastSpeechProb)]
    pub fn last_speech_prob(&self) -> f64 {
        self.inner.last_speech_prob() as f64
    }
}

impl Default for Denoiser {
    fn default() -> Self {
        Self::new()
    }
}
