//! Parity gate: the Rust formant port vs. the Python/Parselmouth oracle.
//!
//! For each fixture we read the SAME PCM16/48k WAV and the oracle JSON, run the
//! Rust formant analyzer over the SAME analysis windows, and compare F1/F2
//! against Praat's `to_formant_burg().get_value_at_time(i, t_mid)`. Formants are
//! intrinsically noisier than pitch (LPC root assignment, resampling, Praat's
//! exact windowing), so we hold a LENIENT median-absolute-error bar on F1/F2 —
//! the values that drive the app's resonance score. F3+ is reported only.

use std::path::PathBuf;

use dsp::formant::{analyze_formants, FormantParams};
use serde::Deserialize;

#[derive(Deserialize)]
struct Oracle {
    samplerate: u32,
    sample_scale: f64,
    window_samples: usize,
    config: Config,
    windows: Vec<Window>,
}

#[derive(Deserialize)]
struct Config {
    max_number_of_formants: f64,
    maximum_formant_hz: f64,
    window_length_s: f64,
    pre_emphasis_from_hz: f64,
    rms_threshold: f64,
}

#[derive(Deserialize)]
struct Window {
    start_sample: usize,
    rms: f64,
    voiced: bool,
    #[serde(default)]
    formants_hz: Vec<f64>,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn read_wav(name: &str, scale: f64) -> Vec<f64> {
    let path = root().join("tools/audio/fixtures").join(name);
    let mut reader =
        hound::WavReader::open(&path).unwrap_or_else(|e| panic!("open {:?}: {e}", path));
    reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f64 / scale)
        .collect()
}

fn load_oracle(stem: &str) -> Oracle {
    let path = root().join("tools/oracle/expected").join(format!("{stem}.json"));
    let txt =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {:?}: {e}", path));
    serde_json::from_str(&txt).unwrap()
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

struct FStats {
    compared: usize,
    median_f1_pct: f64,
    median_f2_pct: f64,
    median_f3_pct: f64,
}

fn run_fixture(stem: &str, wav: &str) -> FStats {
    let o = load_oracle(stem);
    let samples = read_wav(wav, o.sample_scale);
    let p = FormantParams {
        samplerate: o.samplerate as f64,
        max_number_of_formants: o.config.max_number_of_formants as usize,
        maximum_formant_hz: o.config.maximum_formant_hz,
        window_length_s: o.config.window_length_s,
        pre_emphasis_from_hz: o.config.pre_emphasis_from_hz,
        robust: false, // this gate validates the Burg path vs the Burg oracle
    };

    let (mut e1, mut e2, mut e3) = (Vec::new(), Vec::new(), Vec::new());
    let mut compared = 0usize;

    for w in &o.windows {
        // Only compare on windows Praat treated as voiced & above the RMS gate,
        // and that actually have oracle formants.
        if !w.voiced || w.rms < o.config.rms_threshold || w.formants_hz.is_empty() {
            continue;
        }
        let frame = &samples[w.start_sample..w.start_sample + o.window_samples];
        let got = analyze_formants(frame, &p).formants;
        if got.is_empty() {
            continue;
        }
        compared += 1;
        let exp = &w.formants_hz;
        if !exp.is_empty() && !got.is_empty() {
            e1.push((got[0] - exp[0]).abs() / exp[0] * 100.0);
        }
        if exp.len() >= 2 && got.len() >= 2 {
            e2.push((got[1] - exp[1]).abs() / exp[1] * 100.0);
        }
        if exp.len() >= 3 && got.len() >= 3 {
            e3.push((got[2] - exp[2]).abs() / exp[2] * 100.0);
        }
    }

    let s = FStats {
        compared,
        median_f1_pct: median(e1),
        median_f2_pct: median(e2),
        median_f3_pct: median(e3),
    };
    println!(
        "[{stem}] compared={} medianErr F1={:.1}% F2={:.1}% F3={:.1}%",
        s.compared, s.median_f1_pct, s.median_f2_pct, s.median_f3_pct
    );
    s
}

#[test]
fn formant_parity_vowels() {
    // Sustained synthetic vowels — the cleanest formant targets.
    for (stem, wav) in [
        ("vowel_a_150hz", "vowel_a_150hz.wav"),
        ("vowel_i_220hz", "vowel_i_220hz.wav"),
    ] {
        let s = run_fixture(stem, wav);
        assert!(s.compared > 0, "{stem}: no comparable windows");
        assert!(
            s.median_f1_pct < 12.0,
            "{stem}: median F1 error {:.1}% >= 12%",
            s.median_f1_pct
        );
        assert!(
            s.median_f2_pct < 12.0,
            "{stem}: median F2 error {:.1}% >= 12%",
            s.median_f2_pct
        );
    }
}

#[test]
fn formant_parity_nasal_and_resonant_vowels() {
    // Regression guard for nasalised vowels. Praat keeps a WIDE (heavily
    // damped) F3 pole on these — e.g. nasal /a/ F3 ≈ 1920 Hz, bw ≈ 1450 Hz.
    // An earlier MAX_BANDWIDTH cap dropped that pole, so our "F3" became
    // Praat's F4 (~2600+ Hz), a ~30% error. The Fourier-domain resampler +
    // full-frame Burg now recover the real wide F3, so we assert tight F1/F2/F3
    // parity here. /u/, nasal /u/ and nasal /a/ are clean; the resonant /a/ has
    // a genuine near-degenerate F3/F4 pair so it gets a looser F3 bar.
    for (stem, wav, f3_bar) in [
        ("vowel_u_130hz", "vowel_u_130hz.wav", 8.0),
        ("vowel_u_nasal_130hz", "vowel_u_nasal_130hz.wav", 8.0),
        ("vowel_a_nasal_130hz", "vowel_a_nasal_130hz.wav", 8.0),
        ("vowel_a_res_130hz", "vowel_a_res_130hz.wav", 15.0),
    ] {
        let s = run_fixture(stem, wav);
        assert!(s.compared > 0, "{stem}: no comparable windows");
        assert!(
            s.median_f1_pct < 8.0,
            "{stem}: median F1 error {:.1}% >= 8%",
            s.median_f1_pct
        );
        assert!(
            s.median_f2_pct < 8.0,
            "{stem}: median F2 error {:.1}% >= 8%",
            s.median_f2_pct
        );
        assert!(
            s.median_f3_pct < f3_bar,
            "{stem}: median F3 error {:.1}% >= {f3_bar}% (nasal F3 regression?)",
            s.median_f3_pct
        );
    }
}

#[test]
fn formant_parity_real_speech_report() {
    // Natural speech: report numbers, hold a very lenient bar on F1/F2.
    let s = run_fixture("real_speech_48k", "real_speech_48k.wav");
    assert!(s.compared > 0, "real speech: no comparable windows");
    assert!(
        s.median_f1_pct < 10.0 && s.median_f2_pct < 10.0 && s.median_f3_pct < 10.0,
        "real speech: F1={:.1}% F2={:.1}% F3={:.1}% (10% bar)",
        s.median_f1_pct,
        s.median_f2_pct,
        s.median_f3_pct
    );
}
