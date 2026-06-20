//! Parity gate: the Rust pitch port vs. the Python/Parselmouth oracle.
//!
//! For each fixture we read the SAME PCM16/48k WAV and the oracle JSON, run the
//! Rust pitch analyzer over the SAME analysis windows, and compare f0 on the
//! windows Praat marked voiced. Tones are held to a tight tolerance; natural
//! speech is reported and held to an aggregate threshold (octave errors on a
//! few frames are expected and absorbed by downstream median filtering).

use std::path::PathBuf;

use dsp::pitch::{analyze_pitch, PitchParams};
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
    pitch_floor_hz: f64,
    pitch_ceiling_hz: f64,
    pitch_silence_threshold: f64,
    pitch_voicing_threshold: f64,
    pitch_very_accurate: bool,
    rms_threshold: f64,
}

#[derive(Deserialize)]
struct Window {
    start_sample: usize,
    rms: f64,
    voiced: bool,
    pitch_hz: Option<f64>,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn read_wav(name: &str, scale: f64) -> Vec<f64> {
    let path = root().join("tools/audio/fixtures").join(name);
    let mut reader = hound::WavReader::open(&path)
        .unwrap_or_else(|e| panic!("open {:?}: {e}", path));
    reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f64 / scale)
        .collect()
}

fn load_oracle(stem: &str) -> Oracle {
    let path = root().join("tools/oracle/expected").join(format!("{stem}.json"));
    let txt = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {:?}: {e}", path));
    serde_json::from_str(&txt).unwrap()
}

struct Stats {
    voiced_compared: usize,
    within_tol: usize,
    median_abs_pct: f64,
    voicing_agree: usize,
    total: usize,
}

fn run_fixture(stem: &str, wav: &str, tol_pct: f64) -> Stats {
    let o = load_oracle(stem);
    let samples = read_wav(wav, o.sample_scale);
    let p = PitchParams {
        samplerate: o.samplerate as f64,
        floor: o.config.pitch_floor_hz,
        ceiling: o.config.pitch_ceiling_hz,
        silence_threshold: o.config.pitch_silence_threshold,
        voicing_threshold: o.config.pitch_voicing_threshold,
        octave_cost: 0.01,
        very_accurate: o.config.pitch_very_accurate,
    };

    let mut pct_errors = Vec::new();
    let mut within = 0usize;
    let mut voicing_agree = 0usize;
    let mut voiced_compared = 0usize;

    for w in &o.windows {
        let frame = &samples[w.start_sample..w.start_sample + o.window_samples];
        let res = analyze_pitch(frame, &p);

        // Apply the same RMS gate the Python path applies upstream of pitch.
        let gated_voiced = res.voiced && w.rms >= o.config.rms_threshold;
        if gated_voiced == w.voiced {
            voicing_agree += 1;
        }

        if w.voiced {
            if let (Some(exp), Some(got)) = (w.pitch_hz, res.f0) {
                voiced_compared += 1;
                let pct = (got - exp).abs() / exp * 100.0;
                pct_errors.push(pct);
                if pct <= tol_pct {
                    within += 1;
                }
            }
        }
    }

    pct_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if pct_errors.is_empty() {
        0.0
    } else {
        pct_errors[pct_errors.len() / 2]
    };

    let stats = Stats {
        voiced_compared,
        within_tol: within,
        median_abs_pct: median,
        voicing_agree,
        total: o.windows.len(),
    };
    println!(
        "[{stem}] windows={} voicing_agree={}/{} voiced_compared={} within {:.1}%tol={}/{} median_err={:.3}%",
        stats.total,
        stats.voicing_agree,
        stats.total,
        stats.voiced_compared,
        tol_pct,
        stats.within_tol,
        stats.voiced_compared,
        stats.median_abs_pct,
    );
    stats
}

#[test]
fn pitch_parity_tones_tight() {
    // Pure tones: the algorithm should land essentially on Praat's value.
    for (stem, wav) in [
        ("tone_150hz", "tone_150hz.wav"),
        ("tone_220hz", "tone_220hz.wav"),
    ] {
        let s = run_fixture(stem, wav, 1.0);
        assert!(
            s.within_tol as f64 >= 0.98 * s.voiced_compared as f64,
            "{stem}: only {}/{} within 1% (median {:.3}%)",
            s.within_tol,
            s.voiced_compared,
            s.median_abs_pct
        );
        assert!(
            s.voicing_agree as f64 >= 0.98 * s.total as f64,
            "{stem}: voicing agreement {}/{}",
            s.voicing_agree,
            s.total
        );
    }
}

#[test]
fn pitch_parity_vowels_and_sweep() {
    // Harmonic-rich synthetic signals: allow a little more slack.
    for (stem, wav) in [
        ("vowel_a_150hz", "vowel_a_150hz.wav"),
        ("vowel_i_220hz", "vowel_i_220hz.wav"),
        ("sweep_120_300hz", "sweep_120_300hz.wav"),
    ] {
        let s = run_fixture(stem, wav, 2.0);
        assert!(
            s.median_abs_pct <= 2.0,
            "{stem}: median pitch error {:.3}% > 2%",
            s.median_abs_pct
        );
    }
}

#[test]
fn pitch_parity_real_speech_aggregate() {
    // Natural speech: report and hold an aggregate bar. Per-frame octave slips
    // are expected; downstream median filtering handles them.
    let s = run_fixture("real_speech_48k", "real_speech_48k.wav", 5.0);
    let frac = s.within_tol as f64 / s.voiced_compared.max(1) as f64;
    assert!(
        frac >= 0.70,
        "real speech: only {:.0}% of voiced frames within 5% ({}/{})",
        frac * 100.0,
        s.within_tol,
        s.voiced_compared
    );
    assert!(
        s.voicing_agree as f64 >= 0.75 * s.total as f64,
        "real speech: voicing agreement {}/{}",
        s.voicing_agree,
        s.total
    );
}

#[test]
fn silence_unvoiced() {
    let s = run_fixture("silence", "silence.wav", 1.0);
    assert_eq!(s.voicing_agree, s.total, "silence should be fully unvoiced");
}
