//! How close the live (per-window) pitch is to the BEST-case Praat: whole-file
//! `to_pitch_ac`, whose Viterbi path suppresses octave jumps. The oracle
//! (tools/oracle/gen_pitch_oracle.py) samples that whole-file contour at each
//! window midpoint; here we run our per-window analyzer on the same frames and
//! compare. Clean tones/vowels should match exactly; real speech currently
//! diverges (we jump per window, the reference is smooth) and this test reports
//! that gap so a cross-frame continuity fix can be measured against it.

use std::path::PathBuf;

use dsp::pitch::{PitchParams, PitchTracker};
use serde::Deserialize;

#[derive(Deserialize)]
struct Oracle {
    samplerate: u32,
    sample_scale: f64,
    window_samples: usize,
    pitch_floor_hz: f64,
    pitch_ceiling_hz: f64,
    windows: Vec<Window>,
}

#[derive(Deserialize)]
struct Window {
    start_sample: usize,
    whole_pitch_hz: Option<f64>,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn read_wav(name: &str, scale: f64) -> Vec<f64> {
    let path = root().join("tools/audio/fixtures").join(name);
    let mut reader =
        hound::WavReader::open(&path).unwrap_or_else(|e| panic!("open {:?}: {e}", path));
    reader.samples::<i16>().map(|s| s.unwrap() as f64 / scale).collect()
}

fn load(stem: &str) -> Oracle {
    let path = root().join("tools/oracle/expected_pitch").join(format!("{stem}.json"));
    let txt = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {:?}: {e}", path));
    serde_json::from_str(&txt).unwrap()
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn jump_pct(series: &[f64]) -> usize {
    if series.len() < 2 {
        return 0;
    }
    let j = (1..series.len()).filter(|&i| (series[i] / series[i - 1] - 1.0).abs() > 0.15).count();
    100 * j / series.len()
}

struct Stats {
    compared: usize,
    median_err_pct: f64,
    our_jump: usize,
    ref_jump: usize,
}

fn run(stem: &str, wav: &str) -> Stats {
    let o = load(stem);
    let samples = read_wav(wav, o.sample_scale);
    let p = PitchParams {
        samplerate: o.samplerate as f64,
        floor: o.pitch_floor_hz,
        ceiling: o.pitch_ceiling_hz,
        ..Default::default()
    };

    // Run the live ONLINE tracker across the windows in order (cross-frame path),
    // exactly as the app does block to block.
    let mut tracker = PitchTracker::new(p);
    let mut errs = Vec::new();
    let (mut ours, mut refs) = (Vec::new(), Vec::new());
    for w in &o.windows {
        let frame = &samples[w.start_sample..w.start_sample + o.window_samples];
        let got = tracker.analyze(frame).f0;
        let Some(reference) = w.whole_pitch_hz else { continue };
        if let Some(f) = got {
            errs.push((f - reference).abs() / reference * 100.0);
            ours.push(f);
            refs.push(reference);
        }
    }

    let s = Stats {
        compared: ours.len(),
        median_err_pct: median(errs),
        our_jump: jump_pct(&ours),
        ref_jump: jump_pct(&refs),
    };
    println!(
        "[{stem}] compared={} medianErr_vs_bestPraat={:.1}% jumpRate ours={}% bestPraat={}%",
        s.compared, s.median_err_pct, s.our_jump, s.ref_jump
    );
    s
}

#[test]
fn live_pitch_matches_best_praat_on_clean_signals() {
    for (stem, wav) in [
        ("tone_220hz", "tone_220hz.wav"),
        ("tone_150hz", "tone_150hz.wav"),
        ("vowel_a_150hz", "vowel_a_150hz.wav"),
        ("vowel_i_220hz", "vowel_i_220hz.wav"),
        ("vowel_u_130hz", "vowel_u_130hz.wav"),
    ] {
        let s = run(stem, wav);
        assert!(s.compared > 0, "{stem}: no comparable windows");
        assert!(
            s.median_err_pct < 5.0,
            "{stem}: live pitch differs from whole-file Praat by {:.1}%",
            s.median_err_pct
        );
    }
}

#[test]
fn live_pitch_tracks_best_praat_on_real_speech() {
    // The online octave-continuity tracker brings the live per-window jump rate
    // down to whole-file Praat's level (it was ~2x without it). Values track
    // closely and the jump rate is no worse than best-case Praat.
    let s = run("real_speech_48k", "real_speech_48k.wav");
    assert!(s.compared > 20, "real_speech: too few comparable windows");
    assert!(s.median_err_pct < 5.0, "value should track best Praat (got {:.1}%)", s.median_err_pct);
    assert!(
        s.our_jump <= s.ref_jump + 2,
        "live jump rate should match whole-file Praat (ours={}%, best={}%)",
        s.our_jump,
        s.ref_jump
    );
}
