//! us-vs-Praat LPC *coefficient* comparator (diagnostic infra).
//!
//! Reads `tools/oracle/expected/lpc_dump.json` (produced by
//! `tools/oracle/dump_lpc.py`) which holds Praat's Burg LPC coefficients for a
//! few vowel windows, recomputes our Burg coefficients for the same windows,
//! and prints them side by side. Built to localize a formant divergence to
//! Burg/preprocessing (coefficients differ) vs root extraction (coefficients
//! match but formants differ).
//!
//! NOTE on interpretation: `dump_lpc.py` computes Praat's coefficients via
//! `To LPC (burg)` over a `window_length` *sub-window*, whereas the production
//! `analyze_lpc` now models the *whole resampled frame* (which conditions the
//! ill-posed nasal LPC far better — see `formant.rs::preprocess`). So the
//! coefficients here are NOT expected to match bit-for-bit; the tool stays
//! useful for re-localizing future divergences (feed it matching windows).
//! The authoritative parity check is the formant-level `formant_parity` gate.
//!
//! Run: cargo test --release --test lpc_compare -- --nocapture

use std::path::PathBuf;

use dsp::formant::{analyze_lpc, FormantParams};
use serde::Deserialize;

#[derive(Deserialize)]
struct Dump {
    config: Config,
    samplerate: u32,
    sample_scale: f64,
    entries: Vec<Entry>,
}
#[derive(Deserialize)]
struct Config {
    max_number_of_formants: f64,
    maximum_formant_hz: f64,
    window_length_s: f64,
    pre_emphasis_from_hz: f64,
}
#[derive(Deserialize)]
struct Entry {
    window_index: usize,
    start_sample: usize,
    window_samples: usize,
    order: usize,
    praat_lpc: Vec<f64>,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

#[test]
fn compare_lpc() {
    let txt = std::fs::read_to_string(root().join("tools/oracle/expected/lpc_dump.json"))
        .expect("run tools/oracle/dump_lpc.py first");
    let dumps: std::collections::BTreeMap<String, Dump> =
        serde_json::from_str(&txt).unwrap();

    for (stem, d) in &dumps {
        let mut reader = hound::WavReader::open(
            root().join("tools/audio/fixtures").join(format!("{stem}.wav")),
        )
        .unwrap();
        let samples: Vec<f64> = reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f64 / d.sample_scale)
            .collect();
        let p = FormantParams {
            samplerate: d.samplerate as f64,
            max_number_of_formants: d.config.max_number_of_formants as usize,
            maximum_formant_hz: d.config.maximum_formant_hz,
            window_length_s: d.config.window_length_s,
            pre_emphasis_from_hz: d.config.pre_emphasis_from_hz,
            robust: false,
        };

        for e in &d.entries {
            let frame = &samples[e.start_sample..e.start_sample + e.window_samples];
            let (a, _fs) = analyze_lpc(frame, &p);
            // a[1..=order] is the analysis filter; compare to praat_lpc.
            let ours: Vec<f64> = (1..=e.order).map(|k| a[k]).collect();
            let mut max_abs = 0.0f64;
            for (o, pr) in ours.iter().zip(&e.praat_lpc) {
                max_abs = max_abs.max((o - pr).abs());
            }
            println!("[{stem} w{}]  order={}  maxCoeffDiff={:.4}", e.window_index, e.order, max_abs);
            println!("  praat: {:?}", round_vec(&e.praat_lpc));
            println!("  ours : {:?}", round_vec(&ours));
        }
        println!();
    }
}

fn round_vec(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| (x * 10000.0).round() / 10000.0).collect()
}
