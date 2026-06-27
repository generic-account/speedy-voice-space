//! us-vs-Praat formant comparator (diagnostic, not a hard gate).
//!
//! For each fixture, aligns our Rust formants against Praat's per frame and
//! reports: per-slot median error, how often F1/F2/F3 match within tolerance,
//! and how often we emit extra/missing poles vs Praat (the slot-shift cause).
//! Also dumps a few sample frames with bandwidths.
//!
//! Run: cargo test --release --test formant_compare -- --nocapture

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
    pre_emphasis_from_hz: f64,
    rms_threshold: f64,
}
#[derive(Deserialize)]
struct Window {
    start_sample: usize,
    rms: f64,
    voiced: bool,
    #[serde(default)]
    formant_freq_hz: Vec<f64>,
    #[serde(default)]
    formant_bw_hz: Vec<f64>,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn load(stem: &str) -> (Oracle, Vec<f64>) {
    let txt = std::fs::read_to_string(
        root().join("tools/oracle/expected").join(format!("{stem}.json")),
    )
    .unwrap();
    let o: Oracle = serde_json::from_str(&txt).unwrap();
    let mut reader = hound::WavReader::open(
        root().join("tools/audio/fixtures").join(format!("{stem}.wav")),
    )
    .unwrap();
    let scale = o.sample_scale;
    let samples = reader.samples::<i16>().map(|s| s.unwrap() as f64 / scale).collect();
    (o, samples)
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn params(o: &Oracle) -> FormantParams {
    FormantParams {
        samplerate: o.samplerate as f64,
        max_number_of_formants: o.config.max_number_of_formants as usize,
        maximum_formant_hz: o.config.maximum_formant_hz,
        pre_emphasis_from_hz: o.config.pre_emphasis_from_hz,
        robust: false,
    }
}

fn compare(stem: &str, verbose: usize) {
    let (o, samples) = load(stem);
    let p = params(&o);

    let mut slot_err: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut slot_within10: [usize; 3] = [0; 3];
    let mut frames = 0usize;
    let mut extra = 0usize; // we emit MORE poles than Praat
    let mut missing = 0usize; // we emit FEWER poles than Praat
    let mut shown = 0usize;

    for w in &o.windows {
        if !w.voiced || w.rms < o.config.rms_threshold || w.formant_freq_hz.is_empty() {
            continue;
        }
        let frame = &samples[w.start_sample..w.start_sample + o.window_samples];
        let res = analyze_formants(frame, &p);
        frames += 1;

        if res.formants.len() > w.formant_freq_hz.len() {
            extra += 1;
        } else if res.formants.len() < w.formant_freq_hz.len() {
            missing += 1;
        }

        for s in 0..3 {
            if let (Some(&got), Some(&exp)) =
                (res.formants.get(s), w.formant_freq_hz.get(s))
            {
                let e = (got - exp).abs() / exp * 100.0;
                slot_err[s].push(e);
                if e <= 10.0 {
                    slot_within10[s] += 1;
                }
            }
        }

        if shown < verbose {
            shown += 1;
            let ours: Vec<String> = res
                .formants
                .iter()
                .zip(&res.bandwidths)
                .map(|(f, b)| format!("{:.0}/{:.0}", f, b))
                .collect();
            let theirs: Vec<String> = w
                .formant_freq_hz
                .iter()
                .zip(&w.formant_bw_hz)
                .map(|(f, b)| format!("{:.0}/{:.0}", f, b))
                .collect();
            println!("    ours  (f/bw): {ours:?}");
            println!("    praat (f/bw): {theirs:?}");
        }
    }

    println!(
        "[{stem}] frames={frames}  extraPole={extra} missingPole={missing}",
    );
    for s in 0..3 {
        let n = slot_err[s].len();
        let pct = if n > 0 {
            100.0 * slot_within10[s] as f64 / n as f64
        } else {
            0.0
        };
        println!(
            "    F{}: medianErr={:.1}%  within10%={:.0}% ({}/{})",
            s + 1,
            median(slot_err[s].clone()),
            pct,
            slot_within10[s],
            n
        );
    }
}

#[test]
fn compare_all() {
    for stem in [
        "vowel_u_130hz",
        "vowel_u_nasal_130hz",
        "vowel_a_res_130hz",
        "vowel_a_nasal_130hz",
        "real_speech_48k",
    ] {
        compare(stem, 2);
        println!();
    }
}
