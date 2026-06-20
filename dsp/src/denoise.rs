//! Streaming noise suppression backed by `nnnoiseless` (pure-Rust RNNoise). The
//! pipeline runs at 48 kHz (RNNoise's native rate), so we just accumulate
//! 480-sample frames, denoise, and dry/wet mix — no resampling.
//!
//! nnnoiseless expects f32 samples in i16 amplitude range (±32768), so we scale
//! up on the way in and back down on the way out.

use nnnoiseless::DenoiseState;

const FRAME: usize = DenoiseState::FRAME_SIZE; // 480
const SCALE: f32 = 32768.0;

pub struct Denoiser {
    state: Box<DenoiseState<'static>>,
    in_buf: Vec<f32>,
    out_buf: Vec<f32>,
    last_vad: f32,
}

impl Denoiser {
    pub fn new() -> Self {
        Denoiser {
            state: DenoiseState::new(),
            in_buf: Vec::new(),
            out_buf: Vec::new(),
            last_vad: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.state = DenoiseState::new();
        self.in_buf.clear();
        self.out_buf.clear();
        self.last_vad = 0.0;
    }

    pub fn last_speech_prob(&self) -> f32 {
        self.last_vad
    }

    /// Process one block, returning the same number of samples. `mix` in [0,1]
    /// blends denoised (1.0) vs. raw (0.0), matching denoise.py.
    pub fn process_block(&mut self, block: &[f32], mix: f32) -> Vec<f32> {
        let mix = mix.clamp(0.0, 1.0);
        self.in_buf.extend_from_slice(block);

        let mut frame_in = [0.0f32; FRAME];
        let mut frame_out = [0.0f32; FRAME];

        let mut consumed = 0;
        while self.in_buf.len() - consumed >= FRAME {
            let raw = &self.in_buf[consumed..consumed + FRAME];
            for i in 0..FRAME {
                frame_in[i] = raw[i] * SCALE;
            }
            self.last_vad = self.state.process_frame(&mut frame_out, &frame_in);
            for i in 0..FRAME {
                let denoised = frame_out[i] / SCALE;
                self.out_buf.push(mix * denoised + (1.0 - mix) * raw[i]);
            }
            consumed += FRAME;
        }
        if consumed > 0 {
            self.in_buf.drain(0..consumed);
        }

        // Return exactly block.len() samples; pad with the raw block tail if the
        // streaming denoiser hasn't produced enough yet (mirrors denoise.py).
        let needed = block.len();
        if self.out_buf.len() >= needed {
            self.out_buf.drain(0..needed).collect()
        } else {
            let mut out: Vec<f32> = self.out_buf.drain(..).collect();
            let shortage = needed - out.len();
            out.extend_from_slice(&block[block.len() - shortage..]);
            out
        }
    }
}

impl Default for Denoiser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_preserving_and_reduces_noise() {
        let mut d = Denoiser::new();
        // Pseudo-random white noise (deterministic LCG — no Date/rand needed).
        let mut seed: u32 = 12345;
        let mut rng = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1 << 24) as f32 * 2.0 - 1.0
        };
        let block: Vec<f32> = (0..1024).map(|_| 0.2 * rng()).collect();

        // Warm up the denoiser over several blocks, then measure.
        let mut last = vec![];
        for _ in 0..30 {
            last = d.process_block(&block, 1.0);
            assert_eq!(last.len(), block.len());
        }
        let raw_rms =
            (block.iter().map(|x| x * x).sum::<f32>() / block.len() as f32).sqrt();
        let out_rms =
            (last.iter().map(|x| x * x).sum::<f32>() / last.len() as f32).sqrt();
        assert!(
            out_rms < raw_rms,
            "denoised noise rms {out_rms} should be below raw {raw_rms}"
        );
    }

    #[test]
    fn dry_mix_is_passthrough() {
        let mut d = Denoiser::new();
        let block: Vec<f32> = (0..512)
            .map(|i| 0.1 * (i as f32 * 0.05).sin())
            .collect();
        let out = d.process_block(&block, 0.0);
        for (a, b) in block.iter().zip(&out) {
            assert!((a - b).abs() < 1e-6, "mix=0 must pass through");
        }
    }
}
