"""
Generate deterministic synthetic audio fixtures for pipeline validation.

Outputs 48 kHz / 16-bit / mono WAVs into ./fixtures so they can be fed into
Chromium's --use-file-for-fake-audio-capture (which loops the file) and used by
unit/parity tests later.

Run:  python tools/audio/gen_fixtures.py
"""
from __future__ import annotations

import os
import numpy as np
import soundfile as sf

SR = 48000
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures")
os.makedirs(OUT, exist_ok=True)


def write(name: str, x: np.ndarray, sr: int = SR) -> None:
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    path = os.path.join(OUT, name)
    sf.write(path, x, sr, subtype="PCM_16")
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))
    print(f"  {name:24s} {len(x)/sr:5.2f}s  rms={rms:.4f}")


def t(seconds: float, sr: int = SR) -> np.ndarray:
    return np.arange(int(seconds * sr)) / sr


def tone(freq: float, seconds: float, amp: float = 0.3) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * t(seconds))


def vowel(f0: float, formants, seconds: float, amp: float = 0.25) -> np.ndarray:
    """Crude source-filter vowel: harmonic-rich glottal source shaped by formant peaks."""
    time = t(seconds)
    # glottal-ish source: sum of decaying harmonics of f0
    src = np.zeros_like(time)
    for k in range(1, 40):
        if k * f0 > SR / 2:
            break
        src += (1.0 / k) * np.sin(2 * np.pi * k * f0 * time)
    src /= np.max(np.abs(src)) + 1e-9
    # emphasize formant regions by adding resonant tones at the formant freqs
    shaped = src.copy()
    for ff, g in formants:
        shaped += g * np.sin(2 * np.pi * ff * time)
    shaped /= np.max(np.abs(shaped)) + 1e-9
    return amp * shaped


def resonant_vowel(
    f0: float,
    formants,  # list of (freq_hz, bandwidth_hz)
    seconds: float,
    amp: float = 0.25,
) -> np.ndarray:
    """
    Source-filter vowel: glottal impulse train at f0 through a cascade of 2-pole
    resonators (one per formant). This produces overlapping broadband formant
    peaks like real speech — so closely spaced formants (e.g. /u/'s F1/F2) can
    merge under LPC, unlike the additive-sine `vowel()`. Better for stressing
    formant-tracking robustness.
    """
    n = int(seconds * SR)
    # Glottal source: impulse train at f0.
    src = np.zeros(n)
    period = SR / f0
    idx = 0.0
    while idx < n:
        src[int(idx)] = 1.0
        idx += period
    y = src.copy()
    for ff, bw in formants:
        r = np.exp(-np.pi * bw / SR)
        theta = 2.0 * np.pi * ff / SR
        a1 = 2.0 * r * np.cos(theta)
        a2 = -(r * r)
        out = np.zeros(n)
        for i in range(n):
            out[i] = y[i] + (a1 * out[i - 1] if i >= 1 else 0.0) + (
                a2 * out[i - 2] if i >= 2 else 0.0
            )
        y = out
    y /= np.max(np.abs(y)) + 1e-9
    return amp * y


def sweep(f_lo: float, f_hi: float, seconds: float, amp: float = 0.3) -> np.ndarray:
    time = t(seconds)
    k = (f_hi / f_lo) ** (1.0 / seconds)
    phase = 2 * np.pi * f_lo * (k**time - 1) / np.log(k)
    return amp * np.sin(phase)


def main() -> None:
    print(f"Writing fixtures to {OUT}")

    # Silence — meter/gate should read ~0, voiced=False
    write("silence.wav", np.zeros(int(2.0 * SR)))

    # Pure tones — deterministic level + known "pitch"
    write("tone_220hz.wav", tone(220.0, 3.0))
    write("tone_150hz.wav", tone(150.0, 3.0))

    # Synthetic vowels — known f0 + formant targets for later pitch/formant parity
    # /a/-ish: F1~700 F2~1200 ; /i/-ish: F1~300 F2~2300
    write("vowel_a_150hz.wav",
          vowel(150.0, [(700, 0.5), (1200, 0.35), (2600, 0.15)], 3.0))
    write("vowel_i_220hz.wav",
          vowel(220.0, [(300, 0.5), (2300, 0.4), (3000, 0.2)], 3.0))

    # Realistic source-filter vowels (broadband formant peaks).
    # /u/: F1~350 F2~800 (close together → the merging case the UI showed) F3~2400.
    write("vowel_u_130hz.wav",
          resonant_vowel(130.0, [(350, 60), (800, 90), (2400, 150)], 3.0))
    # /a/: F1~730 F2~1090 F3~2440 for a second realistic reference.
    write("vowel_a_res_130hz.wav",
          resonant_vowel(130.0, [(730, 80), (1090, 90), (2440, 150)], 3.0))

    # Nasalized vowels: nasalization adds a low nasal pole (~250 Hz) and broadens
    # / damps the oral formants (wider bandwidths). This is the noisy real-world
    # case the UI struggles with. Extra pole near F1 + widened bandwidths.
    write("vowel_u_nasal_130hz.wav",
          resonant_vowel(130.0,
                         [(250, 120), (380, 160), (800, 200), (2300, 250), (3300, 300)],
                         3.0))
    write("vowel_a_nasal_130hz.wav",
          resonant_vowel(130.0,
                         [(250, 120), (720, 180), (1100, 220), (2450, 260), (3400, 320)],
                         3.0))

    # Pitch glide — exercises smoothing/median over time
    write("sweep_120_300hz.wav", sweep(120.0, 300.0, 4.0))

    # Resample the downloaded real speech to 48k mono for a consistent fixture set
    real_8k = os.path.join(OUT, "real_speech_8k.wav")
    if os.path.exists(real_8k):
        data, sr = sf.read(real_8k, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data[:, 0]
        from scipy.signal import resample_poly
        data48 = resample_poly(data, up=SR, down=sr).astype(np.float32)
        write("real_speech_48k.wav", data48)
    else:
        print("  (real_speech_8k.wav not found — skipping 48k resample)")

    print("done.")


if __name__ == "__main__":
    main()
