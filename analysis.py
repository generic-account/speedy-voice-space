from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional

import librosa
import numpy as np
import scipy.signal


@dataclass(frozen=True)
class AnalysisConfig:
    samplerate: int = 16000
    frame_size: int = 1024
    lpc_order: int = 12
    rms_threshold: float = 0.0005
    pre_emphasis: float = 0.97
    max_formants: int = 4
    max_formant_bandwidth: float = 500.0
    min_formant_hz: float = 90.0
    max_formant_hz_margin: float = 50.0
    pitch_min_hz: float = 60.0
    pitch_max_hz: float = 500.0
    pitch_clarity_threshold: float = 0.30

    max_f1_hz: float = 1200.0
    max_f2_hz: float = 3000.0
    max_f3_hz: float = 4500.0


@dataclass(frozen=True)
class AnalysisResult:
    voiced: bool
    rms: float
    pitch_hz: Optional[float]
    pitch_clarity: float
    formants_hz: List[float]
    primary_resonance_hz: Optional[float]
    secondary_resonance_hz: Optional[float]
    lpc_spectrum_hz: Optional[np.ndarray] = None
    lpc_spectrum_db: Optional[np.ndarray] = None


def rms(frame: np.ndarray) -> float:
    frame = np.asarray(frame, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(frame)) + 1e-12))


def apply_pre_emphasis(frame: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    if len(frame) == 0:
        return frame
    return np.append(frame[0], frame[1:] - coeff * frame[:-1])

def extract_formants_from_lpc_roots(
    lpc_coeffs: np.ndarray,
    samplerate: int,
    *,
    max_formants: int = 4,
    max_bandwidth: float = 500.0,
    min_freq_hz: float = 90.0,
    max_freq_margin_hz: float = 50.0,
) -> List[float]:
    roots = np.roots(lpc_coeffs)
    roots = roots[np.imag(roots) >= 0]
    roots = roots[np.abs(roots) < 1.0]

    if len(roots) == 0:
        return []

    angles = np.angle(roots)
    freqs = angles * (samplerate / (2.0 * np.pi))
    bandwidths = -0.5 * (samplerate / (2.0 * np.pi)) * np.log(np.abs(roots))

    mask = (
        (freqs > min_freq_hz)
        & (freqs < samplerate / 2.0 - max_freq_margin_hz)
        & (bandwidths < max_bandwidth)
    )

    filtered = np.sort(freqs[mask])[:max_formants]
    return [float(f) for f in filtered]

def compute_lpc_spectrum(
    frame: np.ndarray,
    samplerate: int,
    lpc_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute LPC coefficients and the LPC spectral envelope.
    """
    coeffs = librosa.lpc(y=frame, order=lpc_order)
    freqs, response = scipy.signal.freqz(
        [1.0], coeffs, worN=len(frame) // 2, fs=samplerate
    )
    spectrum_db = 20.0 * np.log10(np.abs(response) + 1e-10)
    return coeffs, freqs, spectrum_db


def estimate_pitch_autocorrelation(
    frame: np.ndarray,
    samplerate: int,
    *,
    min_hz: float = 60.0,
    max_hz: float = 500.0,
    clarity_threshold: float = 0.30,
) -> tuple[Optional[float], float]:
    """
    Simple normalized autocorrelation pitch estimator.

    Returns:
        (pitch_hz, clarity)
    """
    x = np.asarray(frame, dtype=np.float32)
    if len(x) < 8:
        return None, 0.0

    x = x - np.mean(x)
    energy = float(np.sum(x * x))
    if energy <= 1e-10:
        return None, 0.0

    min_lag = max(1, int(samplerate / max_hz))
    max_lag = min(len(x) - 1, int(samplerate / min_hz))
    if max_lag <= min_lag:
        return None, 0.0

    corr = np.correlate(x, x, mode="full")
    corr = corr[len(corr) // 2 :]  # non-negative lags

    denom = corr[0] + 1e-12
    norm_corr = corr / denom

    search_region = norm_corr[min_lag : max_lag + 1]
    if len(search_region) == 0:
        return None, 0.0

    best_offset = int(np.argmax(search_region))
    best_lag = min_lag + best_offset
    clarity = float(search_region[best_offset])

    if clarity < clarity_threshold or best_lag <= 0:
        return None, clarity

    pitch_hz = float(samplerate / best_lag)
    return pitch_hz, clarity


class RealtimeAnalyzer:
    """
    Rolling-frame analyzer that consumes audio callback blocks and emits
    modular analysis results.

    Usage:
        analyzer = RealtimeAnalyzer(config)
        analyzer.set_result_callback(on_result)
        analyzer.push_audio(block)
    """

    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config
        self._buffer: Deque[float] = deque(maxlen=config.frame_size)
        self._result_callback: Optional[Callable[[AnalysisResult], None]] = None
        self.frames_seen = 0

    def set_result_callback(self, callback: Callable[[AnalysisResult], None]) -> None:
        self._result_callback = callback

    def reset(self) -> None:
        self._buffer = deque(maxlen=self.config.frame_size)
        self.frames_seen = 0

    def update_config(self, config: AnalysisConfig) -> None:
        self.config = config
        self.reset()

    def push_audio(self, block: np.ndarray) -> Optional[AnalysisResult]:
        mono = np.asarray(block, dtype=np.float32).reshape(-1)
        self._buffer.extend(mono.tolist())
        self.frames_seen += 1

        if len(self._buffer) < self.config.frame_size:
            return None

        frame = np.asarray(self._buffer, dtype=np.float32)
        result = self.analyze_frame(frame)

        if self._result_callback is not None:
            self._result_callback(result)

        return result

    def analyze_frame(self, frame: np.ndarray) -> AnalysisResult:
        current_rms = rms(frame)
        if current_rms < self.config.rms_threshold:
            return AnalysisResult(
                voiced=False,
                rms=current_rms,
                pitch_hz=None,
                pitch_clarity=0.0,
                formants_hz=[],
                primary_resonance_hz=None,
                secondary_resonance_hz=None,
                lpc_spectrum_hz=None,
                lpc_spectrum_db=None,
            )

        pitch_hz, pitch_clarity = estimate_pitch_autocorrelation(
            frame,
            self.config.samplerate,
            min_hz=self.config.pitch_min_hz,
            max_hz=self.config.pitch_max_hz,
            clarity_threshold=self.config.pitch_clarity_threshold,
        )

        emphasized = apply_pre_emphasis(frame, self.config.pre_emphasis)
        windowed = emphasized * np.hamming(len(emphasized))

        try:
            lpc_coeffs, spec_hz, spec_db = compute_lpc_spectrum(
                windowed,
                self.config.samplerate,
                self.config.lpc_order,
            )
            formants = extract_formants_from_lpc_roots(
                lpc_coeffs,
                self.config.samplerate,
                max_formants=self.config.max_formants,
                max_bandwidth=self.config.max_formant_bandwidth,
                min_freq_hz=self.config.min_formant_hz,
                max_freq_margin_hz=self.config.max_formant_hz_margin,
            )
        except Exception as exc:
            print(f"LPC analysis error: {exc!r}")
            spec_hz = None
            spec_db = None
            formants = []

        primary = formants[0] if len(formants) > 0 else None
        secondary = formants[1] if len(formants) > 1 else None

        return AnalysisResult(
            voiced=True,
            rms=current_rms,
            pitch_hz=pitch_hz,
            pitch_clarity=pitch_clarity,
            formants_hz=formants,
            primary_resonance_hz=primary,
            secondary_resonance_hz=secondary,
            lpc_spectrum_hz=spec_hz,
            lpc_spectrum_db=spec_db,
        )
