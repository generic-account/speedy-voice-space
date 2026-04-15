from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Deque, Optional, Sequence
import math

from analysis import AnalysisResult


@dataclass(frozen=True)
class ProcessingSettings:
    pitch_median_window: int = 5
    resonance_median_window: int = 7
    pitch_alpha: float = 0.25
    resonance_alpha: float = 0.15

    f2_low_hz: float = 600.0
    f2_high_hz: float = 3000.0
    f3_low_hz: float = 1500.0
    f3_high_hz: float = 4500.0
    f2_weight: float = 0.6
    f3_weight: float = 0.4

    f2_max_jump_hz: float = 500.0
    f3_max_jump_hz: float = 700.0

@dataclass(frozen=True)
class DisplayState:
    voiced: bool
    rms: float
    raw_pitch_hz: Optional[float]
    filtered_pitch_hz: Optional[float]

    raw_resonance: Optional[float]
    filtered_resonance: Optional[float]
    resonance_confidence: float

    raw_f2_hz: Optional[float]
    raw_f3_hz: Optional[float]
    norm_f2: Optional[float]
    norm_f3: Optional[float]

    formants_hz: list[float]


def clamp(lo: float, hi: float, x: float) -> float:
    return max(lo, min(hi, x))


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def norm_mel_01(hz: float, lo_hz: float, hi_hz: float) -> float:
    mel = hz_to_mel(hz)
    mel_lo = hz_to_mel(lo_hz)
    mel_hi = hz_to_mel(hi_hz)
    if mel_hi <= mel_lo:
        return 0.0
    return clamp(0.0, 1.0, (mel - mel_lo) / (mel_hi - mel_lo))


def smooth_value(
    previous: Optional[float],
    new_value: Optional[float],
    alpha: float,
) -> Optional[float]:
    if new_value is None:
        return previous
    if previous is None:
        return new_value
    return alpha * new_value + (1.0 - alpha) * previous


def median_filtered_value(
    window: Deque[float],
    new_value: Optional[float],
) -> Optional[float]:
    if new_value is not None:
        window.append(float(new_value))
    if not window:
        return None
    return float(median(window))


class VoiceProcessor:
    def __init__(self, settings: Optional[ProcessingSettings] = None) -> None:
        self.settings = settings or ProcessingSettings()
        self.tracked_f2_hz: Optional[float] = None
        self.tracked_f3_hz: Optional[float] = None
        self._init_buffers()

    def _init_buffers(self) -> None:
        self.raw_pitch_window: Deque[float] = deque(
            maxlen=max(1, int(self.settings.pitch_median_window))
        )
        self.raw_resonance_window: Deque[float] = deque(
            maxlen=max(1, int(self.settings.resonance_median_window))
        )
        self.smoothed_pitch: Optional[float] = None
        self.smoothed_resonance: Optional[float] = None

    def _pick_closest_candidate(
        self,
        candidates: Sequence[float],
        previous: Optional[float],
        max_jump_hz: float,
    ) -> Optional[float]:
        if not candidates:
            return None

        if previous is None:
            return float(candidates[0])

        best = min(candidates, key=lambda x: abs(x - previous))
        if abs(best - previous) > max_jump_hz:
            return None

        return float(best)

    def select_tracked_formants(
        self,
        candidates_hz: Sequence[float],
    ) -> tuple[Optional[float], Optional[float]]:
        s = self.settings
        candidates = sorted(float(f) for f in candidates_hz)

        f2_candidates = [f for f in candidates if s.f2_low_hz <= f <= s.f2_high_hz]
        f3_candidates = [f for f in candidates if s.f3_low_hz <= f <= s.f3_high_hz]

        f2 = self._pick_closest_candidate(
            f2_candidates,
            self.tracked_f2_hz,
            s.f2_max_jump_hz,
        )

        if f2 is not None:
            f3_candidates = [f for f in f3_candidates if abs(f - f2) > 1e-6]

        f3 = self._pick_closest_candidate(
            f3_candidates,
            self.tracked_f3_hz,
            s.f3_max_jump_hz,
        )

        # Enforce ordering if both exist
        if f2 is not None and f3 is not None and f3 <= f2:
            valid_f3 = [f for f in f3_candidates if f > f2]
            f3 = self._pick_closest_candidate(
                valid_f3,
                self.tracked_f3_hz,
                s.f3_max_jump_hz,
            )

        self.tracked_f2_hz = f2
        self.tracked_f3_hz = f3

        return f2, f3

    def update_settings(self, settings: ProcessingSettings) -> None:
        self.settings = settings

        self.raw_pitch_window = deque(
            self.raw_pitch_window,
            maxlen=max(1, int(settings.pitch_median_window)),
        )
        self.raw_resonance_window = deque(
            self.raw_resonance_window,
            maxlen=max(1, int(settings.resonance_median_window)),
        )

        self.smoothed_pitch = None
        self.smoothed_resonance = None
        self.tracked_f2_hz = None
        self.tracked_f3_hz = None

    def reset(self) -> None:
        self.tracked_f2_hz = None
        self.tracked_f3_hz = None
        self._init_buffers()

    def compute_resonance(
        self,
        formants_hz: Sequence[float],
    ) -> tuple[Optional[float], float, Optional[float], Optional[float], Optional[float], Optional[float]]:
        f2, f3 = self.select_tracked_formants(formants_hz)

        norm_f2: Optional[float] = None
        norm_f3: Optional[float] = None

        values: list[float] = []
        weights: list[float] = []

        if f2 is not None:
            norm_f2 = norm_mel_01(
                float(f2),
                self.settings.f2_low_hz,
                self.settings.f2_high_hz,
            )
            values.append(norm_f2)
            weights.append(self.settings.f2_weight)

        if f3 is not None:
            norm_f3 = norm_mel_01(
                float(f3),
                self.settings.f3_low_hz,
                self.settings.f3_high_hz,
            )
            values.append(norm_f3)
            weights.append(self.settings.f3_weight)

        if not values:
            return None, 0.0, f2, f3, norm_f2, norm_f3

        total_weight = sum(weights)
        if total_weight <= 0:
            return None, 0.0, f2, f3, norm_f2, norm_f3

        score = sum(v * (w / total_weight) for v, w in zip(values, weights))
        confidence = clamp(0.0, 1.0, total_weight)
        return float(score), float(confidence), f2, f3, norm_f2, norm_f3

    def process(self, result: AnalysisResult) -> DisplayState:
        raw_resonance, confidence, raw_f2, raw_f3, norm_f2, norm_f3 = self.compute_resonance(
            result.formants_hz
        )

        median_pitch = median_filtered_value(
            self.raw_pitch_window,
            result.pitch_hz,
        )
        median_resonance = median_filtered_value(
            self.raw_resonance_window,
            raw_resonance,
        )

        self.smoothed_pitch = smooth_value(
            self.smoothed_pitch,
            median_pitch,
            self.settings.pitch_alpha,
        )
        self.smoothed_resonance = smooth_value(
            self.smoothed_resonance,
            median_resonance,
            self.settings.resonance_alpha,
        )

        return DisplayState(
            voiced=result.voiced,
            rms=result.rms,
            raw_pitch_hz=result.pitch_hz,
            filtered_pitch_hz=self.smoothed_pitch,
            raw_resonance=raw_resonance,
            filtered_resonance=self.smoothed_resonance,
            resonance_confidence=confidence,
            raw_f2_hz=raw_f2,
            raw_f3_hz=raw_f3,
            norm_f2=norm_f2,
            norm_f3=norm_f3,
            formants_hz=list(result.formants_hz),
        )