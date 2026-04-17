from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional

import numpy as np
import parselmouth


@dataclass(frozen=True)
class AnalysisConfig:
    samplerate: int = 48000

    # Streaming / buffering
    block_size: int = 1024
    buffer_duration_s: float = 0.06  # rolling analysis buffer in seconds

    # Gating
    rms_threshold: float = 0.0005

    # Pitch (Praat / Parselmouth)
    pitch_time_step: float = 0.01
    pitch_floor_hz: float = 75.0
    pitch_ceiling_hz: float = 400.0
    pitch_silence_threshold: float = 0.03
    pitch_voicing_threshold: float = 0.45
    pitch_very_accurate: bool = False

    # Formants (Praat Burg)
    formant_time_step: float = 0.005
    max_number_of_formants: float = 5.0
    maximum_formant_hz: float = 5500.0
    window_length_s: float = 0.025
    pre_emphasis_from_hz: float = 50.0


@dataclass(frozen=True)
class AnalysisResult:
    voiced: bool
    rms: float
    pitch_hz: Optional[float]
    formants_hz: List[float]


def rms(frame: np.ndarray) -> float:
    frame = np.asarray(frame, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(frame)) + 1e-12))


class RealtimeAnalyzer:
    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config
        self.frames_seen = 0
        self._result_callback: Optional[Callable[[AnalysisResult], None]] = None
        self._init_buffer()

    def _init_buffer(self) -> None:
        maxlen = max(1, int(round(self.config.buffer_duration_s * self.config.samplerate)))
        self._buffer: Deque[float] = deque(maxlen=maxlen)

    def set_result_callback(self, callback: Callable[[AnalysisResult], None]) -> None:
        self._result_callback = callback

    def update_config(self, config: AnalysisConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.frames_seen = 0
        self._init_buffer()

    def push_audio(self, block: np.ndarray) -> Optional[AnalysisResult]:
        mono = np.asarray(block, dtype=np.float64).reshape(-1)
        self._buffer.extend(mono.tolist())
        self.frames_seen += 1

        if len(self._buffer) < 32:
            return None

        frame = np.asarray(self._buffer, dtype=np.float64)
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
                formants_hz=[],
            )

        snd = parselmouth.Sound(frame, sampling_frequency=float(self.config.samplerate))

        # Pitch
        pitch_obj = snd.to_pitch_ac(
            time_step=self.config.pitch_time_step,
            pitch_floor=self.config.pitch_floor_hz,
            very_accurate=self.config.pitch_very_accurate,
            silence_threshold=self.config.pitch_silence_threshold,
            voicing_threshold=self.config.pitch_voicing_threshold,
            pitch_ceiling=self.config.pitch_ceiling_hz,
        )

        t = snd.get_total_duration() / 2.0
        pitch_hz = pitch_obj.get_value_at_time(t)
        if pitch_hz is None or (isinstance(pitch_hz, float) and np.isnan(pitch_hz)):
            pitch_hz = None
            voiced = False
        else:
            pitch_hz = float(pitch_hz)
            voiced = True

        # Formants
        formant_obj = snd.to_formant_burg(
            time_step=self.config.formant_time_step,
            max_number_of_formants=self.config.max_number_of_formants,
            maximum_formant=self.config.maximum_formant_hz,
            window_length=self.config.window_length_s,
            pre_emphasis_from=self.config.pre_emphasis_from_hz,
        )

        formants: List[float] = []
        for i in range(1, int(self.config.max_number_of_formants) + 1):
            value = formant_obj.get_value_at_time(i, t)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            formants.append(float(value))

        return AnalysisResult(
            voiced=voiced,
            rms=current_rms,
            pitch_hz=pitch_hz,
            formants_hz=formants,
        )