from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional
from denoise import DenoiseSettings, RNNoisePreprocessor
from settings_defaults import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_BUFFER_DURATION_S,
    DEFAULT_FORMANT_TIME_STEP,
    DEFAULT_MAX_NUMBER_OF_FORMANTS,
    DEFAULT_MAXIMUM_FORMANT_HZ,
    DEFAULT_NOISE_SUPPRESSION_ENABLED,
    DEFAULT_NOISE_SUPPRESSION_MIX,
    DEFAULT_PITCH_CEILING_HZ,
    DEFAULT_PITCH_FLOOR_HZ,
    DEFAULT_PITCH_SILENCE_THRESHOLD,
    DEFAULT_PITCH_TIME_STEP,
    DEFAULT_PITCH_VERY_ACCURATE,
    DEFAULT_PITCH_VOICING_THRESHOLD,
    DEFAULT_PRE_EMPHASIS_FROM_HZ,
    DEFAULT_RMS_THRESHOLD,
    DEFAULT_SAMPLERATE,
    DEFAULT_WINDOW_LENGTH_S,
)

import numpy as np
import parselmouth


@dataclass(frozen=True)
class AnalysisConfig:
    samplerate: int = DEFAULT_SAMPLERATE

    # Streaming / buffering
    block_size: int = DEFAULT_BLOCK_SIZE
    buffer_duration_s: float = DEFAULT_BUFFER_DURATION_S  # rolling analysis buffer in seconds

    # Gating
    rms_threshold: float = DEFAULT_RMS_THRESHOLD

    # Pitch (Praat / Parselmouth)
    pitch_time_step: float = DEFAULT_PITCH_TIME_STEP
    pitch_floor_hz: float = DEFAULT_PITCH_FLOOR_HZ
    pitch_ceiling_hz: float = DEFAULT_PITCH_CEILING_HZ
    pitch_silence_threshold: float = DEFAULT_PITCH_SILENCE_THRESHOLD
    pitch_voicing_threshold: float = DEFAULT_PITCH_VOICING_THRESHOLD
    pitch_very_accurate: bool = DEFAULT_PITCH_VERY_ACCURATE

    # Formants (Praat Burg)
    formant_time_step: float = DEFAULT_FORMANT_TIME_STEP
    max_number_of_formants: float = DEFAULT_MAX_NUMBER_OF_FORMANTS
    maximum_formant_hz: float = DEFAULT_MAXIMUM_FORMANT_HZ
    window_length_s: float = DEFAULT_WINDOW_LENGTH_S
    pre_emphasis_from_hz: float = DEFAULT_PRE_EMPHASIS_FROM_HZ

    noise_suppression_enabled: bool = DEFAULT_NOISE_SUPPRESSION_ENABLED
    noise_suppression_mix: float = DEFAULT_NOISE_SUPPRESSION_MIX


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
        self.preprocessor = RNNoisePreprocessor(self._denoise_settings())
        self.frames_seen = 0
        self._result_callback: Optional[Callable[[AnalysisResult], None]] = None
        self._init_buffer()

    def _denoise_settings(self) -> DenoiseSettings:
        return DenoiseSettings(
            enabled=self.config.noise_suppression_enabled,
            mix=self.config.noise_suppression_mix,
        )

    def _init_buffer(self) -> None:
        maxlen = max(
            1, int(round(self.config.buffer_duration_s * self.config.samplerate))
        )
        self._buffer: Deque[float] = deque(maxlen=maxlen)

    def set_result_callback(self, callback: Callable[[AnalysisResult], None]) -> None:
        self._result_callback = callback

    def update_config(self, config: AnalysisConfig) -> None:
        self.config = config
        self.preprocessor.update_settings(self._denoise_settings())
        self.reset()

    def reset(self) -> None:
        self.frames_seen = 0
        self.preprocessor.reset()
        self._init_buffer()

    def push_audio(self, block: np.ndarray) -> Optional[AnalysisResult]:
        mono = np.asarray(block, dtype=np.float32).reshape(-1)
        processed, _ = self.preprocessor.process_block(mono, self.config.samplerate)
        self._buffer.extend(processed.astype(np.float64).tolist())
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
