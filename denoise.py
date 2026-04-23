from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import resample_poly

from pyrnnoise import RNNoise
from settings_defaults import (
    DEFAULT_DENOISE_FRAME_SIZE,
    DEFAULT_DENOISE_TARGET_SAMPLE_RATE,
    DEFAULT_NOISE_SUPPRESSION_ENABLED,
    DEFAULT_NOISE_SUPPRESSION_MIX,
)


def clamp(lo: float, hi: float, x: float) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class DenoiseSettings:
    enabled: bool = DEFAULT_NOISE_SUPPRESSION_ENABLED
    mix: float = DEFAULT_NOISE_SUPPRESSION_MIX
    target_sample_rate: int = DEFAULT_DENOISE_TARGET_SAMPLE_RATE
    frame_size: int = DEFAULT_DENOISE_FRAME_SIZE  # RNNoise native frame size at 48 kHz


class AudioPreprocessor:
    def process_block(
        self,
        block: np.ndarray,
        samplerate: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Returns:
            processed_block: same length as input block
            speech_prob: 0..1 estimate if available, else 0.0
        """
        raise NotImplementedError

    def reset(self) -> None:
        pass


class BypassPreprocessor(AudioPreprocessor):
    def process_block(
        self,
        block: np.ndarray,
        samplerate: int,
    ) -> Tuple[np.ndarray, float]:
        return np.asarray(block, dtype=np.float32), 0.0

    def reset(self) -> None:
        pass


class RNNoisePreprocessor(AudioPreprocessor):
    """
    Streaming RNNoise adapter.

    Internally:
    - resamples incoming audio to 48 kHz
    - accumulates 480-sample frames
    - denoises frame-by-frame
    - overlap-free concatenation
    - resamples back to original samplerate
    - returns exactly the same number of samples as input
    """

    def __init__(self, settings: Optional[DenoiseSettings] = None) -> None:
        self.settings = settings or DenoiseSettings()
        self._rnnoise = RNNoise(sample_rate=self.settings.target_sample_rate)
        self._in_buffer = np.zeros(0, dtype=np.float32)
        self._out_buffer = np.zeros(0, dtype=np.float32)
        self._last_speech_prob = 0.0

    def update_settings(self, settings: DenoiseSettings) -> None:
        self.settings = settings
        self.reset()

    def reset(self) -> None:
        self._in_buffer = np.zeros(0, dtype=np.float32)
        self._out_buffer = np.zeros(0, dtype=np.float32)
        self._last_speech_prob = 0.0
        self._rnnoise = RNNoise(sample_rate=self.settings.target_sample_rate)

    def _to_int16(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x = np.clip(x, -1.0, 1.0)
        return (x * 32767.0).astype(np.int16)

    def _to_float32(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if np.issubdtype(x.dtype, np.integer):
            return (x.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
        return x.astype(np.float32)

    def _resample_if_needed(
        self, x: np.ndarray, src_sr: int, dst_sr: int
    ) -> np.ndarray:
        x = self._ensure_mono_1d(x)
        if src_sr == dst_sr:
            return x.astype(np.float32, copy=False)
        return resample_poly(x, up=dst_sr, down=src_sr).astype(np.float32)

    def _ensure_mono_1d(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)

        if x.ndim == 0:
            return x.reshape(1).astype(np.float32)

        if x.ndim == 1:
            return x.astype(np.float32)

        # If shape is [channels, samples], use first channel
        if x.ndim == 2:
            if x.shape[0] == 1:
                return x[0].astype(np.float32)
            if x.shape[1] == 1:
                return x[:, 0].astype(np.float32)

            # Fall back to first channel
            return x[0].astype(np.float32)

        # Last-resort flatten
        return x.reshape(-1).astype(np.float32)

    def process_block(
        self,
        block: np.ndarray,
        samplerate: int,
    ) -> Tuple[np.ndarray, float]:
        block = np.asarray(block, dtype=np.float32).reshape(-1)

        if not self.settings.enabled:
            return block, 0.0

        mix = clamp(0.0, 1.0, float(self.settings.mix))
        target_sr = int(self.settings.target_sample_rate)
        frame_size = int(self.settings.frame_size)

        # Resample input block to RNNoise native rate
        block_48k = self._resample_if_needed(block, samplerate, target_sr)

        # Append to streaming input buffer
        self._in_buffer = np.concatenate([self._in_buffer, block_48k])

        denoised_frames = []

        while len(self._in_buffer) >= frame_size:
            frame = self._in_buffer[:frame_size]
            self._in_buffer = self._in_buffer[frame_size:]

            frame = self._ensure_mono_1d(frame)
            frame_i16 = self._to_int16(frame)

            # pyrnnoise chunk API expects [num_channels, num_samples]
            chunk_i16 = frame_i16.reshape(1, -1)

            chunk_results = list(self._rnnoise.denoise_chunk(chunk_i16))
            if not chunk_results:
                denoised_frame = frame.astype(np.float32)
                self._last_speech_prob = 0.0
            else:
                speech_prob, denoised_i16 = chunk_results[0]

                if isinstance(speech_prob, (list, tuple, np.ndarray)):
                    try:
                        self._last_speech_prob = float(np.mean(speech_prob))
                    except Exception:
                        self._last_speech_prob = 0.0
                else:
                    try:
                        self._last_speech_prob = float(speech_prob)
                    except Exception:
                        self._last_speech_prob = 0.0

                denoised_i16 = self._ensure_mono_1d(denoised_i16)
                denoised_frame = self._to_float32(denoised_i16)

            denoised_frame = self._ensure_mono_1d(denoised_frame)

            blended = (mix * denoised_frame) + ((1.0 - mix) * frame)
            denoised_frames.append(self._ensure_mono_1d(blended))

        if denoised_frames:
            produced_48k = np.concatenate(denoised_frames)
            produced_src = self._ensure_mono_1d(
                self._resample_if_needed(produced_48k, target_sr, samplerate)
            )
            self._out_buffer = np.concatenate(
                [self._out_buffer, produced_src.astype(np.float32)]
            )

        # Return exactly same number of samples as input
        needed = len(block)
        if len(self._out_buffer) >= needed:
            out = self._out_buffer[:needed]
            self._out_buffer = self._out_buffer[needed:]
        else:
            # Not enough denoised output yet; pad remainder with raw input tail
            shortage = needed - len(self._out_buffer)
            out = np.concatenate([self._out_buffer, block[-shortage:]]).astype(
                np.float32
            )
            self._out_buffer = np.zeros(0, dtype=np.float32)

        return out.astype(np.float32), clamp(0.0, 1.0, self._last_speech_prob)
