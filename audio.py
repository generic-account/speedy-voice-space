from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd


AudioFrameCallback = Callable[[np.ndarray], None]


@dataclass(frozen=True)
class AudioDevice:
    """Simple wrapper around an input-capable sounddevice entry."""
    index: int
    name: str
    max_input_channels: int
    default_samplerate: float

    @property
    def display_name(self) -> str:
        return f"{self.index}: {self.name}"


def list_input_devices() -> List[AudioDevice]:
    """
    Return all input-capable audio devices.

    The original AURORA code filters sounddevice devices by
    max_input_channels > 0. This keeps the same behavior.
    """
    devices = sd.query_devices()
    result: List[AudioDevice] = []

    for idx, dev in enumerate(devices):
        if int(dev.get("max_input_channels", 0)) > 0:
            result.append(
                AudioDevice(
                    index=idx,
                    name=str(dev.get("name", f"Device {idx}")),
                    max_input_channels=int(dev.get("max_input_channels", 0)),
                    default_samplerate=float(dev.get("default_samplerate", 16000.0)),
                )
            )
    return result


class AudioInputManager:
    """
    Manages a single-channel sounddevice InputStream.

    This class is intentionally UI-agnostic. You can use it from PyQt,
    a CLI tool, or another application layer.
    """

    def __init__(
        self,
        samplerate: int,
        blocksize: int,
        channels: int = 1,
        dtype: str = "float32",
    ) -> None:
        self.samplerate = int(samplerate)
        self.blocksize = int(blocksize)
        self.channels = int(channels)
        self.dtype = dtype

        self._devices: List[AudioDevice] = list_input_devices()
        self._selected_device_index: Optional[int] = None
        self._stream: Optional[sd.InputStream] = None
        self._frame_callback: Optional[AudioFrameCallback] = None

    @property
    def devices(self) -> List[AudioDevice]:
        return self._devices

    @property
    def selected_device(self) -> Optional[AudioDevice]:
        if self._selected_device_index is None:
            return None
        for dev in self._devices:
            if dev.index == self._selected_device_index:
                return dev
        return None

    @property
    def is_running(self) -> bool:
        return self._stream is not None

    def refresh_devices(self) -> List[AudioDevice]:
        self._devices = list_input_devices()
        return self._devices

    def set_frame_callback(self, callback: AudioFrameCallback) -> None:
        self._frame_callback = callback

    def _audio_callback(self, indata, frames, time, status) -> None:
        if status:
            print(f"Audio status: {status}")

        if self._frame_callback is None:
            return

        try:
            # flatten to mono float array
            frame = np.asarray(indata[:, 0], dtype=np.float32).copy()
            self._frame_callback(frame)
        except Exception as exc:
            print(f"Audio callback error: {exc!r}")

    def check_input_settings(
        self,
        device_index: Optional[int] = None,
        samplerate: Optional[int] = None,
        channels: Optional[int] = None,
    ) -> Optional[str]:
        """
        Returns None if settings are valid, otherwise returns the error string.
        """
        if device_index is None:
            device_index = self._selected_device_index
        if samplerate is None:
            samplerate = self.samplerate
        if channels is None:
            channels = self.channels

        try:
            sd.check_input_settings(
                device=device_index,
                channels=channels,
                samplerate=samplerate,
                dtype=self.dtype,
            )
            return None
        except Exception as exc:
            return str(exc)

    def start(self, device_index: Optional[int] = None) -> None:
        if self._frame_callback is None:
            raise RuntimeError("No frame callback set. Call set_frame_callback() first.")

        if device_index is not None:
            self._selected_device_index = int(device_index)

        if self._selected_device_index is None:
            if not self._devices:
                raise RuntimeError("No audio input devices found.")
            self._selected_device_index = self._devices[0].index

        self.stop()

        selected = self.selected_device
        if selected is None:
            raise RuntimeError("Selected audio device not found.")

        requested_samplerate = int(self.samplerate)
        fallback_samplerate = int(round(selected.default_samplerate))

        # Try requested samplerate first
        try_rates = [requested_samplerate]
        if fallback_samplerate not in try_rates:
            try_rates.append(fallback_samplerate)

        last_exc = None

        for rate in try_rates:
            try:
                self._stream = sd.InputStream(
                    device=self._selected_device_index,
                    callback=self._audio_callback,
                    channels=self.channels,
                    samplerate=rate,
                    blocksize=self.blocksize,
                    dtype=self.dtype,
                )
                self._stream.start()
                self.samplerate = rate
                print(
                    f"Started input stream on '{selected.name}' "
                    f"at {rate} Hz (blocksize={self.blocksize})"
                )
                return
            except Exception as exc:
                last_exc = exc
                self._stream = None
                print(
                    f"Failed to start '{selected.name}' at {rate} Hz: {exc}"
                )

        raise RuntimeError(
            f"Could not start audio stream for device '{selected.name}'. "
            f"Tried sample rates: {try_rates}. Last error: {last_exc}"
        ) from last_exc


    def stop(self) -> None:
        if self._stream is None:
            return

        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

    def restart(
        self,
        *,
        device_index: Optional[int] = None,
        samplerate: Optional[int] = None,
        blocksize: Optional[int] = None,
    ) -> None:
        if samplerate is not None:
            self.samplerate = int(samplerate)
        if blocksize is not None:
            self.blocksize = int(blocksize)
        self.start(device_index=device_index)

    def close(self) -> None:
        self.stop()
