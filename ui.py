from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Deque, Optional

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from audio import AudioInputManager
from analysis import AnalysisConfig, AnalysisResult, RealtimeAnalyzer

from statistics import median


class AnalysisEmitter(QtCore.QObject):
    result_ready = QtCore.Signal(object)


class MainWindow(QtWidgets.QWidget):
    """
    Minimal PyQtGraph UI for live pitch vs resonance plotting.

    X-axis: pitch (Hz)
    Y-axis: primary resonance / F1 (Hz)

    A secondary resonance label is also shown, and the latest point leaves
    a short trail for motion visibility.
    """

    def __init__(self, config: Optional[AnalysisConfig] = None) -> None:
        super().__init__()

        self.config = config or AnalysisConfig()
        self.analyzer = RealtimeAnalyzer(self.config)
        self.audio = AudioInputManager(
            samplerate=self.config.samplerate,
            blocksize=self.config.frame_size,
            channels=1,
        )

        self.pitch_history: Deque[float] = deque(maxlen=120)
        self.resonance_history: Deque[float] = deque(maxlen=120)

        self.smoothed_pitch: Optional[float] = None
        self.smoothed_resonance: Optional[float] = None

        self.raw_pitch_window: Deque[float] = deque(maxlen=5)
        self.raw_resonance_window: Deque[float] = deque(maxlen=7)

        self.emitter = AnalysisEmitter()
        self.emitter.result_ready.connect(self._update_ui)

        self._build_ui()
        self._wire_callbacks()
        self._populate_devices()

    def _update_window_size(self, window: Deque[float], new_size: int) -> Deque[float]:
        return deque(window, maxlen=new_size)

    def _median_filtered_value(
        self,
        window: Deque[float],
        new_value: Optional[float],
    ) -> Optional[float]:
        if new_value is None:
            if not window:
                return None
            return float(median(window))

        window.append(float(new_value))
        if not window:
            return None
        return float(median(window))

    def _smooth_value(
        self,
        previous: Optional[float],
        new_value: Optional[float],
        alpha: float,
    ) -> Optional[float]:
        if new_value is None:
            return previous
        if previous is None:
            return new_value
        return alpha * new_value + (1.0 - alpha) * previous

    def _build_ui(self) -> None:
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")

        self.setWindowTitle("Pitch / Resonance Monitor")
        self.resize(1100, 700)

        layout = QtWidgets.QHBoxLayout(self)

        # Left: chart
        self.graphics = pg.GraphicsLayoutWidget()
        self.plot = self.graphics.addPlot(title="Real-time Pitch vs Resonance")
        self.plot.setLabel("bottom", "Pitch (Hz)")
        self.plot.setLabel("left", "Primary Resonance / F1 (Hz)")
        self.plot.setXRange(60, 500)
        self.plot.setYRange(100, 1200)
        self.plot.showGrid(x=True, y=True, alpha=0.25)

        self.trail_curve = self.plot.plot(
            pen=None,
            symbol="o",
            symbolSize=6,
        )
        self.current_point = self.plot.plot(
            pen=None,
            symbol="o",
            symbolSize=12,
        )

        layout.addWidget(self.graphics, stretch=4)

        # Right: controls
        controls = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(controls)

        self.device_dropdown = QtWidgets.QComboBox()

        self.fs_box = QtWidgets.QSpinBox()
        self.fs_box.setRange(8000, 48000)
        self.fs_box.setValue(self.config.samplerate)

        self.frame_box = QtWidgets.QSpinBox()
        self.frame_box.setRange(256, 4096)
        self.frame_box.setSingleStep(128)
        self.frame_box.setValue(self.config.frame_size)

        self.lpc_box = QtWidgets.QSpinBox()
        self.lpc_box.setRange(4, 30)
        self.lpc_box.setValue(self.config.lpc_order)

        self.threshold_box = QtWidgets.QDoubleSpinBox()
        self.threshold_box.setRange(0.0, 0.20)
        self.threshold_box.setDecimals(6)
        self.threshold_box.setSingleStep(0.0001)
        self.threshold_box.setValue(self.config.rms_threshold)

        self.pitch_min_box = QtWidgets.QSpinBox()
        self.pitch_min_box.setRange(40, 400)
        self.pitch_min_box.setValue(int(self.config.pitch_min_hz))

        self.pitch_max_box = QtWidgets.QSpinBox()
        self.pitch_max_box.setRange(80, 1200)
        self.pitch_max_box.setValue(int(self.config.pitch_max_hz))

        self.pitch_smoothing_box = QtWidgets.QDoubleSpinBox()
        self.pitch_smoothing_box.setRange(0.01, 1.00)
        self.pitch_smoothing_box.setDecimals(2)
        self.pitch_smoothing_box.setSingleStep(0.05)
        self.pitch_smoothing_box.setValue(0.25)

        self.resonance_smoothing_box = QtWidgets.QDoubleSpinBox()
        self.resonance_smoothing_box.setRange(0.01, 1.00)
        self.resonance_smoothing_box.setDecimals(2)
        self.resonance_smoothing_box.setSingleStep(0.05)
        self.resonance_smoothing_box.setValue(0.15)

        self.pitch_median_window_box = QtWidgets.QSpinBox()
        self.pitch_median_window_box.setRange(1, 15)
        self.pitch_median_window_box.setValue(5)

        self.resonance_median_window_box = QtWidgets.QSpinBox()
        self.resonance_median_window_box.setRange(1, 15)
        self.resonance_median_window_box.setValue(7)

        self.start_button = QtWidgets.QPushButton("Start")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.apply_button = QtWidgets.QPushButton("Apply Settings")

        self.pitch_label = QtWidgets.QLabel("Pitch: —")
        self.res1_label = QtWidgets.QLabel("Resonance 1: —")
        self.res2_label = QtWidgets.QLabel("Resonance 2: —")
        self.rms_label = QtWidgets.QLabel("RMS: —")
        self.voice_label = QtWidgets.QLabel("Voiced: —")

        form.addRow("Input Device", self.device_dropdown)
        form.addRow("Sample Rate", self.fs_box)
        form.addRow("Frame Size", self.frame_box)
        form.addRow("LPC Order", self.lpc_box)
        form.addRow("RMS Threshold", self.threshold_box)
        form.addRow("Pitch Min (Hz)", self.pitch_min_box)
        form.addRow("Pitch Max (Hz)", self.pitch_max_box)
        form.addRow("Pitch Median Window", self.pitch_median_window_box)
        form.addRow("Resonance Median Window", self.resonance_median_window_box)
        form.addRow("Pitch Smoothing", self.pitch_smoothing_box)
        form.addRow("Resonance Smoothing", self.resonance_smoothing_box)
        form.addRow(self.start_button)
        form.addRow(self.stop_button)
        form.addRow(self.apply_button)
        form.addRow("Pitch", self.pitch_label)
        form.addRow("Primary Resonance", self.res1_label)
        form.addRow("Secondary Resonance", self.res2_label)
        form.addRow("RMS", self.rms_label)
        form.addRow("Voiced", self.voice_label)

        layout.addWidget(controls, stretch=1)

        # Keeps Qt responsive similarly to the original app pattern.
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(lambda: None)
        self.timer.start(30)

    def _wire_callbacks(self) -> None:
        self.audio.set_frame_callback(self.analyzer.push_audio)
        self.analyzer.set_result_callback(self._handle_analysis_result)

        self.start_button.clicked.connect(self.start_audio)
        self.stop_button.clicked.connect(self.stop_audio)
        self.apply_button.clicked.connect(self.apply_settings)
        self.device_dropdown.currentIndexChanged.connect(self._on_device_changed)

    def _populate_devices(self) -> None:
        self.audio.refresh_devices()
        self.device_dropdown.clear()

        for dev in self.audio.devices:
            self.device_dropdown.addItem(dev.display_name, userData=dev.index)

    def _get_selected_device_index(self) -> Optional[int]:
        idx = self.device_dropdown.currentIndex()
        if idx < 0:
            return None
        return self.device_dropdown.itemData(idx)

    def _on_device_changed(self, _: int) -> None:
        device_index = self._get_selected_device_index()
        if device_index is None:
            return

        selected = None
        for dev in self.audio.devices:
            if dev.index == device_index:
                selected = dev
                break

        if selected is not None:
            device_rate = int(round(selected.default_samplerate))
            self.fs_box.blockSignals(True)
            self.fs_box.setValue(device_rate)
            self.fs_box.blockSignals(False)

        if self.audio.is_running:
            self.apply_settings()

    def _handle_analysis_result(self, result: AnalysisResult) -> None:
        # audio callback thread -> marshal onto Qt thread
        # QtCore.QTimer.singleShot(0, lambda r=result: self._update_ui(r))
        self.emitter.result_ready.emit(result)

    def _update_ui(self, result: AnalysisResult) -> None:
        self.rms_label.setText(f"{result.rms:.4f}")
        self.voice_label.setText("Yes" if result.voiced else "No")

        pitch_window_size = int(self.pitch_median_window_box.value())
        res_window_size = int(self.resonance_median_window_box.value())

        if self.raw_pitch_window.maxlen != pitch_window_size:
            self.raw_pitch_window = deque(
                self.raw_pitch_window, maxlen=pitch_window_size
            )

        if self.raw_resonance_window.maxlen != res_window_size:
            self.raw_resonance_window = deque(
                self.raw_resonance_window, maxlen=res_window_size
            )

        median_pitch = self._median_filtered_value(
            self.raw_pitch_window,
            result.pitch_hz,
        )

        median_resonance = self._median_filtered_value(
            self.raw_resonance_window,
            result.primary_resonance_hz,
        )

        pitch_alpha = float(self.pitch_smoothing_box.value())
        resonance_alpha = float(self.resonance_smoothing_box.value())

        self.smoothed_pitch = self._smooth_value(
            self.smoothed_pitch,
            median_pitch,
            pitch_alpha,
        )

        self.smoothed_resonance = self._smooth_value(
            self.smoothed_resonance,
            median_resonance,
            resonance_alpha,
        )

        if self.smoothed_pitch is None or self.smoothed_resonance is None:
            self.current_point.clear()
            return

        self.pitch_history.append(self.smoothed_pitch)
        self.resonance_history.append(self.smoothed_resonance)

        x = np.asarray(self.pitch_history, dtype=float)
        y = np.asarray(self.resonance_history, dtype=float)

        self.trail_curve.setData(x, y)
        self.current_point.setData([x[-1]], [y[-1]])

        if self.smoothed_pitch is None:
            self.pitch_label.setText("—")
        else:
            self.pitch_label.setText(f"{self.smoothed_pitch:.1f} Hz")

        if self.smoothed_resonance is None:
            self.res1_label.setText("—")
        else:
            self.res1_label.setText(f"{self.smoothed_resonance:.1f} Hz")

        if result.secondary_resonance_hz is None:
            self.res2_label.setText("—")
        else:
            self.res2_label.setText(f"{result.secondary_resonance_hz:.1f} Hz")

    def _read_config_from_controls(self) -> AnalysisConfig:
        pitch_min = float(self.pitch_min_box.value())
        pitch_max = float(self.pitch_max_box.value())
        if pitch_max <= pitch_min:
            pitch_max = pitch_min + 1.0

        return AnalysisConfig(
            samplerate=int(self.fs_box.value()),
            frame_size=int(self.frame_box.value()),
            lpc_order=int(self.lpc_box.value()),
            rms_threshold=float(self.threshold_box.value()),
            pitch_min_hz=pitch_min,
            pitch_max_hz=pitch_max,
        )

    def apply_settings(self) -> None:
        was_running = self.audio.is_running
        if was_running:
            self.stop_audio()

        self.config = self._read_config_from_controls()
        self.analyzer.update_config(self.config)
        self.audio.samplerate = self.config.samplerate
        self.audio.blocksize = self.config.frame_size

        self.pitch_history.clear()
        self.resonance_history.clear()
        self.trail_curve.clear()
        self.current_point.clear()

        self.raw_pitch_window = deque(maxlen=self.pitch_median_window_box.value())
        self.raw_resonance_window = deque(
            maxlen=self.resonance_median_window_box.value()
        )

        self.smoothed_pitch = None
        self.smoothed_resonance = None

        if was_running:
            self.start_audio()

    def start_audio(self) -> None:
        device_index = self._get_selected_device_index()
        self.audio.restart(
            device_index=device_index,
            samplerate=self.config.samplerate,
            blocksize=self.config.frame_size,
        )

        # If audio.py fell back to a different sample rate, keep UI/config in sync
        if self.audio.samplerate != self.config.samplerate:
            self.config = replace(self.config, samplerate=self.audio.samplerate)
            self.fs_box.blockSignals(True)
            self.fs_box.setValue(self.audio.samplerate)
            self.fs_box.blockSignals(False)
            self.analyzer.update_config(self.config)

    def stop_audio(self) -> None:
        self.audio.stop()

    def closeEvent(self, event) -> None:
        self.audio.close()
        super().closeEvent(event)


def run() -> int:
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication([])

    window = MainWindow()
    window.show()

    if owns_app:
        return app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
