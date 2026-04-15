from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Deque, Optional

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from audio import AudioInputManager
from analysis import AnalysisConfig, AnalysisResult, RealtimeAnalyzer
from processing import ProcessingSettings, VoiceProcessor


class AnalysisEmitter(QtCore.QObject):
    result_ready = QtCore.pyqtSignal(object)


class MainWindow(QtWidgets.QWidget):
    """
    UI-only layer.

    Responsibilities:
    - create controls
    - read control values into configs/settings
    - start/stop audio
    - render processed display state

    All resonance calculation, missing-data handling, rolling median,
    and exponential smoothing live in processing.py.
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

        self.processor = VoiceProcessor()
        self.pitch_history: Deque[float] = deque(maxlen=120)
        self.resonance_history: Deque[float] = deque(maxlen=120)

        self.emitter = AnalysisEmitter()
        self.emitter.result_ready.connect(self._update_ui)

        self._build_ui()
        self._wire_callbacks()
        self._populate_devices()
        self.apply_processing_settings()

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
        self.plot.setLabel("left", "Resonance (0–1)")
        self.plot.setXRange(60, 500)
        self.plot.setYRange(0.0, 1.0)
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

        self.pitch_median_window_box = QtWidgets.QSpinBox()
        self.pitch_median_window_box.setRange(1, 15)
        self.pitch_median_window_box.setValue(5)

        self.resonance_median_window_box = QtWidgets.QSpinBox()
        self.resonance_median_window_box.setRange(1, 15)
        self.resonance_median_window_box.setValue(7)

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

        self.f2_low_box = QtWidgets.QSpinBox()
        self.f2_low_box.setRange(200, 4000)
        self.f2_low_box.setValue(800)

        self.f2_high_box = QtWidgets.QSpinBox()
        self.f2_high_box.setRange(200, 4000)
        self.f2_high_box.setValue(2500)

        self.f3_low_box = QtWidgets.QSpinBox()
        self.f3_low_box.setRange(500, 5000)
        self.f3_low_box.setValue(1500)

        self.f3_high_box = QtWidgets.QSpinBox()
        self.f3_high_box.setRange(500, 5000)
        self.f3_high_box.setValue(3500)

        self.f2_weight_box = QtWidgets.QDoubleSpinBox()
        self.f2_weight_box.setRange(0.0, 1.0)
        self.f2_weight_box.setDecimals(2)
        self.f2_weight_box.setSingleStep(0.05)
        self.f2_weight_box.setValue(0.60)

        self.f3_weight_box = QtWidgets.QDoubleSpinBox()
        self.f3_weight_box.setRange(0.0, 1.0)
        self.f3_weight_box.setDecimals(2)
        self.f3_weight_box.setSingleStep(0.05)
        self.f3_weight_box.setValue(0.40)

        self.start_button = QtWidgets.QPushButton("Start")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.apply_button = QtWidgets.QPushButton("Apply Settings")

        self.pitch_label = QtWidgets.QLabel("—")
        self.resonance_label = QtWidgets.QLabel("—")
        self.confidence_label = QtWidgets.QLabel("—")
        self.rms_label = QtWidgets.QLabel("—")
        self.voice_label = QtWidgets.QLabel("—")
        self.frames_label = QtWidgets.QLabel("0")

        self.raw_f2_label = QtWidgets.QLabel("—")
        self.raw_f3_label = QtWidgets.QLabel("—")
        self.norm_f2_label = QtWidgets.QLabel("—")
        self.norm_f3_label = QtWidgets.QLabel("—")
        self.raw_resonance_label = QtWidgets.QLabel("—")
        self.formants_label = QtWidgets.QLabel("—")

        form.addRow("Input Device", self.device_dropdown)
        form.addRow("Sample Rate", self.fs_box)
        form.addRow("Frame Size", self.frame_box)
        form.addRow("LPC Order", self.lpc_box)
        form.addRow("RMS Threshold", self.threshold_box)
        form.addRow("Pitch Min (Hz)", self.pitch_min_box)
        form.addRow("Pitch Max (Hzz)", self.pitch_max_box)

        form.addRow("Pitch Median Window", self.pitch_median_window_box)
        form.addRow("Resonance Median Window", self.resonance_median_window_box)
        form.addRow("Pitch Smoothing", self.pitch_smoothing_box)
        form.addRow("Resonance Smoothing", self.resonance_smoothing_box)

        form.addRow("F2 Low (Hz)", self.f2_low_box)
        form.addRow("F2 High (Hz)", self.f2_high_box)
        form.addRow("F3 Low (Hz)", self.f3_low_box)
        form.addRow("F3 High (Hz)", self.f3_high_box)
        form.addRow("F2 Weight", self.f2_weight_box)
        form.addRow("F3 Weight", self.f3_weight_box)

        form.addRow(self.start_button)
        form.addRow(self.stop_button)
        form.addRow(self.apply_button)

        form.addRow("Pitch", self.pitch_label)
        form.addRow("Resonance", self.resonance_label)
        form.addRow("Confidence", self.confidence_label)
        form.addRow("RMS", self.rms_label)
        form.addRow("Voiced", self.voice_label)
        form.addRow("Frames Seen", self.frames_label)

        form.addRow("Raw F2", self.raw_f2_label)
        form.addRow("Raw F3", self.raw_f3_label)
        form.addRow("Norm F2", self.norm_f2_label)
        form.addRow("Norm F3", self.norm_f3_label)
        form.addRow("Raw Resonance", self.raw_resonance_label)
        form.addRow("Formants", self.formants_label)

        layout.addWidget(controls, stretch=1)

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
        self.emitter.result_ready.emit(result)

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

    def _read_processing_settings_from_controls(self) -> ProcessingSettings:
        f2_low = float(self.f2_low_box.value())
        f2_high = float(self.f2_high_box.value())
        f3_low = float(self.f3_low_box.value())
        f3_high = float(self.f3_high_box.value())

        if f2_high <= f2_low:
            f2_high = f2_low + 1.0
        if f3_high <= f3_low:
            f3_high = f3_low + 1.0

        f2_weight = float(self.f2_weight_box.value())
        f3_weight = float(self.f3_weight_box.value())
        weight_sum = f2_weight + f3_weight
        if weight_sum <= 0:
            f2_weight, f3_weight = 0.6, 0.4

        return ProcessingSettings(
            pitch_median_window=int(self.pitch_median_window_box.value()),
            resonance_median_window=int(self.resonance_median_window_box.value()),
            pitch_alpha=float(self.pitch_smoothing_box.value()),
            resonance_alpha=float(self.resonance_smoothing_box.value()),
            f2_low_hz=f2_low,
            f2_high_hz=f2_high,
            f3_low_hz=f3_low,
            f3_high_hz=f3_high,
            f2_weight=f2_weight,
            f3_weight=f3_weight,
        )

    def apply_processing_settings(self) -> None:
        self.processor.update_settings(
            self._read_processing_settings_from_controls()
        )

    def apply_settings(self) -> None:
        was_running = self.audio.is_running
        if was_running:
            self.stop_audio()

        self.config = self._read_config_from_controls()
        self.analyzer.update_config(self.config)
        self.audio.samplerate = self.config.samplerate
        self.audio.blocksize = self.config.frame_size

        self.apply_processing_settings()
        self.processor.reset()

        self.pitch_history.clear()
        self.resonance_history.clear()
        self.trail_curve.clear()
        self.current_point.clear()

        if was_running:
            self.start_audio()

    def _update_ui(self, result: AnalysisResult) -> None:
        state = self.processor.process(result)

        self.frames_label.setText(str(self.analyzer.frames_seen))
        self.rms_label.setText(f"{state.rms:.4f}")
        self.voice_label.setText("Yes" if state.voiced else "No")

        if state.filtered_pitch_hz is None:
            self.pitch_label.setText("—")
        else:
            self.pitch_label.setText(f"{state.filtered_pitch_hz:.1f} Hz")

        if state.filtered_resonance is None:
            self.resonance_label.setText("—")
        else:
            self.resonance_label.setText(f"{state.filtered_resonance:.3f}")

        self.confidence_label.setText(f"{state.resonance_confidence:.2f}")

        if state.filtered_pitch_hz is None or state.filtered_resonance is None:
            self.current_point.clear()
            return

        self.pitch_history.append(state.filtered_pitch_hz)
        self.resonance_history.append(state.filtered_resonance)

        x = np.asarray(self.pitch_history, dtype=float)
        y = np.asarray(self.resonance_history, dtype=float)

        self.trail_curve.setData(x, y)
        self.current_point.setData([x[-1]], [y[-1]])

        if state.raw_f2_hz is None:
            self.raw_f2_label.setText("missing")
        else:
            self.raw_f2_label.setText(f"{state.raw_f2_hz:.1f} Hz")

        if state.raw_f3_hz is None:
            self.raw_f3_label.setText("missing")
        else:
            self.raw_f3_label.setText(f"{state.raw_f3_hz:.1f} Hz")

        if state.norm_f2 is None:
            self.norm_f2_label.setText("missing")
        else:
            self.norm_f2_label.setText(f"{state.norm_f2:.3f}")

        if state.norm_f3 is None:
            self.norm_f3_label.setText("missing")
        else:
            self.norm_f3_label.setText(f"{state.norm_f3:.3f}")

        if state.raw_resonance is None:
            self.raw_resonance_label.setText("missing")
        else:
            self.raw_resonance_label.setText(f"{state.raw_resonance:.3f}")

        if state.formants_hz:
            self.formants_label.setText(
                ", ".join(f"{f:.0f}" for f in state.formants_hz[:4])
            )
        else:
            self.formants_label.setText("missing")

    def start_audio(self) -> None:
        device_index = self._get_selected_device_index()
        self.audio.restart(
            device_index=device_index,
            samplerate=self.config.samplerate,
            blocksize=self.config.frame_size,
        )

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