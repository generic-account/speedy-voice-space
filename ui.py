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
    def __init__(self, config: Optional[AnalysisConfig] = None) -> None:
        super().__init__()

        self.config = config or AnalysisConfig()
        self.analyzer = RealtimeAnalyzer(self.config)
        self.audio = AudioInputManager(
            samplerate=self.config.samplerate,
            blocksize=self.config.block_size,
            channels=1,
        )

        self.processor = VoiceProcessor()

        self.pitch_history: Deque[float] = deque(maxlen=120)
        self.resonance_history: Deque[float] = deque(maxlen=120)

        self.formant_plot_history_len = 240
        self.f2_history: Deque[float] = deque(maxlen=self.formant_plot_history_len)
        self.f3_history: Deque[float] = deque(maxlen=self.formant_plot_history_len)

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
        self.resize(1500, 900)

        root_layout = QtWidgets.QHBoxLayout(self)

        #
        # Left: main plot
        #
        self.graphics = pg.GraphicsLayoutWidget()
        self.main_plot = self.graphics.addPlot(title="Real-time Pitch vs Resonance")
        self.main_plot.setLabel("bottom", "Pitch (Hz)")
        self.main_plot.setLabel("left", "Resonance (0–1)")
        self.main_plot.setXRange(60, 500)
        self.main_plot.setYRange(0.0, 1.0)
        self.main_plot.showGrid(x=True, y=True, alpha=0.25)

        self.trail_curve = self.main_plot.plot(
            pen=None,
            symbol="o",
            symbolSize=6,
        )
        self.current_point = self.main_plot.plot(
            pen=None,
            symbol="o",
            symbolSize=12,
        )

        root_layout.addWidget(self.graphics, stretch=4)

        #
        # Right: mini plots + settings/readouts
        #
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(1)

        #
        # Top-right compact formant plots
        #
        formant_graphs_box = QtWidgets.QWidget()
        formant_graphs_layout = QtWidgets.QVBoxLayout(formant_graphs_box)
        formant_graphs_layout.setContentsMargins(0, 0, 0, 0)
        formant_graphs_layout.setSpacing(1)

        self.f2_title = QtWidgets.QLabel("F2 / time")
        self.f2_title.setStyleSheet("font-size: 11px; padding: 0px; margin: 0px;")
        self.f2_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.f2_title.setContentsMargins(0, 0, 0, 0)

        self.f2_plot = pg.PlotWidget()
        self.f2_plot.setLabel("left", "Hz")
        self.f2_plot.setYRange(0, 5000)
        self.f2_plot.showGrid(x=True, y=True, alpha=0.2)
        self.f2_plot.setMouseEnabled(x=False, y=False)
        self.f2_plot.setMinimumHeight(85)
        self.f2_plot.setMaximumHeight(95)
        self.f2_plot.hideAxis("bottom")
        self.f2_plot.getPlotItem().getViewBox().setDefaultPadding(0.0)
        self.f2_plot.getPlotItem().layout.setContentsMargins(0, 0, 0, 0)
        self.f2_curve = self.f2_plot.plot(pen=pg.mkPen(width=2))

        self.f3_title = QtWidgets.QLabel("F3 / time")
        self.f3_title.setStyleSheet("font-size: 11px; padding: 0px; margin: 0px;")
        self.f3_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.f3_title.setContentsMargins(0, 0, 0, 0)

        self.f3_plot = pg.PlotWidget()
        self.f3_plot.setLabel("left", "Hz")
        self.f3_plot.setYRange(0, 7000)
        self.f3_plot.showGrid(x=True, y=True, alpha=0.2)
        self.f3_plot.setMouseEnabled(x=False, y=False)
        self.f3_plot.setMinimumHeight(85)
        self.f3_plot.setMaximumHeight(95)
        self.f3_plot.hideAxis("bottom")
        self.f3_plot.getPlotItem().getViewBox().setDefaultPadding(0.0)
        self.f3_plot.getPlotItem().layout.setContentsMargins(0, 0, 0, 0)
        self.f3_curve = self.f3_plot.plot(pen=pg.mkPen(width=2))

        formant_graphs_layout.addWidget(self.f2_title)
        formant_graphs_layout.addWidget(self.f2_plot)
        formant_graphs_layout.addWidget(self.f3_title)
        formant_graphs_layout.addWidget(self.f3_plot)

        right_layout.addWidget(formant_graphs_box, stretch=0)

        #
        # Bottom-right: controls and readouts
        #
        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(12)

        left_col = QtWidgets.QWidget()
        left_form = QtWidgets.QFormLayout(left_col)
        left_form.setContentsMargins(0, 0, 0, 0)
        left_form.setSpacing(4)

        right_col = QtWidgets.QWidget()
        right_grid = QtWidgets.QGridLayout(right_col)
        right_grid.setContentsMargins(0, 0, 0, 0)
        right_grid.setHorizontalSpacing(8)
        right_grid.setVerticalSpacing(4)

        controls_layout.addWidget(left_col, stretch=1)
        controls_layout.addWidget(right_col, stretch=1)

        #
        # Controls
        #
        self.device_dropdown = QtWidgets.QComboBox()

        self.fs_box = QtWidgets.QSpinBox()
        self.fs_box.setRange(8000, 48000)
        self.fs_box.setValue(self.config.samplerate)

        self.block_size_box = QtWidgets.QSpinBox()
        self.block_size_box.setRange(128, 4096)
        self.block_size_box.setSingleStep(128)
        self.block_size_box.setValue(self.config.block_size)

        self.buffer_duration_box = QtWidgets.QDoubleSpinBox()
        self.buffer_duration_box.setRange(0.02, 0.30)
        self.buffer_duration_box.setDecimals(3)
        self.buffer_duration_box.setSingleStep(0.005)
        self.buffer_duration_box.setValue(self.config.buffer_duration_s)

        self.threshold_box = QtWidgets.QDoubleSpinBox()
        self.threshold_box.setRange(0.0, 0.20)
        self.threshold_box.setDecimals(6)
        self.threshold_box.setSingleStep(0.0001)
        self.threshold_box.setValue(self.config.rms_threshold)

        self.pitch_floor_box = QtWidgets.QDoubleSpinBox()
        self.pitch_floor_box.setRange(40.0, 300.0)
        self.pitch_floor_box.setDecimals(1)
        self.pitch_floor_box.setValue(self.config.pitch_floor_hz)

        self.pitch_ceiling_box = QtWidgets.QDoubleSpinBox()
        self.pitch_ceiling_box.setRange(100.0, 800.0)
        self.pitch_ceiling_box.setDecimals(1)
        self.pitch_ceiling_box.setValue(self.config.pitch_ceiling_hz)

        self.pitch_silence_box = QtWidgets.QDoubleSpinBox()
        self.pitch_silence_box.setRange(0.0, 1.0)
        self.pitch_silence_box.setDecimals(2)
        self.pitch_silence_box.setSingleStep(0.01)
        self.pitch_silence_box.setValue(self.config.pitch_silence_threshold)

        self.pitch_voicing_box = QtWidgets.QDoubleSpinBox()
        self.pitch_voicing_box.setRange(0.0, 1.0)
        self.pitch_voicing_box.setDecimals(2)
        self.pitch_voicing_box.setSingleStep(0.01)
        self.pitch_voicing_box.setValue(self.config.pitch_voicing_threshold)

        self.pitch_accurate_box = QtWidgets.QCheckBox()
        self.pitch_accurate_box.setChecked(self.config.pitch_very_accurate)

        self.formant_time_step_box = QtWidgets.QDoubleSpinBox()
        self.formant_time_step_box.setRange(0.001, 0.05)
        self.formant_time_step_box.setDecimals(3)
        self.formant_time_step_box.setSingleStep(0.001)
        self.formant_time_step_box.setValue(self.config.formant_time_step)

        self.max_formants_box = QtWidgets.QDoubleSpinBox()
        self.max_formants_box.setRange(3.0, 7.0)
        self.max_formants_box.setDecimals(1)
        self.max_formants_box.setSingleStep(0.5)
        self.max_formants_box.setValue(self.config.max_number_of_formants)

        self.maximum_formant_box = QtWidgets.QSpinBox()
        self.maximum_formant_box.setRange(3000, 8000)
        self.maximum_formant_box.setValue(int(self.config.maximum_formant_hz))

        self.window_length_box = QtWidgets.QDoubleSpinBox()
        self.window_length_box.setRange(0.010, 0.080)
        self.window_length_box.setDecimals(3)
        self.window_length_box.setSingleStep(0.001)
        self.window_length_box.setValue(self.config.window_length_s)

        self.pre_emphasis_box = QtWidgets.QSpinBox()
        self.pre_emphasis_box.setRange(0, 1000)
        self.pre_emphasis_box.setValue(int(self.config.pre_emphasis_from_hz))

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
        self.f2_low_box.setRange(200, 10000)
        self.f2_low_box.setValue(600)

        self.f2_high_box = QtWidgets.QSpinBox()
        self.f2_high_box.setRange(200, 10000)
        self.f2_high_box.setValue(3000)

        self.f3_low_box = QtWidgets.QSpinBox()
        self.f3_low_box.setRange(500, 12000)
        self.f3_low_box.setValue(1500)

        self.f3_high_box = QtWidgets.QSpinBox()
        self.f3_high_box.setRange(500, 12000)
        self.f3_high_box.setValue(4500)

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

        #
        # Readout labels
        #
        self.pitch_label = QtWidgets.QLabel("—")
        self.resonance_label = QtWidgets.QLabel("—")
        self.confidence_label = QtWidgets.QLabel("—")
        self.rms_label = QtWidgets.QLabel("—")
        self.voice_label = QtWidgets.QLabel("—")
        self.frames_label = QtWidgets.QLabel("0")
        self.raw_f2_label = QtWidgets.QLabel("—")
        self.raw_f3_label = QtWidgets.QLabel("—")
        self.formants_label = QtWidgets.QLabel("—")

        readout_value_labels = [
            self.pitch_label,
            self.resonance_label,
            self.confidence_label,
            self.rms_label,
            self.voice_label,
            self.frames_label,
            self.raw_f2_label,
            self.raw_f3_label,
            self.formants_label,
        ]

        for label in readout_value_labels:
            label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            label.setMinimumWidth(140)

        self.formants_label.setWordWrap(True)
        self.formants_label.setMinimumWidth(180)

        #
        # Left column: controls
        #
        left_form.addRow("Input Device", self.device_dropdown)
        left_form.addRow("Sample Rate", self.fs_box)
        left_form.addRow("Block Size", self.block_size_box)
        left_form.addRow("Buffer Duration (s)", self.buffer_duration_box)
        left_form.addRow("RMS Threshold", self.threshold_box)

        left_form.addRow("Pitch Floor (Hz)", self.pitch_floor_box)
        left_form.addRow("Pitch Ceiling (Hz)", self.pitch_ceiling_box)
        left_form.addRow("Pitch Silence Thresh", self.pitch_silence_box)
        left_form.addRow("Pitch Voicing Thresh", self.pitch_voicing_box)
        left_form.addRow("Pitch Very Accurate", self.pitch_accurate_box)

        left_form.addRow("Formant Time Step (s)", self.formant_time_step_box)
        left_form.addRow("Max # Formants", self.max_formants_box)
        left_form.addRow("Maximum Formant (Hz)", self.maximum_formant_box)
        left_form.addRow("Window Length (s)", self.window_length_box)
        left_form.addRow("Pre-emphasis From (Hz)", self.pre_emphasis_box)

        left_form.addRow("Pitch Median Window", self.pitch_median_window_box)
        left_form.addRow("Resonance Median Window", self.resonance_median_window_box)
        left_form.addRow("Pitch Smoothing", self.pitch_smoothing_box)
        left_form.addRow("Resonance Smoothing", self.resonance_smoothing_box)

        left_form.addRow("F2 Low (Hz)", self.f2_low_box)
        left_form.addRow("F2 High (Hz)", self.f2_high_box)
        left_form.addRow("F3 Low (Hz)", self.f3_low_box)
        left_form.addRow("F3 High (Hz)", self.f3_high_box)
        left_form.addRow("F2 Weight", self.f2_weight_box)
        left_form.addRow("F3 Weight", self.f3_weight_box)

        button_row = QtWidgets.QWidget()
        button_layout = QtWidgets.QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(6)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.apply_button)
        left_form.addRow(button_row)

        #
        # Right column: compact readouts
        #
        readout_names = [
            "Pitch",
            "Resonance",
            "Confidence",
            "RMS",
            "Voiced",
            "Frames Seen",
            "F2",
            "F3",
            "All Formants",
        ]

        readout_values = [
            self.pitch_label,
            self.resonance_label,
            self.confidence_label,
            self.rms_label,
            self.voice_label,
            self.frames_label,
            self.raw_f2_label,
            self.raw_f3_label,
            self.formants_label,
        ]

        for row, (name, value_widget) in enumerate(zip(readout_names, readout_values)):
            name_label = QtWidgets.QLabel(name)
            name_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop
            )
            name_label.setMinimumWidth(95)

            right_grid.addWidget(name_label, row, 0)
            right_grid.addWidget(value_widget, row, 1)

        right_grid.setColumnStretch(0, 0)
        right_grid.setColumnStretch(1, 1)

        right_layout.addWidget(controls, stretch=1)
        root_layout.addWidget(right_panel, stretch=2)

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
        pitch_floor = float(self.pitch_floor_box.value())
        pitch_ceiling = float(self.pitch_ceiling_box.value())
        if pitch_ceiling <= pitch_floor:
            pitch_ceiling = pitch_floor + 1.0

        return AnalysisConfig(
            samplerate=int(self.fs_box.value()),
            block_size=int(self.block_size_box.value()),
            buffer_duration_s=float(self.buffer_duration_box.value()),
            rms_threshold=float(self.threshold_box.value()),
            pitch_time_step=0.01,
            pitch_floor_hz=pitch_floor,
            pitch_ceiling_hz=pitch_ceiling,
            pitch_silence_threshold=float(self.pitch_silence_box.value()),
            pitch_voicing_threshold=float(self.pitch_voicing_box.value()),
            pitch_very_accurate=bool(self.pitch_accurate_box.isChecked()),
            formant_time_step=float(self.formant_time_step_box.value()),
            max_number_of_formants=float(self.max_formants_box.value()),
            maximum_formant_hz=float(self.maximum_formant_box.value()),
            window_length_s=float(self.window_length_box.value()),
            pre_emphasis_from_hz=float(self.pre_emphasis_box.value()),
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
        if (f2_weight + f3_weight) <= 0:
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
        self.processor.update_settings(self._read_processing_settings_from_controls())

    def apply_settings(self) -> None:
        was_running = self.audio.is_running
        if was_running:
            self.stop_audio()

        self.config = self._read_config_from_controls()
        self.analyzer.update_config(self.config)
        self.audio.samplerate = self.config.samplerate
        self.audio.blocksize = self.config.block_size

        self.apply_processing_settings()
        self.processor.reset()

        self.pitch_history.clear()
        self.resonance_history.clear()
        self.f2_history.clear()
        self.f3_history.clear()

        self.trail_curve.clear()
        self.current_point.clear()
        self.f2_curve.clear()
        self.f3_curve.clear()

        if was_running:
            self.start_audio()

    def _update_formant_plots(self, state) -> None:
        f2 = state.raw_f2_hz
        f3 = state.raw_f3_hz

        self.f2_history.append(np.nan if f2 is None else float(f2))
        self.f3_history.append(np.nan if f3 is None else float(f3))

        x1 = np.arange(len(self.f2_history), dtype=float)
        y1 = np.asarray(self.f2_history, dtype=float)

        x2 = np.arange(len(self.f3_history), dtype=float)
        y2 = np.asarray(self.f3_history, dtype=float)

        self.f2_curve.setData(x1, y1)
        self.f3_curve.setData(x2, y2)

    def _update_ui(self, result: AnalysisResult) -> None:
        state = self.processor.process(result)

        self.frames_label.setText(str(getattr(self.analyzer, "frames_seen", 0)))
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

        if state.raw_f2_hz is None:
            self.raw_f2_label.setText("missing")
        else:
            self.raw_f2_label.setText(f"{state.raw_f2_hz:.1f} Hz")

        if state.raw_f3_hz is None:
            self.raw_f3_label.setText("missing")
        else:
            self.raw_f3_label.setText(f"{state.raw_f3_hz:.1f} Hz")

        if state.formants_hz:
            self.formants_label.setText(", ".join(f"{f:.0f}" for f in state.formants_hz[:5]))
        else:
            self.formants_label.setText("missing")

        self._update_formant_plots(state)

        if state.filtered_pitch_hz is None or state.filtered_resonance is None:
            self.current_point.clear()
            return

        self.pitch_history.append(state.filtered_pitch_hz)
        self.resonance_history.append(state.filtered_resonance)

        x = np.asarray(self.pitch_history, dtype=float)
        y = np.asarray(self.resonance_history, dtype=float)

        self.trail_curve.setData(x, y)
        self.current_point.setData([x[-1]], [y[-1]])

    def start_audio(self) -> None:
        device_index = self._get_selected_device_index()
        self.audio.restart(
            device_index=device_index,
            samplerate=self.config.samplerate,
            blocksize=self.config.block_size,
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