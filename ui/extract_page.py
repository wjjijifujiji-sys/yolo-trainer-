"""Video frame extraction page."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QProgressBar, QSlider, QGroupBox,
    QRadioButton, QButtonGroup,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QFont

from ui.components import styled_button
from utils.i18n import t

# Import extractor class (inline to avoid separate module)
import time
from PyQt6.QtCore import QThread


class FrameExtractor(QThread):
    """Background thread for extracting frames from video."""

    progress = pyqtSignal(int, int, float)
    frame_preview = pyqtSignal(np.ndarray)
    log = pyqtSignal(str)
    done = pyqtSignal(int, str)

    MODE_FPS = 0
    MODE_INTERVAL = 1
    MODE_SCENE = 2
    MODE_MOTION = 3

    def __init__(self, video_path, output_dir, mode, target_fps=1.0, interval_sec=2.0,
                 scene_threshold=0.4, motion_threshold=25.0, output_format="jpg", quality=95):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.mode = mode
        self.target_fps = target_fps
        self.interval_sec = interval_sec
        self.scene_threshold = scene_threshold
        self.motion_threshold = motion_threshold
        self.output_format = output_format
        self.quality = quality
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        import cv2
        import numpy as np
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.log.emit(f"Cannot open video: {self.video_path}")
                self.done.emit(0, "")
                return

            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.log.emit(f"Video: {width}x{height}, {video_fps:.1f}fps, {total_frames} frames")

            self.output_dir = os.path.abspath(self.output_dir)
            os.makedirs(self.output_dir, exist_ok=True)
            self.log.emit(f"Output: {self.output_dir}")

            start_time = time.time()
            count = 0
            frame_idx = 0
            prev_frame = None

            if self.mode == self.MODE_FPS:
                skip = max(1, int(video_fps / self.target_fps))
                self.log.emit(f"Mode: {self.target_fps} fps, skip {skip} frames")
            elif self.mode == self.MODE_INTERVAL:
                skip = max(1, int(video_fps * self.interval_sec))
                self.log.emit(f"Mode: every {self.interval_sec}s, skip {skip} frames")
            elif self.mode == self.MODE_SCENE:
                skip = 1
                self.log.emit(f"Mode: scene change (threshold={self.scene_threshold})")
            elif self.mode == self.MODE_MOTION:
                skip = 1
                self.log.emit(f"Mode: motion detection (threshold={self.motion_threshold})")

            while not self._stop:
                ret, frame = cap.read()
                if not ret:
                    break

                should_save = False

                if self.mode in (self.MODE_FPS, self.MODE_INTERVAL):
                    if frame_idx % skip == 0:
                        should_save = True
                elif self.mode == self.MODE_SCENE:
                    if prev_frame is not None:
                        hist1 = cv2.calcHist([prev_frame], [0], None, [256], [0, 256])
                        hist2 = cv2.calcHist([frame], [0], None, [256], [0, 256])
                        cv2.normalize(hist1, hist1)
                        cv2.normalize(hist2, hist2)
                        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
                        if diff > self.scene_threshold:
                            should_save = True
                    else:
                        should_save = True
                elif self.mode == self.MODE_MOTION:
                    if prev_frame is not None:
                        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        diff = cv2.absdiff(gray1, gray2)
                        _, thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
                        if np.count_nonzero(thresh) / thresh.size > 0.01:
                            should_save = True
                    else:
                        should_save = True

                if should_save:
                    count += 1
                    filename = f"frame_{count:06d}.{self.output_format}"
                    filepath = os.path.join(self.output_dir, filename)
                    try:
                        if self.output_format == "jpg":
                            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                        else:
                            ok, buf = cv2.imencode('.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                        if ok:
                            buf.tofile(filepath)
                    except Exception as e:
                        self.log.emit(f"  [ERROR] frame {count}: {e}")

                prev_frame = frame.copy()
                if frame_idx % 5 == 0:
                    self.frame_preview.emit(frame)
                frame_idx += 1
                if frame_idx % 10 == 0:
                    self.progress.emit(frame_idx, total_frames, time.time() - start_time)

            cap.release()
            elapsed = time.time() - start_time
            actual = len([f for f in os.listdir(self.output_dir) if f.endswith(self.output_format)]) if os.path.exists(self.output_dir) else 0
            self.log.emit(f"\nDone! {actual} frames in {elapsed:.1f}s")
            self.log.emit(f"Output: {self.output_dir}")
            self.done.emit(actual, self.output_dir)

        except Exception as e:
            self.log.emit(f"Error: {str(e)}")
            self.done.emit(0, str(e))


class VideoInfo:
    @staticmethod
    def get_info(video_path):
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Cannot open video"}
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS) or 0,
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        }
        info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
        codec = info["codec"]
        info["codec_str"] = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
        cap.release()
        return info

    @staticmethod
    def get_frame_at(video_path, frame_idx):
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None


class ExtractPage(QWidget):
    """Video frame extraction page."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._video_path = None
        self._video_info = None
        self._output_dir = None
        self._extractor = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ── Video info bar ──
        info_bar = QHBoxLayout()
        info_bar.setSpacing(12)

        self.file_label = QLabel(t("no_video"))
        self.file_label.setObjectName("hint")
        info_bar.addWidget(self.file_label, 1)

        self.browse_btn = styled_button(t("select_video"), "#f97316", 13)
        self.browse_btn.clicked.connect(self._browse_video)
        info_bar.addWidget(self.browse_btn)

        self.video_info_label = QLabel("")
        self.video_info_label.setObjectName("hint")
        info_bar.addWidget(self.video_info_label)
        layout.addLayout(info_bar)

        # ── Preview ──
        self.preview_label = QLabel(t("drag_video_hint"))
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(280)
        self.preview_label.setStyleSheet("""
            QLabel {
                background: #0c1017;
                border: 2px dashed #374151;
                border-radius: 12px;
                color: #6b7280;
                font-size: 16px;
            }
        """)
        self.preview_label.setAcceptDrops(True)
        layout.addWidget(self.preview_label, 1)

        # ── Seek slider ──
        seek_row = QHBoxLayout()
        seek_row.setSpacing(8)
        self.seek_label = QLabel("0:00 / 0:00")
        self.seek_label.setObjectName("hint")
        self.seek_label.setMinimumWidth(100)
        seek_row.addWidget(self.seek_label)

        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 100)
        self.seek_slider.sliderMoved.connect(self._on_seek)
        seek_row.addWidget(self.seek_slider, 1)

        self.time_label = QLabel("0:00")
        self.time_label.setObjectName("hint")
        self.time_label.setMinimumWidth(60)
        seek_row.addWidget(self.time_label)
        layout.addLayout(seek_row)

        # ── Settings: mode + params + action ──
        settings = QHBoxLayout()
        settings.setSpacing(12)

        # Mode selection
        mode_group = QGroupBox(t("extract_mode"))
        mode_layout = QVBoxLayout(mode_group)

        self.mode_group = QButtonGroup()
        self.mode_fps = QRadioButton(t("mode_fps"))
        self.mode_interval = QRadioButton(t("mode_interval"))
        self.mode_scene = QRadioButton(t("mode_scene"))
        self.mode_motion = QRadioButton(t("mode_motion"))
        self.mode_fps.setChecked(True)

        self.mode_group.addButton(self.mode_fps, 0)
        self.mode_group.addButton(self.mode_interval, 1)
        self.mode_group.addButton(self.mode_scene, 2)
        self.mode_group.addButton(self.mode_motion, 3)

        mode_layout.addWidget(self.mode_fps)
        mode_layout.addWidget(self.mode_interval)
        mode_layout.addWidget(self.mode_scene)
        mode_layout.addWidget(self.mode_motion)
        self.mode_group.idClicked.connect(self._on_mode_changed)
        settings.addWidget(mode_group)

        # Parameters
        param_group = QGroupBox(t("params"))
        param_layout = QVBoxLayout(param_group)

        # FPS param
        fps_row = QHBoxLayout()
        fps_row.setSpacing(8)
        self.fps_label = QLabel(t("fps_per_sec"))
        self.fps_label.setMinimumWidth(70)
        fps_row.addWidget(self.fps_label)
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.1, 60.0)
        self.fps_spin.setValue(1.0)
        self.fps_spin.setSingleStep(1.0)
        self.fps_spin.setDecimals(1)
        self.fps_spin.setSuffix(t("unit_frames"))
        self.fps_spin.setMaximumWidth(160)
        fps_row.addWidget(self.fps_spin)
        fps_row.addStretch()
        param_layout.addLayout(fps_row)

        # Interval param
        self.interval_widget = QWidget()
        iw = QHBoxLayout(self.interval_widget)
        iw.setContentsMargins(0, 0, 0, 0)
        iw.setSpacing(8)
        self.interval_label = QLabel(t("every_n_sec"))
        self.interval_label.setMinimumWidth(70)
        iw.addWidget(self.interval_label)
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.5, 3600.0)
        self.interval_spin.setValue(2.0)
        self.interval_spin.setSingleStep(1.0)
        self.interval_spin.setDecimals(1)
        self.interval_spin.setSuffix(t("unit_sec_per_frame"))
        self.interval_spin.setMaximumWidth(160)
        iw.addWidget(self.interval_spin)
        iw.addStretch()
        param_layout.addWidget(self.interval_widget)

        # Scene threshold
        self.scene_widget = QWidget()
        sw = QHBoxLayout(self.scene_widget)
        sw.setContentsMargins(0, 0, 0, 0)
        sw.setSpacing(8)
        self.scene_label = QLabel(t("sensitivity"))
        self.scene_label.setMinimumWidth(70)
        sw.addWidget(self.scene_label)
        self.scene_spin = QDoubleSpinBox()
        self.scene_spin.setRange(0.1, 1.0)
        self.scene_spin.setValue(0.4)
        self.scene_spin.setSingleStep(0.05)
        self.scene_spin.setDecimals(2)
        self.scene_spin.setMaximumWidth(160)
        sw.addWidget(self.scene_spin)
        self.scene_hint = QLabel(t("higher_less"))
        self.scene_hint.setObjectName("hint")
        sw.addWidget(self.scene_hint)
        sw.addStretch()
        param_layout.addWidget(self.scene_widget)

        # Motion threshold
        self.motion_widget = QWidget()
        mw = QHBoxLayout(self.motion_widget)
        mw.setContentsMargins(0, 0, 0, 0)
        mw.setSpacing(8)
        self.motion_label = QLabel(t("sensitivity"))
        self.motion_label.setMinimumWidth(70)
        mw.addWidget(self.motion_label)
        self.motion_spin = QDoubleSpinBox()
        self.motion_spin.setRange(1.0, 100.0)
        self.motion_spin.setValue(25.0)
        self.motion_spin.setSingleStep(5.0)
        self.motion_spin.setDecimals(0)
        self.motion_spin.setMaximumWidth(160)
        mw.addWidget(self.motion_spin)
        self.motion_hint = QLabel(t("higher_less"))
        self.motion_hint.setObjectName("hint")
        mw.addWidget(self.motion_hint)
        mw.addStretch()
        param_layout.addWidget(self.motion_widget)

        # Output format
        fmt_row = QHBoxLayout()
        fmt_row.setSpacing(8)
        fmt_label = QLabel(t("format"))
        fmt_label.setMinimumWidth(70)
        fmt_row.addWidget(fmt_label)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["jpg", "png"])
        self.format_combo.setCurrentText("jpg")
        self.format_combo.setMaximumWidth(80)
        fmt_row.addWidget(self.format_combo)

        qual_label = QLabel(t("quality"))
        qual_label.setMinimumWidth(40)
        fmt_row.addWidget(qual_label)
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(95)
        self.quality_spin.setSuffix("%")
        self.quality_spin.setMaximumWidth(80)
        fmt_row.addWidget(self.quality_spin)
        fmt_row.addStretch()
        param_layout.addLayout(fmt_row)

        settings.addWidget(param_group, 1)

        # Output & action
        action_group = QGroupBox(t("output"))
        action_layout = QVBoxLayout(action_group)

        out_row = QHBoxLayout()
        out_row.setSpacing(8)
        self.out_label = QLabel(t("no_output_dir"))
        self.out_label.setObjectName("hint")
        out_row.addWidget(self.out_label, 1)
        self.out_btn = styled_button(t("select_dir"), "#6b7280", 11)
        self.out_btn.setMaximumWidth(90)
        self.out_btn.clicked.connect(self._browse_output)
        out_row.addWidget(self.out_btn)
        action_layout.addLayout(out_row)

        action_layout.addStretch()

        self.extract_btn = styled_button(t("start_extract"), "#f97316", 15)
        self.extract_btn.setMinimumHeight(44)
        self.extract_btn.setEnabled(False)
        self.extract_btn.clicked.connect(self._start_extract)
        action_layout.addWidget(self.extract_btn)

        self.stop_btn = styled_button(t("stop"), "#ef4444", 15)
        self.stop_btn.setMinimumHeight(44)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_extract)
        action_layout.addWidget(self.stop_btn)

        settings.addWidget(action_group)

        layout.addLayout(settings)

        # ── Progress & Log ──
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel(t("ready"))
        self.status_label.setObjectName("hint")
        layout.addWidget(self.status_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        layout.addWidget(self.log_text)

        self._on_mode_changed(0)

    # ─── i18n ───

    def retranslate(self):
        self.browse_btn.setText(t("select_video"))
        self.extract_btn.setText(t("start_extract"))
        self.stop_btn.setText(t("stop"))
        self.out_btn.setText(t("select_dir"))
        if self.file_label.text() in ("未选择视频", "No video selected"):
            self.file_label.setText(t("no_video"))
        if self.out_label.text() in ("未选择输出目录", "No output dir"):
            self.out_label.setText(t("no_output_dir"))
        if self.status_label.text() in ("就绪", "Ready"):
            self.status_label.setText(t("ready"))
        if self.preview_label.text() in ("拖放视频文件到此处，或点击「选择视频」", "Drop video here or click Browse"):
            self.preview_label.setText(t("drag_video_hint"))

    def _on_mode_changed(self, mode_id):
        self.fps_label.setVisible(mode_id == 0)
        self.fps_spin.setVisible(mode_id == 0)
        self.interval_widget.setVisible(mode_id == 1)
        self.scene_widget.setVisible(mode_id == 2)
        self.motion_widget.setVisible(mode_id == 3)

    def _browse_video(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, t("select_video"), "",
            "Video (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All (*)"
        )
        if filepath:
            self._load_video(filepath)

    def _load_video(self, filepath):
        self._video_path = filepath
        info = VideoInfo.get_info(filepath)
        if "error" in info:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", info["error"])
            return

        self._video_info = info
        self.file_label.setText(Path(filepath).name)
        self.file_label.setStyleSheet("color: #22c55e; font-size: 12px;")

        dur = info["duration"]
        self.video_info_label.setText(
            f"{info['width']}x{info['height']} | {info['fps']:.1f}fps | "
            f"{int(dur//60)}:{int(dur%60):02d} | {info['total_frames']} frames | {info['codec_str']}"
        )

        self.seek_slider.setRange(0, info["total_frames"] - 1)
        self.seek_slider.setValue(0)
        self._update_time_label(0)

        frame = VideoInfo.get_frame_at(filepath, 0)
        if frame is not None:
            self._show_frame(frame)

        self.extract_btn.setEnabled(True)
        self.log(f"Loaded: {Path(filepath).name}")

    def _show_frame(self, frame):
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def _on_seek(self, frame_idx):
        if self._video_path is None:
            return
        self._update_time_label(frame_idx)
        frame = VideoInfo.get_frame_at(self._video_path, frame_idx)
        if frame is not None:
            self._show_frame(frame)

    def _update_time_label(self, frame_idx):
        if self._video_info is None:
            return
        fps = self._video_info["fps"] or 30.0
        total = self._video_info["total_frames"]
        cur = frame_idx / fps
        tot = total / fps
        self.seek_label.setText(f"{int(cur//60)}:{int(cur%60):02d} / {int(tot//60)}:{int(tot%60):02d}")
        self.time_label.setText(f"{int(cur//60)}:{int(cur%60):02d}")

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, t("select_dir"))
        if folder:
            self._output_dir = folder
            self.out_label.setText(folder)
            self.out_label.setStyleSheet("color: #22c55e; font-size: 12px;")

    def _start_extract(self):
        if not self._video_path:
            return
        if not self._output_dir:
            self._output_dir = str(Path(self._video_path).parent / "extracted_frames")
            os.makedirs(self._output_dir, exist_ok=True)
            self.out_label.setText(self._output_dir)
            self.out_label.setStyleSheet("color: #22c55e; font-size: 12px;")

        self.extract_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(t("extracting"))
        self.status_label.setStyleSheet("color: #f97316; font-size: 13px; font-weight: bold;")

        self._extractor = FrameExtractor(
            video_path=self._video_path,
            output_dir=self._output_dir,
            mode=self.mode_group.checkedId(),
            target_fps=self.fps_spin.value(),
            interval_sec=self.interval_spin.value(),
            scene_threshold=self.scene_spin.value(),
            motion_threshold=self.motion_spin.value(),
            output_format=self.format_combo.currentText(),
            quality=self.quality_spin.value(),
        )
        self._extractor.progress.connect(self._on_progress)
        self._extractor.frame_preview.connect(self._show_frame)
        self._extractor.log.connect(self.log)
        self._extractor.done.connect(self._on_done)
        self._extractor.start()

    def _stop_extract(self):
        if self._extractor:
            self._extractor.stop()

    def _on_progress(self, current, total, elapsed):
        if total > 0:
            self.progress_bar.setValue(int(current / total * 100))
            if elapsed > 0 and current > 0:
                rem = (total - current) / (current / elapsed)
                self.status_label.setText(f"{t('extracting')} {current}/{total} | {int(rem//60)}:{int(rem%60):02d}")

    def _on_done(self, count, output):
        self.extract_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if count > 0:
            self.progress_bar.setValue(100)
            self.status_label.setText(f"{t('extract_done')} {count}")
            self.status_label.setStyleSheet("color: #22c55e; font-size: 13px; font-weight: bold;")
        else:
            self.status_label.setText(t("extract_fail"))
            self.status_label.setStyleSheet("color: #ef4444; font-size: 13px; font-weight: bold;")

    def log(self, msg):
        self.log_text.append(msg)
        scroll = self.log_text.verticalScrollBar()
        scroll.setValue(scroll.maximum())

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if Path(url.toLocalFile()).suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}:
                    event.acceptProposedAction()
                    return
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}:
                self._load_video(str(path))
                return
