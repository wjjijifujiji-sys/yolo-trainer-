"""Detection page - detect objects in images, videos, and camera."""

from __future__ import annotations

import os
import csv
import time
from pathlib import Path
from datetime import datetime

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFileDialog, QComboBox, QDoubleSpinBox,
    QTextEdit, QProgressBar, QGroupBox, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont

from ui.components import styled_button
from utils.i18n import t


class DetectionWorker(QThread):
    """Background thread for running detection."""
    progress = pyqtSignal(int, int)
    frame_result = pyqtSignal(np.ndarray, list)  # frame, detections
    log = pyqtSignal(str)
    done = pyqtSignal(int)  # total detections

    def __init__(self, model, source, conf, iou, output_dir, source_type="image"):
        super().__init__()
        self.model = model
        self.source = source
        self.conf = conf
        self.iou = iou
        self.output_dir = output_dir
        self.source_type = source_type
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            if self.source_type == "image":
                self._detect_image()
            elif self.source_type == "video":
                self._detect_video()
        except Exception as e:
            self.log.emit(f"Error: {e}")
            self.done.emit(0)

    def _detect_image(self):
        import cv2
        results = self.model.predict(
            source=self.source,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        total_det = 0
        for r in results:
            img = r.plot()
            dets = []
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = r.names[cls_id]
                    conf_val = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    dets.append({
                        "class": cls_name,
                        "confidence": conf_val,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    })
                    total_det += 1
            self.frame_result.emit(img, dets)

            # Save annotated image
            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                name = Path(self.source).stem
                cv2.imwrite(
                    os.path.join(self.output_dir, f"{name}_detected.jpg"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )

        self.log.emit(f"Image: {total_det} objects detected")
        self.done.emit(total_det)

    def _detect_video(self):
        import cv2
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.log.emit("Cannot open video")
            self.done.emit(0)
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            out_path = os.path.join(self.output_dir, "detected_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        total_det = 0
        frame_idx = 0

        while not self._stop:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(
                source=frame,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )

            dets = []
            for r in results:
                img = r.plot()
                if r.boxes is not None:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = r.names[cls_id]
                        conf_val = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        dets.append({
                            "frame": frame_idx,
                            "class": cls_name,
                            "confidence": conf_val,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        })
                        total_det += 1

            self.frame_result.emit(img if 'img' in dir() else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dets)

            if writer and 'img' in dir():
                writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            frame_idx += 1
            if frame_idx % 5 == 0:
                self.progress.emit(frame_idx, total_frames)

        cap.release()
        if writer:
            writer.release()

        self.log.emit(f"Video: {total_det} objects in {frame_idx} frames")
        self.done.emit(total_det)

    def get_last_detections(self):
        return self._last_dets if hasattr(self, '_last_dets') else []


class DetectPage(object):
    pass


class DetectPage(QWidget):
    """Detection page for images, videos, and camera."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._model = None
        self._model_path = None
        self._worker = None
        self._camera_cap = None
        self._camera_timer = QTimer()
        self._camera_timer.timeout.connect(self._camera_frame)
        self._all_detections = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ── Model selection ──
        model_row = QHBoxLayout()
        model_row.setSpacing(8)

        model_label = QLabel(t("select_model_pt"))
        model_label.setObjectName("sectionTitle")
        model_row.addWidget(model_label)

        self.model_label = QLabel(t("no_model_loaded"))
        self.model_label.setObjectName("hint")
        model_row.addWidget(self.model_label, 1)

        self.load_btn = styled_button(t("load_model"), "#f97316", 12)
        self.load_btn.setMaximumWidth(120)
        self.load_btn.clicked.connect(self._load_model)
        model_row.addWidget(self.load_btn)
        layout.addLayout(model_row)

        # ── Preview ──
        self.preview_label = QLabel(t("no_model_loaded"))
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(320)
        self.preview_label.setStyleSheet("""
            QLabel {
                background: #0c1017;
                border: 2px dashed #374151;
                border-radius: 12px;
                color: #6b7280;
                font-size: 16px;
            }
        """)
        layout.addWidget(self.preview_label, 1)

        # ── Settings row ──
        settings = QHBoxLayout()
        settings.setSpacing(12)

        # Confidence
        conf_label = QLabel(t("confidence"))
        conf_label.setObjectName("sectionTitleSmall")
        settings.addWidget(conf_label)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setMaximumWidth(100)
        settings.addWidget(self.conf_spin)

        # IoU
        iou_label = QLabel(t("iou_thresh"))
        iou_label.setObjectName("sectionTitleSmall")
        settings.addWidget(iou_label)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.1, 1.0)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setMaximumWidth(100)
        settings.addWidget(self.iou_spin)

        settings.addStretch()

        # Source buttons
        self.img_btn = styled_button(t("detect_image"), "#3b82f6", 12)
        self.img_btn.clicked.connect(self._detect_image)
        settings.addWidget(self.img_btn)

        self.video_btn = styled_button(t("detect_video"), "#8b5cf6", 12)
        self.video_btn.clicked.connect(self._detect_video)
        settings.addWidget(self.video_btn)

        self.cam_btn = styled_button(t("detect_camera"), "#22c55e", 12)
        self.cam_btn.clicked.connect(self._toggle_camera)
        settings.addWidget(self.cam_btn)

        layout.addLayout(settings)

        # ── Export row ──
        export_row = QHBoxLayout()
        export_row.setSpacing(8)

        self.csv_btn = styled_button(t("export_csv"), "#6b7280", 11)
        self.csv_btn.setMaximumWidth(120)
        self.csv_btn.clicked.connect(self._export_csv)
        export_row.addWidget(self.csv_btn)

        self.save_img_btn = styled_button(t("save_annotated"), "#6b7280", 11)
        self.save_img_btn.setMaximumWidth(140)
        self.save_img_btn.clicked.connect(self._save_last_annotated)
        export_row.addWidget(self.save_img_btn)

        export_row.addStretch()

        self.status_label = QLabel(t("ready"))
        self.status_label.setObjectName("hint")
        export_row.addWidget(self.status_label)
        layout.addLayout(export_row)

        # ── Progress ──
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

    # ─── Model ───

    def _load_model(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, t("select_model_pt"), "",
            "YOLO Model (*.pt);;All (*)"
        )
        if not filepath:
            return

        self.model_label.setText("Loading...")
        self.model_label.setStyleSheet("color: #f97316; font-size: 12px;")

        try:
            from ultralytics import YOLO
            self._model = YOLO(filepath)
            self._model_path = filepath
            self.model_label.setText(Path(filepath).name)
            self.model_label.setStyleSheet("color: #22c55e; font-size: 12px;")
        except Exception as e:
            self.model_label.setText(f"Error: {e}")
            self.model_label.setStyleSheet("color: #ef4444; font-size: 12px;")

    # ─── Detect Image ───

    def _detect_image(self):
        if not self._model:
            QMessageBox.warning(self, t("title"), t("no_model_loaded"))
            return

        filepath, _ = QFileDialog.getOpenFileName(
            self, t("detect_image"), "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All (*)"
        )
        if not filepath:
            return

        self.status_label.setText(t("detecting"))
        self.status_label.setStyleSheet("color: #f97316; font-size: 12px;")

        output_dir = str(Path(filepath).parent / "detection_results")
        self._worker = DetectionWorker(
            self._model, filepath, self.conf_spin.value(), self.iou_spin.value(),
            output_dir, "image"
        )
        self._worker.frame_result.connect(self._on_frame_result)
        self._worker.log.connect(self._log)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    # ─── Detect Video ───

    def _detect_video(self):
        if not self._model:
            QMessageBox.warning(self, t("title"), t("no_model_loaded"))
            return

        filepath, _ = QFileDialog.getOpenFileName(
            self, t("detect_video"), "",
            "Video (*.mp4 *.avi *.mov *.mkv *.wmv);;All (*)"
        )
        if not filepath:
            return

        output_dir = str(Path(filepath).parent / "detection_results")
        self.status_label.setText(t("detecting"))
        self.status_label.setStyleSheet("color: #f97316; font-size: 12px;")

        self._worker = DetectionWorker(
            self._model, filepath, self.conf_spin.value(), self.iou_spin.value(),
            output_dir, "video"
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.frame_result.connect(self._on_frame_result)
        self._worker.log.connect(self._log)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    # ─── Camera ───

    def _toggle_camera(self):
        if self._camera_cap and self._camera_cap.isOpened():
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        import cv2
        if not self._model:
            QMessageBox.warning(self, t("title"), t("no_model_loaded"))
            return

        self._camera_cap = cv2.VideoCapture(0)
        if not self._camera_cap.isOpened():
            QMessageBox.warning(self, t("title"), "Cannot open camera")
            return

        self.cam_btn.setText(t("stop"))
        self.cam_btn.setStyleSheet("""
            HoverButton {
                background: #ef4444; color: #fff; border: none;
                padding: 8px 18px; border-radius: 8px; font-size: 12px; font-weight: bold;
            }
        """)
        self._camera_timer.start(33)  # ~30fps

    def _stop_camera(self):
        self._camera_timer.stop()
        if self._camera_cap:
            self._camera_cap.release()
            self._camera_cap = None
        self.cam_btn.setText(t("detect_camera"))
        self.cam_btn._base_color = "#22c55e"
        self.cam_btn.setStyleSheet(f"""
            HoverButton {{
                background: #22c55e; color: #fff; border: none;
                padding: 8px 18px; border-radius: 8px; font-size: 12px; font-weight: bold;
            }}
        """)

    def _camera_frame(self):
        import cv2
        if not self._camera_cap or not self._camera_cap.isOpened():
            return

        ret, frame = self._camera_cap.read()
        if not ret:
            return

        results = self._model.predict(
            source=frame, conf=self.conf_spin.value(),
            iou=self.iou_spin.value(), verbose=False,
        )

        for r in results:
            img = r.plot()
            self._show_frame(img)

    # ─── UI helpers ───

    def _on_frame_result(self, frame, dets):
        self._last_frame = frame
        self._last_dets = dets
        self._all_detections.extend(dets)
        self._show_frame(frame)

    def _show_frame(self, frame):
        import cv2
        import numpy as np
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb = frame
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape
        # Copy data to prevent garbage collection
        data = rgb.copy()
        qimg = QImage(data.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        scaled = pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def _on_progress(self, current, total):
        if total > 0:
            self.progress_bar.setValue(int(current / total * 100))

    def _on_done(self, count):
        self.progress_bar.setValue(100)
        self.status_label.setText(f"{t('detect_done')}: {count}")
        self.status_label.setStyleSheet("color: #22c55e; font-size: 12px; font-weight: bold;")

    def _log(self, msg):
        self.status_label.setText(msg)

    # ─── Export ───

    def _export_csv(self):
        if not self._all_detections:
            QMessageBox.warning(self, t("title"), t("no_result"))
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, t("export_csv"), "detections.csv", "CSV (*.csv)"
        )
        if not filepath:
            return

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "class", "confidence", "x1", "y1", "x2", "y2"])
            writer.writeheader()
            for det in self._all_detections:
                writer.writerow(det)

        self.status_label.setText(f"CSV saved: {filepath}")
        self.status_label.setStyleSheet("color: #22c55e; font-size: 12px;")

    def _save_last_annotated(self):
        import cv2
        if not hasattr(self, '_last_frame') or self._last_frame is None:
            QMessageBox.warning(self, t("title"), t("no_result"))
            return

        folder = QFileDialog.getExistingDirectory(self, t("save_annotated"))
        if not folder:
            return

        filename = f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(folder, filename)

        frame = self._last_frame
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        cv2.imwrite(filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

        self.status_label.setText(f"Saved: {filepath}")
        self.status_label.setStyleSheet("color: #22c55e; font-size: 12px;")

    def retranslate(self):
        self.load_btn.setText(t("load_model"))
        self.img_btn.setText(t("detect_image"))
        self.video_btn.setText(t("detect_video"))
        self.cam_btn.setText(t("detect_camera"))
        self.csv_btn.setText(t("export_csv"))
        self.save_img_btn.setText(t("save_annotated"))
        if self.model_label.text() in ("未加载模型", "No model loaded"):
            self.model_label.setText(t("no_model_loaded"))
        if self.status_label.text() in ("就绪", "Ready"):
            self.status_label.setText(t("ready"))
