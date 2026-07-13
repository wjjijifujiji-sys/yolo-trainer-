"""Train page - model training with responsive layout."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog, QComboBox, QSpinBox,
    QTextEdit, QProgressBar, QMessageBox, QFrame,
)

from ui.components import styled_button
from utils.dataset_parser import DatasetParser
from utils.i18n import t


class TrainPage(QWidget):
    """Training page with model selection, parameter config, and training control."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.trainer = None
        self.parser = DatasetParser()
        self._gpu_available = self._check_gpu()
        self._setup_ui()

    def _check_gpu(self) -> tuple:
        """Check GPU and CUDA status. Returns (status_code, message).
        status_code: 0=no GPU, 1=NVIDIA+CUDA ready, 2=NVIDIA no CUDA, 3=AMD/other
        """
        import subprocess

        # Try to detect CUDA via system Python (works in exe too)
        try:
            result = subprocess.run(
                ["py", "-c", "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.version.cuda)"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) >= 3 and lines[0].strip() == "True":
                    return 1, f"GPU: {lines[1].strip()} | CUDA {lines[2].strip()}"
        except Exception:
            pass

        # Fallback: try importing torch directly (works when running from source)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                cuda_ver = torch.version.cuda
                return 1, f"GPU: {gpu_name} | CUDA {cuda_ver}"
        except ImportError:
            pass

        # Check NVIDIA GPU via nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_info = result.stdout.strip().split(",")[0]
                return 2, f"NVIDIA: {gpu_info} | 无CUDA加速"
        except Exception:
            pass

        # Check AMD GPU via WMI (Windows)
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "Name"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    name = line.strip()
                    if "AMD" in name.upper() or "RADEON" in name.upper():
                        return 3, f"AMD: {name} | 需要ROCm加速"
                    if "NVIDIA" in name.upper() or "GEFORCE" in name.upper() or "QUADRO" in name.upper() or "RTX" in name.upper() or "GTX" in name.upper():
                        return 2, f"NVIDIA: {name} | 无CUDA加速"
        except Exception:
            pass

        return 0, t("no_gpu")

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # ── Title row ──
        title_row = QHBoxLayout()
        title = QLabel(t("model_train"))
        title.setFont(QFont("Microsoft YaHei UI", 22, QFont.Weight.Bold))
        title.setStyleSheet("color: #f97316;")
        title_row.addWidget(title)
        title_row.addStretch()

        gpu_status = self._gpu_available[0]
        gpu_text = self._gpu_available[1]

        # Color coding: 0=gray, 1=green, 2=yellow(warning), 3=red(warning)
        if gpu_status == 1:
            gpu_color, gpu_bg = "#22c55e", "#052e16"
        elif gpu_status == 0:
            gpu_color, gpu_bg = "#6b7280", "#1f2937"
        else:
            gpu_color, gpu_bg = "#eab308", "#422006"

        self.gpu_label = QLabel(gpu_text)
        self.gpu_label.setStyleSheet(f"""
            background: {gpu_bg}; color: {gpu_color};
            padding: 6px 16px; border-radius: 20px;
            font-size: 12px; font-weight: bold;
            border: 1px solid {gpu_color};
        """)

        # GPU tooltips
        if gpu_status == 1:
            self.gpu_label.setToolTip("GPU加速已就绪，训练速度将大幅提升")
        elif gpu_status == 2:
            self.gpu_label.setToolTip("检测到NVIDIA显卡但未安装CUDA\n请向AI助手询问如何安装对应版本的CUDA")
        elif gpu_status == 3:
            self.gpu_label.setToolTip("检测到AMD显卡\n需要安装ROCm加速（Linux）或DirectML（Windows）\n请向AI助手询问具体方案")
        else:
            self.gpu_label.setToolTip("未检测到独立显卡\n将使用CPU训练，速度较慢")
        title_row.addWidget(self.gpu_label)
        layout.addLayout(title_row)

        # ── Split into left (controls) and right (log) ──
        from PyQt6.QtWidgets import QSplitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        # ── Left: controls ──
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(12)

        # Dataset section
        ds_title = QLabel(t("dataset"))
        ds_title.setObjectName("sectionTitle")
        left_layout.addWidget(ds_title)

        ds_row = QHBoxLayout()
        ds_row.setSpacing(8)
        self.ds_path_label = QLabel(t("no_dataset"))
        self.ds_path_label.setObjectName("hint")
        self.ds_path_label.setWordWrap(True)
        ds_row.addWidget(self.ds_path_label, 1)

        self.browse_btn = styled_button(t("browse"), "#f97316", 12)
        self.browse_btn.setMaximumWidth(80)
        self.browse_btn.clicked.connect(self._browse_dataset)
        ds_row.addWidget(self.browse_btn)

        self.clear_ds_btn = styled_button(t("clear"), "#6b7280", 12)
        self.clear_ds_btn.setMaximumWidth(80)
        self.clear_ds_btn.clicked.connect(self._clear_dataset)
        ds_row.addWidget(self.clear_ds_btn)
        left_layout.addLayout(ds_row)

        # Model section
        model_title = QLabel(t("model"))
        model_title.setObjectName("sectionTitle")
        left_layout.addWidget(model_title)

        self.model_combo = QComboBox()
        self._all_models = [
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
            "yolov8l.pt", "yolov8x.pt",
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
            "yolo11l.pt", "yolo11x.pt",
        ]
        tools_dir = Path(__file__).parent.parent / "tools"
        available = [m for m in self._all_models if (tools_dir / m).exists()]
        if not available:
            available = self._all_models
        self._available_models = available
        self.model_combo.addItems(available)
        self.model_combo.setCurrentText("yolov8n.pt" if "yolov8n.pt" in available else available[0])
        left_layout.addWidget(self.model_combo)

        # Parameters section
        param_title = QLabel(t("params"))
        param_title.setObjectName("sectionTitle")
        left_layout.addWidget(param_title)

        param_grid = QGridLayout()
        param_grid.setSpacing(10)

        # Epochs
        ep_label = QLabel(t("epochs"))
        ep_label.setObjectName("sectionTitleSmall")
        param_grid.addWidget(ep_label, 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        param_grid.addWidget(self.epochs_spin, 1, 0)

        # Batch Size
        bs_label = QLabel(t("batch_size"))
        bs_label.setObjectName("sectionTitleSmall")
        param_grid.addWidget(bs_label, 0, 1)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(16)
        param_grid.addWidget(self.batch_spin, 1, 1)

        # Image Size
        im_label = QLabel(t("img_size"))
        im_label.setObjectName("sectionTitleSmall")
        param_grid.addWidget(im_label, 0, 2)
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(320, 1280)
        self.img_size_spin.setValue(640)
        self.img_size_spin.setSingleStep(32)
        param_grid.addWidget(self.img_size_spin, 1, 2)

        # Device
        dev_label = QLabel(t("device"))
        dev_label.setObjectName("sectionTitleSmall")
        param_grid.addWidget(dev_label, 0, 3)
        self.device_combo = QComboBox()
        self.device_combo.addItems([t("auto_device"), "CPU", "0", "0,1"])
        self.device_combo.setCurrentIndex(0)
        param_grid.addWidget(self.device_combo, 1, 3)

        param_grid.setColumnStretch(0, 1)
        param_grid.setColumnStretch(1, 1)
        param_grid.setColumnStretch(2, 1)
        param_grid.setColumnStretch(3, 1)
        left_layout.addLayout(param_grid)

        # ── Divider ──
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet("color: #374151;")
        left_layout.addWidget(div)

        # ── Train / Stop buttons ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self.start_btn = styled_button(t("start_train"), "#f97316", 15)
        self.start_btn.setMinimumHeight(44)
        self.start_btn.clicked.connect(self._start_training)
        btn_row.addWidget(self.start_btn)

        self.stop_btn = styled_button(t("stop"), "#ef4444", 15)
        self.stop_btn.setMinimumHeight(44)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)
        btn_row.addWidget(self.stop_btn)
        left_layout.addLayout(btn_row)

        # ── Progress ──
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        left_layout.addWidget(self.progress_bar)

        self.status_label = QLabel(t("ready"))
        self.status_label.setObjectName("hint")
        left_layout.addWidget(self.status_label)

        left_layout.addStretch()
        splitter.addWidget(left)

        # ── Right: log + export ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(12)

        log_title = QLabel("Log")
        log_title.setObjectName("sectionTitle")
        right_layout.addWidget(log_title)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        right_layout.addWidget(self.log_text, 1)

        # Model export section
        div2 = QFrame()
        div2.setFrameShape(QFrame.Shape.HLine)
        div2.setStyleSheet("color: #374151;")
        right_layout.addWidget(div2)

        export_title = QLabel(t("model_export"))
        export_title.setObjectName("sectionTitle")
        right_layout.addWidget(export_title)

        conv_row = QHBoxLayout()
        conv_row.setSpacing(8)
        self.conv_pt_label = QLabel(t("no_model_selected"))
        self.conv_pt_label.setObjectName("hint")
        self.conv_pt_label.setWordWrap(True)
        conv_row.addWidget(self.conv_pt_label, 1)

        self.conv_select_btn = styled_button(t("select"), "#f97316", 11)
        self.conv_select_btn.setMaximumWidth(80)
        self.conv_select_btn.clicked.connect(self._select_convert_pt)
        conv_row.addWidget(self.conv_select_btn)
        right_layout.addLayout(conv_row)

        fmt_row = QHBoxLayout()
        fmt_row.setSpacing(8)
        self.conv_format_combo = QComboBox()
        self.conv_format_combo.addItems(["onnx", "engine", "tflite", "coreml"])
        fmt_row.addWidget(self.conv_format_combo, 1)

        self.convert_btn = styled_button(t("convert"), "#8b5cf6", 12)
        self.convert_btn.setMinimumWidth(100)
        self.convert_btn.clicked.connect(self._convert_model)
        fmt_row.addWidget(self.convert_btn)
        right_layout.addLayout(fmt_row)

        right_layout.addStretch()
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([600, 300])
        layout.addWidget(splitter, 1)

    # ─── i18n ───

    def retranslate(self):
        self.start_btn.setText(t("start_train"))
        self.stop_btn.setText(t("stop"))
        self.conv_select_btn.setText(t("select"))
        self.convert_btn.setText(t("convert"))
        self.browse_btn.setText(t("browse"))
        self.clear_ds_btn.setText(t("clear"))
        self.gpu_label.setText(t("gpu_ready") if self._gpu_available[0] else t("no_gpu"))
        if self.ds_path_label.text() in ("未选择数据集", "No dataset selected"):
            self.ds_path_label.setText(t("no_dataset"))
        if self.conv_pt_label.text() in ("未选择模型", "No model selected"):
            self.conv_pt_label.setText(t("no_model_selected"))
        if self.status_label.text() in ("就绪", "Ready"):
            self.status_label.setText(t("ready"))
        # Update device combo
        current_device = self.device_combo.currentText()
        self.device_combo.clear()
        self.device_combo.addItems([t("auto_device"), "CPU", "0", "0,1"])
        # Re-select
        for i in range(self.device_combo.count()):
            if self.device_combo.itemText(i) == current_device:
                self.device_combo.setCurrentIndex(i)
                break

    # ─── Handlers ───

    def _browse_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, t("dataset"))
        if folder:
            self.ds_path_label.setText(folder)
            self.ds_path_label.setStyleSheet("color: #22c55e; font-size: 12px;")
            self._validate_dataset(Path(folder))

    def _clear_dataset(self):
        self.ds_path_label.setText(t("no_dataset"))
        self.ds_path_label.setStyleSheet("color: #6b7280; font-size: 12px;")

    def _validate_dataset(self, dataset_dir: Path):
        stats = self.parser.parse_and_validate(str(dataset_dir))
        if stats.get("valid"):
            self.log(t("dataset_valid", count=stats["total_images"], classes=stats.get("classes", [])))
        else:
            self.log(t("invalid_dataset", err=stats.get("error", "unknown")))

    def _start_training(self):
        dataset_dir_str = self.ds_path_label.text()
        if dataset_dir_str in (t("no_dataset"), "未选择数据集", "No dataset selected", "未选择数据集"):
            self.log(t("please_select_ds"))
            return

        dataset_dir = Path(dataset_dir_str)
        if not dataset_dir.exists():
            self.log(t("path_not_exist", path=dataset_dir))
            return

        tools_dir = Path(__file__).parent.parent / "tools"
        model_name = self.model_combo.currentText()
        model_path = tools_dir / model_name
        if not model_path.exists():
            self.log(t("model_not_exist", model=model_name))
            self.log(t("download_hint"))
            return

        epochs = self.epochs_spin.value()
        batch_size = self.batch_spin.value()
        img_size = self.img_size_spin.value()
        device_str = self.device_combo.currentText()
        if "cpu" in device_str.lower():
            device = "cpu"
        elif t("auto_device") in device_str or "auto" in device_str.lower() or "自动" in device_str:
            try:
                import subprocess
                result = subprocess.run(
                    ["py", "-c", "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"],
                    capture_output=True, text=True, timeout=10
                )
                device = result.stdout.strip() if result.returncode == 0 else "cpu"
            except Exception:
                device = "cpu"
        else:
            device = device_str

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(t("training"))
        self.status_label.setStyleSheet("color: #f97316; font-size: 13px; font-weight: bold;")

        output_dir = dataset_dir.parent / "outputs"
        output_dir.mkdir(exist_ok=True)

        self.log(t("train_log_start"))
        self.log(t("train_log_model", model=model_name))
        self.log(t("train_log_dataset", dataset=dataset_dir))
        self.log(t("train_log_device", device=device))
        self.log(t("train_log_params", ep=epochs, bs=batch_size, isize=img_size))
        self.log("")

        self.trainer = YOLOTrainerThread(
            dataset_dir=str(dataset_dir),
            model=str(model_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            project=str(output_dir),
            name="run",
        )
        self.trainer.log_signal.connect(self.log)
        self.trainer.done_signal.connect(self._on_train_done)
        self.trainer.progress_signal.connect(self._on_progress)
        self.trainer.start()

    def _on_progress(self, epoch, total):
        if total > 0:
            pct = int(epoch / total * 100)
            self.progress_bar.setValue(pct)

    def _on_train_done(self, success, output_path):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if success:
            self.progress_bar.setValue(100)
            self.status_label.setText(t("train_done"))
            self.status_label.setStyleSheet("color: #22c55e; font-size: 13px; font-weight: bold;")
            self.log(f"\n{output_path}")
        else:
            self.status_label.setText(t("train_fail"))
            self.status_label.setStyleSheet("color: #ef4444; font-size: 13px; font-weight: bold;")
            self.log(f"\n{output_path}")

    def _stop_training(self):
        if self.trainer:
            self.trainer.stop()
            self.log(t("train_user_stop"))

    def _select_convert_pt(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "YOLO Model (.pt)", "", "YOLO Models (*.pt)"
        )
        if filepath:
            self.conv_pt_label.setText(filepath)
            self.conv_pt_label.setStyleSheet("color: #22c55e; font-size: 12px;")

    def _convert_model(self):
        pt_path = self.conv_pt_label.text()
        if pt_path in (t("no_model_selected"), "未选择模型", "No model selected"):
            QMessageBox.warning(self, t("title"), t("select_model_first"))
            return

        fmt = self.conv_format_combo.currentText()
        self.convert_btn.setEnabled(False)
        self.convert_btn.setText(t("converting"))
        self.status_label.setText(t("exporting", fmt=fmt.upper()))
        self.status_label.setStyleSheet("color: #8b5cf6; font-size: 13px; font-weight: bold;")

        try:
            from ultralytics import YOLO
            self.log(f"Loading model: {pt_path}")
            model = YOLO(pt_path)
            self.log(t("exporting", fmt=fmt.upper()))

            output_path = model.export(format=fmt)
            self.log(f"Exported: {output_path}")
            self.status_label.setText(t("export_done"))
            self.status_label.setStyleSheet("color: #22c55e; font-size: 13px; font-weight: bold;")
        except Exception as e:
            self.log(t("export_fail_msg", err=str(e)))
            self.status_label.setText(t("train_fail"))
            self.status_label.setStyleSheet("color: #ef4444; font-size: 13px; font-weight: bold;")
        finally:
            self.convert_btn.setEnabled(True)
            self.convert_btn.setText(t("convert"))

    def log(self, msg):
        self.log_text.append(msg)
        scroll = self.log_text.verticalScrollBar()
        scroll.setValue(scroll.maximum())


class YOLOTrainerThread(QThread):
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(bool, str)
    progress_signal = pyqtSignal(int, int)

    def __init__(self, dataset_dir, model, epochs, batch, imgsz, device, project, name):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.model = model
        self.epochs = epochs
        self.batch = batch
        self.imgsz = imgsz
        self.device = device
        self.project = project
        self.name = name
        self._stop = False

    def run(self):
        try:
            from ultralytics import YOLO
            import time

            self.log_signal.emit("Generating dataset config...")
            yaml_path = self._create_yaml_config()
            self.log_signal.emit(f"  Config: {yaml_path}")

            self.log_signal.emit(f"Loading model: {Path(self.model).name}")
            model = YOLO(self.model)

            self.log_signal.emit("Training started...")
            start_time = time.time()

            results = model.train(
                data=yaml_path,
                epochs=self.epochs,
                batch=self.batch,
                imgsz=self.imgsz,
                device=self.device,
                project=self.project,
                name=self.name,
                exist_ok=True,
                verbose=True,
            )

            elapsed = time.time() - start_time

            if self._stop:
                self.log_signal.emit("\nTraining interrupted")
                self.done_signal.emit(False, "Interrupted")
                return

            if hasattr(results, 'box'):
                box = results.box
                map50 = box.map50 if hasattr(box, 'map50') else "N/A"
                map50_95 = box.map50_95 if hasattr(box, 'map50_95') else "N/A"
            else:
                map50 = "N/A"
                map50_95 = "N/A"

            self.log_signal.emit(f"\nTraining complete! Elapsed: {elapsed:.1f}s")
            self.log_signal.emit(f"  mAP50: {map50}, mAP50-95: {map50_95}")
            best_weight = Path(self.project) / self.name / "weights" / "best.pt"
            self.done_signal.emit(True, str(best_weight))

        except Exception as e:
            self.log_signal.emit(f"\nError: {str(e)}")
            self.done_signal.emit(False, str(e))

    def _create_yaml_config(self):
        from utils.dataset_parser import DatasetParser
        parser = DatasetParser()
        stats = parser.parse_and_validate(self.dataset_dir)
        if not stats.get("valid"):
            raise ValueError(f"Invalid dataset: {stats.get('error', 'unknown')}")
        yaml_path = parser.create_yaml_config(
            self.dataset_dir,
            num_classes=len(stats.get("classes", [])),
            class_names=[f"class_{i}" for i in stats.get("classes", [])],
        )
        return yaml_path

    def stop(self):
        self._stop = True
