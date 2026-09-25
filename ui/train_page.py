"""Train page - model training with responsive layout."""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QScrollArea, QTextEdit, QProgressBar, QMessageBox, QFrame,
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
        self._advanced = False
        self._setup_ui()
        self._update_resume_state()

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

        # Check NVIDIA GPU via nvidia-smi​‌‌​‌​‌​​‌‌​‌​​‌​‌‌​‌​‌​​‌‌​‌​​‌​‌‌​​‌‌​​‌‌‌​‌​‌
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_info = result.stdout.strip().split(",")[0]
                return 2, f"NVIDIA: {gpu_info} | {t('no_cuda')}"
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
                        return 3, f"AMD: {name} | {t('need_rocm')}"
                    if "NVIDIA" in name.upper() or "GEFORCE" in name.upper() or "QUADRO" in name.upper() or "RTX" in name.upper() or "GTX" in name.upper():
                        return 2, f"NVIDIA: {name} | {t('no_cuda')}"
        except Exception:
            pass

        return 0, t("no_gpu")

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # ── Title row ──
        title_row = QHBoxLayout()
        self.title_label = QLabel(t("model_train"))
        self.title_label.setFont(QFont("Microsoft YaHei UI", 22, QFont.Weight.Bold))
        self.title_label.setStyleSheet("color: #f97316;")
        title_row.addWidget(self.title_label)
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
            self.gpu_label.setToolTip(t("gpu_ready_tooltip"))
        elif gpu_status == 2:
            self.gpu_label.setToolTip(t("no_cuda_tooltip"))
        elif gpu_status == 3:
            self.gpu_label.setToolTip(t("amd_tooltip"))
        else:
            self.gpu_label.setToolTip(t("no_gpu_tooltip"))
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
        self.ds_title = QLabel(t("dataset"))
        self.ds_title.setObjectName("sectionTitle")
        left_layout.addWidget(self.ds_title)

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
        self.model_title = QLabel(t("model"))
        self.model_title.setObjectName("sectionTitle")
        left_layout.addWidget(self.model_title)

        self.model_combo = QComboBox()
        self._all_models = [
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
            "yolov8l.pt", "yolov8x.pt",
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
            "yolo11l.pt", "yolo11x.pt",
            "yolo26n.pt", "yolo26s.pt", "yolo26m.pt",
            "yolo26l.pt", "yolo26x.pt",
        ]#新增不同版本的YOLO
        tools_dir = Path(__file__).parent.parent / "tools"
        available = [m for m in self._all_models if (tools_dir / m).exists()]
        if not available:
            available = self._all_models
        self._available_models = available
        self.model_combo.addItems(available)
        self.model_combo.setCurrentText("yolov8n.pt" if "yolov8n.pt" in available else available[0])
        left_layout.addWidget(self.model_combo)

        # Parameters section
        self.param_title = QLabel(t("params"))
        self.param_title.setObjectName("sectionTitle")
        left_layout.addWidget(self.param_title)

        param_grid = QGridLayout()
        param_grid.setSpacing(10)

        # Epochs
        self.ep_label = QLabel(t("epochs"))
        self.ep_label.setObjectName("sectionTitleSmall")
        param_grid.addWidget(self.ep_label, 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        param_grid.addWidget(self.epochs_spin, 1, 0)

        # Batch Size
        self.bs_label = QLabel(t("batch_size"))
        self.bs_label.setObjectName("sectionTitleSmall")
        param_grid.addWidget(self.bs_label, 0, 1)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(16)
        param_grid.addWidget(self.batch_spin, 1, 1)

        # Image Size
        self.im_label = QLabel(t("img_size"))
        self.im_label.setObjectName("sectionTitleSmall")
        param_grid.addWidget(self.im_label, 0, 2)
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(320, 1280)
        self.img_size_spin.setValue(640)
        self.img_size_spin.setSingleStep(32)
        param_grid.addWidget(self.img_size_spin, 1, 2)

        # Device
        self.dev_label = QLabel(t("device"))
        self.dev_label.setObjectName("sectionTitleSmall")
        param_grid.addWidget(self.dev_label, 0, 3)
        self.device_combo = QComboBox()
        self.device_combo.addItems([t("auto_device"), "CPU", "0", "0,1"])
        self.device_combo.setCurrentIndex(0)
        param_grid.addWidget(self.device_combo, 1, 3)

        param_grid.setColumnStretch(0, 1)
        param_grid.setColumnStretch(1, 1)
        param_grid.setColumnStretch(2, 1)
        param_grid.setColumnStretch(3, 1)
        left_layout.addLayout(param_grid)

        # ── Advanced mode toggle ──​‌‌​‌​‌​​‌‌​‌​​‌​‌‌​‌​‌​​‌‌​‌​​‌​‌​‌‌‌‌‌​‌‌​​​‌​
        self.adv_toggle_btn = styled_button(t("advanced_mode"), "#8b5cf6", 11)
        self.adv_toggle_btn.setMaximumWidth(140)
        self.adv_toggle_btn.clicked.connect(self._toggle_advanced)
        left_layout.addWidget(self.adv_toggle_btn)

        # Advanced params panel (hidden until advanced mode is on)
        self.adv_scroll = QScrollArea()
        self.adv_scroll.setWidgetResizable(True)
        self.adv_scroll.setMaximumHeight(420)
        self.adv_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.adv_scroll.setStyleSheet("""
            QScrollArea { background: #111827; border: none; }
            QScrollArea > QWidget > QWidget { background: #111827; }
        """)
        adv_widget = QWidget()
        adv_grid = QGridLayout(adv_widget)
        adv_grid.setSpacing(8)
        self._build_advanced_params(adv_grid)
        self.adv_scroll.setWidget(adv_widget)
        self.adv_scroll.setVisible(False)
        left_layout.addWidget(self.adv_scroll)

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
        self.resume_btn = styled_button(t("resume_train"), "#06b6d4", 15)
        self.resume_btn.setMinimumHeight(44)
        self.resume_btn.clicked.connect(self._resume_training)
        btn_row.addWidget(self.resume_btn)
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

        self.log_title = QLabel(t("log_title"))
        self.log_title.setObjectName("sectionTitle")
        right_layout.addWidget(self.log_title)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        right_layout.addWidget(self.log_text, 1)

        # Model export section
        div2 = QFrame()
        div2.setFrameShape(QFrame.Shape.HLine)
        div2.setStyleSheet("color: #374151;")
        right_layout.addWidget(div2)

        self.export_title = QLabel(t("model_export"))
        self.export_title.setObjectName("sectionTitle")
        right_layout.addWidget(self.export_title)

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

    # ─── Advanced mode ─── 高级模式(https://docs.ultralytics.com/zh/models/yolo26)

    def _toggle_advanced(self):
        self._advanced = not self._advanced
        self.adv_scroll.setVisible(self._advanced)
        self.adv_toggle_btn.setText(t("hide_advanced") if self._advanced else t("advanced_mode"))

    def _build_advanced_params(self, grid):
        # 2 columns of [enable] [label] [control]
        items = self._create_advanced_controls()
        cols = 2
        for i, field in enumerate(items):
            row, col = divmod(i, cols)
            box = QWidget()
            h = QHBoxLayout(box)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            h.addWidget(field["enable"])
            field["label"].setObjectName("sectionTitleSmall")
            h.addWidget(field["label"])
            field["ctrl"].setMaximumWidth(120)
            h.addWidget(field["ctrl"])
            h.addStretch(1)
            grid.addWidget(box, row, col)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

    def _add_adv_field(self, items, key, label_key, ctrl):
        """One advanced row: enable checkbox (default OFF) + label + control."""
        enable = QCheckBox()
        enable.setChecked(False)
        enable.setToolTip(t("adv_enable_hint"))
        label = QLabel(t(label_key))
        ctrl.setEnabled(False)
        enable.toggled.connect(ctrl.setEnabled)
        items.append({
            "key": key,
            "enable": enable,
            "label": label,
            "ctrl": ctrl,
            "label_key": label_key,
        })
        return enable, label, ctrl

    def _create_advanced_controls(self):
        items = []

        def make_spin(spin_cls, lo, hi, default, decimals=None, step=None):
            s = spin_cls()
            s.setRange(lo, hi)
            if decimals is not None:
                s.setDecimals(decimals)
            if step is not None:
                s.setSingleStep(step)
            s.setValue(default)
            return s

        optimizer_combo = QComboBox()
        optimizer_combo.addItems(["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"])
        self._add_adv_field(items, "optimizer", "optimizer", optimizer_combo)

        self._add_adv_field(items, "lr0", "lr0", make_spin(QDoubleSpinBox, 0.00001, 0.1, 0.01, decimals=5))
        self._add_adv_field(items, "lrf", "lrf", make_spin(QDoubleSpinBox, 0.0001, 1.0, 0.01, decimals=4))
        self._add_adv_field(items, "momentum", "momentum", make_spin(QDoubleSpinBox, 0.0, 0.99, 0.937, decimals=3))
        self._add_adv_field(items, "weight_decay", "weight_decay", make_spin(QDoubleSpinBox, 0.0, 0.01, 0.0005, decimals=5))
        self._add_adv_field(items, "warmup_epochs", "warmup_epochs", make_spin(QDoubleSpinBox, 0.0, 10.0, 3.0, decimals=1))
        self._add_adv_field(items, "warmup_momentum", "warmup_momentum", make_spin(QDoubleSpinBox, 0.0, 0.99, 0.8, decimals=3))

        self._add_adv_field(items, "box", "box_loss", make_spin(QDoubleSpinBox, 0.0, 20.0, 7.5, decimals=2))
        self._add_adv_field(items, "cls", "cls_loss", make_spin(QDoubleSpinBox, 0.0, 20.0, 0.5, decimals=2))
        self._add_adv_field(items, "dfl", "dfl_loss", make_spin(QDoubleSpinBox, 0.0, 10.0, 1.5, decimals=2))

        self._add_adv_field(items, "patience", "patience", make_spin(QSpinBox, 0, 200, 100))
        self._add_adv_field(items, "seed", "seed", make_spin(QSpinBox, 0, 2147483647, 0))
        self._add_adv_field(items, "workers", "workers", make_spin(QSpinBox, 0, 16, 8))
        self._add_adv_field(items, "close_mosaic", "close_mosaic", make_spin(QSpinBox, 0, 100, 10))

        # Augmentations: probability 0-1 (only sent when enable is ON)
        self._add_adv_field(items, "mosaic", "mosaic", make_spin(QDoubleSpinBox, 0.0, 1.0, 1.0, decimals=2, step=0.05))
        self._add_adv_field(items, "mixup", "mixup", make_spin(QDoubleSpinBox, 0.0, 1.0, 0.0, decimals=2, step=0.05))
        self._add_adv_field(items, "fliplr", "flip_lr", make_spin(QDoubleSpinBox, 0.0, 1.0, 0.5, decimals=2, step=0.05))
        self._add_adv_field(items, "flipud", "flip_ud", make_spin(QDoubleSpinBox, 0.0, 1.0, 0.0, decimals=2, step=0.05))

        self._add_adv_field(items, "degrees", "degrees", make_spin(QDoubleSpinBox, 0.0, 180.0, 0.0, decimals=1))
        self._add_adv_field(items, "translate", "translate", make_spin(QDoubleSpinBox, 0.0, 1.0, 0.1, decimals=2))
        self._add_adv_field(items, "scale", "scale", make_spin(QDoubleSpinBox, 0.0, 1.0, 0.5, decimals=2))
        self._add_adv_field(items, "shear", "shear", make_spin(QDoubleSpinBox, 0.0, 180.0, 0.0, decimals=1))
        self._add_adv_field(items, "perspective", "perspective", make_spin(QDoubleSpinBox, 0.0, 0.001, 0.0, decimals=4))

        self._add_adv_field(items, "hsv_h", "hsv_h", make_spin(QDoubleSpinBox, 0.0, 1.0, 0.015, decimals=3))
        self._add_adv_field(items, "hsv_s", "hsv_s", make_spin(QDoubleSpinBox, 0.0, 1.0, 0.7, decimals=2))
        self._add_adv_field(items, "hsv_v", "hsv_v", make_spin(QDoubleSpinBox, 0.0, 1.0, 0.4, decimals=2))

        # Legacy attribute aliases for retranslate()​‌‌‌‌​​‌​‌​‌‌‌‌‌​​‌​​​​​‌‌‌​​‌‌‌‌​​​‌‌‌​‌​​​‌​‌‌
        by_key = {f["key"]: f for f in items}
        self.optimizer_label = by_key["optimizer"]["label"]
        self.optimizer_combo = by_key["optimizer"]["ctrl"]
        self.lr0_label = by_key["lr0"]["label"]
        self.lr0_spin = by_key["lr0"]["ctrl"]
        self.lrf_label = by_key["lrf"]["label"]
        self.lrf_spin = by_key["lrf"]["ctrl"]
        self.momentum_label = by_key["momentum"]["label"]
        self.momentum_spin = by_key["momentum"]["ctrl"]
        self.wd_label = by_key["weight_decay"]["label"]
        self.wd_spin = by_key["weight_decay"]["ctrl"]
        self.warmup_ep_label = by_key["warmup_epochs"]["label"]
        self.warmup_ep_spin = by_key["warmup_epochs"]["ctrl"]
        self.warmup_mom_label = by_key["warmup_momentum"]["label"]
        self.warmup_mom_spin = by_key["warmup_momentum"]["ctrl"]
        self.box_label = by_key["box"]["label"]
        self.box_spin = by_key["box"]["ctrl"]
        self.cls_label = by_key["cls"]["label"]
        self.cls_spin = by_key["cls"]["ctrl"]
        self.dfl_label = by_key["dfl"]["label"]
        self.dfl_spin = by_key["dfl"]["ctrl"]
        self.patience_label = by_key["patience"]["label"]
        self.patience_spin = by_key["patience"]["ctrl"]
        self.seed_label = by_key["seed"]["label"]
        self.seed_spin = by_key["seed"]["ctrl"]
        self.workers_label = by_key["workers"]["label"]
        self.workers_spin = by_key["workers"]["ctrl"]
        self.close_mosaic_label = by_key["close_mosaic"]["label"]
        self.close_mosaic_spin = by_key["close_mosaic"]["ctrl"]
        # mosaic/mixup/flip are now 0-1 probability spins (aliases kept)
        self.mosaic_chk = by_key["mosaic"]["ctrl"]
        self.mixup_chk = by_key["mixup"]["ctrl"]
        self.fliplr_chk = by_key["fliplr"]["ctrl"]
        self.flipud_chk = by_key["flipud"]["ctrl"]
        self.degrees_label = by_key["degrees"]["label"]
        self.degrees_spin = by_key["degrees"]["ctrl"]
        self.translate_label = by_key["translate"]["label"]
        self.translate_spin = by_key["translate"]["ctrl"]
        self.scale_label = by_key["scale"]["label"]
        self.scale_spin = by_key["scale"]["ctrl"]
        self.shear_label = by_key["shear"]["label"]
        self.shear_spin = by_key["shear"]["ctrl"]
        self.perspective_label = by_key["perspective"]["label"]
        self.perspective_spin = by_key["perspective"]["ctrl"]
        self.hsv_h_label = by_key["hsv_h"]["label"]
        self.hsv_h_spin = by_key["hsv_h"]["ctrl"]
        self.hsv_s_label = by_key["hsv_s"]["label"]
        self.hsv_s_spin = by_key["hsv_s"]["ctrl"]
        self.hsv_v_label = by_key["hsv_v"]["label"]
        self.hsv_v_spin = by_key["hsv_v"]["ctrl"]

        self._adv_fields = items
        return items

    def _collect_hyperparams(self):
        """Only fields whose enable checkbox is checked are sent.

        Default all OFF -> {} (stock ultralytics). Experts enable what they need.
        """
        hyper = {}
        for field in self._adv_fields:
            if not field["enable"].isChecked():
                continue
            ctrl = field["ctrl"]
            if isinstance(ctrl, QComboBox):
                hyper[field["key"]] = ctrl.currentText()
            elif isinstance(ctrl, (QDoubleSpinBox, QSpinBox)):
                hyper[field["key"]] = ctrl.value()
            elif isinstance(ctrl, QCheckBox):
                hyper[field["key"]] = 1.0 if ctrl.isChecked() else 0.0
        return hyper

    # ─── i18n ───

    def retranslate(self):
        self.title_label.setText(t("model_train"))
        self.start_btn.setText(t("start_train"))
        self.stop_btn.setText(t("stop"))
        self.resume_btn.setText(t("resume_train"))
        self.conv_select_btn.setText(t("select"))
        self.convert_btn.setText(t("convert"))
        self.browse_btn.setText(t("browse"))
        self.clear_ds_btn.setText(t("clear"))
        self.gpu_label.setText(t("gpu_ready") if self._gpu_available[0] else t("no_gpu"))
        # Update GPU tooltips
        gpu_status = self._gpu_available[0]
        if gpu_status == 1:
            self.gpu_label.setToolTip(t("gpu_ready_tooltip"))
        elif gpu_status == 2:
            self.gpu_label.setToolTip(t("no_cuda_tooltip"))
        elif gpu_status == 3:
            self.gpu_label.setToolTip(t("amd_tooltip"))
        else:
            self.gpu_label.setToolTip(t("no_gpu_tooltip"))
        # Section titles
        self.ds_title.setText(t("dataset"))
        self.model_title.setText(t("model"))
        self.param_title.setText(t("params"))
        self.log_title.setText(t("log_title"))
        self.export_title.setText(t("model_export"))
        # Param labels
        self.ep_label.setText(t("epochs"))
        self.bs_label.setText(t("batch_size"))
        self.im_label.setText(t("img_size"))
        self.dev_label.setText(t("device"))
        self.adv_toggle_btn.setText(t("hide_advanced") if self._advanced else t("advanced_mode"))
        # Advanced field labels (via stored fields)
        for field in getattr(self, "_adv_fields", []):
            field["label"].setText(t(field["label_key"]))
            field["enable"].setToolTip(t("adv_enable_hint"))
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
        # Re-select‌‌‌​​‌​​‌​‌‌‌‌‌‌‌​​​‌​‌​‌‌‌​​‌‌​‌​​‌‌‌​‌‌​‌‌​​​​
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
            self._update_resume_state()

    def _clear_dataset(self):
        self.ds_path_label.setText(t("no_dataset"))
        self.ds_path_label.setStyleSheet("color: #6b7280; font-size: 12px;")
        self._update_resume_state()

    def _validate_dataset(self, dataset_dir: Path):
        stats = self.parser.parse_and_validate(str(dataset_dir))
        if stats.get("valid"):
            self.log(t("dataset_valid", count=stats["total_images"], classes=stats.get("classes", [])))
        else:
            self.log(t("invalid_dataset", err=stats.get("error", "unknown")))

    def _start_training(self):
        dataset_dir_str = self.ds_path_label.text()
        if dataset_dir_str in (t("no_dataset"), "未选择数据集", "No dataset selected"):
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

        hyper = self._collect_hyperparams() if self._advanced else {}
        if hyper:
            self.log(f"  Advanced: optimizer={hyper['optimizer']}, lr0={hyper['lr0']}, "
                     f"lrf={hyper['lrf']}, momentum={hyper['momentum']}")

        output_dir = dataset_dir / "outputs"
        output_dir.mkdir(exist_ok=True)

        self.log(t("train_log_start"))
        self.log(t("train_log_model", model=model_name))
        self.log(t("train_log_dataset", dataset=dataset_dir))
        self.log(t("train_log_device", device=device))
        self.log(t("train_log_params", ep=epochs, bs=batch_size, isize=img_size))
        self.log("")

        self._start_trainer(
            dataset_dir=str(dataset_dir),
            model=str(model_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            project=str(output_dir),
            name="run",
            hyperparams=hyper,
        )

    def _on_progress(self, epoch, total):
        if total > 0:
            pct = int(epoch / total * 100)
            self.progress_bar.setValue(pct)

    def _start_trainer(self, **kwargs):
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText(t("training"))
        self.status_label.setStyleSheet("color: #f97316; font-size: 13px; font-weight: bold;")
        self.trainer = YOLOTrainerThread(**kwargs)
        self.trainer.log_signal.connect(self.log)
        self.trainer.done_signal.connect(self._on_train_done)
        self.trainer.progress_signal.connect(self._on_progress)
        self.trainer.start()

    def _resume_training(self):
        dataset_dir_str = self.ds_path_label.text()
        if dataset_dir_str in (t("no_dataset"), "未选择数据集", "No dataset selected"):
            self.log(t("please_select_ds"))
            return
        dataset_dir = Path(dataset_dir_str)
        if not dataset_dir.exists():
            self.log(t("path_not_exist", path=dataset_dir))
            return
        last_pt = dataset_dir / "outputs" / "run" / "weights" / "last.pt"
        if not last_pt.exists():
            self.log(t("no_checkpoint", path=last_pt))
            return
        self.log(t("resume_log", path=last_pt))
        self._start_trainer(
            dataset_dir=str(dataset_dir),
            model=str(last_pt),
            project=str(dataset_dir / "outputs"),
            name="run",
            resume=str(last_pt),
        )

    def _update_resume_state(self):
        if self.trainer and self.trainer.isRunning():
            return
        dataset_dir_str = self.ds_path_label.text()
        last_pt = None
        if dataset_dir_str not in (t("no_dataset"), "未选择数据集", "No dataset selected"):
            d = Path(dataset_dir_str)
            if d.exists():
                p = d / "outputs" / "run" / "weights" / "last.pt"
                if p.exists():
                    last_pt = p
        self.resume_btn.setEnabled(last_pt is not None)
        self.resume_btn.setToolTip(str(last_pt) if last_pt else "")

    def _on_train_done(self, success, output_path):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._update_resume_state()
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
        import subprocess
        import json
        import tempfile

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
            cfg = {"model": pt_path, "format": fmt}
            cfg_path = os.path.join(tempfile.gettempdir(), f"_export_cfg_{os.getpid()}.json")
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False)

            helper = str(Path(__file__).parent.parent / "export_helper.py")
            self.log(f"Loading model: {pt_path}")

            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            result = subprocess.run(
                ["py", helper, cfg_path],
                capture_output=True, text=False, timeout=300,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )

            stdout = result.stdout.decode('utf-8', errors='ignore')
            for line in stdout.strip().split('\n'):
                line = line.strip()
                if line:
                    self.log(line)
                if line.startswith('JSON:'):
                    try:
                        resp = json.loads(line[5:])
                        if resp.get("status") == "done":
                            output_path = resp.get("path", "")
                            self.log(f"Exported: {output_path}")
                            self.status_label.setText(t("export_done"))
                            self.status_label.setStyleSheet("color: #22c55e; font-size: 13px; font-weight: bold;")
                        else:
                            self.log(t("export_fail_msg", err=resp.get("msg", "unknown")))
                            self.status_label.setText(t("train_fail"))
                            self.status_label.setStyleSheet("color: #ef4444; font-size: 13px; font-weight: bold;")
                    except Exception:
                        pass

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

    def __init__(self, dataset_dir, model, epochs=100, batch=16, imgsz=640,
                 device="cpu", project="runs", name="run", hyperparams=None, resume=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.model = model
        self.epochs = epochs
        self.batch = batch
        self.imgsz = imgsz
        self.device = device
        self.project = project
        self.name = name
        self.hyperparams = hyperparams or {}
        self.resume = resume
        self._stop = False
        self._proc = None

    def stop(self):
        self._stop = True
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        import subprocess
        try:
            # Kill the whole tree so ultralytics worker processes die too
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=5,
            )
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def run(self):
        import subprocess
        import json
        import tempfile
        import os

        try:
            self.log_signal.emit("Generating dataset config...")
            yaml_path = self._create_yaml_config()
            self.log_signal.emit(f"  Config: {yaml_path}")

            cfg = {
                "dataset": self.dataset_dir,
                "model": self.model,
                "yaml": yaml_path,
                "epochs": self.epochs,
                "batch": self.batch,
                "imgsz": self.imgsz,
                "device": self.device,
                "project": self.project,
                "name": self.name,
                "hyper": self.hyperparams,
                "resume": self.resume,
            }
            cfg_path = os.path.join(tempfile.gettempdir(), f"_train_cfg_{os.getpid()}.json")
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False)

            helper = str(Path(__file__).parent.parent / "train_helper.py")
            self.log_signal.emit(f"Starting training via system Python...")

            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            self._proc = subprocess.Popen(
                ["py", helper, cfg_path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=False, bufsize=0,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            proc = self._proc

            while not self._stop:
                line = proc.stdout.readline()
                if not line:
                    break
                text = line.decode('utf-8', errors='ignore').strip()
                if text:
                    self.log_signal.emit(text)

            if self._stop:
                self.stop()
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

            best_weight = os.path.join(self.project, self.name, "weights", "best.pt")
            if self._stop:
                self.done_signal.emit(False, "Training stopped by user")
            else:
                self.done_signal.emit(os.path.exists(best_weight), best_weight)

        except Exception as e:
            self.log_signal.emit(f"\nError: {str(e)}")
            self.done_signal.emit(False, str(e))
        finally:
            self._proc = None

    def _create_yaml_config(self):
        from utils.dataset_parser import DatasetParser
        parser = DatasetParser()
        stats = parser.parse_and_validate(self.dataset_dir)
        if not stats.get("valid"):
            raise ValueError(f"Invalid dataset: {stats.get('error', 'unknown')}")
        num_classes = len(stats.get("classes", []))
        yaml_path = parser.create_yaml_config(
            self.dataset_dir,
            num_classes=num_classes,
            class_names=self._load_class_names(num_classes),
        )
        return yaml_path

    def _load_class_names(self, num_classes: int) -> list[str]:
        """Real class names from annotations.json (written by the annotate page).
        Falls back to class_N placeholders for externally-built datasets."""
        import json
        ann_file = os.path.join(self.dataset_dir, "annotations.json")
        if os.path.exists(ann_file):
            try:
                with open(ann_file, "r", encoding="utf-8") as f:
                    cats = json.load(f).get("categories", [])
                if len(cats) == num_classes:
                    return cats
            except (OSError, ValueError):
                pass
        return [f"class_{i}" for i in range(num_classes)]
#唧唧复唧唧著
