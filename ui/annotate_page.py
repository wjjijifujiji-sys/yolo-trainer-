"""Annotation page - image labeling with rectangle drawing."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFileDialog, QListWidget, QListWidgetItem,
    QComboBox, QSpinBox, QMessageBox, QSplitter,
    QInputDialog, QGraphicsTextItem, QGraphicsView, QGraphicsScene,
    QFrame,
)
from PyQt6.QtCore import Qt, QRectF, pyqtSignal as Signal
from PyQt6.QtGui import QPixmap, QPen, QColor, QFont, QPainter

from ui.components import styled_button
from utils.annotations import Annotation, AnnotationManager
from utils.i18n import t


class AnnotationCanvas(QGraphicsView):
    """Custom QGraphicsView that handles rectangle drawing via mouse."""

    rect_drawing = Signal(object)

    def __init__(self, scene):
        super().__init__(scene)
        self._start_point = None
        self._dragging = False
        self._rubber_band = None
        self._pixmap_item = None

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

    def wheelEvent(self, event):
        factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(factor, factor)
        else:
            self.scale(1 / factor, 1 / factor)

    def load_image(self, filepath):
        pixmap = QPixmap(filepath)
        if self._pixmap_item:
            self.scene().removeItem(self._pixmap_item)
        self._pixmap_item = self.scene().addPixmap(pixmap)
        self.scene().setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        return pixmap.width(), pixmap.height()

    def start_draw(self, scene_point):
        self._start_point = scene_point
        self._dragging = True
        pen = QPen(QColor(249, 115, 22))
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        self._rubber_band = self.scene().addRect(QRectF(scene_point, scene_point), pen)

    def update_draw(self, scene_point):
        if not self._dragging or not self._rubber_band:
            return
        rect = QRectF(self._start_point, scene_point).normalized()
        self._rubber_band.setRect(rect)

    def finish_draw(self):
        if not self._dragging or not self._rubber_band:
            return None, True
        self._dragging = False
        rect = self._rubber_band.rect()
        self.scene().removeItem(self._rubber_band)
        self._rubber_band = None
        if rect.width() < 3 or rect.height() < 3:
            return None, True
        return rect, False

    def cancel_draw(self):
        if self._rubber_band:
            self.scene().removeItem(self._rubber_band)
            self._rubber_band = None
        self._dragging = False

    def add_rect(self, rect, class_id, categories):
        colors = [
            "#ef4444", "#22c55e", "#3b82f6", "#eab308", "#06b6d4",
            "#d946ef", "#f97316", "#8b5cf6", "#14b8a6", "#ec4899",
        ]
        color = QColor(colors[class_id % len(colors)])
        pen = QPen(color, 2)
        self.scene().addRect(rect, pen)
        label = categories[class_id] if class_id < len(categories) else str(class_id)

        bg = QColor(color)
        bg.setAlpha(180)
        text_item = QGraphicsTextItem(f" {label} ")
        text_item.setDefaultTextColor(QColor("#fff"))
        text_item.setFont(QFont("Microsoft YaHei UI", 9, QFont.Weight.Bold))
        text_item.setPos(rect.left(), rect.top() - 22)
        self.scene().addItem(text_item)

    def clear_rects(self):
        for item in list(self.scene().items()):
            if item != self._pixmap_item:
                self.scene().removeItem(item)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        if not self._pixmap_item:
            return
        self.start_draw(self.mapToScene(event.position().toPoint()))

    def mouseMoveEvent(self, event):
        if not self._dragging:
            super().mouseMoveEvent(event)
            return
        self.update_draw(self.mapToScene(event.position().toPoint()))

    def mouseReleaseEvent(self, event):
        if not self._dragging:
            super().mouseReleaseEvent(event)
            return
        rect, ignore = self.finish_draw()
        if ignore or rect is None:
            return
        self.rect_drawing.emit(rect)


class AnnotatePage(QWidget):
    """Full annotation page with image browsing, rectangle drawing, and YOLO export."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.manager = AnnotationManager()
        self._current_image_idx = 0
        self._image_files = []
        self._current_image_dir = None
        self._current_filename = None

        self._setup_ui()
        self._refresh_category_combo()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # ── Top toolbar ──
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.import_btn = styled_button(t("import_folder"), "#f97316", 13)
        self.import_btn.clicked.connect(self._import_folder)
        toolbar.addWidget(self.import_btn)

        toolbar.addSpacing(16)

        cat_label = QLabel(t("class_list") + ":")
        cat_label.setObjectName("sectionTitleSmall")
        toolbar.addWidget(cat_label)

        self.cat_combo = QComboBox()
        self.cat_combo.setMinimumWidth(120)
        self.cat_combo.setMaximumWidth(200)
        toolbar.addWidget(self.cat_combo)

        self.add_cat_btn = styled_button(t("add"), "#22c55e", 12)
        self.add_cat_btn.setMaximumWidth(80)
        self.add_cat_btn.clicked.connect(self._add_category)
        toolbar.addWidget(self.add_cat_btn)

        self.del_cat_btn = styled_button(t("remove"), "#ef4444", 12)
        self.del_cat_btn.setMaximumWidth(80)
        self.del_cat_btn.clicked.connect(self._remove_category)
        toolbar.addWidget(self.del_cat_btn)

        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        # ── Main content: splitter ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        # Left: canvas + nav
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.scene = QGraphicsScene()
        self.canvas = AnnotationCanvas(self.scene)
        self.canvas.rect_drawing.connect(self._on_rect_drawn)
        left_layout.addWidget(self.canvas, 1)

        # Navigation bar
        nav = QHBoxLayout()
        nav.setSpacing(6)
        self.prev_btn = styled_button(t("prev"), "#6b7280", 12)
        self.prev_btn.setMaximumWidth(100)
        self.prev_btn.setEnabled(False)
        self.prev_btn.clicked.connect(self._prev_image)
        nav.addWidget(self.prev_btn)

        self.next_btn = styled_button(t("next"), "#6b7280", 12)
        self.next_btn.setMaximumWidth(100)
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self._next_image)
        nav.addWidget(self.next_btn)

        self.delete_btn = styled_button(t("delete_img"), "#ef4444", 12)
        self.delete_btn.setMaximumWidth(100)
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self._delete_current)
        nav.addWidget(self.delete_btn)

        nav.addStretch()

        self.img_counter = QLabel("")
        self.img_counter.setObjectName("hint")
        nav.addWidget(self.img_counter)
        left_layout.addLayout(nav)

        splitter.addWidget(left_widget)

        # Right: control panel
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(10)

        # Class list section
        section1 = QLabel(t("class_list"))
        section1.setObjectName("sectionTitle")
        right_layout.addWidget(section1)

        self.cat_list = QListWidget()
        self.cat_list.setMaximumWidth(320)
        right_layout.addWidget(self.cat_list)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet("color: #374151;")
        right_layout.addWidget(div)

        # Annotations section
        section2 = QLabel(t("current_annotations"))
        section2.setObjectName("sectionTitle")
        right_layout.addWidget(section2)

        self.ann_list = QListWidget()
        self.ann_list.setMaximumWidth(320)
        self.ann_list.itemDoubleClicked.connect(self._remove_annotation_by_click)
        right_layout.addWidget(self.ann_list)

        ann_btns = QHBoxLayout()
        ann_btns.setSpacing(6)
        self.del_ann_btn = styled_button(t("delete_selected"), "#ef4444", 11)
        self.del_ann_btn.clicked.connect(self._remove_selected_annotation)
        ann_btns.addWidget(self.del_ann_btn)

        self.clear_ann_btn = styled_button(t("clear_all"), "#6b7280", 11)
        self.clear_ann_btn.clicked.connect(self._clear_current_annotations)
        ann_btns.addWidget(self.clear_ann_btn)
        right_layout.addLayout(ann_btns)

        right_layout.addStretch()
        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1000, 300])
        main_layout.addWidget(splitter, 1)

        # ── Bottom export bar ──
        export_bar = QHBoxLayout()
        export_bar.setSpacing(8)

        ratio_label = QLabel(t("val_ratio"))
        ratio_label.setObjectName("sectionTitleSmall")
        export_bar.addWidget(ratio_label)

        self.val_ratio_spin = QSpinBox()
        self.val_ratio_spin.setRange(5, 50)
        self.val_ratio_spin.setValue(20)
        self.val_ratio_spin.setSuffix("%")
        self.val_ratio_spin.setSingleStep(5)
        self.val_ratio_spin.setMaximumWidth(80)
        export_bar.addWidget(self.val_ratio_spin)

        export_bar.addStretch()

        self.export_btn = styled_button(t("export_yolo"), "#f97316", 14)
        self.export_btn.setMinimumWidth(200)
        self.export_btn.setMinimumHeight(42)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_yolo)
        export_bar.addWidget(self.export_btn)
        main_layout.addLayout(export_bar)

    # ─── i18n ───

    def retranslate(self):
        self.import_btn.setText(t("import_folder"))
        self.export_btn.setText(t("export_yolo"))
        self.prev_btn.setText(t("prev"))
        self.next_btn.setText(t("next"))
        self.delete_btn.setText(t("delete_img"))
        self.add_cat_btn.setText(t("add"))
        self.del_cat_btn.setText(t("remove"))
        self.del_ann_btn.setText(t("delete_selected"))
        self.clear_ann_btn.setText(t("clear_all"))
        # Section labels
        for lbl in self.findChildren(QLabel):
            if lbl.text() in ("类别列表", "Classes", "当前图片标注", "Annotations"):
                if lbl.text() in ("类别列表", "Classes"):
                    lbl.setText(t("class_list"))
                elif lbl.text() in ("当前图片标注", "Annotations"):
                    lbl.setText(t("current_annotations"))

    # ─── Category management ───

    def _add_category(self):
        name, ok = QInputDialog.getText(self, t("add_category"), t("category_name"))
        if ok and name.strip():
            idx = self.manager.add_category(name.strip())
            self._refresh_category_combo()
            self._refresh_cat_list()
            self.cat_combo.setCurrentIndex(idx)

    def _remove_category(self):
        row = self.cat_list.currentRow()
        if row < 0:
            return
        if self.manager.remove_category(row):
            self._refresh_category_combo()
            self._refresh_cat_list()
            self._refresh_annotation_list()

    def _refresh_category_combo(self):
        current = self.cat_combo.currentText()
        self.cat_combo.clear()
        self.cat_combo.addItems(self.manager.categories)
        if current in self.manager.categories:
            self.cat_combo.setCurrentText(current)

    def _refresh_cat_list(self):
        self.cat_list.clear()
        for name in self.manager.categories:
            self.cat_list.addItem(name)

    # ─── Image navigation ───

    def _import_folder(self):
        folder = QFileDialog.getExistingDirectory(self, t("import_folder"))
        if not folder:
            return

        self._current_image_dir = Path(folder)
        supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self._image_files = sorted(
            f for f in self._current_image_dir.iterdir()
            if f.is_file() and f.suffix.lower() in supported
        )

        if not self._image_files:
            QMessageBox.warning(self, t("title"), t("no_image"))
            return

        self._current_image_idx = 0
        self._enable_navigation(True)
        self._load_current_image()

    def _enable_navigation(self, enabled):
        self.prev_btn.setEnabled(enabled)
        self.next_btn.setEnabled(enabled)
        self.delete_btn.setEnabled(enabled)
        self.export_btn.setEnabled(enabled)

    def _load_current_image(self):
        if not self._image_files:
            return
        img_path = self._image_files[self._current_image_idx]
        self._current_filename = img_path.name
        self.canvas.clear_rects()
        self.canvas.load_image(str(img_path))
        self._restore_annotations()
        self._refresh_annotation_list()
        self.img_counter.setText(
            f"{self._current_image_idx + 1}/{len(self._image_files)}  {self._current_filename}"
        )

    def _prev_image(self):
        if self._current_image_idx > 0:
            self._current_image_idx -= 1
            self._load_current_image()

    def _next_image(self):
        if self._current_image_idx < len(self._image_files) - 1:
            self._current_image_idx += 1
            self._load_current_image()

    def _delete_current(self):
        if not self._image_files:
            return
        reply = QMessageBox.question(
            self, t("confirm_delete"),
            t("confirm_delete_msg", name=self._current_filename),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._image_files.pop(self._current_image_idx)
            self.manager.annotations.pop(self._current_filename, None)
            if not self._image_files:
                self._enable_navigation(False)
                self.canvas.clear_rects()
                self.img_counter.setText("")
                return
            if self._current_image_idx >= len(self._image_files):
                self._current_image_idx = len(self._image_files) - 1
            self._load_current_image()

    # ─── Drawing interaction ───

    def _on_rect_drawn(self, rect):
        class_id = self.cat_combo.currentIndex()
        if class_id >= len(self.manager.categories):
            QMessageBox.warning(self, t("title"), t("no_category"))
            return

        ann = Annotation(
            class_id=class_id,
            x=rect.x(), y=rect.y(), w=rect.width(), h=rect.height(),
        )
        self.manager.add_annotation(self._current_filename, ann)
        self.canvas.add_rect(rect, class_id, self.manager.categories)
        self._refresh_annotation_list()

    # ─── Annotation list ───

    def _restore_annotations(self):
        anns = self.manager.get_annotations(self._current_filename)
        for ann in anns:
            rect = QRectF(ann.x, ann.y, ann.w, ann.h)
            self.canvas.add_rect(rect, ann.class_id, self.manager.categories)

    def _refresh_annotation_list(self):
        self.ann_list.clear()
        anns = self.manager.get_annotations(self._current_filename)
        for i, ann in enumerate(anns):
            cat = self.manager.categories[ann.class_id] if ann.class_id < len(self.manager.categories) else "?"
            item = QListWidgetItem(f"[{i}] {cat}  ({ann.x:.0f}, {ann.y:.0f})  {ann.w:.0f}×{ann.h:.0f}")
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.ann_list.addItem(item)

    def _remove_annotation_by_click(self, item):
        idx = item.data(Qt.ItemDataRole.UserRole)
        self._remove_annotation_at(idx)

    def _remove_selected_annotation(self):
        row = self.ann_list.currentRow()
        if row >= 0:
            self._remove_annotation_at(row)

    def _remove_annotation_at(self, index):
        anns = self.manager.get_annotations(self._current_filename)
        if 0 <= index < len(anns):
            anns.pop(index)
            self.manager.annotations[self._current_filename] = anns
            self._restore_annotations()
            self._refresh_annotation_list()

    def _clear_current_annotations(self):
        if self._current_filename:
            self.manager.annotations.pop(self._current_filename, None)
            self.canvas.clear_rects()
            img_path = self._image_files[self._current_image_idx]
            self.canvas.load_image(str(img_path))
            self._refresh_annotation_list()

    # ─── Export ───

    def _export_yolo(self):
        if not self._image_files:
            QMessageBox.warning(self, t("title"), t("no_image"))
            return
        if not self.manager.categories:
            QMessageBox.warning(self, t("title"), t("no_category"))
            return

        out_folder = QFileDialog.getExistingDirectory(self, t("export_yolo"))
        if not out_folder:
            return

        val_ratio = self.val_ratio_spin.value() / 100.0
        try:
            yaml_path = self.manager.export_yolo(
                image_folder=str(self._current_image_dir),
                output_dir=out_folder,
                val_ratio=val_ratio,
            )
            QMessageBox.information(
                self, t("export_success"),
                t("export_info", path=out_folder, cats=self.manager.categories,
                  count=len(self._image_files), yaml=yaml_path),
            )
        except Exception as e:
            QMessageBox.critical(self, t("export_fail"), str(e))
