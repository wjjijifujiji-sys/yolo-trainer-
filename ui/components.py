"""Shared UI components - styled buttons, widgets."""

from PyQt6.QtWidgets import QPushButton, QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtProperty, QSize
from PyQt6.QtGui import QFont


# ─── Global Stylesheet ───

STYLESHEET = """
QMainWindow {
    background: #111827;
}
QWidget {
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    color: #e5e7eb;
}

/* ── Tab Bar ── */
QTabBar::tab {
    background: #1f2937;
    color: #9ca3af;
    padding: 14px 40px;
    font-size: 14px;
    font-weight: bold;
    border: none;
    border-bottom: 3px solid transparent;
    margin-right: 0px;
}
QTabBar::tab:selected {
    background: #1f2937;
    color: #f97316;
    border-bottom: 3px solid #f97316;
}
QTabBar::tab:hover {
    background: #374151;
    color: #d1d5db;
}

/* ── Menu Bar ── */
QMenuBar {
    background: #111827;
    color: #d1d5db;
    border-bottom: 1px solid #1f2937;
    padding: 2px;
    font-size: 13px;
}
QMenuBar::item {
    padding: 6px 14px;
    border-radius: 6px;
}
QMenuBar::item:selected {
    background: #374151;
}
QMenu {
    background: #1f2937;
    color: #e5e7eb;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 4px;
}
QMenu::item {
    padding: 8px 20px;
    border-radius: 4px;
}
QMenu::item:selected {
    background: #f97316;
    color: #fff;
}

/* ── ScrollBar ── */
QScrollBar:vertical {
    background: #111827;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #4b5563;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #6b7280;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    background: #111827;
    height: 8px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal {
    background: #4b5563;
    border-radius: 4px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background: #6b7280;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* ── Splitter ── */
QSplitter::handle {
    background: #374151;
    width: 2px;
    height: 2px;
}
QSplitter::handle:hover {
    background: #f97316;
}

/* ── GraphicsView (canvas) ── */
QGraphicsView {
    background: #0c1017;
    border: 1px solid #1f2937;
    border-radius: 8px;
}

/* ── ListWidget ── */
QListWidget {
    background: #111827;
    color: #d1d5db;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 4px;
    font-size: 12px;
    outline: none;
}
QListWidget::item {
    padding: 6px 8px;
    border-radius: 4px;
    margin: 1px 2px;
}
QListWidget::item:selected {
    background: #f97316;
    color: #fff;
}
QListWidget::item:hover {
    background: #374151;
}

/* ── ComboBox ── */
QComboBox {
    background: #1f2937;
    color: #e5e7eb;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
    min-width: 100px;
}
QComboBox:hover {
    border-color: #f97316;
}
QComboBox::drop-down {
    border: none;
    width: 28px;
}
QComboBox::down-arrow {
    image: none;
    border: none;
}
QComboBox QAbstractItemView {
    background: #1f2937;
    color: #e5e7eb;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 4px;
    selection-background-color: #f97316;
    selection-color: #fff;
}

/* ── SpinBox ── */
QSpinBox, QDoubleSpinBox {
    background: #1f2937;
    color: #e5e7eb;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
    min-width: 80px;
    selection-background-color: #f97316;
    selection-color: #fff;
}
QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #f97316;
}
QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #f97316;
    color: #f97316;
}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background: #374151;
    border: none;
    width: 20px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: #4b5563;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow,
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: none;
    border: none;
}

/* ── ProgressBar ── */
QProgressBar {
    background: #1f2937;
    border: none;
    border-radius: 6px;
    text-align: center;
    color: #fff;
    font-weight: bold;
    font-size: 12px;
    max-height: 14px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #f97316, stop:1 #fb923c);
    border-radius: 6px;
}

/* ── QTextEdit ── */
QTextEdit {
    background: #0c1017;
    color: #9ca3af;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 8px;
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 12px;
}

/* ── QLabel accents ── */
QLabel#sectionTitle {
    color: #f97316;
    font-size: 14px;
    font-weight: bold;
    padding: 4px 0px;
}
QLabel#sectionTitleSmall {
    color: #9ca3af;
    font-size: 12px;
    font-weight: bold;
}
QLabel#hint {
    color: #6b7280;
    font-size: 12px;
}
"""


def styled_button(text, color="#f97316", size=13, bold=True):
    """Create a styled button with hover/press effects."""
    btn = HoverButton(text)
    btn._base_color = color
    weight = "bold" if bold else "normal"
    btn.setStyleSheet(f"""
        HoverButton {{
            background: {color};
            color: #fff;
            border: none;
            padding: 8px 18px;
            border-radius: 8px;
            font-size: {size}px;
            font-weight: {weight};
        }}
    """)
    btn.setMinimumHeight(36)
    return btn


class HoverButton(QPushButton):
    """Button with scale animation on press and color shift on hover."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._anim = None
        self._scale = 1.0

    def _get_scale(self):
        return self._scale

    def _set_scale(self, val):
        self._scale = val
        s = val
        pad_v = int(8 * s)
        pad_h = int(18 * s)
        size = int(self.font().pointSize() * s) if self.font().pointSize() > 0 else 13
        color = getattr(self, '_base_color', '#f97316')
        self.setStyleSheet(f"""
            HoverButton {{
                background: {color};
                color: #fff;
                border: none;
                padding: {pad_v}px {pad_h}px;
                border-radius: 8px;
                font-size: {size}px;
                font-weight: bold;
            }}
        """)

    scale_prop = pyqtProperty(float, _get_scale, _set_scale)

    def mousePressEvent(self, event):
        self._anim = QPropertyAnimation(self, b"scale_prop")
        self._anim.setDuration(80)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.95)
        self._anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        self._anim.start()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self._anim = QPropertyAnimation(self, b"scale_prop")
        self._anim.setDuration(120)
        self._anim.setStartValue(0.95)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.OutBack)
        self._anim.start()
        super().mouseReleaseEvent(event)
