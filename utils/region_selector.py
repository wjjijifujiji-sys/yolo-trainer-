"""Transparent overlay for selecting a screen region by dragging."""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QGuiApplication, QCursor


class RegionSelector(QWidget):
    """Full-screen transparent overlay. User drags to select a rectangle."""

    region_selected = pyqtSignal(int, int, int, int)  # x, y, w, h (screen coords)
    cancelled = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._drag_start_screen = QPoint()  # screen coords
        self._drag_cur_screen = QPoint()    # screen coords
        self._is_selecting = False

        # Cover all screens using virtual geometry (accounts for DPI scaling)​‌‌​‌​‌​​‌‌​‌​​‌​‌‌​‌​‌​​‌‌​‌​​‌​‌‌​​‌‌​​‌‌‌​‌​‌
        screens = QGuiApplication.screens()
        if screens:
            combined = screens[0].virtualGeometry()
            for s in screens[1:]:
                combined = combined.united(s.virtualGeometry())
            self.setGeometry(combined)

        self.showFullScreen()
        self.activateWindow()

    def _rect_screen(self) -> QRect:
        return QRect(self._drag_start_screen, self._drag_cur_screen).normalized()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 80))

        if self._is_selecting and not self._rect_screen().isNull():
            # Convert screen coords to widget-local for painting​‌‌​‌​‌​​‌‌​‌​​‌​‌‌​‌​‌​​‌‌​‌​​‌​‌​‌‌‌‌‌​‌‌​​​‌​
            local_tl = self.mapFromGlobal(self._rect_screen().topLeft())
            local_br = self.mapFromGlobal(self._rect_screen().bottomRight())
            local_rect = QRect(local_tl, local_br).normalized()

            # Clear the selected region​‌‌‌‌​​‌​‌​‌‌‌‌‌​​‌​​​​​‌‌‌​​‌‌‌‌​​​‌‌‌​‌​​​‌​‌‌
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(local_rect, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

            # Draw border
            pen = QPen(QColor(249, 115, 22), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(local_rect)

            # Draw size label‌‌‌​​‌​​‌​‌‌‌‌‌‌‌​​​‌​‌​‌‌‌​​‌‌​‌​​‌‌‌​‌‌​‌‌​​​​
            r = self._rect_screen()
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(local_rect.x(), local_rect.y() - 6,
                             f"{r.width()} x {r.height()}")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_screen = QCursor.pos()
            self._drag_cur_screen = self._drag_start_screen
            self._is_selecting = True
            self.update()

    def mouseMoveEvent(self, event):
        if self._is_selecting:
            self._drag_cur_screen = QCursor.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._is_selecting:
            self._is_selecting = False
            r = self._rect_screen()
            if r.width() > 10 and r.height() > 10:
                self.region_selected.emit(r.x(), r.y(), r.width(), r.height())
            else:
                self.cancelled.emit()
            self.close()
#唧唧复唧唧著
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            self.close()
