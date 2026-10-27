"""
Interactive zoomable/pannable image canvas with drag-and-drop support
and helper conversions from ImageContainer to QPixmap.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from config.settings import SUPPORTED_EXTENSIONS
from core.image_io import ImageContainer


def container_to_qpixmap(container: ImageContainer | None) -> QPixmap:
    """Convert an ImageContainer to a QPixmap safely without memory aliasing."""
    if container is None:
        return QPixmap()

    if container.has_alpha:
        rgba = np.ascontiguousarray(container.to_rgba())
        h, w, _ = rgba.shape
        qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        return QPixmap.fromImage(qimg)

    rgb = np.ascontiguousarray(container.to_rgb())
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class InteractiveImageCanvas(QWidget):
    """
    Single zoomable and pannable image canvas widget supporting file Drag-and-Drop.
    Also emits view transformation changes so paired canvases can stay synchronized.
    """

    file_dropped = Signal(str)
    open_requested = Signal()
    view_changed = Signal(float, QPointF)  # (zoom_factor, pan_offset)

    def __init__(self, label_tag: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label_tag = label_tag
        self._pixmap = QPixmap()
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._dragging = False
        self._last_mouse_pos = QPointF(0.0, 0.0)
        self.setAcceptDrops(True)
        self.setMinimumSize(260, 220)
        self.setMouseTracking(True)

    def set_pixmap(self, pixmap: QPixmap, fit: bool = True) -> None:
        self._pixmap = pixmap
        if fit and not self._pixmap.isNull():
            self.fit_to_window()
        else:
            self.update()

    def has_image(self) -> bool:
        return not self._pixmap.isNull()

    def set_view_state(self, zoom: float, pan: QPointF) -> None:
        """Synchronize zoom and pan without re-emitting view_changed."""
        self._zoom = float(np.clip(zoom, 0.05, 16.0))
        self._pan = QPointF(pan)
        self.update()

    def fit_to_window(self) -> None:
        if self._pixmap.isNull() or self.width() <= 10 or self.height() <= 10:
            self._zoom = 1.0
            self._pan = QPointF(0.0, 0.0)
            self.update()
            return
        pw = self._pixmap.width()
        ph = self._pixmap.height()
        scale_x = (self.width() - 24) / max(1, pw)
        scale_y = (self.height() - 24) / max(1, ph)
        self._zoom = float(np.clip(min(scale_x, scale_y), 0.05, 4.0))
        self._pan = QPointF(0.0, 0.0)
        self.view_changed.emit(self._zoom, self._pan)
        self.update()

    def reset_zoom_100(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.view_changed.emit(self._zoom, self._pan)
        self.update()

    def zoom_by(self, factor: float) -> None:
        self._zoom = float(np.clip(self._zoom * factor, 0.05, 16.0))
        self.view_changed.emit(self._zoom, self._pan)
        self.update()

    @property
    def zoom_factor(self) -> float:
        return self._zoom

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                local = url.toLocalFile()
                if local and Path(local).suffix.lower() in SUPPORTED_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                local = url.toLocalFile()
                if local and Path(local).suffix.lower() in SUPPORTED_EXTENSIONS:
                    self.file_dropped.emit(local)
                    event.acceptProposedAction()
                    return

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._pixmap.isNull():
            return
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        self.zoom_by(factor)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._pixmap.isNull() and event.button() == Qt.MouseButton.LeftButton:
            self.open_requested.emit()
            return
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._dragging = True
            self._last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging and not self._pixmap.isNull():
            delta = event.position() - self._last_mouse_pos
            self._last_mouse_pos = event.position()
            self._pan += delta
            self.view_changed.emit(self._zoom, self._pan)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._zoom < 2.5)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect()
        painter.fillRect(rect, QColor("#10131A"))

        if self._pixmap.isNull():
            pen = QPen(QColor("#363D52"), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            drop_rect = rect.adjusted(24, 24, -24, -24)
            painter.drawRoundedRect(drop_rect, 12, 12)

            painter.setPen(QColor("#9CA3AF"))
            painter.drawText(
                drop_rect,
                Qt.AlignmentFlag.AlignCenter,
                "Kéo & Thả ảnh vào đây (JPG, PNG, WEBP)\nhoặc Nhấn vào khung để chọn ảnh từ máy tính",
            )
            return

        draw_w = self._pixmap.width() * self._zoom
        draw_h = self._pixmap.height() * self._zoom
        cx = rect.width() / 2.0 + self._pan.x()
        cy = rect.height() / 2.0 + self._pan.y()
        target_rect = QRectF(cx - draw_w / 2.0, cy - draw_h / 2.0, draw_w, draw_h)

        painter.drawPixmap(target_rect, self._pixmap, QRectF(self._pixmap.rect()))

        if self.label_tag:
            badge_rect = QRectF(12, 12, 140, 26)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(15, 17, 23, 200))
            painter.drawRoundedRect(badge_rect, 6, 6)
            painter.setPen(QColor("#E5E7EB"))
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, self.label_tag)
