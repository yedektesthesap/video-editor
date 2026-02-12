
from __future__ import annotations

import colorsys
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QSpinBox,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from event_detector import EVENT_DEFINITIONS, REQUIRED_TARGET_ROIS
from event_worker import EventDetectionWorker, RoiColorAnalysisWorker

MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 1200
MIN_RELATIVE_SIZE = 0.002
MAX_SHORTCUT_ROIS = 9
TEMPLATE_VERSION = "1.1"

SETTINGS_ORGANIZATION = "YSN"
SETTINGS_APPLICATION = "VideoEditorROI"
SETTINGS_LAST_TEMPLATE_PATH = "last_template_path"
SETTINGS_LAST_VIDEO_DIR = "last_video_dir"
EVENT_COL_START = 4
EVENT_COL_END = 5
DETECTION_MODE_AUTO = "auto"
DETECTION_MODE_MANUAL = "manual"

SOURCE_START_SENSITIVITY_PRESETS: dict[str, dict[str, float]] = {
    "Hassas": {
        "source_color_k": 3.0,
        "source_color_min": 1.4,
        "source_consecutive": 1.0,
        "source_soft_ratio": 0.7,
    },
    "Dengeli": {
        "source_color_k": 4.0,
        "source_color_min": 1.8,
        "source_consecutive": 2.0,
        "source_soft_ratio": 0.8,
    },
    "Muhafazakar": {
        "source_color_k": 5.0,
        "source_color_min": 2.3,
        "source_consecutive": 3.0,
        "source_soft_ratio": 0.9,
    },
}

DARK_STYLESHEET = """
QWidget {
    background-color: #15171c;
    color: #e5e7eb;
    selection-background-color: #2f78d4;
    selection-color: #ffffff;
}

QMainWindow {
    background-color: #15171c;
}

QGroupBox {
    border: 1px solid #313744;
    border-radius: 8px;
    margin-top: 0.8em;
    padding-top: 0.6em;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #f4f5f7;
}

QPushButton {
    background-color: #2a303a;
    color: #f4f5f7;
    border: 1px solid #3d4554;
    border-radius: 6px;
    padding: 6px 10px;
}

QPushButton:hover {
    background-color: #333b47;
}

QPushButton:pressed {
    background-color: #222831;
}

QPushButton:disabled {
    background-color: #1f242c;
    color: #818897;
    border-color: #2e3440;
}

QLineEdit,
QPlainTextEdit,
QListWidget,
QTableWidget,
QSpinBox {
    background-color: #1b2028;
    border: 1px solid #313744;
    border-radius: 5px;
    padding: 4px;
}

QHeaderView::section {
    background-color: #2a303a;
    color: #edf0f4;
    border: 1px solid #3d4554;
    padding: 4px;
}

QProgressBar {
    border: 1px solid #3d4554;
    border-radius: 6px;
    background: #1b2028;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #2f8f4e;
    border-radius: 5px;
}

QScrollBar:vertical {
    background: #15171c;
    width: 12px;
    margin: 2px;
}

QScrollBar::handle:vertical {
    background: #3a414f;
    border-radius: 5px;
    min-height: 20px;
}

QScrollBar:horizontal {
    background: #15171c;
    height: 12px;
    margin: 2px;
}

QScrollBar::handle:horizontal {
    background: #3a414f;
    border-radius: 5px;
    min-width: 20px;
}
"""


@dataclass(frozen=True)
class RelativeRect:
    x: float
    y: float
    w: float
    h: float

    def to_dict(self) -> dict:
        return {"x": float(self.x), "y": float(self.y), "w": float(self.w), "h": float(self.h)}


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_index: int
    source_video: str


@dataclass(frozen=True)
class RoiStats:
    pixel_rect: Tuple[int, int, int, int]
    relative_rect: RelativeRect
    preview_pixmap: QPixmap


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_relative_rect(x: float, y: float, w: float, h: float) -> Optional[RelativeRect]:
    x0 = min(float(x), float(x) + float(w))
    y0 = min(float(y), float(y) + float(h))
    x1 = max(float(x), float(x) + float(w))
    y1 = max(float(y), float(y) + float(h))

    x0 = clamp01(x0)
    y0 = clamp01(y0)
    x1 = clamp01(x1)
    y1 = clamp01(y1)

    if x1 <= x0 or y1 <= y0:
        return None

    return RelativeRect(x=x0, y=y0, w=x1 - x0, h=y1 - y0)


def parse_relative_rect(raw: object) -> Optional[RelativeRect]:
    if not isinstance(raw, dict):
        return None
    try:
        return normalize_relative_rect(
            x=float(raw["x"]),
            y=float(raw["y"]),
            w=float(raw["w"]),
            h=float(raw["h"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def relative_to_pixel_rect(rect: RelativeRect, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if width <= 0 or height <= 0:
        return None

    x0 = int(np.floor(clamp01(rect.x) * width))
    y0 = int(np.floor(clamp01(rect.y) * height))
    x1 = int(np.ceil(clamp01(rect.x + rect.w) * width))
    y1 = int(np.ceil(clamp01(rect.y + rect.h) * height))

    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))

    w = x1 - x0
    h = y1 - y0
    if w <= 0 or h <= 0:
        return None

    return x0, y0, w, h


def format_relative_rect(rect: Optional[RelativeRect]) -> str:
    if rect is None:
        return "-"
    return f"x={rect.x:.6f}, y={rect.y:.6f}, w={rect.w:.6f}, h={rect.h:.6f}"


def format_pixel_rect(rect: Optional[Tuple[int, int, int, int]]) -> str:
    if rect is None:
        return "-"
    x, y, w, h = rect
    return f"x={x}, y={y}, w={w}, h={h}"


def format_time_dk_sn_ms(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"

    try:
        total_ms = int(round(float(seconds) * 1000.0))
    except (TypeError, ValueError):
        return "-"

    total_ms = max(0, total_ms)
    minutes, remainder = divmod(total_ms, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{minutes:02d} dk {whole_seconds:02d} sn {millis:03d} ms"


def roi_color(roi_name: str) -> QColor:
    digest = hashlib.sha1(roi_name.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") % 360
    sat = 0.65 + (digest[2] / 255.0) * 0.20
    val = 0.85 + (digest[3] / 255.0) * 0.10
    red, green, blue = colorsys.hsv_to_rgb(hue / 360.0, sat, val)
    return QColor(int(red * 255), int(green * 255), int(blue * 255))


def numpy_rgb_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    rgb_contiguous = np.ascontiguousarray(rgb)
    height, width, channels = rgb_contiguous.shape
    if channels != 3:
        raise ValueError("RGB image must have 3 channels")

    qimage = QImage(
        rgb_contiguous.data,
        width,
        height,
        channels * width,
        QImage.Format.Format_RGB888,
    )
    return QPixmap.fromImage(qimage.copy())

class VideoCanvas(QWidget):
    roi_drawn = pyqtSignal(str, object)
    draw_requested_without_selection = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._image_pixmap: Optional[QPixmap] = None
        self._rois_rel: Dict[str, RelativeRect] = {}
        self._active_roi: Optional[str] = None
        self._drag_start: Optional[QPointF] = None
        self._drag_current: Optional[QPointF] = None

        self.setMouseTracking(True)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        self._image_pixmap = pixmap
        self.update()

    def set_rois(self, rois_rel: Dict[str, RelativeRect]) -> None:
        self._rois_rel = dict(rois_rel)
        self.update()

    def set_active_roi(self, roi_name: Optional[str]) -> None:
        self._active_roi = roi_name
        self.update()

    def _image_draw_rect(self) -> QRectF:
        if self._image_pixmap is None or self._image_pixmap.isNull():
            return QRectF()

        pix_w = float(self._image_pixmap.width())
        pix_h = float(self._image_pixmap.height())
        if pix_w <= 0.0 or pix_h <= 0.0:
            return QRectF()

        widget_w = float(self.width())
        widget_h = float(self.height())
        if widget_w <= 0.0 or widget_h <= 0.0:
            return QRectF()

        scale = min(widget_w / pix_w, widget_h / pix_h)
        draw_w = pix_w * scale
        draw_h = pix_h * scale
        draw_x = (widget_w - draw_w) / 2.0
        draw_y = (widget_h - draw_h) / 2.0
        return QRectF(draw_x, draw_y, draw_w, draw_h)

    def _widget_to_normalized(self, pos: QPointF, clamp_to_bounds: bool) -> Optional[QPointF]:
        rect = self._image_draw_rect()
        if rect.isEmpty():
            return None

        nx = (pos.x() - rect.x()) / rect.width()
        ny = (pos.y() - rect.y()) / rect.height()

        if clamp_to_bounds:
            nx = clamp01(nx)
            ny = clamp01(ny)
        elif nx < 0.0 or nx > 1.0 or ny < 0.0 or ny > 1.0:
            return None

        return QPointF(nx, ny)

    def _relative_to_canvas_rect(self, rel_rect: RelativeRect, draw_rect: QRectF) -> QRectF:
        return QRectF(
            draw_rect.x() + rel_rect.x * draw_rect.width(),
            draw_rect.y() + rel_rect.y * draw_rect.height(),
            rel_rect.w * draw_rect.width(),
            rel_rect.h * draw_rect.height(),
        )

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton or self._image_pixmap is None:
            return

        point = self._widget_to_normalized(event.position(), clamp_to_bounds=False)
        if point is None:
            return

        if self._active_roi is None:
            self.draw_requested_without_selection.emit()
            return

        self._drag_start = point
        self._drag_current = point
        self.update()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._drag_start is None:
            return

        point = self._widget_to_normalized(event.position(), clamp_to_bounds=True)
        if point is None:
            return

        self._drag_current = point
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton or self._drag_start is None or self._active_roi is None:
            return

        end_point = self._widget_to_normalized(event.position(), clamp_to_bounds=True)
        if end_point is None:
            self._drag_start = None
            self._drag_current = None
            self.update()
            return

        rect = normalize_relative_rect(
            x=self._drag_start.x(),
            y=self._drag_start.y(),
            w=end_point.x() - self._drag_start.x(),
            h=end_point.y() - self._drag_start.y(),
        )

        if rect is not None and rect.w >= MIN_RELATIVE_SIZE and rect.h >= MIN_RELATIVE_SIZE:
            self.roi_drawn.emit(self._active_roi, rect)

        self._drag_start = None
        self._drag_current = None
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 20))

        draw_rect = self._image_draw_rect()
        if self._image_pixmap is None or draw_rect.isEmpty():
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a video to start")
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        source_rect = QRectF(0.0, 0.0, float(self._image_pixmap.width()), float(self._image_pixmap.height()))
        painter.drawPixmap(draw_rect, self._image_pixmap, source_rect)

        for roi_name, rel_rect in self._rois_rel.items():
            color = roi_color(roi_name)
            canvas_rect = self._relative_to_canvas_rect(rel_rect, draw_rect)
            pen = QPen(color, 2)
            painter.setPen(pen)
            fill_color = QColor(color)
            fill_color.setAlpha(45)
            painter.fillRect(canvas_rect, fill_color)
            painter.drawRect(canvas_rect)

            text = roi_name
            font_metrics = painter.fontMetrics()
            text_width = font_metrics.horizontalAdvance(text)
            text_height = font_metrics.height()
            label_rect = QRectF(
                canvas_rect.x(),
                max(draw_rect.y(), canvas_rect.y() - text_height - 4),
                text_width + 10,
                text_height + 4,
            )
            label_bg = QColor(color)
            label_bg.setAlpha(180)
            painter.fillRect(label_rect, label_bg)
            painter.setPen(Qt.GlobalColor.black)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, text)

        if self._drag_start is not None and self._drag_current is not None:
            temp_rect = normalize_relative_rect(
                x=self._drag_start.x(),
                y=self._drag_start.y(),
                w=self._drag_current.x() - self._drag_start.x(),
                h=self._drag_current.y() - self._drag_start.y(),
            )
            if temp_rect is not None:
                drag_canvas_rect = self._relative_to_canvas_rect(temp_rect, draw_rect)
                active_color = roi_color(self._active_roi) if self._active_roi else QColor(255, 255, 255)
                drag_pen = QPen(active_color, 2, Qt.PenStyle.DashLine)
                painter.setPen(drag_pen)
                painter.drawRect(drag_canvas_rect)

class RoiCard(QGroupBox):
    PREVIEW_WIDTH = 200
    PREVIEW_HEIGHT = 112

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Selected ROI", parent)

        layout = QVBoxLayout(self)

        self.pixel_label = QLabel("Pixel: -")
        self.relative_label = QLabel("Relative: -")

        self.preview_label = QLabel("No Preview")
        self.preview_label.setFixedSize(self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid #666; background: #111; color: #aaa;")

        layout.addWidget(self.pixel_label)
        layout.addWidget(self.relative_label)
        layout.addWidget(self.preview_label)

    def set_roi_name(self, roi_name: Optional[str]) -> None:
        if roi_name:
            self.setTitle(f"Selected ROI: {roi_name}")
        else:
            self.setTitle("Selected ROI: None")

    def set_empty(self, relative_rect: Optional[RelativeRect]) -> None:
        self.pixel_label.setText("Pixel: -")
        self.relative_label.setText(f"Relative: {format_relative_rect(relative_rect)}")
        self.preview_label.clear()
        self.preview_label.setText("No Preview")

    def set_stats(self, stats: RoiStats) -> None:
        self.pixel_label.setText(f"Pixel: {format_pixel_rect(stats.pixel_rect)}")
        self.relative_label.setText(f"Relative: {format_relative_rect(stats.relative_rect)}")

        scaled = stats.preview_pixmap.scaled(
            self.PREVIEW_WIDTH,
            self.PREVIEW_HEIGHT,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)


class RoiColorTimelineWidget(QWidget):
    time_selected = pyqtSignal(float, int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._roi_name = ""
        self._times = np.asarray([], dtype=np.float64)
        self._delta_lab = np.asarray([], dtype=np.float64)
        self._lab_l = np.asarray([], dtype=np.float64)
        self._lab_a = np.asarray([], dtype=np.float64)
        self._lab_b = np.asarray([], dtype=np.float64)
        self._mean_rgb: list[tuple[int, int, int]] = []
        self._selected_index: Optional[int] = None
        self._view_start_index = 0
        self._view_end_index = 0
        self._dragging_strip = False
        self._drag_start_x = 0.0
        self._drag_start_range: tuple[int, int] = (0, 0)
        self._drag_moved = False

        self.setMinimumHeight(90)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def clear_data(self) -> None:
        self._roi_name = ""
        self._times = np.asarray([], dtype=np.float64)
        self._delta_lab = np.asarray([], dtype=np.float64)
        self._lab_l = np.asarray([], dtype=np.float64)
        self._lab_a = np.asarray([], dtype=np.float64)
        self._lab_b = np.asarray([], dtype=np.float64)
        self._mean_rgb = []
        self._selected_index = None
        self._view_start_index = 0
        self._view_end_index = 0
        self._dragging_strip = False
        self._drag_start_x = 0.0
        self._drag_start_range = (0, 0)
        self._drag_moved = False
        self.update()

    def set_data(self, payload: dict) -> None:
        roi_name = str(payload.get("roi_name", "")).strip()
        times = self._to_float_array(payload.get("times"))
        mean_rgb = self._to_rgb_series(payload.get("mean_rgb"))

        count = min(times.size, len(mean_rgb))
        if count <= 0:
            self.clear_data()
            return

        self._roi_name = roi_name
        self._times = times[:count]
        self._delta_lab = np.asarray([], dtype=np.float64)
        self._lab_l = np.asarray([], dtype=np.float64)
        self._lab_a = np.asarray([], dtype=np.float64)
        self._lab_b = np.asarray([], dtype=np.float64)
        self._mean_rgb = mean_rgb[:count]
        self._selected_index = 0
        self._view_start_index = 0
        self._view_end_index = count - 1
        self._dragging_strip = False
        self._drag_moved = False
        self.update()

    def set_selected_index(self, index: Optional[int]) -> None:
        if index is None:
            self._selected_index = None
            self.update()
            return
        if self._times.size == 0:
            self._selected_index = None
            self.update()
            return
        bounded = max(0, min(int(index), int(self._times.size) - 1))
        self._selected_index = bounded
        self.update()

    def has_data(self) -> bool:
        return self._times.size > 0

    def zoom_in(self, focus_index: Optional[int] = None) -> bool:
        start_idx, end_idx = self._visible_range()
        current = end_idx - start_idx + 1
        if current <= 1:
            return False
        target = max(1, int(round(current * 0.75)))
        if target >= current:
            target = current - 1
        return self._apply_zoom_to_count(target, focus_index)

    def zoom_out(self, focus_index: Optional[int] = None) -> bool:
        total = int(self._times.size)
        start_idx, end_idx = self._visible_range()
        current = end_idx - start_idx + 1
        if current >= total:
            return False
        target = min(total, int(round(current / 0.75)))
        if target <= current:
            target = current + 1
        return self._apply_zoom_to_count(target, focus_index)

    def reset_zoom(self) -> bool:
        total = int(self._times.size)
        if total <= 0:
            return False
        if self._view_start_index == 0 and self._view_end_index == total - 1:
            return False
        self._view_start_index = 0
        self._view_end_index = total - 1
        self.update()
        return True

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._times.size == 0:
            return

        chart_rects = self._chart_rects()
        if chart_rects is None:
            return

        timeline_rect, color_rect = chart_rects
        point = event.position()
        if not color_rect.contains(point):
            return

        index = self._nearest_index_for_x(float(point.x()), timeline_rect)
        if index is None:
            return

        self._dragging_strip = True
        self._drag_start_x = float(point.x())
        self._drag_start_range = self._visible_range()
        self._drag_moved = False
        self._selected_index = index
        self.update()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if not self._dragging_strip or self._times.size <= 0:
            return

        chart_rects = self._chart_rects()
        if chart_rects is None:
            return

        timeline_rect, _ = chart_rects
        start_idx, end_idx = self._drag_start_range
        visible_count = end_idx - start_idx + 1
        total = int(self._times.size)
        if visible_count <= 0:
            return

        point = event.position()
        index = self._nearest_index_for_x(float(point.x()), timeline_rect)
        if index is not None:
            self._selected_index = index

        if total <= visible_count:
            self.update()
            return

        px_width = max(1.0, float(timeline_rect.width()))
        px_per_sample = px_width / float(max(1, visible_count - 1))
        shift_samples = int(round((float(point.x()) - self._drag_start_x) / px_per_sample))
        if shift_samples != 0:
            max_start = max(0, total - visible_count)
            new_start = max(0, min(max_start, start_idx - shift_samples))
            new_end = new_start + visible_count - 1
            if new_start != self._view_start_index or new_end != self._view_end_index:
                self._view_start_index = new_start
                self._view_end_index = new_end
                self._drag_moved = True
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not self._dragging_strip:
            return

        self._dragging_strip = False

        chart_rects = self._chart_rects()
        if chart_rects is None or self._times.size == 0:
            return

        timeline_rect, color_rect = chart_rects
        was_dragged = self._drag_moved
        self._drag_moved = False
        point = event.position()
        if was_dragged:
            self.update()
            return
        if color_rect.contains(point):
            index = self._nearest_index_for_x(float(point.x()), timeline_rect)
            if index is not None:
                self._selected_index = index
                self.update()
                self.time_selected.emit(float(self._times[index]), int(index))

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if self._times.size <= 1:
            event.ignore()
            return

        chart_rects = self._chart_rects()
        if chart_rects is None:
            event.ignore()
            return

        timeline_rect, color_rect = chart_rects
        point = event.position()
        if not color_rect.contains(point):
            event.ignore()
            return

        focus_index = self._nearest_index_for_x(float(point.x()), timeline_rect)
        delta = int(event.angleDelta().y())
        changed = False
        if delta > 0:
            changed = self.zoom_in(focus_index)
        elif delta < 0:
            changed = self.zoom_out(focus_index)

        if changed:
            event.accept()
        else:
            event.ignore()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#15171c"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if self._times.size == 0:
            painter.setPen(QColor("#b8c0cc"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Renk analizi verisi yok.")
            return

        chart_rects = self._chart_rects()
        if chart_rects is None:
            painter.setPen(QColor("#b8c0cc"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Panel cok kucuk.")
            return

        timeline_rect, color_rect = chart_rects
        start_idx, end_idx = self._visible_range()
        visible_rgb = self._mean_rgb[start_idx : end_idx + 1]

        self._draw_panel_background(painter, color_rect)
        self._draw_color_strip(painter, color_rect, visible_rgb)

        if self._selected_index is not None and start_idx <= self._selected_index <= end_idx:
            x_selected = self._x_for_index(timeline_rect, self._selected_index)
            select_pen = QPen(QColor("#f4f5f7"), 1, Qt.PenStyle.DashLine)
            painter.setPen(select_pen)
            painter.drawLine(
                int(round(x_selected)),
                int(round(color_rect.top())),
                int(round(x_selected)),
                int(round(color_rect.bottom())),
            )

        painter.setPen(QColor("#d3d9e5"))
        painter.drawText(
            QRectF(color_rect.left() + 8, color_rect.top() + 2, color_rect.width() - 16, 16),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "Ortalama Renk (Surukle: kaydir, Tekerlek: zoom)",
        )

        label = f"ROI: {self._roi_name}" if self._roi_name else "ROI: -"
        visible_count = end_idx - start_idx + 1
        zoom_text = f"Gorunum: {visible_count}/{int(self._times.size)}"
        painter.drawText(
            QRectF(timeline_rect.left(), color_rect.top() - 20, timeline_rect.width() * 0.65, 18),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            label,
        )
        painter.drawText(
            QRectF(
                timeline_rect.left() + (timeline_rect.width() * 0.65),
                color_rect.top() - 20,
                timeline_rect.width() * 0.35,
                18,
            ),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            zoom_text,
        )

        start_text = f"{float(self._times[start_idx]):.2f}s"
        end_text = f"{float(self._times[end_idx]):.2f}s"
        painter.setPen(QColor("#9aa4b7"))
        painter.drawText(
            QRectF(timeline_rect.left(), timeline_rect.bottom() + 4, 110, 18),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            start_text,
        )
        painter.drawText(
            QRectF(timeline_rect.right() - 110, timeline_rect.bottom() + 4, 110, 18),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            end_text,
        )

    @staticmethod
    def _to_float_array(raw: object) -> np.ndarray:
        if raw is None:
            return np.asarray([], dtype=np.float64)
        try:
            arr = np.asarray(list(raw), dtype=np.float64)
        except (TypeError, ValueError):
            return np.asarray([], dtype=np.float64)
        if arr.ndim != 1:
            return np.asarray([], dtype=np.float64)
        return arr

    @staticmethod
    def _to_rgb_series(raw: object) -> list[tuple[int, int, int]]:
        if not isinstance(raw, list):
            return []

        result: list[tuple[int, int, int]] = []
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                continue
            try:
                red = max(0, min(255, int(item[0])))
                green = max(0, min(255, int(item[1])))
                blue = max(0, min(255, int(item[2])))
            except (TypeError, ValueError):
                continue
            result.append((red, green, blue))
        return result

    def _chart_rects(self) -> Optional[tuple[QRectF, QRectF]]:
        outer = QRectF(self.rect()).adjusted(50, 34, -16, -34)
        if outer.width() < 120 or outer.height() < 28:
            return None

        color_rect = outer
        timeline_rect = outer
        return timeline_rect, color_rect

    def _visible_range(self) -> tuple[int, int]:
        total = int(self._times.size)
        if total <= 0:
            return 0, -1
        start_idx = max(0, min(int(self._view_start_index), total - 1))
        end_idx = max(start_idx, min(int(self._view_end_index), total - 1))
        return start_idx, end_idx

    def _apply_zoom_to_count(self, new_count: int, focus_index: Optional[int]) -> bool:
        total = int(self._times.size)
        if total <= 0:
            return False

        new_count = max(1, min(total, int(new_count)))
        start_idx, end_idx = self._visible_range()
        current_count = end_idx - start_idx + 1
        if current_count == new_count:
            return False

        if focus_index is None:
            if self._selected_index is not None:
                focus_index = self._selected_index
            else:
                focus_index = start_idx + (current_count // 2)
        focus_index = max(0, min(total - 1, int(focus_index)))

        if current_count <= 1:
            focus_ratio = 0.5
        else:
            focus_ratio = (focus_index - start_idx) / float(current_count - 1)

        new_start = int(round(focus_index - focus_ratio * float(max(1, new_count - 1))))
        max_start = max(0, total - new_count)
        new_start = max(0, min(max_start, new_start))
        new_end = new_start + new_count - 1

        self._view_start_index = new_start
        self._view_end_index = new_end
        self.update()
        return True

    def _nearest_index_for_x(self, x_pos: float, timeline_rect: QRectF) -> Optional[int]:
        if self._times.size == 0:
            return None
        start_idx, end_idx = self._visible_range()
        visible_count = end_idx - start_idx + 1
        if visible_count <= 1:
            return start_idx

        left = float(timeline_rect.left())
        width = max(1.0, float(timeline_rect.width()))
        clamped_x = max(left, min(float(timeline_rect.right()), x_pos))
        ratio = (clamped_x - left) / width
        local_index = int(round(ratio * float(visible_count - 1)))
        return start_idx + max(0, min(visible_count - 1, local_index))

    @staticmethod
    def _series_limits(values: np.ndarray) -> tuple[float, float]:
        if values.size == 0:
            return 0.0, 1.0

        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0, 1.0

        low = float(np.min(finite))
        high = float(np.max(finite))
        if abs(high - low) <= 1e-9:
            padding = max(1.0, abs(low) * 0.1)
            return low - padding, high + padding

        padding = (high - low) * 0.1
        return low - padding, high + padding

    def _x_for_index(self, rect: QRectF, index: int) -> float:
        start_idx, end_idx = self._visible_range()
        visible_count = end_idx - start_idx + 1
        if visible_count <= 1:
            return float(rect.center().x())

        index = max(start_idx, min(int(index), end_idx))
        ratio = float(index - start_idx) / float(max(1, visible_count - 1))
        ratio = max(0.0, min(1.0, ratio))
        return float(rect.left()) + ratio * float(rect.width())

    @staticmethod
    def _y_for_value(rect: QRectF, value: float, min_value: float, max_value: float) -> float:
        span = max(1e-9, max_value - min_value)
        ratio = (value - min_value) / span
        ratio = max(0.0, min(1.0, ratio))
        return float(rect.bottom()) - ratio * float(rect.height())

    @staticmethod
    def _draw_panel_background(painter: QPainter, rect: QRectF) -> None:
        painter.fillRect(rect, QColor("#1b2028"))
        pen = QPen(QColor("#313744"), 1)
        painter.setPen(pen)
        painter.drawRect(rect)

    @staticmethod
    def _draw_guides(painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(QColor("#2a313c"), 1))
        for ratio in (0.25, 0.5, 0.75):
            y_pos = rect.top() + (rect.height() * ratio)
            painter.drawLine(
                int(round(rect.left())),
                int(round(y_pos)),
                int(round(rect.right())),
                int(round(y_pos)),
            )

    def _draw_series(
        self,
        painter: QPainter,
        rect: QRectF,
        values: np.ndarray,
        start_index: int,
        end_index: int,
        min_value: float,
        max_value: float,
        color: QColor,
        width: float,
    ) -> None:
        if values.size == 0 or end_index < start_index:
            return

        pen = QPen(color, width)
        painter.setPen(pen)

        visible_count = end_index - start_index + 1
        if visible_count == 1:
            x_pos = self._x_for_index(rect, start_index)
            y_pos = self._y_for_value(rect, float(values[start_index]), min_value, max_value)
            painter.drawEllipse(QRectF(x_pos - 2.5, y_pos - 2.5, 5.0, 5.0))
            return

        path = QPainterPath()
        for idx in range(start_index, end_index + 1):
            value = values[idx]
            x_pos = self._x_for_index(rect, idx)
            y_pos = self._y_for_value(rect, float(value), min_value, max_value)
            if idx == start_index:
                path.moveTo(x_pos, y_pos)
            else:
                path.lineTo(x_pos, y_pos)
        painter.drawPath(path)

    def _draw_color_strip(self, painter: QPainter, rect: QRectF, visible_rgb: list[tuple[int, int, int]]) -> None:
        if not visible_rgb:
            return

        count = len(visible_rgb)
        if count == 1:
            red, green, blue = visible_rgb[0]
            painter.fillRect(rect, QColor(red, green, blue))
            painter.setPen(QPen(QColor("#313744"), 1))
            painter.drawRect(rect)
            return

        left = float(rect.left())
        width = float(rect.width())
        top = float(rect.top())
        height = float(rect.height())

        for idx, (red, green, blue) in enumerate(visible_rgb):
            x0 = left + (float(idx) / float(count)) * width
            x1 = left + (float(idx + 1) / float(count)) * width
            segment = QRectF(x0, top, max(1.0, x1 - x0), height)
            painter.fillRect(segment, QColor(red, green, blue))

        painter.setPen(QPen(QColor("#313744"), 1))
        painter.drawRect(rect)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ROI Template Editor")
        self.resize(1360, 860)

        self.settings = QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)

        self.original_bgr: Optional[np.ndarray] = None
        self.video_meta: Optional[VideoMeta] = None
        self.video_frame_count = 0

        self.rois_rel: Dict[str, RelativeRect] = {}
        self.roi_order: list[str] = []
        self.active_roi_name: Optional[str] = None

        self.pending_template_rois: Optional[Dict[str, RelativeRect]] = None
        self.pending_template_order: Optional[list[str]] = None

        self.current_template_path: Optional[str] = None
        self.shortcuts: list[QShortcut] = []
        self._syncing_roi_list = False
        self.detect_thread: Optional[QThread] = None
        self.detect_worker: Optional[EventDetectionWorker] = None
        self.color_thread: Optional[QThread] = None
        self.color_worker: Optional[RoiColorAnalysisWorker] = None
        self.last_detected_events: list[dict] = []
        self.last_roi_color_payload: Optional[dict] = None
        self.last_detection_sample_hz = 10
        self._last_detection_log_message = ""
        self._syncing_color_roi_combo = False
        self._syncing_mode_combo = False
        self._syncing_manual_slider = False
        self._event_detection_busy = False
        self._color_analysis_busy = False
        self._event_detection_cancel_requested = False
        self._color_analysis_cancel_requested = False
        self.detection_mode: Optional[str] = None
        self.timeline_dirty = False
        self.startup_tab: Optional[QWidget] = None
        self.startup_completed = False
        self._screen_history: list[str] = []
        self._current_screen = "startup"

        self.canvas = VideoCanvas(self)
        self.canvas.roi_drawn.connect(self.on_roi_drawn)
        self.canvas.draw_requested_without_selection.connect(self.on_draw_without_selection)

        self._build_ui()
        self._setup_shortcuts()
        self._try_autoload_last_template()
        self.update_selected_roi_panel()
        self._update_event_video_label()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        nav_bar = QWidget(self)
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(10)

        self.back_button = QPushButton("< Geri")
        self.back_button.clicked.connect(self.on_back_button_clicked)
        self.back_button.setMinimumWidth(90)

        self.screen_title_label = QLabel("Baslangic")
        self.screen_title_label.setStyleSheet("font-size: 16px; font-weight: 600; color: #f4f5f7;")

        nav_layout.addWidget(self.back_button)
        nav_layout.addWidget(self.screen_title_label)
        nav_layout.addStretch(1)
        root_layout.addWidget(nav_bar)

        self.main_stack = QStackedWidget(self)
        root_layout.addWidget(self.main_stack, stretch=1)

        self.startup_tab = QWidget(self)
        self._build_startup_tab(self.startup_tab)
        self.main_stack.addWidget(self.startup_tab)

        self.roi_tab = QWidget(self)
        self._build_roi_tab(self.roi_tab)
        self.main_stack.addWidget(self.roi_tab)

        self.event_tab = QWidget(self)
        self._build_event_tab(self.event_tab)
        self.main_stack.addWidget(self.event_tab)

        self._set_current_screen("startup", push_history=False)

    def _screen_title(self, screen: str) -> str:
        if screen == "startup":
            return "Baslangic"
        if screen == "roi":
            return "ROI Secimi"
        if screen == "event":
            return "Olay Tespit"
        return "Ekran"

    def _screen_widget(self, screen: str) -> Optional[QWidget]:
        if screen == "startup":
            return self.startup_tab
        if screen == "roi":
            return self.roi_tab
        if screen == "event":
            return self.event_tab
        return None

    def _set_current_screen(self, screen: str, push_history: bool) -> bool:
        target = self._screen_widget(screen)
        if target is None:
            return False

        if screen == self._current_screen:
            self._update_navigation_bar()
            return True

        current = self._screen_widget(self._current_screen)
        if push_history and current is not None:
            self._screen_history.append(self._current_screen)

        self._current_screen = screen
        self.main_stack.setCurrentWidget(target)
        self._update_navigation_bar()
        return True

    def _update_navigation_bar(self) -> None:
        self.back_button.setEnabled(bool(self._screen_history))
        self.screen_title_label.setText(self._screen_title(self._current_screen))

    def on_back_button_clicked(self) -> None:
        if not self._screen_history:
            return
        previous = self._screen_history.pop()
        self._set_current_screen(previous, push_history=False)

    def _build_startup_tab(self, container: QWidget) -> None:
        layout = QVBoxLayout(container)
        layout.addStretch(1)

        title = QLabel("Olay Tespit Modu Secimi")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: 700;")

        subtitle = QLabel("Devam etmeden once bir calisma modu secin.")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #b8c0cc;")

        auto_button = QPushButton("Otomatik Olay Tespiti")
        auto_button.setMinimumHeight(46)
        auto_button.clicked.connect(self.on_startup_auto_clicked)

        manual_button = QPushButton("Manuel Olay Tespiti")
        manual_button.setMinimumHeight(46)
        manual_button.clicked.connect(self.on_startup_manual_clicked)

        button_box = QWidget(container)
        button_layout = QVBoxLayout(button_box)
        button_layout.setContentsMargins(0, 10, 0, 0)
        button_layout.setSpacing(10)
        button_layout.addWidget(auto_button)
        button_layout.addWidget(manual_button)

        wrapper = QWidget(container)
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(8)
        wrapper_layout.addWidget(title)
        wrapper_layout.addWidget(subtitle)
        wrapper_layout.addWidget(button_box)
        wrapper.setMaximumWidth(460)

        center_row = QHBoxLayout()
        center_row.addStretch(1)
        center_row.addWidget(wrapper)
        center_row.addStretch(1)

        layout.addLayout(center_row)
        layout.addStretch(2)

    def _build_roi_tab(self, container: QWidget) -> None:
        main_layout = QHBoxLayout(container)
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        controls_layout = QHBoxLayout()
        self.open_button = QPushButton("Open Video")
        self.load_button = QPushButton("Load Template")
        self.save_button = QPushButton("Save Template")
        self.reset_button = QPushButton("Reset ROIs")

        controls_layout.addWidget(self.open_button)
        controls_layout.addWidget(self.load_button)
        controls_layout.addWidget(self.save_button)
        controls_layout.addWidget(self.reset_button)

        self.open_button.clicked.connect(self.open_video_dialog)
        self.load_button.clicked.connect(self.load_template_dialog)
        self.save_button.clicked.connect(self.save_template)
        self.reset_button.clicked.connect(self.reset_rois)

        self.goto_event_tab_button = QPushButton("Olay Tespit'e Gec")
        self.goto_event_tab_button.clicked.connect(self.switch_to_event_tab)

        roi_manager_box = QGroupBox("ROI Manager")
        roi_manager_layout = QVBoxLayout(roi_manager_box)
        self.roi_list = QListWidget()
        self.roi_list.currentItemChanged.connect(self.on_roi_list_current_item_changed)

        roi_buttons_layout = QHBoxLayout()
        self.add_roi_button = QPushButton("Add ROI")
        self.rename_roi_button = QPushButton("Rename ROI")
        self.delete_roi_button = QPushButton("Delete ROI")
        self.add_roi_button.clicked.connect(self.add_roi)
        self.rename_roi_button.clicked.connect(self.rename_roi)
        self.delete_roi_button.clicked.connect(self.delete_roi)
        roi_buttons_layout.addWidget(self.add_roi_button)
        roi_buttons_layout.addWidget(self.rename_roi_button)
        roi_buttons_layout.addWidget(self.delete_roi_button)

        roi_manager_layout.addWidget(self.roi_list)
        roi_manager_layout.addLayout(roi_buttons_layout)

        left_layout.addLayout(controls_layout)
        left_layout.addWidget(self.goto_event_tab_button)
        left_layout.addWidget(self.canvas, stretch=1)

        self.selected_roi_card = RoiCard()
        right_layout.addWidget(roi_manager_box)
        right_layout.addWidget(self.selected_roi_card)
        right_layout.addStretch(1)

        right_panel = QWidget()
        right_panel.setLayout(right_layout)
        right_panel.setMinimumWidth(360)
        right_panel.setMaximumWidth(520)

        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addWidget(right_panel)

    def _build_event_tab(self, container: QWidget) -> None:
        root_layout = QVBoxLayout(container)
        root_layout.setContentsMargins(0, 0, 0, 0)

        left_panel = QWidget(container)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.event_video_label = QLabel("Aktif video: -")
        self.event_frame_preview = QLabel("Start/End hucrelerine tiklayinca frame burada gorunecek.")
        self.event_frame_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.event_frame_preview.setMinimumSize(280, 220)
        self.event_frame_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.event_frame_preview.setStyleSheet("border: 1px solid #666; background: #111; color: #aaa;")
        left_layout.addWidget(self.event_frame_preview, stretch=1)

        right_panel = QWidget(container)
        layout = QVBoxLayout(right_panel)
        layout.setContentsMargins(0, 0, 0, 0)
        right_panel.setMinimumWidth(460)
        layout.addWidget(self.event_video_label)

        controls_layout = QHBoxLayout()
        self.event_mode_label = QLabel("Mod:")
        self.event_mode_combo = QComboBox()
        self.event_mode_combo.addItem("Otomatik", DETECTION_MODE_AUTO)
        self.event_mode_combo.addItem("Manuel", DETECTION_MODE_MANUAL)
        self.event_mode_combo.currentIndexChanged.connect(self.on_event_mode_combo_changed)

        self.event_open_video_button = QPushButton("Video Ac")
        self.event_open_video_button.clicked.connect(self.open_video_dialog)

        self.sample_hz_spin = QSpinBox()
        self.sample_hz_spin.setRange(5, 12)
        self.sample_hz_spin.setValue(10)
        self.sample_hz_spin.setPrefix("sample_hz: ")
        self.source_sensitivity_label = QLabel("Ilk4 start hass.:")
        self.source_sensitivity_combo = QComboBox()
        self.source_sensitivity_combo.addItems(list(SOURCE_START_SENSITIVITY_PRESETS.keys()))
        self.source_sensitivity_combo.setCurrentText("Dengeli")
        self.source_sensitivity_combo.setToolTip("Ilk 4 olayin start zamani icin kaynak ROI renk degisim hassasiyeti.")

        self.detect_button = QPushButton("Olaylari Tespit Et")
        self.detect_button.clicked.connect(self.on_detect_button_clicked)

        self.save_timeline_button = QPushButton("timeline.json Kaydet")
        self.save_timeline_button.setEnabled(False)
        self.save_timeline_button.clicked.connect(self.save_timeline_json)

        self.color_roi_label = QLabel("Renk ROI:")
        self.color_roi_combo = QComboBox()
        self.color_roi_combo.currentTextChanged.connect(self.on_color_roi_changed)
        self.color_analyze_button = QPushButton("ROI Renk Analizi")
        self.color_analyze_button.clicked.connect(self.on_color_analyze_button_clicked)

        controls_layout.addWidget(self.event_mode_label)
        controls_layout.addWidget(self.event_mode_combo)
        controls_layout.addWidget(self.event_open_video_button)
        controls_layout.addWidget(self.sample_hz_spin)
        controls_layout.addWidget(self.source_sensitivity_label)
        controls_layout.addWidget(self.source_sensitivity_combo)
        controls_layout.addWidget(self.detect_button)
        controls_layout.addWidget(self.save_timeline_button)
        controls_layout.addStretch(1)

        self.event_progress = QProgressBar()
        self.event_progress.setRange(0, 100)
        self.event_progress.setValue(0)

        self.event_log = QPlainTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setPlaceholderText("Olay tespit loglari burada gorunecek.")
        self.event_log.setMaximumBlockCount(300)
        self.event_log.setMinimumWidth(320)
        self.event_log.setMaximumHeight(110)

        top_row_layout = QHBoxLayout()
        top_left_layout = QVBoxLayout()
        top_left_layout.setContentsMargins(0, 0, 0, 0)
        top_left_layout.addLayout(controls_layout)
        top_left_layout.addWidget(self.event_progress)
        top_row_layout.addLayout(top_left_layout, stretch=3)
        top_row_layout.addWidget(self.event_log, stretch=2)

        self.event_table = QTableWidget(len(EVENT_DEFINITIONS), 7)
        self.event_table.setHorizontalHeaderLabels(
            ["id", "name", "target_roi", "type", "start", "end", "confidence"]
        )
        self.event_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.event_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.event_table.verticalHeader().setVisible(False)
        header = self.event_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.event_table.cellClicked.connect(self.on_event_table_cell_clicked)
        self.event_table.itemSelectionChanged.connect(self.on_event_table_selection_changed)

        self.manual_controls_box = QGroupBox("Manuel Olay Atama")
        manual_layout = QVBoxLayout(self.manual_controls_box)
        manual_layout.setContentsMargins(8, 8, 8, 8)
        manual_layout.setSpacing(6)

        self.manual_time_label = QLabel("Frame: -")
        self.manual_time_label.setStyleSheet("color: #9aa4b7;")

        self.manual_frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.manual_frame_slider.setRange(0, 0)
        self.manual_frame_slider.valueChanged.connect(self.on_manual_frame_slider_changed)

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(QLabel("Timeline:"))
        nav_layout.addWidget(self.manual_time_label, stretch=1)

        step_layout = QHBoxLayout()
        self.manual_step_minus_sec_button = QPushButton("-1 sn")
        self.manual_step_minus_frame_button = QPushButton("-1 f")
        self.manual_step_plus_frame_button = QPushButton("+1 f")
        self.manual_step_plus_sec_button = QPushButton("+1 sn")
        self.manual_step_minus_sec_button.clicked.connect(lambda: self.step_manual_seconds(-1.0))
        self.manual_step_minus_frame_button.clicked.connect(lambda: self.step_manual_frame(-1))
        self.manual_step_plus_frame_button.clicked.connect(lambda: self.step_manual_frame(1))
        self.manual_step_plus_sec_button.clicked.connect(lambda: self.step_manual_seconds(1.0))
        step_layout.addWidget(self.manual_step_minus_sec_button)
        step_layout.addWidget(self.manual_step_minus_frame_button)
        step_layout.addWidget(self.manual_step_plus_frame_button)
        step_layout.addWidget(self.manual_step_plus_sec_button)
        step_layout.addStretch(1)

        assign_layout = QHBoxLayout()
        self.manual_assign_start_button = QPushButton("Start Ata")
        self.manual_assign_end_button = QPushButton("End Ata")
        self.manual_clear_row_button = QPushButton("Satiri Temizle")
        self.manual_assign_start_button.clicked.connect(self.on_manual_assign_start_clicked)
        self.manual_assign_end_button.clicked.connect(self.on_manual_assign_end_clicked)
        self.manual_clear_row_button.clicked.connect(self.on_manual_clear_row_clicked)
        assign_layout.addWidget(self.manual_assign_start_button)
        assign_layout.addWidget(self.manual_assign_end_button)
        assign_layout.addWidget(self.manual_clear_row_button)
        assign_layout.addStretch(1)

        manual_layout.addLayout(nav_layout)
        manual_layout.addWidget(self.manual_frame_slider)
        manual_layout.addLayout(step_layout)
        manual_layout.addLayout(assign_layout)

        color_controls_layout = QHBoxLayout()
        color_controls_layout.addWidget(self.color_roi_label)
        color_controls_layout.addWidget(self.color_roi_combo)
        color_controls_layout.addWidget(self.color_analyze_button)
        color_controls_layout.addStretch(1)

        self.color_progress = QProgressBar()
        self.color_progress.setRange(0, 100)
        self.color_progress.setValue(0)

        self.roi_color_timeline = RoiColorTimelineWidget(self)
        self.roi_color_timeline.time_selected.connect(self.on_roi_color_time_selected)

        upper_section = QWidget(right_panel)
        upper_layout = QVBoxLayout(upper_section)
        upper_layout.setContentsMargins(0, 0, 0, 0)
        upper_layout.addLayout(top_row_layout)
        upper_layout.addWidget(self.manual_controls_box)
        upper_layout.addWidget(self.event_table, stretch=1)

        self.event_lower_section = QWidget(right_panel)
        lower_layout = QVBoxLayout(self.event_lower_section)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        lower_layout.addLayout(color_controls_layout)
        lower_layout.addWidget(self.color_progress)
        lower_layout.addWidget(self.roi_color_timeline, stretch=1)

        self.event_vertical_splitter = QSplitter(Qt.Orientation.Vertical, right_panel)
        self.event_vertical_splitter.setChildrenCollapsible(False)
        self.event_vertical_splitter.setHandleWidth(8)
        self.event_vertical_splitter.addWidget(upper_section)
        self.event_vertical_splitter.addWidget(self.event_lower_section)
        self.event_vertical_splitter.setStretchFactor(0, 3)
        self.event_vertical_splitter.setStretchFactor(1, 2)
        self.event_vertical_splitter.setSizes([420, 320])
        layout.addWidget(self.event_vertical_splitter, stretch=1)

        self.event_splitter = QSplitter(Qt.Orientation.Horizontal, container)
        self.event_splitter.setChildrenCollapsible(False)
        self.event_splitter.setHandleWidth(8)
        self.event_splitter.addWidget(left_panel)
        self.event_splitter.addWidget(right_panel)
        self.event_splitter.setStretchFactor(0, 0)
        self.event_splitter.setStretchFactor(1, 1)
        left_initial = max(220, self.event_frame_preview.minimumWidth())
        self.event_splitter.setSizes([left_initial, 1800])
        root_layout.addWidget(self.event_splitter)

        self._reset_event_table()
        self._refresh_color_roi_combo()
        self._update_analysis_controls()

    def _setup_shortcuts(self) -> None:
        for index in range(1, MAX_SHORTCUT_ROIS + 1):
            shortcut = QShortcut(QKeySequence(str(index)), self)
            shortcut.activated.connect(lambda roi_index=index - 1: self.select_roi_by_index(roi_index))
            self.shortcuts.append(shortcut)

    def on_startup_auto_clicked(self) -> None:
        if not self._apply_detection_mode(DETECTION_MODE_AUTO, source="startup"):
            return
        self._complete_startup_selection(open_event_tab=False)

    def on_startup_manual_clicked(self) -> None:
        if not self._apply_detection_mode(DETECTION_MODE_MANUAL, source="startup"):
            return
        self._complete_startup_selection(open_event_tab=True)

    def _complete_startup_selection(self, open_event_tab: bool) -> None:
        self.startup_completed = True
        self._update_analysis_controls()

        if open_event_tab:
            self.switch_to_event_tab()
        else:
            self.switch_to_roi_tab()

    def _build_manual_events_payload(self) -> list[dict]:
        return [
            {
                "id": int(event_info["id"]),
                "name": str(event_info["name"]),
                "target_roi": str(event_info["target_roi"]),
                "type": str(event_info["type"]),
                "start": None,
                "end": None,
                "confidence": None,
            }
            for event_info in EVENT_DEFINITIONS
        ]

    def _selected_mode_from_combo(self) -> Optional[str]:
        index = self.event_mode_combo.currentIndex()
        if index < 0:
            return None
        data = self.event_mode_combo.itemData(index)
        if data in (DETECTION_MODE_AUTO, DETECTION_MODE_MANUAL):
            return str(data)
        return None

    def _sync_event_mode_combo_to_state(self) -> None:
        target_mode = self.detection_mode if self.detection_mode in (DETECTION_MODE_AUTO, DETECTION_MODE_MANUAL) else None
        if target_mode is None:
            target_mode = DETECTION_MODE_AUTO

        target_index = self.event_mode_combo.findData(target_mode)
        if target_index < 0:
            return
        if self.event_mode_combo.currentIndex() == target_index:
            return

        self._syncing_mode_combo = True
        try:
            self.event_mode_combo.setCurrentIndex(target_index)
        finally:
            self._syncing_mode_combo = False

    def _has_usable_video(self) -> bool:
        return self.video_meta is not None and os.path.isfile(self.video_meta.source_video)

    def _confirm_discard_timeline_if_needed(self) -> bool:
        if not self.timeline_dirty or not self.last_detected_events:
            return True

        response = QMessageBox.question(
            self,
            "Mod Degisimi",
            "Kaydedilmemis olay degisiklikleri var. Mod degistirirseniz veriler temizlenecek.\nDevam etmek istiyor musunuz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return response == QMessageBox.StandardButton.Yes

    def _initialize_manual_events(self) -> None:
        self.last_detected_events = self._build_manual_events_payload()
        self.event_log.clear()
        self.event_progress.setValue(0)
        self.timeline_dirty = False
        self._populate_event_table_from_results()

    def _apply_detection_mode(self, mode: Optional[str], source: str) -> bool:
        if mode not in (DETECTION_MODE_AUTO, DETECTION_MODE_MANUAL, None):
            return False
        if mode == self.detection_mode:
            self._sync_event_mode_combo_to_state()
            self._update_analysis_controls()
            return True

        if self._is_event_detection_running() or self._is_color_analysis_running():
            QMessageBox.information(self, "Mod Degisimi", "Calisan analiz varken mod degistirilemez.")
            self._sync_event_mode_combo_to_state()
            return False

        if self.detection_mode is not None and not self._confirm_discard_timeline_if_needed():
            self._sync_event_mode_combo_to_state()
            return False

        if mode == DETECTION_MODE_MANUAL and not self._has_usable_video():
            if not self.open_video_dialog():
                if source == "startup":
                    self.statusBar().showMessage("Manuel mod icin video secimi gerekli.", 2600)
                self._sync_event_mode_combo_to_state()
                return False

        self.detection_mode = mode
        if mode == DETECTION_MODE_MANUAL:
            self._initialize_manual_events()
            self._sync_manual_timeline_to_current_frame()
        else:
            self._invalidate_event_results()
            self.timeline_dirty = False

        self._sync_event_mode_combo_to_state()
        self._update_analysis_controls()
        return True

    def on_event_mode_combo_changed(self, index: int) -> None:
        del index
        if self._syncing_mode_combo:
            return
        if not self.startup_completed:
            self._sync_event_mode_combo_to_state()
            return

        selected_mode = self._selected_mode_from_combo()
        if selected_mode is None:
            self._sync_event_mode_combo_to_state()
            return
        self._apply_detection_mode(selected_mode, source="event")

    def on_event_table_selection_changed(self) -> None:
        self._update_analysis_controls()

    def _manual_selected_row(self) -> Optional[int]:
        row = int(self.event_table.currentRow())
        if row < 0 or row >= len(self.last_detected_events):
            return None
        return row

    def _manual_current_frame_and_seconds(self) -> Optional[tuple[int, float]]:
        if self.video_meta is None:
            return None
        frame_index = max(0, int(self.video_meta.frame_index))
        fps = float(self.video_meta.fps) if self.video_meta.fps > 0 else 30.0
        seconds = float(frame_index) / fps
        return frame_index, seconds

    def _sync_manual_timeline_to_current_frame(self) -> None:
        has_video = self.video_meta is not None
        max_index = 0
        if has_video:
            if self.video_frame_count > 0:
                max_index = max(0, self.video_frame_count - 1)
            else:
                max_index = max(0, int(self.video_meta.frame_index))

        self._syncing_manual_slider = True
        try:
            self.manual_frame_slider.setRange(0, max_index)
            if has_video:
                self.manual_frame_slider.setValue(max(0, min(max_index, int(self.video_meta.frame_index))))
            else:
                self.manual_frame_slider.setValue(0)
        finally:
            self._syncing_manual_slider = False

        if not has_video:
            self.manual_time_label.setText("Frame: -")
            return

        frame_and_seconds = self._manual_current_frame_and_seconds()
        if frame_and_seconds is None:
            self.manual_time_label.setText("Frame: -")
            return

        frame_index, seconds = frame_and_seconds
        total_text = "?"
        if self.video_frame_count > 0:
            total_text = str(self.video_frame_count - 1)
        pretty_time = format_time_dk_sn_ms(seconds)
        self.manual_time_label.setText(f"Frame: {frame_index}/{total_text} | {seconds:.2f} sn | {pretty_time}")

    def on_manual_frame_slider_changed(self, frame_index: int) -> None:
        if self._syncing_manual_slider:
            return
        if self.detection_mode != DETECTION_MODE_MANUAL:
            return

        frame_payload = self._read_video_frame_at_index(int(frame_index))
        if frame_payload is None:
            self.statusBar().showMessage("Frame okunamadi.", 2200)
            return

        frame_bgr, actual_frame = frame_payload
        self._apply_current_frame(frame_bgr, actual_frame)
        self._sync_manual_timeline_to_current_frame()

    def step_manual_frame(self, delta: int) -> None:
        if self.detection_mode != DETECTION_MODE_MANUAL:
            return
        if self.video_meta is None:
            QMessageBox.warning(self, "Manuel Olay", "Once bir video acin.")
            return
        target = int(self.manual_frame_slider.value()) + int(delta)
        target = max(self.manual_frame_slider.minimum(), min(self.manual_frame_slider.maximum(), target))
        self.manual_frame_slider.setValue(target)

    def step_manual_seconds(self, delta_seconds: float) -> None:
        if self.video_meta is None:
            return
        fps = float(self.video_meta.fps) if self.video_meta.fps > 0 else 30.0
        frame_delta = int(round(float(delta_seconds) * fps))
        if frame_delta == 0 and delta_seconds != 0.0:
            frame_delta = 1 if delta_seconds > 0 else -1
        self.step_manual_frame(frame_delta)

    def _assign_manual_event_time(self, field_name: str) -> None:
        if self.detection_mode != DETECTION_MODE_MANUAL:
            return
        row = self._manual_selected_row()
        if row is None:
            QMessageBox.information(self, "Manuel Olay", "Start/End atamak icin bir event satiri secin.")
            return

        frame_and_seconds = self._manual_current_frame_and_seconds()
        if frame_and_seconds is None:
            QMessageBox.warning(self, "Manuel Olay", "Once bir video acin.")
            return

        _, seconds = frame_and_seconds
        payload = dict(self.last_detected_events[row])
        payload[field_name] = round(float(seconds), 2)
        payload["confidence"] = None
        self.last_detected_events[row] = payload
        self.timeline_dirty = True
        self._populate_event_table_from_results()
        self.event_table.selectRow(row)
        event_id = payload.get("id", row + 1)
        self.statusBar().showMessage(f"Event {event_id} {field_name} atandi: {format_time_dk_sn_ms(seconds)}", 2600)

    def on_manual_assign_start_clicked(self) -> None:
        self._assign_manual_event_time("start")

    def on_manual_assign_end_clicked(self) -> None:
        self._assign_manual_event_time("end")

    def on_manual_clear_row_clicked(self) -> None:
        if self.detection_mode != DETECTION_MODE_MANUAL:
            return
        row = self._manual_selected_row()
        if row is None:
            QMessageBox.information(self, "Manuel Olay", "Temizlemek icin bir event satiri secin.")
            return

        payload = dict(self.last_detected_events[row])
        payload["start"] = None
        payload["end"] = None
        payload["confidence"] = None
        self.last_detected_events[row] = payload
        self.timeline_dirty = True
        self._populate_event_table_from_results()
        self.event_table.selectRow(row)
        event_id = payload.get("id", row + 1)
        self.statusBar().showMessage(f"Event {event_id} zamanlari temizlendi.", 2300)

    def switch_to_roi_tab(self) -> None:
        self._set_current_screen("roi", push_history=True)

    def switch_to_event_tab(self) -> None:
        self._set_current_screen("event", push_history=True)

    def _set_table_item(self, row: int, col: int, value: str) -> None:
        item = QTableWidgetItem(value)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.event_table.setItem(row, col, item)

    def _set_time_table_item(self, row: int, col: int, seconds: Optional[float]) -> None:
        item = QTableWidgetItem(format_time_dk_sn_ms(seconds))
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if seconds is None:
            item.setForeground(QColor("#8d97ab"))
            item.setToolTip("Bu olay icin zaman bilgisi yok.")
        else:
            item.setForeground(QColor("#7ec8ff"))
            item.setToolTip("Tikla: ilgili frame'i goster.")
        self.event_table.setItem(row, col, item)

    def _reset_event_table(self) -> None:
        self.event_table.setRowCount(len(EVENT_DEFINITIONS))
        for row, event_info in enumerate(EVENT_DEFINITIONS):
            self._set_table_item(row, 0, str(event_info["id"]))
            self._set_table_item(row, 1, str(event_info["name"]))
            self._set_table_item(row, 2, str(event_info["target_roi"]))
            self._set_table_item(row, 3, str(event_info["type"]))
            self._set_time_table_item(row, EVENT_COL_START, None)
            self._set_time_table_item(row, EVENT_COL_END, None)
            self._set_table_item(row, 6, "0.00")

    def _populate_event_table_from_results(self) -> None:
        self._reset_event_table()
        for row, event in enumerate(self.last_detected_events):
            if row >= self.event_table.rowCount():
                break

            try:
                start_seconds = None if event.get("start") is None else float(event.get("start"))
            except (TypeError, ValueError):
                start_seconds = None
            try:
                end_seconds = None if event.get("end") is None else float(event.get("end"))
            except (TypeError, ValueError):
                end_seconds = None

            confidence_text = "0.00"
            confidence_raw = event.get("confidence")
            if self.detection_mode == DETECTION_MODE_MANUAL and confidence_raw is None:
                confidence_text = ""
            else:
                try:
                    confidence_text = f"{float(confidence_raw):.2f}"
                except (TypeError, ValueError):
                    confidence_text = "0.00"

            self._set_table_item(row, 0, str(event.get("id", row + 1)))
            self._set_table_item(row, 1, str(event.get("name", "")))
            self._set_table_item(row, 2, str(event.get("target_roi", "")))
            self._set_table_item(row, 3, str(event.get("type", "")))
            self._set_time_table_item(row, EVENT_COL_START, start_seconds)
            self._set_time_table_item(row, EVENT_COL_END, end_seconds)
            self._set_table_item(row, 6, confidence_text)

    def _set_event_frame_preview_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        if pixmap is None or pixmap.isNull():
            self.event_frame_preview.clear()
            self.event_frame_preview.setText("Start/End hucrelerine tiklayinca frame burada gorunecek.")
            return

        target_width = max(1, self.event_frame_preview.width() - 10)
        target_height = max(1, self.event_frame_preview.height() - 10)
        scaled = pixmap.scaled(
            target_width,
            target_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.event_frame_preview.setPixmap(scaled)

    def _update_event_video_label(self) -> None:
        if self.video_meta is None:
            self.event_video_label.setText("Aktif video: -")
            return
        base_name = os.path.basename(self.video_meta.source_video)
        self.event_video_label.setText(f"Aktif video: {base_name}")

    def _append_event_log(self, message: str) -> None:
        cleaned = message.strip()
        if not cleaned:
            return
        self.event_log.appendPlainText(cleaned)
        self._last_detection_log_message = cleaned

    def _is_event_detection_running(self) -> bool:
        return self._event_detection_busy or (self.detect_thread is not None and self.detect_thread.isRunning())

    def _is_color_analysis_running(self) -> bool:
        return self._color_analysis_busy or (self.color_thread is not None and self.color_thread.isRunning())

    def _selected_color_roi_name(self) -> Optional[str]:
        if self.color_roi_combo.count() <= 0:
            return None
        roi_name = self.color_roi_combo.currentText().strip()
        if not roi_name or roi_name not in self.rois_rel:
            return None
        return roi_name

    def _update_analysis_controls(self) -> None:
        event_running = self._is_event_detection_running()
        color_running = self._is_color_analysis_running()
        any_running = event_running or color_running
        is_auto = self.detection_mode == DETECTION_MODE_AUTO
        is_manual = self.detection_mode == DETECTION_MODE_MANUAL
        mode_ready = is_auto or is_manual

        self._sync_event_mode_combo_to_state()
        self.event_mode_combo.setEnabled(self.startup_completed and mode_ready and not any_running)
        self.event_open_video_button.setEnabled(not any_running)

        self.sample_hz_spin.setVisible(is_auto)
        self.source_sensitivity_label.setVisible(is_auto)
        self.source_sensitivity_combo.setVisible(is_auto)
        self.detect_button.setVisible(is_auto)
        self.manual_controls_box.setVisible(is_manual)
        if is_auto and not self.event_lower_section.isVisible():
            self.event_lower_section.setVisible(True)
            self.event_vertical_splitter.setSizes([420, 320])
        elif is_manual and self.event_lower_section.isVisible():
            self.event_lower_section.setVisible(False)
            self.event_vertical_splitter.setSizes([740, 0])

        self.sample_hz_spin.setEnabled(is_auto and not any_running)
        self.source_sensitivity_combo.setEnabled(is_auto and not any_running)

        if is_auto and event_running:
            if self._event_detection_cancel_requested:
                self.detect_button.setText("Olay Tespiti Durduruluyor...")
                self.detect_button.setEnabled(False)
            else:
                self.detect_button.setText("Olay Tespitini Durdur")
                self.detect_button.setEnabled(True)
        elif is_auto:
            self.detect_button.setText("Olaylari Tespit Et")
            self.detect_button.setEnabled(not color_running)
        else:
            self.detect_button.setText("Olaylari Tespit Et")
            self.detect_button.setEnabled(False)

        has_roi_choice = self.color_roi_combo.count() > 0
        can_start_color = (
            is_auto and has_roi_choice and self.video_meta is not None and self._selected_color_roi_name() is not None
        )
        self.color_roi_combo.setEnabled(is_auto and has_roi_choice and not any_running)
        if is_auto and color_running:
            if self._color_analysis_cancel_requested:
                self.color_analyze_button.setText("Renk Analizi Durduruluyor...")
                self.color_analyze_button.setEnabled(False)
            else:
                self.color_analyze_button.setText("Renk Analizini Durdur")
                self.color_analyze_button.setEnabled(True)
        elif is_auto:
            self.color_analyze_button.setText("ROI Renk Analizi")
            self.color_analyze_button.setEnabled(can_start_color and not event_running)
        else:
            self.color_analyze_button.setText("ROI Renk Analizi")
            self.color_analyze_button.setEnabled(False)

        if is_manual:
            self.event_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.event_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        else:
            self.event_table.clearSelection()
            self.event_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            self.event_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)

        manual_video_ready = is_manual and self.video_meta is not None
        manual_row_selected = self._manual_selected_row() is not None
        manual_controls_enabled = manual_video_ready and not any_running
        self.manual_frame_slider.setEnabled(manual_controls_enabled)
        self.manual_step_minus_sec_button.setEnabled(manual_controls_enabled)
        self.manual_step_minus_frame_button.setEnabled(manual_controls_enabled)
        self.manual_step_plus_frame_button.setEnabled(manual_controls_enabled)
        self.manual_step_plus_sec_button.setEnabled(manual_controls_enabled)
        self.manual_assign_start_button.setEnabled(manual_controls_enabled and manual_row_selected)
        self.manual_assign_end_button.setEnabled(manual_controls_enabled and manual_row_selected)
        self.manual_clear_row_button.setEnabled(manual_controls_enabled and manual_row_selected)

        self.save_timeline_button.setEnabled(mode_ready and (not any_running) and bool(self.last_detected_events))

    def _set_detection_controls(self, running: bool) -> None:
        self._event_detection_busy = bool(running)
        self._update_analysis_controls()

    def _set_color_analysis_controls(self, running: bool) -> None:
        self._color_analysis_busy = bool(running)
        self._update_analysis_controls()

    def _invalidate_event_results(self, force: bool = False) -> None:
        if self.detection_mode == DETECTION_MODE_MANUAL and not force:
            self._update_analysis_controls()
            return
        self.last_detected_events = []
        self.timeline_dirty = False
        self.event_progress.setValue(0)
        self._reset_event_table()
        self._update_analysis_controls()

    def _invalidate_roi_color_results(self, refresh_combo: bool = False) -> None:
        self.last_roi_color_payload = None
        self.color_progress.setValue(0)
        self.roi_color_timeline.clear_data()
        if refresh_combo:
            self._refresh_color_roi_combo()
        else:
            self._update_analysis_controls()

    def _refresh_color_roi_combo(self) -> None:
        current_name = self.color_roi_combo.currentText().strip()
        names: list[str] = []
        for roi_name in self.roi_order:
            if roi_name in self.rois_rel and roi_name not in names:
                names.append(roi_name)
        for roi_name in self.rois_rel:
            if roi_name not in names:
                names.append(roi_name)

        target_name: Optional[str] = None
        if current_name and current_name in names:
            target_name = current_name
        elif self.active_roi_name is not None and self.active_roi_name in names:
            target_name = self.active_roi_name
        elif names:
            target_name = names[0]

        self._syncing_color_roi_combo = True
        try:
            self.color_roi_combo.clear()
            if names:
                self.color_roi_combo.addItems(names)
                if target_name is not None:
                    target_index = self.color_roi_combo.findText(target_name)
                    if target_index >= 0:
                        self.color_roi_combo.setCurrentIndex(target_index)
        finally:
            self._syncing_color_roi_combo = False

        self._update_analysis_controls()

    def on_color_roi_changed(self, roi_name: str) -> None:
        if self._syncing_color_roi_combo:
            return

        selected_name = roi_name.strip()
        if self.last_roi_color_payload is not None:
            payload_roi = str(self.last_roi_color_payload.get("roi_name", "")).strip()
            if payload_roi and payload_roi != selected_name:
                self._invalidate_roi_color_results(refresh_combo=False)

        if selected_name:
            self.statusBar().showMessage(f"Renk ROI: {selected_name}", 2200)

    def on_color_zoom_in_clicked(self) -> None:
        if self.roi_color_timeline.zoom_in():
            self.statusBar().showMessage("Renk serisi: zoom in", 1800)

    def on_color_zoom_out_clicked(self) -> None:
        if self.roi_color_timeline.zoom_out():
            self.statusBar().showMessage("Renk serisi: zoom out", 1800)

    def on_color_zoom_reset_clicked(self) -> None:
        if self.roi_color_timeline.reset_zoom():
            self.statusBar().showMessage("Renk serisi: zoom sifirlandi", 1800)

    def _rois_to_serializable(self) -> dict:
        return {name: rect.to_dict() for name, rect in self.rois_rel.items()}

    def _build_event_detection_params(self) -> dict[str, object]:
        preset_name = self.source_sensitivity_combo.currentText().strip()
        if preset_name not in SOURCE_START_SENSITIVITY_PRESETS:
            preset_name = "Dengeli"
        preset = SOURCE_START_SENSITIVITY_PRESETS[preset_name]

        params: dict[str, object] = {
            "source_start_first4_enabled": True,
            "source_color_k": float(preset["source_color_k"]),
            "source_color_min": float(preset["source_color_min"]),
            "source_consecutive": int(round(float(preset["source_consecutive"]))),
            "source_soft_ratio": float(preset["source_soft_ratio"]),
        }
        return params

    def on_detect_button_clicked(self) -> None:
        if self._is_event_detection_running():
            self.stop_event_detection()
            return
        self.start_event_detection()

    def on_color_analyze_button_clicked(self) -> None:
        if self._is_color_analysis_running():
            self.stop_roi_color_analysis()
            return
        self.start_roi_color_analysis()

    def stop_event_detection(self) -> None:
        if not self._is_event_detection_running() or self.detect_worker is None:
            self.statusBar().showMessage("Calisan olay tespiti yok.", 2000)
            return
        if self._event_detection_cancel_requested:
            return
        self._event_detection_cancel_requested = True
        self.detect_worker.cancel()
        self.event_progress.setValue(0)
        self._append_event_log("Olay tespiti icin durdurma istendi.")
        self.statusBar().showMessage("Olay tespiti durduruluyor...", 2500)
        self._update_analysis_controls()

    def stop_roi_color_analysis(self) -> None:
        if not self._is_color_analysis_running() or self.color_worker is None:
            self.statusBar().showMessage("Calisan ROI renk analizi yok.", 2000)
            return
        if self._color_analysis_cancel_requested:
            return
        self._color_analysis_cancel_requested = True
        self.color_worker.cancel()
        self.color_progress.setValue(0)
        self._append_event_log("ROI renk analizi icin durdurma istendi.")
        self.statusBar().showMessage("ROI renk analizi durduruluyor...", 2500)
        self._update_analysis_controls()

    def start_event_detection(self) -> None:
        if self.detection_mode != DETECTION_MODE_AUTO:
            QMessageBox.information(self, "Olay Tespit", "Bu islem sadece otomatik modda kullanilir.")
            return
        if self._is_event_detection_running():
            QMessageBox.information(self, "Olay Tespit", "Analiz zaten calisiyor.")
            return
        if self._is_color_analysis_running():
            QMessageBox.information(self, "Olay Tespit", "Renk analizi calisiyor. Once tamamlanmasini bekleyin.")
            return

        if self.video_meta is None:
            QMessageBox.warning(self, "Olay Tespit", "Once bir video acin.")
            return
        if not os.path.isfile(self.video_meta.source_video):
            QMessageBox.warning(self, "Olay Tespit", "Aktif video yolu bulunamadi.")
            return

        missing = [name for name in REQUIRED_TARGET_ROIS if name not in self.rois_rel]
        if missing:
            joined = ", ".join(missing)
            QMessageBox.warning(self, "Olay Tespit", f"Eksik hedef ROI: {joined}")
            return

        self.switch_to_event_tab()
        self.last_detected_events = []
        self.timeline_dirty = False
        self.last_detection_sample_hz = int(self.sample_hz_spin.value())
        self._last_detection_log_message = ""
        self._event_detection_cancel_requested = False
        self._reset_event_table()
        self.event_log.clear()
        self.event_progress.setValue(0)
        self._set_detection_controls(running=True)
        self._append_event_log(
            f"Analiz basladi... (Ilk4 start hass.: {self.source_sensitivity_combo.currentText().strip() or 'Dengeli'})"
        )

        self.detect_thread = QThread(self)
        self.detect_worker = EventDetectionWorker(
            video_path=self.video_meta.source_video,
            rois_relative=self._rois_to_serializable(),
            sample_hz=self.last_detection_sample_hz,
            params=self._build_event_detection_params(),
        )
        self.detect_worker.moveToThread(self.detect_thread)

        self.detect_thread.started.connect(self.detect_worker.run)
        self.detect_worker.progress.connect(self.on_event_detection_progress)
        self.detect_worker.result.connect(self.on_event_detection_result)
        self.detect_worker.error.connect(self.on_event_detection_error)
        self.detect_worker.finished.connect(self.on_event_detection_finished)
        self.detect_worker.finished.connect(self.detect_thread.quit)
        self.detect_worker.finished.connect(self.detect_worker.deleteLater)
        self.detect_thread.finished.connect(self.detect_thread.deleteLater)
        self.detect_thread.finished.connect(self.on_event_detection_thread_finished)

        self.detect_thread.start()

    def on_event_detection_progress(self, percent: int, message: str) -> None:
        if self._event_detection_cancel_requested:
            return
        self.event_progress.setValue(max(0, min(100, int(percent))))
        if message.strip():
            self._append_event_log(message)
            self.statusBar().showMessage(message, 2200)

    def on_event_detection_result(self, events: list) -> None:
        self.last_detected_events = [dict(item) for item in events]
        self.timeline_dirty = True
        self._populate_event_table_from_results()
        self.event_progress.setValue(100)
        self._append_event_log("Analiz sonucu tabloya yazildi.")

    def start_roi_color_analysis(self) -> None:
        if self.detection_mode != DETECTION_MODE_AUTO:
            QMessageBox.information(self, "ROI Renk Analizi", "Renk analizi sadece otomatik modda kullanilir.")
            return
        if self._is_color_analysis_running():
            QMessageBox.information(self, "ROI Renk Analizi", "Renk analizi zaten calisiyor.")
            return
        if self._is_event_detection_running():
            QMessageBox.information(self, "ROI Renk Analizi", "Olay tespiti calisiyor. Once bitmesini bekleyin.")
            return

        if self.video_meta is None:
            QMessageBox.warning(self, "ROI Renk Analizi", "Once bir video acin.")
            return
        if not os.path.isfile(self.video_meta.source_video):
            QMessageBox.warning(self, "ROI Renk Analizi", "Aktif video yolu bulunamadi.")
            return

        roi_name = self._selected_color_roi_name()
        if roi_name is None:
            QMessageBox.warning(self, "ROI Renk Analizi", "Renk analizi icin bir ROI secin.")
            return

        roi_rect = self.rois_rel.get(roi_name)
        if roi_rect is None:
            QMessageBox.warning(self, "ROI Renk Analizi", f"Secili ROI bulunamadi: {roi_name}")
            return

        self.switch_to_event_tab()
        self._invalidate_roi_color_results(refresh_combo=False)
        self._append_event_log(f"ROI renk analizi basladi: {roi_name}")
        self._color_analysis_cancel_requested = False
        self._set_color_analysis_controls(running=True)

        self.color_thread = QThread(self)
        self.color_worker = RoiColorAnalysisWorker(
            video_path=self.video_meta.source_video,
            roi_name=roi_name,
            roi_relative=roi_rect.to_dict(),
            sample_hz=int(self.sample_hz_spin.value()),
        )
        self.color_worker.moveToThread(self.color_thread)

        self.color_thread.started.connect(self.color_worker.run)
        self.color_worker.progress.connect(self.on_roi_color_progress)
        self.color_worker.result.connect(self.on_roi_color_result)
        self.color_worker.error.connect(self.on_roi_color_error)
        self.color_worker.finished.connect(self.on_roi_color_finished)
        self.color_worker.finished.connect(self.color_thread.quit)
        self.color_worker.finished.connect(self.color_worker.deleteLater)
        self.color_thread.finished.connect(self.color_thread.deleteLater)
        self.color_thread.finished.connect(self.on_roi_color_thread_finished)

        self.color_thread.start()

    def on_roi_color_progress(self, percent: int, message: str) -> None:
        if self._color_analysis_cancel_requested:
            return
        self.color_progress.setValue(max(0, min(100, int(percent))))
        if message.strip():
            self._append_event_log(message)
            self.statusBar().showMessage(message, 2200)

    def on_roi_color_result(self, payload: dict) -> None:
        self.last_roi_color_payload = dict(payload)
        self.roi_color_timeline.set_data(self.last_roi_color_payload)
        self.color_progress.setValue(100)

        roi_name = str(self.last_roi_color_payload.get("roi_name", "")).strip() or "-"
        raw_times = self.last_roi_color_payload.get("times")
        sample_count = len(raw_times) if isinstance(raw_times, list) else 0
        self._append_event_log(f"ROI renk analizi guncellendi: {roi_name} ({sample_count} ornek)")

    def on_roi_color_time_selected(self, seconds: float, sample_index: int) -> None:
        if self.video_meta is None:
            return

        frame_payload = self._read_video_frame_at_seconds(seconds)
        if frame_payload is None:
            self.statusBar().showMessage("Frame okunamadi.", 2500)
            return

        frame_bgr, frame_index = frame_payload
        self._apply_current_frame(frame_bgr, frame_index)
        self.roi_color_timeline.set_selected_index(sample_index)

        roi_name = "-"
        if self.last_roi_color_payload is not None:
            roi_name = str(self.last_roi_color_payload.get("roi_name", "")).strip() or "-"
        pretty_time = format_time_dk_sn_ms(seconds)
        self.statusBar().showMessage(f"ROI {roi_name} zaman: {pretty_time}", 2800)

    def on_roi_color_error(self, message: str) -> None:
        self._append_event_log(f"Hata: {message}")
        QMessageBox.critical(self, "ROI Renk Analizi", message)

    def on_roi_color_finished(self) -> None:
        self._set_color_analysis_controls(running=False)
        if self._color_analysis_cancel_requested and not self.last_roi_color_payload:
            self.color_progress.setValue(0)
            self._append_event_log("ROI renk analizi kullanici tarafindan durduruldu.")
            self.statusBar().showMessage("ROI renk analizi durduruldu.", 3000)
        elif self.last_roi_color_payload:
            self.statusBar().showMessage("ROI renk analizi tamamlandi.", 3000)
        else:
            self.statusBar().showMessage("ROI renk analizi bitti, sonuc uretilmedi.", 3000)
        self._color_analysis_cancel_requested = False

    def on_roi_color_thread_finished(self) -> None:
        self.color_worker = None
        self.color_thread = None
        self._update_analysis_controls()

    def on_event_table_cell_clicked(self, row: int, col: int) -> None:
        if col not in (EVENT_COL_START, EVENT_COL_END):
            return
        if row < 0 or row >= len(self.last_detected_events):
            return
        if self.video_meta is None:
            return

        event = self.last_detected_events[row]
        field_name = "start" if col == EVENT_COL_START else "end"
        raw_seconds = event.get(field_name)
        if raw_seconds is None:
            event_id = event.get("id", row + 1)
            self.statusBar().showMessage(f"Event {event_id} icin {field_name} zamani yok.", 2500)
            return

        try:
            seconds = max(0.0, float(raw_seconds))
        except (TypeError, ValueError):
            self.statusBar().showMessage("Zaman bilgisi gecersiz.", 2500)
            return

        frame_payload = self._read_video_frame_at_seconds(seconds)
        if frame_payload is None:
            self.statusBar().showMessage("Frame okunamadi.", 2500)
            return

        frame_bgr, frame_index = frame_payload
        self._apply_current_frame(frame_bgr, frame_index)

        event_id = event.get("id", row + 1)
        pretty_time = format_time_dk_sn_ms(seconds)
        self.statusBar().showMessage(f"Event {event_id} {field_name}: {pretty_time}", 2800)

    def _read_video_frame_at_index(self, frame_index: int) -> Optional[Tuple[np.ndarray, int]]:
        if self.video_meta is None:
            return None

        video_path = self.video_meta.source_video
        if not os.path.isfile(video_path):
            return None

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            return None

        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            target_frame = max(0, int(frame_index))
            if frame_count > 0:
                target_frame = max(0, min(frame_count - 1, target_frame))
            self.video_frame_count = max(self.video_frame_count, frame_count)

            capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ok, frame = capture.read()
            if not ok or frame is None:
                return None

            reported_next = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or (target_frame + 1))
            actual_frame = max(0, reported_next - 1)
            return frame, actual_frame
        finally:
            capture.release()

    def _read_video_frame_at_seconds(self, seconds: float) -> Optional[Tuple[np.ndarray, int]]:
        if self.video_meta is None:
            return None
        fps = float(self.video_meta.fps) if self.video_meta.fps > 0 else 30.0
        target_frame = int(round(max(0.0, float(seconds)) * fps))
        return self._read_video_frame_at_index(target_frame)

    def _apply_current_frame(self, frame_bgr: np.ndarray, frame_index: int) -> None:
        if self.video_meta is None:
            return

        self.original_bgr = frame_bgr
        self.video_meta = VideoMeta(
            width=self.video_meta.width,
            height=self.video_meta.height,
            fps=self.video_meta.fps,
            frame_index=max(0, int(frame_index)),
            source_video=self.video_meta.source_video,
        )

        display_rgb = self._build_display_rgb(frame_bgr)
        display_pixmap = numpy_rgb_to_qpixmap(display_rgb)
        self.canvas.set_image_pixmap(display_pixmap)
        self.canvas.set_rois(self.rois_rel)
        self._set_event_frame_preview_pixmap(display_pixmap)
        self.update_selected_roi_panel()
        self._sync_manual_timeline_to_current_frame()

    def on_event_detection_error(self, message: str) -> None:
        self._append_event_log(f"Hata: {message}")
        QMessageBox.critical(self, "Olay Tespit", message)

    def on_event_detection_finished(self) -> None:
        self._set_detection_controls(running=False)
        if self._event_detection_cancel_requested and not self.last_detected_events:
            self.event_progress.setValue(0)
            self._append_event_log("Olay tespiti kullanici tarafindan durduruldu.")
            self.statusBar().showMessage("Olay tespiti durduruldu.", 3000)
        elif self.last_detected_events:
            self.statusBar().showMessage("Olay tespiti tamamlandi.", 3000)
        else:
            self.statusBar().showMessage("Olay tespiti bitti, sonuc uretilmedi.", 3000)
        self._event_detection_cancel_requested = False

    def on_event_detection_thread_finished(self) -> None:
        self.detect_worker = None
        self.detect_thread = None
        self._update_analysis_controls()

    def _validate_manual_events_for_save(self) -> Optional[str]:
        if len(self.last_detected_events) < len(EVENT_DEFINITIONS):
            return "Manuel kayit icin tum event satirlari olusmamis."

        for row, event_info in enumerate(EVENT_DEFINITIONS):
            event_payload = self.last_detected_events[row] if row < len(self.last_detected_events) else {}
            event_id = int(event_payload.get("id", event_info["id"]))
            raw_start = event_payload.get("start")
            raw_end = event_payload.get("end")

            if raw_start is None or raw_end is None:
                return f"Event {event_id} icin start/end bos birakilamaz."

            try:
                start_seconds = float(raw_start)
                end_seconds = float(raw_end)
            except (TypeError, ValueError):
                return f"Event {event_id} zaman bilgisi gecersiz."

            if end_seconds < start_seconds:
                return f"Event {event_id} icin start zamani end zamanindan buyuk olamaz."

        return None

    def save_timeline_json(self) -> None:
        if not self.last_detected_events:
            if self.detection_mode == DETECTION_MODE_MANUAL:
                QMessageBox.information(self, "timeline.json", "Once manuel event zamanlarini atayin.")
            else:
                QMessageBox.information(self, "timeline.json", "Once olay tespiti calistirin.")
            return
        if self.video_meta is None:
            QMessageBox.warning(self, "timeline.json", "Aktif video bilgisi yok.")
            return

        if self.detection_mode == DETECTION_MODE_MANUAL:
            validation_error = self._validate_manual_events_for_save()
            if validation_error is not None:
                QMessageBox.warning(self, "timeline.json", validation_error)
                return

        initial_dir = os.path.dirname(self.current_template_path) if self.current_template_path else os.getcwd()
        video_name = os.path.splitext(os.path.basename(self.video_meta.source_video))[0].strip()
        if not video_name:
            video_name = "video"
        initial_path = os.path.join(initial_dir, f"{video_name}_timeline.json")
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "timeline.json Kaydet",
            initial_path,
            "JSON Files (*.json);;All Files (*.*)",
        )
        if not save_path:
            return
        if not save_path.lower().endswith(".json"):
            save_path += ".json"

        template_value: object
        if self.current_template_path:
            template_value = self.current_template_path
        else:
            template_value = self._rois_to_serializable()

        payload = {
            "source_video": self.video_meta.source_video,
            "template_path": template_value,
            "sample_hz": int(self.last_detection_sample_hz),
            "events": self.last_detected_events,
        }

        try:
            with open(save_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2)
        except OSError as exc:
            QMessageBox.critical(self, "timeline.json", f"Kayit basarisiz:\n{exc}")
            return

        self.timeline_dirty = False
        self._update_analysis_controls()
        self._append_event_log(f"timeline kaydedildi: {save_path}")
        self.statusBar().showMessage(f"timeline kaydedildi: {save_path}", 4000)

    def _sanitize_roi_name(self, roi_name: str) -> str:
        return roi_name.strip()

    def _is_duplicate_roi_name(self, roi_name: str, exclude_name: Optional[str] = None) -> bool:
        target = roi_name.casefold()
        exclude = exclude_name.casefold() if exclude_name is not None else None
        for existing in self.roi_order:
            existing_folded = existing.casefold()
            if existing_folded == target and existing_folded != exclude:
                return True
        return False

    def _refresh_roi_list_widget(self) -> None:
        self._syncing_roi_list = True
        try:
            self.roi_list.clear()
            for name in self.roi_order:
                self.roi_list.addItem(name)
            if self.active_roi_name and self.active_roi_name in self.roi_order:
                self.roi_list.setCurrentRow(self.roi_order.index(self.active_roi_name))
            else:
                self.roi_list.setCurrentRow(-1)
        finally:
            self._syncing_roi_list = False

    def set_active_roi_name(self, roi_name: Optional[str], announce: bool = True) -> None:
        if roi_name is not None and roi_name not in self.roi_order:
            roi_name = None

        self.active_roi_name = roi_name
        self.canvas.set_active_roi(roi_name)
        self._refresh_roi_list_widget()
        self.update_selected_roi_panel()

        if announce:
            if roi_name is None:
                self.statusBar().showMessage("No active ROI selected", 2200)
            else:
                self.statusBar().showMessage(f"Active ROI: {roi_name}", 2200)

    def select_roi_by_index(self, roi_index: int) -> None:
        if roi_index < 0 or roi_index >= len(self.roi_order):
            return
        self.set_active_roi_name(self.roi_order[roi_index])

    def on_roi_list_current_item_changed(
        self,
        current: Optional[QListWidgetItem],
        previous: Optional[QListWidgetItem],
    ) -> None:
        del previous
        if self._syncing_roi_list:
            return
        self.set_active_roi_name(current.text() if current is not None else None)

    def add_roi(self) -> None:
        roi_name, accepted = QInputDialog.getText(self, "Add ROI", "ROI name:")
        if not accepted:
            return

        cleaned_name = self._sanitize_roi_name(roi_name)
        if not cleaned_name:
            QMessageBox.warning(self, "Add ROI", "ROI name cannot be empty.")
            return
        if self._is_duplicate_roi_name(cleaned_name):
            QMessageBox.warning(self, "Add ROI", "ROI name already exists.")
            return

        self.roi_order.append(cleaned_name)
        self.set_active_roi_name(cleaned_name)
        self._invalidate_event_results()
        self._invalidate_roi_color_results(refresh_combo=True)

    def rename_roi(self) -> None:
        current_name = self.active_roi_name
        if current_name is None:
            QMessageBox.information(self, "Rename ROI", "Select an ROI first.")
            return

        new_name, accepted = QInputDialog.getText(self, "Rename ROI", "New name:", text=current_name)
        if not accepted:
            return

        cleaned_name = self._sanitize_roi_name(new_name)
        if not cleaned_name:
            QMessageBox.warning(self, "Rename ROI", "ROI name cannot be empty.")
            return
        if self._is_duplicate_roi_name(cleaned_name, exclude_name=current_name):
            QMessageBox.warning(self, "Rename ROI", "ROI name already exists.")
            return

        if cleaned_name == current_name:
            return

        roi_index = self.roi_order.index(current_name)
        self.roi_order[roi_index] = cleaned_name

        if current_name in self.rois_rel:
            self.rois_rel[cleaned_name] = self.rois_rel.pop(current_name)
            self.canvas.set_rois(self.rois_rel)

        self.set_active_roi_name(cleaned_name)
        self._invalidate_event_results()
        self._invalidate_roi_color_results(refresh_combo=True)

    def delete_roi(self) -> None:
        current_name = self.active_roi_name
        if current_name is None:
            QMessageBox.information(self, "Delete ROI", "Select an ROI first.")
            return

        current_index = self.roi_order.index(current_name)
        self.roi_order.pop(current_index)
        self.rois_rel.pop(current_name, None)
        self.canvas.set_rois(self.rois_rel)

        next_active: Optional[str] = None
        if self.roi_order:
            if current_index < len(self.roi_order):
                next_active = self.roi_order[current_index]
            else:
                next_active = self.roi_order[-1]

        self.set_active_roi_name(next_active, announce=False)
        self._invalidate_event_results()
        self._invalidate_roi_color_results(refresh_combo=True)
        self.statusBar().showMessage(f"Deleted ROI: {current_name}", 2200)

    def on_draw_without_selection(self) -> None:
        self.statusBar().showMessage("Add or select an ROI before drawing.", 2600)

    def open_video_dialog(self) -> bool:
        initial_dir = self.settings.value(SETTINGS_LAST_VIDEO_DIR, "", type=str).strip()
        if not initial_dir or not os.path.isdir(initial_dir):
            if self.video_meta is not None:
                current_video_dir = os.path.dirname(self.video_meta.source_video)
                if os.path.isdir(current_video_dir):
                    initial_dir = current_video_dir
        if not initial_dir:
            initial_dir = os.getcwd()

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            initial_dir,
            "MP4 Files (*.mp4);;Video Files (*.mp4 *.mov *.mkv *.avi);;All Files (*.*)",
        )
        if not path:
            return False

        selected_dir = os.path.dirname(path)
        if selected_dir and os.path.isdir(selected_dir):
            self.settings.setValue(SETTINGS_LAST_VIDEO_DIR, selected_dir)

        return self.load_video(path)

    def load_video(self, path: str) -> bool:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            QMessageBox.warning(self, "Open Video", "Failed to open video file.")
            return False

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        frame_index = frame_count // 2 if frame_count > 0 else 0
        if frame_index > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_index = 0
            ok, frame = capture.read()

        capture.release()

        if not ok or frame is None:
            QMessageBox.warning(self, "Open Video", "Could not read a frame from this video.")
            return False

        self.video_meta = VideoMeta(
            width=width if width > 0 else frame.shape[1],
            height=height if height > 0 else frame.shape[0],
            fps=fps,
            frame_index=frame_index,
            source_video=path,
        )
        self.video_frame_count = max(0, frame_count)
        video_dir = os.path.dirname(path)
        if video_dir and os.path.isdir(video_dir):
            self.settings.setValue(SETTINGS_LAST_VIDEO_DIR, video_dir)
        self._apply_current_frame(frame, frame_index)

        if self.pending_template_rois is not None:
            pending_order = self.pending_template_order or list(self.pending_template_rois.keys())
            self._apply_roi_state(self.pending_template_rois, pending_order)
            self.pending_template_rois = None
            self.pending_template_order = None
        else:
            self.canvas.set_rois(self.rois_rel)
            self.set_active_roi_name(self.active_roi_name, announce=False)

        self._update_event_video_label()
        if self.detection_mode == DETECTION_MODE_MANUAL:
            self._initialize_manual_events()
            self.switch_to_event_tab()
        else:
            self._invalidate_event_results()
        self._invalidate_roi_color_results(refresh_combo=True)
        self.statusBar().showMessage(f"Video loaded: {os.path.basename(path)}", 3000)
        return True

    def _build_display_rgb(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_h, frame_w = frame_bgr.shape[:2]
        scale = min(1.0, MAX_DISPLAY_WIDTH / float(frame_w), MAX_DISPLAY_HEIGHT / float(frame_h))

        if scale < 1.0:
            target_w = max(1, int(round(frame_w * scale)))
            target_h = max(1, int(round(frame_h * scale)))
            resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            resized = frame_bgr

        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def _compute_roi_stats(self, rect: RelativeRect) -> Optional[RoiStats]:
        if self.original_bgr is None:
            return None

        frame_h, frame_w = self.original_bgr.shape[:2]
        pixel_rect = relative_to_pixel_rect(rect, width=frame_w, height=frame_h)
        if pixel_rect is None:
            return None

        x, y, w, h = pixel_rect
        crop = self.original_bgr[y : y + h, x : x + w]
        if crop.size == 0:
            return None

        preview_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        preview_pixmap = numpy_rgb_to_qpixmap(preview_rgb)

        return RoiStats(
            pixel_rect=pixel_rect,
            relative_rect=rect,
            preview_pixmap=preview_pixmap,
        )

    def update_selected_roi_panel(self) -> None:
        selected_name = self.active_roi_name
        self.selected_roi_card.set_roi_name(selected_name)

        if selected_name is None:
            self.selected_roi_card.set_empty(None)
            return

        rect = self.rois_rel.get(selected_name)
        if rect is None:
            self.selected_roi_card.set_empty(None)
            return

        stats = self._compute_roi_stats(rect)
        if stats is None:
            self.selected_roi_card.set_empty(rect)
        else:
            self.selected_roi_card.set_stats(stats)

    def on_roi_drawn(self, roi_name: str, rect: RelativeRect) -> None:
        if roi_name not in self.roi_order:
            self.roi_order.append(roi_name)
        self.rois_rel[roi_name] = rect
        self.canvas.set_rois(self.rois_rel)
        self.set_active_roi_name(roi_name, announce=False)
        self._invalidate_event_results()
        self._invalidate_roi_color_results(refresh_combo=True)

    def save_template(self) -> None:
        if self.video_meta is None or self.original_bgr is None:
            QMessageBox.warning(self, "Save Template", "Open a video before saving a template.")
            return

        ordered_saved_names: list[str] = []
        for roi_name in self.roi_order:
            if roi_name in self.rois_rel:
                ordered_saved_names.append(roi_name)
        for roi_name in self.rois_rel:
            if roi_name not in ordered_saved_names:
                ordered_saved_names.append(roi_name)

        if not ordered_saved_names:
            QMessageBox.warning(self, "Save Template", "At least one ROI rectangle is required.")
            return

        initial_path = self.current_template_path or os.path.join(os.getcwd(), "template.json")
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Template",
            initial_path,
            "JSON Files (*.json);;All Files (*.*)",
        )
        if not save_path:
            return

        if not save_path.lower().endswith(".json"):
            save_path += ".json"

        payload = {
            "version": TEMPLATE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_video": self.video_meta.source_video,
            "video": {
                "width": self.video_meta.width,
                "height": self.video_meta.height,
                "fps": self.video_meta.fps,
                "frame_index": self.video_meta.frame_index,
            },
            "roi_order": ordered_saved_names,
            "rois": {roi_name: self.rois_rel[roi_name].to_dict() for roi_name in ordered_saved_names},
        }

        try:
            with open(save_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2)
        except OSError as exc:
            QMessageBox.critical(self, "Save Template", f"Failed to save template:\n{exc}")
            return

        self.current_template_path = save_path
        self.settings.setValue(SETTINGS_LAST_TEMPLATE_PATH, save_path)
        self.statusBar().showMessage(f"Template saved: {save_path}", 4000)

    def load_template_dialog(self) -> None:
        initial_path = self.current_template_path or os.getcwd()
        load_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Template",
            initial_path,
            "JSON Files (*.json);;All Files (*.*)",
        )
        if not load_path:
            return

        self.load_template_from_path(load_path, silent=False)

    def _extract_template_data(self, payload: object) -> Optional[Tuple[Dict[str, RelativeRect], list[str]]]:
        if not isinstance(payload, dict):
            return None

        rois_raw = payload.get("rois")
        if not isinstance(rois_raw, dict):
            return None

        parsed_rois: Dict[str, RelativeRect] = {}
        seen_casefold: set[str] = set()
        for raw_name, raw_rect in rois_raw.items():
            if not isinstance(raw_name, str):
                continue
            cleaned_name = raw_name.strip()
            if not cleaned_name:
                continue
            folded_name = cleaned_name.casefold()
            if folded_name in seen_casefold:
                continue

            rect = parse_relative_rect(raw_rect)
            if rect is None:
                continue

            parsed_rois[cleaned_name] = rect
            seen_casefold.add(folded_name)

        if not parsed_rois:
            return None

        parsed_names = list(parsed_rois.keys())
        order_raw = payload.get("roi_order")

        ordered_names: list[str] = []
        used: set[str] = set()
        if isinstance(order_raw, list):
            for raw_name in order_raw:
                if not isinstance(raw_name, str):
                    continue
                target = raw_name.strip()
                if not target:
                    continue
                actual = next((name for name in parsed_names if name.casefold() == target.casefold()), None)
                if actual is None:
                    continue
                folded_name = actual.casefold()
                if folded_name in used:
                    continue
                ordered_names.append(actual)
                used.add(folded_name)

        for name in parsed_names:
            folded_name = name.casefold()
            if folded_name not in used:
                ordered_names.append(name)
                used.add(folded_name)

        return parsed_rois, ordered_names

    def _apply_roi_state(self, rois_rel: Dict[str, RelativeRect], roi_order: list[str]) -> None:
        self.rois_rel = dict(rois_rel)

        normalized_order: list[str] = []
        used: set[str] = set()
        for name in roi_order:
            folded_name = name.casefold()
            if folded_name in used:
                continue
            if name not in self.rois_rel:
                continue
            normalized_order.append(name)
            used.add(folded_name)

        for name in self.rois_rel:
            folded_name = name.casefold()
            if folded_name in used:
                continue
            normalized_order.append(name)
            used.add(folded_name)

        self.roi_order = normalized_order
        self.canvas.set_rois(self.rois_rel)

        if self.active_roi_name in self.roi_order:
            self.set_active_roi_name(self.active_roi_name, announce=False)
        elif self.roi_order:
            self.set_active_roi_name(self.roi_order[0], announce=False)
        else:
            self.set_active_roi_name(None, announce=False)

        self._invalidate_event_results()
        self._invalidate_roi_color_results(refresh_combo=True)

    def load_template_from_path(self, template_path: str, silent: bool) -> bool:
        try:
            with open(template_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            if not silent:
                QMessageBox.warning(self, "Load Template", f"Failed to read template:\n{exc}")
            return False

        extracted = self._extract_template_data(payload)
        if extracted is None:
            if not silent:
                QMessageBox.warning(self, "Load Template", "Template does not contain valid ROI data.")
            return False

        rois, roi_order = extracted
        self.current_template_path = template_path
        self.settings.setValue(SETTINGS_LAST_TEMPLATE_PATH, template_path)

        if self.original_bgr is None:
            self.pending_template_rois = rois
            self.pending_template_order = roi_order
        else:
            self._apply_roi_state(rois, roi_order)
            self.pending_template_rois = None
            self.pending_template_order = None

        if not silent:
            self.statusBar().showMessage(f"Template loaded: {template_path}", 4000)

        return True

    def _try_autoload_last_template(self) -> None:
        stored = self.settings.value(SETTINGS_LAST_TEMPLATE_PATH, "", type=str)
        if not stored:
            return
        if not os.path.isfile(stored):
            return
        self.load_template_from_path(stored, silent=True)

    def reset_rois(self) -> None:
        self.rois_rel.clear()
        self.roi_order.clear()
        self.active_roi_name = None
        self.pending_template_rois = None
        self.pending_template_order = None
        self.canvas.set_rois(self.rois_rel)
        self.set_active_roi_name(None, announce=False)
        self._invalidate_event_results()
        self._invalidate_roi_color_results(refresh_combo=True)
        self.statusBar().showMessage("ROIs reset", 2000)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.detect_worker is not None:
            self.detect_worker.cancel()
        if self.detect_thread is not None and self.detect_thread.isRunning():
            self.detect_thread.quit()
            self.detect_thread.wait(1500)
        if self.color_worker is not None:
            self.color_worker.cancel()
        if self.color_thread is not None and self.color_thread.isRunning():
            self.color_thread.quit()
            self.color_thread.wait(1500)
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
