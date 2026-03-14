
from __future__ import annotations

import colorsys
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QKeyEvent, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QSpinBox,
    QStyledItemDelegate,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from event_detector import EVENT_DEFINITIONS, REQUIRED_TARGET_ROIS
from event_worker import EventDetectionWorker, RoiColorAnalysisWorker, VideoEditWorker

MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 1200
MIN_RELATIVE_SIZE = 0.002
MAX_SHORTCUT_ROIS = 9
TEMPLATE_VERSION = "1.1"

SETTINGS_ORGANIZATION = "YSN"
SETTINGS_APPLICATION = "VideoEditorROI"
SETTINGS_LAST_TEMPLATE_PATH = "last_template_path"
SETTINGS_LAST_VIDEO_DIR = "last_video_dir"
SETTINGS_FFMPEG_PATH = "ffmpeg_path"
AUTO_EVENT_TIMELINE_FILENAME = "olaylar_timeline.json"
EVENT_COL_START = 4
EVENT_COL_END = 5
EVENT_COL_CONFIDENCE = 6
EVENT_COL_TARGET_ROI = 2
EVENT_COL_TYPE = 3
EDIT_TEXT_COL_ID = 0
EDIT_TEXT_COL_TEXT = 1
EDIT_TEXT_COL_START = 2
EDIT_TEXT_COL_END = 3
EDIT_TEXT_COL_POSITION = 4
EDIT_TEXT_COL_SIZE = 5
EDIT_TEXT_COL_COLOR = 6
EDIT_TEXT_COL_BOLD = 7
EDIT_TEXT_COL_ITALIC = 8
EDIT_IMAGE_COL_FILE = 0
EDIT_IMAGE_COL_START = 1
EDIT_IMAGE_COL_END = 2
EDIT_IMAGE_COL_POSITION = 3
EDIT_IMAGE_COL_SIZE = 4
EDIT_EXTERNAL_AUDIO_COL_FILE = 0
EDIT_EXTERNAL_AUDIO_COL_START = 1
EDIT_EXTERNAL_AUDIO_COL_END = 2
EDIT_EXTERNAL_AUDIO_COL_VOLUME = 3
EDIT_EXTERNAL_AUDIO_START_EVENT_ROLE = Qt.ItemDataRole.UserRole
DETECTION_MODE_AUTO = "auto"
DETECTION_MODE_MANUAL = "manual"
EVENT_TABLE_VISIBLE_ROWS = 10
MANUAL_EVENT_NAMES: list[str] = [
    "sol1_al1",
    "sol1_koy1",
    "sol2_al1",
    "sol2_koy1",
    "sol2_al2",
    "sol2_koy2",
    "sol3_al1",
    "sol3_koy1",
    "sol3_al2",
    "sol3_koy2",
    "sol3_al3",
    "sol3_koy3",
    "sol4_al1",
    "sol4_koy1",
    "sol4_al2",
    "sol4_koy2",
    "sol4_al3",
    "sol4_koy3",
    "sol4_al4",
    "sol4_koy4",
    "sag4_al1",
    "sag4_koy1",
    "sag3_al1",
    "sag3_koy1",
    "sag3_al2",
    "sag3_koy2",
    "sag2_al1",
    "sag2_koy1",
    "sag2_al2",
    "sag2_koy2",
    "sag2_al3",
    "sag2_koy3",
    "sag1_al1",
    "sag1_koy1",
    "sag1_al2",
    "sag1_koy2",
    "sag1_al3",
    "sag1_koy3",
    "sag1_al4",
    "sag1_koy4",
]
MANUAL_EVENT_DEFINITIONS: list[dict[str, object]] = [
    {
        "id": idx + 1,
        "name": name,
        "target_roi": "",
        "type": "manual",
    }
    for idx, name in enumerate(MANUAL_EVENT_NAMES)
]
CUT_EVENT_SEGMENT_RULES: list[tuple[str, str, float]] = [
    ("sol1_al1", "sol4_koy4", 0.0),
    ("sag4_al1", "sag4_koy1", 3.0),
    ("sag3_al1", "sag3_koy2", 3.0),
    ("sag2_al1", "sag2_koy3", 3.0),
    ("sag1_al1", "sag1_koy4", 3.0),
]
CUT_VIDEO_TAIL_SECONDS = 3.0
EDIT_RESOLUTION_PRESETS: list[tuple[str, tuple[int, int]]] = [
    ("2160p (3840x2160)", (3840, 2160)),
    ("1440p (2560x1440)", (2560, 1440)),
    ("1080p (1920x1080)", (1920, 1080)),
    ("720p (1280x720)", (1280, 720)),
    ("480p (854x480)", (854, 480)),
]
EDIT_FPS_PRESETS: list[float] = [60.0, 50.0, 30.0, 25.0, 24.0]
EDIT_SPEED_PRESETS: list[tuple[str, float]] = [
    ("0.5x", 0.5),
    ("0.75x", 0.75),
    ("1.0x", 1.0),
    ("1.25x", 1.25),
    ("1.5x", 1.5),
    ("2.0x", 2.0),
]

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

QCheckBox {
    spacing: 6px;
    min-height: 22px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
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


class TimeOverlayProgressBar(QProgressBar):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._elapsed_seconds: Optional[float] = None
        self._remaining_seconds: Optional[float] = None
        self.setTextVisible(False)

    def set_time_fields(self, elapsed_seconds: Optional[float], remaining_seconds: Optional[float]) -> None:
        elapsed_value = None if elapsed_seconds is None else max(0.0, float(elapsed_seconds))
        remaining_value = None if remaining_seconds is None else max(0.0, float(remaining_seconds))
        if self._elapsed_seconds == elapsed_value and self._remaining_seconds == remaining_value:
            return
        self._elapsed_seconds = elapsed_value
        self._remaining_seconds = remaining_value
        self.update()

    @staticmethod
    def _format_clock(seconds: Optional[float]) -> str:
        if seconds is None:
            return "--:--"
        total_seconds = max(0, int(round(float(seconds))))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setPen(QColor("#f4f5f7"))
        text_rect = self.rect().adjusted(8, 0, -8, 0)
        left_text = f"Gecen: {self._format_clock(self._elapsed_seconds)}"
        right_text = f"Kalan: {self._format_clock(self._remaining_seconds)}"
        percent_text = f"{max(0, min(100, int(self.value())))}%"
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, left_text)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, right_text)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, percent_text)


class FloatSpinDelegate(QStyledItemDelegate):
    def __init__(
        self,
        minimum: float,
        maximum: float,
        decimals: int,
        single_step: float,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.decimals = max(0, int(decimals))
        self.single_step = float(single_step)

    def createEditor(self, parent, _option, _index):
        editor = QDoubleSpinBox(parent)
        editor.setFrame(False)
        editor.setDecimals(self.decimals)
        editor.setRange(self.minimum, self.maximum)
        editor.setSingleStep(self.single_step)
        editor.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        return editor

    def setEditorData(self, editor, index) -> None:
        if not isinstance(editor, QDoubleSpinBox):
            return
        raw_value = index.model().data(index, Qt.ItemDataRole.EditRole)
        try:
            value = float(str(raw_value).strip())
        except (TypeError, ValueError):
            value = self.minimum
        editor.setValue(max(self.minimum, min(self.maximum, value)))

    def setModelData(self, editor, model, index) -> None:
        if not isinstance(editor, QDoubleSpinBox):
            return
        value = float(editor.value())
        text = f"{value:.{self.decimals}f}".rstrip("0").rstrip(".")
        if not text:
            text = "0"
        model.setData(index, text, Qt.ItemDataRole.EditRole)


class IntSpinDelegate(QStyledItemDelegate):
    def __init__(self, minimum: int, maximum: int, single_step: int = 1, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.minimum = int(minimum)
        self.maximum = int(maximum)
        self.single_step = max(1, int(single_step))

    def createEditor(self, parent, _option, _index):
        editor = QSpinBox(parent)
        editor.setFrame(False)
        editor.setRange(self.minimum, self.maximum)
        editor.setSingleStep(self.single_step)
        editor.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        return editor

    def setEditorData(self, editor, index) -> None:
        if not isinstance(editor, QSpinBox):
            return
        raw_value = index.model().data(index, Qt.ItemDataRole.EditRole)
        try:
            value = int(float(str(raw_value).strip()))
        except (TypeError, ValueError):
            value = self.minimum
        editor.setValue(max(self.minimum, min(self.maximum, value)))

    def setModelData(self, editor, model, index) -> None:
        if not isinstance(editor, QSpinBox):
            return
        model.setData(index, str(int(editor.value())), Qt.ItemDataRole.EditRole)


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
        self.edit_thread: Optional[QThread] = None
        self.edit_worker: Optional[VideoEditWorker] = None
        self.last_detected_events: list[dict] = []
        self.last_roi_color_payload: Optional[dict] = None
        self.edit_segments: list[tuple[float, float]] = []
        self.last_detection_sample_hz = 10
        self._last_detection_log_message = ""
        self._last_edit_log_message = ""
        self._syncing_color_roi_combo = False
        self._syncing_mode_combo = False
        self._syncing_manual_slider = False
        self._event_detection_busy = False
        self._color_analysis_busy = False
        self._video_edit_busy = False
        self._event_detection_cancel_requested = False
        self._color_analysis_cancel_requested = False
        self._video_edit_cancel_requested = False
        self._video_edit_started_monotonic: Optional[float] = None
        self._edit_preview_first_frame_bgr: Optional[np.ndarray] = None
        self._edit_preview_first_frame_source: str = ""
        self.detection_mode: Optional[str] = None
        self.timeline_dirty = False
        self._suspend_event_timeline_autoload = False
        self.startup_tab: Optional[QWidget] = None
        self.video_select_tab: Optional[QWidget] = None
        self.edit_tab: Optional[QWidget] = None
        self.startup_completed = False
        self._screen_history: list[str] = []
        self._current_screen = "video"

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

        self.screen_title_label = QLabel("Video Secimi")
        self.screen_title_label.setStyleSheet("font-size: 16px; font-weight: 600; color: #f4f5f7;")
        self.screen_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.nav_right_spacer = QWidget(self)
        self.nav_right_spacer.setFixedWidth(90)

        nav_layout.addWidget(self.back_button)
        nav_layout.addStretch(1)
        nav_layout.addWidget(self.screen_title_label)
        nav_layout.addStretch(1)
        nav_layout.addWidget(self.nav_right_spacer)
        root_layout.addWidget(nav_bar)

        self.main_stack = QStackedWidget(self)
        root_layout.addWidget(self.main_stack, stretch=1)

        self.startup_tab = QWidget(self)
        self._build_startup_tab(self.startup_tab)
        self.main_stack.addWidget(self.startup_tab)

        self.video_select_tab = QWidget(self)
        self._build_video_select_tab(self.video_select_tab)
        self.main_stack.addWidget(self.video_select_tab)

        self.roi_tab = QWidget(self)
        self._build_roi_tab(self.roi_tab)
        self.main_stack.addWidget(self.roi_tab)

        self.event_tab = QWidget(self)
        self._build_event_tab(self.event_tab)
        self.main_stack.addWidget(self.event_tab)

        self.edit_tab = QWidget(self)
        self._build_edit_tab(self.edit_tab)
        self.main_stack.addWidget(self.edit_tab)

        self._set_current_screen("video", push_history=False)

    def _screen_title(self, screen: str) -> str:
        if screen == "startup":
            return ""
        if screen == "video":
            return ""
        if screen == "roi":
            return "ROI Secimi"
        if screen == "event":
            return "Olay Tespit"
        if screen == "edit":
            return "Video Edit"
        return "Ekran"

    def _screen_widget(self, screen: str) -> Optional[QWidget]:
        if screen == "startup":
            return self.startup_tab
        if screen == "video":
            return self.video_select_tab
        if screen == "roi":
            return self.roi_tab
        if screen == "event":
            return self.event_tab
        if screen == "edit":
            return self.edit_tab
        return None

    def _set_current_screen(self, screen: str, push_history: bool) -> bool:
        target = self._screen_widget(screen)
        if target is None:
            return False

        if screen == self._current_screen:
            if self.main_stack.currentWidget() is not target:
                self.main_stack.setCurrentWidget(target)
            self._update_navigation_bar()
            return True

        current = self._screen_widget(self._current_screen)
        if push_history and current is not None:
            self._screen_history.append(self._current_screen)

        self._current_screen = screen
        self.main_stack.setCurrentWidget(target)
        self._update_navigation_bar()
        if screen == "event":
            self._try_auto_import_default_timeline()
        return True

    def _update_navigation_bar(self) -> None:
        hide_back_button = self._current_screen == "video"
        back_visible = not hide_back_button
        self.back_button.setVisible(back_visible)
        self.back_button.setEnabled(back_visible and bool(self._screen_history))
        self.nav_right_spacer.setVisible(back_visible)
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
        wrapper_layout.addWidget(button_box)
        wrapper.setMaximumWidth(460)

        center_row = QHBoxLayout()
        center_row.addStretch(1)
        center_row.addWidget(wrapper)
        center_row.addStretch(1)

        layout.addLayout(center_row)
        layout.addStretch(2)

    def _build_video_select_tab(self, container: QWidget) -> None:
        layout = QVBoxLayout(container)
        layout.addStretch(1)

        title = QLabel("Video Secimi")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: 700;")

        self.video_select_continue_button = QPushButton("Video Sec ve Devam Et")
        self.video_select_continue_button.setMinimumHeight(46)
        self.video_select_continue_button.clicked.connect(self.on_video_select_screen_pick_clicked)

        wrapper = QWidget(container)
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(8)
        wrapper_layout.addWidget(title)
        wrapper_layout.addWidget(self.video_select_continue_button)
        wrapper.setMaximumWidth(520)

        center_row = QHBoxLayout()
        center_row.addStretch(1)
        center_row.addWidget(wrapper)
        center_row.addStretch(1)

        layout.addLayout(center_row)
        layout.addStretch(2)
        self._update_video_select_screen_text()

    def _update_video_select_screen_text(self) -> None:
        return

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
        self.load_timeline_button = QPushButton("timeline.json Yukle")
        self.load_timeline_button.clicked.connect(self.load_timeline_json)
        top_header_layout = QHBoxLayout()
        top_header_layout.setContentsMargins(0, 0, 0, 0)
        top_header_layout.addWidget(self.event_video_label, stretch=1)
        top_header_layout.addWidget(self.load_timeline_button)
        layout.addLayout(top_header_layout)

        controls_layout = QHBoxLayout()
        self.event_mode_label = QLabel("Mod:")
        self.event_mode_combo = QComboBox()
        self.event_mode_combo.addItem("Otomatik", DETECTION_MODE_AUTO)
        self.event_mode_combo.addItem("Manuel", DETECTION_MODE_MANUAL)

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

        controls_layout.addWidget(self.sample_hz_spin)
        controls_layout.addWidget(self.source_sensitivity_label)
        controls_layout.addWidget(self.source_sensitivity_combo)
        controls_layout.addWidget(self.detect_button)
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

        self.event_auto_controls_widget = QWidget(right_panel)
        self.event_auto_controls_widget.setLayout(top_row_layout)

        self.event_table = QTableWidget(len(EVENT_DEFINITIONS), 7)
        self.event_table.setHorizontalHeaderLabels(
            ["id", "name", "target_roi", "type", "start", "end", "confidence"]
        )
        self.event_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.event_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.event_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.event_table.verticalHeader().setVisible(False)
        self.event_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        header = self.event_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.event_table.cellClicked.connect(self.on_event_table_cell_clicked)
        self.event_table.itemSelectionChanged.connect(self.on_event_table_selection_changed)

        self.event_table_frame = QFrame(right_panel)
        self.event_table_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.event_table_frame.setFrameShadow(QFrame.Shadow.Plain)
        event_table_frame_layout = QVBoxLayout(self.event_table_frame)
        event_table_frame_layout.setContentsMargins(0, 0, 0, 0)
        event_table_frame_layout.setSpacing(0)
        event_table_frame_layout.addWidget(self.event_table)

        self.manual_controls_box = QGroupBox("Manuel Olay Atama")
        self.manual_controls_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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
        self.manual_step_minus_500ms_button = QPushButton("-500 ms")
        self.manual_step_minus_250ms_button = QPushButton("-250 ms")
        self.manual_step_minus_100ms_button = QPushButton("-100 ms")
        self.manual_step_minus_frame_button = QPushButton("-1 f")
        self.manual_step_plus_frame_button = QPushButton("+1 f")
        self.manual_step_plus_100ms_button = QPushButton("+100 ms")
        self.manual_step_plus_250ms_button = QPushButton("+250 ms")
        self.manual_step_plus_500ms_button = QPushButton("+500 ms")
        self.manual_step_plus_sec_button = QPushButton("+1 sn")
        self.manual_step_minus_sec_button.clicked.connect(lambda: self.step_manual_seconds(-1.0))
        self.manual_step_minus_500ms_button.clicked.connect(lambda: self.step_manual_seconds(-0.5))
        self.manual_step_minus_250ms_button.clicked.connect(lambda: self.step_manual_seconds(-0.25))
        self.manual_step_minus_100ms_button.clicked.connect(lambda: self.step_manual_seconds(-0.1))
        self.manual_step_minus_frame_button.clicked.connect(lambda: self.step_manual_frame(-1))
        self.manual_step_plus_frame_button.clicked.connect(lambda: self.step_manual_frame(1))
        self.manual_step_plus_100ms_button.clicked.connect(lambda: self.step_manual_seconds(0.1))
        self.manual_step_plus_250ms_button.clicked.connect(lambda: self.step_manual_seconds(0.25))
        self.manual_step_plus_500ms_button.clicked.connect(lambda: self.step_manual_seconds(0.5))
        self.manual_step_plus_sec_button.clicked.connect(lambda: self.step_manual_seconds(1.0))
        step_layout.addWidget(self.manual_step_minus_sec_button)
        step_layout.addWidget(self.manual_step_minus_500ms_button)
        step_layout.addWidget(self.manual_step_minus_250ms_button)
        step_layout.addWidget(self.manual_step_minus_100ms_button)
        step_layout.addWidget(self.manual_step_minus_frame_button)
        step_layout.addWidget(self.manual_step_plus_frame_button)
        step_layout.addWidget(self.manual_step_plus_100ms_button)
        step_layout.addWidget(self.manual_step_plus_250ms_button)
        step_layout.addWidget(self.manual_step_plus_500ms_button)
        step_layout.addWidget(self.manual_step_plus_sec_button)
        step_layout.addStretch(1)

        assign_layout = QHBoxLayout()
        self.manual_assign_hint_label = QLabel("Satir secin, sonra Set Start ile atayin.")
        self.manual_assign_hint_label.setStyleSheet("color: #9aa4b7;")
        assign_layout.addWidget(self.manual_assign_hint_label, stretch=1)
        self.manual_set_start_button = QPushButton("Set Start")
        self.manual_set_start_button.clicked.connect(self.on_manual_set_start_clicked)
        assign_layout.addWidget(self.manual_set_start_button)

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
        upper_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        upper_layout.addWidget(self.event_auto_controls_widget)
        upper_layout.addWidget(self.manual_controls_box)
        upper_layout.addWidget(self.event_table_frame)
        event_table_actions_layout = QHBoxLayout()
        event_table_actions_layout.setContentsMargins(0, 0, 0, 0)
        event_table_actions_layout.addStretch(1)
        event_table_actions_layout.addWidget(self.save_timeline_button)
        self.edit_button = QPushButton("Edit")
        self.edit_button.setEnabled(False)
        self.edit_button.clicked.connect(self.on_open_edit_clicked)
        event_table_actions_layout.addWidget(self.edit_button)
        upper_layout.addLayout(event_table_actions_layout)
        upper_layout.addStretch(1)

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
        self._update_event_table_frame_height()
        self._refresh_color_roi_combo()
        self._update_analysis_controls()

    def _build_edit_tab(self, container: QWidget) -> None:
        root_layout = QVBoxLayout(container)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(8)

        edit_splitter = QSplitter(Qt.Orientation.Horizontal, container)
        left_panel = QWidget(edit_splitter)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        top_content = QWidget(left_panel)
        top_layout = QVBoxLayout(top_content)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)

        top_scroll = QScrollArea(left_panel)
        top_scroll.setWidgetResizable(True)
        top_scroll.setFrameShape(QFrame.Shape.NoFrame)
        top_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        top_scroll.setWidget(top_content)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        self.edit_source_video_label = QLabel("Kaynak video: -")
        header_layout.addWidget(self.edit_source_video_label, stretch=1, alignment=Qt.AlignmentFlag.AlignLeft)

        output_header_widget = QWidget(top_content)
        output_header_layout = QHBoxLayout(output_header_widget)
        output_header_layout.setContentsMargins(0, 0, 0, 0)
        output_header_layout.setSpacing(6)
        output_header_layout.addWidget(QLabel("Cikti Klasoru:"))
        self.edit_output_path_edit = QLineEdit()
        self.edit_output_path_edit.setPlaceholderText("Cikti klasor yolu")
        self.edit_output_path_edit.setMinimumWidth(360)
        self.edit_output_path_edit.textChanged.connect(self._update_edit_controls)
        self.edit_output_browse_button = QPushButton("Gozat")
        self.edit_output_browse_button.clicked.connect(self.on_edit_output_browse_clicked)
        output_header_layout.addWidget(self.edit_output_path_edit)
        output_header_layout.addWidget(self.edit_output_browse_button)
        header_layout.addWidget(output_header_widget, 0, Qt.AlignmentFlag.AlignRight)
        top_layout.addLayout(header_layout)

        self.edit_cut_group = QGroupBox("Cut Ayarlari")
        cut_layout = QVBoxLayout(self.edit_cut_group)
        cut_layout.setContentsMargins(8, 8, 8, 8)
        cut_layout.setSpacing(6)

        self.edit_cut_enabled_checkbox = QCheckBox("Enable Cut")
        self.edit_cut_enabled_checkbox.setChecked(True)
        self.edit_cut_enabled_checkbox.stateChanged.connect(self._on_edit_operation_checkbox_changed)

        self.edit_cut_info_label = QLabel("i")
        self.edit_cut_info_label.setFixedSize(18, 18)
        self.edit_cut_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.edit_cut_info_label.setCursor(Qt.CursorShape.WhatsThisCursor)
        self.edit_cut_info_label.setToolTip(self._build_cut_rule_tooltip())
        self.edit_cut_info_label.setStyleSheet(
            "background-color: #263142; color: #dbe5f5; border: 1px solid #3d4c61; "
            "border-radius: 9px; font-weight: 700;"
        )

        self.edit_segments_label = QLabel("Hazir segment: -")
        self.edit_segments_label.setStyleSheet("color: #9aa4b7;")

        cut_header_layout = QHBoxLayout()
        cut_header_layout.setContentsMargins(0, 0, 0, 0)
        cut_header_layout.setSpacing(6)
        cut_header_layout.addWidget(self.edit_cut_enabled_checkbox)
        cut_header_layout.addStretch(1)
        cut_header_layout.addWidget(self.edit_cut_info_label, 0, Qt.AlignmentFlag.AlignVCenter)

        quality_layout = QHBoxLayout()
        quality_layout.setContentsMargins(0, 0, 0, 0)
        self.edit_preset_label = QLabel("Preset:")
        quality_layout.addWidget(self.edit_preset_label)
        self.edit_preset_combo = QComboBox()
        self.edit_preset_combo.addItems(["ultrafast", "fast", "medium", "slow"])
        self.edit_preset_combo.setCurrentText("medium")
        self.edit_preset_combo.currentIndexChanged.connect(self._update_edit_quality_tooltips)
        quality_layout.addWidget(self.edit_preset_combo)
        self.edit_crf_label = QLabel("CRF:")
        quality_layout.addWidget(self.edit_crf_label)
        self.edit_crf_spin = QSpinBox()
        self.edit_crf_spin.setRange(0, 51)
        self.edit_crf_spin.setValue(25)
        self.edit_crf_spin.valueChanged.connect(self._update_edit_quality_tooltips)
        quality_layout.addWidget(self.edit_crf_spin)
        quality_layout.addStretch(1)

        cut_layout.addLayout(cut_header_layout)
        cut_layout.addWidget(self.edit_segments_label)
        cut_layout.addLayout(quality_layout)
        top_layout.addWidget(self.edit_cut_group)

        self.edit_resize_group = QGroupBox("Cozunurluk Ayarlari")
        resize_layout = QVBoxLayout(self.edit_resize_group)
        resize_layout.setContentsMargins(8, 8, 8, 8)
        resize_layout.setSpacing(6)

        self.edit_resize_enabled_checkbox = QCheckBox("Enable Cozunurluk/FPS")
        self.edit_resize_enabled_checkbox.setChecked(False)
        self.edit_resize_enabled_checkbox.stateChanged.connect(self._on_edit_operation_checkbox_changed)

        self.edit_current_specs_label = QLabel("Mevcut: -")
        self.edit_current_specs_label.setStyleSheet("color: #9aa4b7;")

        resize_target_layout = QHBoxLayout()
        resize_target_layout.setContentsMargins(0, 0, 0, 0)
        resize_target_layout.addWidget(QLabel("Hedef Cozunurluk:"))
        self.edit_target_resolution_combo = QComboBox()
        self.edit_target_resolution_combo.currentIndexChanged.connect(self._update_edit_controls)
        resize_target_layout.addWidget(self.edit_target_resolution_combo)
        resize_target_layout.addWidget(QLabel("Hedef FPS:"))
        self.edit_target_fps_combo = QComboBox()
        self.edit_target_fps_combo.currentIndexChanged.connect(self._update_edit_controls)
        resize_target_layout.addWidget(self.edit_target_fps_combo)
        resize_target_layout.addStretch(1)

        resize_layout.addWidget(self.edit_resize_enabled_checkbox)
        resize_layout.addWidget(self.edit_current_specs_label)
        resize_layout.addLayout(resize_target_layout)
        top_layout.addWidget(self.edit_resize_group)

        self.edit_audio_group = QGroupBox("Ses Ayarlari")
        audio_layout = QVBoxLayout(self.edit_audio_group)
        audio_layout.setContentsMargins(8, 8, 8, 8)
        audio_layout.setSpacing(6)
        self.edit_remove_audio_checkbox = QCheckBox("Sesi Sil")
        self.edit_remove_audio_checkbox.setChecked(False)
        self.edit_remove_audio_checkbox.stateChanged.connect(self._on_edit_operation_checkbox_changed)
        audio_layout.addWidget(self.edit_remove_audio_checkbox)
        top_layout.addWidget(self.edit_audio_group)

        self.edit_text_overlay_group = QGroupBox("Yazi Katmanlari")
        text_overlay_layout = QVBoxLayout(self.edit_text_overlay_group)
        text_overlay_layout.setContentsMargins(8, 8, 8, 8)
        text_overlay_layout.setSpacing(6)
        self.edit_text_overlay_enabled_checkbox = QCheckBox("Enable Yazi Katmanlari")
        self.edit_text_overlay_enabled_checkbox.setChecked(False)
        self.edit_text_overlay_enabled_checkbox.stateChanged.connect(self._on_edit_operation_checkbox_changed)
        text_overlay_layout.addWidget(self.edit_text_overlay_enabled_checkbox)
        self.edit_text_overlay_table = QTableWidget(0, 9)
        self._configure_edit_overlay_table(
            self.edit_text_overlay_table,
            ["ID", "Metin", "Baslangic(sn)", "Bitis(sn)", "Pozisyon(X,Y)", "Boyut(px)", "Renk(#RRGGBB)", "Bold", "Italik"],
        )
        self.edit_text_time_delegate = FloatSpinDelegate(0.0, 86400.0, 3, 0.1, self.edit_text_overlay_table)
        self.edit_text_size_delegate = IntSpinDelegate(8, 256, 1, self.edit_text_overlay_table)
        self.edit_text_overlay_table.setItemDelegateForColumn(EDIT_TEXT_COL_START, self.edit_text_time_delegate)
        self.edit_text_overlay_table.setItemDelegateForColumn(EDIT_TEXT_COL_END, self.edit_text_time_delegate)
        self.edit_text_overlay_table.setItemDelegateForColumn(EDIT_TEXT_COL_SIZE, self.edit_text_size_delegate)
        text_table_header = self.edit_text_overlay_table.horizontalHeader()
        text_table_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        text_table_header.setStretchLastSection(False)
        self.edit_text_overlay_table.setColumnWidth(EDIT_TEXT_COL_ID, 70)
        self.edit_text_overlay_table.setColumnWidth(EDIT_TEXT_COL_TEXT, 200)
        self.edit_text_overlay_table.setColumnWidth(EDIT_TEXT_COL_START, 120)
        self.edit_text_overlay_table.setColumnWidth(EDIT_TEXT_COL_END, 120)
        self.edit_text_overlay_table.setColumnWidth(EDIT_TEXT_COL_POSITION, 150)
        self.edit_text_overlay_table.setColumnWidth(EDIT_TEXT_COL_SIZE, 90)
        self.edit_text_overlay_table.setColumnWidth(EDIT_TEXT_COL_COLOR, 130)
        self.edit_text_overlay_table.setColumnWidth(EDIT_TEXT_COL_BOLD, 80)
        self.edit_text_overlay_table.setColumnWidth(EDIT_TEXT_COL_ITALIC, 80)
        self.edit_text_overlay_table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.edit_text_overlay_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.edit_text_overlay_table.setWordWrap(False)
        self.edit_text_overlay_table.itemChanged.connect(self._on_edit_overlay_table_changed)
        self.edit_text_overlay_table.cellClicked.connect(self.on_edit_text_overlay_cell_clicked)
        self.edit_text_overlay_table.installEventFilter(self)
        text_overlay_layout.addWidget(self.edit_text_overlay_table)
        text_button_layout = QHBoxLayout()
        text_button_layout.setContentsMargins(0, 0, 0, 0)
        self.edit_text_overlay_add_button = QPushButton("Satir Ekle")
        self.edit_text_overlay_add_button.clicked.connect(self.on_add_text_overlay_row_clicked)
        self.edit_text_overlay_remove_button = QPushButton("Satir Sil")
        self.edit_text_overlay_remove_button.clicked.connect(self.on_remove_text_overlay_row_clicked)
        self.edit_text_overlay_set_event_time_button = QPushButton("Set Event Time")
        self.edit_text_overlay_set_event_time_button.clicked.connect(self.on_set_text_overlay_times_from_events_clicked)
        self.edit_text_overlay_save_button = QPushButton("Yazilari Kaydet")
        self.edit_text_overlay_save_button.clicked.connect(self.on_save_text_overlay_rows_clicked)
        self.edit_text_overlay_import_button = QPushButton("Yazilari Import Et")
        self.edit_text_overlay_import_button.clicked.connect(self.on_import_text_overlay_rows_clicked)
        text_button_layout.addWidget(self.edit_text_overlay_add_button)
        text_button_layout.addWidget(self.edit_text_overlay_remove_button)
        text_button_layout.addWidget(self.edit_text_overlay_set_event_time_button)
        text_button_layout.addWidget(self.edit_text_overlay_save_button)
        text_button_layout.addWidget(self.edit_text_overlay_import_button)
        text_button_layout.addStretch(1)
        text_overlay_layout.addLayout(text_button_layout)
        top_layout.addWidget(self.edit_text_overlay_group)

        self.edit_image_overlay_group = QGroupBox("PNG Katmanlari")
        image_overlay_layout = QVBoxLayout(self.edit_image_overlay_group)
        image_overlay_layout.setContentsMargins(8, 8, 8, 8)
        image_overlay_layout.setSpacing(6)
        self.edit_image_overlay_enabled_checkbox = QCheckBox("Enable PNG Katmanlari")
        self.edit_image_overlay_enabled_checkbox.setChecked(False)
        self.edit_image_overlay_enabled_checkbox.stateChanged.connect(self._on_edit_operation_checkbox_changed)
        image_overlay_layout.addWidget(self.edit_image_overlay_enabled_checkbox)
        self.edit_image_overlay_table = QTableWidget(0, 5)
        self._configure_edit_overlay_table(
            self.edit_image_overlay_table,
            ["Dosya", "Baslangic(sn)", "Bitis(sn)", "Pozisyon(X,Y)", "Boyut(px)"],
        )
        self.edit_image_time_delegate = FloatSpinDelegate(0.0, 86400.0, 3, 0.1, self.edit_image_overlay_table)
        self.edit_image_size_delegate = IntSpinDelegate(1, 20000, 1, self.edit_image_overlay_table)
        self.edit_image_overlay_table.setItemDelegateForColumn(EDIT_IMAGE_COL_START, self.edit_image_time_delegate)
        self.edit_image_overlay_table.setItemDelegateForColumn(EDIT_IMAGE_COL_END, self.edit_image_time_delegate)
        self.edit_image_overlay_table.setItemDelegateForColumn(EDIT_IMAGE_COL_SIZE, self.edit_image_size_delegate)
        image_table_header = self.edit_image_overlay_table.horizontalHeader()
        image_table_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        image_table_header.setStretchLastSection(False)
        self.edit_image_overlay_table.setColumnWidth(EDIT_IMAGE_COL_FILE, 260)
        self.edit_image_overlay_table.setColumnWidth(EDIT_IMAGE_COL_START, 120)
        self.edit_image_overlay_table.setColumnWidth(EDIT_IMAGE_COL_END, 120)
        self.edit_image_overlay_table.setColumnWidth(EDIT_IMAGE_COL_POSITION, 150)
        self.edit_image_overlay_table.setColumnWidth(EDIT_IMAGE_COL_SIZE, 110)
        self.edit_image_overlay_table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.edit_image_overlay_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.edit_image_overlay_table.setWordWrap(False)
        self.edit_image_overlay_table.itemChanged.connect(self._on_edit_overlay_table_changed)
        self.edit_image_overlay_table.cellClicked.connect(self.on_edit_image_overlay_cell_clicked)
        self.edit_image_overlay_table.installEventFilter(self)
        image_overlay_layout.addWidget(self.edit_image_overlay_table)
        image_button_layout = QHBoxLayout()
        image_button_layout.setContentsMargins(0, 0, 0, 0)
        self.edit_image_overlay_add_button = QPushButton("Satir Ekle")
        self.edit_image_overlay_add_button.clicked.connect(self.on_add_image_overlay_row_clicked)
        self.edit_image_overlay_remove_button = QPushButton("Satir Sil")
        self.edit_image_overlay_remove_button.clicked.connect(self.on_remove_image_overlay_row_clicked)
        image_button_layout.addWidget(self.edit_image_overlay_add_button)
        image_button_layout.addWidget(self.edit_image_overlay_remove_button)
        image_button_layout.addStretch(1)
        image_overlay_layout.addLayout(image_button_layout)
        top_layout.addWidget(self.edit_image_overlay_group)

        self.edit_external_audio_group = QGroupBox("Harici Ses Katmanlari")
        external_audio_layout = QVBoxLayout(self.edit_external_audio_group)
        external_audio_layout.setContentsMargins(8, 8, 8, 8)
        external_audio_layout.setSpacing(6)
        self.edit_external_audio_enabled_checkbox = QCheckBox("Enable Harici Ses")
        self.edit_external_audio_enabled_checkbox.setChecked(False)
        self.edit_external_audio_enabled_checkbox.stateChanged.connect(self._on_edit_operation_checkbox_changed)
        external_audio_layout.addWidget(self.edit_external_audio_enabled_checkbox)
        self.edit_external_audio_table = QTableWidget(0, 4)
        self._configure_edit_overlay_table(
            self.edit_external_audio_table,
            ["Dosya", "Baslangic(sn)", "Bitis(sn, ops)", "Ses Seviyesi"],
        )
        external_audio_table_header = self.edit_external_audio_table.horizontalHeader()
        external_audio_table_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        external_audio_table_header.setStretchLastSection(False)
        self.edit_external_audio_table.setColumnWidth(EDIT_EXTERNAL_AUDIO_COL_FILE, 260)
        self.edit_external_audio_table.setColumnWidth(EDIT_EXTERNAL_AUDIO_COL_START, 120)
        self.edit_external_audio_table.setColumnWidth(EDIT_EXTERNAL_AUDIO_COL_END, 140)
        self.edit_external_audio_table.setColumnWidth(EDIT_EXTERNAL_AUDIO_COL_VOLUME, 120)
        self.edit_external_audio_table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.edit_external_audio_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.edit_external_audio_table.setWordWrap(False)
        self.edit_external_audio_table.itemChanged.connect(self._on_edit_overlay_table_changed)
        self.edit_external_audio_table.cellClicked.connect(self.on_edit_external_audio_cell_clicked)
        external_audio_layout.addWidget(self.edit_external_audio_table)
        external_audio_button_layout = QHBoxLayout()
        external_audio_button_layout.setContentsMargins(0, 0, 0, 0)
        self.edit_external_audio_add_button = QPushButton("Satir Ekle")
        self.edit_external_audio_add_button.clicked.connect(self.on_add_external_audio_row_clicked)
        self.edit_external_audio_remove_button = QPushButton("Satir Sil")
        self.edit_external_audio_remove_button.clicked.connect(self.on_remove_external_audio_row_clicked)
        self.edit_external_audio_save_button = QPushButton("Tablo Kaydet")
        self.edit_external_audio_save_button.clicked.connect(self.on_save_external_audio_rows_clicked)
        self.edit_external_audio_import_button = QPushButton("Tablo Import Et")
        self.edit_external_audio_import_button.clicked.connect(self.on_import_external_audio_rows_clicked)
        external_audio_button_layout.addWidget(self.edit_external_audio_add_button)
        external_audio_button_layout.addWidget(self.edit_external_audio_remove_button)
        external_audio_button_layout.addWidget(self.edit_external_audio_save_button)
        external_audio_button_layout.addWidget(self.edit_external_audio_import_button)
        external_audio_button_layout.addStretch(1)
        external_audio_layout.addLayout(external_audio_button_layout)
        top_layout.addWidget(self.edit_external_audio_group)

        self.edit_speed_group = QGroupBox("Video Hizi")
        speed_layout = QVBoxLayout(self.edit_speed_group)
        speed_layout.setContentsMargins(8, 8, 8, 8)
        speed_layout.setSpacing(6)
        self.edit_speed_enabled_checkbox = QCheckBox("Enable Video Hizi")
        self.edit_speed_enabled_checkbox.setChecked(False)
        self.edit_speed_enabled_checkbox.stateChanged.connect(self._on_edit_operation_checkbox_changed)

        speed_target_layout = QHBoxLayout()
        speed_target_layout.setContentsMargins(0, 0, 0, 0)
        speed_target_layout.addWidget(QLabel("Hiz:"))
        self.edit_speed_combo = QComboBox()
        for label, factor in EDIT_SPEED_PRESETS:
            self.edit_speed_combo.addItem(label, float(factor))
        self.edit_speed_combo.setCurrentText("1.0x")
        self.edit_speed_combo.currentIndexChanged.connect(self._update_edit_controls)
        speed_target_layout.addWidget(self.edit_speed_combo)
        speed_target_layout.addStretch(1)

        speed_layout.addWidget(self.edit_speed_enabled_checkbox)
        speed_layout.addLayout(speed_target_layout)
        top_layout.addWidget(self.edit_speed_group)

        edit_checkbox_style = """
        QCheckBox {
            spacing: 8px;
            min-height: 26px;
            color: #f3f6fb;
            font-size: 14px;
            font-weight: 600;
        }
        QCheckBox::indicator {
            width: 20px;
            height: 20px;
            background-color: #5f6672;
            border: 1px solid #7a8392;
            border-radius: 4px;
        }
        QCheckBox::indicator:checked {
            background-color: #5f6672;
            border: 1px solid #7a8392;
            image: url(checkbox_tick_black.svg);
        }
        QCheckBox:disabled {
            color: #8e97a8;
        }
        """
        for checkbox in (
            self.edit_cut_enabled_checkbox,
            self.edit_resize_enabled_checkbox,
            self.edit_remove_audio_checkbox,
            self.edit_text_overlay_enabled_checkbox,
            self.edit_image_overlay_enabled_checkbox,
            self.edit_external_audio_enabled_checkbox,
            self.edit_speed_enabled_checkbox,
        ):
            checkbox.setStyleSheet(edit_checkbox_style)
        top_layout.addStretch(1)

        bottom_content = QWidget(left_panel)
        bottom_layout = QVBoxLayout(bottom_content)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)

        run_layout = QHBoxLayout()
        run_layout.setContentsMargins(0, 0, 0, 0)
        run_layout.addStretch(1)
        self.edit_run_button = QPushButton("Edit Islemine Basla")
        self.edit_run_button.clicked.connect(self.on_edit_run_button_clicked)
        run_layout.addWidget(self.edit_run_button)
        bottom_layout.addLayout(run_layout)

        self.edit_progress = TimeOverlayProgressBar()
        self.edit_progress.setRange(0, 100)
        self.edit_progress.setValue(0)
        self.edit_progress.set_time_fields(elapsed_seconds=None, remaining_seconds=None)
        self.edit_progress.setVisible(False)
        bottom_layout.addWidget(self.edit_progress)

        self.edit_log = QPlainTextEdit()
        self.edit_log.setReadOnly(True)
        self.edit_log.setPlaceholderText("FFmpeg edit loglari burada gorunecek.")
        self.edit_log.setMaximumBlockCount(400)
        bottom_layout.addWidget(self.edit_log, stretch=1)

        left_layout.addWidget(top_scroll, stretch=3)
        left_layout.addWidget(bottom_content, stretch=1)

        right_panel = QWidget(edit_splitter)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self.edit_overlay_preview_group = QGroupBox("Onizleme (Ilk Kare)")
        preview_group_layout = QVBoxLayout(self.edit_overlay_preview_group)
        preview_group_layout.setContentsMargins(8, 8, 8, 8)
        preview_group_layout.setSpacing(6)

        self.edit_overlay_preview_label = QLabel("Video secildiginde ilk kare burada gorunecek.")
        self.edit_overlay_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.edit_overlay_preview_label.setMinimumSize(320, 240)
        self.edit_overlay_preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.edit_overlay_preview_label.setStyleSheet("border: 1px solid #666; background: #111; color: #aaa;")
        preview_group_layout.addWidget(self.edit_overlay_preview_label, stretch=1)

        self.edit_overlay_preview_status_label = QLabel("Yazi ve PNG katmanlari ilk kare uzerinde gosterilir.")
        self.edit_overlay_preview_status_label.setStyleSheet("color: #9aa4b7;")
        preview_group_layout.addWidget(self.edit_overlay_preview_status_label)

        right_layout.addWidget(self.edit_overlay_preview_group, stretch=1)

        edit_splitter.addWidget(left_panel)
        edit_splitter.addWidget(right_panel)
        edit_splitter.setChildrenCollapsible(False)
        edit_splitter.setStretchFactor(0, 3)
        edit_splitter.setStretchFactor(1, 2)
        edit_splitter.setSizes([900, 540])
        root_layout.addWidget(edit_splitter, stretch=1)
        self._refresh_edit_resolution_options()
        self._update_edit_quality_tooltips()
        self._update_edit_controls()
        self._update_edit_overlay_preview(force_frame_reload=True)

    def _configure_edit_overlay_table(self, table: QTableWidget, headers: list[str]) -> None:
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setMinimumHeight(150)

    def _on_edit_overlay_table_changed(self, item: Optional[QTableWidgetItem] = None) -> None:
        if (
            item is not None
            and hasattr(self, "edit_external_audio_table")
            and item.tableWidget() is self.edit_external_audio_table
            and item.column() == EDIT_EXTERNAL_AUDIO_COL_START
        ):
            self._clear_external_audio_start_item_metadata(item)
        self._update_edit_controls()
        self._update_edit_overlay_preview(force_frame_reload=False)

    @staticmethod
    def _set_table_row_values(table: QTableWidget, row_values: list[str]) -> None:
        row_index = table.rowCount()
        table.insertRow(row_index)
        for col_index, raw_value in enumerate(row_values):
            table.setItem(row_index, col_index, QTableWidgetItem(str(raw_value)))

    @staticmethod
    def _remove_selected_table_row(table: QTableWidget) -> None:
        current_row = table.currentRow()
        if current_row < 0 and table.rowCount() > 0:
            current_row = table.rowCount() - 1
        if current_row >= 0:
            table.removeRow(current_row)

    def _set_text_overlay_style_cell(self, row: int, col: int, checked: bool) -> None:
        if not hasattr(self, "edit_text_overlay_table"):
            return
        table = self.edit_text_overlay_table
        if row < 0 or row >= table.rowCount():
            return
        item = table.item(row, col)
        if item is None:
            item = QTableWidgetItem("")
            table.setItem(row, col, item)
        item_flags = item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        item_flags &= ~Qt.ItemFlag.ItemIsEditable
        item.setFlags(item_flags)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    def _text_overlay_style_value(self, row: int, col: int) -> bool:
        if not hasattr(self, "edit_text_overlay_table"):
            return False
        table = self.edit_text_overlay_table
        if row < 0 or row >= table.rowCount():
            return False
        item = table.item(row, col)
        if item is None:
            return False
        return item.checkState() == Qt.CheckState.Checked

    @staticmethod
    def _parse_overlay_bool(raw_value: object, default: bool = False) -> bool:
        if raw_value is None:
            return bool(default)
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, (int, float)):
            return float(raw_value) != 0.0
        text = str(raw_value).strip().lower()
        if text in ("1", "true", "yes", "evet", "on"):
            return True
        if text in ("0", "false", "no", "hayir", "off", ""):
            return False
        return bool(default)

    @staticmethod
    def _normalize_text_overlay_id(raw_value: object) -> str:
        return str(raw_value).strip()

    def _next_text_overlay_id(self, used_ids: Optional[set[str]] = None, include_table: bool = True) -> str:
        existing_ids = set(used_ids or set())
        if include_table and hasattr(self, "edit_text_overlay_table"):
            table = self.edit_text_overlay_table
            for row in range(table.rowCount()):
                existing_id = self._normalize_text_overlay_id(self._edit_table_cell_text(table, row, EDIT_TEXT_COL_ID))
                if existing_id:
                    existing_ids.add(existing_id)
        candidate = 1
        while str(candidate) in existing_ids:
            candidate += 1
        return str(candidate)

    def _append_text_overlay_row(
        self,
        id_value: str,
        text_value: str,
        start_value: str,
        end_value: str,
        position_value: str,
        size_value: str,
        color_value: str,
        bold: bool = False,
        italic: bool = False,
    ) -> None:
        if not hasattr(self, "edit_text_overlay_table"):
            return
        table = self.edit_text_overlay_table
        self._set_table_row_values(
            table,
            [id_value, text_value, start_value, end_value, position_value, size_value, color_value, "", ""],
        )
        row = table.rowCount() - 1
        self._set_text_overlay_style_cell(row, EDIT_TEXT_COL_BOLD, checked=bold)
        self._set_text_overlay_style_cell(row, EDIT_TEXT_COL_ITALIC, checked=italic)

    def on_add_text_overlay_row_clicked(self) -> None:
        self.edit_text_overlay_table.blockSignals(True)
        try:
            self._append_text_overlay_row(
                id_value=str(self._next_text_overlay_id()),
                text_value="ornek yazi",
                start_value="0.0",
                end_value="1.0",
                position_value="0.05, 0.05",
                size_value="36",
                color_value="#FFFFFF",
                bold=False,
                italic=False,
            )
        finally:
            self.edit_text_overlay_table.blockSignals(False)
        self._update_edit_controls()
        self._update_edit_overlay_preview(force_frame_reload=False)

    def on_remove_text_overlay_row_clicked(self) -> None:
        self._remove_selected_table_row(self.edit_text_overlay_table)
        self._update_edit_controls()
        self._update_edit_overlay_preview(force_frame_reload=False)

    def on_set_text_overlay_times_from_events_clicked(self) -> None:
        if not hasattr(self, "edit_text_overlay_table"):
            return
        if not self.last_detected_events:
            QMessageBox.information(self, "Yazi Katmanlari", "Once olay zamanlarini yukleyin veya olusturun.")
            return

        video_duration = self._video_duration_seconds()
        if video_duration is None or video_duration <= 0.0:
            QMessageBox.warning(self, "Yazi Katmanlari", "Video suresi belirlenemedi.")
            return

        event_start_times = self._available_event_start_times(video_duration)
        if not event_start_times:
            QMessageBox.information(self, "Yazi Katmanlari", "Kullanilabilir event start zamani bulunamadi.")
            return

        table = self.edit_text_overlay_table
        updated_count = 0
        skipped_ids: list[str] = []
        table.blockSignals(True)
        try:
            for row in range(table.rowCount()):
                overlay_id = self._normalize_text_overlay_id(self._edit_table_cell_text(table, row, EDIT_TEXT_COL_ID))
                if not overlay_id:
                    continue
                event_start = event_start_times.get(overlay_id)
                if event_start is None:
                    skipped_ids.append(overlay_id)
                    continue

                start_item = table.item(row, EDIT_TEXT_COL_START)
                if start_item is None:
                    start_item = QTableWidgetItem("")
                    table.setItem(row, EDIT_TEXT_COL_START, start_item)
                end_item = table.item(row, EDIT_TEXT_COL_END)
                if end_item is None:
                    end_item = QTableWidgetItem("")
                    table.setItem(row, EDIT_TEXT_COL_END, end_item)

                start_item.setText(self._format_overlay_number(event_start, 3))
                end_item.setText(self._format_overlay_number(video_duration, 3))
                updated_count += 1
        finally:
            table.blockSignals(False)

        self._update_edit_controls()
        self._update_edit_overlay_preview(force_frame_reload=False)

        if updated_count <= 0:
            QMessageBox.information(self, "Yazi Katmanlari", "Eslesen event adi bulunamadi.")
            return

        skipped_text = ""
        if skipped_ids:
            unique_skipped = ", ".join(dict.fromkeys(skipped_ids))
            skipped_text = f" | Eslesmeyen ID: {unique_skipped}"
        message = f"{updated_count} yazi satiri event zamanina gore guncellendi{skipped_text}"
        self.statusBar().showMessage(message, 4500)
        self._append_edit_log(message)

    @staticmethod
    def _format_overlay_number(value: float, decimals: int = 6) -> str:
        text = f"{float(value):.{max(0, int(decimals))}f}".rstrip("0").rstrip(".")
        return text or "0"

    def _available_event_start_times(self, max_seconds: Optional[float] = None) -> dict[str, float]:
        event_start_times: dict[str, float] = {}
        for event_payload in self.last_detected_events:
            if not isinstance(event_payload, dict):
                continue
            event_name = str(event_payload.get("name", "")).strip()
            if not event_name or event_name in event_start_times:
                continue
            raw_start = event_payload.get("start")
            if raw_start is None:
                continue
            try:
                start_seconds = float(raw_start)
            except (TypeError, ValueError):
                continue
            if start_seconds < 0.0:
                continue
            if max_seconds is not None and start_seconds >= float(max_seconds):
                continue
            event_start_times[event_name] = start_seconds
        return event_start_times

    def on_save_text_overlay_rows_clicked(self) -> None:
        if not hasattr(self, "edit_text_overlay_table"):
            return
        overlays, overlay_error = self._collect_text_overlays()
        if overlay_error is not None:
            QMessageBox.warning(self, "Yazi Katmanlari", overlay_error)
            return
        if not overlays:
            QMessageBox.information(self, "Yazi Katmanlari", "Kaydedilecek gecerli yazi satiri yok.")
            return

        source_video = self.video_meta.source_video if self.video_meta is not None else ""
        initial_dir = self._default_project_directory()

        video_name = os.path.splitext(os.path.basename(source_video))[0].strip() if source_video else "video"
        if not video_name:
            video_name = "video"
        initial_path = os.path.join(initial_dir, f"{video_name}_text_overlays.json")
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Yazi Katmanlarini Kaydet",
            initial_path,
            "JSON Files (*.json);;All Files (*.*)",
        )
        if not save_path:
            return
        if not save_path.lower().endswith(".json"):
            save_path += ".json"

        serializable_items: list[dict] = []
        for item in overlays:
            payload_item = dict(item)
            payload_item["position"] = self._format_text_overlay_position(
                float(item.get("x", 0.0)),
                float(item.get("y", 0.0)),
            )
            serializable_items.append(payload_item)

        payload = {
            "type": "text_overlays",
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_video": source_video,
            "items": serializable_items,
        }
        try:
            with open(save_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)
        except OSError as exc:
            QMessageBox.critical(self, "Yazi Katmanlari", f"Kayit basarisiz:\n{exc}")
            return

        self.statusBar().showMessage(f"Yazi katmanlari kaydedildi: {save_path}", 3500)
        self._append_edit_log(f"Yazi katmanlari kaydedildi: {save_path}")

    def on_import_text_overlay_rows_clicked(self) -> None:
        if not hasattr(self, "edit_text_overlay_table"):
            return
        initial_dir = self._default_project_directory()

        load_path, _ = QFileDialog.getOpenFileName(
            self,
            "Yazi Katmanlarini Import Et",
            initial_dir,
            "JSON Files (*.json);;All Files (*.*)",
        )
        if not load_path:
            return

        try:
            with open(load_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Yazi Katmanlari", f"Dosya okunamadi:\n{exc}")
            return

        raw_items: object = payload
        if isinstance(payload, dict):
            if isinstance(payload.get("items"), list):
                raw_items = payload.get("items")
            elif isinstance(payload.get("text_overlays"), list):
                raw_items = payload.get("text_overlays")
            elif isinstance(payload.get("overlays"), list):
                raw_items = payload.get("overlays")

        if not isinstance(raw_items, list):
            QMessageBox.warning(self, "Yazi Katmanlari", "Dosya formati gecersiz: yazi listesi bulunamadi.")
            return

        imported_rows: list[tuple[list[str], bool, bool]] = []
        used_ids: set[str] = set()
        for index, raw_item in enumerate(raw_items, start=1):
            if not isinstance(raw_item, dict):
                QMessageBox.warning(self, "Yazi Katmanlari", f"Satir {index}: Veri dict formatinda olmali.")
                return
            raw_id = raw_item.get("id")
            overlay_id = self._normalize_text_overlay_id(raw_id) if raw_id is not None else ""
            if not overlay_id:
                overlay_id = self._next_text_overlay_id(used_ids, include_table=False)
            if overlay_id in used_ids:
                QMessageBox.warning(self, "Yazi Katmanlari", f"Satir {index}: ID tekrar ediyor ({overlay_id}).")
                return
            used_ids.add(overlay_id)
            text_value = str(raw_item.get("text", "")).strip()
            if not text_value:
                QMessageBox.warning(self, "Yazi Katmanlari", f"Satir {index}: Metin bos olamaz.")
                return
            try:
                start_seconds = float(raw_item.get("start", 0.0))
                end_seconds = float(raw_item.get("end", 0.0))
            except (TypeError, ValueError):
                QMessageBox.warning(self, "Yazi Katmanlari", f"Satir {index}: Baslangic/bitis degeri gecersiz.")
                return
            if start_seconds < 0.0:
                QMessageBox.warning(self, "Yazi Katmanlari", f"Satir {index}: Baslangic 0'dan kucuk olamaz.")
                return
            if end_seconds <= start_seconds:
                QMessageBox.warning(self, "Yazi Katmanlari", f"Satir {index}: Bitis baslangictan buyuk olmali.")
                return

            x_value: Optional[float] = None
            y_value: Optional[float] = None
            if "x" in raw_item and "y" in raw_item:
                try:
                    x_value = float(raw_item.get("x"))
                    y_value = float(raw_item.get("y"))
                except (TypeError, ValueError):
                    x_value, y_value = None, None
            if x_value is None or y_value is None:
                parsed_position = self._parse_text_overlay_position(str(raw_item.get("position", "")))
                if parsed_position is None:
                    QMessageBox.warning(self, "Yazi Katmanlari", f"Satir {index}: Pozisyon X,Y formatinda olmali.")
                    return
                x_value, y_value = parsed_position
            if x_value < 0.0 or x_value > 1.0 or y_value < 0.0 or y_value > 1.0:
                QMessageBox.warning(self, "Yazi Katmanlari", f"Satir {index}: Pozisyon 0-1 araliginda olmali.")
                return

            try:
                font_size = int(raw_item.get("font_size", 36))
            except (TypeError, ValueError):
                QMessageBox.warning(self, "Yazi Katmanlari", f"Satir {index}: Boyut gecersiz.")
                return
            if font_size < 8 or font_size > 256:
                QMessageBox.warning(self, "Yazi Katmanlari", f"Satir {index}: Boyut 8-256 araliginda olmali.")
                return

            color_value = str(raw_item.get("color", "#FFFFFF")).strip().upper() or "#FFFFFF"
            if not self._is_valid_hex_color(color_value):
                QMessageBox.warning(self, "Yazi Katmanlari", f"Satir {index}: Renk #RRGGBB formatinda olmali.")
                return

            bold_value = self._parse_overlay_bool(raw_item.get("bold", False), default=False)
            italic_value = self._parse_overlay_bool(raw_item.get("italic", False), default=False)

            imported_rows.append(
                (
                    [
                        overlay_id,
                        text_value,
                        self._format_overlay_number(start_seconds, 3),
                        self._format_overlay_number(end_seconds, 3),
                        self._format_text_overlay_position(x_value, y_value),
                        str(font_size),
                        color_value,
                    ],
                    bold_value,
                    italic_value,
                )
            )

        table = self.edit_text_overlay_table
        table.blockSignals(True)
        try:
            table.setRowCount(0)
            for row_values, bold_value, italic_value in imported_rows:
                self._append_text_overlay_row(
                    id_value=row_values[0],
                    text_value=row_values[1],
                    start_value=row_values[2],
                    end_value=row_values[3],
                    position_value=row_values[4],
                    size_value=row_values[5],
                    color_value=row_values[6],
                    bold=bold_value,
                    italic=italic_value,
                )
        finally:
            table.blockSignals(False)

        if imported_rows and hasattr(self, "edit_text_overlay_enabled_checkbox"):
            self.edit_text_overlay_enabled_checkbox.setChecked(True)
        self._update_edit_controls()
        self._update_edit_overlay_preview(force_frame_reload=False)
        self.statusBar().showMessage(f"Yazi katmanlari import edildi: {os.path.basename(load_path)}", 3500)
        self._append_edit_log(f"Yazi katmanlari import edildi: {load_path}")

    def on_edit_text_overlay_cell_clicked(self, row: int, col: int) -> None:
        if not hasattr(self, "edit_text_overlay_table"):
            return
        table = self.edit_text_overlay_table
        if col in (EDIT_TEXT_COL_BOLD, EDIT_TEXT_COL_ITALIC):
            self._update_edit_controls()
            self._update_edit_overlay_preview(force_frame_reload=False)
            return
        if col != EDIT_TEXT_COL_COLOR:
            if col in (EDIT_TEXT_COL_START, EDIT_TEXT_COL_END, EDIT_TEXT_COL_SIZE) and table.isEnabled():
                if table.state() == QAbstractItemView.State.EditingState:
                    return
                item = table.item(row, col)
                if item is None:
                    item = QTableWidgetItem("")
                    table.setItem(row, col, item)
                table.editItem(item)
            return
        self._pick_text_overlay_color_for_row(row)

    def _pick_text_overlay_color_for_row(self, row: int) -> None:
        if not hasattr(self, "edit_text_overlay_table"):
            return
        table = self.edit_text_overlay_table
        if row < 0 or row >= table.rowCount():
            return

        current_value = self._edit_table_cell_text(table, row, EDIT_TEXT_COL_COLOR).upper()
        initial_color = QColor(current_value) if self._is_valid_hex_color(current_value) else QColor("#FFFFFF")
        selected_color = QColorDialog.getColor(initial_color, self, "Yazi Rengi Sec")
        if not selected_color.isValid():
            return

        hex_value = selected_color.name(QColor.NameFormat.HexRgb).upper()
        table.blockSignals(True)
        try:
            item = table.item(row, EDIT_TEXT_COL_COLOR)
            if item is None:
                item = QTableWidgetItem(hex_value)
                table.setItem(row, EDIT_TEXT_COL_COLOR, item)
            else:
                item.setText(hex_value)
        finally:
            table.blockSignals(False)
        self._update_edit_controls()
        self._update_edit_overlay_preview(force_frame_reload=False)

    @staticmethod
    def _parse_text_overlay_position(raw_value: str) -> Optional[Tuple[float, float]]:
        text = str(raw_value).strip()
        if not text:
            return None
        normalized = text.replace(";", ",").replace("(", "").replace(")", "")
        if "," in normalized:
            parts = [part.strip() for part in normalized.split(",") if part.strip()]
        else:
            parts = [part for part in normalized.split() if part]
        if len(parts) != 2:
            return None
        try:
            return float(parts[0]), float(parts[1])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_text_overlay_position(x_value: float, y_value: float) -> str:
        safe_x = max(0.0, min(1.0, float(x_value)))
        safe_y = max(0.0, min(1.0, float(y_value)))
        x_text = f"{safe_x:.4f}".rstrip("0").rstrip(".")
        y_text = f"{safe_y:.4f}".rstrip("0").rstrip(".")
        if not x_text:
            x_text = "0"
        if not y_text:
            y_text = "0"
        return f"{x_text}, {y_text}"

    @staticmethod
    def _image_overlay_source_size(path_value: str) -> Optional[Tuple[int, int]]:
        image_path = str(path_value).strip()
        if not image_path or not os.path.isfile(image_path):
            return None
        image = QImage(image_path)
        if image.isNull():
            return None
        source_width = int(image.width())
        source_height = int(image.height())
        if source_width <= 0 or source_height <= 0:
            return None
        return source_width, source_height

    def _image_overlay_scaled_dimensions(self, path_value: str, size_value: int) -> Optional[Tuple[int, int]]:
        source_size = self._image_overlay_source_size(path_value)
        if source_size is None:
            return None
        source_width, source_height = source_size
        target_width = max(1, int(size_value))
        target_height = max(1, int(round(float(target_width) * float(source_height) / float(source_width))))
        return target_width, target_height

    def eventFilter(self, watched, event) -> bool:
        if isinstance(event, QKeyEvent) and event.type() == QEvent.Type.KeyPress:
            if hasattr(self, "edit_text_overlay_table") and watched is self.edit_text_overlay_table:
                table = self.edit_text_overlay_table
                if (
                    table.isEnabled()
                    and table.currentColumn() == EDIT_TEXT_COL_POSITION
                    and table.state() != QAbstractItemView.State.EditingState
                ):
                    key_value = event.key()
                    if key_value in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
                        row = table.currentRow()
                        if row < 0:
                            return True
                        item = table.item(row, EDIT_TEXT_COL_POSITION)
                        if item is None:
                            item = QTableWidgetItem("0, 0")
                            table.setItem(row, EDIT_TEXT_COL_POSITION, item)
                        parsed = self._parse_text_overlay_position(item.text())
                        current_x, current_y = (0.0, 0.0) if parsed is None else parsed

                        step = 0.01
                        modifiers = event.modifiers()
                        if modifiers & Qt.KeyboardModifier.ShiftModifier:
                            step = 0.05
                        elif modifiers & Qt.KeyboardModifier.ControlModifier:
                            step = 0.002

                        if key_value == Qt.Key.Key_Left:
                            current_x -= step
                        elif key_value == Qt.Key.Key_Right:
                            current_x += step
                        elif key_value == Qt.Key.Key_Up:
                            current_y -= step
                        elif key_value == Qt.Key.Key_Down:
                            current_y += step

                        position_text = self._format_text_overlay_position(current_x, current_y)
                        table.blockSignals(True)
                        try:
                            item.setText(position_text)
                        finally:
                            table.blockSignals(False)
                        self._update_edit_controls()
                        self._update_edit_overlay_preview(force_frame_reload=False)
                        self.statusBar().showMessage(
                            f"Yazi konumu guncellendi: {position_text} (Shift: hizli, Ctrl: hassas)",
                            1200,
                        )
                        return True

            if hasattr(self, "edit_image_overlay_table") and watched is self.edit_image_overlay_table:
                table = self.edit_image_overlay_table
                if table.isEnabled() and table.state() != QAbstractItemView.State.EditingState:
                    row = table.currentRow()
                    col = table.currentColumn()
                    key_value = event.key()
                    modifiers = event.modifiers()

                    if row < 0:
                        return key_value in (
                            Qt.Key.Key_Left,
                            Qt.Key.Key_Right,
                            Qt.Key.Key_Up,
                            Qt.Key.Key_Down,
                        )

                    if (
                        col == EDIT_IMAGE_COL_POSITION
                        and key_value in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down)
                    ):
                        item = table.item(row, EDIT_IMAGE_COL_POSITION)
                        if item is None:
                            item = QTableWidgetItem("0, 0")
                            table.setItem(row, EDIT_IMAGE_COL_POSITION, item)
                        parsed = self._parse_text_overlay_position(item.text())
                        current_x, current_y = (0.0, 0.0) if parsed is None else parsed

                        step = 0.01
                        if modifiers & Qt.KeyboardModifier.ShiftModifier:
                            step = 0.05
                        elif modifiers & Qt.KeyboardModifier.ControlModifier:
                            step = 0.002

                        if key_value == Qt.Key.Key_Left:
                            current_x -= step
                        elif key_value == Qt.Key.Key_Right:
                            current_x += step
                        elif key_value == Qt.Key.Key_Up:
                            current_y -= step
                        elif key_value == Qt.Key.Key_Down:
                            current_y += step

                        position_text = self._format_text_overlay_position(current_x, current_y)
                        table.blockSignals(True)
                        try:
                            item.setText(position_text)
                        finally:
                            table.blockSignals(False)
                        self._update_edit_controls()
                        self._update_edit_overlay_preview(force_frame_reload=False)
                        self.statusBar().showMessage(
                            f"PNG konumu guncellendi: {position_text} (Shift: hizli, Ctrl: hassas)",
                            1200,
                        )
                        return True

                    if col in (EDIT_IMAGE_COL_START, EDIT_IMAGE_COL_END) and key_value in (
                        Qt.Key.Key_Left,
                        Qt.Key.Key_Right,
                        Qt.Key.Key_Up,
                        Qt.Key.Key_Down,
                    ):
                        item = table.item(row, col)
                        if item is None:
                            default_time = "0.0" if col == EDIT_IMAGE_COL_START else "1.0"
                            item = QTableWidgetItem(default_time)
                            table.setItem(row, col, item)
                        try:
                            current_value = float(item.text().strip())
                        except (TypeError, ValueError):
                            current_value = 0.0 if col == EDIT_IMAGE_COL_START else 1.0

                        step = 0.1
                        if modifiers & Qt.KeyboardModifier.ShiftModifier:
                            step = 1.0
                        elif modifiers & Qt.KeyboardModifier.ControlModifier:
                            step = 0.01

                        if key_value in (Qt.Key.Key_Left, Qt.Key.Key_Down):
                            current_value -= step
                        else:
                            current_value += step
                        current_value = max(0.0, current_value)

                        try:
                            row_start = float(self._edit_table_cell_text(table, row, EDIT_IMAGE_COL_START))
                        except (TypeError, ValueError):
                            row_start = None
                        try:
                            row_end = float(self._edit_table_cell_text(table, row, EDIT_IMAGE_COL_END))
                        except (TypeError, ValueError):
                            row_end = None

                        if col == EDIT_IMAGE_COL_START and row_end is not None:
                            current_value = min(current_value, max(0.0, row_end - 0.001))
                        if col == EDIT_IMAGE_COL_END and row_start is not None:
                            current_value = max(current_value, row_start + 0.001)

                        time_text = self._format_overlay_number(current_value, 3)
                        table.blockSignals(True)
                        try:
                            item.setText(time_text)
                        finally:
                            table.blockSignals(False)
                        self._update_edit_controls()
                        self._update_edit_overlay_preview(force_frame_reload=False)
                        self.statusBar().showMessage(
                            f"PNG zamani guncellendi: {time_text} sn (Shift: hizli, Ctrl: hassas)",
                            1200,
                        )
                        return True

                    if col == EDIT_IMAGE_COL_SIZE and key_value in (Qt.Key.Key_Up, Qt.Key.Key_Down):
                        item = table.item(row, EDIT_IMAGE_COL_SIZE)
                        if item is None:
                            item = QTableWidgetItem("")
                            table.setItem(row, EDIT_IMAGE_COL_SIZE, item)

                        size_text_raw = item.text().strip()
                        if size_text_raw:
                            try:
                                current_size = max(1, int(float(size_text_raw)))
                            except (TypeError, ValueError):
                                current_size = 100
                        else:
                            path_value = self._edit_table_cell_text(table, row, EDIT_IMAGE_COL_FILE)
                            source_size = self._image_overlay_source_size(path_value)
                            current_size = source_size[0] if source_size is not None else 100

                        step = 10
                        if modifiers & Qt.KeyboardModifier.ShiftModifier:
                            step = 50
                        elif modifiers & Qt.KeyboardModifier.ControlModifier:
                            step = 2

                        if key_value == Qt.Key.Key_Up:
                            current_size += step
                        else:
                            current_size -= step
                        current_size = max(1, current_size)

                        table.blockSignals(True)
                        try:
                            item.setText(str(current_size))
                        finally:
                            table.blockSignals(False)

                        path_value = self._edit_table_cell_text(table, row, EDIT_IMAGE_COL_FILE)
                        scaled_size = self._image_overlay_scaled_dimensions(path_value, current_size)
                        self._update_edit_controls()
                        self._update_edit_overlay_preview(force_frame_reload=False)
                        if scaled_size is None:
                            size_message = f"{current_size}px"
                        else:
                            size_message = f"{scaled_size[0]}x{scaled_size[1]} px"
                        self.statusBar().showMessage(
                            f"PNG boyutu guncellendi: {size_message} (Shift: hizli, Ctrl: hassas)",
                            1200,
                        )
                        return True

        return super().eventFilter(watched, event)

    def on_add_image_overlay_row_clicked(self) -> None:
        self.edit_image_overlay_table.blockSignals(True)
        try:
            self._set_table_row_values(
                self.edit_image_overlay_table,
                ["", "0.0", "1.0", "0.10, 0.10", ""],
            )
        finally:
            self.edit_image_overlay_table.blockSignals(False)
        self._update_edit_controls()
        self._update_edit_overlay_preview(force_frame_reload=False)

    def on_remove_image_overlay_row_clicked(self) -> None:
        self._remove_selected_table_row(self.edit_image_overlay_table)
        self._update_edit_controls()
        self._update_edit_overlay_preview(force_frame_reload=False)

    def on_edit_image_overlay_cell_clicked(self, row: int, col: int) -> None:
        if not hasattr(self, "edit_image_overlay_table"):
            return

        table = self.edit_image_overlay_table
        if row < 0 or row >= table.rowCount() or not table.isEnabled():
            return

        if col != EDIT_IMAGE_COL_FILE:
            if col in (EDIT_IMAGE_COL_START, EDIT_IMAGE_COL_END, EDIT_IMAGE_COL_SIZE):
                if table.state() == QAbstractItemView.State.EditingState:
                    return
                item = table.item(row, col)
                if item is None:
                    default_value = ""
                    if col == EDIT_IMAGE_COL_START:
                        default_value = "0.0"
                    elif col == EDIT_IMAGE_COL_END:
                        default_value = "1.0"
                    item = QTableWidgetItem(default_value)
                    table.setItem(row, col, item)
                table.editItem(item)
            return

        current_value = self._edit_table_cell_text(table, row, EDIT_IMAGE_COL_FILE)
        initial_dir = ""
        if current_value:
            if os.path.isfile(current_value):
                initial_dir = os.path.dirname(current_value)
            elif os.path.isdir(current_value):
                initial_dir = current_value
        if not initial_dir and self.video_meta is not None and self.video_meta.source_video:
            video_dir = os.path.dirname(self.video_meta.source_video)
            if os.path.isdir(video_dir):
                initial_dir = video_dir
        if not initial_dir:
            initial_dir = self.settings.value(SETTINGS_LAST_VIDEO_DIR, "", type=str).strip()
        if not initial_dir or not os.path.isdir(initial_dir):
            initial_dir = os.getcwd()

        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "PNG Dosyasi Sec",
            initial_dir,
            "PNG Files (*.png);;Image Files (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*.*)",
        )
        if not selected_path:
            return

        table.blockSignals(True)
        try:
            item = table.item(row, EDIT_IMAGE_COL_FILE)
            if item is None:
                item = QTableWidgetItem(selected_path)
                table.setItem(row, EDIT_IMAGE_COL_FILE, item)
            else:
                item.setText(selected_path)
        finally:
            table.blockSignals(False)
        self._update_edit_controls()
        self._update_edit_overlay_preview(force_frame_reload=False)

    def on_add_external_audio_row_clicked(self) -> None:
        self.edit_external_audio_table.blockSignals(True)
        try:
            self._append_external_audio_row("", "0.0", "", "1.0")
        finally:
            self.edit_external_audio_table.blockSignals(False)
        self._update_edit_controls()

    def on_remove_external_audio_row_clicked(self) -> None:
        self._remove_selected_table_row(self.edit_external_audio_table)
        self._update_edit_controls()

    def _append_external_audio_row(
        self,
        path_value: str,
        start_value: str,
        end_value: str,
        volume_value: str,
        start_event_name: Optional[str] = None,
    ) -> None:
        self._set_table_row_values(
            self.edit_external_audio_table,
            [path_value, start_value, end_value, volume_value],
        )
        row = self.edit_external_audio_table.rowCount() - 1
        if row < 0 or not start_event_name:
            return
        start_item = self.edit_external_audio_table.item(row, EDIT_EXTERNAL_AUDIO_COL_START)
        if start_item is None:
            return
        try:
            start_seconds = float(start_value)
        except (TypeError, ValueError):
            return
        self._set_external_audio_start_item_metadata(start_item, start_event_name, start_seconds)

    @staticmethod
    def _clear_external_audio_start_item_metadata(item: QTableWidgetItem) -> None:
        item.setData(EDIT_EXTERNAL_AUDIO_START_EVENT_ROLE, None)
        item.setToolTip("")

    def _set_external_audio_start_item_metadata(
        self,
        item: QTableWidgetItem,
        event_name: str,
        start_seconds: float,
    ) -> None:
        cleaned_name = str(event_name).strip()
        if not cleaned_name:
            self._clear_external_audio_start_item_metadata(item)
            return
        item.setData(EDIT_EXTERNAL_AUDIO_START_EVENT_ROLE, cleaned_name)
        item.setToolTip(f"Event: {cleaned_name} | Baslangic: {format_time_dk_sn_ms(start_seconds)}")

    def _external_audio_start_event_names(self) -> list[Optional[str]]:
        event_names: list[Optional[str]] = []
        if not hasattr(self, "edit_external_audio_table"):
            return event_names

        table = self.edit_external_audio_table
        for row in range(table.rowCount()):
            row_values = [
                self._edit_table_cell_text(table, row, EDIT_EXTERNAL_AUDIO_COL_FILE),
                self._edit_table_cell_text(table, row, EDIT_EXTERNAL_AUDIO_COL_START),
                self._edit_table_cell_text(table, row, EDIT_EXTERNAL_AUDIO_COL_END),
                self._edit_table_cell_text(table, row, EDIT_EXTERNAL_AUDIO_COL_VOLUME),
            ]
            if not any(row_values):
                continue
            start_item = table.item(row, EDIT_EXTERNAL_AUDIO_COL_START)
            raw_event_name = start_item.data(EDIT_EXTERNAL_AUDIO_START_EVENT_ROLE) if start_item is not None else None
            event_name = str(raw_event_name).strip() if raw_event_name is not None else ""
            event_names.append(event_name or None)
        return event_names

    def _begin_external_audio_start_edit(self, row: int) -> None:
        table = self.edit_external_audio_table
        if table.state() == QAbstractItemView.State.EditingState:
            return
        item = table.item(row, EDIT_EXTERNAL_AUDIO_COL_START)
        if item is None:
            item = QTableWidgetItem("0.0")
            table.setItem(row, EDIT_EXTERNAL_AUDIO_COL_START, item)
        table.editItem(item)

    def _pick_external_audio_start_from_events(self, row: int) -> None:
        if not hasattr(self, "edit_external_audio_table"):
            return

        event_start_times = self._available_event_start_times()
        if not event_start_times:
            self._begin_external_audio_start_edit(row)
            return

        table = self.edit_external_audio_table
        start_item = table.item(row, EDIT_EXTERNAL_AUDIO_COL_START)
        if start_item is None:
            start_item = QTableWidgetItem("0.0")
            table.setItem(row, EDIT_EXTERNAL_AUDIO_COL_START, start_item)

        current_event_name_raw = start_item.data(EDIT_EXTERNAL_AUDIO_START_EVENT_ROLE)
        current_event_name = str(current_event_name_raw).strip() if current_event_name_raw is not None else ""

        option_labels = ["Elle Gir"]
        option_map: dict[str, tuple[str, float]] = {}
        current_index = 0
        for index, (event_name, start_seconds) in enumerate(event_start_times.items(), start=1):
            label = f"{event_name} ({format_time_dk_sn_ms(start_seconds)})"
            option_labels.append(label)
            option_map[label] = (event_name, start_seconds)
            if event_name == current_event_name:
                current_index = index

        selected_label, accepted = QInputDialog.getItem(
            self,
            "Harici Ses Baslangici",
            "Event secin:",
            option_labels,
            current_index,
            False,
        )
        if not accepted:
            return
        if selected_label == "Elle Gir":
            self._clear_external_audio_start_item_metadata(start_item)
            self._begin_external_audio_start_edit(row)
            return

        selected_event = option_map.get(selected_label)
        if selected_event is None:
            return

        event_name, start_seconds = selected_event
        table.blockSignals(True)
        try:
            start_item.setText(self._format_overlay_number(start_seconds, 3))
            self._set_external_audio_start_item_metadata(start_item, event_name, start_seconds)
        finally:
            table.blockSignals(False)
        self._update_edit_controls()

    def on_edit_external_audio_cell_clicked(self, row: int, col: int) -> None:
        if not hasattr(self, "edit_external_audio_table"):
            return

        table = self.edit_external_audio_table
        if row < 0 or row >= table.rowCount() or not table.isEnabled():
            return

        if col != EDIT_EXTERNAL_AUDIO_COL_FILE:
            if col == EDIT_EXTERNAL_AUDIO_COL_START:
                self._pick_external_audio_start_from_events(row)
                return
            if col in (EDIT_EXTERNAL_AUDIO_COL_END, EDIT_EXTERNAL_AUDIO_COL_VOLUME):
                if table.state() == QAbstractItemView.State.EditingState:
                    return
                item = table.item(row, col)
                if item is None:
                    default_value = ""
                    if col == EDIT_EXTERNAL_AUDIO_COL_VOLUME:
                        default_value = "1.0"
                    item = QTableWidgetItem(default_value)
                    table.setItem(row, col, item)
                table.editItem(item)
            return

        current_value = self._edit_table_cell_text(table, row, EDIT_EXTERNAL_AUDIO_COL_FILE)
        initial_dir = ""
        if current_value:
            if os.path.isfile(current_value):
                initial_dir = os.path.dirname(current_value)
            elif os.path.isdir(current_value):
                initial_dir = current_value
        if not initial_dir and self.video_meta is not None and self.video_meta.source_video:
            video_dir = os.path.dirname(self.video_meta.source_video)
            if os.path.isdir(video_dir):
                initial_dir = video_dir
        if not initial_dir:
            initial_dir = self.settings.value(SETTINGS_LAST_VIDEO_DIR, "", type=str).strip()
        if not initial_dir or not os.path.isdir(initial_dir):
            initial_dir = os.getcwd()

        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "Ses Dosyasi Sec",
            initial_dir,
            "Audio Files (*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.opus *.wma *.aiff *.aif);;All Files (*.*)",
        )
        if not selected_path:
            return

        table.blockSignals(True)
        try:
            item = table.item(row, EDIT_EXTERNAL_AUDIO_COL_FILE)
            if item is None:
                item = QTableWidgetItem(selected_path)
                table.setItem(row, EDIT_EXTERNAL_AUDIO_COL_FILE, item)
            else:
                item.setText(selected_path)
        finally:
            table.blockSignals(False)
        self._update_edit_controls()

    def on_save_external_audio_rows_clicked(self) -> None:
        if not hasattr(self, "edit_external_audio_table"):
            return
        tracks, track_error = self._collect_external_audio_tracks()
        if track_error is not None:
            QMessageBox.warning(self, "Harici Ses Katmanlari", track_error)
            return
        if not tracks:
            QMessageBox.information(self, "Harici Ses Katmanlari", "Kaydedilecek gecerli harici ses satiri yok.")
            return

        source_video = self.video_meta.source_video if self.video_meta is not None else ""
        initial_dir = self._default_project_directory()
        video_name = os.path.splitext(os.path.basename(source_video))[0].strip() if source_video else "video"
        if not video_name:
            video_name = "video"
        initial_path = os.path.join(initial_dir, f"{video_name}_external_audio_tracks.json")
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Harici Ses Tablosunu Kaydet",
            initial_path,
            "JSON Files (*.json);;All Files (*.*)",
        )
        if not save_path:
            return
        if not save_path.lower().endswith(".json"):
            save_path += ".json"

        serializable_items: list[dict] = []
        for track, start_event_name in zip(tracks, self._external_audio_start_event_names()):
            payload_item = dict(track)
            if start_event_name:
                payload_item["start_event_name"] = start_event_name
            serializable_items.append(payload_item)

        payload = {
            "type": "external_audio_tracks",
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_video": source_video,
            "items": serializable_items,
        }
        try:
            with open(save_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)
        except OSError as exc:
            QMessageBox.critical(self, "Harici Ses Katmanlari", f"Kayit basarisiz:\n{exc}")
            return

        self.statusBar().showMessage(f"Harici ses tablosu kaydedildi: {save_path}", 3500)
        self._append_edit_log(f"Harici ses tablosu kaydedildi: {save_path}")

    def on_import_external_audio_rows_clicked(self) -> None:
        if not hasattr(self, "edit_external_audio_table"):
            return
        initial_dir = self._default_project_directory()

        load_path, _ = QFileDialog.getOpenFileName(
            self,
            "Harici Ses Tablosunu Import Et",
            initial_dir,
            "JSON Files (*.json);;All Files (*.*)",
        )
        if not load_path:
            return

        try:
            with open(load_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Harici Ses Katmanlari", f"Dosya okunamadi:\n{exc}")
            return

        raw_items: object = payload
        if isinstance(payload, dict):
            if isinstance(payload.get("items"), list):
                raw_items = payload.get("items")
            elif isinstance(payload.get("external_audio_tracks"), list):
                raw_items = payload.get("external_audio_tracks")
            elif isinstance(payload.get("tracks"), list):
                raw_items = payload.get("tracks")

        if not isinstance(raw_items, list):
            QMessageBox.warning(self, "Harici Ses Katmanlari", "Dosya formati gecersiz: ses listesi bulunamadi.")
            return

        imported_rows: list[tuple[str, str, str, str, Optional[str]]] = []
        for index, raw_item in enumerate(raw_items, start=1):
            if not isinstance(raw_item, dict):
                QMessageBox.warning(self, "Harici Ses Katmanlari", f"Satir {index}: Veri dict formatinda olmali.")
                return

            path_value = str(raw_item.get("path", raw_item.get("file", ""))).strip()
            if not path_value:
                QMessageBox.warning(self, "Harici Ses Katmanlari", f"Satir {index}: Dosya yolu bos olamaz.")
                return

            try:
                start_seconds = float(raw_item.get("start", ""))
            except (TypeError, ValueError):
                QMessageBox.warning(self, "Harici Ses Katmanlari", f"Satir {index}: Baslangic gecersiz.")
                return
            if start_seconds < 0.0:
                QMessageBox.warning(self, "Harici Ses Katmanlari", f"Satir {index}: Baslangic 0'dan kucuk olamaz.")
                return

            end_seconds: Optional[float] = None
            raw_end = raw_item.get("end")
            if raw_end not in (None, ""):
                try:
                    end_seconds = float(raw_end)
                except (TypeError, ValueError):
                    QMessageBox.warning(self, "Harici Ses Katmanlari", f"Satir {index}: Bitis gecersiz.")
                    return
                if end_seconds <= start_seconds:
                    QMessageBox.warning(
                        self,
                        "Harici Ses Katmanlari",
                        f"Satir {index}: Bitis baslangictan buyuk olmali.",
                    )
                    return

            try:
                volume_value = float(raw_item.get("volume", 1.0))
            except (TypeError, ValueError):
                QMessageBox.warning(self, "Harici Ses Katmanlari", f"Satir {index}: Ses seviyesi gecersiz.")
                return
            if volume_value < 0.0 or volume_value > 4.0:
                QMessageBox.warning(
                    self,
                    "Harici Ses Katmanlari",
                    f"Satir {index}: Ses seviyesi 0.0-4.0 araliginda olmali.",
                )
                return

            start_event_name_raw = raw_item.get("start_event_name")
            start_event_name = str(start_event_name_raw).strip() if start_event_name_raw is not None else ""
            imported_rows.append(
                (
                    path_value,
                    self._format_overlay_number(start_seconds, 3),
                    self._format_overlay_number(end_seconds, 3) if end_seconds is not None else "",
                    self._format_overlay_number(volume_value, 3),
                    start_event_name or None,
                )
            )

        table = self.edit_external_audio_table
        table.blockSignals(True)
        try:
            table.setRowCount(0)
            for path_value, start_value, end_value, volume_value, start_event_name in imported_rows:
                self._append_external_audio_row(
                    path_value,
                    start_value,
                    end_value,
                    volume_value,
                    start_event_name=start_event_name,
                )
        finally:
            table.blockSignals(False)

        if imported_rows and hasattr(self, "edit_external_audio_enabled_checkbox"):
            self.edit_external_audio_enabled_checkbox.setChecked(True)
        self._update_edit_controls()
        self.statusBar().showMessage(f"Harici ses tablosu import edildi: {os.path.basename(load_path)}", 3500)
        self._append_edit_log(f"Harici ses tablosu import edildi: {load_path}")

    def _clear_edit_overlay_preview_cache(self) -> None:
        self._edit_preview_first_frame_bgr = None
        self._edit_preview_first_frame_source = ""

    def _load_edit_overlay_preview_frame(self, force_reload: bool) -> Optional[np.ndarray]:
        if self.video_meta is None:
            return None
        source_path = self.video_meta.source_video
        if not source_path or not os.path.isfile(source_path):
            return None
        if (
            (not force_reload)
            and self._edit_preview_first_frame_bgr is not None
            and self._edit_preview_first_frame_source == source_path
        ):
            return self._edit_preview_first_frame_bgr

        capture = cv2.VideoCapture(source_path)
        if not capture.isOpened():
            return None
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.video_frame_count = max(self.video_frame_count, frame_count)
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
            if not ok or frame is None:
                return None
        finally:
            capture.release()

        self._edit_preview_first_frame_bgr = frame
        self._edit_preview_first_frame_source = source_path
        return frame

    def _set_edit_overlay_preview_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        if not hasattr(self, "edit_overlay_preview_label"):
            return
        if pixmap is None:
            self.edit_overlay_preview_label.clear()
            self.edit_overlay_preview_label.setText("Video secildiginde ilk kare burada gorunecek.")
            return
        target_width = max(1, self.edit_overlay_preview_label.width() - 10)
        target_height = max(1, self.edit_overlay_preview_label.height() - 10)
        scaled = pixmap.scaled(
            target_width,
            target_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.edit_overlay_preview_label.setPixmap(scaled)

    def _draw_image_overlays_on_pixmap(self, base_pixmap: QPixmap, image_overlays: list[dict]) -> Tuple[QPixmap, int]:
        rendered = QPixmap(base_pixmap)
        painter = QPainter(rendered)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        canvas_width = rendered.width()
        canvas_height = rendered.height()
        drawn_count = 0

        for overlay in image_overlays:
            image_path = str(overlay.get("path", "")).strip()
            if not image_path or not os.path.isfile(image_path):
                continue
            image = QImage(image_path)
            if image.isNull():
                continue
            try:
                x_value = float(overlay.get("x", 0.0))
                y_value = float(overlay.get("y", 0.0))
            except (TypeError, ValueError):
                continue
            draw_x = int(round(float(canvas_width) * x_value))
            draw_y = int(round(float(canvas_height) * y_value))

            overlay_pixmap = QPixmap.fromImage(image)
            width_value = overlay.get("width")
            height_value = overlay.get("height")
            if width_value is not None or height_value is not None:
                try:
                    target_width = max(1, int(width_value)) if width_value is not None else overlay_pixmap.width()
                except (TypeError, ValueError):
                    target_width = overlay_pixmap.width()
                try:
                    target_height = max(1, int(height_value)) if height_value is not None else overlay_pixmap.height()
                except (TypeError, ValueError):
                    target_height = overlay_pixmap.height()
                overlay_pixmap = overlay_pixmap.scaled(
                    target_width,
                    target_height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            painter.drawPixmap(draw_x, draw_y, overlay_pixmap)
            drawn_count += 1

        painter.end()
        return rendered, drawn_count

    def _draw_text_overlays_on_pixmap(self, base_pixmap: QPixmap, text_overlays: list[dict]) -> Tuple[QPixmap, int]:
        rendered = QPixmap(base_pixmap)
        painter = QPainter(rendered)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        canvas_width = rendered.width()
        canvas_height = rendered.height()
        drawn_count = 0

        for overlay in text_overlays:
            text_value = str(overlay.get("text", "")).strip()
            if not text_value:
                continue
            try:
                x_value = float(overlay.get("x", 0.0))
                y_value = float(overlay.get("y", 0.0))
            except (TypeError, ValueError):
                continue

            drawn_count += 1
            font_size = max(8, int(overlay.get("font_size", 24)))
            font = painter.font()
            font.setPixelSize(font_size)
            font.setBold(bool(overlay.get("bold", False)))
            font.setItalic(bool(overlay.get("italic", False)))
            painter.setFont(font)
            metrics = painter.fontMetrics()
            text_width = metrics.horizontalAdvance(text_value)
            text_height = metrics.height()
            draw_x = int(round((canvas_width - text_width) * x_value))
            draw_y_top = int(round((canvas_height - text_height) * y_value))
            baseline = draw_y_top + metrics.ascent()
            color_value = str(overlay.get("color", "#FFFFFF")).strip() or "#FFFFFF"
            pen_color = QColor(color_value)
            if not pen_color.isValid():
                pen_color = QColor("#FFFFFF")
            painter.setPen(pen_color)
            painter.drawText(draw_x, baseline, text_value)

        painter.end()
        return rendered, drawn_count

    def _update_edit_overlay_preview(self, force_frame_reload: bool) -> None:
        if not hasattr(self, "edit_overlay_preview_label"):
            return
        if not hasattr(self, "edit_overlay_preview_status_label"):
            return

        has_video = self.video_meta is not None and os.path.isfile(self.video_meta.source_video)
        if not has_video:
            self._clear_edit_overlay_preview_cache()
            self._set_edit_overlay_preview_pixmap(None)
            self.edit_overlay_preview_status_label.setText("Onizleme icin once video secin.")
            return

        frame_bgr = self._load_edit_overlay_preview_frame(force_reload=force_frame_reload)
        if frame_bgr is None:
            self._clear_edit_overlay_preview_cache()
            self._set_edit_overlay_preview_pixmap(None)
            self.edit_overlay_preview_status_label.setText("Ilk kare okunamadi.")
            return

        preview_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rendered_pixmap = numpy_rgb_to_qpixmap(preview_rgb)
        status_parts: list[str] = []

        image_preview_requested = self._is_edit_image_overlay_enabled() or (
            hasattr(self, "edit_image_overlay_table") and self.edit_image_overlay_table.rowCount() > 0
        )
        if image_preview_requested:
            image_overlays, image_error = self._collect_image_overlays()
            if image_error is not None:
                status_parts.append(f"PNG hata: {image_error}")
            elif not image_overlays:
                status_parts.append("PNG: 0 satir")
            else:
                rendered_pixmap, png_count = self._draw_image_overlays_on_pixmap(rendered_pixmap, image_overlays)
                png_state = "" if self._is_edit_image_overlay_enabled() else " (pasif)"
                status_parts.append(f"PNG: {png_count}/{len(image_overlays)}{png_state}")
        else:
            status_parts.append("PNG kapali")

        text_preview_requested = self._is_edit_text_overlay_enabled() or (
            hasattr(self, "edit_text_overlay_table") and self.edit_text_overlay_table.rowCount() > 0
        )
        if text_preview_requested:
            text_overlays, text_error = self._collect_text_overlays()
            if text_error is not None:
                status_parts.append(f"Yazi hata: {text_error}")
            elif not text_overlays:
                status_parts.append("Yazi: 0 satir")
            else:
                rendered_pixmap, text_count = self._draw_text_overlays_on_pixmap(rendered_pixmap, text_overlays)
                text_state = "" if self._is_edit_text_overlay_enabled() else " (pasif)"
                status_parts.append(f"Yazi: {text_count}/{len(text_overlays)}{text_state}")
        else:
            status_parts.append("Yazi kapali")

        self._set_edit_overlay_preview_pixmap(rendered_pixmap)
        self.edit_overlay_preview_status_label.setText(" | ".join(status_parts))

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

    def on_video_select_screen_pick_clicked(self) -> None:
        if not self.open_video_dialog():
            return
        self._set_current_screen("startup", push_history=True)

    def _complete_startup_selection(self, open_event_tab: bool) -> None:
        self.startup_completed = True
        self._update_analysis_controls()
        if open_event_tab:
            self.switch_to_event_tab()
        else:
            self.switch_to_roi_tab()

    def _event_definitions_for_current_mode(self) -> list[dict]:
        if self.detection_mode == DETECTION_MODE_MANUAL:
            return MANUAL_EVENT_DEFINITIONS
        return EVENT_DEFINITIONS

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
            for event_info in MANUAL_EVENT_DEFINITIONS
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
        self.edit_segments = []
        self.event_log.clear()
        self.event_progress.setValue(0)
        self.timeline_dirty = False
        self._populate_event_table_from_results()
        self._update_edit_segments_label()

    def _apply_detection_mode(
        self,
        mode: Optional[str],
        source: str,
        require_video_if_manual: bool = True,
    ) -> bool:
        if mode not in (DETECTION_MODE_AUTO, DETECTION_MODE_MANUAL, None):
            return False
        if mode == self.detection_mode:
            self._sync_event_mode_combo_to_state()
            self._update_video_select_screen_text()
            self._update_analysis_controls()
            return True

        if self._is_event_detection_running() or self._is_color_analysis_running() or self._is_video_edit_running():
            QMessageBox.information(self, "Mod Degisimi", "Calisan analiz varken mod degistirilemez.")
            self._sync_event_mode_combo_to_state()
            return False

        if self.detection_mode is not None and not self._confirm_discard_timeline_if_needed():
            self._sync_event_mode_combo_to_state()
            return False

        if mode == DETECTION_MODE_MANUAL and require_video_if_manual and not self._has_usable_video():
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
        self._update_video_select_screen_text()
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

    def on_manual_set_start_clicked(self) -> None:
        self._assign_manual_event_time("start")

    def _assign_manual_event_time(self, field_name: str, row_override: Optional[int] = None) -> None:
        if self.detection_mode != DETECTION_MODE_MANUAL:
            return
        if row_override is not None:
            row = int(row_override)
            if row < 0 or row >= len(self.last_detected_events):
                return
        else:
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

    def switch_to_roi_tab(self) -> None:
        self._set_current_screen("roi", push_history=True)

    def switch_to_event_tab(self) -> None:
        self._set_current_screen("event", push_history=True)

    def switch_to_edit_tab(self) -> None:
        self._set_current_screen("edit", push_history=True)

    def on_open_edit_clicked(self) -> None:
        if self.video_meta is None or not os.path.isfile(self.video_meta.source_video):
            QMessageBox.warning(self, "Edit", "Aktif video bulunamadi.")
            return
        if self._is_video_edit_running():
            QMessageBox.information(self, "Edit", "Video edit islemi zaten calisiyor.")
            return

        segments, error = self._build_merged_cut_segments()
        if error is None and segments is not None:
            self.edit_segments = segments
            self.edit_cut_enabled_checkbox.setChecked(True)
        else:
            self.edit_segments = []
            self.edit_cut_enabled_checkbox.setChecked(False)

        source_path = self.video_meta.source_video
        self._update_edit_video_label()
        self._refresh_edit_resolution_options()
        self._update_edit_segments_label()
        self.edit_output_path_edit.setText(self._default_edit_output_directory(source_path))
        self.edit_progress.setValue(0)
        self._reset_edit_progress_time_fields()
        self.edit_log.clear()
        if self.edit_segments:
            self._append_edit_log(self._format_segments_summary(self.edit_segments))
        else:
            self._append_edit_log("Cut adimi varsayilan olarak pasif baslatildi (event start/end eksik).")
            if error:
                self._append_edit_log(f"Cut nedeni: {error}")
        self._update_edit_controls()
        self._update_edit_overlay_preview(force_frame_reload=True)
        self.switch_to_edit_tab()

    def on_edit_output_browse_clicked(self) -> None:
        if self.video_meta is None:
            QMessageBox.warning(self, "Edit", "Once bir video acin.")
            return
        base_dir = self.edit_output_path_edit.text().strip()
        if not base_dir:
            base_dir = self._default_edit_output_directory(self.video_meta.source_video)
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Cikti Klasoru Sec",
            base_dir,
        )
        if not selected_dir:
            return
        self.edit_output_path_edit.setText(selected_dir)
        self._update_edit_controls()

    def on_edit_run_button_clicked(self) -> None:
        if hasattr(self, "edit_progress") and not self.edit_progress.isVisible():
            self.edit_progress.setVisible(True)
        if self._is_video_edit_running():
            self.stop_video_edit()
            return
        self.start_video_edit()

    def _on_edit_operation_checkbox_changed(self, *_args: object) -> None:
        self._update_edit_controls()
        self._update_edit_overlay_preview(force_frame_reload=False)

    @staticmethod
    def _build_cut_rule_tooltip() -> str:
        return (
            "Cut sabit plan ile calisir:\n"
            "1. sol1_al1 -> sol4_koy4\n"
            "2. sag4_al1 -> sag4_koy1 + 3 sn\n"
            "3. sag3_al1 -> sag3_koy2 + 3 sn\n"
            "4. sag2_al1 -> sag2_koy3 + 3 sn\n"
            "5. sag1_al1 -> sag1_koy4 + 3 sn\n"
            "6. Videonun son 3 saniyesi\n"
            "Not: eski start/end eslestirmesi artik kullanilmaz."
        )

    def _video_duration_seconds(self) -> Optional[float]:
        if self.video_meta is None:
            return None
        fps = max(0.0, float(self.video_meta.fps))
        if fps <= 0.0:
            return None
        if self.video_frame_count > 0:
            return max(0.0, float(self.video_frame_count) / fps)
        if int(self.video_meta.frame_index) > 0:
            return max(0.0, float(self.video_meta.frame_index) / fps)
        return None

    def _build_merged_cut_segments(self) -> Tuple[Optional[list[tuple[float, float]]], Optional[str]]:
        if not self.last_detected_events:
            return None, "Kesim icin timeline olaylari bulunamadi."

        video_duration = self._video_duration_seconds()
        if video_duration is None or video_duration <= 0.0:
            return None, "Videonun suresi belirlenemedi."

        event_times: dict[str, float] = {}
        for event_payload in self.last_detected_events:
            if not isinstance(event_payload, dict):
                continue
            event_name = str(event_payload.get("name", "")).strip()
            if not event_name:
                continue
            raw_start = event_payload.get("start")
            if raw_start is None:
                continue
            try:
                event_times[event_name] = float(raw_start)
            except (TypeError, ValueError):
                return None, f"{event_name} zaman bilgisi gecersiz."

        raw_segments: list[tuple[float, float]] = []
        for start_name, end_name, end_offset_seconds in CUT_EVENT_SEGMENT_RULES:
            start_seconds = event_times.get(start_name)
            if start_seconds is None:
                return None, f"Cut icin {start_name} zamani gerekli."
            end_marker_seconds = event_times.get(end_name)
            if end_marker_seconds is None:
                return None, f"Cut icin {end_name} zamani gerekli."
            segment_start = max(0.0, min(float(start_seconds), video_duration))
            segment_end = max(0.0, min(float(end_marker_seconds) + float(end_offset_seconds), video_duration))
            if segment_end <= segment_start:
                return None, f"Cut araligi gecersiz: {start_name} -> {end_name}"
            raw_segments.append((segment_start, segment_end))

        tail_start = max(0.0, float(video_duration) - CUT_VIDEO_TAIL_SECONDS)
        if video_duration > tail_start:
            raw_segments.append((tail_start, float(video_duration)))

        if not raw_segments:
            return None, "Gecerli kesim araligi bulunamadi."

        sorted_segments = sorted(raw_segments, key=lambda item: (item[0], item[1]))
        merged_segments: list[tuple[float, float]] = []
        for start_seconds, end_seconds in sorted_segments:
            if not merged_segments:
                merged_segments.append((start_seconds, end_seconds))
                continue
            last_start, last_end = merged_segments[-1]
            if start_seconds <= last_end:
                merged_segments[-1] = (last_start, max(last_end, end_seconds))
            else:
                merged_segments.append((start_seconds, end_seconds))

        if not merged_segments:
            return None, "Gecerli kesim araligi bulunamadi."
        return merged_segments, None

    @staticmethod
    def _default_edit_output_directory(source_video: str) -> str:
        source_dir = os.path.dirname(os.path.abspath(source_video))
        if source_dir and os.path.isdir(source_dir):
            return source_dir
        return os.getcwd()

    @staticmethod
    def _default_project_directory() -> str:
        if getattr(sys, "frozen", False):
            project_dir = os.path.dirname(os.path.abspath(sys.executable))
        else:
            project_dir = os.path.dirname(os.path.abspath(__file__))
        if project_dir and os.path.isdir(project_dir):
            return project_dir
        return os.getcwd()

    def _default_edit_output_path(self, source_video: str, output_directory: str) -> str:
        source_dir = str(output_directory).strip()
        if not source_dir:
            source_dir = self._default_edit_output_directory(source_video)
        source_name = os.path.splitext(os.path.basename(source_video))[0].strip()
        if not source_name:
            source_name = "video"

        timestamp_suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
        cut_timestamp_pattern = re.compile(r"(?i)(?:_)?cut\d{8}-\d{6}")
        cut_matches = list(cut_timestamp_pattern.finditer(source_name))
        if cut_matches:
            last_match = cut_matches[-1]
            source_name = (
                f"{source_name[:last_match.start()]}"
                f"_cut{timestamp_suffix}"
                f"{source_name[last_match.end():]}"
            )
        else:
            source_name = f"{source_name}_cut{timestamp_suffix}"
        return os.path.join(source_dir, f"{source_name}.mp4")

    @staticmethod
    def _suggest_non_conflicting_output_path(path: str) -> str:
        base_dir = os.path.dirname(path)
        base_name, extension = os.path.splitext(os.path.basename(path))
        extension = extension or ".mp4"
        for index in range(1, 10000):
            candidate = os.path.join(base_dir, f"{base_name}_{index}{extension}")
            if not os.path.exists(candidate):
                return candidate
        return os.path.join(base_dir, f"{base_name}_{int(time.time())}{extension}")

    def _prompt_rename_output_path(self, existing_path: str) -> Optional[str]:
        suggested_path = self._suggest_non_conflicting_output_path(existing_path)
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Cikti Dosyasi Zaten Var",
            suggested_path,
            "MP4 Files (*.mp4);;All Files (*.*)",
        )
        if not save_path:
            return None
        normalized = save_path.strip()
        if not normalized:
            return None
        if not normalized.lower().endswith(".mp4"):
            normalized += ".mp4"
        return normalized

    def _format_segments_summary(self, segments: list[tuple[float, float]]) -> str:
        total = sum(max(0.0, end_seconds - start_seconds) for start_seconds, end_seconds in segments)
        return f"Kesim segmentleri hazirlandi: {len(segments)} adet | Toplam: {total:.2f} sn"

    def _describe_preset_effect(self, preset: str) -> str:
        preset_key = str(preset).strip().lower()
        if preset_key == "ultrafast":
            return "ultrafast: en hizli, dosya boyutu genelde daha buyuk."
        if preset_key == "fast":
            return "fast: hizli, dosya boyutu medium/slow'dan genelde buyuk."
        if preset_key == "medium":
            return "medium: hiz ve boyut dengeli."
        if preset_key == "slow":
            return "slow: daha yavas, dosya boyutu genelde daha kucuk."
        return f"{preset_key}: secili preset hiz/boyut dengesini belirler."

    def _describe_crf_effect(self, crf: int) -> str:
        value = max(0, min(51, int(crf)))
        if value == 0:
            return "CRF 0: neredeyse kayipsiz, dosya cok buyuk."
        if 1 <= value <= 17:
            return "CRF 1-17: cok yuksek kalite, buyuk dosya."
        if 18 <= value <= 22:
            return "CRF 18-22: yuksek kalite, dengeli boyut."
        if 23 <= value <= 28:
            return "CRF 23-28: orta kalite, daha kucuk dosya."
        if 29 <= value <= 40:
            return "CRF 29-40: dusuk kaliteye iner, dosya kuculur."
        return "CRF 41-51: cok dusuk kalite, en kucuk dosya."

    def _build_edit_quality_tooltip(self) -> str:
        preset = "medium"
        if hasattr(self, "edit_preset_combo"):
            selected_preset = self.edit_preset_combo.currentText().strip().lower()
            if selected_preset:
                preset = selected_preset
        crf_value = int(self.edit_crf_spin.value()) if hasattr(self, "edit_crf_spin") else 25

        preset_effect = self._describe_preset_effect(preset)
        crf_effect = self._describe_crf_effect(crf_value)
        return (
            "Preset: hiz/boyut dengesini belirler (ultrafast > fast > medium > slow).\n"
            f"{preset_effect}\n"
            "CRF: dusuk = daha iyi kalite, yuksek = daha kucuk dosya.\n"
            f"{crf_effect}\n"
            f"Secili: preset={preset}, crf={crf_value}"
        )

    def _update_edit_quality_tooltips(self, *_args: object) -> None:
        if not hasattr(self, "edit_preset_label") or not hasattr(self, "edit_crf_label"):
            return
        tooltip_text = self._build_edit_quality_tooltip()
        self.edit_preset_label.setToolTip(tooltip_text)
        self.edit_crf_label.setToolTip(tooltip_text)

    def _update_edit_video_label(self) -> None:
        if not hasattr(self, "edit_source_video_label"):
            return
        if self.video_meta is None:
            self.edit_source_video_label.setText("Kaynak video: -")
            self._refresh_edit_resolution_options()
            self._update_edit_overlay_preview(force_frame_reload=True)
            return
        self.edit_source_video_label.setText(f"Kaynak video: {self.video_meta.source_video}")
        self._refresh_edit_resolution_options()
        self._update_edit_overlay_preview(force_frame_reload=True)

    @staticmethod
    def _format_edit_fps_value(value: float) -> str:
        rounded = round(float(value), 2)
        if abs(rounded - round(rounded)) < 0.0001:
            return str(int(round(rounded)))
        return f"{rounded:.2f}".rstrip("0").rstrip(".")

    @staticmethod
    def _closest_p_label(height: int) -> str:
        candidates = [2160, 1440, 1080, 720, 480]
        target = min(candidates, key=lambda item: abs(int(height) - int(item)))
        return f"{int(target)}p"

    @staticmethod
    def _k_label(width: int) -> Optional[str]:
        if int(width) >= 3840:
            return "4K"
        if int(width) >= 2560:
            return "2K"
        return None

    def _resolution_class_text(self, width: int, height: int) -> str:
        labels: list[str] = [self._closest_p_label(height)]
        k_label = self._k_label(width)
        if k_label is not None:
            labels.append(k_label)
        return ", ".join(labels)

    def _is_edit_remove_audio_enabled(self) -> bool:
        return hasattr(self, "edit_remove_audio_checkbox") and self.edit_remove_audio_checkbox.isChecked()

    def _is_edit_speed_enabled(self) -> bool:
        return hasattr(self, "edit_speed_enabled_checkbox") and self.edit_speed_enabled_checkbox.isChecked()

    def _is_edit_text_overlay_enabled(self) -> bool:
        return hasattr(self, "edit_text_overlay_enabled_checkbox") and self.edit_text_overlay_enabled_checkbox.isChecked()

    def _is_edit_image_overlay_enabled(self) -> bool:
        return hasattr(self, "edit_image_overlay_enabled_checkbox") and self.edit_image_overlay_enabled_checkbox.isChecked()

    def _is_edit_external_audio_enabled(self) -> bool:
        return hasattr(self, "edit_external_audio_enabled_checkbox") and self.edit_external_audio_enabled_checkbox.isChecked()

    @staticmethod
    def _edit_table_cell_text(table: QTableWidget, row: int, col: int) -> str:
        item = table.item(row, col)
        if item is None:
            return ""
        return item.text().strip()

    @staticmethod
    def _is_valid_hex_color(value: str) -> bool:
        text = value.strip()
        if len(text) != 7 or not text.startswith("#"):
            return False
        try:
            int(text[1:], 16)
        except ValueError:
            return False
        return True

    def _collect_text_overlays(self) -> Tuple[list[dict], Optional[str]]:
        overlays: list[dict] = []
        if not hasattr(self, "edit_text_overlay_table"):
            return overlays, None

        table = self.edit_text_overlay_table
        seen_ids: set[str] = set()
        for row in range(table.rowCount()):
            overlay_id_raw = self._edit_table_cell_text(table, row, EDIT_TEXT_COL_ID)
            text_value = self._edit_table_cell_text(table, row, EDIT_TEXT_COL_TEXT)
            start_raw = self._edit_table_cell_text(table, row, EDIT_TEXT_COL_START)
            end_raw = self._edit_table_cell_text(table, row, EDIT_TEXT_COL_END)
            position_raw = self._edit_table_cell_text(table, row, EDIT_TEXT_COL_POSITION)
            font_size_raw = self._edit_table_cell_text(table, row, EDIT_TEXT_COL_SIZE)
            color_raw = self._edit_table_cell_text(table, row, EDIT_TEXT_COL_COLOR)
            bold_enabled = self._text_overlay_style_value(row, EDIT_TEXT_COL_BOLD)
            italic_enabled = self._text_overlay_style_value(row, EDIT_TEXT_COL_ITALIC)

            row_values = [text_value, start_raw, end_raw, position_raw, font_size_raw, color_raw]
            if not any(row_values):
                continue
            overlay_id = self._normalize_text_overlay_id(overlay_id_raw)
            if not overlay_id:
                return [], f"Yazi katmani satir {row + 1}: ID bos olamaz."
            if overlay_id in seen_ids:
                return [], f"Yazi katmani satir {row + 1}: ID tekrar ediyor ({overlay_id})."
            seen_ids.add(overlay_id)
            if not text_value:
                return [], f"Yazi katmani satir {row + 1}: Metin bos olamaz."
            if not start_raw or not end_raw:
                return [], f"Yazi katmani satir {row + 1}: Baslangic ve bitis zorunlu."

            position_pair = self._parse_text_overlay_position(position_raw)
            if position_pair is None:
                return [], f"Yazi katmani satir {row + 1}: Pozisyon X,Y formatinda olmali."

            try:
                start_seconds = float(start_raw)
                end_seconds = float(end_raw)
                x_value = float(position_pair[0])
                y_value = float(position_pair[1])
                font_size = int(font_size_raw)
            except (TypeError, ValueError):
                return [], f"Yazi katmani satir {row + 1}: Sayisal alanlar gecersiz."

            if start_seconds < 0.0:
                return [], f"Yazi katmani satir {row + 1}: Baslangic 0'dan kucuk olamaz."
            if end_seconds <= start_seconds:
                return [], f"Yazi katmani satir {row + 1}: Bitis baslangictan buyuk olmali."
            if x_value < 0.0 or x_value > 1.0 or y_value < 0.0 or y_value > 1.0:
                return [], f"Yazi katmani satir {row + 1}: X ve Y 0-1 araliginda olmali."
            if font_size < 8 or font_size > 256:
                return [], f"Yazi katmani satir {row + 1}: Boyut 8-256 araliginda olmali."

            color_value = color_raw.upper() if color_raw else "#FFFFFF"
            if not self._is_valid_hex_color(color_value):
                return [], f"Yazi katmani satir {row + 1}: Renk #RRGGBB formatinda olmali."

            overlays.append(
                {
                    "id": overlay_id,
                    "text": text_value,
                    "start": round(start_seconds, 6),
                    "end": round(end_seconds, 6),
                    "x": round(x_value, 6),
                    "y": round(y_value, 6),
                    "font_size": int(font_size),
                    "color": color_value,
                    "bold": bool(bold_enabled),
                    "italic": bool(italic_enabled),
                }
            )
        return overlays, None

    def _collect_image_overlays(self) -> Tuple[list[dict], Optional[str]]:
        overlays: list[dict] = []
        if not hasattr(self, "edit_image_overlay_table"):
            return overlays, None

        table = self.edit_image_overlay_table
        for row in range(table.rowCount()):
            path_value = self._edit_table_cell_text(table, row, EDIT_IMAGE_COL_FILE)
            start_raw = self._edit_table_cell_text(table, row, EDIT_IMAGE_COL_START)
            end_raw = self._edit_table_cell_text(table, row, EDIT_IMAGE_COL_END)
            position_raw = self._edit_table_cell_text(table, row, EDIT_IMAGE_COL_POSITION)
            size_raw = self._edit_table_cell_text(table, row, EDIT_IMAGE_COL_SIZE)

            row_values = [path_value, start_raw, end_raw, position_raw, size_raw]
            if not any(row_values):
                continue
            if not path_value:
                return [], f"PNG katmani satir {row + 1}: Dosya yolu bos olamaz."
            if not os.path.isfile(path_value):
                return [], f"PNG katmani satir {row + 1}: Dosya bulunamadi ({path_value})."
            if not start_raw or not end_raw:
                return [], f"PNG katmani satir {row + 1}: Baslangic ve bitis zorunlu."
            position_pair = self._parse_text_overlay_position(position_raw)
            if position_pair is None:
                return [], f"PNG katmani satir {row + 1}: Pozisyon X,Y formatinda olmali."

            try:
                start_seconds = float(start_raw)
                end_seconds = float(end_raw)
                x_value = float(position_pair[0])
                y_value = float(position_pair[1])
            except (TypeError, ValueError):
                return [], f"PNG katmani satir {row + 1}: Sayisal alanlar gecersiz."

            if start_seconds < 0.0:
                return [], f"PNG katmani satir {row + 1}: Baslangic 0'dan kucuk olamaz."
            if end_seconds <= start_seconds:
                return [], f"PNG katmani satir {row + 1}: Bitis baslangictan buyuk olmali."
            if x_value < 0.0 or x_value > 1.0 or y_value < 0.0 or y_value > 1.0:
                return [], f"PNG katmani satir {row + 1}: X ve Y 0-1 araliginda olmali."

            width_value: Optional[int] = None
            height_value: Optional[int] = None
            if size_raw:
                try:
                    requested_width = int(float(size_raw))
                except (TypeError, ValueError):
                    return [], f"PNG katmani satir {row + 1}: Boyut tam sayi olmali."
                if requested_width <= 0:
                    return [], f"PNG katmani satir {row + 1}: Boyut pozitif olmali."
                scaled_size = self._image_overlay_scaled_dimensions(path_value, requested_width)
                if scaled_size is None:
                    return [], f"PNG katmani satir {row + 1}: Boyut icin gorsel okunamadi."
                width_value, height_value = scaled_size

            overlays.append(
                {
                    "path": path_value,
                    "start": round(start_seconds, 6),
                    "end": round(end_seconds, 6),
                    "x": round(x_value, 6),
                    "y": round(y_value, 6),
                    "width": width_value,
                    "height": height_value,
                }
            )

        return overlays, None

    def _collect_external_audio_tracks(self) -> Tuple[list[dict], Optional[str]]:
        tracks: list[dict] = []
        if not hasattr(self, "edit_external_audio_table"):
            return tracks, None

        table = self.edit_external_audio_table
        for row in range(table.rowCount()):
            path_value = self._edit_table_cell_text(table, row, EDIT_EXTERNAL_AUDIO_COL_FILE)
            start_raw = self._edit_table_cell_text(table, row, EDIT_EXTERNAL_AUDIO_COL_START)
            end_raw = self._edit_table_cell_text(table, row, EDIT_EXTERNAL_AUDIO_COL_END)
            volume_raw = self._edit_table_cell_text(table, row, EDIT_EXTERNAL_AUDIO_COL_VOLUME)

            row_values = [path_value, start_raw, end_raw, volume_raw]
            if not any(row_values):
                continue
            if not path_value:
                return [], f"Harici ses satir {row + 1}: Dosya yolu bos olamaz."
            if not os.path.isfile(path_value):
                return [], f"Harici ses satir {row + 1}: Dosya bulunamadi ({path_value})."
            if not start_raw:
                return [], f"Harici ses satir {row + 1}: Baslangic zorunlu."

            try:
                start_seconds = float(start_raw)
            except (TypeError, ValueError):
                return [], f"Harici ses satir {row + 1}: Baslangic gecersiz."
            if start_seconds < 0.0:
                return [], f"Harici ses satir {row + 1}: Baslangic 0'dan kucuk olamaz."

            end_seconds: Optional[float] = None
            if end_raw:
                try:
                    end_seconds = float(end_raw)
                except (TypeError, ValueError):
                    return [], f"Harici ses satir {row + 1}: Bitis gecersiz."
                if end_seconds <= start_seconds:
                    return [], f"Harici ses satir {row + 1}: Bitis baslangictan buyuk olmali."

            volume_value = 1.0
            if volume_raw:
                try:
                    volume_value = float(volume_raw)
                except (TypeError, ValueError):
                    return [], f"Harici ses satir {row + 1}: Ses seviyesi gecersiz."
            if volume_value < 0.0 or volume_value > 4.0:
                return [], f"Harici ses satir {row + 1}: Ses seviyesi 0.0-4.0 araliginda olmali."

            tracks.append(
                {
                    "path": path_value,
                    "start": round(start_seconds, 6),
                    "end": round(end_seconds, 6) if end_seconds is not None else None,
                    "volume": round(volume_value, 6),
                }
            )

        return tracks, None

    def _refresh_edit_resolution_options(self) -> None:
        if not hasattr(self, "edit_target_resolution_combo"):
            return

        current_resolution = self.edit_target_resolution_combo.currentData()
        current_fps = self.edit_target_fps_combo.currentData()

        source_width = 0
        source_height = 0
        source_fps = 0.0
        if self.video_meta is not None:
            source_width = max(0, int(self.video_meta.width))
            source_height = max(0, int(self.video_meta.height))
            source_fps = max(0.0, float(self.video_meta.fps))

        self.edit_target_resolution_combo.blockSignals(True)
        self.edit_target_fps_combo.blockSignals(True)
        try:
            self.edit_target_resolution_combo.clear()
            self.edit_target_resolution_combo.addItem("Orijinal", None)
            if self.video_meta is not None:
                for label, (target_w, target_h) in EDIT_RESOLUTION_PRESETS:
                    if source_width > 0 and source_height > 0 and (target_w > source_width or target_h > source_height):
                        continue
                    self.edit_target_resolution_combo.addItem(label, (int(target_w), int(target_h)))

            self.edit_target_fps_combo.clear()
            self.edit_target_fps_combo.addItem("Orijinal", None)
            if self.video_meta is not None:
                for fps_value in EDIT_FPS_PRESETS:
                    if source_fps > 0.0 and fps_value > (source_fps + 0.001):
                        continue
                    self.edit_target_fps_combo.addItem(f"{self._format_edit_fps_value(fps_value)} FPS", float(fps_value))

            def restore_combo_data(combo: QComboBox, data: object) -> None:
                for idx in range(combo.count()):
                    if combo.itemData(idx) == data:
                        combo.setCurrentIndex(idx)
                        return
                combo.setCurrentIndex(0)

            restore_combo_data(self.edit_target_resolution_combo, current_resolution)
            restore_combo_data(self.edit_target_fps_combo, current_fps)
        finally:
            self.edit_target_resolution_combo.blockSignals(False)
            self.edit_target_fps_combo.blockSignals(False)

        if hasattr(self, "edit_current_specs_label"):
            if source_width <= 0 or source_height <= 0:
                self.edit_current_specs_label.setText("Mevcut: -")
            else:
                fps_text = "-" if source_fps <= 0.0 else f"{self._format_edit_fps_value(source_fps)} FPS"
                class_text = self._resolution_class_text(source_width, source_height)
                self.edit_current_specs_label.setText(f"Mevcut: {source_width}x{source_height} ({class_text}) | {fps_text}")

    def _selected_resize_targets(self) -> Tuple[Optional[tuple[int, int]], Optional[float]]:
        target_resolution: Optional[tuple[int, int]] = None
        target_fps: Optional[float] = None

        resolution_data = self.edit_target_resolution_combo.currentData() if hasattr(self, "edit_target_resolution_combo") else None
        if isinstance(resolution_data, (tuple, list)) and len(resolution_data) == 2:
            try:
                target_w = int(resolution_data[0])
                target_h = int(resolution_data[1])
                if target_w > 0 and target_h > 0:
                    target_resolution = (target_w, target_h)
            except (TypeError, ValueError):
                target_resolution = None

        fps_data = self.edit_target_fps_combo.currentData() if hasattr(self, "edit_target_fps_combo") else None
        if fps_data is not None:
            try:
                fps_value = float(fps_data)
                if fps_value > 0.0:
                    target_fps = fps_value
            except (TypeError, ValueError):
                target_fps = None

        return target_resolution, target_fps

    def _effective_resize_targets(self) -> Tuple[Optional[tuple[int, int]], Optional[float]]:
        target_resolution, target_fps = self._selected_resize_targets()
        if self.video_meta is None:
            return target_resolution, target_fps

        source_width = max(0, int(self.video_meta.width))
        source_height = max(0, int(self.video_meta.height))
        source_fps = max(0.0, float(self.video_meta.fps))

        if target_resolution is not None:
            target_w, target_h = target_resolution
            if source_width > 0 and source_height > 0:
                if target_w > source_width or target_h > source_height:
                    target_resolution = None
                elif target_w >= source_width and target_h >= source_height:
                    target_resolution = None

        if target_fps is not None and source_fps > 0.0 and target_fps >= (source_fps - 0.001):
            target_fps = None

        return target_resolution, target_fps

    def _selected_speed_factor(self) -> Optional[float]:
        if not hasattr(self, "edit_speed_combo"):
            return None
        raw_data = self.edit_speed_combo.currentData()
        if raw_data is None:
            return None
        try:
            value = float(raw_data)
        except (TypeError, ValueError):
            return None
        if value <= 0.0:
            return None
        return value

    def _effective_speed_factor(self) -> Optional[float]:
        if not self._is_edit_speed_enabled():
            return None
        value = self._selected_speed_factor()
        if value is None:
            return None
        if abs(value - 1.0) < 0.001:
            return None
        return value

    def _update_edit_segments_label(self) -> None:
        if not hasattr(self, "edit_segments_label"):
            return
        if not self.edit_segments:
            self.edit_segments_label.setText("Hazir segment: -")
            return
        self.edit_segments_label.setText(self._format_segments_summary(self.edit_segments))

    def _append_edit_log(self, message: str) -> None:
        if not hasattr(self, "edit_log"):
            return
        cleaned = message.strip()
        if not cleaned:
            return
        self.edit_log.appendPlainText(cleaned)
        self._last_edit_log_message = cleaned

    def _reset_edit_progress_time_fields(self) -> None:
        self._video_edit_started_monotonic = None
        if hasattr(self, "edit_progress") and isinstance(self.edit_progress, TimeOverlayProgressBar):
            self.edit_progress.set_time_fields(elapsed_seconds=None, remaining_seconds=None)

    def _sync_edit_progress_time_fields(self, percent: int) -> None:
        if not hasattr(self, "edit_progress") or not isinstance(self.edit_progress, TimeOverlayProgressBar):
            return
        bounded = max(0, min(100, int(percent)))
        if self._video_edit_started_monotonic is None:
            self.edit_progress.set_time_fields(
                elapsed_seconds=0.0 if bounded <= 0 else None,
                remaining_seconds=0.0 if bounded >= 100 else None,
            )
            return
        elapsed_seconds = max(0.0, time.monotonic() - self._video_edit_started_monotonic)
        remaining_seconds: Optional[float]
        if bounded <= 0:
            remaining_seconds = None
        elif bounded >= 100:
            remaining_seconds = 0.0
        else:
            remaining_seconds = elapsed_seconds * (100.0 - float(bounded)) / float(bounded)
        self.edit_progress.set_time_fields(elapsed_seconds=elapsed_seconds, remaining_seconds=remaining_seconds)

    def _is_video_edit_running(self) -> bool:
        return self._video_edit_busy or (self.edit_thread is not None and self.edit_thread.isRunning())

    def _update_edit_controls(self, *_args: object) -> None:
        if not hasattr(self, "edit_run_button"):
            return
        is_running = self._is_video_edit_running()
        has_video = self.video_meta is not None and os.path.isfile(self.video_meta.source_video)
        output_directory = self.edit_output_path_edit.text().strip()
        output_directory_ready = bool(output_directory) and os.path.isdir(output_directory)
        cut_enabled = self.edit_cut_enabled_checkbox.isChecked()
        resize_enabled = self.edit_resize_enabled_checkbox.isChecked() if hasattr(self, "edit_resize_enabled_checkbox") else False
        remove_audio_enabled = self._is_edit_remove_audio_enabled()
        speed_enabled = self._is_edit_speed_enabled()
        text_overlay_enabled = self._is_edit_text_overlay_enabled()
        image_overlay_enabled = self._is_edit_image_overlay_enabled()
        external_audio_enabled = self._is_edit_external_audio_enabled()

        has_segments = bool(self.edit_segments)
        cut_ready = True
        merged_segments: Optional[list[tuple[float, float]]] = None
        if cut_enabled:
            merged_segments, merge_error = self._build_merged_cut_segments()
            cut_ready = merge_error is None and merged_segments is not None and bool(merged_segments)
            has_segments = cut_ready

        target_resolution, target_fps = self._effective_resize_targets()
        resize_ready = (not resize_enabled) or (target_resolution is not None or target_fps is not None)
        speed_factor = self._selected_speed_factor()
        speed_ready = (not speed_enabled) or (speed_factor is not None)
        effective_speed_factor = self._effective_speed_factor()
        text_overlays, text_overlay_error = self._collect_text_overlays()
        image_overlays, image_overlay_error = self._collect_image_overlays()
        external_audio_tracks, external_audio_error = self._collect_external_audio_tracks()
        text_overlay_ready = (not text_overlay_enabled) or (text_overlay_error is None and bool(text_overlays))
        image_overlay_ready = (not image_overlay_enabled) or (image_overlay_error is None and bool(image_overlays))
        external_audio_ready = (not external_audio_enabled) or (
            external_audio_error is None and bool(external_audio_tracks)
        )
        visual_overlay_effective = (
            (text_overlay_enabled and bool(text_overlays)) or (image_overlay_enabled and bool(image_overlays))
        )
        external_audio_effective = external_audio_enabled and bool(external_audio_tracks)

        has_selected_operation = (
            cut_enabled
            or resize_enabled
            or remove_audio_enabled
            or speed_enabled
            or text_overlay_enabled
            or image_overlay_enabled
            or external_audio_enabled
        )
        has_effective_operation = (
            (cut_enabled and has_segments)
            or (resize_enabled and (target_resolution is not None or target_fps is not None))
            or remove_audio_enabled
            or (effective_speed_factor is not None)
            or visual_overlay_effective
            or external_audio_effective
        )

        analysis_running = self._is_event_detection_running() or self._is_color_analysis_running()
        can_run = (
            has_video
            and output_directory_ready
            and has_selected_operation
            and has_effective_operation
            and cut_ready
            and resize_ready
            and speed_ready
            and text_overlay_ready
            and image_overlay_ready
            and external_audio_ready
            and (not is_running)
            and (not analysis_running)
        )

        self.edit_cut_enabled_checkbox.setEnabled(not is_running)
        if hasattr(self, "edit_resize_enabled_checkbox"):
            self.edit_resize_enabled_checkbox.setEnabled(not is_running)
        self.edit_output_path_edit.setEnabled(not is_running)
        self.edit_output_browse_button.setEnabled(not is_running)
        self.edit_preset_combo.setEnabled(not is_running)
        self.edit_crf_spin.setEnabled(not is_running)
        if hasattr(self, "edit_target_resolution_combo"):
            self.edit_target_resolution_combo.setEnabled((not is_running) and resize_enabled)
        if hasattr(self, "edit_target_fps_combo"):
            self.edit_target_fps_combo.setEnabled((not is_running) and resize_enabled)
        if hasattr(self, "edit_remove_audio_checkbox"):
            self.edit_remove_audio_checkbox.setEnabled(not is_running)
        if hasattr(self, "edit_text_overlay_enabled_checkbox"):
            self.edit_text_overlay_enabled_checkbox.setEnabled(not is_running)
        if hasattr(self, "edit_text_overlay_table"):
            self.edit_text_overlay_table.setEnabled((not is_running) and text_overlay_enabled)
        if hasattr(self, "edit_text_overlay_add_button"):
            self.edit_text_overlay_add_button.setEnabled((not is_running) and text_overlay_enabled)
        if hasattr(self, "edit_text_overlay_remove_button"):
            self.edit_text_overlay_remove_button.setEnabled((not is_running) and text_overlay_enabled)
        if hasattr(self, "edit_text_overlay_save_button"):
            self.edit_text_overlay_save_button.setEnabled(not is_running)
        if hasattr(self, "edit_text_overlay_import_button"):
            self.edit_text_overlay_import_button.setEnabled(not is_running)
        if hasattr(self, "edit_text_overlay_set_event_time_button"):
            self.edit_text_overlay_set_event_time_button.setEnabled(not is_running)
        if hasattr(self, "edit_image_overlay_enabled_checkbox"):
            self.edit_image_overlay_enabled_checkbox.setEnabled(not is_running)
        if hasattr(self, "edit_image_overlay_table"):
            self.edit_image_overlay_table.setEnabled((not is_running) and image_overlay_enabled)
        if hasattr(self, "edit_image_overlay_add_button"):
            self.edit_image_overlay_add_button.setEnabled((not is_running) and image_overlay_enabled)
        if hasattr(self, "edit_image_overlay_remove_button"):
            self.edit_image_overlay_remove_button.setEnabled((not is_running) and image_overlay_enabled)
        if hasattr(self, "edit_external_audio_enabled_checkbox"):
            self.edit_external_audio_enabled_checkbox.setEnabled(not is_running)
        if hasattr(self, "edit_external_audio_table"):
            self.edit_external_audio_table.setEnabled((not is_running) and external_audio_enabled)
        if hasattr(self, "edit_external_audio_add_button"):
            self.edit_external_audio_add_button.setEnabled((not is_running) and external_audio_enabled)
        if hasattr(self, "edit_external_audio_remove_button"):
            self.edit_external_audio_remove_button.setEnabled((not is_running) and external_audio_enabled)
        if hasattr(self, "edit_external_audio_save_button"):
            self.edit_external_audio_save_button.setEnabled(not is_running)
        if hasattr(self, "edit_external_audio_import_button"):
            self.edit_external_audio_import_button.setEnabled(not is_running)
        if hasattr(self, "edit_speed_enabled_checkbox"):
            self.edit_speed_enabled_checkbox.setEnabled(not is_running)
        if hasattr(self, "edit_speed_combo"):
            self.edit_speed_combo.setEnabled((not is_running) and speed_enabled)
        if is_running:
            if self._video_edit_cancel_requested:
                self.edit_run_button.setText("Edit Islemi Durduruluyor...")
                self.edit_run_button.setEnabled(False)
            else:
                self.edit_run_button.setText("Edit Islemeyi Durdur")
                self.edit_run_button.setEnabled(True)
        else:
            self.edit_run_button.setText("Edit Islemine Basla")
            self.edit_run_button.setEnabled(can_run)

    def _set_video_edit_controls(self, running: bool) -> None:
        self._video_edit_busy = bool(running)
        self._update_analysis_controls()

    def _resolve_ffmpeg_path(self) -> Optional[str]:
        stored_path = self.settings.value(SETTINGS_FFMPEG_PATH, "", type=str).strip()
        if stored_path and os.path.isfile(stored_path):
            return stored_path

        path_ffmpeg = shutil.which("ffmpeg")
        if path_ffmpeg and os.path.isfile(path_ffmpeg):
            return path_ffmpeg

        initial_dir = ""
        if stored_path:
            candidate_dir = os.path.dirname(stored_path)
            if os.path.isdir(candidate_dir):
                initial_dir = candidate_dir
        if not initial_dir:
            initial_dir = os.getcwd()

        ffmpeg_path, _ = QFileDialog.getOpenFileName(
            self,
            "ffmpeg.exe Sec",
            initial_dir,
            "FFmpeg Executable (ffmpeg.exe);;Executable Files (*.exe);;All Files (*.*)",
        )
        if not ffmpeg_path:
            return None
        ffmpeg_path = ffmpeg_path.strip()
        if not ffmpeg_path or not os.path.isfile(ffmpeg_path):
            QMessageBox.warning(self, "FFmpeg", "Secilen ffmpeg yolu gecersiz.")
            return None
        self.settings.setValue(SETTINGS_FFMPEG_PATH, ffmpeg_path)
        return ffmpeg_path

    def start_video_edit(self) -> None:
        if self._is_video_edit_running():
            QMessageBox.information(self, "Edit", "Video edit islemi zaten calisiyor.")
            return
        if self._is_event_detection_running() or self._is_color_analysis_running():
            QMessageBox.information(self, "Edit", "Analiz calisirken edit baslatilamaz.")
            return
        if self.video_meta is None or not os.path.isfile(self.video_meta.source_video):
            QMessageBox.warning(self, "Edit", "Aktif video bulunamadi.")
            return

        cut_enabled = self.edit_cut_enabled_checkbox.isChecked()
        resize_enabled = self.edit_resize_enabled_checkbox.isChecked() if hasattr(self, "edit_resize_enabled_checkbox") else False
        remove_audio = self._is_edit_remove_audio_enabled()
        speed_enabled = self._is_edit_speed_enabled()
        text_overlay_enabled = self._is_edit_text_overlay_enabled()
        image_overlay_enabled = self._is_edit_image_overlay_enabled()
        external_audio_enabled = self._is_edit_external_audio_enabled()

        if not (
            cut_enabled
            or resize_enabled
            or remove_audio
            or speed_enabled
            or text_overlay_enabled
            or image_overlay_enabled
            or external_audio_enabled
        ):
            QMessageBox.warning(self, "Edit", "En az bir islem icin enable secilmelidir.")
            return

        text_overlays: list[dict] = []
        if text_overlay_enabled:
            text_overlays, text_error = self._collect_text_overlays()
            if text_error is not None:
                QMessageBox.warning(self, "Edit", text_error)
                return
            if not text_overlays:
                QMessageBox.warning(self, "Edit", "Yazi katmani aktifken en az bir gecerli satir olmalidir.")
                return

        image_overlays: list[dict] = []
        if image_overlay_enabled:
            image_overlays, image_error = self._collect_image_overlays()
            if image_error is not None:
                QMessageBox.warning(self, "Edit", image_error)
                return
            if not image_overlays:
                QMessageBox.warning(self, "Edit", "PNG katmani aktifken en az bir gecerli satir olmalidir.")
                return

        external_audio_tracks: list[dict] = []
        if external_audio_enabled:
            external_audio_tracks, external_audio_error = self._collect_external_audio_tracks()
            if external_audio_error is not None:
                QMessageBox.warning(self, "Edit", external_audio_error)
                return
            if not external_audio_tracks:
                QMessageBox.warning(self, "Edit", "Harici ses aktifken en az bir gecerli satir olmalidir.")
                return

        segments: list[tuple[float, float]] = []
        if cut_enabled:
            parsed_segments, error = self._build_merged_cut_segments()
            if error is not None or parsed_segments is None:
                QMessageBox.warning(self, "Edit", error or "Kesim segmentleri hazirlanamadi.")
                return
            segments = parsed_segments
        self.edit_segments = segments
        self._update_edit_segments_label()

        target_resolution: Optional[tuple[int, int]] = None
        target_fps: Optional[float] = None
        if resize_enabled:
            target_resolution, target_fps = self._effective_resize_targets()
            if target_resolution is None and target_fps is None:
                QMessageBox.warning(self, "Edit", "Cozunurluk/FPS icin en az bir dusurme secilmelidir.")
                return

        speed_factor = self._effective_speed_factor()
        if speed_enabled and speed_factor is None:
            QMessageBox.warning(self, "Edit", "Video hizi icin 1.0x disinda bir deger secilmelidir.")
            return

        cut_effective = cut_enabled and bool(self.edit_segments)
        resize_effective = resize_enabled and (target_resolution is not None or target_fps is not None)
        speed_effective = speed_factor is not None
        visual_overlay_effective = bool(text_overlays or image_overlays)
        external_audio_effective = bool(external_audio_tracks)
        has_effective_operation = (
            visual_overlay_effective
            or remove_audio
            or external_audio_effective
            or cut_effective
            or resize_effective
            or speed_effective
        )
        if not has_effective_operation:
            QMessageBox.warning(self, "Edit", "Secilen ayarlarda uygulanacak bir islem bulunamadi.")
            return

        output_directory = self.edit_output_path_edit.text().strip()
        if not output_directory:
            output_directory = self._default_edit_output_directory(self.video_meta.source_video)
            self.edit_output_path_edit.setText(output_directory)
        if not os.path.isdir(output_directory):
            QMessageBox.warning(self, "Edit", "Cikti klasoru bulunamadi.")
            return
        output_path = self._default_edit_output_path(
            source_video=self.video_meta.source_video,
            output_directory=output_directory,
        )

        source_path = self.video_meta.source_video
        if os.path.abspath(source_path) == os.path.abspath(output_path):
            QMessageBox.warning(self, "Edit", "Cikti dosyasi kaynak video ile ayni olamaz.")
            return

        while os.path.exists(output_path):
            renamed_output = self._prompt_rename_output_path(output_path)
            if renamed_output is None:
                return
            output_path = renamed_output

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.isdir(output_dir):
            QMessageBox.warning(self, "Edit", "Cikti klasoru bulunamadi.")
            return
        self.edit_output_path_edit.setText(output_dir or output_directory)
        if os.path.abspath(source_path) == os.path.abspath(output_path):
            QMessageBox.warning(self, "Edit", "Cikti dosyasi kaynak video ile ayni olamaz.")
            return

        ffmpeg_path = self._resolve_ffmpeg_path()
        if ffmpeg_path is None:
            QMessageBox.warning(self, "FFmpeg", "FFmpeg yolu bulunamadi. Islem baslatilmadi.")
            return

        preset = self.edit_preset_combo.currentText().strip() or "medium"
        crf_value = int(self.edit_crf_spin.value())

        self.edit_progress.setValue(0)
        self._video_edit_started_monotonic = time.monotonic()
        self._sync_edit_progress_time_fields(0)
        self.edit_log.clear()
        self._append_edit_log("Zaman Ekseni: Edit Oncesi (kaynak video).")

        if text_overlay_enabled:
            self._append_edit_log(f"Yazi katmani: {len(text_overlays)} satir uygulanacak.")
        else:
            self._append_edit_log("Yazi katmani devre disi.")
        if image_overlay_enabled:
            self._append_edit_log(f"PNG katmani: {len(image_overlays)} satir uygulanacak.")
        else:
            self._append_edit_log("PNG katmani devre disi.")
        if external_audio_enabled:
            self._append_edit_log(f"Harici ses katmani: {len(external_audio_tracks)} satir uygulanacak (mix).")
        else:
            self._append_edit_log("Harici ses katmani devre disi.")

        if cut_enabled:
            self._append_edit_log(self._format_segments_summary(self.edit_segments))
            self._append_edit_log("Cut modu: Re-encode (frame hassas).")
        else:
            self._append_edit_log("Cut adimi devre disi.")

        if resize_enabled:
            if target_resolution is not None:
                self._append_edit_log(f"Hedef cozunurluk: {target_resolution[0]}x{target_resolution[1]}")
            else:
                self._append_edit_log("Hedef cozunurluk: Orijinal")
            if target_fps is not None:
                self._append_edit_log(f"Hedef FPS: {self._format_edit_fps_value(target_fps)}")
            else:
                self._append_edit_log("Hedef FPS: Orijinal")
        else:
            self._append_edit_log("Cozunurluk/FPS adimi devre disi.")

        if remove_audio:
            self._append_edit_log("Ses adimi: Orijinal ses silinecek.")
        else:
            self._append_edit_log("Ses adimi: Orijinal ses korunacak.")

        if speed_enabled and speed_factor is not None:
            self._append_edit_log(f"Hedef hiz: {speed_factor:.2f}x")
        else:
            self._append_edit_log("Video hizi adimi devre disi.")

        planned_steps: list[str] = []
        if visual_overlay_effective:
            planned_steps.append("Yazi/PNG")
        if remove_audio:
            planned_steps.append("Ses Silme")
        if external_audio_effective:
            planned_steps.append("Harici Ses")
        if cut_effective:
            planned_steps.append("Cut")
        if resize_effective:
            planned_steps.append("Cozunurluk/FPS")
        if speed_effective:
            planned_steps.append("Video Hizi")
        self._append_edit_log(f"Planlanan adimlar: {' -> '.join(planned_steps)}")
        self._append_edit_log(f"FFmpeg: {ffmpeg_path}")
        self._append_edit_log(f"Cikti: {output_path}")
        self.statusBar().showMessage("Video edit islemi baslatiliyor...", 2200)

        self._video_edit_cancel_requested = False
        self.edit_thread = QThread(self)
        self.edit_worker = VideoEditWorker(
            ffmpeg_path=ffmpeg_path,
            input_path=source_path,
            output_path=output_path,
            cut_segments=self.edit_segments,
            preset=preset,
            crf=crf_value,
            enable_cut=cut_effective,
            enable_resize=resize_effective,
            target_width=target_resolution[0] if target_resolution is not None else None,
            target_height=target_resolution[1] if target_resolution is not None else None,
            target_fps=target_fps,
            remove_audio=remove_audio,
            enable_speed=speed_effective,
            speed_factor=speed_factor,
            text_overlays=text_overlays,
            image_overlays=image_overlays,
            external_audio_tracks=external_audio_tracks,
            external_audio_mode="mix",
        )
        self.edit_worker.moveToThread(self.edit_thread)

        self.edit_thread.started.connect(self.edit_worker.run)
        self.edit_worker.progress.connect(self.on_video_edit_progress)
        self.edit_worker.result.connect(self.on_video_edit_result)
        self.edit_worker.error.connect(self.on_video_edit_error)
        self.edit_worker.finished.connect(self.on_video_edit_finished)
        self.edit_worker.finished.connect(self.edit_thread.quit)
        self.edit_worker.finished.connect(self.edit_worker.deleteLater)
        self.edit_thread.finished.connect(self.edit_thread.deleteLater)
        self.edit_thread.finished.connect(self.on_video_edit_thread_finished)

        self._set_video_edit_controls(running=True)
        self.edit_thread.start()

    def stop_video_edit(self) -> None:
        if not self._is_video_edit_running() or self.edit_worker is None:
            self.statusBar().showMessage("Calisan video edit islemi yok.", 2200)
            return
        if self._video_edit_cancel_requested:
            return
        self._video_edit_cancel_requested = True
        self.edit_worker.cancel()
        self._append_edit_log("Video edit islemi icin durdurma istendi.")
        self.statusBar().showMessage("Video edit islemi durduruluyor...", 2500)
        self._update_edit_controls()

    def on_video_edit_progress(self, percent: int, message: str) -> None:
        bounded = max(0, min(100, int(percent)))
        self.edit_progress.setValue(bounded)
        self._sync_edit_progress_time_fields(bounded)
        if message.strip():
            self._append_edit_log(message)

    def on_video_edit_result(self, payload: dict) -> None:
        operations = payload.get("operations")
        if isinstance(operations, list) and operations:
            self._append_edit_log(f"Uygulanan adimlar: {', '.join(str(item) for item in operations)}")
        output_path = str(payload.get("output_path", "")).strip()
        if output_path:
            self._append_edit_log(f"Olusan dosya: {output_path}")
            self.statusBar().showMessage(f"Video edit tamamlandi: {output_path}", 4000)
        self.edit_progress.setValue(100)
        self._sync_edit_progress_time_fields(100)

    def on_video_edit_error(self, message: str) -> None:
        self._sync_edit_progress_time_fields(self.edit_progress.value())
        self._append_edit_log(f"Hata: {message}")
        QMessageBox.critical(self, "Video Edit", message)

    def on_video_edit_finished(self) -> None:
        self._sync_edit_progress_time_fields(self.edit_progress.value())
        self._set_video_edit_controls(running=False)
        if self._video_edit_cancel_requested:
            self._append_edit_log("Video edit islemi kullanici tarafindan durduruldu.")
            self.statusBar().showMessage("Video edit islemi durduruldu.", 3200)
        elif self.edit_progress.value() < 100:
            self.statusBar().showMessage("Video edit islemi bitti.", 3000)
        self._video_edit_cancel_requested = False
        self._video_edit_started_monotonic = None

    def on_video_edit_thread_finished(self) -> None:
        self.edit_worker = None
        self.edit_thread = None
        self._update_edit_controls()

    def _update_event_table_frame_height(self) -> None:
        row_height = max(1, int(self.event_table.verticalHeader().defaultSectionSize()))
        header_height = max(24, int(self.event_table.horizontalHeader().height()))
        frame_border = max(0, int(self.event_table.frameWidth()) * 2)
        target_height = header_height + (EVENT_TABLE_VISIBLE_ROWS * row_height) + frame_border
        self.event_table_frame.setMinimumHeight(target_height)
        self.event_table_frame.setMaximumHeight(target_height)

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
        event_definitions = self._event_definitions_for_current_mode()
        self.event_table.setRowCount(len(event_definitions))
        for row, event_info in enumerate(event_definitions):
            self._set_table_item(row, 0, str(event_info["id"]))
            self._set_table_item(row, 1, str(event_info["name"]))
            self._set_table_item(row, 2, str(event_info["target_roi"]))
            self._set_table_item(row, 3, str(event_info["type"]))
            self._set_time_table_item(row, EVENT_COL_START, None)
            self._set_time_table_item(row, EVENT_COL_END, None)
            self._set_table_item(row, EVENT_COL_CONFIDENCE, "0.00")
        self._update_event_table_frame_height()

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
            self._set_table_item(row, EVENT_COL_CONFIDENCE, confidence_text)
        self._update_analysis_controls()

    def _has_complete_event_times(self) -> bool:
        for event_payload in self.last_detected_events:
            for field_name in ("start", "end"):
                raw_value = event_payload.get(field_name)
                if raw_value is None:
                    continue
                try:
                    float(raw_value)
                except (TypeError, ValueError):
                    continue
                return True
        return False

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
            self._update_edit_video_label()
            return
        base_name = os.path.basename(self.video_meta.source_video)
        self.event_video_label.setText(f"Aktif video: {base_name}")
        self._update_edit_video_label()

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
        edit_running = self._is_video_edit_running()
        any_running = event_running or color_running or edit_running
        is_auto = self.detection_mode == DETECTION_MODE_AUTO
        is_manual = self.detection_mode == DETECTION_MODE_MANUAL
        mode_ready = is_auto or is_manual

        self.event_table.setColumnHidden(EVENT_COL_TARGET_ROI, is_manual)
        self.event_table.setColumnHidden(EVENT_COL_TYPE, is_manual)
        self.event_table.setColumnHidden(EVENT_COL_END, is_manual)
        self.event_table.setColumnHidden(EVENT_COL_CONFIDENCE, is_manual)

        self._sync_event_mode_combo_to_state()
        self.event_mode_combo.setEnabled(self.startup_completed and mode_ready and not any_running)
        self.sample_hz_spin.setVisible(is_auto)
        self.source_sensitivity_label.setVisible(is_auto)
        self.source_sensitivity_combo.setVisible(is_auto)
        if is_auto:
            self.event_auto_controls_widget.setVisible(True)
            self.event_auto_controls_widget.setMinimumHeight(0)
            self.event_auto_controls_widget.setMaximumHeight(16777215)
        else:
            self.event_auto_controls_widget.setVisible(False)
            self.event_auto_controls_widget.setMinimumHeight(0)
            self.event_auto_controls_widget.setMaximumHeight(0)
        self.detect_button.setVisible(is_auto)
        self.event_progress.setVisible(is_auto)
        self.event_log.setVisible(is_auto)
        self.manual_controls_box.setVisible(is_manual)
        self.color_roi_label.setVisible(is_auto)
        self.color_roi_combo.setVisible(is_auto)
        self.color_analyze_button.setVisible(is_auto)
        self.color_progress.setVisible(is_auto)
        self.roi_color_timeline.setVisible(is_auto)
        if is_auto and not self.event_lower_section.isVisible():
            self.event_lower_section.setVisible(True)
            self.event_vertical_splitter.setHandleWidth(8)
            self.event_vertical_splitter.setSizes([420, 320])
        elif is_manual and self.event_lower_section.isVisible():
            self.event_lower_section.setVisible(False)
            self.event_vertical_splitter.setHandleWidth(0)
            self.event_vertical_splitter.setSizes([740, 0])
        elif is_manual:
            self.event_vertical_splitter.setHandleWidth(0)

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
            self.detect_button.setEnabled((not color_running) and (not edit_running))
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
            self.color_analyze_button.setEnabled(can_start_color and (not event_running) and (not edit_running))
        else:
            self.color_analyze_button.setText("ROI Renk Analizi")
            self.color_analyze_button.setEnabled(False)

        manual_video_ready = is_manual and self.video_meta is not None
        manual_controls_enabled = manual_video_ready and not any_running
        manual_row_selected = self._manual_selected_row() is not None
        self.manual_frame_slider.setEnabled(manual_controls_enabled)
        self.manual_step_minus_sec_button.setEnabled(manual_controls_enabled)
        self.manual_step_minus_500ms_button.setEnabled(manual_controls_enabled)
        self.manual_step_minus_250ms_button.setEnabled(manual_controls_enabled)
        self.manual_step_minus_100ms_button.setEnabled(manual_controls_enabled)
        self.manual_step_minus_frame_button.setEnabled(manual_controls_enabled)
        self.manual_step_plus_frame_button.setEnabled(manual_controls_enabled)
        self.manual_step_plus_100ms_button.setEnabled(manual_controls_enabled)
        self.manual_step_plus_250ms_button.setEnabled(manual_controls_enabled)
        self.manual_step_plus_500ms_button.setEnabled(manual_controls_enabled)
        self.manual_step_plus_sec_button.setEnabled(manual_controls_enabled)
        self.manual_set_start_button.setEnabled(manual_controls_enabled and manual_row_selected)

        self.load_timeline_button.setEnabled(mode_ready and (not any_running))
        self.save_timeline_button.setEnabled(mode_ready and (not any_running) and self._has_complete_event_times())
        self.edit_button.setEnabled((not any_running) and self._has_usable_video())
        self._update_edit_controls()

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
        self.edit_segments = []
        self.timeline_dirty = False
        self.event_progress.setValue(0)
        self._reset_event_table()
        self._update_edit_segments_label()
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
        if self._is_video_edit_running():
            QMessageBox.information(self, "Olay Tespit", "Video edit calisirken olay tespiti baslatilamaz.")
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

        self._suspend_event_timeline_autoload = True
        try:
            self.switch_to_event_tab()
        finally:
            self._suspend_event_timeline_autoload = False
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
        self.edit_segments = []
        self.timeline_dirty = True
        self._populate_event_table_from_results()
        self._update_edit_segments_label()
        self.event_progress.setValue(100)
        self._append_event_log("Analiz sonucu tabloya yazildi.")

    def start_roi_color_analysis(self) -> None:
        if self.detection_mode != DETECTION_MODE_AUTO:
            QMessageBox.information(self, "ROI Renk Analizi", "Renk analizi sadece otomatik modda kullanilir.")
            return
        if self._is_color_analysis_running():
            QMessageBox.information(self, "ROI Renk Analizi", "Renk analizi zaten calisiyor.")
            return
        if self._is_video_edit_running():
            QMessageBox.information(self, "ROI Renk Analizi", "Video edit calisirken renk analizi baslatilamaz.")
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

        self._suspend_event_timeline_autoload = True
        try:
            self.switch_to_event_tab()
        finally:
            self._suspend_event_timeline_autoload = False
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
        allowed_cols = (EVENT_COL_START,) if self.detection_mode == DETECTION_MODE_MANUAL else (
            EVENT_COL_START,
            EVENT_COL_END,
        )
        if col not in allowed_cols:
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
        if not self._has_complete_event_times():
            return "Kaydetmek icin en az bir event zamani girilmelidir."
        return None

    def _build_events_from_timeline_payload(self, payload: object) -> Tuple[Optional[list[dict]], Optional[str]]:
        if not isinstance(payload, dict):
            return None, "Dosya formati gecersiz."

        raw_events = payload.get("events")
        if not isinstance(raw_events, list):
            return None, "'events' listesi bulunamadi."

        events_by_id: Dict[int, dict] = {}
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                continue
            raw_id = raw_event.get("id")
            try:
                event_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            events_by_id[event_id] = raw_event

        event_definitions = self._event_definitions_for_current_mode()
        normalized_events: list[dict] = []
        for row, event_info in enumerate(event_definitions):
            event_id = int(event_info["id"])
            source_event = events_by_id.get(event_id)
            if source_event is None and row < len(raw_events) and isinstance(raw_events[row], dict):
                source_event = raw_events[row]

            start_seconds: Optional[float] = None
            end_seconds: Optional[float] = None
            confidence_value: Optional[float] = None
            if source_event is not None:
                raw_start = source_event.get("start")
                raw_end = source_event.get("end")

                if raw_start is not None:
                    try:
                        start_seconds = round(float(raw_start), 2)
                    except (TypeError, ValueError):
                        return None, f"Event {event_id} start degeri gecersiz."

                if raw_end is not None:
                    try:
                        end_seconds = round(float(raw_end), 2)
                    except (TypeError, ValueError):
                        return None, f"Event {event_id} end degeri gecersiz."

                raw_confidence = source_event.get("confidence")
                if raw_confidence is not None:
                    try:
                        confidence_value = float(raw_confidence)
                    except (TypeError, ValueError):
                        confidence_value = None

            normalized_events.append(
                {
                    "id": event_id,
                    "name": str(event_info["name"]),
                    "target_roi": str(event_info["target_roi"]),
                    "type": str(event_info["type"]),
                    "start": start_seconds,
                    "end": end_seconds,
                    "confidence": confidence_value,
                }
            )

        return normalized_events, None

    def _timeline_payload_matches_current_mode(self, payload: object) -> bool:
        if not isinstance(payload, dict):
            return False
        raw_events = payload.get("events")
        if not isinstance(raw_events, list):
            return False

        event_definitions = self._event_definitions_for_current_mode()
        if len(raw_events) != len(event_definitions):
            return False

        expected_names = [str(event_info["name"]).strip() for event_info in event_definitions]
        actual_names: list[str] = []
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                return False
            actual_names.append(str(raw_event.get("name", "")).strip())
        return actual_names == expected_names

    def _load_timeline_from_path(
        self,
        load_path: str,
        *,
        quiet: bool = False,
        success_log_message: Optional[str] = None,
        success_status_message: Optional[str] = None,
        require_mode_match: bool = False,
    ) -> bool:
        normalized_path = str(load_path).strip()
        if not normalized_path:
            return False

        try:
            with open(normalized_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            if quiet:
                self.statusBar().showMessage(f"timeline otomatik yuklenemedi: {os.path.basename(normalized_path)}", 4000)
                self._append_event_log(f"timeline otomatik yuklenemedi: {normalized_path} ({exc})")
            else:
                QMessageBox.warning(self, "timeline.json", f"Dosya okunamadi:\n{exc}")
            return False

        if require_mode_match and not self._timeline_payload_matches_current_mode(payload):
            return False

        parsed_events, parse_error = self._build_events_from_timeline_payload(payload)
        if parse_error is not None or parsed_events is None:
            if quiet:
                self.statusBar().showMessage(f"timeline otomatik yuklenemedi: {os.path.basename(normalized_path)}", 4000)
                self._append_event_log(f"timeline otomatik yuklenemedi: {normalized_path} ({parse_error or 'gecersiz veri'})")
            else:
                QMessageBox.warning(self, "timeline.json", parse_error or "timeline verisi gecersiz.")
            return False

        if isinstance(payload, dict):
            raw_sample_hz = payload.get("sample_hz")
            try:
                sample_hz = int(raw_sample_hz)
            except (TypeError, ValueError):
                sample_hz = None
            if sample_hz is not None:
                sample_hz = max(self.sample_hz_spin.minimum(), min(self.sample_hz_spin.maximum(), sample_hz))
                self.last_detection_sample_hz = int(sample_hz)
                self.sample_hz_spin.setValue(int(sample_hz))

        self.last_detected_events = parsed_events
        self.edit_segments = []
        self.timeline_dirty = False
        self._populate_event_table_from_results()
        self._update_edit_segments_label()
        self._append_event_log(success_log_message or f"timeline yuklendi: {normalized_path}")
        self.statusBar().showMessage(
            success_status_message or f"timeline yuklendi: {os.path.basename(normalized_path)}",
            4000,
        )
        return True

    def _try_auto_import_default_timeline(self) -> None:
        if self._suspend_event_timeline_autoload:
            return
        if self._is_event_detection_running() or self._is_color_analysis_running() or self._is_video_edit_running():
            return
        if self.timeline_dirty and self.last_detected_events:
            return

        default_timeline_path = os.path.join(self._default_project_directory(), AUTO_EVENT_TIMELINE_FILENAME)
        if not os.path.isfile(default_timeline_path):
            return

        self._load_timeline_from_path(
            default_timeline_path,
            quiet=True,
            success_log_message=f"timeline otomatik yuklendi: {default_timeline_path}",
            success_status_message=f"{AUTO_EVENT_TIMELINE_FILENAME} otomatik yuklendi",
            require_mode_match=True,
        )

    def load_timeline_json(self) -> None:
        if self._is_event_detection_running() or self._is_color_analysis_running() or self._is_video_edit_running():
            QMessageBox.information(self, "timeline.json", "Analiz calisirken timeline yuklenemez.")
            return

        if getattr(sys, "frozen", False):
            initial_dir = os.path.dirname(os.path.abspath(sys.executable))
        else:
            initial_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isdir(initial_dir):
            initial_dir = os.getcwd()

        load_path, _ = QFileDialog.getOpenFileName(
            self,
            "timeline.json Yukle",
            initial_dir,
            "JSON Files (*.json);;All Files (*.*)",
        )
        if not load_path:
            return

        self._load_timeline_from_path(load_path)

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
        if self._is_video_edit_running():
            QMessageBox.information(self, "Open Video", "Video edit calisirken yeni video acilamaz.")
            return False

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
        else:
            self._invalidate_event_results()
        self._update_video_select_screen_text()
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

        initial_path = self.current_template_path or os.path.join(os.getcwd(), "roi_template.json")
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
        if self.edit_worker is not None:
            self.edit_worker.cancel()
        if self.edit_thread is not None and self.edit_thread.isRunning():
            self.edit_thread.quit()
            self.edit_thread.wait(1500)
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
