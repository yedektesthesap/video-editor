
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
from PyQt6.QtCore import QPointF, QRectF, QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 1200
MIN_RELATIVE_SIZE = 0.002
MAX_SHORTCUT_ROIS = 9
TEMPLATE_VERSION = "1.1"

SETTINGS_ORGANIZATION = "YSN"
SETTINGS_APPLICATION = "VideoEditorROI"
SETTINGS_LAST_TEMPLATE_PATH = "last_template_path"


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
    mean_rgb: Tuple[int, int, int]
    hex_color: str
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
        self.color_label = QLabel("Mean RGB / HEX: -")

        self.preview_label = QLabel("No Preview")
        self.preview_label.setFixedSize(self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid #666; background: #111; color: #aaa;")

        layout.addWidget(self.pixel_label)
        layout.addWidget(self.relative_label)
        layout.addWidget(self.color_label)
        layout.addWidget(self.preview_label)

    def set_roi_name(self, roi_name: Optional[str]) -> None:
        if roi_name:
            self.setTitle(f"Selected ROI: {roi_name}")
        else:
            self.setTitle("Selected ROI: None")

    def set_empty(self, relative_rect: Optional[RelativeRect]) -> None:
        self.pixel_label.setText("Pixel: -")
        self.relative_label.setText(f"Relative: {format_relative_rect(relative_rect)}")
        self.color_label.setText("Mean RGB / HEX: -")
        self.preview_label.clear()
        self.preview_label.setText("No Preview")

    def set_stats(self, stats: RoiStats) -> None:
        self.pixel_label.setText(f"Pixel: {format_pixel_rect(stats.pixel_rect)}")
        self.relative_label.setText(f"Relative: {format_relative_rect(stats.relative_rect)}")
        r, g, b = stats.mean_rgb
        self.color_label.setText(f"Mean RGB / HEX: RGB({r}, {g}, {b}) {stats.hex_color}")

        scaled = stats.preview_pixmap.scaled(
            self.PREVIEW_WIDTH,
            self.PREVIEW_HEIGHT,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ROI Template Editor")
        self.resize(1360, 860)

        self.settings = QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)

        self.original_bgr: Optional[np.ndarray] = None
        self.video_meta: Optional[VideoMeta] = None

        self.rois_rel: Dict[str, RelativeRect] = {}
        self.roi_order: list[str] = []
        self.active_roi_name: Optional[str] = None

        self.pending_template_rois: Optional[Dict[str, RelativeRect]] = None
        self.pending_template_order: Optional[list[str]] = None

        self.current_template_path: Optional[str] = None
        self.shortcuts: list[QShortcut] = []
        self._syncing_roi_list = False

        self.canvas = VideoCanvas(self)
        self.canvas.roi_drawn.connect(self.on_roi_drawn)
        self.canvas.draw_requested_without_selection.connect(self.on_draw_without_selection)

        self._build_ui()
        self._setup_shortcuts()
        self._try_autoload_last_template()
        self.update_selected_roi_panel()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
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

    def _setup_shortcuts(self) -> None:
        for index in range(1, MAX_SHORTCUT_ROIS + 1):
            shortcut = QShortcut(QKeySequence(str(index)), self)
            shortcut.activated.connect(lambda roi_index=index - 1: self.select_roi_by_index(roi_index))
            self.shortcuts.append(shortcut)

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
        self.statusBar().showMessage(f"Deleted ROI: {current_name}", 2200)

    def on_draw_without_selection(self) -> None:
        self.statusBar().showMessage("Add or select an ROI before drawing.", 2600)

    def open_video_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "",
            "MP4 Files (*.mp4);;Video Files (*.mp4 *.mov *.mkv *.avi);;All Files (*.*)",
        )
        if not path:
            return
        self.load_video(path)

    def load_video(self, path: str) -> None:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            QMessageBox.warning(self, "Open Video", "Failed to open video file.")
            return

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
            return

        self.original_bgr = frame
        self.video_meta = VideoMeta(
            width=width if width > 0 else frame.shape[1],
            height=height if height > 0 else frame.shape[0],
            fps=fps,
            frame_index=frame_index,
            source_video=path,
        )

        display_rgb = self._build_display_rgb(frame)
        display_pixmap = numpy_rgb_to_qpixmap(display_rgb)
        self.canvas.set_image_pixmap(display_pixmap)

        if self.pending_template_rois is not None:
            pending_order = self.pending_template_order or list(self.pending_template_rois.keys())
            self._apply_roi_state(self.pending_template_rois, pending_order)
            self.pending_template_rois = None
            self.pending_template_order = None
        else:
            self.canvas.set_rois(self.rois_rel)
            self.set_active_roi_name(self.active_roi_name, announce=False)

        self.statusBar().showMessage(f"Video loaded: {os.path.basename(path)}", 3000)

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

        mean_bgr = crop.mean(axis=(0, 1))
        blue = int(np.clip(round(float(mean_bgr[0])), 0, 255))
        green = int(np.clip(round(float(mean_bgr[1])), 0, 255))
        red = int(np.clip(round(float(mean_bgr[2])), 0, 255))
        hex_color = f"#{red:02X}{green:02X}{blue:02X}"

        preview_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        preview_pixmap = numpy_rgb_to_qpixmap(preview_rgb)

        return RoiStats(
            pixel_rect=pixel_rect,
            relative_rect=rect,
            mean_rgb=(red, green, blue),
            hex_color=hex_color,
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
        self.statusBar().showMessage("ROIs reset", 2000)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
