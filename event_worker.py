from __future__ import annotations

from typing import Any, Mapping, Optional

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from event_detector import EventDetectionCancelled, analyze_roi_color_timeline, detect_events


class EventDetectionWorker(QObject):
    progress = pyqtSignal(int, str)
    result = pyqtSignal(list)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        video_path: str,
        rois_relative: Mapping[str, Any],
        sample_hz: int = 10,
        params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.rois_relative = dict(rois_relative)
        self.sample_hz = int(sample_hz)
        self.params = dict(params) if params is not None else None
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    @pyqtSlot()
    def run(self) -> None:
        try:
            events = detect_events(
                video_path=self.video_path,
                rois_relative_dict=self.rois_relative,
                sample_hz=self.sample_hz,
                params=self.params,
                progress_cb=self._emit_progress,
                cancel_cb=self._is_cancel_requested,
            )
            if not self._cancel_requested:
                self.result.emit(events)
        except EventDetectionCancelled:
            pass
        except Exception as exc:
            if not self._cancel_requested:
                self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def _is_cancel_requested(self) -> bool:
        return self._cancel_requested

    def _emit_progress(self, percent: int, message: str) -> None:
        bounded = max(0, min(100, int(percent)))
        self.progress.emit(bounded, str(message))


class RoiColorAnalysisWorker(QObject):
    progress = pyqtSignal(int, str)
    result = pyqtSignal(dict)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        video_path: str,
        roi_name: str,
        roi_relative: Mapping[str, Any],
        sample_hz: int = 10,
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.roi_name = str(roi_name)
        self.roi_relative = dict(roi_relative)
        self.sample_hz = int(sample_hz)
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    @pyqtSlot()
    def run(self) -> None:
        try:
            payload = analyze_roi_color_timeline(
                video_path=self.video_path,
                roi_name=self.roi_name,
                roi_relative=self.roi_relative,
                sample_hz=self.sample_hz,
                progress_cb=self._emit_progress,
                cancel_cb=self._is_cancel_requested,
            )
            if not self._cancel_requested:
                self.result.emit(payload)
        except EventDetectionCancelled:
            pass
        except Exception as exc:
            if not self._cancel_requested:
                self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def _is_cancel_requested(self) -> bool:
        return self._cancel_requested

    def _emit_progress(self, percent: int, message: str) -> None:
        bounded = max(0, min(100, int(percent)))
        self.progress.emit(bounded, str(message))
