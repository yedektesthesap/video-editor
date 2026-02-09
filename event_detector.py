from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import cv2
import numpy as np

EVENT_DEFINITIONS = [
    {"id": 1, "name": "sol1_to_mix1", "source_roi": "sol1", "target_roi": "mix1", "type": "pour"},
    {"id": 2, "name": "sol2_to_mix2", "source_roi": "sol2", "target_roi": "mix2", "type": "pour"},
    {"id": 3, "name": "sol3_to_mix3", "source_roi": "sol3", "target_roi": "mix3", "type": "pour"},
    {"id": 4, "name": "sol4_to_mix4", "source_roi": "sol4", "target_roi": "mix4", "type": "pour"},
    {"id": 5, "name": "sag4_to_mix4_mix", "source_roi": "sag4", "target_roi": "mix4", "type": "pour_mix"},
    {"id": 6, "name": "sag3_to_mix3_mix", "source_roi": "sag3", "target_roi": "mix3", "type": "pour_mix"},
    {"id": 7, "name": "sag2_to_mix2_mix", "source_roi": "sag2", "target_roi": "mix2", "type": "pour_mix"},
    {"id": 8, "name": "sag1_to_mix1_mix", "source_roi": "sag1", "target_roi": "mix1", "type": "pour_mix"},
]

REQUIRED_TARGET_ROIS = ("mix1", "mix2", "mix3", "mix4")


class EventDetectionError(RuntimeError):
    pass


class EventDetectionCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class DetectorParams:
    sample_hz: int = 10
    baseline_sec: float = 1.0
    confirm_sec: float = 1.5
    quiet_pour_sec: float = 0.35
    quiet_mix_sec: float = 0.8
    stable_window_sec: float = 0.5
    color_stable_max: float = 1.8
    motion_k: float = 6.0
    color_k: float = 6.0
    motion_min: float = 2.0
    color_min: float = 2.5


ProgressCallback = Callable[[int, str], None]
CancelCallback = Callable[[], bool]


def detect_events(
    video_path: str,
    rois_relative_dict: Mapping[str, Any],
    sample_hz: int = 10,
    params: Optional[Mapping[str, float]] = None,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_cb: Optional[CancelCallback] = None,
) -> list[dict]:
    cfg = _merge_params(sample_hz=sample_hz, override=params)

    missing_targets = [name for name in REQUIRED_TARGET_ROIS if name not in rois_relative_dict]
    if missing_targets:
        joined = ", ".join(missing_targets)
        raise EventDetectionError(f"Eksik hedef ROI: {joined}")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise EventDetectionError(f"Video acilamadi: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0:
            fps = float(cfg.sample_hz)

        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_step = max(1, int(round(fps / float(cfg.sample_hz))))

        if progress_cb is not None:
            progress_cb(1, "Video ornekleniyor...")

        roi_pixels: dict[str, tuple[int, int, int, int]] = {}
        for target_name in REQUIRED_TARGET_ROIS:
            rel_rect = _coerce_relative_rect(rois_relative_dict[target_name])
            if rel_rect is None:
                raise EventDetectionError(f"Gecersiz ROI: {target_name}")
            pixel_rect = _relative_to_pixel_rect(rel_rect, frame_width, frame_height)
            if pixel_rect is None:
                raise EventDetectionError(f"ROI goruntu boyutuna sigmiyor: {target_name}")
            roi_pixels[target_name] = pixel_rect

        timeline = _sample_video_features(
            capture=capture,
            fps=fps,
            frame_step=frame_step,
            frame_count=frame_count,
            roi_pixels=roi_pixels,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )
    finally:
        capture.release()

    if timeline["times"].size == 0:
        raise EventDetectionError("Videodan ornek frame cikartilamadi.")

    motion_thresholds: dict[str, float] = {}
    color_thresholds: dict[str, float] = {}
    baseline_count = max(3, int(math.ceil(cfg.baseline_sec * float(cfg.sample_hz))))
    for target_name in REQUIRED_TARGET_ROIS:
        motion_values = timeline["motion"][target_name]
        color_values = timeline["color_delta"][target_name]
        motion_thresholds[target_name] = _robust_threshold(
            motion_values[:baseline_count], min_limit=cfg.motion_min, k=cfg.motion_k
        )
        color_thresholds[target_name] = _robust_threshold(
            color_values[:baseline_count], min_limit=cfg.color_min, k=cfg.color_k
        )

    if progress_cb is not None:
        progress_cb(60, "Olay tespiti basladi")

    events: list[dict] = []
    search_start_idx = 0
    sample_count = int(timeline["times"].size)
    confirm_window = max(1, int(math.ceil(cfg.confirm_sec * float(cfg.sample_hz))))
    quiet_pour_count = max(1, int(math.ceil(cfg.quiet_pour_sec * float(cfg.sample_hz))))
    quiet_mix_count = max(1, int(math.ceil(cfg.quiet_mix_sec * float(cfg.sample_hz))))
    stable_window_count = max(2, int(math.ceil(cfg.stable_window_sec * float(cfg.sample_hz))))

    for event_pos, event_info in enumerate(EVENT_DEFINITIONS):
        _check_cancel(cancel_cb)
        target_roi = str(event_info["target_roi"])
        event_type = str(event_info["type"])
        motion_series = timeline["motion"][target_roi]
        color_series = timeline["color_delta"][target_roi]
        color_means = timeline["color_mean"][target_roi]
        motion_threshold = motion_thresholds[target_roi]
        color_threshold = color_thresholds[target_roi]

        found_payload: Optional[dict] = None
        idx = search_start_idx
        while idx < sample_count:
            _check_cancel(cancel_cb)
            if motion_series[idx] <= motion_threshold:
                idx += 1
                continue

            start_idx = idx
            confirm_until = min(sample_count - 1, start_idx + confirm_window)
            confirm_idx = -1
            for probe in range(start_idx, confirm_until + 1):
                if color_series[probe] > color_threshold:
                    confirm_idx = probe
                    break

            if confirm_idx < 0:
                idx = start_idx + 1
                continue

            required_quiet = quiet_pour_count if event_type == "pour" else quiet_mix_count
            quiet_counter = 0
            end_idx = -1
            stable_span = 0.0

            for probe in range(confirm_idx, sample_count):
                _check_cancel(cancel_cb)
                if motion_series[probe] <= motion_threshold:
                    quiet_counter += 1
                else:
                    quiet_counter = 0

                if quiet_counter < required_quiet:
                    continue

                candidate_end = probe
                if event_type == "pour_mix":
                    stable_span = _color_span(color_means, candidate_end, stable_window_count)
                    if stable_span > cfg.color_stable_max:
                        continue

                end_idx = candidate_end
                break

            if end_idx < 0:
                idx = start_idx + 1
                continue

            motion_peak = float(np.max(motion_series[start_idx : end_idx + 1]))
            color_peak = float(np.max(color_series[start_idx : end_idx + 1]))
            confidence = _build_confidence(
                event_type=event_type,
                motion_peak=motion_peak,
                color_peak=color_peak,
                motion_threshold=motion_threshold,
                color_threshold=color_threshold,
                color_span=stable_span,
                color_stable_max=cfg.color_stable_max,
            )

            found_payload = {
                "id": int(event_info["id"]),
                "name": str(event_info["name"]),
                "target_roi": target_roi,
                "type": event_type,
                "start": round(float(timeline["times"][start_idx]), 2),
                "end": round(float(timeline["times"][end_idx]), 2),
                "confidence": round(float(confidence), 2),
            }

            search_start_idx = end_idx + 1
            break

        if found_payload is None:
            found_payload = {
                "id": int(event_info["id"]),
                "name": str(event_info["name"]),
                "target_roi": target_roi,
                "type": event_type,
                "start": None,
                "end": None,
                "confidence": 0.0,
            }
            if progress_cb is not None:
                pct = 60 + int(((event_pos + 1) / len(EVENT_DEFINITIONS)) * 40)
                progress_cb(min(99, pct), f"Event {event_info['id']} bulunamadi")
        else:
            if progress_cb is not None:
                pct = 60 + int(((event_pos + 1) / len(EVENT_DEFINITIONS)) * 40)
                progress_cb(min(99, pct), f"Event {event_info['id']} bulundu")

        events.append(found_payload)

    if progress_cb is not None:
        progress_cb(100, "Olay tespiti tamamlandi")

    return events


def _sample_video_features(
    capture: cv2.VideoCapture,
    fps: float,
    frame_step: int,
    frame_count: int,
    roi_pixels: Mapping[str, tuple[int, int, int, int]],
    progress_cb: Optional[ProgressCallback],
    cancel_cb: Optional[CancelCallback],
) -> dict:
    times: list[float] = []
    motion: dict[str, list[float]] = {name: [] for name in roi_pixels}
    color_delta: dict[str, list[float]] = {name: [] for name in roi_pixels}
    color_mean: dict[str, list[np.ndarray]] = {name: [] for name in roi_pixels}

    prev_gray: dict[str, Optional[np.ndarray]] = {name: None for name in roi_pixels}
    prev_color: dict[str, Optional[np.ndarray]] = {name: None for name in roi_pixels}

    current_frame_index = 0
    last_progress = -1
    reached_end = False
    while not reached_end:
        _check_cancel(cancel_cb)
        ok, frame = capture.read()
        if not ok or frame is None:
            break

        frame_h, frame_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        t = float(current_frame_index) / float(fps) if fps > 0.0 else float(len(times))
        times.append(t)

        for roi_name, (x, y, w, h) in roi_pixels.items():
            x1 = min(frame_w, x + w)
            y1 = min(frame_h, y + h)
            if x < 0 or y < 0 or x1 <= x or y1 <= y:
                motion[roi_name].append(0.0)
                color_delta[roi_name].append(0.0)
                color_mean[roi_name].append(np.zeros((3,), dtype=np.float64))
                prev_gray[roi_name] = None
                prev_color[roi_name] = None
                continue

            crop_gray = gray[y:y1, x:x1]
            crop_lab = lab[y:y1, x:x1]
            if crop_gray.size == 0 or crop_lab.size == 0:
                motion[roi_name].append(0.0)
                color_delta[roi_name].append(0.0)
                color_mean[roi_name].append(np.zeros((3,), dtype=np.float64))
                prev_gray[roi_name] = None
                prev_color[roi_name] = None
                continue

            mean_color = crop_lab.reshape(-1, 3).mean(axis=0).astype(np.float64)
            previous_gray = prev_gray[roi_name]
            previous_color = prev_color[roi_name]

            if previous_gray is None or previous_gray.shape != crop_gray.shape:
                motion_value = 0.0
            else:
                motion_value = float(np.mean(cv2.absdiff(crop_gray, previous_gray)))

            if previous_color is None:
                color_value = 0.0
            else:
                color_value = float(np.linalg.norm(mean_color - previous_color))

            motion[roi_name].append(motion_value)
            color_delta[roi_name].append(color_value)
            color_mean[roi_name].append(mean_color)

            prev_gray[roi_name] = crop_gray.copy()
            prev_color[roi_name] = mean_color

        if progress_cb is not None and frame_count > 0:
            pct = int((current_frame_index / float(frame_count)) * 55.0)
            pct = max(1, min(55, pct))
            if pct != last_progress and pct % 5 == 0:
                last_progress = pct
                progress_cb(pct, "")

        # Sample frame'leri disindakiler decode edilmeden atlanir.
        for _ in range(frame_step - 1):
            _check_cancel(cancel_cb)
            if not capture.grab():
                reached_end = True
                break
            current_frame_index += 1

        current_frame_index += 1

    return {
        "times": np.asarray(times, dtype=np.float64),
        "motion": {name: np.asarray(values, dtype=np.float64) for name, values in motion.items()},
        "color_delta": {name: np.asarray(values, dtype=np.float64) for name, values in color_delta.items()},
        "color_mean": color_mean,
    }


def _coerce_relative_rect(raw: Any) -> Optional[tuple[float, float, float, float]]:
    x: float
    y: float
    w: float
    h: float
    if isinstance(raw, Mapping):
        try:
            x = float(raw["x"])
            y = float(raw["y"])
            w = float(raw["w"])
            h = float(raw["h"])
        except (KeyError, TypeError, ValueError):
            return None
    else:
        try:
            x = float(getattr(raw, "x"))
            y = float(getattr(raw, "y"))
            w = float(getattr(raw, "w"))
            h = float(getattr(raw, "h"))
        except (AttributeError, TypeError, ValueError):
            return None

    x0 = max(0.0, min(1.0, min(x, x + w)))
    y0 = max(0.0, min(1.0, min(y, y + h)))
    x1 = max(0.0, min(1.0, max(x, x + w)))
    y1 = max(0.0, min(1.0, max(y, y + h)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def _relative_to_pixel_rect(
    rel_rect: tuple[float, float, float, float], width: int, height: int
) -> Optional[tuple[int, int, int, int]]:
    if width <= 0 or height <= 0:
        return None

    x, y, w, h = rel_rect
    x0 = int(math.floor(max(0.0, min(1.0, x)) * width))
    y0 = int(math.floor(max(0.0, min(1.0, y)) * height))
    x1 = int(math.ceil(max(0.0, min(1.0, x + w)) * width))
    y1 = int(math.ceil(max(0.0, min(1.0, y + h)) * height))

    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    rect_w = x1 - x0
    rect_h = y1 - y0
    if rect_w <= 0 or rect_h <= 0:
        return None
    return x0, y0, rect_w, rect_h


def _robust_threshold(values: np.ndarray, min_limit: float, k: float) -> float:
    if values.size == 0:
        return float(min_limit)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    robust_sigma = 1.4826 * mad
    return float(max(min_limit, med + k * robust_sigma))


def _build_confidence(
    event_type: str,
    motion_peak: float,
    color_peak: float,
    motion_threshold: float,
    color_threshold: float,
    color_span: float,
    color_stable_max: float,
) -> float:
    motion_score = _clip01(((motion_peak / max(1e-6, motion_threshold)) - 1.0) / 2.0)
    color_score = _clip01(((color_peak / max(1e-6, color_threshold)) - 1.0) / 2.0)
    stability_score = _clip01(1.0 - (color_span / max(1e-6, 2.0 * color_stable_max)))
    if event_type == "pour_mix":
        return 0.45 * motion_score + 0.35 * color_score + 0.20 * stability_score
    return 0.60 * motion_score + 0.40 * color_score


def _color_span(color_mean: list[np.ndarray], end_idx: int, window_count: int) -> float:
    if not color_mean:
        return 0.0
    start_idx = max(0, end_idx - window_count + 1)
    window = color_mean[start_idx : end_idx + 1]
    if len(window) < 2:
        return 0.0

    max_distance = 0.0
    for idx in range(len(window)):
        for jdx in range(idx + 1, len(window)):
            distance = float(np.linalg.norm(window[idx] - window[jdx]))
            if distance > max_distance:
                max_distance = distance
    return max_distance


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _merge_params(sample_hz: int, override: Optional[Mapping[str, float]]) -> DetectorParams:
    cfg = DetectorParams(sample_hz=max(5, min(12, int(sample_hz))))
    if override is None:
        return cfg

    values = {
        "sample_hz": cfg.sample_hz,
        "baseline_sec": cfg.baseline_sec,
        "confirm_sec": cfg.confirm_sec,
        "quiet_pour_sec": cfg.quiet_pour_sec,
        "quiet_mix_sec": cfg.quiet_mix_sec,
        "stable_window_sec": cfg.stable_window_sec,
        "color_stable_max": cfg.color_stable_max,
        "motion_k": cfg.motion_k,
        "color_k": cfg.color_k,
        "motion_min": cfg.motion_min,
        "color_min": cfg.color_min,
    }
    for key in values:
        if key == "sample_hz":
            continue
        if key in override:
            try:
                values[key] = float(override[key])
            except (TypeError, ValueError):
                continue
    return DetectorParams(**values)


def _check_cancel(cancel_cb: Optional[CancelCallback]) -> None:
    if cancel_cb is not None and bool(cancel_cb()):
        raise EventDetectionCancelled("Olay tespiti iptal edildi.")
