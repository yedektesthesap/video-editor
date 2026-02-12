from __future__ import annotations

import os
import shutil
import subprocess
from collections import deque
from typing import Any, Mapping, Optional, Sequence

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


class VideoEditWorker(QObject):
    progress = pyqtSignal(int, str)
    result = pyqtSignal(dict)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        ffmpeg_path: str,
        input_path: str,
        output_path: str,
        cut_segments: Sequence[tuple[float, float]],
        preset: str = "medium",
        crf: int = 18,
        enable_cut: bool = True,
    ) -> None:
        super().__init__()
        self.ffmpeg_path = str(ffmpeg_path)
        self.input_path = str(input_path)
        self.output_path = str(output_path)
        self.cut_segments = [(float(start), float(end)) for start, end in cut_segments]
        self.preset = str(preset).strip() or "medium"
        self.crf = max(0, min(51, int(crf)))
        self.enable_cut = bool(enable_cut)
        self._cancel_requested = False
        self._process: Optional[subprocess.Popen[str]] = None

    def cancel(self) -> None:
        self._cancel_requested = True
        process = self._process
        if process is None:
            return
        if process.poll() is not None:
            return
        try:
            process.terminate()
        except OSError:
            pass

    @pyqtSlot()
    def run(self) -> None:
        try:
            payload = self._run_ffmpeg_edit()
            if not self._cancel_requested:
                self.result.emit(payload)
        except Exception as exc:
            if not self._cancel_requested:
                self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def _run_ffmpeg_edit(self) -> dict:
        if not self.enable_cut:
            raise RuntimeError("Bu surumde yalnizca cut islemi destekleniyor.")
        if not self.cut_segments:
            raise RuntimeError("Cut segment listesi bos.")
        if not os.path.isfile(self.input_path):
            raise RuntimeError(f"Girdi videosu bulunamadi: {self.input_path}")

        ffmpeg_binary = self._resolve_ffmpeg_binary(self.ffmpeg_path)
        if ffmpeg_binary is None:
            raise RuntimeError("FFmpeg bulunamadi.")

        has_audio = self._detect_audio_stream(ffmpeg_binary, self.input_path)
        total_duration = sum(max(0.0, end_seconds - start_seconds) for start_seconds, end_seconds in self.cut_segments)
        if total_duration <= 0.0:
            raise RuntimeError("Toplam kesim suresi sifir veya gecersiz.")

        filter_complex = self._build_filter_complex(self.cut_segments, has_audio=has_audio)
        command = [
            ffmpeg_binary,
            "-y",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "error",
            "-progress",
            "pipe:1",
            "-i",
            self.input_path,
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-c:v",
            "libx264",
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
        ]
        if has_audio:
            command.extend(["-map", "[aout]", "-c:a", "aac"])
        command.extend(["-movflags", "+faststart", self.output_path])

        self.progress.emit(0, "FFmpeg islemi baslatiliyor...")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._process = process

        last_percent = 0
        error_tail: deque[str] = deque(maxlen=30)
        try:
            if process.stdout is not None:
                for raw_line in process.stdout:
                    if self._cancel_requested:
                        self.cancel()
                        break
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line.startswith("out_time_ms="):
                        value_text = line.split("=", 1)[1].strip()
                        try:
                            out_time_ms = int(value_text)
                        except ValueError:
                            continue
                        percent = int((float(out_time_ms) / float(total_duration * 1_000_000.0)) * 100.0)
                        percent = max(1, min(99, percent))
                        if percent > last_percent:
                            last_percent = percent
                            self.progress.emit(percent, "")
                        continue
                    if line == "progress=end":
                        last_percent = 100
                        self.progress.emit(100, "FFmpeg islemi tamamlandi.")
                        continue
                    if "=" not in line:
                        error_tail.append(line)
            return_code = process.wait()
            if self._cancel_requested:
                raise RuntimeError("Video edit islemi durduruldu.")
            if return_code != 0:
                summary = "\n".join(error_tail).strip()
                if not summary:
                    summary = f"FFmpeg cikis kodu: {return_code}"
                raise RuntimeError(f"FFmpeg islemi basarisiz:\n{summary}")
        finally:
            self._process = None

        self.progress.emit(100, "Video edit tamamlandi.")
        return {
            "output_path": self.output_path,
            "segments": len(self.cut_segments),
            "has_audio": bool(has_audio),
            "duration_seconds": round(total_duration, 3),
        }

    @staticmethod
    def _resolve_ffmpeg_binary(raw_path: str) -> Optional[str]:
        candidate = str(raw_path).strip().strip('"')
        if not candidate:
            return None
        if os.path.isfile(candidate):
            return candidate
        resolved = shutil.which(candidate)
        if resolved and os.path.isfile(resolved):
            return resolved
        return None

    @staticmethod
    def _build_filter_complex(cut_segments: Sequence[tuple[float, float]], has_audio: bool) -> str:
        parts: list[str] = []
        segment_count = len(cut_segments)
        for index, (start_seconds, end_seconds) in enumerate(cut_segments):
            parts.append(
                f"[0:v]trim=start={start_seconds:.6f}:end={end_seconds:.6f},setpts=PTS-STARTPTS[v{index}]"
            )
            if has_audio:
                parts.append(
                    f"[0:a]atrim=start={start_seconds:.6f}:end={end_seconds:.6f},asetpts=PTS-STARTPTS[a{index}]"
                )

        video_concat_inputs = "".join(f"[v{index}]" for index in range(segment_count))
        parts.append(f"{video_concat_inputs}concat=n={segment_count}:v=1:a=0[vout]")
        if has_audio:
            audio_concat_inputs = "".join(f"[a{index}]" for index in range(segment_count))
            parts.append(f"{audio_concat_inputs}concat=n={segment_count}:v=0:a=1[aout]")

        return ";".join(parts)

    def _detect_audio_stream(self, ffmpeg_binary: str, input_path: str) -> bool:
        ffprobe_binary = self._resolve_ffprobe_binary(ffmpeg_binary)
        if ffprobe_binary is not None:
            command = [
                ffprobe_binary,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                input_path,
            ]
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                    timeout=8,
                )
                if completed.returncode == 0 and completed.stdout.strip():
                    return True
            except (OSError, subprocess.SubprocessError):
                pass

        command = [ffmpeg_binary, "-hide_banner", "-i", input_path]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=8,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        output_text = (completed.stdout or "") + "\n" + (completed.stderr or "")
        return "Audio:" in output_text

    @staticmethod
    def _resolve_ffprobe_binary(ffmpeg_binary: str) -> Optional[str]:
        ffmpeg_dir = os.path.dirname(ffmpeg_binary)
        if ffmpeg_dir:
            ffprobe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
            candidate = os.path.join(ffmpeg_dir, ffprobe_name)
            if os.path.isfile(candidate):
                return candidate

        resolved = shutil.which("ffprobe")
        if resolved and os.path.isfile(resolved):
            return resolved
        return None
