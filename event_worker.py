from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
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
        preset: str = "slow",
        crf: int = 0,
        enable_cut: bool = True,
        enable_resize: bool = False,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        target_fps: Optional[float] = None,
        remove_audio: bool = False,
        enable_speed: bool = False,
        speed_factor: Optional[float] = None,
        enable_audio_effect: bool = False,
        audio_effect_preset: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.ffmpeg_path = str(ffmpeg_path)
        self.input_path = str(input_path)
        self.output_path = str(output_path)
        self.cut_segments = [(float(start), float(end)) for start, end in cut_segments]
        self.preset = str(preset).strip() or "slow"
        self.crf = max(0, min(51, int(crf)))
        self.enable_cut = bool(enable_cut)
        self.enable_resize = bool(enable_resize)
        self.target_width = int(target_width) if target_width is not None else None
        self.target_height = int(target_height) if target_height is not None else None
        self.target_fps = float(target_fps) if target_fps is not None else None
        self.remove_audio = bool(remove_audio)
        self.enable_speed = bool(enable_speed)
        self.speed_factor = float(speed_factor) if speed_factor is not None else None
        self.enable_audio_effect = bool(enable_audio_effect)
        self.audio_effect_preset = str(audio_effect_preset).strip().lower() if audio_effect_preset is not None else None
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
        if not self.enable_cut and not self.enable_resize and (not self.remove_audio) and (not self.enable_speed) and (not self.enable_audio_effect):
            raise RuntimeError("En az bir edit islemi secilmelidir.")
        if not os.path.isfile(self.input_path):
            raise RuntimeError(f"Girdi videosu bulunamadi: {self.input_path}")

        ffmpeg_binary = self._resolve_ffmpeg_binary(self.ffmpeg_path)
        if ffmpeg_binary is None:
            raise RuntimeError("FFmpeg bulunamadi.")

        cut_duration = sum(max(0.0, end_seconds - start_seconds) for start_seconds, end_seconds in self.cut_segments)
        if self.enable_cut:
            if not self.cut_segments:
                raise RuntimeError("Cut segment listesi bos.")
            if cut_duration <= 0.0:
                raise RuntimeError("Toplam kesim suresi sifir veya gecersiz.")

        if self.enable_resize:
            if (self.target_width is None) != (self.target_height is None):
                raise RuntimeError("Cozunurluk icin genislik/yukseklik birlikte verilmelidir.")
            if self.target_width is not None and self.target_width <= 0:
                raise RuntimeError("Hedef genislik pozitif olmali.")
            if self.target_height is not None and self.target_height <= 0:
                raise RuntimeError("Hedef yukseklik pozitif olmali.")
            if self.target_fps is not None and self.target_fps <= 0.0:
                raise RuntimeError("Hedef FPS pozitif olmali.")
            if self.target_width is None and self.target_fps is None:
                raise RuntimeError("Cozunurluk/FPS adimi icin en az bir hedef secilmelidir.")

        if self.enable_speed:
            if self.speed_factor is None or self.speed_factor <= 0.0:
                raise RuntimeError("Video hizi icin gecerli hiz degeri gereklidir.")
            if abs(self.speed_factor - 1.0) < 0.001:
                raise RuntimeError("Video hizi 1.0x olamaz.")

        if self.enable_audio_effect:
            if not self.audio_effect_preset:
                raise RuntimeError("Ses efekti secilmedi.")
            if self.audio_effect_preset == "none":
                raise RuntimeError("Ses efekti 'Yok' olamaz.")

        operation_names: list[str] = []
        if self.enable_cut:
            operation_names.append("cut")
        if self.enable_resize:
            operation_names.append("resize")
        if self.remove_audio:
            operation_names.append("audio_remove")
        if self.enable_speed:
            operation_names.append("speed")
        if self.enable_audio_effect:
            operation_names.append("audio_effect")

        input_path = self.input_path
        output_path = self.output_path
        has_audio = self._detect_audio_stream(ffmpeg_binary, input_path)
        estimated_duration = self._probe_duration_seconds(ffmpeg_binary, input_path)

        temp_files: list[str] = []
        completed_ops: list[str] = []
        try:
            step_count = len(operation_names)
            for index, operation_name in enumerate(operation_names):
                is_last_step = index == (step_count - 1)
                step_output = output_path if is_last_step else self._create_temp_output_path(temp_files)

                if operation_name == "cut":
                    command = self._build_cut_command(
                        ffmpeg_binary=ffmpeg_binary,
                        input_path=input_path,
                        output_path=step_output,
                        has_audio=has_audio,
                    )
                    step_label = "Cut"
                    step_duration = cut_duration
                elif operation_name == "resize":
                    command = self._build_resize_command(
                        ffmpeg_binary=ffmpeg_binary,
                        input_path=input_path,
                        output_path=step_output,
                        has_audio=has_audio,
                        target_width=self.target_width,
                        target_height=self.target_height,
                        target_fps=self.target_fps,
                    )
                    step_label = "Cozunurluk/FPS"
                    step_duration = estimated_duration
                elif operation_name == "audio_remove":
                    command = self._build_remove_audio_command(
                        ffmpeg_binary=ffmpeg_binary,
                        input_path=input_path,
                        output_path=step_output,
                    )
                    step_label = "Ses Silme"
                    step_duration = estimated_duration
                elif operation_name == "speed":
                    command = self._build_speed_command(
                        ffmpeg_binary=ffmpeg_binary,
                        input_path=input_path,
                        output_path=step_output,
                        has_audio=has_audio,
                        speed_factor=self.speed_factor,
                    )
                    step_label = "Video Hizi"
                    step_duration = estimated_duration
                else:
                    command = self._build_audio_effect_command(
                        ffmpeg_binary=ffmpeg_binary,
                        input_path=input_path,
                        output_path=step_output,
                        has_audio=has_audio,
                        effect_preset=self.audio_effect_preset,
                    )
                    step_label = "Ses Efekti"
                    step_duration = estimated_duration

                self._run_ffmpeg_step(
                    command=command,
                    step_index=index,
                    step_count=step_count,
                    step_label=step_label,
                    expected_duration=step_duration,
                )
                completed_ops.append(operation_name)

                input_path = step_output
                has_audio = self._detect_audio_stream(ffmpeg_binary, input_path)
                if operation_name == "cut":
                    estimated_duration = cut_duration
                elif operation_name == "speed" and self.speed_factor is not None and estimated_duration is not None:
                    estimated_duration = float(estimated_duration) / float(self.speed_factor)

            final_duration = self._probe_duration_seconds(ffmpeg_binary, self.output_path)
            if final_duration is None:
                final_duration = estimated_duration if estimated_duration is not None else 0.0
        finally:
            self._process = None
            for temp_path in temp_files:
                self._safe_remove_file(temp_path)

        self.progress.emit(100, "Video edit tamamlandi.")
        return {
            "output_path": self.output_path,
            "segments": len(self.cut_segments),
            "has_audio": bool(has_audio),
            "duration_seconds": round(float(final_duration), 3),
            "operations": completed_ops,
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

    def _build_cut_command(
        self,
        ffmpeg_binary: str,
        input_path: str,
        output_path: str,
        has_audio: bool,
    ) -> list[str]:
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
            input_path,
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
            command.extend(["-map", "[aout]", "-c:a", "alac"])
        command.extend(["-movflags", "+faststart", output_path])
        return command

    def _build_resize_command(
        self,
        ffmpeg_binary: str,
        input_path: str,
        output_path: str,
        has_audio: bool,
        target_width: Optional[int],
        target_height: Optional[int],
        target_fps: Optional[float],
    ) -> list[str]:
        filters: list[str] = []
        if target_width is not None and target_height is not None:
            filters.append(f"scale={int(target_width)}:{int(target_height)}:flags=lanczos")
        if target_fps is not None:
            filters.append(f"fps=fps={float(target_fps):.6f}")
        if not filters:
            raise RuntimeError("Cozunurluk/FPS adimi icin filtre olusturulamadi.")

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
            input_path,
            "-vf",
            ",".join(filters),
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
        ]
        if has_audio:
            command.extend(["-map", "0:a?", "-c:a", "alac"])
        command.extend(["-movflags", "+faststart", output_path])
        return command

    def _build_remove_audio_command(
        self,
        ffmpeg_binary: str,
        input_path: str,
        output_path: str,
    ) -> list[str]:
        return [
            ffmpeg_binary,
            "-y",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "error",
            "-progress",
            "pipe:1",
            "-i",
            input_path,
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            "-movflags",
            "+faststart",
            output_path,
        ]

    def _build_speed_command(
        self,
        ffmpeg_binary: str,
        input_path: str,
        output_path: str,
        has_audio: bool,
        speed_factor: Optional[float],
    ) -> list[str]:
        if speed_factor is None or speed_factor <= 0.0:
            raise RuntimeError("Video hizi icin gecersiz hiz degeri.")

        atempo_filter = self._build_atempo_filter(speed_factor)
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
            input_path,
            "-filter:v",
            f"setpts=PTS/{float(speed_factor):.6f}",
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
        ]
        if has_audio:
            command.extend(
                [
                    "-filter:a",
                    atempo_filter,
                    "-map",
                    "0:a:0",
                    "-c:a",
                    "alac",
                ]
            )
        command.extend(["-movflags", "+faststart", output_path])
        return command

    def _build_audio_effect_command(
        self,
        ffmpeg_binary: str,
        input_path: str,
        output_path: str,
        has_audio: bool,
        effect_preset: Optional[str],
    ) -> list[str]:
        effect_chain = self._audio_effect_filter_chain(effect_preset)
        if not effect_chain:
            raise RuntimeError("Ses efekti filtresi olusturulamadi.")

        common_prefix = [
            ffmpeg_binary,
            "-y",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "error",
            "-progress",
            "pipe:1",
            "-i",
            input_path,
        ]
        if has_audio:
            filter_complex = (
                f"[0:a]aformat=sample_rates=48000:channel_layouts=stereo[orig];"
                f"[0:a]{effect_chain},aformat=sample_rates=48000:channel_layouts=stereo,volume=0.55[fx];"
                f"[orig][fx]amix=inputs=2:normalize=0:dropout_transition=0[aout]"
            )
            command = common_prefix + [
                "-filter_complex",
                filter_complex,
                "-map",
                "0:v:0",
                "-map",
                "[aout]",
                "-c:v",
                "copy",
                "-c:a",
                "alac",
                "-movflags",
                "+faststart",
                output_path,
            ]
            return command

        filter_complex = (
            f"[1:a]{effect_chain},aformat=sample_rates=48000:channel_layouts=stereo,volume=0.8[aout]"
        )
        command = common_prefix + [
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:sample_rate=48000",
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "alac",
            "-shortest",
            "-movflags",
            "+faststart",
            output_path,
        ]
        return command

    @staticmethod
    def _audio_effect_filter_chain(effect_preset: Optional[str]) -> str:
        key = str(effect_preset or "").strip().lower()
        if key == "bass_boost":
            return "bass=g=8:f=110:w=0.6"
        if key == "echo":
            return "aecho=0.8:0.88:45:0.35"
        if key == "phone":
            return "highpass=f=300,lowpass=f=3200,acompressor=threshold=-19dB:ratio=3:attack=5:release=50"
        return ""

    @staticmethod
    def _build_atempo_filter(speed_factor: float) -> str:
        if speed_factor <= 0.0:
            raise RuntimeError("Atempo icin hiz degeri pozitif olmali.")

        working = float(speed_factor)
        chain: list[str] = []
        while working > 2.0:
            chain.append("atempo=2.0")
            working /= 2.0
        while working < 0.5:
            chain.append("atempo=0.5")
            working /= 0.5
        chain.append(f"atempo={working:.6f}")
        return ",".join(chain)

    def _run_ffmpeg_step(
        self,
        command: Sequence[str],
        step_index: int,
        step_count: int,
        step_label: str,
        expected_duration: Optional[float],
    ) -> None:
        start_percent = int(round((float(step_index) / float(max(1, step_count))) * 100.0))
        end_percent = int(round((float(step_index + 1) / float(max(1, step_count))) * 100.0))
        step_no = step_index + 1

        self.progress.emit(start_percent, f"{step_no}/{step_count} {step_label} basladi.")
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._process = process

        last_percent = start_percent
        duration_seconds = float(expected_duration) if expected_duration is not None else 0.0
        has_duration = duration_seconds > 0.0
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

                    if has_duration and line.startswith("out_time_ms="):
                        value_text = line.split("=", 1)[1].strip()
                        try:
                            out_time_ms = int(value_text)
                        except ValueError:
                            continue
                        ratio = float(out_time_ms) / float(duration_seconds * 1_000_000.0)
                        ratio = max(0.0, min(1.0, ratio))
                        dynamic = start_percent + int(round(ratio * float(max(0, end_percent - start_percent - 1))))
                        dynamic = max(start_percent, min(end_percent - 1, dynamic))
                        if dynamic > last_percent:
                            last_percent = dynamic
                            self.progress.emit(dynamic, "")
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

        self.progress.emit(end_percent, f"{step_no}/{step_count} {step_label} tamamlandi.")

    @staticmethod
    def _create_temp_output_path(bucket: list[str]) -> str:
        fd, temp_path = tempfile.mkstemp(prefix="video_edit_step_", suffix=".mp4")
        os.close(fd)
        bucket.append(temp_path)
        return temp_path

    def _probe_duration_seconds(self, ffmpeg_binary: str, input_path: str) -> Optional[float]:
        ffprobe_binary = self._resolve_ffprobe_binary(ffmpeg_binary)
        if ffprobe_binary is None:
            return None

        command = [
            ffprobe_binary,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
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
        except (OSError, subprocess.SubprocessError):
            return None

        if completed.returncode != 0:
            return None

        try:
            value = float((completed.stdout or "").strip())
        except ValueError:
            return None
        if value <= 0.0:
            return None
        return value

    @staticmethod
    def _safe_remove_file(path: str) -> None:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass

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
