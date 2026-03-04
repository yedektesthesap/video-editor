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
        text_overlays: Optional[Sequence[dict]] = None,
        image_overlays: Optional[Sequence[dict]] = None,
        external_audio_tracks: Optional[Sequence[dict]] = None,
        external_audio_mode: str = "mix",
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
        self.text_overlays = [dict(item) for item in (text_overlays or []) if isinstance(item, Mapping)]
        self.image_overlays = [dict(item) for item in (image_overlays or []) if isinstance(item, Mapping)]
        self.external_audio_tracks = [dict(item) for item in (external_audio_tracks or []) if isinstance(item, Mapping)]
        self.external_audio_mode = str(external_audio_mode).strip().lower() or "mix"
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
        has_visual_overlay = bool(self.text_overlays) or bool(self.image_overlays)
        has_external_audio = bool(self.external_audio_tracks)
        if (
            (not has_visual_overlay)
            and (not self.enable_cut)
            and (not self.enable_resize)
            and (not self.remove_audio)
            and (not self.enable_speed)
            and (not self.enable_audio_effect)
            and (not has_external_audio)
        ):
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

        for index, image_item in enumerate(self.image_overlays):
            image_path = str(image_item.get("path", "")).strip()
            if not image_path:
                raise RuntimeError(f"PNG katmani {index + 1} icin dosya yolu bos.")
            if not os.path.isfile(image_path):
                raise RuntimeError(f"PNG katmani {index + 1} icin dosya bulunamadi: {image_path}")

        if has_external_audio:
            if self.external_audio_mode != "mix":
                raise RuntimeError("Harici ses modu yalnizca 'mix' olabilir.")
            for index, track_item in enumerate(self.external_audio_tracks):
                track_path = str(track_item.get("path", "")).strip()
                if not track_path:
                    raise RuntimeError(f"Harici ses {index + 1} icin dosya yolu bos.")
                if not os.path.isfile(track_path):
                    raise RuntimeError(f"Harici ses {index + 1} dosyasi bulunamadi: {track_path}")

        operation_names: list[str] = []
        if has_visual_overlay:
            operation_names.append("visual_overlay")
        if self.remove_audio:
            operation_names.append("audio_remove")
        if has_external_audio:
            operation_names.append("external_audio")
        if self.enable_cut:
            operation_names.append("cut")
        if self.enable_resize:
            operation_names.append("resize")
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

                if operation_name == "visual_overlay":
                    command = self._build_visual_overlay_command(
                        ffmpeg_binary=ffmpeg_binary,
                        input_path=input_path,
                        output_path=step_output,
                        has_audio=has_audio,
                        text_overlays=self.text_overlays,
                        image_overlays=self.image_overlays,
                    )
                    step_label = "Yazi/PNG Katmanlari"
                    step_duration = estimated_duration
                elif operation_name == "audio_remove":
                    command = self._build_remove_audio_command(
                        ffmpeg_binary=ffmpeg_binary,
                        input_path=input_path,
                        output_path=step_output,
                    )
                    step_label = "Ses Silme"
                    step_duration = estimated_duration
                elif operation_name == "external_audio":
                    command = self._build_external_audio_mix_command(
                        ffmpeg_binary=ffmpeg_binary,
                        input_path=input_path,
                        output_path=step_output,
                        has_audio=has_audio,
                        external_audio_tracks=self.external_audio_tracks,
                    )
                    step_label = "Harici Ses Mix"
                    step_duration = estimated_duration
                elif operation_name == "cut":
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
        command.extend(["-shortest", "-movflags", "+faststart", output_path])
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

    def _build_visual_overlay_command(
        self,
        ffmpeg_binary: str,
        input_path: str,
        output_path: str,
        has_audio: bool,
        text_overlays: Sequence[Mapping[str, Any]],
        image_overlays: Sequence[Mapping[str, Any]],
    ) -> list[str]:
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
        ]
        for image_item in image_overlays:
            image_path = str(image_item.get("path", "")).strip()
            command.extend(["-loop", "1", "-i", image_path])

        filter_parts: list[str] = []
        current_video = "[0:v]"
        image_input_index = 1

        for image_index, image_item in enumerate(image_overlays):
            start_seconds = float(image_item.get("start", 0.0))
            end_seconds = float(image_item.get("end", 0.0))
            x_value = float(image_item.get("x", 0.0))
            y_value = float(image_item.get("y", 0.0))
            width_value = image_item.get("width")
            height_value = image_item.get("height")

            source_label = f"[{image_input_index}:v]"
            prepared_label = f"[img{image_index}]"
            try:
                target_width = max(1, int(float(width_value))) if width_value is not None else None
            except (TypeError, ValueError):
                target_width = None
            try:
                target_height = max(1, int(float(height_value))) if height_value is not None else None
            except (TypeError, ValueError):
                target_height = None

            if target_width is not None:
                filter_parts.append(f"{source_label}scale={target_width}:-1:flags=lanczos,format=rgba{prepared_label}")
            elif target_height is not None:
                filter_parts.append(f"{source_label}scale=-1:{target_height}:flags=lanczos,format=rgba{prepared_label}")
            else:
                filter_parts.append(f"{source_label}format=rgba{prepared_label}")

            out_label = f"[vimg{image_index}]"
            filter_parts.append(
                f"{current_video}{prepared_label}"
                f"overlay=x=main_w*{x_value:.6f}:y=main_h*{y_value:.6f}:"
                f"enable='between(t,{start_seconds:.6f},{end_seconds:.6f})'{out_label}"
            )
            current_video = out_label
            image_input_index += 1

        for text_index, text_item in enumerate(text_overlays):
            text_value = self._escape_drawtext_text(str(text_item.get("text", "")))
            start_seconds = float(text_item.get("start", 0.0))
            end_seconds = float(text_item.get("end", 0.0))
            x_value = float(text_item.get("x", 0.0))
            y_value = float(text_item.get("y", 0.0))
            font_size = int(text_item.get("font_size", 24))
            color = str(text_item.get("color", "FFFFFF")).strip().lstrip("#") or "FFFFFF"
            bold_enabled = bool(text_item.get("bold", False))
            italic_enabled = bool(text_item.get("italic", False))
            fontfile_option = ""
            if bold_enabled or italic_enabled:
                style_fontfile = self._resolve_drawtext_fontfile(bold_enabled=bold_enabled, italic_enabled=italic_enabled)
                if style_fontfile:
                    escaped_fontfile = self._escape_drawtext_text(style_fontfile)
                    fontfile_option = f"fontfile='{escaped_fontfile}':"
            out_label = f"[vtxt{text_index}]"
            filter_parts.append(
                f"{current_video}drawtext=text='{text_value}':"
                f"expansion=none:"
                f"{fontfile_option}"
                f"x=(w-text_w)*{x_value:.6f}:y=(h-text_h)*{y_value:.6f}:"
                f"fontsize={font_size}:fontcolor={color}:"
                f"enable='between(t,{start_seconds:.6f},{end_seconds:.6f})'{out_label}"
            )
            current_video = out_label

        if not filter_parts:
            raise RuntimeError("Yazi/PNG katmani aktif ancak gecerli filtre olusturulamadi.")

        command.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                current_video,
                "-c:v",
                "libx264",
                "-preset",
                self.preset,
                "-crf",
                str(self.crf),
            ]
        )
        if has_audio:
            command.extend(["-map", "0:a?", "-c:a", "alac"])
        command.extend(["-movflags", "+faststart", output_path])
        return command

    def _build_external_audio_mix_command(
        self,
        ffmpeg_binary: str,
        input_path: str,
        output_path: str,
        has_audio: bool,
        external_audio_tracks: Sequence[Mapping[str, Any]],
    ) -> list[str]:
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
        ]
        for track_item in external_audio_tracks:
            command.extend(["-i", str(track_item.get("path", "")).strip()])

        filter_parts: list[str] = []
        audio_inputs: list[str] = []
        if has_audio:
            filter_parts.append("[0:a]aformat=sample_rates=48000:channel_layouts=stereo[aorig]")
            audio_inputs.append("[aorig]")

        for track_index, track_item in enumerate(external_audio_tracks, start=1):
            start_seconds = float(track_item.get("start", 0.0))
            end_seconds_raw = track_item.get("end")
            duration_expr = "atrim=0"
            if end_seconds_raw is not None:
                end_seconds = float(end_seconds_raw)
                duration = max(0.0, end_seconds - start_seconds)
                duration_expr = f"atrim=0:{duration:.6f}"
            delay_ms = max(0, int(round(start_seconds * 1000.0)))
            volume_value = float(track_item.get("volume", 1.0))
            out_label = f"[aext{track_index}]"
            filter_parts.append(
                f"[{track_index}:a]{duration_expr},asetpts=PTS-STARTPTS,"
                f"adelay={delay_ms}|{delay_ms},volume={volume_value:.6f},"
                f"aformat=sample_rates=48000:channel_layouts=stereo{out_label}"
            )
            audio_inputs.append(out_label)

        if not audio_inputs:
            raise RuntimeError("Harici ses adimi icin ses girisi bulunamadi.")

        if len(audio_inputs) == 1:
            filter_parts.append(f"{audio_inputs[0]}alimiter=limit=0.95[aout]")
        else:
            mix_inputs = "".join(audio_inputs)
            filter_parts.append(
                f"{mix_inputs}amix=inputs={len(audio_inputs)}:normalize=0:dropout_transition=0,"
                f"alimiter=limit=0.95[aout]"
            )

        command.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
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
        )
        return command

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
    def _resolve_drawtext_fontfile(bold_enabled: bool, italic_enabled: bool) -> Optional[str]:
        style_key = (bool(bold_enabled), bool(italic_enabled))
        candidates_by_style: dict[tuple[bool, bool], tuple[str, ...]] = {
            (False, True): (
                r"C:\Windows\Fonts\ariali.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans-Oblique.ttf",
            ),
            (True, False): (
                r"C:\Windows\Fonts\arialbd.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            ),
            (True, True): (
                r"C:\Windows\Fonts\arialbi.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans-BoldOblique.ttf",
            ),
        }
        for candidate in candidates_by_style.get(style_key, ()):
            expanded = os.path.expandvars(os.path.expanduser(candidate))
            if os.path.isfile(expanded):
                return expanded
        return None

    @staticmethod
    def _escape_drawtext_text(raw_text: str) -> str:
        escaped = str(raw_text)
        replacements = (
            ("\\", r"\\"),
            ("'", r"\'"),
            (":", r"\:"),
            (",", r"\,"),
            ("[", r"\["),
            ("]", r"\]"),
        )
        for source, target in replacements:
            escaped = escaped.replace(source, target)
        return escaped

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

    @staticmethod
    def _subprocess_no_console_kwargs() -> dict[str, Any]:
        if os.name != "nt":
            return {}

        kwargs: dict[str, Any] = {}
        create_no_window = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if create_no_window:
            kwargs["creationflags"] = create_no_window

        startupinfo_factory = getattr(subprocess, "STARTUPINFO", None)
        if startupinfo_factory is not None:
            startupinfo = startupinfo_factory()
            use_show_window = int(getattr(subprocess, "STARTF_USESHOWWINDOW", 0))
            if use_show_window:
                startupinfo.dwFlags |= use_show_window
            startupinfo.wShowWindow = int(getattr(subprocess, "SW_HIDE", 0))
            kwargs["startupinfo"] = startupinfo
        return kwargs

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
        subprocess_kwargs = self._subprocess_no_console_kwargs()
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **subprocess_kwargs,
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
            subprocess_kwargs = self._subprocess_no_console_kwargs()
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=8,
                **subprocess_kwargs,
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
                subprocess_kwargs = self._subprocess_no_console_kwargs()
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                    timeout=8,
                    **subprocess_kwargs,
                )
                if completed.returncode == 0 and completed.stdout.strip():
                    return True
            except (OSError, subprocess.SubprocessError):
                pass

        command = [ffmpeg_binary, "-hide_banner", "-i", input_path]
        try:
            subprocess_kwargs = self._subprocess_no_console_kwargs()
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=8,
                **subprocess_kwargs,
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
