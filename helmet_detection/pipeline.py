"""端到端检测流程：图片、视频文件、摄像头。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .alerts import AlertDecision, AlertManager
from .association import AssociationResult, PersonHelmetAssociator
from .config import SystemConfig, config_to_dict
from .detector import YoloDetector
from .render import Renderer
from .stats import FrameStats, RunStats
from .types import ComplianceStatus
from .writers import (
    CsvStatsWriter,
    JsonlWriter,
    RunArtifacts,
    SnapshotSaver,
    safe_stem,
    write_image,
    write_json,
)

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class ProcessedFrame:
    annotated: np.ndarray
    result: AssociationResult
    stats: FrameStats
    alert: AlertDecision


def read_image(path: Path) -> np.ndarray | None:
    """支持中文路径的图片读取。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def resize_max_width(frame: np.ndarray, max_width: int | None) -> np.ndarray:
    if not max_width or frame.shape[1] <= max_width:
        return frame
    width = max(2, int(max_width) // 2 * 2)
    scale = width / frame.shape[1]
    height = max(2, int(round(frame.shape[0] * scale / 2)) * 2)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


class HelmetDetectionSystem:
    def __init__(
        self,
        config: SystemConfig,
        source: str = "",
        model_path: str | None = None,
    ) -> None:
        self.config = config
        if model_path:
            config.model.path = model_path
        aliases = {
            "person": config.classes.person,
            "helmet": config.classes.helmet,
            "no_helmet": config.classes.no_helmet,
        }
        self.detector = YoloDetector(config.model, aliases)
        self.associator = PersonHelmetAssociator(config.association_config())
        self.renderer = Renderer(config.render)
        self.source = source
        self.run_stats = RunStats(source=source)
        self.alerts = AlertManager(config.alert, source=source)
        self._show_warned = False
        # 预热一次，避免首帧推理耗时被误计入统计
        self.detector.warmup()

    # ------------------------------------------------------------------ #
    def capabilities(self) -> dict[str, bool]:
        return self.detector.capabilities()

    def validate_model(self) -> list[str]:
        caps = self.capabilities()
        warnings: list[str] = []
        if not caps.get("person"):
            warnings.append("模型未包含 Person 类别，无法建立人员-安全帽关联。")
        if not caps.get("helmet") and not caps.get("no_helmet"):
            warnings.append("模型未包含 helmet / no_helmet 类别，无法判断是否佩戴安全帽。")
        return warnings

    # ------------------------------------------------------------------ #
    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float | None = None,
        now: float | None = None,
    ) -> ProcessedFrame:
        timestamp = time.time() if timestamp is None else timestamp
        detections = self.detector.detect(frame)
        result = self.associator.analyze(detections, frame_size=(frame.shape[1], frame.shape[0]))
        decision = self.alerts.update(
            result,
            now=now,
            extra={"frame": frame_index, "source": self.source},
        )
        inference_ms = self.detector.inference_ms[-1] if self.detector.inference_ms else None
        stats = FrameStats.from_result(
            result=result,
            frame_index=frame_index,
            timestamp=timestamp,
            new_violation_events=decision.event_update.new_count if decision.event_update else 0,
            alert_triggered=decision.triggered,
            inference_ms=inference_ms,
        )
        self.run_stats.update(stats, self.alerts.tracker)

        if self.renderer.use_cjk:
            extra_lines = [
                f"累计违规事件 {self.alerts.tracker.total_events}",
                f"帧 {frame_index}",
            ]
        else:
            extra_lines = [
                f"Cumulative violation events {self.alerts.tracker.total_events}",
                f"Frame {frame_index}",
            ]
        annotated = self.renderer.draw(
            frame, result, stats,
            alert=decision.triggered,
            extra_lines=extra_lines,
        )
        return ProcessedFrame(annotated=annotated, result=result, stats=stats, alert=decision)

    # ------------------------------------------------------------------ #
    # 运行目录与产物
    # ------------------------------------------------------------------ #
    def create_run_dir(self, source: str) -> RunArtifacts:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.config.output_dir() / f"{safe_stem(source)}_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return RunArtifacts(run_dir=run_dir)

    def _summary(self, source_type: str, artifacts: RunArtifacts,
                 extra: dict[str, Any] | None = None) -> dict[str, Any]:
        self.run_stats.finish()
        summary = {
            "source": self.source,
            "source_type": source_type,
            "model": str(self.detector.model_path),
            "model_classes": self.detector.class_names,
            "capabilities": self.capabilities(),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "stats": self.run_stats.to_dict(),
            "artifacts": artifacts.to_dict(),
            "config": {
                "conf": self.config.model.conf,
                "iou": self.config.model.iou,
                "imgsz": self.config.model.imgsz,
                "device": self.detector.device,
                "association": {
                    "head_height_ratio": self.config.association.head_height_ratio,
                    "head_width_shrink": self.config.association.head_width_shrink,
                    "unmatched_policy": self.config.association.unmatched_policy,
                },
            },
        }
        summary.update(extra or {})
        return summary

    # ------------------------------------------------------------------ #
    def run_image(
        self,
        source: Path,
        output_path: Path | None = None,
        show: bool | None = None,
    ) -> dict[str, Any]:
        frame = read_image(source)
        if frame is None:
            raise RuntimeError(f"无法读取图片: {source}")

        artifacts = self.create_run_dir(str(source))
        processed = self.process_frame(frame, frame_index=0)
        artifacts.annotated_path = output_path or (artifacts.run_dir / f"{safe_stem(source)}_detected.jpg")
        write_image(artifacts.annotated_path, processed.annotated, self.config.output.jpeg_quality)

        artifacts.summary_path = artifacts.run_dir / "summary.json"
        summary = self._summary("image", artifacts, {
            "image": {
                "path": str(source),
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
            },
            "frame_result": processed.result.to_dict(),
        })
        write_json(artifacts.summary_path, summary)

        if self.config.output.save_csv:
            csv_path = artifacts.run_dir / "stats.csv"
            writer = CsvStatsWriter(csv_path)
            writer.write(processed.stats)
            writer.close()
            artifacts.csv_path = csv_path
        if self.config.output.save_json:
            jsonl_path = artifacts.run_dir / "frames.jsonl"
            jsonl = JsonlWriter(jsonl_path)
            jsonl.write({"stats": processed.stats.to_dict(), "result": processed.result.to_dict()})
            jsonl.close()
            artifacts.jsonl_path = jsonl_path

        show = self.config.output.display if show is None else show
        if show:
            self._display(processed.annotated, wait_ms=0)

        summary["artifacts"] = artifacts.to_dict()
        write_json(artifacts.summary_path, summary)
        logger.info("图片处理完成: %s", artifacts.annotated_path)
        return summary

    # ------------------------------------------------------------------ #
    def run_video(
        self,
        source: str | int,
        output_path: Path | None = None,
        show: bool | None = None,
        max_frames: int | None = None,
        record: bool | None = None,
    ) -> dict[str, Any]:
        cap = self._open_capture(source)
        source_type = "webcam" if isinstance(source, int) else "video"
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        source_label = f"camera_{source}" if isinstance(source, int) else str(source)

        artifacts = self.create_run_dir(source_label)
        show = self.config.output.display if show is None else show
        record = self.config.output.save_annotated if record is None else record

        video_writer: cv2.VideoWriter | None = None
        csv_writer: CsvStatsWriter | None = None
        jsonl_writer: JsonlWriter | None = None
        snapshot_saver: SnapshotSaver | None = None
        if self.config.alert.snapshot_on_new_event and self.config.output.save_violation_snapshots:
            snapshot_saver = SnapshotSaver(
                artifacts.run_dir / "violations",
                max_per_event=self.config.alert.max_snapshots_per_event,
                quality=self.config.output.jpeg_quality,
                source=source_label,
            )
            artifacts.snapshot_dir = snapshot_saver.directory
            snapshot_saver.directory.mkdir(parents=True, exist_ok=True)

        frame_index = 0
        last_render_time = time.perf_counter()
        render_fps = 0.0
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                if max_frames is not None and frame_index >= max_frames:
                    break
                frame = resize_max_width(frame, self.config.output.max_width)
                processed = self.process_frame(frame, frame_index, timestamp=time.time())

                if record:
                    if video_writer is None:
                        path = output_path or (artifacts.run_dir / f"{safe_stem(source_label)}_detected.mp4")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        fourcc = cv2.VideoWriter_fourcc(*self.config.output.video_codec[:4])
                        h, w = processed.annotated.shape[:2]
                        video_writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
                        if not video_writer.isOpened():
                            logger.error("无法创建输出视频: %s（尝试更换 --codec）", path)
                            video_writer = None
                            record = False
                        else:
                            artifacts.annotated_path = path

                if video_writer is not None:
                    video_writer.write(processed.annotated)
                if csv_writer is None and self.config.output.save_csv:
                    csv_writer = CsvStatsWriter(artifacts.run_dir / "stats.csv")
                    artifacts.csv_path = csv_writer.path
                if csv_writer is not None:
                    csv_writer.write(processed.stats)
                if jsonl_writer is None and self.config.output.save_json:
                    jsonl_writer = JsonlWriter(artifacts.run_dir / "frames.jsonl")
                    artifacts.jsonl_path = jsonl_writer.path
                if jsonl_writer is not None:
                    jsonl_writer.write({
                        "stats": processed.stats.to_dict(),
                        "result": processed.result.to_dict(),
                    })

                if snapshot_saver is not None and processed.alert.new_events:
                    for event in processed.alert.new_events:
                        snapshot_saver.save(frame, event.event_id, frame_index)

                now_render = time.perf_counter()
                dt = now_render - last_render_time
                last_render_time = now_render
                if dt > 0:
                    render_fps = 0.9 * render_fps + 0.1 * (1.0 / dt) if render_fps else 1.0 / dt
                if show and not self._show_warned:
                    if not self._display(processed.annotated, wait_ms=1, fps=render_fps):
                        break

                frame_index += 1
                if frame_index % 30 == 0:
                    logger.info(
                        "已处理 %d 帧 | 人员 %d | 违规 %d | 累计事件 %d | %.1f FPS",
                        frame_index, processed.stats.persons, processed.stats.violations,
                        self.alerts.tracker.total_events, render_fps,
                    )
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
            if csv_writer is not None:
                csv_writer.close()
            if jsonl_writer is not None:
                jsonl_writer.close()

        if snapshot_saver is not None and snapshot_saver.paths:
            artifacts.snapshot_index = artifacts.run_dir / "violations.json"
            snapshot_saver.write_index(artifacts.snapshot_index)

        artifacts.summary_path = artifacts.run_dir / "summary.json"
        summary = self._summary(source_type, artifacts, {
            "video": {
                "fps": fps,
                "total_frames_reported": total_frames,
                "frames_processed": frame_index,
            },
        })
        write_json(artifacts.summary_path, summary)
        logger.info("视频处理完成: %d 帧，产物目录 %s", frame_index, artifacts.run_dir)
        return summary

    # ------------------------------------------------------------------ #
    @staticmethod
    def _open_capture(source: str | int) -> cv2.VideoCapture:
        if isinstance(source, int):
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened() and Path(source).exists():
                cap = cv2.VideoCapture(str(Path(source).resolve()))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {source}")
        return cap

    def _display(self, frame: np.ndarray, wait_ms: int = 1, fps: float | None = None) -> bool:
        """返回 False 表示用户请求退出。无 GUI 环境下仅警告一次并继续。"""
        try:
            window = "Helmet Detection"
            display = frame.copy()
            if fps:
                cv2.putText(display, f"FPS {fps:.1f}", (12, display.shape[0] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(window, display)
            key = cv2.waitKey(wait_ms) & 0xFF
            return key not in (27, ord("q"))
        except cv2.error as exc:
            if not self._show_warned:
                logger.warning("当前环境无法显示窗口（可能无 GUI），已继续保存结果: %s", exc)
                self._show_warned = True
            return False

    def close(self) -> None:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
