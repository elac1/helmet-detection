"""Ultralytics YOLO 检测器封装。"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .association import normalize_class_name
from .config import ModelConfig, SystemConfig
from .types import Detection


class ModelLoadError(RuntimeError):
    pass


class YoloDetector:
    def __init__(self, config: ModelConfig, class_aliases: dict[str, list[str]] | None = None) -> None:
        self.config = config
        self.model_path = self._resolve_path(config.path)
        if not self.model_path.exists():
            raise ModelLoadError(
                f"模型文件不存在: {self.model_path}\n"
                "请先运行训练脚本生成权重，或通过 --model 指定已有 .pt 文件。\n"
                "提示：COCO 预训练的 yolov8n.pt 不含安全帽类别，不能直接用于合规判断。"
            )
        self.model = YOLO(str(self.model_path))
        self.names: dict[int, str] = {int(k): str(v) for k, v in self.model.names.items()}
        self.device = self._resolve_device(config.device)
        self.half = bool(config.half and self.device != "cpu")
        self._aliases = class_aliases or {}
        self.inference_ms: list[float] = []

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_path(value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        if path.exists():
            return path.resolve()
        # 常见本地预训练模型位置（无网络环境下的兜底）
        fallback = Path.home() / "yolo" / path.name
        if fallback.exists() and path.parent == Path("."):
            return fallback
        return Path(__file__).resolve().parent.parent / path

    @staticmethod
    def _resolve_device(value: str) -> str:
        if value and value != "auto":
            return value
        try:
            import torch
            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"

    # ------------------------------------------------------------------ #
    @property
    def class_names(self) -> list[str]:
        return [self.names[i] for i in sorted(self.names)]

    def normalized_names(self) -> set[str]:
        return {normalize_class_name(n) for n in self.names.values()}

    def capabilities(self) -> dict[str, bool]:
        names = self.normalized_names()
        return {
            "person": any(n in names for n in map(normalize_class_name, self._aliases.get("person", []))),
            "helmet": any(n in names for n in map(normalize_class_name, self._aliases.get("helmet", []))),
            "no_helmet": any(n in names for n in map(normalize_class_name, self._aliases.get("no_helmet", []))),
        }

    # ------------------------------------------------------------------ #
    def warmup(self) -> None:
        dummy = np.zeros((self.config.imgsz, self.config.imgsz, 3), dtype=np.uint8)
        kwargs = {}
        if self.half:
            kwargs["half"] = True
        self.model.predict(
            dummy, imgsz=self.config.imgsz, device=self.device,
            verbose=False, conf=self.config.conf, iou=self.config.iou, **kwargs,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        t0 = time.perf_counter()
        kwargs = {"half": True} if self.half else {}
        results = self.model.predict(
            source=frame,
            conf=self.config.conf,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            device=self.device,
            max_det=self.config.max_det,
            verbose=self.config.verbose,
            **kwargs,
        )
        self.inference_ms.append((time.perf_counter() - t0) * 1000.0)
        if not results:
            return []

        result = results[0]
        names = getattr(result, "names", self.names) or self.names
        boxes = getattr(result, "boxes", None)
        detections: list[Detection] = []
        if boxes is None or len(boxes) == 0:
            return detections
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, class_id in zip(xyxy, conf, cls):
            name = str(names.get(int(class_id), class_id)) if isinstance(names, dict) else str(names[int(class_id)])
            detections.append(Detection(
                class_id=int(class_id),
                class_name=name,
                confidence=float(score),
                xyxy=tuple(float(v) for v in box),
            ))
        return detections

    def average_inference_ms(self) -> float | None:
        return sum(self.inference_ms) / len(self.inference_ms) if self.inference_ms else None
