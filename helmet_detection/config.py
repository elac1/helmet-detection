"""配置加载与默认配置。"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .association import (
    AssociationConfig,
    DEFAULT_HELMET_NAMES,
    DEFAULT_NO_HELMET_NAMES,
    DEFAULT_PERSON_NAMES,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass
class ModelConfig:
    path: str = "runs/ppe/yolov8n_baseline/weights/best.pt"
    device: str = "auto"          # auto / cpu / 0 / cuda:0
    conf: float = 0.35
    iou: float = 0.45
    imgsz: int = 640
    max_det: int = 100
    half: bool = False
    verbose: bool = False


@dataclass
class ClassNamesConfig:
    person: list[str] = field(default_factory=lambda: list(DEFAULT_PERSON_NAMES))
    helmet: list[str] = field(default_factory=lambda: list(DEFAULT_HELMET_NAMES))
    no_helmet: list[str] = field(default_factory=lambda: list(DEFAULT_NO_HELMET_NAMES))


@dataclass
class RenderConfig:
    language: str = "zh"                 # zh / en
    show_labels: bool = True
    show_confidence: bool = True
    show_head_region: bool = False
    show_stats_panel: bool = True
    show_alert_banner: bool = True
    show_unmatched_detections: bool = True
    show_other_detections: bool = False
    font_path: str | None = None
    line_thickness: int = 2


@dataclass
class AlertConfig:
    enabled: bool = True
    cooldown_seconds: float = 5.0
    event_ttl_seconds: float = 6.0
    event_iou_threshold: float = 0.20
    min_violations: int = 1
    console: bool = True
    terminal_bell: bool = False
    snapshot_on_new_event: bool = True
    max_snapshots_per_event: int = 3
    webhook_url: str | None = None


@dataclass
class OutputConfig:
    dir: str = "outputs"
    save_annotated: bool = True
    save_csv: bool = True
    save_json: bool = False
    save_violation_snapshots: bool = True
    display: bool = False
    max_width: int | None = 1280
    video_codec: str = "mp4v"
    jpeg_quality: int = 92


@dataclass
class SystemConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    classes: ClassNamesConfig = field(default_factory=ClassNamesConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def association_config(self) -> AssociationConfig:
        """把类别别名注入关联配置，返回独立副本。"""
        cfg = self.association
        cfg.person_names = tuple(self.classes.person)
        cfg.helmet_names = tuple(self.classes.helmet)
        cfg.no_helmet_names = tuple(self.classes.no_helmet)
        cfg.__post_init__()
        return cfg

    def output_dir(self) -> Path:
        path = Path(self.output.dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def model_path(self) -> Path:
        path = Path(self.model.path)
        if path.is_absolute():
            return path
        candidate = PROJECT_ROOT / path
        if candidate.exists():
            return candidate
        local = Path.cwd() / path
        return local if local.exists() else candidate


def _filtered(cls: type, values: dict[str, Any] | None) -> Any:
    """只保留 dataclass 声明过的字段，避免配置里多余键导致崩溃。"""
    values = values or {}
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in values.items() if k in allowed})


def load_config(path: str | Path | None = None) -> SystemConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        if path:
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        return SystemConfig()

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg = SystemConfig(
        model=_filtered(ModelConfig, raw.get("model")),
        classes=_filtered(ClassNamesConfig, raw.get("classes")),
        association=_filtered(AssociationConfig, raw.get("association")),
        render=_filtered(RenderConfig, raw.get("render")),
        alert=_filtered(AlertConfig, raw.get("alert")),
        output=_filtered(OutputConfig, raw.get("output")),
    )
    return cfg


def config_to_dict(cfg: SystemConfig) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if is_dataclass(value):
            return {f.name: convert(getattr(value, f.name)) for f in fields(value)}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return list(value)
        return value

    return convert(cfg)
