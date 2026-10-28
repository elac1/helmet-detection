"""Construction-PPE 数据集解析工具（YOLO 格式）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import yaml
from PIL import Image

from .types import Detection

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


@dataclass
class DatasetSpec:
    yaml_path: Path
    root: Path
    class_names: dict[int, str]
    split_dirs: dict[str, list[Path]] = field(default_factory=dict)

    def names(self) -> list[str]:
        return [self.class_names[i] for i in sorted(self.class_names)]

    def split_images(self, split: str) -> list[Path]:
        images: list[Path] = []
        for directory in self.split_dirs.get(split, []):
            if directory.is_file():
                for line in directory.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        candidate = Path(line.strip())
                        images.append(candidate if candidate.is_absolute() else self.root / candidate)
            elif directory.is_dir():
                images.extend(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        return sorted(set(images))

    def label_path(self, image: Path) -> Path | None:
        """根据 image 路径中的 images 组件推导 labels 目录，兼容任意嵌套。"""
        parts = image.resolve().parts
        if "images" in parts:
            index = len(parts) - 1 - parts[::-1].index("images")
            labels_root = Path(*parts[:index]) / "labels"
            relative = Path(*parts[index + 1:])
            return (labels_root / relative).with_suffix(".txt")
        # 兜底：同目录同名 txt
        return image.with_suffix(".txt")


def load_dataset_spec(data: str | Path) -> DatasetSpec:
    path = Path(data).expanduser()
    if path.is_dir():
        for candidate in ("data.yaml", "data.yml", "dataset.yaml"):
            if (path / candidate).exists():
                path = path / candidate
                break
    if not path.exists():
        raise FileNotFoundError(f"数据集 YAML 不存在: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    root_value = raw.get("path") or "."
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        relative = (path.parent / root).resolve()
        if relative.exists():
            root = relative
        else:
            try:
                from ultralytics import settings
                datasets_dir = Path(settings.get("datasets_dir", "datasets"))
                candidate = datasets_dir / root
                root = candidate if candidate.exists() else relative
            except Exception:  # noqa: BLE001
                root = relative
    root = root.resolve()

    names_raw = raw.get("names") or {}
    if isinstance(names_raw, list):
        class_names = {i: str(name) for i, name in enumerate(names_raw)}
    else:
        class_names = {int(k): str(v) for k, v in names_raw.items()}

    spec = DatasetSpec(yaml_path=path.resolve(), root=root, class_names=class_names)
    image_roots: list[Path] = []
    for split in SPLITS:
        value = raw.get(split)
        if value is None:
            continue
        entries = value if isinstance(value, list) else [value]
        dirs: list[Path] = []
        for entry in entries:
            entry_path = Path(entry).expanduser()
            if not entry_path.is_absolute():
                entry_path = root / entry_path
            dirs.append(entry_path.resolve())
            if entry_path.is_dir() and entry_path.name == "images":
                image_roots.append(entry_path.resolve())
        spec.split_dirs[split] = dirs
    spec.split_dirs["_all_image_roots"] = image_roots
    return spec


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.width, image.height


def read_yolo_labels(
    label_path: Path,
    class_names: dict[int, str],
    image_width: int,
    image_height: int,
    source: str = "ground_truth",
) -> list[Detection]:
    detections: list[Detection] = []
    if not label_path.exists():
        return detections
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        if class_id not in class_names:
            continue
        x1 = (cx - w / 2) * image_width
        y1 = (cy - h / 2) * image_height
        x2 = (cx + w / 2) * image_width
        y2 = (cy + h / 2) * image_height
        detections.append(Detection(
            class_id=class_id,
            class_name=class_names[class_id],
            confidence=1.0,
            xyxy=(x1, y1, x2, y2),
            source=source,
        ))
    return detections


def load_ground_truth(
    image_path: Path,
    spec: DatasetSpec,
    image_dims: tuple[int, int] | None = None,
) -> list[Detection]:
    label_path = spec.label_path(image_path)
    if label_path is None:
        return []
    width, height = image_dims or image_size(image_path)
    return read_yolo_labels(label_path, spec.class_names, width, height)


def count_labels(label_path: Path) -> tuple[int, int]:
    """返回 (有效框数, 非法行数)。"""
    valid = invalid = 0
    if not label_path.exists():
        return 0, 0
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            invalid += 1
            continue
        try:
            [float(v) for v in parts]
        except ValueError:
            invalid += 1
            continue
        valid += 1
    return valid, invalid


def iter_split(
    spec: DatasetSpec,
    split: str,
    max_images: int | None = None,
) -> Iterator[Path]:
    images = spec.split_images(split)
    if max_images is not None:
        images = images[:max_images]
    yield from images


def summarize_split_sizes(spec: DatasetSpec) -> dict[str, int]:
    return {split: len(spec.split_images(split)) for split in SPLITS}
