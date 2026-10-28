"""结果持久化：标注图片/视频、逐帧 CSV、JSON 汇总、违规截图。"""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .stats import FrameStats


def safe_stem(value: str) -> str:
    stem = Path(value).stem if value else "stream"
    stem = re.sub(r"[^\w\-.]+", "_", stem, flags=re.UNICODE).strip("._")
    return stem[:80] or "stream"


def write_image(path: Path, frame: np.ndarray, quality: int = 92) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return False
    path.write_bytes(buffer.tobytes())
    return True


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


class CsvStatsWriter:
    HEADER = [
        "frame", "timestamp", "persons", "compliant", "violations", "unknown",
        "compliance_rate", "new_violation_events", "alert_triggered", "inference_ms",
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", newline="", encoding="utf-8-sig")
        self._writer = csv.DictWriter(self._file, fieldnames=self.HEADER)
        self._writer.writeheader()

    def write(self, stats: FrameStats) -> None:
        row = stats.to_dict()
        self._writer.writerow({key: row.get(key) for key in self.HEADER})
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")

    def write(self, data: dict[str, Any]) -> None:
        self._file.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class SnapshotSaver:
    def __init__(
        self,
        directory: Path,
        max_per_event: int = 3,
        quality: int = 92,
        source: str = "",
    ) -> None:
        self.directory = directory
        self.max_per_event = max_per_event
        self.quality = quality
        self.source = source
        self._counts: dict[int, int] = {}
        self.paths: list[Path] = []

    def save(self, frame: np.ndarray, event_id: int, frame_index: int) -> Path | None:
        count = self._counts.get(event_id, 0)
        if count >= self.max_per_event:
            return None
        self._counts[event_id] = count + 1
        name = f"{safe_stem(self.source)}_e{event_id}_f{frame_index}_{int(time.time() * 1000)}.jpg"
        path = self.directory / name
        if write_image(path, frame, self.quality):
            self.paths.append(path)
            return path
        return None

    def write_index(self, path: Path) -> None:
        write_json(path, {"snapshots": [str(p) for p in self.paths]})


@dataclass
class RunArtifacts:
    run_dir: Path
    csv_path: Path | None = None
    jsonl_path: Path | None = None
    summary_path: Path | None = None
    annotated_path: Path | None = None
    snapshot_dir: Path | None = None
    snapshot_index: Path | None = None

    def to_dict(self) -> dict[str, str]:
        return {
            key: str(value) for key, value in (
                ("run_dir", self.run_dir),
                ("csv", self.csv_path),
                ("jsonl", self.jsonl_path),
                ("summary", self.summary_path),
                ("annotated", self.annotated_path),
                ("snapshots", self.snapshot_dir),
                ("snapshot_index", self.snapshot_index),
            ) if value is not None
        }
