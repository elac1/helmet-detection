"""检测框、几何计算与数据结构。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

Box = tuple[float, float, float, float]
Point = tuple[float, float]


def box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_center(box: Box) -> Point:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def box_width(box: Box) -> float:
    return max(0.0, box[2] - box[0])


def box_height(box: Box) -> float:
    return max(0.0, box[3] - box[1])


def intersection_area(a: Box, b: Box) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: Box, b: Box) -> float:
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def coverage(inner: Box, outer: Box) -> float:
    """inner 框被 outer 框覆盖的比例，用于判断安全帽是否落在头部区域。"""
    inner_area = box_area(inner)
    if inner_area <= 0:
        return 0.0
    return min(1.0, intersection_area(inner, outer) / inner_area)


def point_in_box(point: Point, box: Box, margin: float = 0.0) -> bool:
    x, y = point
    w = box_width(box)
    h = box_height(box)
    return (box[0] - w * margin <= x <= box[2] + w * margin
            and box[1] - h * margin <= y <= box[3] + h * margin)


def expand_box(box: Box, ratio_x: float = 0.0, ratio_y: float | None = None) -> Box:
    if ratio_y is None:
        ratio_y = ratio_x
    w = box_width(box)
    h = box_height(box)
    return (box[0] - w * ratio_x, box[1] - h * ratio_y,
            box[2] + w * ratio_x, box[3] + h * ratio_y)


def clip_box(box: Box, width: int, height: int) -> Box:
    return (min(max(box[0], 0.0), float(width)),
            min(max(box[1], 0.0), float(height)),
            min(max(box[2], 0.0), float(width)),
            min(max(box[3], 0.0), float(height)))


def center_distance(a: Box, b: Box) -> float:
    ax, ay = box_center(a)
    bx, by = box_center(b)
    return math.hypot(ax - bx, ay - by)


def normalized_center_distance(a: Box, b: Box) -> float:
    """中心距离除以外框对角线长度，便于跨尺度比较。"""
    diag = math.hypot(max(box_width(a), box_width(b)), max(box_height(a), box_height(b)))
    return center_distance(a, b) / diag if diag > 0 else float("inf")


@dataclass(frozen=True)
class Detection:
    """统一的检测结果，不依赖 Ultralytics 对象。"""

    class_id: int
    class_name: str
    confidence: float
    xyxy: Box
    source: str = "model"  # model / ground_truth

    @property
    def center(self) -> Point:
        return box_center(self.xyxy)

    @property
    def area(self) -> float:
        return box_area(self.xyxy)

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "xyxy": [round(float(v), 2) for v in self.xyxy],
            "source": self.source,
        }


class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"      # 已正确佩戴
    VIOLATION = "violation"      # 未佩戴 / 违规
    UNKNOWN = "unknown"          # 无法确认（遮挡、出画、无头部检测）

    @property
    def zh(self) -> str:
        return {
            ComplianceStatus.COMPLIANT: "已佩戴",
            ComplianceStatus.VIOLATION: "未佩戴",
            ComplianceStatus.UNKNOWN: "待确认",
        }[self]


def count_by_status(items: Iterable["PersonAssessment"]) -> dict[str, int]:
    counts = {status.value: 0 for status in ComplianceStatus}
    for item in items:
        counts[item.status.value] += 1
    return counts
