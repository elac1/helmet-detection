"""帧级统计、违规事件去重与运行汇总。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .association import AssociationResult
from .types import Box, ComplianceStatus, Detection, iou


@dataclass
class FrameStats:
    frame_index: int
    timestamp: float
    persons: int = 0
    compliant: int = 0
    violations: int = 0
    unknown: int = 0
    compliance_rate: float | None = None
    new_violation_events: int = 0
    alert_triggered: bool = False
    inference_ms: float | None = None

    def to_dict(self) -> dict:
        return {
            "frame": self.frame_index,
            "timestamp": round(self.timestamp, 3),
            "persons": self.persons,
            "compliant": self.compliant,
            "violations": self.violations,
            "unknown": self.unknown,
            "compliance_rate": None if self.compliance_rate is None else round(self.compliance_rate, 2),
            "new_violation_events": self.new_violation_events,
            "alert_triggered": self.alert_triggered,
            "inference_ms": None if self.inference_ms is None else round(self.inference_ms, 2),
        }

    @classmethod
    def from_result(
        cls,
        result: AssociationResult,
        frame_index: int,
        timestamp: float,
        new_violation_events: int = 0,
        alert_triggered: bool = False,
        inference_ms: float | None = None,
    ) -> "FrameStats":
        return cls(
            frame_index=frame_index,
            timestamp=timestamp,
            persons=result.persons,
            compliant=result.compliant,
            violations=result.violations,
            unknown=result.unknown,
            compliance_rate=result.compliance_rate,
            new_violation_events=new_violation_events,
            alert_triggered=alert_triggered,
            inference_ms=inference_ms,
        )


@dataclass
class _ViolationEvent:
    event_id: int
    box: Box
    first_seen: float
    last_seen: float
    hits: int = 1


@dataclass
class EventUpdate:
    new_events: list[_ViolationEvent] = field(default_factory=list)
    active_events: int = 0

    @property
    def new_count(self) -> int:
        return len(self.new_events)

    @property
    def triggered(self) -> bool:
        return bool(self.new_events)


class ViolationEventTracker:
    """基于空间重叠和时间 TTL 的轻量违规事件去重。

    没有接入多目标跟踪时，用 IoU/中心距离把连续帧中的同一个违规人员归为同一事件，
    避免视频里每帧都重复报警。后续可替换为 ByteTrack 的 track_id 方案。
    """

    def __init__(
        self,
        iou_threshold: float = 0.2,
        ttl_seconds: float = 6.0,
        max_center_distance: float = 0.35,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.ttl_seconds = ttl_seconds
        self.max_center_distance = max_center_distance
        self.events: list[_ViolationEvent] = []
        self._next_id = 1

    def _center_distance(self, a: Box, b: Box) -> float:
        aw, ah = a[2] - a[0], a[3] - a[1]
        bw, bh = b[2] - b[0], b[3] - b[1]
        diag = max((aw * aw + ah * ah) ** 0.5, (bw * bw + bh * bh) ** 0.5, 1.0)
        ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
        bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 / diag

    def update(self, violation_boxes: list[Box], now: float | None = None) -> EventUpdate:
        now = time.monotonic() if now is None else now
        self.events = [e for e in self.events if now - e.last_seen <= self.ttl_seconds]

        unmatched = list(range(len(violation_boxes)))
        matched_events: set[int] = set()
        # 先匹配 IoU 最高的组合，再回退到中心距离
        candidates: list[tuple[float, int, int]] = []
        for bi, box in enumerate(violation_boxes):
            for ei, event in enumerate(self.events):
                score = iou(box, event.box)
                if score >= self.iou_threshold:
                    candidates.append((score, bi, ei))
        candidates.sort(reverse=True)
        for _, bi, ei in candidates:
            if bi not in unmatched or ei in matched_events:
                continue
            event = self.events[ei]
            event.box = violation_boxes[bi]
            event.last_seen = now
            event.hits += 1
            unmatched.remove(bi)
            matched_events.add(ei)

        for bi in list(unmatched):
            box = violation_boxes[bi]
            best_ei, best_dist = None, self.max_center_distance
            for ei, event in enumerate(self.events):
                if ei in matched_events:
                    continue
                dist = self._center_distance(box, event.box)
                if dist < best_dist:
                    best_ei, best_dist = ei, dist
            if best_ei is not None:
                event = self.events[best_ei]
                event.box = box
                event.last_seen = now
                event.hits += 1
                matched_events.add(best_ei)
                unmatched.remove(bi)

        new_events: list[_ViolationEvent] = []
        for bi in unmatched:
            event = _ViolationEvent(
                event_id=self._next_id,
                box=violation_boxes[bi],
                first_seen=now,
                last_seen=now,
            )
            self._next_id += 1
            self.events.append(event)
            new_events.append(event)

        return EventUpdate(new_events=new_events, active_events=len(self.events))

    @property
    def total_events(self) -> int:
        return self._next_id - 1


@dataclass
class RunStats:
    source: str = ""
    frames_processed: int = 0
    person_observations: int = 0
    compliant_observations: int = 0
    violation_observations: int = 0
    unknown_observations: int = 0
    max_persons: int = 0
    max_violations: int = 0
    total_violation_events: int = 0
    alerts: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    inference_ms_sum: float = 0.0
    inference_frames: int = 0

    def update(self, stats: FrameStats, event_tracker: ViolationEventTracker | None = None) -> None:
        self.frames_processed += 1
        self.person_observations += stats.persons
        self.compliant_observations += stats.compliant
        self.violation_observations += stats.violations
        self.unknown_observations += stats.unknown
        self.max_persons = max(self.max_persons, stats.persons)
        self.max_violations = max(self.max_violations, stats.violations)
        self.alerts += int(stats.alert_triggered)
        if stats.inference_ms is not None:
            self.inference_ms_sum += stats.inference_ms
            self.inference_frames += 1
        if event_tracker is not None:
            self.total_violation_events = event_tracker.total_events

    def finish(self) -> None:
        self.finished_at = time.time()

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at or time.time()
        return max(0.0, end - self.started_at)

    @property
    def processing_fps(self) -> float | None:
        return self.frames_processed / self.duration_seconds if self.duration_seconds > 0 else None

    @property
    def average_inference_ms(self) -> float | None:
        return self.inference_ms_sum / self.inference_frames if self.inference_frames else None

    @property
    def average_compliance_rate(self) -> float | None:
        evaluated = self.compliant_observations + self.violation_observations
        return self.compliant_observations / evaluated * 100.0 if evaluated else None

    def to_dict(self) -> dict:
        return {
            "frames_processed": self.frames_processed,
            "person_observations": self.person_observations,
            "compliant_observations": self.compliant_observations,
            "violation_observations": self.violation_observations,
            "unknown_observations": self.unknown_observations,
            "max_persons": self.max_persons,
            "max_violations": self.max_violations,
            "total_violation_events": self.total_violation_events,
            "alerts": self.alerts,
            "duration_seconds": round(self.duration_seconds, 3),
            "processing_fps": None if self.processing_fps is None else round(self.processing_fps, 2),
            "average_inference_ms": None if self.average_inference_ms is None
            else round(self.average_inference_ms, 2),
            "average_compliance_rate": None if self.average_compliance_rate is None
            else round(self.average_compliance_rate, 2),
        }


def status_counts(result: AssociationResult) -> dict[str, int]:
    counts = {s.value: 0 for s in ComplianceStatus}
    for assessment in result.assessments:
        counts[assessment.status.value] += 1
    return counts


def violation_boxes(result: AssociationResult) -> list[Box]:
    return [a.person.xyxy for a in result.assessments if a.status == ComplianceStatus.VIOLATION]


def detections_to_boxes(detections: list[Detection]) -> list[Box]:
    return [d.xyxy for d in detections]
