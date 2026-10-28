"""人员与安全帽的关联逻辑。

核心原则：**检测到安全帽 != 人员正确佩戴安全帽**。

因此不直接根据画面中是否存在 helmet 判断合规，而是：
1. 以 Person 检测框为基准，计算其头部区域；
2. 只把落入头部区域、且置信度足够的 helmet / no_helmet 框作为候选证据；
3. 用贪心匹配保证一个安全帽只归属一名人员；
4. 人员未被匹配到任何头部证据时，默认标记为 unknown（而不是违规），
   避免把“遮挡/出画/模型漏检”误报成违规；
5. 安全帽被拿在手里、放在地上或属于画面外人员时，不会使任何人变为合规。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import (
    Box,
    ComplianceStatus,
    Detection,
    box_area,
    box_center,
    box_height,
    box_width,
    clip_box,
    coverage,
    expand_box,
    iou,
    normalized_center_distance,
    point_in_box,
)


def normalize_class_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


DEFAULT_PERSON_NAMES = ("person", "worker", "people", "pedestrian")
DEFAULT_HELMET_NAMES = ("helmet", "hardhat", "safety_helmet", "helmet_on")
DEFAULT_NO_HELMET_NAMES = (
    "no_helmet", "no_hardhat", "nohelmet", "without_helmet", "no_safety_helmet",
)


@dataclass
class AssociationConfig:
    # 头部区域：Person 框顶部一定比例、左右内缩一定比例
    head_height_ratio: float = 0.38
    head_width_shrink: float = 0.12
    # 标注中安全帽可能略高于 Person 框顶部，因此头部区域向上扩展少量
    head_top_margin: float = 0.08
    # 匹配阈值
    min_match_score: float = 0.28
    min_head_iou: float = 0.03
    min_coverage: float = 0.10
    min_person_confidence: float = 0.35
    min_helmet_confidence: float = 0.25
    min_no_helmet_confidence: float = 0.25
    # 安全帽中心到头部中心的归一化距离上限
    max_normalized_distance: float = 1.10
    # 证据框面积相对头部区域面积的上限，过滤异常大框
    max_evidence_area_ratio: float = 2.50
    # 一个人员同时匹配到 helmet 与 no_helmet 时如何处理：unknown / violation
    conflict_policy: str = "unknown"
    conflict_score_margin: float = 0.06
    # 无任何头部证据时的策略：unknown / violation
    unmatched_policy: str = "unknown"
    # 类别别名（小写、连字符统一为下划线）
    person_names: tuple[str, ...] = DEFAULT_PERSON_NAMES
    helmet_names: tuple[str, ...] = DEFAULT_HELMET_NAMES
    no_helmet_names: tuple[str, ...] = DEFAULT_NO_HELMET_NAMES

    def __post_init__(self) -> None:
        self.person_names = tuple(normalize_class_name(n) for n in self.person_names)
        self.helmet_names = tuple(normalize_class_name(n) for n in self.helmet_names)
        self.no_helmet_names = tuple(normalize_class_name(n) for n in self.no_helmet_names)
        if self.unmatched_policy not in ("unknown", "violation"):
            raise ValueError("unmatched_policy 必须是 unknown 或 violation")
        if self.conflict_policy not in ("unknown", "violation"):
            raise ValueError("conflict_policy 必须是 unknown 或 violation")


@dataclass
class PersonAssessment:
    person: Detection
    status: ComplianceStatus
    head_box: Box
    evidence: Detection | None = None
    score: float = 0.0
    reason: str = ""

    @property
    def is_violation(self) -> bool:
        return self.status == ComplianceStatus.VIOLATION

    @property
    def is_compliant(self) -> bool:
        return self.status == ComplianceStatus.COMPLIANT

    def to_dict(self) -> dict:
        return {
            "person": self.person.to_dict(),
            "status": self.status.value,
            "status_zh": self.status.zh,
            "head_box": [round(float(v), 2) for v in self.head_box],
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "match_score": round(float(self.score), 4),
            "reason": self.reason,
        }


@dataclass
class AssociationResult:
    assessments: list[PersonAssessment] = field(default_factory=list)
    unmatched_helmets: list[Detection] = field(default_factory=list)
    other_detections: list[Detection] = field(default_factory=list)

    @property
    def persons(self) -> int:
        return len(self.assessments)

    @property
    def compliant(self) -> int:
        return sum(a.status == ComplianceStatus.COMPLIANT for a in self.assessments)

    @property
    def violations(self) -> int:
        return sum(a.status == ComplianceStatus.VIOLATION for a in self.assessments)

    @property
    def unknown(self) -> int:
        return sum(a.status == ComplianceStatus.UNKNOWN for a in self.assessments)

    @property
    def evaluated(self) -> int:
        """排除 unknown 后参与合规率计算的人数。"""
        return self.compliant + self.violations

    @property
    def compliance_rate(self) -> float | None:
        return self.compliant / self.evaluated * 100.0 if self.evaluated else None

    def counts(self) -> dict[str, int]:
        return {
            "persons": self.persons,
            "compliant": self.compliant,
            "violations": self.violations,
            "unknown": self.unknown,
        }

    def to_dict(self) -> dict:
        return {
            "counts": self.counts(),
            "compliance_rate": None if self.compliance_rate is None
            else round(self.compliance_rate, 2),
            "assessments": [a.to_dict() for a in self.assessments],
            "unmatched_helmets": [d.to_dict() for d in self.unmatched_helmets],
        }


@dataclass
class _Candidate:
    person_index: int
    detection: Detection
    status: ComplianceStatus
    score: float
    head_iou: float
    cover: float


class PersonHelmetAssociator:
    """将 helmet / no_helmet 检测框关联到具体人员。"""

    def __init__(self, config: AssociationConfig | None = None) -> None:
        self.config = config or AssociationConfig()

    # ------------------------------------------------------------------ #
    # 类别判断
    # ------------------------------------------------------------------ #
    def is_person(self, det: Detection) -> bool:
        return normalize_class_name(det.class_name) in self.config.person_names

    def is_helmet(self, det: Detection) -> bool:
        return normalize_class_name(det.class_name) in self.config.helmet_names

    def is_no_helmet(self, det: Detection) -> bool:
        return normalize_class_name(det.class_name) in self.config.no_helmet_names

    def is_head_evidence(self, det: Detection) -> bool:
        return self.is_helmet(det) or self.is_no_helmet(det)

    # ------------------------------------------------------------------ #
    # 头部区域
    # ------------------------------------------------------------------ #
    def head_region(self, person: Detection, frame_size: tuple[int, int] | None = None) -> Box:
        x1, y1, x2, y2 = person.xyxy
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        shrink = self.config.head_width_shrink
        box = (
            x1 + w * shrink,
            y1 - h * self.config.head_top_margin,
            x2 - w * shrink,
            y1 + h * self.config.head_height_ratio,
        )
        if frame_size is not None:
            box = clip_box(box, *frame_size)
        return box

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        detections: list[Detection],
        frame_size: tuple[int, int] | None = None,
    ) -> AssociationResult:
        cfg = self.config
        persons = [
            d for d in detections
            if self.is_person(d) and d.confidence >= cfg.min_person_confidence
        ]
        evidences = [
            d for d in detections
            if self.is_head_evidence(d)
            and d.confidence >= (
                cfg.min_helmet_confidence if self.is_helmet(d) else cfg.min_no_helmet_confidence
            )
        ]
        other = [d for d in detections if not self.is_person(d) and not self.is_head_evidence(d)]

        heads = [self.head_region(p, frame_size) for p in persons]
        candidates: dict[int, list[_Candidate]] = {i: [] for i in range(len(persons))}

        for pi, (person, head) in enumerate(zip(persons, heads)):
            head_area = box_area(head)
            if head_area <= 0:
                continue
            for det in evidences:
                cand = self._score_candidate(person, head, det, pi)
                if cand is not None:
                    candidates[pi].append(cand)

        for pi in candidates:
            candidates[pi].sort(key=lambda c: c.score, reverse=True)

        # 贪心匹配：分数从高到低，一人一个头部证据，一个证据只归属一人
        pairs = [c for cs in candidates.values() for c in cs]
        pairs.sort(key=lambda c: c.score, reverse=True)
        used_persons: set[int] = set()
        used_evidence: set[int] = set()
        matched: dict[int, _Candidate] = {}
        ambiguous: set[int] = set()

        for cand in pairs:
            if cand.person_index in used_persons or id(cand.detection) in used_evidence:
                continue
            if cand.score < cfg.min_match_score:
                continue
            matched[cand.person_index] = cand
            used_persons.add(cand.person_index)
            used_evidence.add(id(cand.detection))
            # 同一人员存在分数接近的相反类别证据 -> 冲突
            for other_cand in candidates[cand.person_index]:
                if other_cand.status != cand.status and other_cand.score >= cand.score - cfg.conflict_score_margin:
                    ambiguous.add(cand.person_index)
                    break

        assessments: list[PersonAssessment] = []
        for pi, (person, head) in enumerate(zip(persons, heads)):
            cand = matched.get(pi)
            if cand is not None:
                if pi in ambiguous:
                    status = (ComplianceStatus.VIOLATION if cfg.conflict_policy == "violation"
                              else ComplianceStatus.UNKNOWN)
                    reason = "conflicting_helmet_and_no_helmet"
                elif cand.status == ComplianceStatus.COMPLIANT:
                    status, reason = ComplianceStatus.COMPLIANT, "helmet_matched"
                else:
                    status, reason = ComplianceStatus.VIOLATION, "no_helmet_matched"
                assessments.append(PersonAssessment(
                    person=person,
                    status=status,
                    head_box=head,
                    evidence=cand.detection,
                    score=cand.score,
                    reason=reason,
                ))
                continue

            status, reason = self._status_without_evidence(person, head, frame_size)
            assessments.append(PersonAssessment(
                person=person, status=status, head_box=head, evidence=None, score=0.0, reason=reason,
            ))

        helmet_dets = [d for d in evidences if self.is_helmet(d)]
        unmatched_helmets = [d for d in helmet_dets if id(d) not in used_evidence]
        return AssociationResult(
            assessments=assessments,
            unmatched_helmets=unmatched_helmets,
            other_detections=other,
        )

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _score_candidate(
        self,
        person: Detection,
        head: Box,
        det: Detection,
        person_index: int,
    ) -> _Candidate | None:
        cfg = self.config
        head_area = box_area(head)
        if head_area <= 0 or det.area <= 0:
            return None
        if det.area > head_area * cfg.max_evidence_area_ratio:
            return None

        head_iou = iou(det.xyxy, head)
        cover = coverage(det.xyxy, head)
        center_inside = point_in_box(det.center, head, margin=0.10)
        if head_iou < cfg.min_head_iou and cover < cfg.min_coverage and not center_inside:
            return None

        dist = normalized_center_distance(det.xyxy, head)
        if dist > cfg.max_normalized_distance:
            return None

        # 证据中心必须位于人员上半身，避免把拿在手里的安全帽算作佩戴
        px1, py1, px2, py2 = person.xyxy
        if det.center[1] > py1 + (py2 - py1) * 0.55:
            return None

        score = (
            0.45 * min(cover, 1.0)
            + 0.25 * min(head_iou * 3.0, 1.0)
            + 0.20 * (1.0 if center_inside else 0.0)
            + 0.10 * max(0.0, 1.0 - dist / cfg.max_normalized_distance)
        )
        status = ComplianceStatus.COMPLIANT if self.is_helmet(det) else ComplianceStatus.VIOLATION
        return _Candidate(
            person_index=person_index,
            detection=det,
            status=status,
            score=score,
            head_iou=head_iou,
            cover=cover,
        )

    def _status_without_evidence(
        self,
        person: Detection,
        head: Box,
        frame_size: tuple[int, int] | None,
    ) -> tuple[ComplianceStatus, str]:
        cfg = self.config
        if box_height(person.xyxy) < 40 or box_width(person.xyxy) < 20:
            return ComplianceStatus.UNKNOWN, "person_too_small"
        # Person 框贴住画面顶部：头部可能出画，不能直接判违规
        if person.xyxy[1] <= 2.0:
            return ComplianceStatus.UNKNOWN, "head_out_of_frame"
        if frame_size is not None and (
            head[0] <= 1.0 or head[1] <= 1.0
            or head[2] >= frame_size[0] - 1.0 or head[3] >= frame_size[1] - 1.0
        ):
            return ComplianceStatus.UNKNOWN, "head_partially_out_of_frame"
        if cfg.unmatched_policy == "violation":
            return ComplianceStatus.VIOLATION, "no_head_evidence_strict"
        return ComplianceStatus.UNKNOWN, "no_head_evidence"
