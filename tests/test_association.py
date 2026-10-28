from __future__ import annotations

import numpy as np
import pytest

from helmet_detection.association import (
    AssociationConfig,
    PersonHelmetAssociator,
)
from helmet_detection.stats import ViolationEventTracker
from helmet_detection.types import ComplianceStatus, Detection


def det(name: str, box, conf: float = 0.9) -> Detection:
    return Detection(class_id=0, class_name=name, confidence=conf, xyxy=box)


PERSON = det("Person", (100, 100, 300, 500))
HELMET = det("helmet", (150, 105, 250, 165))
NO_HELMET = det("no_helmet", (150, 105, 250, 165))


@pytest.fixture()
def associator() -> PersonHelmetAssociator:
    return PersonHelmetAssociator(AssociationConfig())


def test_helmet_in_head_region_is_compliant(associator):
    result = associator.analyze([PERSON, HELMET], frame_size=(800, 600))
    assert result.persons == 1
    assert result.compliant == 1
    assert result.violations == 0
    assert result.assessments[0].reason == "helmet_matched"
    assert result.assessments[0].evidence is HELMET


def test_no_helmet_in_head_region_is_violation(associator):
    result = associator.analyze([PERSON, NO_HELMET], frame_size=(800, 600))
    assert result.violations == 1
    assert result.assessments[0].status == ComplianceStatus.VIOLATION


def test_helmet_held_at_waist_does_not_make_person_compliant(associator):
    held = det("helmet", (160, 350, 240, 420))
    result = associator.analyze([PERSON, held], frame_size=(800, 600))
    assert result.persons == 1
    assert result.compliant == 0
    assert result.assessments[0].status == ComplianceStatus.UNKNOWN
    assert len(result.unmatched_helmets) == 1


def test_helmet_is_assigned_to_at_most_one_person(associator):
    person2 = det("Person", (400, 100, 600, 500))
    result = associator.analyze([PERSON, person2, HELMET], frame_size=(800, 600))
    assert result.compliant == 1
    assert result.unknown == 1
    statuses = {a.person.xyxy: a.status for a in result.assessments}
    assert statuses[PERSON.xyxy] == ComplianceStatus.COMPLIANT
    assert statuses[person2.xyxy] == ComplianceStatus.UNKNOWN


def test_low_confidence_helmet_is_ignored(associator):
    weak = det("helmet", (150, 105, 250, 165), conf=0.10)
    result = associator.analyze([PERSON, weak], frame_size=(800, 600))
    assert result.compliant == 0
    assert result.assessments[0].reason == "no_head_evidence"


def test_strict_unmatched_policy_marks_violation():
    strict = PersonHelmetAssociator(AssociationConfig(unmatched_policy="violation"))
    result = strict.analyze([PERSON], frame_size=(800, 600))
    assert result.violations == 1
    assert result.assessments[0].reason == "no_head_evidence_strict"


def test_person_touching_top_edge_is_unknown_even_in_strict_mode():
    strict = PersonHelmetAssociator(AssociationConfig(unmatched_policy="violation"))
    at_edge = det("Person", (100, 0, 300, 400))
    result = strict.analyze([at_edge], frame_size=(800, 600))
    assert result.assessments[0].status == ComplianceStatus.UNKNOWN
    assert result.assessments[0].reason == "head_out_of_frame"


def test_conflicting_evidence_is_unknown(associator):
    result = associator.analyze([PERSON, HELMET, NO_HELMET], frame_size=(800, 600))
    assert result.assessments[0].status == ComplianceStatus.UNKNOWN
    assert result.assessments[0].reason == "conflicting_helmet_and_no_helmet"


def test_helmet_without_person_is_not_compliance(associator):
    result = associator.analyze([HELMET], frame_size=(800, 600))
    assert result.persons == 0
    assert result.compliant == 0
    assert result.violations == 0
    assert len(result.unmatched_helmets) == 1
    assert result.compliance_rate is None


def test_compliance_rate_excludes_unknown(associator):
    person2 = det("Person", (400, 100, 600, 500))
    result = associator.analyze([PERSON, person2, HELMET], frame_size=(800, 600))
    assert result.compliance_rate == pytest.approx(100.0)


def _boxes(result):
    return [a.person.xyxy for a in result.assessments]


def test_violation_event_tracker_deduplicates():
    tracker = ViolationEventTracker(iou_threshold=0.2, ttl_seconds=5.0)
    box = (100.0, 100.0, 300.0, 500.0)
    first = tracker.update([box], now=100.0)
    second = tracker.update([box], now=101.0)
    assert first.new_count == 1
    assert second.new_count == 0
    assert tracker.total_events == 1
    assert second.active_events == 1


def test_violation_event_tracker_handles_two_people():
    tracker = ViolationEventTracker(ttl_seconds=5.0)
    a = (100.0, 100.0, 200.0, 400.0)
    b = (500.0, 100.0, 600.0, 400.0)
    update = tracker.update([a, b], now=10.0)
    assert update.new_count == 2
    assert tracker.total_events == 2


def test_renderer_produces_same_shape(associator):
    from helmet_detection.config import RenderConfig
    from helmet_detection.render import Renderer
    from helmet_detection.stats import FrameStats

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = associator.analyze([PERSON, HELMET], frame_size=(640, 480))
    stats = FrameStats.from_result(result, frame_index=0, timestamp=0.0)
    rendered = Renderer(RenderConfig()).draw(frame, result, stats, alert=False)
    assert rendered.shape == frame.shape
    assert rendered.dtype == np.uint8
