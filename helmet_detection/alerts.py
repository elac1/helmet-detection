"""违规提醒：控制台、告警横幅、可选 webhook 与截图触发。"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from .association import AssociationResult
from .config import AlertConfig
from .stats import EventUpdate, ViolationEventTracker, violation_boxes

logger = logging.getLogger(__name__)


@dataclass
class AlertDecision:
    triggered: bool = False
    message: str = ""
    new_events: list[Any] = field(default_factory=list)
    active_events: int = 0
    event_update: EventUpdate | None = None


class AlertManager:
    def __init__(self, config: AlertConfig, source: str = "") -> None:
        self.config = config
        self.source = source
        self.tracker = ViolationEventTracker(
            iou_threshold=config.event_iou_threshold,
            ttl_seconds=config.event_ttl_seconds,
        )
        self._last_alert_at = 0.0

    def update(
        self,
        result: AssociationResult,
        now: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> AlertDecision:
        now = time.monotonic() if now is None else now
        update = self.tracker.update(violation_boxes(result), now=now)
        active = update.active_events

        if not self.config.enabled or result.violations < self.config.min_violations:
            return AlertDecision(active_events=active, event_update=update)

        cooldown_ok = (now - self._last_alert_at) >= self.config.cooldown_seconds
        if not update.triggered or not cooldown_ok:
            return AlertDecision(active_events=active, event_update=update)

        self._last_alert_at = now
        message = f"检测到 {result.violations} 名人员未佩戴安全帽（新事件 {update.new_count} 个）"
        if self.config.console:
            indices = [i for i, a in enumerate(result.assessments) if a.is_violation]
            logger.warning("%s，人员序号: %s", message, indices)
        if self.config.terminal_bell:
            print("\a", end="", flush=True)
        if self.config.webhook_url:
            self._send_webhook(message, result, extra or {})

        return AlertDecision(
            triggered=True,
            message=message,
            new_events=list(update.new_events),
            active_events=active,
            event_update=update,
        )

    def _send_webhook(self, message: str, result: AssociationResult, extra: dict[str, Any]) -> None:
        payload = {
            "source": self.source,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "message": message,
            "counts": result.counts(),
            "extra": extra,
        }

        def post() -> None:
            try:
                requests.post(self.config.webhook_url, json=payload, timeout=3)
            except Exception as exc:  # noqa: BLE001
                logger.error("webhook 推送失败: %s", exc)

        threading.Thread(target=post, daemon=True).start()
