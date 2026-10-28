"""检测结果可视化：检测框、类别、置信度、统计面板与告警横幅。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .association import AssociationResult
from .config import RenderConfig
from .stats import FrameStats
from .types import ComplianceStatus, Detection

STATUS_COLORS = {
    ComplianceStatus.COMPLIANT: (46, 204, 113),
    ComplianceStatus.VIOLATION: (231, 76, 60),
    ComplianceStatus.UNKNOWN: (243, 156, 18),
}
EVIDENCE_COLORS = {
    "helmet": (39, 174, 96),
    "no_helmet": (192, 57, 43),
}
OTHER_COLOR = (189, 195, 199)
UNMATCHED_COLOR = (127, 140, 141)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
STATUS_EN = {
    ComplianceStatus.COMPLIANT: "OK",
    ComplianceStatus.VIOLATION: "VIOLATION",
    ComplianceStatus.UNKNOWN: "UNKNOWN",
}


def find_cjk_font() -> str | None:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    for pattern in ("/usr/share/fonts/**/NotoSansCJK*", "/usr/share/fonts/**/wqy*"):
        matches = sorted(Path("/").glob(pattern.lstrip("/")))
        if matches:
            regular = [m for m in matches if "Regular" in m.name]
            return str(regular[0] if regular else matches[0])
    return None


class Renderer:
    def __init__(self, config: RenderConfig) -> None:
        self.config = config
        self.font_path = config.font_path or find_cjk_font()
        self.use_cjk = bool(self.font_path and config.language == "zh")
        self._font_cache: dict[int, ImageFont.FreeTypeFont] = {}

    # ------------------------------------------------------------------ #
    def draw(
        self,
        frame: np.ndarray,
        result: AssociationResult,
        stats: FrameStats,
        alert: bool = False,
        extra_lines: list[str] | None = None,
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        scale = max(0.6, min(2.2, width / 1280.0))
        thickness = max(2, int(round(self.config.line_thickness * scale)))
        font_size = max(14, int(round(20 * scale)))
        small_size = max(12, int(round(16 * scale)))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image, "RGBA")
        font = self._font(font_size)
        small_font = self._font(small_size)

        # 1. 人员框与状态
        for i, assessment in enumerate(result.assessments):
            box = tuple(int(round(v)) for v in assessment.person.xyxy)
            color = STATUS_COLORS[assessment.status]
            draw.rectangle(box, outline=color + (255,), width=thickness)
            if self.config.show_head_region:
                self._dashed_rect(draw, assessment.head_box, (241, 196, 15, 220), max(1, thickness - 1))

            parts = [self._person_label(i, assessment.person, assessment.status)]
            if self.config.show_confidence:
                parts.append(f"{assessment.person.confidence:.2f}")
            text = " ".join(parts)
            self._label(draw, box, text, font, WHITE, color + (235,))

        # 2. 头部证据框（安全帽 / 未佩戴）
        drawn_evidence: set[int] = set()
        for assessment in result.assessments:
            evidence = assessment.evidence
            if evidence is None or id(evidence) in drawn_evidence:
                continue
            drawn_evidence.add(id(evidence))
            key = "helmet" if assessment.is_compliant else "no_helmet"
            color = EVIDENCE_COLORS[key]
            self._detection_box(draw, evidence, color, font, thickness,
                                label_zh=self._evidence_label(evidence))

        # 3. 未关联到人员的安全帽：明确标注“未关联”，不计入合规
        if self.config.show_unmatched_detections:
            for det in result.unmatched_helmets:
                if id(det) in drawn_evidence:
                    continue
                self._detection_box(
                    draw, det, UNMATCHED_COLOR, small_font, max(1, thickness - 1),
                    label_zh="安全帽(未关联)" if self.use_cjk else "Helmet (unassigned)",
                )

        # 4. 其他 PPE 类别（可选）
        if self.config.show_other_detections:
            for det in result.other_detections:
                self._detection_box(draw, det, OTHER_COLOR, small_font,
                                    max(1, thickness - 1), label_zh=self._class_label(det))

        # 5. 统计面板
        if self.config.show_stats_panel:
            self._stats_panel(draw, width, height, stats, font, small_font, scale, extra_lines)

        # 6. 告警横幅
        if alert and self.config.show_alert_banner:
            self._alert_banner(draw, width, height, stats, scale)

        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------ #
    def _font(self, size: int) -> ImageFont.ImageFont:
        if not self.use_cjk or not self.font_path:
            return ImageFont.load_default()
        if size not in self._font_cache:
            self._font_cache[size] = ImageFont.truetype(self.font_path, size=size)
        return self._font_cache[size]

    def _person_label(self, index: int, person: Detection, status: ComplianceStatus) -> str:
        if self.config.language == "zh" and self.use_cjk:
            return f"人员#{index} {status.zh}"
        return f"Person#{index} {STATUS_EN[status]}"

    def _evidence_label(self, det: Detection) -> str:
        zh = self.config.language == "zh" and self.use_cjk
        if det.class_name.lower() in ("no_helmet", "no_hardhat", "no-helmet"):
            name = "未佩戴安全帽" if zh else "NO HELMET"
        else:
            name = "安全帽" if zh else "HELMET"
        return f"{name} {det.confidence:.2f}"

    @staticmethod
    def _class_label(det: Detection) -> str:
        return f"{det.class_name} {det.confidence:.2f}"

    def _label(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        text: str,
        font: ImageFont.ImageFont,
        fg: tuple[int, int, int],
        bg: tuple[int, int, int, int],
    ) -> None:
        pad = 3
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:  # noqa: BLE001
            tw, th = len(text) * 8, 12
        x = max(0, box[0])
        y = box[1] - th - 2 * pad
        if y < 0:
            y = box[1] + 2
        draw.rectangle((x, y, x + tw + 2 * pad, y + th + 2 * pad), fill=bg)
        draw.text((x + pad, y + pad - 1), text, font=font, fill=fg)

    def _detection_box(
        self,
        draw: ImageDraw.ImageDraw,
        det: Detection,
        color: tuple[int, int, int],
        font: ImageFont.ImageFont,
        thickness: int,
        label_zh: str,
    ) -> None:
        box = tuple(int(round(v)) for v in det.xyxy)
        draw.rectangle(box, outline=color + (255,), width=thickness)
        text = label_zh
        if self.config.show_confidence and not label_zh.endswith(f"{det.confidence:.2f}"):
            text = f"{label_zh} {det.confidence:.2f}"
        self._label(draw, box, text, font, WHITE, color + (235,))

    def _dashed_rect(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[float, float, float, float],
        color: tuple[int, int, int, int],
        width: int,
        dash: int = 6,
    ) -> None:
        x1, y1, x2, y2 = (int(v) for v in box)
        for x in range(x1, x2, dash * 2):
            draw.line((x, y1, min(x + dash, x2), y1), fill=color, width=width)
            draw.line((x, y2, min(x + dash, x2), y2), fill=color, width=width)
        for y in range(y1, y2, dash * 2):
            draw.line((x1, y, x1, min(y + dash, y2)), fill=color, width=width)
            draw.line((x2, y, x2, min(y + dash, y2)), fill=color, width=width)

    def _stats_panel(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        stats: FrameStats,
        font: ImageFont.ImageFont,
        small_font: ImageFont.ImageFont,
        scale: float,
        extra_lines: list[str] | None,
    ) -> None:
        zh = self.config.language == "zh" and self.use_cjk
        lines = [
            (f"人员 {stats.persons}   已佩戴 {stats.compliant}   "
             f"未佩戴 {stats.violations}   待确认 {stats.unknown}" if zh
             else f"Person {stats.persons}  OK {stats.compliant}  "
                  f"Violation {stats.violations}  Unknown {stats.unknown}"),
            (f"合规率 {stats.compliance_rate:.1f}% (当前帧)" if stats.compliance_rate is not None
             else ("合规率 -- (当前帧)" if zh else "Compliance -- (frame)")),
        ]
        lines.extend(extra_lines or [])
        pad = int(10 * scale)
        line_h = int(26 * scale)
        text_widths = []
        for i, line in enumerate(lines):
            current_font = font if i < 2 else small_font
            try:
                text_widths.append(draw.textlength(line, font=current_font))
            except Exception:  # noqa: BLE001
                text_widths.append(len(line) * 10 * scale)
        panel_w = int(max(text_widths, default=200) + pad * 3)
        panel_w = min(panel_w, width - pad * 2)
        panel_h = line_h * len(lines) + pad
        draw.rectangle((pad, pad, pad + panel_w, pad + panel_h), fill=(0, 0, 0, 175))
        y = pad + pad // 2
        for i, line in enumerate(lines):
            draw.text((pad * 2, y), line, font=font if i < 2 else small_font, fill=(255, 255, 255))
            y += line_h

    def _alert_banner(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        stats: FrameStats,
        scale: float,
    ) -> None:
        zh = self.config.language == "zh" and self.use_cjk
        text = (f"警告：检测到 {stats.violations} 人未佩戴安全帽"
                if zh else f"ALERT: {stats.violations} worker(s) without helmet")
        font = self._font(max(18, int(26 * scale)))
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:  # noqa: BLE001
            tw, th = len(text) * 14, 26
        margin = int(12 * scale)
        x1 = max(margin, (width - tw) // 2 - margin)
        y1 = int(80 * scale)
        x2 = min(width - margin, (width + tw) // 2 + margin)
        y2 = y1 + th + margin * 2
        draw.rectangle((x1, y1, x2, y2), fill=(192, 57, 43, 230))
        draw.text(((width - tw) // 2, y1 + margin), text, font=font, fill=(255, 255, 255))
