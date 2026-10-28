"""数据集统计与 GT 层面的人员-安全帽关联分析。"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .association import PersonHelmetAssociator
from .config import PROJECT_ROOT
from .dataset import (
    SPLITS,
    DatasetSpec,
    count_labels,
    image_size,
    load_dataset_spec,
    read_yolo_labels,
)
from .types import ComplianceStatus


@dataclass
class SplitAnalysis:
    split: str
    images: int = 0
    labeled_images: int = 0
    missing_labels: int = 0
    malformed_lines: int = 0
    boxes: int = 0
    class_counts: Counter = field(default_factory=Counter)
    images_with_class: Counter = field(default_factory=Counter)
    boxes_per_image: Counter = field(default_factory=Counter)
    image_sizes: Counter = field(default_factory=Counter)
    gt_person_status: Counter = field(default_factory=Counter)
    gt_unmatched_helmet: int = 0
    gt_unmatched_no_helmet: int = 0
    images_with_helmet_but_no_person: int = 0
    images_with_person_but_no_head_evidence: int = 0
    association_errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "images": self.images,
            "labeled_images": self.labeled_images,
            "missing_labels": self.missing_labels,
            "malformed_lines": self.malformed_lines,
            "boxes": self.boxes,
            "boxes_per_image": dict(sorted(self.boxes_per_image.items())),
            "class_counts": dict(self.class_counts.most_common()),
            "images_with_class": dict(self.images_with_class.most_common()),
            "image_sizes": {f"{w}x{h}": c for (w, h), c in self.image_sizes.most_common(10)},
            "gt_person_status": dict(self.gt_person_status),
            "gt_unmatched_helmet": self.gt_unmatched_helmet,
            "gt_unmatched_no_helmet": self.gt_unmatched_no_helmet,
            "images_with_helmet_but_no_person": self.images_with_helmet_but_no_person,
            "images_with_person_but_no_head_evidence": self.images_with_person_but_no_head_evidence,
            "association_errors": self.association_errors,
        }


def _analyze_split(spec: DatasetSpec, split: str, max_images: int | None = None) -> SplitAnalysis:
    analysis = SplitAnalysis(split=split)
    images = spec.split_images(split)
    if max_images is not None:
        images = images[:max_images]
    associator = PersonHelmetAssociator()

    for image_path in tqdm(images, desc=f"分析 {split}", unit="img", leave=False):
        analysis.images += 1
        try:
            width, height = image_size(image_path)
        except Exception:  # noqa: BLE001
            analysis.association_errors += 1
            continue
        analysis.image_sizes[(width, height)] += 1

        label_path = spec.label_path(image_path)
        if label_path is None or not label_path.exists():
            analysis.missing_labels += 1
            continue
        analysis.labeled_images += 1
        valid, invalid = count_labels(label_path)
        analysis.malformed_lines += invalid
        analysis.boxes += valid
        analysis.boxes_per_image[valid] += 1

        detections = read_yolo_labels(label_path, spec.class_names, width, height)
        present = {d.class_name for d in detections}
        for name in present:
            analysis.images_with_class[name] += 1
        for det in detections:
            analysis.class_counts[det.class_name] += 1

        lower = {name.lower() for name in present}
        if lower & {"helmet", "hardhat"} and not lower & {"person", "worker"}:
            analysis.images_with_helmet_but_no_person += 1

        try:
            result = associator.analyze(detections, frame_size=(width, height))
        except Exception:  # noqa: BLE001
            analysis.association_errors += 1
            continue

        for assessment in result.assessments:
            analysis.gt_person_status[assessment.status.value] += 1
        analysis.gt_unmatched_helmet += len(result.unmatched_helmets)
        analysis.gt_unmatched_no_helmet += sum(
            1 for det in detections
            if associator.is_no_helmet(det) and all(a.evidence is not det for a in result.assessments)
        )
        if result.persons and not any(a.evidence for a in result.assessments):
            analysis.images_with_person_but_no_head_evidence += 1

    return analysis


def analyze_dataset(
    data: str | Path,
    output_dir: str | Path | None = None,
    max_images: int | None = None,
) -> dict[str, Any]:
    spec = load_dataset_spec(data)
    output = Path(output_dir) if output_dir else PROJECT_ROOT / "outputs" / "dataset_analysis"
    output.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "dataset": {
            "yaml": str(spec.yaml_path),
            "root": str(spec.root),
            "classes": spec.names(),
            "class_count": len(spec.class_names),
        },
        "splits": {},
    }
    totals = {
        "images": 0, "boxes": 0, "missing_labels": 0, "malformed_lines": 0,
        "class_counts": Counter(), "gt_person_status": Counter(),
        "gt_unmatched_helmet": 0,
        "images_with_helmet_but_no_person": 0,
        "images_with_person_but_no_head_evidence": 0,
    }

    for split in SPLITS:
        if not spec.split_dirs.get(split):
            continue
        analysis = _analyze_split(spec, split, max_images=max_images)
        results["splits"][split] = analysis.to_dict()
        totals["images"] += analysis.images
        totals["boxes"] += analysis.boxes
        totals["missing_labels"] += analysis.missing_labels
        totals["malformed_lines"] += analysis.malformed_lines
        totals["class_counts"].update(analysis.class_counts)
        totals["gt_person_status"].update(analysis.gt_person_status)
        totals["gt_unmatched_helmet"] += analysis.gt_unmatched_helmet
        totals["images_with_helmet_but_no_person"] += analysis.images_with_helmet_but_no_person
        totals["images_with_person_but_no_head_evidence"] += analysis.images_with_person_but_no_head_evidence

    results["totals"] = {
        "images": totals["images"],
        "boxes": totals["boxes"],
        "missing_labels": totals["missing_labels"],
        "malformed_lines": totals["malformed_lines"],
        "class_counts": dict(totals["class_counts"].most_common()),
        "gt_person_status": dict(totals["gt_person_status"]),
        "gt_unmatched_helmet": totals["gt_unmatched_helmet"],
        "images_with_helmet_but_no_person": totals["images_with_helmet_but_no_person"],
        "images_with_person_but_no_head_evidence": totals["images_with_person_but_no_head_evidence"],
    }
    results["recommendations"] = _recommendations(results)

    json_path = output / "dataset_analysis.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = output / "dataset_analysis.md"
    report_path.write_text(_markdown_report(results), encoding="utf-8")
    results["_artifacts"] = {"json": str(json_path), "report": str(report_path)}
    return results


def _recommendations(results: dict[str, Any]) -> list[str]:
    totals = results["totals"]
    classes = totals["class_counts"]
    notes: list[str] = []
    person = sum(v for k, v in classes.items() if k.lower() in ("person", "worker"))
    helmet = sum(v for k, v in classes.items() if k.lower() in ("helmet", "hardhat"))
    no_helmet = sum(v for k, v in classes.items() if k.lower() in ("no_helmet", "no_hardhat"))
    status = totals["gt_person_status"]
    unknown = status.get("unknown", 0)
    persons = sum(status.values())

    if totals["images"] < 2000:
        notes.append(
            f"数据集仅 {totals['images']} 张图片，建议使用 yolov8n + 强增强 + 迁移学习；"
            "若 mAP 已饱和，再考虑 yolov8s/m，而不必一开始换大模型。"
        )
    if person == 0:
        notes.append("未找到 Person 标注，无法做人员-安全帽关联；需要补充人员框标注。")
    if helmet == 0 or no_helmet == 0:
        notes.append("helmet 或 no_helmet 类别缺失，合规判断会大量落入 unknown。")
    if persons and unknown / persons > 0.25:
        notes.append(
            f"GT 关联中 unknown 占 {unknown / persons:.1%}，需检查头部区域阈值、Person 标注质量，"
            "以及数据集中是否存在大量遮挡/出画人员。"
        )
    if totals["images_with_helmet_but_no_person"]:
        notes.append(
            f"有 {totals['images_with_helmet_but_no_person']} 张图片只标注了安全帽而没有人，"
            "这正好说明不能以“画面存在安全帽”判断合规，系统已忽略这些孤立安全帽。"
        )
    if no_helmet and helmet and max(helmet, no_helmet) / max(1, min(helmet, no_helmet)) > 2:
        notes.append("helmet / no_helmet 类别不平衡，建议在训练时关注这类别 AP 与召回率。")
    if not notes:
        notes.append("数据分布未发现明显问题，可按 baseline → 评估 → 难例分析 → 调参的流程迭代。")
    return notes


def _markdown_report(results: dict[str, Any]) -> str:
    dataset = results["dataset"]
    lines = [
        "# Construction-PPE 数据集分析",
        "",
        f"- YAML: `{dataset['yaml']}`",
        f"- 根目录: `{dataset['root']}`",
        f"- 类别数: {dataset['class_count']}",
        f"- 类别: {', '.join(dataset['classes'])}",
        "",
        "## 划分统计",
        "",
        "| Split | 图片 | 有标注 | 缺失标注 | 框数 | 非法行 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, stats in results["splits"].items():
        lines.append(
            f"| {split} | {stats['images']} | {stats['labeled_images']} | "
            f"{stats['missing_labels']} | {stats['boxes']} | {stats['malformed_lines']} |"
        )
    lines.extend(["", "## 类别分布", "", "| 类别 | 框数 | 出现图片数 |", "| --- | ---: | ---: |"])
    images_with = defaultdict(int)
    for stats in results["splits"].values():
        for name, count in stats["images_with_class"].items():
            images_with[name] += count
    for name, count in results["totals"]["class_counts"].items():
        lines.append(f"| {name} | {count} | {images_with[name]} |")
    lines.extend([
        "",
        "## GT 人员合规关联",
        "",
        f"- Person 实例: {sum(results['totals']['gt_person_status'].values())}",
        f"- 已佩戴: {results['totals']['gt_person_status'].get('compliant', 0)}",
        f"- 未佩戴: {results['totals']['gt_person_status'].get('violation', 0)}",
        f"- 待确认: {results['totals']['gt_person_status'].get('unknown', 0)}",
        f"- 未关联安全帽: {results['totals']['gt_unmatched_helmet']}",
        f"- 有人但无任何头部证据的图片: "
        f"{results['totals']['images_with_person_but_no_head_evidence']}",
        "",
        "## 建议",
        "",
    ])
    lines.extend(f"- {note}" for note in results["recommendations"])
    lines.append("")
    return "\n".join(lines)
