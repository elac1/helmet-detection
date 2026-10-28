"""模型评估：标准检测指标 + 业务层人员合规指标。"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from .association import PersonHelmetAssociator
from .config import PROJECT_ROOT, SystemConfig, load_config
from .dataset import load_dataset_spec, load_ground_truth, iter_split
from .detector import YoloDetector
from .types import ComplianceStatus, Detection, iou


def metrics_to_dict(metrics: Any) -> dict[str, Any]:
    """兼容不同 Ultralytics 版本，尽量完整地导出检测指标。"""
    box = getattr(metrics, "box", metrics)
    data: dict[str, Any] = {
        "mAP50-95": float(getattr(box, "map", float("nan"))),
        "mAP50": float(getattr(box, "map50", float("nan"))),
        "mAP75": float(getattr(box, "map75", float("nan"))),
        "precision": float(getattr(box, "mp", float("nan"))),
        "recall": float(getattr(box, "mr", float("nan"))),
    }
    try:
        data["f1"] = 2 * data["precision"] * data["recall"] / (data["precision"] + data["recall"])
    except ZeroDivisionError:
        data["f1"] = 0.0

    names = getattr(metrics, "names", {}) or {}
    ap_class_index = list(getattr(box, "ap_class_index", []))
    per_class: dict[str, dict[str, float]] = {}
    for position, class_id in enumerate(ap_class_index):
        name = str(names.get(int(class_id), class_id))
        entry = {}
        for key, attr in (("precision", "p"), ("recall", "r"), ("f1", "f1"),
                          ("mAP50", "ap50"), ("mAP50-95", "ap")):
            values = getattr(box, attr, None)
            if values is None:
                continue
            try:
                entry[key] = float(values[position])
            except (IndexError, TypeError, ValueError):
                continue
        per_class[name] = entry
    data["per_class"] = per_class
    data["classes_evaluated"] = [str(names.get(int(c), c)) for c in ap_class_index]
    return data


def run_validation(
    model_path: str | Path,
    data: str | Path,
    split: str = "val",
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    conf: float = 0.001,
    iou: float = 0.6,
    project: str | Path = "runs/ppe_eval",
    name: str | None = None,
    plots: bool = True,
) -> dict[str, Any]:
    model = YOLO(str(model_path))
    name = name or f"{split}_{time.strftime('%Y%m%d_%H%M%S')}"
    metrics = model.val(
        data=str(data),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou=iou,
        plots=plots,
        project=str(project),
        name=name,
        verbose=True,
    )
    result = metrics_to_dict(metrics)
    result.update({
        "model": str(model_path),
        "data": str(data),
        "split": split,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "save_dir": str(getattr(metrics, "save_dir", Path(project) / name)),
    })
    out_dir = Path(result["save_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "detection_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


# --------------------------------------------------------------------------- #
# 业务层：人员合规状态评估
# --------------------------------------------------------------------------- #
@dataclass
class PairMatch:
    gt: Any
    pred: Any


@dataclass
class ComplianceMetrics:
    images: int = 0
    gt_persons: int = 0
    pred_persons: int = 0
    matched_persons: int = 0
    unmatched_gt: int = 0
    unmatched_pred: int = 0
    gt_unknown: int = 0
    pred_unknown: int = 0
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    per_image: list[dict[str, Any]] = field(default_factory=list)

    def _count(self, key: str) -> int:
        return self.confusion.get("violation", {}).get(key, 0)

    def to_dict(self) -> dict[str, Any]:
        labels = [s.value for s in ComplianceStatus]
        confusion = {
            g: {p: self.confusion.get(g, {}).get(p, 0) for p in labels}
            for g in labels
        }
        total = sum(sum(row.values()) for row in confusion.values())
        correct = sum(confusion[l][l] for l in labels)
        known_total = sum(confusion[g][p] for g in ("violation", "compliant") for p in labels)
        known_correct = sum(confusion[g][g] for g in ("violation", "compliant"))

        def prf(label: str) -> dict[str, float | None]:
            tp = confusion[label][label]
            fp = sum(confusion[g][label] for g in labels if g != label)
            fn = sum(confusion[label][p] for p in labels if p != label)
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            f1 = (2 * precision * recall / (precision + recall)
                  if precision is not None and recall is not None and precision + recall else None)
            return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

        return {
            "images": self.images,
            "gt_persons": self.gt_persons,
            "pred_persons": self.pred_persons,
            "matched_persons": self.matched_persons,
            "unmatched_gt": self.unmatched_gt,
            "unmatched_pred": self.unmatched_pred,
            "pred_unknown_rate": (self.pred_unknown / self.pred_persons if self.pred_persons else None),
            "status_accuracy": correct / total if total else None,
            "status_accuracy_excluding_gt_unknown": known_correct / known_total if known_total else None,
            "violation": prf("violation"),
            "compliant": prf("compliant"),
            "confusion_matrix": confusion,
            "note": "匹配基于 Person 框 IoU>=0.5；未匹配人员不参与状态准确率计算。",
        }


def _greedy_person_match(gt_persons: list[Any], pred_persons: list[Any], iou_threshold: float = 0.5) -> list[PairMatch]:
    candidates: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_persons):
        for pi, pred in enumerate(pred_persons):
            score = iou(gt.person.xyxy, pred.person.xyxy)
            if score >= iou_threshold:
                candidates.append((score, gi, pi))
    candidates.sort(reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    pairs: list[PairMatch] = []
    for _, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        pairs.append(PairMatch(gt=gt_persons[gi], pred=pred_persons[pi]))
    return pairs


def evaluate_compliance(
    model_path: str | Path,
    data: str | Path,
    split: str = "val",
    conf: float | None = None,
    iou: float | None = None,
    imgsz: int | None = None,
    device: str | None = None,
    max_images: int | None = None,
    output: str | Path | None = None,
    config: SystemConfig | None = None,
) -> dict[str, Any]:
    cfg = config or load_config()
    cfg.model.path = str(model_path)
    if conf is not None:
        cfg.model.conf = conf
    if iou is not None:
        cfg.model.iou = iou
    if imgsz is not None:
        cfg.model.imgsz = imgsz
    if device is not None:
        cfg.model.device = device

    detector = YoloDetector(cfg.model, {
        "person": cfg.classes.person,
        "helmet": cfg.classes.helmet,
        "no_helmet": cfg.classes.no_helmet,
    })
    associator = PersonHelmetAssociator(cfg.association_config())
    spec = load_dataset_spec(data)
    images = list(iter_split(spec, split, max_images=max_images))
    if not images:
        raise RuntimeError(f"{split} 划分没有图片，请检查数据集 YAML")

    metrics = ComplianceMetrics()
    metrics.confusion = {s.value: {p.value: 0 for p in ComplianceStatus} for s in ComplianceStatus}
    t0 = time.time()

    for image_path in tqdm(images, desc=f"合规评估 {split}", unit="img"):
        import cv2

        data = np.fromfile(str(image_path), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if frame is None:
            continue
        height, width = frame.shape[:2]
        gt_detections = load_ground_truth(image_path, spec, (width, height))
        pred_detections = detector.detect(frame)
        gt = associator.analyze(gt_detections, frame_size=(width, height))
        pred = associator.analyze(pred_detections, frame_size=(width, height))

        pairs = _greedy_person_match(gt.assessments, pred.assessments)
        metrics.images += 1
        metrics.gt_persons += len(gt.assessments)
        metrics.pred_persons += len(pred.assessments)
        metrics.matched_persons += len(pairs)
        metrics.unmatched_gt += len(gt.assessments) - len(pairs)
        metrics.unmatched_pred += len(pred.assessments) - len(pairs)
        metrics.gt_unknown += sum(a.status == ComplianceStatus.UNKNOWN for a in gt.assessments)
        metrics.pred_unknown += sum(a.status == ComplianceStatus.UNKNOWN for a in pred.assessments)

        for pair in pairs:
            metrics.confusion[pair.gt.status.value][pair.pred.status.value] += 1

        metrics.per_image.append({
            "image": str(image_path),
            "gt_persons": len(gt.assessments),
            "pred_persons": len(pred.assessments),
            "matched": len(pairs),
            "gt_compliant": gt.compliant,
            "gt_violations": gt.violations,
            "gt_unknown": gt.unknown,
            "pred_compliant": pred.compliant,
            "pred_violations": pred.violations,
            "pred_unknown": pred.unknown,
        })

    result = metrics.to_dict()
    result.update({
        "model": str(model_path),
        "data": str(spec.yaml_path),
        "split": split,
        "conf": cfg.model.conf,
        "iou": cfg.model.iou,
        "imgsz": cfg.model.imgsz,
        "device": detector.device,
        "elapsed_seconds": round(time.time() - t0, 2),
        "avg_inference_ms": detector.average_inference_ms(),
    })

    out_dir = Path(output) if output else PROJECT_ROOT / "outputs" / "compliance_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"compliance_{split}_{stamp}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out_dir / f"compliance_{split}_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.per_image[0]) if metrics.per_image else ["image"])
        writer.writeheader()
        writer.writerows(metrics.per_image)
    result["artifacts"] = {"json": str(json_path), "csv": str(csv_path)}
    # 同步写入 JSON，包含 artifacts
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
