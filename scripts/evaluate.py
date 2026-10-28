#!/usr/bin/env python3
"""评估安全帽模型：检测 mAP + 人员级合规准确率/违规召回率。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from helmet_detection.config import load_config  # noqa: E402
from helmet_detection.evaluation import evaluate_compliance, run_validation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models" / "best.pt"))
    parser.add_argument("--data", default=str(PROJECT_ROOT / "datasets" / "construction-ppe" / "data.yaml"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.35, help="合规评估使用的置信度")
    parser.add_argument("--iou", type=float, default=0.45, help="合规评估使用的 NMS IoU")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--skip-detection", action="store_true", help="跳过 mAP 评估")
    parser.add_argument("--skip-compliance", action="store_true", help="跳过人员合规评估")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "evaluation"))
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"模型不存在: {args.model}\n请先运行 scripts/train.py", file=sys.stderr)
        return 1

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"

    result: dict = {"model": args.model, "data": args.data, "split": args.split}
    if not args.skip_detection:
        print("=== 检测指标（mAP） ===")
        det = run_validation(
            model_path=args.model,
            data=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=str(out / "detection"),
            name=args.split,
        )
        result["detection"] = det
        print(json.dumps({k: det[k] for k in ("mAP50-95", "mAP50", "precision", "recall")},
                         ensure_ascii=False, indent=2))
        print("各类别 AP50:")
        for name, values in det.get("per_class", {}).items():
            print(f"  {name:12s} AP50={values.get('mAP50', float('nan')):.4f} "
                  f"P={values.get('precision', float('nan')):.4f} R={values.get('recall', float('nan')):.4f}")

    if not args.skip_compliance:
        print("\n=== 人员级合规指标 ===")
        comp = evaluate_compliance(
            model_path=args.model,
            data=args.data,
            split=args.split,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            max_images=args.max_images,
            output=out / "compliance",
            config=cfg,
        )
        result["compliance"] = comp
        print(f"匹配人员: {comp['matched_persons']} / GT {comp['gt_persons']} "
              f"/ 预测 {comp['pred_persons']}")
        print(f"状态准确率: {_fmt(comp['status_accuracy'])} "
              f"(排除 GT unknown: {_fmt(comp['status_accuracy_excluding_gt_unknown'])})")
        for label in ("violation", "compliant"):
            m = comp[label]
            print(f"{label:10s} P={_fmt(m['precision'])} R={_fmt(m['recall'])} "
                  f"F1={_fmt(m['f1'])} (tp={m['tp']}, fp={m['fp']}, fn={m['fn']})")
        print(f"预测 unknown 比例: {_fmt(comp['pred_unknown_rate'])}")
        print(f"混淆矩阵: {json.dumps(comp['confusion_matrix'], ensure_ascii=False)}")

    summary_path = out / "evaluation_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n评估汇总: {summary_path}")
    return 0


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
