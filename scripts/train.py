#!/usr/bin/env python3
"""训练 YOLOv8 安全帽检测模型。

示例:
    python scripts/train.py --epochs 100 --batch 16 --device 0
    python scripts/train.py --model yolov8s.pt --epochs 150 --imgsz 640 --name yolov8s
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from helmet_detection.evaluation import run_validation  # noqa: E402


def resolve_model(value: str) -> str:
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())
    local = Path.home() / "yolo" / path.name
    if local.exists():
        print(f"使用本地预训练权重: {local}")
        return str(local)
    return value  # 交给 Ultralytics 下载


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PROJECT_ROOT / "datasets" / "construction-ppe" / "data.yaml"))
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--cache", action="store_true", help="缓存图片到内存/磁盘，加速训练")
    parser.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "ppe"))
    parser.add_argument("--name", default="yolov8n_baseline")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-test", action="store_true", help="训练后不在 test 划分上评估")
    parser.add_argument("--copy-to", default=str(PROJECT_ROOT / "models" / "best.pt"),
                        help="将 best.pt 复制到固定路径，供推理/评估使用")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"数据集 YAML 不存在: {data_path}\n"
              "请先运行 scripts/download_file.py 与 scripts/prepare_dataset.py", file=sys.stderr)
        return 1
    raw = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    names = raw.get("names") or {}
    print(f"数据: {data_path}\n类别({len(names)}): {names}")

    from ultralytics import YOLO

    model_path = resolve_model(args.model)
    device = resolve_device(args.device)
    print(f"模型: {model_path}\n设备: {device}\n图片尺寸: {args.imgsz} | batch: {args.batch}")

    model = YOLO(model_path)
    t0 = time.time()
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cache=args.cache,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        resume=args.resume,
        pretrained=True,
        amp=device != "cpu",
        plots=True,
        val=True,
        save=True,
        verbose=True,
        close_mosaic=10,
    )
    elapsed = time.time() - t0

    trainer = model.trainer
    save_dir = Path(getattr(trainer, "save_dir", Path(args.project) / args.name))
    best = Path(getattr(trainer, "best", save_dir / "weights" / "best.pt"))
    if not best.exists():
        best = save_dir / "weights" / "best.pt"
    if not best.exists():
        print("训练结束但未找到 best.pt", file=sys.stderr)
        return 2

    copy_to = Path(args.copy_to)
    copy_to.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, copy_to)
    print(f"best.pt: {best}\n已复制到: {copy_to}")

    train_metrics: dict[str, Any] = {}
    raw_metrics = getattr(trainer, "metrics", {}) or {}
    for key, value in raw_metrics.items():
        try:
            train_metrics[key] = float(value)
        except (TypeError, ValueError):
            continue

    test_metrics = None
    if not args.no_test:
        print("在 test 划分上评估...")
        test_metrics = run_validation(
            model_path=copy_to,
            data=data_path,
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=str(save_dir / "test_eval"),
            name="test",
        )

    info = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": model_path,
        "dataset": str(data_path),
        "classes": names,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "elapsed_seconds": round(elapsed, 1),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "best_weights": str(copy_to),
        "run_dir": str(save_dir),
        "args": vars(args),
    }
    info_path = copy_to.parent / "model_info.json"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (save_dir / "run_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"训练信息: {info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
