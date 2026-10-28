#!/usr/bin/env python3
"""解压并准备 Construction-PPE 数据集，生成绝对路径的 data.yaml。"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_NAMES = {
    0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles",
    5: "none", 6: "Person", 7: "no_helmet", 8: "no_goggle",
    9: "no_gloves", 10: "no_boots",
}


def find_dataset_root(base: Path) -> Path:
    for candidate in [base, *[p for p in base.iterdir() if p.is_dir()]]:
        if (candidate / "images").is_dir() and (candidate / "labels").is_dir():
            return candidate
    raise RuntimeError(f"解压目录中未找到 images/ 与 labels/: {base}")


def read_existing_names(base: Path) -> dict[int, str]:
    for path in base.rglob("data.yaml"):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            names = raw.get("names")
        except Exception:  # noqa: BLE001
            continue
        if isinstance(names, list):
            return {i: str(n) for i, n in enumerate(names)}
        if isinstance(names, dict) and names:
            return {int(k): str(v) for k, v in names.items()}
    return dict(OFFICIAL_NAMES)


def write_data_yaml(dest: Path, names: dict[int, str]) -> Path:
    yaml_path = dest / "data.yaml"
    lines = [
        "# 由 scripts/prepare_dataset.py 生成",
        f"path: {dest}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"  {i}: {names[i]}" for i in sorted(names))
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path,
                        default=PROJECT_ROOT / "datasets" / "construction-ppe.zip")
    parser.add_argument("--dest", type=Path,
                        default=PROJECT_ROOT / "datasets" / "construction-ppe")
    parser.add_argument("--force", action="store_true", help="删除已存在的目标目录后重新解压")
    args = parser.parse_args()

    if not args.zip.exists():
        print(f"未找到数据集压缩包: {args.zip}\n"
              f"请先运行: python scripts/download_file.py "
              f"https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip "
              f"{args.zip}", file=sys.stderr)
        return 1

    try:
        with zipfile.ZipFile(args.zip) as zf:
            bad = zf.testzip()
            if bad:
                raise RuntimeError(f"zip 校验失败: {bad}")
            print(f"zip 校验通过，共 {len(zf.namelist())} 个条目")
    except zipfile.BadZipFile as exc:
        print(f"数据集压缩包不完整或损坏（可能仍在下载）: {exc}", file=sys.stderr)
        return 2

    if args.dest.exists():
        if not args.force:
            print(f"目标目录已存在: {args.dest}（使用 --force 重新解压）")
            return 0
        shutil.rmtree(args.dest)
    args.dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=args.dest.parent, prefix="ppe_extract_") as tmp:
        tmp_path = Path(tmp)
        print("正在解压...")
        with zipfile.ZipFile(args.zip) as zf:
            zf.extractall(tmp_path)
        source = find_dataset_root(tmp_path)
        names = read_existing_names(source)
        if len(names) != 11:
            names = dict(OFFICIAL_NAMES)
        print(f"数据集根目录: {source}")
        shutil.move(str(source), str(args.dest))

    yaml_path = write_data_yaml(args.dest, names)
    print(f"data.yaml: {yaml_path}")
    for split in ("train", "val", "test"):
        images = list((args.dest / "images" / split).glob("*"))
        labels = list((args.dest / "labels" / split).glob("*.txt"))
        print(f"{split:5s}: images={len(images):4d}, labels={len(labels):4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
