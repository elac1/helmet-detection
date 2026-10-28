#!/usr/bin/env python3
"""分析 Construction-PPE 数据集类别、划分、标注质量与 GT 关联情况。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from helmet_detection.dataset_analysis import analyze_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PROJECT_ROOT / "datasets" / "construction-ppe" / "data.yaml"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "dataset_analysis"))
    parser.add_argument("--max-images", type=int, default=None, help="每个划分最多分析多少张（调试）")
    args = parser.parse_args()

    result = analyze_dataset(args.data, output_dir=args.output, max_images=args.max_images)
    print(json.dumps(result["dataset"], ensure_ascii=False, indent=2))
    print("\n划分:")
    for split, stats in result["splits"].items():
        print(f"  {split:5s} images={stats['images']:4d} boxes={stats['boxes']:5d} "
              f"missing_labels={stats['missing_labels']} malformed={stats['malformed_lines']}")
    print("\n类别分布:")
    for name, count in result["totals"]["class_counts"].items():
        print(f"  {name:12s} {count}")
    print("\nGT 人员-安全帽关联:")
    print(json.dumps(result["totals"]["gt_person_status"], ensure_ascii=False))
    print(f"  孤立安全帽标注: {result['totals']['gt_unmatched_helmet']}")
    print(f"  只有安全帽没有人的图片: {result['totals']['images_with_helmet_but_no_person']}")
    print("\n建议:")
    for note in result["recommendations"]:
        print(f"  - {note}")
    print(f"\n报告: {result['_artifacts']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
