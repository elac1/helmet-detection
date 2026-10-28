"""命令行入口：图片 / 视频文件 / 摄像头实时检测。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config
from .detector import ModelLoadError
from .pipeline import HelmetDetectionSystem, IMAGE_SUFFIXES
from .writers import safe_stem

logger = logging.getLogger("helmet_detection")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="helmet-detection",
        description="安全帽佩戴检测系统（YOLOv8 + 人员-安全帽关联）",
    )
    parser.add_argument("-s", "--source", required=True,
                        help="图片来源/视频文件/摄像头编号(如 0) 或 RTSP 地址")
    parser.add_argument("-m", "--model", default=None,
                        help="YOLO 权重路径，默认读取 configs/default.yaml")
    parser.add_argument("-c", "--config", default=None, help="配置文件路径")
    parser.add_argument("--conf", type=float, default=None, help="检测置信度阈值")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=None, help="推理尺寸")
    parser.add_argument("--device", default=None, help="auto / cpu / 0 / cuda:0")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理帧数（调试）")
    parser.add_argument("--max-width", type=int, default=None, help="缩放最大宽度，0 表示不缩放")
    parser.add_argument("--output", type=Path, default=None,
                        help="标注图片/视频输出路径；默认写入 outputs/ 运行目录")
    parser.add_argument("--show", dest="show", action="store_true", default=None, help="显示窗口")
    parser.add_argument("--no-show", dest="show", action="store_false", help="不显示窗口")
    parser.add_argument("--record", dest="record", action="store_true", default=None,
                        help="摄像头也录制标注视频")
    parser.add_argument("--no-record", dest="record", action="store_false", help="不录制视频")
    parser.add_argument("--unmatched-policy", choices=["unknown", "violation"], default=None,
                        help="人员无头部证据时的策略")
    parser.add_argument("--no-alert", action="store_true", help="关闭违规提醒")
    parser.add_argument("--cooldown", type=float, default=None, help="重复告警冷却秒数")
    parser.add_argument("--no-snapshot", action="store_true", help="不保存违规截图")
    parser.add_argument("--show-other", action="store_true", help="显示手套/马甲等其他类别")
    parser.add_argument("--head-region", action="store_true", help="显示头部关联区域（调试）")
    parser.add_argument("--save-json", action="store_true", help="保存逐帧 JSONL")
    parser.add_argument("--codec", default=None, help="视频编码 fourcc，默认 mp4v")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def apply_overrides(cfg, args: argparse.Namespace) -> None:
    if args.model:
        cfg.model.path = args.model
    if args.conf is not None:
        cfg.model.conf = args.conf
    if args.iou is not None:
        cfg.model.iou = args.iou
    if args.imgsz is not None:
        cfg.model.imgsz = args.imgsz
    if args.device:
        cfg.model.device = args.device
    if args.max_width is not None:
        cfg.output.max_width = args.max_width or None
    if args.unmatched_policy:
        cfg.association.unmatched_policy = args.unmatched_policy
    if args.no_alert:
        cfg.alert.enabled = False
    if args.cooldown is not None:
        cfg.alert.cooldown_seconds = args.cooldown
    if args.no_snapshot:
        cfg.alert.snapshot_on_new_event = False
    if args.show_other:
        cfg.render.show_other_detections = True
    if args.head_region:
        cfg.render.show_head_region = True
    if args.save_json:
        cfg.output.save_json = True
    if args.codec:
        cfg.output.video_codec = args.codec


def parse_source(value: str) -> str | int:
    if value.isdigit():
        return int(value)
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")
    return str(path)


def _resolve_output_path(output: Path | None, source: str | int, is_image: bool) -> Path | None:
    if output is None:
        return None
    if output.is_dir():
        suffix = ".jpg" if is_image else ".mp4"
        return output / f"{safe_stem(str(source))}_detected{suffix}"
    if output.suffix:
        return output
    return output.with_suffix(".jpg" if is_image else ".mp4")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    apply_overrides(cfg, args)

    try:
        source = parse_source(args.source)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    is_image = isinstance(source, int) is False and Path(source).suffix.lower() in IMAGE_SUFFIXES
    output_path = _resolve_output_path(args.output, source, is_image)
    show = args.show

    system = None
    try:
        system = HelmetDetectionSystem(cfg, source=str(source))
        for warning in system.validate_model():
            logger.warning(warning)
        logger.info("模型: %s | 设备: %s | 类别: %s",
                    system.detector.model_path, system.detector.device,
                    ", ".join(system.detector.class_names))

        if is_image:
            if show is None:
                show = False
            summary = system.run_image(Path(source), output_path=output_path, show=show)
        else:
            is_camera = isinstance(source, int)
            record = args.record
            if record is None:
                # 摄像头默认只预览，不录制；视频文件默认保存标注视频
                record = False if is_camera else cfg.output.save_annotated
            if show is None:
                show = True if is_camera else cfg.output.display
            summary = system.run_video(
                source=source,
                output_path=output_path,
                show=show,
                max_frames=args.max_frames,
                record=record,
            )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    except KeyboardInterrupt:
        logger.info("用户中断，已保存当前结果")
        return 130
    except ModelLoadError as exc:
        logger.error("%s", exc)
        return 3
    except Exception as exc:  # noqa: BLE001
        logger.exception("运行失败: %s", exc)
        return 1
    finally:
        if system is not None:
            try:
                system.close()
            except Exception:  # noqa: BLE001
                pass
