#!/usr/bin/env python3
"""安全帽检测命令行入口。

示例:
    python main.py --source data/sample.jpg --model models/best.pt --show
    python main.py --source data/demo.mp4
    python main.py --source 0 --show
"""

from helmet_detection.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
