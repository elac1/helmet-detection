# 安全帽佩戴检测系统（Construction-PPE / YOLOv8）

面向工地安全的端到端检测系统：支持图片、视频文件和摄像头实时检测，输出检测框、类别、置信度、人员合规统计、违规提醒与结构化结果，并提供训练与评估脚本。

> **核心设计**：检测到安全帽 ≠ 人员正确佩戴安全帽。系统以 `Person` 检测框为锚点，只有落在人员头部区域内的 `helmet` / `no_helmet` 证据才会影响该人员的合规状态；画面中孤立的安全帽、拿在手里的安全帽、其他人员的安全帽都不会被误算为合规。详见 [人员—安全帽关联](#人员安全帽关联为什么重要)。

---

## 1. 功能清单

| 编号 | 功能 | 实现 |
| --- | --- | --- |
| 1 | 图片安全帽检测 | `python main.py --source xxx.jpg` |
| 2 | 视频文件检测 | `python main.py --source xxx.mp4` |
| 3 | 摄像头实时检测 | `python main.py --source 0 --show` |
| 4 | 判断人员是否佩戴安全帽 | `helmet_detection/association.py` |
| 5 | 显示检测框、类别、置信度 | `helmet_detection/render.py` |
| 6 | 统计正常佩戴/违规/待确认人数 | `helmet_detection/stats.py`，逐帧 CSV |
| 7 | 未佩戴提醒 | 控制台告警、画面红色横幅、违规截图、可选 webhook |
| 8 | 保存检测结果 | 标注图片/视频、CSV、JSON 汇总、违规截图 |
| 9 | 模型评估与优化 | `scripts/evaluate.py`、mAP + 人员级合规指标、训练调参脚本 |

默认使用 **YOLOv8n**，训练数据为 Ultralytics **Construction-PPE**。

---

## 2. 数据集：Construction-PPE

来源：<https://docs.ultralytics.com/datasets/detect/construction-ppe/>

- 共 **1,416 张**图片：train 1,132 / val 143 / test 141
- **11 类**：

| 类别 | 含义 |
| --- | --- |
| `helmet` | 安全帽（已佩戴） |
| `gloves` | 手套 |
| `vest` | 反光马甲 |
| `boots` | 安全鞋 |
| `goggles` | 护目镜 |
| `none` | 其他/无 |
| `Person` | 人员 |
| `no_helmet` | 未佩戴安全帽 |
| `no_goggle` | 未佩戴护目镜 |
| `no_gloves` | 未佩戴手套 |
| `no_boots` | 未穿安全鞋 |

- YOLO 格式标注：`class_id cx cy w h`（归一化到 0~1）
- 官方 YAML：<https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/construction-ppe.yaml>
- 官方压缩包约 178.4 MB，首次训练会自动下载；本仓库提供多线程下载与解压脚本。

### 数据集先分析再训练

```bash
python scripts/analyze_dataset.py --data datasets/construction-ppe/data.yaml
```

输出 `outputs/dataset_analysis/dataset_analysis.json` 与 `.md`，包括：

- 各划分图片数、缺失标注、非法标注行
- 每类框数、出现图片数、每图框数分布、分辨率分布
- **GT 层面**的人员—安全帽关联结果（已佩戴 / 未佩戴 / 待确认）
- 孤立安全帽（有安全帽但无人员）图片数
- 类别不平衡、数据量、标注覆盖等优化建议

> 先看分析结果再决定训练策略，而不是直接跑 100 epochs。

---

## 3. 人员—安全帽关联（为什么重要）

简单做法是“画面里出现 helmet 就算有人合规”，这在工地场景会严重误判：

- 工人把安全帽拿在手里 / 放在地上；
- 画面里只有其他工人的安全帽；
- 安全帽属于已经离开画面的人；
- 模型漏检了 `Person`，但仍检测到安全帽。

本系统的关联流程：

1. 以 `Person` 框顶部 `head_height_ratio`（默认 38%）、左右内缩 `head_width_shrink`（默认 12%）得到头部区域；
2. 只接受中心位于头部区域附近、与头部区域有足够 IoU/覆盖、且位于人员上半身的 `helmet` / `no_helmet` 框作为证据；
3. 贪心匹配：按匹配分数从高到低，一名人员最多绑定一个头部证据，一个证据最多绑定一名人员；
4. 人员没有匹配到头部证据时，默认状态为 `unknown`，而不是直接判违规；
5. 贴边/出画/过小的人员也标为 `unknown`，避免把遮挡误报为违规；
6. 未关联到任何人员的 `helmet` 会在画面上标为“未关联”，但不计入合规人数；
7. 同时匹配到 `helmet` 和 `no_helmet`（冲突）时默认 `unknown`。

状态定义：

| 状态 | 含义 |
| --- | --- |
| `compliant` | 头部区域匹配到 `helmet` |
| `violation` | 头部区域匹配到 `no_helmet` |
| `unknown` | 无头部证据、出画、太小、证据冲突等，无法确认 |

合规率 = `compliant / (compliant + violation)`，`unknown` 不参与分母。

可通过配置切换严格模式：

```yaml
association:
  unmatched_policy: violation   # 人员头部可见但无任何头部证据时判违规（会增加误报）
  conflict_policy: violation
```

---

## 4. 项目结构

```
helmet-detection/
├── configs/default.yaml          # 阈值、关联、渲染、告警、输出配置
├── helmet_detection/
│   ├── association.py            # ★ 人员—安全帽关联
│   ├── detector.py               # Ultralytics YOLO 封装
│   ├── pipeline.py               # 图片/视频/摄像头流程
│   ├── render.py                 # 框、类别、置信度、统计面板、告警横幅
│   ├── stats.py                  # 帧统计、违规事件去重、运行汇总
│   ├── alerts.py                 # 控制台/横幅/webhook 提醒
│   ├── writers.py                # 图片/视频/CSV/JSON/截图
│   ├── dataset.py                # YOLO 数据集解析
│   ├── dataset_analysis.py       # 数据集与 GT 关联分析
│   ├── evaluation.py             # mAP + 人员级合规指标
│   └── cli.py                    # 检测命令行
├── scripts/
│   ├── download_file.py          # 多线程分块下载
│   ├── prepare_dataset.py        # 解压并生成 data.yaml
│   ├── analyze_dataset.py        # 数据集分析
│   ├── train.py                  # YOLOv8 训练
│   └── evaluate.py               # 检测 + 合规评估
├── tests/test_association.py     # 13 个关联/事件/渲染测试
├── main.py                       # 检测入口
└── requirements.txt
```

---

## 5. 环境安装

```bash
cd helmet-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

已验证环境：Python 3.10、ultralytics 8.4.190、torch 2.14.1+cu130、OpenCV 5.0、RTX 4060 Laptop 8GB。

CUDA 是否可用：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 6. 准备数据集

```bash
# 1) 下载（网络慢时支持断点/多线程，默认 16 连接）
python scripts/download_file.py \
  https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip \
  datasets/construction-ppe.zip --connections 16

# 2) 校验、解压、生成绝对路径 data.yaml
python scripts/prepare_dataset.py

# 3) 先分析再训练
python scripts/analyze_dataset.py
```

也可以让 Ultralytics 自动下载（需要能稳定访问 GitHub）：

```bash
yolo detect train data=construction-ppe.yaml model=yolov8n.pt epochs=100 imgsz=640
```

---

## 7. 训练

```bash
# baseline：YOLOv8n，100 epochs
python scripts/train.py --epochs 100 --batch 16 --device 0

# 训练完成后自动：
# - 复制 best.pt 到 models/best.pt
# - 在 test 划分评估并写 models/model_info.json
```

常用参数：

```bash
python scripts/train.py \
  --model /home/elac/yolo/yolov8n.pt \  # 无外网时使用本地预训练权重
  --data datasets/construction-ppe/data.yaml \
  --epochs 100 --imgsz 640 --batch 16 --device 0 \
  --patience 25 --cache --name yolov8n_baseline
```

> 小数据集优先 `yolov8n` + 迁移学习 + 数据增强；只有在验证集指标饱和、且误报主要来自模型容量时才换 `yolov8s/m`。换更大模型后需重新评估推理延迟是否满足摄像头实时性。

---

## 8. 运行检测

```bash
# 图片
python main.py --source data/site.jpg --model models/best.pt --show

# 视频文件（默认保存标注视频、逐帧 CSV、summary.json）
python main.py --source data/site.mp4 --model models/best.pt

# 摄像头实时预览（默认不录制，按 Q/ESC 退出）
python main.py --source 0 --model models/best.pt --show

# 摄像头同时录制
python main.py --source 0 --show --record

# 严格模式 + 其他 PPE 类别 + 头部区域调试
python main.py --source data/site.mp4 --unmatched-policy violation --show-other --head-region
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--conf` / `--iou` / `--imgsz` | 检测阈值与推理尺寸 |
| `--max-width` | 缩放最大宽度，0 表示原图 |
| `--max-frames` | 只处理前 N 帧（调试） |
| `--show` / `--no-show` | 是否打开窗口 |
| `--record` / `--no-record` | 摄像头是否录制 |
| `--unmatched-policy` | `unknown`（默认，低误报）或 `violation`（严格） |
| `--cooldown` | 重复告警冷却秒数 |
| `--no-alert` / `--no-snapshot` | 关闭提醒 / 关闭违规截图 |
| `--show-other` | 显示手套、马甲等类别 |
| `--head-region` | 画出关联用头部区域 |
| `--save-json` | 保存逐帧 JSONL |

> **注意**：COCO 预训练的 `yolov8n.pt` 没有 helmet / no_helmet / Person-PPE 类别，不能直接用于合规判断。必须使用本数据集微调后的 `models/best.pt`。

---

## 9. 输出结果

每次运行创建 `outputs/<source>_<timestamp>/`：

```
outputs/site_20261027_213000/
├── site_detected.jpg / site_detected.mp4   # 标注结果
├── stats.csv                              # 逐帧：人数、合规、违规、unknown、合规率
├── summary.json                           # 参数、耗时、汇总、聚合指标、产物路径
├── frames.jsonl                           # 可选：逐帧检测详情
└── violations/                            # 新违规事件截图
    ├── site_e1_f123_....jpg
    └── violations.json
```

- 视频/摄像头：`total_violation_events` 通过空间重叠 + TTL 去重，避免同一违规人员每帧重复计数；
- `alert_triggered` 受冷却时间与事件 TTL 控制；
- 图片的 `summary.json` 还包含每个人与安全帽证据的框、匹配分数、判定原因。

---

## 10. 评估与优化

### 10.1 标准检测指标

```bash
python scripts/evaluate.py --split test --skip-compliance
```

- mAP50-95 / mAP50 / mAP75 / Precision / Recall / F1
- 每类 AP50、P、R（重点看 `helmet` 与 `no_helmet`）
- Ultralytics 自动生成 PR 曲线、混淆矩阵、验证集预测图

### 10.2 业务层人员合规指标

```bash
python scripts/evaluate.py --split test --conf 0.35
```

除了 mAP，还会评估真正的业务目标：

- 将预测人员与 GT 人员按 Person 框 IoU ≥ 0.5 匹配；
- 分别用同一套关联逻辑计算 GT 与预测的 `compliant / violation / unknown`；
- 输出 3×3 混淆矩阵、状态准确率、违规精确率/召回率/F1、预测 unknown 比例；
- 输出每张图片的 GT/预测人数与状态统计 CSV。

意义：mAP 高不等于违规召回率高。实际部署应优先保证 **违规召回率**，再通过 `unmatched_policy`、置信度阈值和头部区域参数控制误报。

### 10.3 优化路线（按优先级）

1. **先修数据与关联**：查看数据集分析中 unknown 比例、`Person` 标注覆盖、孤立安全帽比例；调整 `head_height_ratio` / `head_width_shrink` / 匹配阈值。
2. **阈值扫描**：在 val 上扫描 `conf`、`min_helmet_confidence`、`min_no_helmet_confidence`，以违规 F1 或“违规召回 ≥ 95% 下的最低误报”为目标，而不是只看 mAP。
3. **难例分析**：针对 no_helmet 漏检、背影、遮挡、远距离小目标、夜间逆光做切片评估；可增加相应数据增强或补采数据。
4. **模型升级**：在实时性允许下尝试 `yolov8s`，对比人物与 no_helmet 的召回和延迟；导出 ONNX/TensorRT 提升摄像头帧率。
5. **多目标跟踪**：用 ByteTrack 的 track_id 替换当前违规事件的空间去重，得到更准确的“独立违规人数”和停留时长。
6. **关联升级**：引入姿态/头部关键点（YOLOv8-pose）精确估计头部框；对“安全帽戴在头上但未系下颚带”等更细粒度违规，增加专用分类头或规则。
7. **业务闭环**：webhook/邮件/声光报警、按区域/班组统计、违规趋势报表、误报人工复核。

---

## 11. 已知限制

- `unknown` 不是“合规”，也不是“违规”，代表证据不足；报表中应单独统计，不能并入合规率。
- 当前违规事件去重是空间 + TTL 启发式，不是真正的 ReID；人员快速移动或换位时可能重复计数，建议生产环境接跟踪器。
- 模型对远距离小目标、严重遮挡、逆光、安全帽颜色与背景接近等场景仍可能漏检。`no_helmet` 的召回率是安全系统的关键指标。
- Construction-PPE 数据集采用 AGPL-3.0，商用前请确认许可证与数据授权。
- 摄像头实时性取决于分辨率、`imgsz`、模型大小和硬件；RTX 4060 上 `yolov8n@640` 通常可实时，CPU 环境会明显变慢。

---

## 12. 测试

```bash
python -m pytest -q tests
```

覆盖：

- 安全帽在头部区域 → compliant
- no_helmet 在头部区域 → violation
- 安全帽拿在腰部 → unknown，不计合规
- 一个安全帽不会被多名人员重复使用
- 低置信度安全帽被忽略
- 严格模式、贴边人员、helmet/no_helmet 冲突
- 画面只有安全帽没有人员 → 不产生合规
- 违规事件跨帧去重、多人事件、渲染形状
