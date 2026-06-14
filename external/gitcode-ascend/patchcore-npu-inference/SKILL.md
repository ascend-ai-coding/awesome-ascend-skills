---
name: external-gitcode-ascend-patchcore-npu-inference
description: PatchCore 工业异常检测在昇腾 NPU 上的端到端推理 Skill。基于 WideResNet-50 骨干网络，在昇腾 800I
  A2 NPU 上实现全链路异常检测推理流水线，零 CPU 回退。特色：BaihuNN（基于 torch.mm 矩阵乘的最近邻搜索，替代 Faiss GPU）、自动环境检测与参数调优、三档加速模式（normal/fast/turbo）。
keywords:
- NPU
- anomaly-detection
- patchcore
- mvtec-ad
- ascend
- 800I-A2
- 工业异常检测
- BaihuNN
- wide-resnet50
original-name: patchcore-npu-inference
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-06-14'
synced-commit: 62f55373f87e10e1d64f3f82f369be625c56b9fc
license: UNKNOWN
---

# PatchCore 昇腾 NPU 工业异常检测推理

基于 **PatchCore** 算法（Roth et al., CVPR 2022），使用 **WideResNet-50** 骨干网络，在 **昇腾 800I A2 NPU** 上实现端到端的工业异常检测推理流水线。

## 触发条件

当用户请求：
- "NPU 工业异常检测"
- "PatchCore 在 NPU 上推理"
- "MVTec-AD 异常检测 NPU"
- "在昇腾 NPU 上跑异常检测"
- "BaihuNN"
- "工业缺陷检测"

## 概述

本 Skill 的核心贡献是 **BaihuNN**——一个基于 `torch.mm` 矩阵乘法的两阶段最近邻搜索引擎，替代原版 Faiss GPU/CUDA 依赖，使整个流水线（从骨干网络推理到最近邻搜索）全程运行在 NPU 上。

### 主要特性

| 特性 | 说明 |
|:-----|:------|
| **全链路 NPU 迁移** | 从骨干网络推理到最近邻搜索全程运行在 NPU 上，零 CPU 回退 |
| **BaihuNN 搜索引擎** | 自研百行级 NN 搜索库，用 `torch.mm` 矩阵乘替代 Faiss GPU/CUDA 依赖 |
| **自动环境检测** | 自动检测 NPU 可用性、CPU 核数、内存/显存，动态调优 batch_size 和采样率 |
| **三档加速模式** | `normal`（保精度）、`fast`（2–3×）、`turbo`（5–8×） |
| **15 类 MVTec-AD 全覆盖** | bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper |

---

## 前置条件

### 1. 环境准备

确保环境中已安装昇腾 NPU 驱动（CANN ≥ 8.5.1）及 PyTorch NPU 插件：

```bash
# 验证 NPU 可用
python -c "import torch; print(torch.npu.is_available())"
# 应输出: True
```

### 2. 安装依赖

```bash
pip install torch torchvision timm faiss-cpu scikit-learn scikit-image tqdm
# 昇腾 NPU 额外安装 torch_npu
pip install torch_npu
```

---

## 快速开始

### 一键运行（全 15 类，自动检测 NPU）

```bash
python scripts/inference.py --data_dir /path/to/mvtec_ad
```

### 仅指定类别调试

```bash
python scripts/inference.py --data_dir /path/to/mvtec_ad --categories bottle cable
```

### 加速模式

```bash
# normal 模式 — 保精度（比赛提交推荐）
python scripts/inference.py --data_dir /path/to/mvtec_ad --mode normal

# fast 模式 — 约 2–3× 加速，精度微降 ~0.5%
python scripts/inference.py --data_dir /path/to/mvtec_ad --mode fast

# turbo 模式 — 约 5–8× 加速，精度降 ~1–2%
python scripts/inference.py --data_dir /path/to/mvtec_ad --mode turbo
```

### CPU 模式（调试/对比）

```bash
python scripts/inference.py --data_dir /path/to/mvtec_ad --device cpu --batch_size 1
```

---

## 工作流程

### 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                   用户输入: --data_dir /path/to/mvtec_ad          │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 1: 环境检测 + 参数自动调优                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐   │
│  │ NPU/CPU  │  │ 显存/内存 │  │ CPU 核数 │  │ batch_size /  │   │
│  │ 自动检测  │→ │ 自动检测  │→ │ 自动检测  │→ │ num_workers   │   │
│  └──────────┘  └──────────┘  └──────────┘  │ 采样率等调优   │   │
│                                             └───────────────┘   │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 2: 逐类别处理（15 类 / 或用户指定部分类别）               │
│                                                                   │
│  对每个类别:                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  Step 1: 加载训练集（仅 good 样本）                        │   │
│  │  Step 2: 提取特征（WideResNet-50 layer2+layer3）          │   │
│  │  Step 3: Coreset 采样（Greedy / Random）                  │   │
│  │  Step 4: BaihuNN 索引拟合（NPU 矩阵乘加速）               │   │
│  │  Step 5: 测试集推理（异常评分 + 分割图生成）              │   │
│  │  Step 6: 指标计算（ImgAUROC / PixAUROC / PRO）           │   │
│  └────────────────────────────────────────────────────────────┘   │
│                           │                                        │
│                   串行 / 并行（turbo 模式）                        │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 3: 汇总输出                                               │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  控制台: 逐类打印 + 平均值                                  │   │
│  │  CSV:    results/inference_<timestamp>/results.csv          │   │
│  └────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 核心算法：BaihuNN 两阶段检索

### 设计原理

BaihuNN 使用恒等式计算 L2 距离，完全基于 `torch.mm` 矩阵乘，充分利用 NPU MATRIX 单元：

```
dist² = ||q||² + ||x||² - 2 · q · xᵀ
```

### 两阶段检索流程

```
阶段 1（NPU AI Core, FP16）：
  ┌──────────┐    ┌──────────┐    ┌───────────────┐
  │ Q 查询集  │    │ X 索引集  │    │ 距离矩阵计算   │
  │ (B, D)   │───→│ (N, D)   │───→│ torch.mm       │
  └──────────┘    └──────────┘    │ (B, N) FP16   │
                                  └───────┬───────┘
                                          │ top-K * 8
                                          ▼
                                  ┌───────────────┐
                                  │ 粗筛候选集     │
                                  │ (B, K*8)      │
                                  └───────┬───────┘
                                          │
阶段 2（CPU, FP32）：                     │
  ┌──────────┐    ┌──────────────────┐    │
  │ Q (FP32) │───→│ 候选精确距离计算  │←───┘
  └──────────┘    │ torch.bmm (FP32) │
                  └───────┬──────────┘
                          │ top-K
                          ▼
                  ┌──────────────────┐
                  │ 最终 top-K 结果   │
                  │ 距离 + 索引      │
                  └──────────────────┘
```

### 核心优势

| 对比项 | Faiss GPU (原版) | BaihuNN NPU (本 Skill) |
|:-------|:----------------:|:---------------------:|
| **依赖** | faiss-gpu (CUDA) | 零额外依赖，仅 PyTorch |
| **设备** | NVIDIA GPU | 昇腾 NPU (MATRIX 单元) |
| **精度** | FP32 全精度 | FP32 精排，精度无损 |
| **部署** | 需要 CUDA 运行时 | 纯 PyTorch 生态 |
| **速度** | CUDA 核函数 | 矩阵乘算子级加速 |

---

## 加速模式对比

| 模式 | 训练 batch | 采样器 | 采样率 | 训练子集 | 并行 | 预期加速 | 精度影响 |
|:----:|:----------:|:------:|:------:|:--------:|:----:|:--------:|:--------:|
| normal | 1 | Greedy | 10% | 100% | 否 | 1× | 基准 |
| fast | 4 | Random | 2% | 100% | 否 | 2–3× | ~0.5% |
| turbo | 8 | Random | 1% | 30% | 是 | 5–8× | ~1–2% |

---

## 参数说明

### 命令行参数

| 参数 | 默认值 | 说明 |
|:-----|:------:|:-----|
| `--data_dir` | (必填) | MVTec-AD 数据集根目录 |
| `--device` | auto | 运行设备：auto/npu:0/cpu |
| `--mode` | normal | 加速模式：normal/fast/turbo |
| `--categories` | 全 15 类 | 指定类别子集 |
| `--batch_size` | 自动调优 | 推理阶段 batch size |
| `--train_batch_size` | 自动调优 | 训练阶段 batch size |
| `--num_workers` | 自动检测 | DataLoader 工作进程数 |
| `--parallel` | auto | 并行类别数（仅 turbo） |
| `--backbone` | wideresnet50 | 骨干网络 |
| `--sampler_percentage` | 自动调优 | Coreset 采样比例 |
| `--seed` | 0 | 随机种子 |

---

## 性能与精度

### 测试环境

| 项目 | 配置 |
|:-----|:-----|
| **硬件** | 昇腾 800I A2 (Atlas 800T A2) |
| **显存** | 64 GiB |
| **CANN** | 8.5.1 |
| **PyTorch** | 2.9.0 / torch_npu 2.9.0 |
| **数据集** | MVTec-AD（15 类，~5 GB） |

### 精度指标（MVTec-AD 15 类平均）

| 指标 | 论文基线 (GPU) | NPU 实测 | 偏差 |
|:-----|:--------------:|:--------:|:----:|
| **Image AUROC** | 0.990 | **0.9905** | +0.0005 |
| **Pixel AUROC** | 0.980 | **0.9812** | +0.0012 |
| **PRO** | — | **0.9963** | — |

### 优化前后对比

| 指标 | 优化前（Faiss CPU） | 优化后（BaihuNN NPU） | 提升 |
|:-----|:------------------:|:--------------------:|:----:|
| **15 类总耗时** | ~1710s (28.5min) | **1363.7s (22.7min)** | **快 25%** |
| **特征搜索方式** | Faiss CPU 暴力搜索 | BaihuNN (`torch.mm` 矩阵乘) | 算子级加速 |
| **设备利用率** | CPU 核利用率低 | NPU MATRIX 单元满负荷 | 架构匹配 |
| **精度 (ImgAUROC)** | 0.990 | **0.9905** | 持平 |
| **依赖** | faiss-cpu（额外安装） | **零额外依赖，仅 PyTorch** | 部署更轻量 |

---

## 文件结构

```
patchcore-npu-inference/
├── SKILL.md                                # 本文件
├── scripts/
│   ├── inference.py                        # 主入口脚本（根目录版）
│   ├── main.py                             # 比赛入口别名
│   ├── setup.py                            # 安装脚本
│   ├── src/
│   │   ├── inference.py                    # 核心推理逻辑
│   │   └── patchcore/
│   │       ├── __init__.py
│   │       ├── backbones.py                # 骨干网络定义
│   │       ├── baihu_nn.py                 # BaihuNN 引擎（核心 NPU 优化）
│   │       ├── common.py                   # 通用工具
│   │       ├── datasets/
│   │       │   ├── __init__.py
│   │       │   └── mvtec.py                # MVTec-AD 数据集加载
│   │       ├── metrics.py                  # 评估指标计算
│   │       ├── networks/
│   │       │   └── __init__.py
│   │       ├── patchcore.py                # PatchCore 算法核心
│   │       ├── sampler.py                  # Coreset 采样
│   │       └── utils.py                    # 工具函数
│   └── bin/
│       ├── load_and_evaluate_patchcore.py  # 模型加载与评估
│       ├── run_patchcore.py                # 运行脚本
│       └── verify_precision.py             # 精度验证
└── references/
    ├── validation_report.md                # NPU 适配验证报告
    ├── code_review_report.md               # 代码审查报告
    └── model_card.md                       # 模型卡
```

---

## 输出说明

### 控制台输出

运行结束后逐类打印指标并汇总：

```
类别             ImgAUROC   PixAUROC   PRO       训练(s)    推理(s)    总计(s)
--------------------------------------------------------------------------------
bottle           1.0000     0.9848     0.9999    52.4       5.2        57.6
...
平均值           0.9905     0.9812     0.9963    66.4       6.8        73.2

端到端总耗时（挂钟）: 1363.7 秒 (22.7 分钟)
```

### CSV 文件

保存至 `results/inference_<timestamp>/results.csv`，含逐类和平均结果。

---

## 适配改造点摘要

| # | 改造点 | 原实现 (CUDA) | NPU 适配后 |
|:-:|:-------|:-------------|:-----------|
| 1 | **设备抽象** | `tensor.cuda()` / `model.to('cuda')` | `tensor.to(device)` / `model.to(device)`，由 `--device auto` 自动选择 |
| 2 | **NN 搜索** | Faiss GPU (`faiss-gpu`) | **BaihuNN** — `torch.mm` 矩阵乘 + 两阶段检索 |
| 3 | **显存管理** | `torch.cuda.empty_cache()` | `torch.npu.empty_cache()`，兼容两者 |
| 4 | **随机种子** | `torch.cuda.manual_seed()` | `torch.npu.manual_seed()`，兼容两者 |
| 5 | **数据加载** | 无特殊处理 | `pin_memory=True` 配合 NPU 异步传输 |
| 6 | **自动环境检测** | 无 | 检测 torch_npu、NPU 设备/显存、CPU、内存 |
| 7 | **参数自动调优** | 无 | 根据显存/内存自动设置 batch/采样率 |

---

## 依赖清单

| 库 | 最低版本 | 说明 |
|:---|:--------:|:-----|
| Python | ≥ 3.8 | — |
| torch | ≥ 1.9.0 | PyTorch 框架 |
| torch_npu | ≥ 2.1.0 | 昇腾 NPU PyTorch 插件 |
| torchvision | ≥ 0.16.0 | 图像处理 |
| timm | ≥ 0.6.0 | 骨干网络（WideResNet-50） |
| faiss-cpu | ≥ 1.7.0 | CPU 近邻搜索（仅回退用） |
| scikit-learn | ≥ 1.0 | 评估指标 |
| scikit-image | ≥ 0.19 | 图像后处理 |
| tqdm | ≥ 4.60 | 进度条 |

---

## 注意事项

1. 首次运行会自动下载 WideResNet-50 骨干网络权重（需联网）
2. MVTec-AD 数据集需提前下载并按 `bottle/train/good/` 等子目录结构存放
3. NPU 模式下自动启用 BaihuNN 替代 FaissNN 做最近邻搜索
4. 如 NPU 显存不足，脚本自动降低 batch_size 和采样率

---

## 参考文档

- [validation_report.md](references/validation_report.md) — NPU 适配验证报告（算子兼容性、精度对齐、性能对比）
- [code_review_report.md](references/code_review_report.md) — 代码审查报告
- [model_card.md](references/model_card.md) — 模型卡

## 引用

```bibtex
@misc{roth2021total,
    title={Towards Total Recall in Industrial Anomaly Detection},
    author={Karsten Roth and Latha Pemula and Joaquin Zepeda
            and Bernhard Schölkopf and Thomas Brox and Peter Gehler},
    year={2021},
    eprint={2106.08265},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
