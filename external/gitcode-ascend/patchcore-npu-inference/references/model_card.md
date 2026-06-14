---
license: apache-2.0
language:
  - en
  - zh
hardware: NPU
tags:
  - NPU
  - anomaly-detection
  - patchcore
  - mvtec-ad
  - ascend
  - 800I-A2
pipeline: image-anomaly-detection
---

# WideResNet-50 — 昇腾 800I A2 NPU 工业异常检测 (PatchCore)

## 1. 简介

本模型基于 **WideResNet-50**（`torchvision.models.wide_resnet50_2`，原始权重：https://huggingface.co/pytorch/wide_resnet50_2）作为骨干网络，结合 **PatchCore**（Roth et al., CVPR 2022）算法，从 GPU/CUDA 迁移至 **华为昇腾 800I A2 NPU**，在保持原论文精度水平的前提下，实现了纯 PyTorch + torch_npu 的原生 NPU 推理。

核心贡献是 **BaihuNN**——一个基于矩阵分解的两阶段最近邻搜索引擎，替代原版中依赖 CUDA 的 Faiss GPU。

**功能特性：**
- ✅ 端到端训练+推理，一键运行
- ✅ 自动环境检测（NPU/CPU/内存/显存）
- ✅ 参数自动调优
- ✅ 三档加速模式（normal / fast / turbo）
- ✅ 输出 ImgAUROC / PixAUROC / PRO / 端到端耗时

## 2. 环境要求

### 硬件
- **昇腾 800I A2**（Atlas 800T A2）— 推荐
- 内存 ≥ 32 GB（推荐 64 GB）
- 磁盘 ≥ 50 GB（含 MVTec-AD 数据集 ~5 GB）

### 软件
| 组件 | 版本要求 |
|:-----|:--------:|
| Python | 3.8 – 3.11 |
| CANN | 8.5.1 |
| PyTorch | ≥ 1.9.0 |
| torch_npu | ≥ 2.1.0 |
| timm | ≥ 0.6.0 |

## 3. 快速开始

```bash
# 一键运行（自动检测 NPU，全 15 类）
python inference.py --data_dir datasets/mvtec_ad
```

更多用法参见 [README.md](./README.md)。

## 4. 性能指标

| 指标 | 论文基线 (GPU) | 本方案 (NPU) | 偏差 |
|:----|:--------------:|:------------:|:----:|
| Image AUROC 均值 | **0.990** | **0.9905** | +0.0005 |
| Pixel AUROC 均值 | **0.980** | **0.9812** | +0.0012 |
| 端到端总耗时 | — | **1363.7 秒** (22.7 分钟) | — |

## 5. 数据集

使用 **MVTec-AD** 数据集（15 个子类，约 5 GB），包含 4 类纹理 + 11 类物体缺陷检测场景。

## 6. 仓库地址

- 仓库: https://ai.gitcode.com/quzhi_1981/ascend-patchcore

## 7. 引用

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
