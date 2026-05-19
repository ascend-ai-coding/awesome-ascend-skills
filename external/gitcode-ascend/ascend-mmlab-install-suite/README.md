# ascend-mmlab-install-suite

在昇腾NPU容器环境中安装 OpenMMLab 系列库的 Skill Suite。

## 概述

本 Suite 提供在昇腾 NPU 容器中安装 OpenMMLab 系列库的完整能力：
- mmcv：计算机视觉基础库
- mmdet：2D 目标检测库
- mmdet3d：3D 目标检测库
- detectron2：Facebook 的检测分割库

## 核心特性

1. **NPU 环境适配**：所有库均适配昇腾 NPU 环境
2. **网络容错**：支持本地克隆 + scp 传输方案
3. **版本兼容**：明确的版本依赖关系
4. **CUDA 检查**：安装后验证是否为 NPU 版本
5. **前置依赖**：根据 PyTorch 版本安装对应 requirements 文件

## 使用方法

### 1. 确认 PyTorch 版本和前置依赖

向用户确认目标 PyTorch 版本，安装对应的前置 requirements：

| PyTorch版本 | requirements文件 |
|-------------|------------------|
| torch 2.1.x | requirements.txt |
| torch 2.6.0 | requirements_pytorch2.6.0.txt |
| torch 2.7.1 | requirements_pytorch2.7.1.txt |
| torch 2.7.1 (Ascend NPU) | requirements_pytorch2.7.1_a5.txt |

### 2. 选择要安装的库

根据需求选择对应的模块：
- 需要 2D 检测 → mmdet
- 需要 3D 检测 → mmdet3d
- 需要实例分割 → detectron2

### 3. 确认环境信息

向用户确认以下信息：
- conda 环境名称
- 服务器 SSH 连接方式
- 工作目录路径
- 网络情况（能否访问 GitHub）

### 4. 执行安装

按照对应模块的 SKILL.md 执行安装

## 模块

| 模块 | 功能 | 依赖 |
|------|------|------|
| mmcv | 计算机视觉基础库 | - |
| mmdet | 2D目标检测 | mmcv |
| mmdet3d | 3D目标检测 | mmcv, mmsegmentation |
| detectron2 | 实例分割 | mmcv |

## 安装顺序建议

```
1. mmcv（基础）
   ├── 2. mmdet
   ├── 2. mmdet3d（需要先装 mmsegmentation）
   └── 2. detectron2
```

## NPU 环境检查

安装完成后必须执行：

```bash
python -c "import site; print([p for p in site.getsitepackages() if 'cuda' in p.lower()])"
```

如果输出非空（包含 cuda 路径），说明安装的是 CUDA 版本，需要重新安装。

## 网络策略

| 情况 | 策略 |
|------|------|
| 服务器直连正常 | 服务器上直接 git clone |
| 服务器网络超时 | 本地克隆 → scp 传输 |
| 有代理 | 配置代理后操作 |

## 前置条件

- 昇腾 NPU 环境（CANN + torch_npu）
- 工作目录已挂载
- conda 环境已配置
