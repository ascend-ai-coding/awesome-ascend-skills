---
name: external-gitcode-ascend-detectron2
description: 在昇腾NPU容器中从源码安装detectron2。适用于实例分割、目标检测等模型的开发。
type: capability
original-name: ascend-detectron2-install
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# ascend-detectron2-install

## 功能

在昇腾NPU容器中从源码安装 detectron2（NPU兼容版本）

## 前置条件

- mmcv 已安装
- 容器中已安装 Ascend CANN 和 torch_npu
- 工作目录已挂载

## 使用

### 步骤1：确认前置条件

确保 mmcv 已安装：
```bash
python -c "import mmcv; print(mmcv.__version__)"
```

### 步骤2：从源码安装 detectron2

**关键**：在NPU环境下必须从源码安装，不能使用预编译的CUDA版本！

```bash
source /opt/conda/bin/activate <conda-env-name>
pip install git+https://github.com/facebookresearch/detectron2.git
```

### 步骤3：验证安装

```bash
python -c "import detectron2; print('detectron2 installed')"
```

**NPU验证**（关键步骤）：
```bash
python -c "import site; print([p for p in site.getsitepackages() if 'cuda' in p.lower()])"
# 应输出空列表，确保没有安装CUDA版本
```

如果输出包含 cuda 相关路径，说明安装的是 CUDA 版本，需要卸载并重新从源码安装：
```bash
pip uninstall detectron2 -y
pip install git+https://github.com/facebookresearch/detectron2.git
```

## 常见问题

- **CUDA版本问题**：NPU环境必须从源码安装，使用预编译CUDA版本会导致运行时错误
- **依赖冲突**：可能需要先卸载已安装的CUDA版本detectron2

## 决策指引

**向用户提问**：
1. 是否已安装detectron2？是什么版本？
2. 当前环境是NPU还是CUDA？

**安装后必须验证**：
- 确认 site.getsitepackages() 中没有 cuda 相关路径
- 否则需要卸载重装
