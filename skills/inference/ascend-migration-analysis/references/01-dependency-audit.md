# Domain 1: Third-Party Dependency Audit

## Overview

审查项目所有外部依赖，识别 NVIDIA/CUDA 专属库，评估 Ascend NPU 兼容性。

## Scan Method

### 1.1 Direct Dependencies

```bash
# Scan requirements files
cat requirements*.txt setup.py pyproject.toml setup.cfg 2>/dev/null | grep -iE "cuda|nvidia|flash.attn|triton|xformers|cupy|cufile|cudf|cuml|rapids|nvmag|nvtx|apex|deepspeed"
```

### 1.2 Indirect Imports in Source Code

```bash
# Scan Python source for CUDA-specific imports
rg "import (flash_attn|triton|xformers|cupy|cuda|apex)" --include="*.py"
rg "from (flash_attn|triton|xformers|cupy|cuda|apex)" --include="*.py"
```

### 1.3 Build-Time Dependencies

```bash
# Scan for CUDA extension builds
rg "CUDAExtension|CppExtension|load\(|nvcc" --include="*.py"
find . -name "*.cu" -o -name "*.cuh" -o -name "Makefile" | head -20
```

## Compatibility Classification

对每个依赖标注以下分类之一：

### Compatible (直接可用)

| Package | Notes |
|---------|-------|
| `torch` | 需替换为 `torch` + `torch_npu` |
| `transformers` | 纯模型推理，与设备无关 |
| `diffusers` | ConfigMixin/ModelMixin 基类，与设备无关 |
| `safetensors` | 权重序列化，与设备无关 |
| `einops` | 纯张量操作库 |
| `numpy`, `opencv-python`, `imageio`, `PIL` | 标准 Python 库 |
| `loguru`, `tqdm`, `ftfy`, `regex` | 工具库 |
| `scipy`, `scikit-learn`, `scikit-image` | CPU 科学计算 |

### Need Replacement (有成熟替代)

| Package | Ascend Replacement | Notes |
|---------|-------------------|-------|
| `flash-attn` (FA2) | `mindiesd.attention_forward()` | 通过 `op_type` 参数选择后端 |
| `flash-attn-interface` (FA3) | `mindiesd.attention_forward()` | 同上 |
| `xformers` | `mindiesd.attention_forward()` | memory_efficient_attention 替代 |
| `triton` (kernel) | 需逐个分析替代方案 | 见 Domain 4 |
| `nvidia-ml-py` | 项目/任务相关 | 可能为可选依赖 |

### Not Supported (无替代，需评估是否必需)

| Package | Impact Assessment |
|---------|-------------------|
| `cuda-python` | 底层 CUDA runtime 绑定，通常可绕过 |
| `cupy` | GPU 数组计算，可用 numpy/torch 替代 |
| `apex` | NVIDIA 混合精度训练，torch_npu.amp 已覆盖 |
| `deepspeed` | 部分功能有 Ascend 版本 |

### Not Involved (当前任务不涉及)

标注任务不使用的依赖，避免过度评估。

## Output Template

```markdown
### Dependency Compatibility Matrix

| Package | Version | Classification | Replacement | Notes |
|---------|---------|---------------|-------------|-------|
| torch | 2.6.0 | Need Replacement | torch + torch_npu | 版本需匹配 CANN |
| flash-attn | 2.7.4 | Need Replacement | mindiesd.attention_forward() | |
| transformers | 4.41.0 | Compatible | - | |
| triton | - | Need Replacement | per-kernel analysis | |
| ... | | | | |
```
