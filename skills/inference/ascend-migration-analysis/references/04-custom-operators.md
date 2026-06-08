# Domain 4: Custom Operator / Kernel Analysis

## Overview

扫描项目中的自定义 GPU 内核（CUDA C++ / Triton），评估 NPU 替代路径。

## Scan Commands

```bash
# CUDA C++ kernels
find . -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" | grep -v build
rg "CUDAExtension|CppExtension" --include="*.py" -n
rg "torch\.utils\.cpp_extension" --include="*.py" -n
rg "load\(" --include="*.py" -n | grep -i cuda

# Triton kernels
rg "import triton" --include="*.py" -n
rg "@triton\.jit" --include="*.py" -n
rg "triton\.Config" --include="*.py" -n
rg "triton\.autotune" --include="*.py" -n
rg "triton\.language" --include="*.py" -n
rg "tl\." --include="*.py" -n

# torch.autograd.Function subclasses (may contain custom ops)
rg "class.*\(torch\.autograd\.Function\)" --include="*.py" -n

# NVIDIA-specific features
rg "nv_tma_desc\|TMA\|get_device_capability\|tensor_map" --include="*.py" -n
rg "cuda\.profiler\|nvtx\|NVTX" --include="*.py" -n
```

## Custom Kernel Categories

### Category 1: CUDA C++ Extensions (.cu/.cpp files)

**特征**: 使用 CUDA C/C++ 编写，通过 `CUDAExtension` 或 `torch.utils.cpp_extension.load()` 编译加载。

**NPU 替代路径**：

| 替代策略 | 适用场景 | 难度 |
|----------|----------|------|
| 使用 NPU 原生算子组合 | 简单融合算子 | 低 |
| 使用 torch_npu 扩展 API | 已有 NPU 等价 API | 低 |
| 使用 Ascend C 重写 | 性能关键的专用算子 | 高 |
| 使用 MindIE/ATB 算子 | Attention/RoPE/Norm 类 | 中 |
| 退化为纯 PyTorch 实现 | 非性能关键路径 | 低 |

### Category 2: Triton Kernels (@triton.jit)

**特征**: 使用 Triton 语言编写，运行在 CUDA 后端。

**常见 Triton kernel 类型及替代方案**：

#### 2a. 稀疏/Block-Sparse Attention

**典型模式**: 将 Q/K/V 分块，通过 gating score 选择性计算部分 block 的注意力。

```python
@triton.jit
def sparse_attn_forward(Q, K, V, block_indices, ...):
    # 逐 block 加载 Q/K/V
    # 计算局部注意力
    # 根据 block_indices 选择性更新
```

**NPU 替代方案**: RainFusion v2 (blockwise Top-K sparse attention)

```python
from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import rain_fusion_attention

output = rain_fusion_attention(q, k, v, block_mask=mask, block_size=128)
```

**适配要点**：
- 保留 gating/打分逻辑（通常为纯 PyTorch，可移除 `@torch.compile` 后直接使用）
- 仅替换最终的注意力执行 kernel
- 需适配 mask 格式（int indices → boolean mask）
- 推理时通常不需要 backward kernel

#### 2b. 激活函数 / 归一化融合

**典型模式**: 将 SiLU + 乘法、或 RMSNorm 融合为单个 kernel。

**NPU 替代方案**：
- `torch_npu.npu_rms_norm()` — RMSNorm 融合
- `mindiesd.fast_layernorm` — LayerNorm 融合
- 纯 PyTorch 实现通常在 NPU 上也能获得可接受的性能

#### 2c. Gating / Score 计算

**典型模式**: 平均池化压缩 + Q×K^T 打分 + Top-K 选择。

**NPU 替代方案**: 直接使用纯 PyTorch 实现（`torch.matmul` + `torch.topk`），移除 `@torch.compile`。这些操作在 NPU 上有原生支持。

#### 2d. Mask/Index 操作

**典型模式**: 根据 block indices 生成 boolean mask。

**NPU 替代方案**: 纯 PyTorch 的 scatter/gather 操作即可实现。

### Category 3: torch.autograd.Function with Custom Forward/Backward

**特征**: 继承 `torch.autograd.Function`，在 forward/backward 中调用自定义 kernel。

**分析要点**：
- 如果推理时不触发 backward，**backward 中的 CUDA kernel 无需适配**
- 仅需替换 forward path 中的 kernel

### Category 4: NVIDIA Hardware-Specific Features

| Feature | Description | NPU Status |
|---------|-------------|------------|
| TMA Descriptor | Hopper GPU Tensor Memory Accelerator | 无等价，需移除 |
| `get_device_capability()` | SM 版本检测 | 无等价，需移除或改写 |
| `triton.runtime.driver.active` | CUDA 后端检测 | 需移除 |
| `num_stages` / `num_warps` | SM 层级调度参数 | 不适用 |

## Analysis Checklist

对每个自定义 kernel，回答以下问题：

1. **功能描述**: 该 kernel 计算什么？
2. **调用路径**: 在什么场景下被调用？（训练/推理，哪个阶段）
3. **是否有纯 PyTorch 等价实现?** 如果有，可直接替换
4. **是否有 NPU 原生算子?** 查询 torch_npu / mindiesd API
5. **是否需要 backward?** 推理场景通常不需要
6. **性能关键程度?** 决定是否需要用 Ascend C 重写

## Output Template

```markdown
### Custom Operator Analysis

| # | Type | File | Function | Purpose | Called In | Replacement | Effort |
|---|------|------|----------|---------|-----------|-------------|--------|
| 1 | Triton | bsa.py:96 | _attn_fwd_bsa_varlen | Block-sparse forward | Stage 3 only | RainFusion v2 | 3 days |
| 2 | Triton | bsa.py:289 | _attn_bwd_dkdv | BSA backward dK/dV | Training only | Skip (inference) | 0 |
| 3 | Triton | common.py:32 | _attn_fwd_gating | Gating score | Stage 3 | Pure PyTorch | 0.5 day |
| 4 | CUDA | ops/kernels.cu | fused_bias_gelu | Activation fusion | All stages | NPU native ops | 1 day |
```
