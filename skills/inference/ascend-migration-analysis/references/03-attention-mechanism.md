# Domain 3: Attention Mechanism Analysis

## Overview

扫描项目中所有注意力机制实现，评估 NPU 替代方案。

## Scan Commands

```bash
# Flash Attention 2
rg "from flash_attn import" --include="*.py" -n
rg "flash_attn_func|flash_attn_varlen_func" --include="*.py" -n

# Flash Attention 3
rg "from flash_attn_interface import" --include="*.py" -n
rg "flash_attn_interface" --include="*.py" -n

# xformers
rg "xformers\.ops" --include="*.py" -n
rg "memory_efficient_attention" --include="*.py" -n

# PyTorch native SDPA
rg "scaled_dot_product_attention" --include="*.py" -n

# Custom attention implementations
rg "torch\.matmul.*transpose" --include="*.py" -n
rg "softmax.*dim" --include="*.py" -n | grep -i attn
rg "class.*Attention" --include="*.py" -n

# Attention config flags
rg "enable_flashattn|enable_xformers|flash_attention|attn_backend" --include="*.py" -n
```

## Attention Type Classification

项目中的注意力实现通常分为以下几类：

### Type 1: Flash Attention 2 (FA2)

```python
from flash_attn import flash_attn_func, flash_attn_varlen_func

# Standard: [B, S, H, D] layout
output = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale)

# Varlen: variable-length sequences
output = flash_attn_varlen_func(q, k, v, cu_seqlens_q=..., cu_seqlens_k=..., max_seqlen_q=..., max_seqlen_k=...)
```

### Type 2: Flash Attention 3 (FA3)

```python
from flash_attn_interface import flash_attn_func, flash_attn_varlen_func

output, *_ = flash_attn_func(q, k, v, softmax_scale=scale)
```

### Type 3: xformers

```python
import xformers.ops

# Standard
output = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)

# BlockDiagonal (varlen)
attn_bias = xformers.ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
output = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=attn_bias)
```

### Type 4: PyTorch Native SDPA

```python
output = F.scaled_dot_product_attention(q, k, v)
```

### Type 5: Custom/Sparse Attention

使用 Triton/CUDA kernel 实现的稀疏注意力（如 Block Sparse Attention, Window Attention 等）。

### Type 6: Manual Implementation

手动 Q×K^T × softmax × V 实现。

## Replacement Solutions

### Solution Matrix

| Type | Self-Attention Replacement | Cross-Attention Replacement |
|------|---------------------------|----------------------------|
| FA2 | `mindiesd.attention_forward(op_type="ascend_laser_attention")` | `mindiesd.attention_forward(op_type="fused_attn_score")` |
| FA3 | 同上 | 同上 |
| xformers | 同上 | 同上 |
| SDPA | torch_npu 原生支持，**无需修改** | 同左 |
| Custom Sparse | 见 Domain 4 | 通常为标准 Cross-Attention |
| Manual | 考虑替换为 NPU 优化版本 | 同左 |

### mindiesd.attention_forward() Details

```python
from mindiesd import attention_forward

# ALGO=1: ascend_laser_attention (Self-Attention 最优)
output = attention_forward(q, k, v, op_type="ascend_laser_attention")

# ALGO=0: fused_attn_score (通用, 含 Cross-Attention)
output = attention_forward(q, k, v, op_type="fused_attn_score")

# ALGO=3: npu_fused_infer_attention_score (量化推理)
output = torch_npu.npu_fused_infer_attention_score(
    q, k, v, num_heads=num_heads, input_layout="BSND", ...)
```

### Key Adaptation Points

#### Tensor Layout Conversion

| Library | Default Layout | mindiesd Expected | Conversion |
|---------|---------------|-------------------|------------|
| FA2 | `[B, S, H, D]` | `[B, S, H, D]` (BSHD) | 可能无需转换 |
| FA3 | `[B, S, H, D]` | `[B, S, H, D]` (BSHD) | 可能无需转换 |
| xformers | `[B, M, H, K]` | `[B, S, H, D]` (BSHD) | 可能无需转换 |
| Manual | `[B, H, S, D]` (BNSD) | `[B, S, H, D]` (BSHD) | 需要 transpose |

#### Varlen (Variable-Length) Attention

Cross-Attention 中常见的变长序列模式：

```python
# FA2 varlen: 使用 cu_seqlens 参数
flash_attn_varlen_func(q, k, v, cu_seqlens_q=..., cu_seqlens_k=...)
```

**NPU 替代策略**：
1. 优先验证 `mindiesd` 是否支持 varlen 模式
2. 如不支持，手动按 batch 拆分为多次标准 attention 调用
3. 或将变长序列 pad 到等长后调用标准 attention

#### Input dtype Enforcement

NPU attention 算子要求 bfloat16 输入：

```python
q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
```

## ALGO Selection Guide

| ALGO | Operator | Best For | Performance |
|------|----------|----------|-------------|
| 0 | `fused_attn_score` | 通用（含 Cross-Attention） | 基线 |
| 1 | `ascend_laser_attention` | Self-Attention only | **最优** |
| 3 | `npu_fused_infer_attention_score` | 量化推理 | 量化场景优 |

## Output Template

```markdown
### Attention Mechanism Analysis

| # | File | Line | Type | Function | Replacement | Layout Conversion | Risk |
|---|------|------|------|----------|-------------|-------------------|------|
| 1 | attention.py | 85 | Self-Attn | flash_attn_func (FA2) | mindiesd ALGO=1 | BSHD→BSHD none | Low |
| 2 | attention.py | 234 | Cross-Attn | flash_attn_varlen_func (FA2) | mindiesd ALGO=0 | Varlen→pad | Medium |
| 3 | vae.py | 416 | Self-Attn | F.scaled_dot_product_attention | No change needed | - | None |
```
