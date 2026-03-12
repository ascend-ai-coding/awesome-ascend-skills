# Attention Replacement Patterns: CUDA → Ascend NPU

Complete code patterns for replacing CUDA attention implementations with Ascend-native alternatives.

## Table of Contents
1. [Overview: Attention Tier System](#1-overview-attention-tier-system)
2. [Replacing flash_attn Standard Attention](#2-replacing-flash_attn-standard-attention)
3. [Replacing flash_attn_varlen_func](#3-replacing-flash_attn_varlen_func)
4. [mindiesd Attention Algorithms](#4-mindiesd-attention-algorithms)
5. [torch_npu Native Attention](#5-torch_npu-native-attention)
6. [Environment Variable Dispatch](#6-environment-variable-dispatch)
7. [Dtype Handling](#7-dtype-handling)
8. [Sequence Parallel Attention](#8-sequence-parallel-attention)
9. [Sparse Attention (Advanced)](#9-sparse-attention-advanced)

---

## 1. Overview: Attention Tier System

Ascend supports multiple attention backends. Use this priority order:

```
Tier 1: attention_forward (from mindiesd)    — Fastest, requires mindiesd package
Tier 2: torch_npu.npu_fused_infer_attention_score — Native NPU op, no extra deps
Tier 3: torch.nn.functional.scaled_dot_product_attention — Portable fallback
```

In practice, wrap all three in a dispatcher function and select via env var:

```python
import os
ALGO = int(os.environ.get("ALGO", "0"))

def attention_op(q, k, v, mask=None):
    if ALGO == 0:
        return _mindiesd_attention(q, k, v)
    elif ALGO == 3:
        return _npu_fused_attention(q, k, v, mask)
    else:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

---

## 2. Replacing flash_attn Standard Attention

### Original CUDA Code
```python
from flash_attn import flash_attn_func

# q, k, v: [batch, seq_len, num_heads, head_dim]
output = flash_attn_func(q, k, v, causal=False)
```

### Ascend Replacement
```python
from mindiesd import attention_forward

def ascend_attention(q, k, v):
    """
    Replace flash_attn_func with mindiesd attention_forward.
    
    Input shapes:  q, k, v: [batch, seq_len, num_heads, head_dim]
    Output shape:  [batch, seq_len, num_heads, head_dim]
    
    NOTE: attention_forward does NOT accept a scale parameter.
    Scale is handled internally. It uses keyword args:
      opt_mode, op_type, layout
    """
    # Cast to bfloat16 — Ascend attention is optimized for bf16
    orig_dtype = q.dtype
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)
    
    # Transpose to [batch, num_heads, seq_len, head_dim] — BNSD layout
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    output = attention_forward(q, k, v, opt_mode="manual", op_type="fused_attn_score", layout="BNSD")
    
    # Transpose back to [batch, seq_len, num_heads, head_dim]
    output = output.transpose(1, 2).to(orig_dtype)
    return output
```

---

## 3. Replacing flash_attn_varlen_func

Variable-length attention (used for packed sequences) requires more care.

### Original CUDA Code
```python
from flash_attn import flash_attn_varlen_func

# q, k, v: [total_tokens, num_heads, head_dim]
# cu_seqlens_q, cu_seqlens_k: cumulative sequence lengths
# max_seqlen_q, max_seqlen_k: maximum sequence lengths
output = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    causal=False
)
```

### Ascend Replacement Strategy

On Ascend, there's no direct `varlen` equivalent in mindiesd. Two approaches:

**Approach A: Unpack to padded batch, run standard attention, repack**
```python
def varlen_attention_ascend(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k):
    """
    Simulate varlen attention by unpacking to padded sequences.
    """
    batch_size = len(cu_seqlens_q) - 1
    
    # Unpack to [batch, max_seq, heads, dim]
    q_padded = torch.zeros(batch_size, max_seqlen_q, q.shape[1], q.shape[2], 
                          dtype=q.dtype, device=q.device)
    k_padded = torch.zeros(batch_size, max_seqlen_k, k.shape[1], k.shape[2],
                          dtype=k.dtype, device=k.device)
    v_padded = torch.zeros_like(k_padded)
    
    for i in range(batch_size):
        sq = cu_seqlens_q[i+1] - cu_seqlens_q[i]
        sk = cu_seqlens_k[i+1] - cu_seqlens_k[i]
        q_padded[i, :sq] = q[cu_seqlens_q[i]:cu_seqlens_q[i+1]]
        k_padded[i, :sk] = k[cu_seqlens_k[i]:cu_seqlens_k[i+1]]
        v_padded[i, :sk] = v[cu_seqlens_k[i]:cu_seqlens_k[i+1]]
    
    # Run standard attention
    out = attention_forward(q_padded, k_padded, v_padded)
    
    # Repack
    output = torch.zeros_like(q)
    for i in range(batch_size):
        sq = cu_seqlens_q[i+1] - cu_seqlens_q[i]
        output[cu_seqlens_q[i]:cu_seqlens_q[i+1]] = out[i, :sq]
    
    return output
```

**Approach B: Single-sequence loop (simpler, slower)**
```python
def varlen_attention_loop(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k):
    batch_size = len(cu_seqlens_q) - 1
    output = torch.zeros_like(q)
    
    for i in range(batch_size):
        q_i = q[cu_seqlens_q[i]:cu_seqlens_q[i+1]].unsqueeze(0)
        k_i = k[cu_seqlens_k[i]:cu_seqlens_k[i+1]].unsqueeze(0)
        v_i = v[cu_seqlens_k[i]:cu_seqlens_k[i+1]].unsqueeze(0)
        out_i = attention_forward(q_i, k_i, v_i)
        output[cu_seqlens_q[i]:cu_seqlens_q[i+1]] = out_i.squeeze(0)
    
    return output
```

Approach A is faster for large batches; Approach B is simpler and sufficient for inference with batch_size=1.

---

## 4. mindiesd Attention Algorithms

mindiesd supports multiple algorithms via the `ALGO` environment variable:

### ALGO=0: fused_attn_score (Default)
```python
from mindiesd import attention_forward

# General-purpose fused attention
# q, k, v: [batch, num_heads, seq_len, head_dim] — BNSD layout
output = attention_forward(q, k, v, opt_mode="manual", op_type="fused_attn_score", layout="BNSD")
```
Best for: Most attention patterns, good balance of speed and precision.

### ALGO=1: ascend_laser_attention
```python
from mindiesd import attention_forward

# Laser attention — optimized for long sequences
# Uses op_type="ascend_laser_attention", NOT an algo= parameter
output = attention_forward(q, k, v, opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD")
```
Best for: Long sequence lengths (>4K tokens), video models.

### ALGO=3: npu_fused_infer_attention_score (torch_npu native)
```python
# Bypass mindiesd, use torch_npu directly
output = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    num_heads=q.shape[1],
    input_layout="BNSD",  # [batch, num_heads, seq, dim]
    scale=scale,
    pre_tokens=65535,
    next_tokens=65535,
)
```
Best for: When mindiesd is unavailable, or for specific operator fusion patterns.

---

## 5. torch_npu Native Attention

When mindiesd is not installed, use torch_npu directly:

```python
import torch_npu

def npu_attention(q, k, v, scale=None):
    """
    Native NPU fused attention without mindiesd dependency.
    q, k, v: [batch, num_heads, seq_len, head_dim]
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    
    output = torch_npu.npu_fused_infer_attention_score(
        q, k, v,
        num_heads=q.shape[1],
        input_layout="BNSD",
        scale=scale,
        pre_tokens=65535,
        next_tokens=65535,
    )
    return output
```

---

## 6. Environment Variable Dispatch

The recommended pattern for production: runtime-configurable attention backend.

```python
import os
import torch
import torch.nn.functional as F

ALGO = int(os.environ.get("ALGO", "0"))

try:
    from mindiesd import attention_forward as mindiesd_attention_forward
    HAS_MINDIESD = True
except ImportError:
    HAS_MINDIESD = False

try:
    import torch_npu
    HAS_TORCH_NPU = True
except ImportError:
    HAS_TORCH_NPU = False


def attention_op(q, k, v, mask=None):
    """
    Universal attention dispatcher for Ascend NPU.
    
    Args:
        q, k, v: [batch, num_heads, seq_len, head_dim] — BNSD layout
        mask: Optional attention mask
    
    NOTE: mindiesd.attention_forward does NOT accept a scale parameter.
    Scale is computed internally. Do not pass scale=... to it.
    For torch_npu and F.sdpa, scale can be passed separately.
    """
    scale = q.shape[-1] ** -0.5
    
    # Always cast to bf16 for attention on Ascend
    orig_dtype = q.dtype
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)
    
    if ALGO == 0 and HAS_MINDIESD:
        output = mindiesd_attention_forward(q, k, v, opt_mode="manual", op_type="fused_attn_score", layout="BNSD")
    elif ALGO == 1 and HAS_MINDIESD:
        output = mindiesd_attention_forward(q, k, v, opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD")
    elif ALGO == 3 and HAS_TORCH_NPU:
        output = torch_npu.npu_fused_infer_attention_score(
            q, k, v,
            num_heads=q.shape[1],
            input_layout="BNSD",
            scale=scale,
            pre_tokens=65535,
            next_tokens=65535,
        )
    else:
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale
        )
    
    return output.to(orig_dtype)
```

---

## 7. Dtype Handling

Ascend attention has specific dtype requirements:

```python
# CRITICAL: Always cast to bfloat16 for attention
# fp16 has known precision issues on Ascend 910B for attention operations
# fp32 works but is much slower

original_dtype = hidden_states.dtype
q = q.to(torch.bfloat16)
k = k.to(torch.bfloat16)
v = v.to(torch.bfloat16)

output = attention_op(q, k, v)

# Cast back to original dtype for downstream ops
output = output.to(original_dtype)
```

---

## 8. Sequence Parallel Attention

When using Ulysses or Ring sequence parallelism, attention needs special handling:

```python
from yunchang import UlyssesAttention, set_seq_parallel_pg

# Initialize sequence parallel group
set_seq_parallel_pg(sp_group)

# Wrap attention for sequence parallelism
ulysses_attn = UlyssesAttention()

# In forward:
# q, k, v are already split along sequence dimension by SP
output = ulysses_attn(
    q, k, v,
    joint_tensor_key=k if cross_attn else None,
    joint_tensor_value=v if cross_attn else None,
)
```

For Ring attention:
```python
from yunchang import LongContextAttention

ring_attn = LongContextAttention(ring_impl_type="zigzag")
output = ring_attn(q, k, v)
```

---

## 9. Sparse Attention (Advanced)

For video models with very long sequences (>16K tokens), sparse attention can skip computation:

### Grid-Based Sparsity (Rainfusion v1)
```python
def create_grid_mask(height, width, num_frames, sparsity_ratio=0.5):
    """
    Create a grid-based attention mask that samples tokens
    at regular intervals. Used in early denoising steps where
    fine-grained attention is less important.
    """
    total = height * width * num_frames
    keep = int(total * (1 - sparsity_ratio))
    
    # Sample on a regular spatial grid
    step_h = max(1, int(height / (keep / num_frames) ** 0.5))
    step_w = max(1, int(width / (keep / num_frames) ** 0.5))
    
    mask = torch.zeros(num_frames, height, width, dtype=torch.bool)
    mask[:, ::step_h, ::step_w] = True
    
    return mask.reshape(-1)
```

### Blockwise Sparsity (Rainfusion v2)
```python
def blockwise_sparse_attention(q, k, v, block_size=256, threshold=0.1):
    """
    Skip attention blocks where the estimated attention score
    is below threshold. More adaptive than grid-based.
    """
    seq_len = q.shape[2]
    num_blocks = (seq_len + block_size - 1) // block_size
    
    # Estimate block importance via mean query-key dot product
    outputs = []
    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, seq_len)
        q_block = q[:, :, start:end]
        
        # Quick importance estimate
        importance = (q_block.mean(dim=2, keepdim=True) @ k.mean(dim=2, keepdim=True).transpose(-1, -2)).abs().mean()
        
        if importance > threshold:
            out_block = attention_op(q_block, k, v)
        else:
            # Skip: use previous block's output or zero
            out_block = torch.zeros_like(q_block)
        
        outputs.append(out_block)
    
    return torch.cat(outputs, dim=2)
```

Sparse attention is model-specific. Tune `sparsity_ratio` and `threshold` per model. Start with no sparsity and add gradually while monitoring output quality.
