# Domain 2: Operator Replacement

## Overview

Replace CUDA-optimized operators with Ascend NPU equivalents. This domain covers normalization layers, positional encoding, and attention operators — the three most performance-critical operator categories in DiT models.

## 2.1 RMSNorm → `torch_npu.npu_rms_norm()`

The original implementation manually computes RMS normalization with float32 casting:

```python
# Original
class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight
```

Replace with NPU fused operator:

```python
# Ascend
import torch_npu

class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]
```

**Benefits:**
- Eliminates `.float()` → `.type_as()` overhead
- Single fused kernel on NPU vs. multiple operations
- Maintains bfloat16 precision throughout

## 2.2 LayerNorm — Remove Type Casting

```python
# Original
class WanLayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)

# Ascend
class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)
        self.dim = dim  # Store dim for direct F.layer_norm call

    def forward(self, x):
        return torch.nn.functional.layer_norm(
            x, normalized_shape=[self.dim],
            weight=self.weight, bias=self.bias, eps=self.eps,
        )
```

**Key change:** Remove `.float()` type conversion. NPU handles bfloat16 LayerNorm natively.

## 2.3 Optional Fast LayerNorm via MindIE

Controlled by environment variable `FAST_LAYERNORM`:

```python
FAST_LAYERNORM = int(os.getenv('FAST_LAYERNORM', 0))
if FAST_LAYERNORM:
    from mindiesd import fast_layernorm

# Usage in forward pass
if FAST_LAYERNORM == 1:
    norm_out = fast_layernorm(self.norm1, x)
else:
    norm_out = self.norm1(x)
```

## 2.4 RoPE → `mindiesd.rotary_position_embedding()`

This is one of the most significant operator replacements.

### Original: Manual Complex Multiplication (~30 lines)

```python
@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()
```

### Ascend: Fused Operator (3 lines)

```python
from mindiesd import rotary_position_embedding

@torch.amp.autocast('npu', enabled=False)
def rope_apply(x, grid_sizes, freqs_list):
    cos, sin = freqs_list[0]
    return rotary_position_embedding(x, cos, sin, rotated_mode="rotated_interleaved", fused=True)
```

### Required: Pre-compute cos/sin in Model Forward

The frequency precomputation must happen in `WanModel.forward()`:

```python
# In WanModel.forward(), before the block loop:
if self.freqs_list is None:
    c = (self.dim // self.num_heads) // 2
    freqs = self.freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    freqs_list = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)
        # Convert complex to cos/sin pair
        cos, sin = torch.chunk(
            torch.view_as_real(freqs_i.to(torch.complex64)), 2, dim=-1)
        cos = cos.unsqueeze(0).expand(-1, -1, -1, -1, 2).flatten(-2)
        sin = sin.unsqueeze(0).expand(-1, -1, -1, -1, 2).flatten(-2)
        freqs_list.append((cos, sin))
    self.freqs_list = freqs_list
```

**Important:** Clear `freqs_list` after each generation to avoid cross-resolution cache conflicts:
```python
# After generation completes:
model.freqs_list = None  # or model._fsdp_wrapped_module.freqs_list = None for FSDP
```

## 2.5 Flash Attention → Multi-Backend Dispatch

Replace the single `flash_attention()` call with a multi-backend dispatch system controlled by `ALGO` environment variable:

```python
import os
ALGO = int(os.getenv('ALGO', 0))

def _attention_op(self, q, k, v, **kwargs):
    if torch.npu.is_available():
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

        if kwargs.get('rainfusion_config') is not None and q.shape[1] == k.shape[1]:
            # RainFusion sparse attention path
            return self._rainfusion_attention(q, k, v, **kwargs)

        if ALGO == 1 and q.shape[1] == k.shape[1]:  # Self-attention only
            from mindiesd import attention_forward
            return attention_forward(q, k, v, op_type="ascend_laser_attention")

        if ALGO == 3 and q.shape[1] == k.shape[1]:
            return torch_npu.npu_fused_infer_attention_score(
                q, k, v, num_heads=self.num_heads, input_layout="BSND", ...)

        # Default: ALGO == 0
        from mindiesd import attention_forward
        return attention_forward(q, k, v, op_type="fused_attn_score")
    else:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

### ALGO Selection Guide

| ALGO | Operator | Best For | Performance |
|------|----------|----------|-------------|
| 0 | `fused_attn_score` | General use, good compatibility | Baseline |
| 1 | `ascend_laser_attention` | Self-attention, high throughput | **Best** |
| 3 | `npu_fused_infer_attention_score` | Quantized inference | Good with quantization |

## Pitfalls

1. **Cross-attention uses ALGO=0 only**: ALGO=1 and ALGO=3 optimizations apply only when `q.shape[1] == k.shape[1]` (self-attention). Cross-attention always falls through to the default path.
2. **bfloat16 enforcement**: All NPU attention paths explicitly cast Q/K/V to bfloat16 before computation.
3. **`rotary_position_embedding` mode**: Must use `rotated_mode="rotated_interleaved"` and `fused=True` for correct results with Wan's interleaved RoPE format.
