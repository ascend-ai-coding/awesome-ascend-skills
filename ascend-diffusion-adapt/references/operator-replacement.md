# Operator Replacement Guide: Fused NPU Kernels

Before/after code for replacing standard PyTorch operators with Ascend-optimized fused kernels.

## Table of Contents
1. [RMSNorm](#1-rmsnorm)
2. [LayerNorm](#2-layernorm)
3. [Rotary Position Embedding (RoPE)](#3-rotary-position-embedding-rope)
4. [CausalConv3d Padding](#4-causalconv3d-padding)
5. [SiLU/GELU Activation](#5-silugelu-activation)
6. [Summary Table](#6-summary-table)

---

## 1. RMSNorm

RMSNorm is used extensively in DiT (Diffusion Transformers) and modern UNet architectures. The fused NPU kernel avoids materializing intermediate tensors.

### Original (Manual Computation)
```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # Manual computation — 3 separate ops
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight
```

### Ascend Replacement
```python
import torch_npu

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # Single fused kernel — no intermediate tensors
        output, _ = torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)
        return output
```

**Note:** `npu_rms_norm` returns a tuple `(output, inverse_rms)`. The second value is only needed for backward pass in training; discard it during inference.

### Conditional Pattern (Supporting Both Backends)
```python
import os

try:
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.use_npu = HAS_NPU and torch.npu.is_available()

    def forward(self, x):
        if self.use_npu:
            output, _ = torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)
            return output
        else:
            norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x * norm * self.weight
```

---

## 2. LayerNorm

### Original
```python
output = F.layer_norm(x, normalized_shape, weight, bias, eps)
# or
layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
output = layer_norm(x)
```

### Ascend Replacement (mindiesd fast_layernorm)
```python
import os
FAST_LN = int(os.environ.get("FAST_LAYERNORM", "0"))

if FAST_LN:
    from mindiesd import fast_layernorm
    # fast_layernorm takes (nn.Module, tensor) — NOT separate weight/bias/eps
    # Pass the LayerNorm module itself as the first argument
    output = fast_layernorm(self.norm1, x)
else:
    output = self.norm1(x)  # Standard nn.LayerNorm forward
```

**IMPORTANT**: `fast_layernorm` signature is `fast_layernorm(norm_module, x)` where `norm_module` is an `nn.LayerNorm` instance. It extracts weight, bias, and eps from the module internally. Do NOT pass weight/bias/eps as separate arguments.

The mindiesd `fast_layernorm` is particularly beneficial for large hidden dimensions (>2048) where the standard LayerNorm becomes memory-bandwidth-bound.

---

## 3. Rotary Position Embedding (RoPE)

RoPE is critical for DiT architectures and modern transformer-based diffusion models (FLUX, SD3, Wan2.x). The CUDA version typically uses complex number multiplication; the Ascend version uses fused kernels.

### Original (Complex Number Application)
```python
def apply_rotary_emb(x, freqs):
    """
    x: [batch, seq, heads, dim]
    freqs: [seq, dim//2, 2]  (cos, sin pairs)
    """
    # Reshape x to complex
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = torch.view_as_complex(freqs)
    
    # Apply rotation
    x_rotated = x_complex * freqs_complex
    
    # Back to real
    return torch.view_as_real(x_rotated).reshape(*x.shape).type_as(x)
```

### Ascend Replacement Option A: mindiesd
```python
from mindiesd import rotary_position_embedding

def apply_rotary_emb(x, freqs_cos, freqs_sin):
    """
    mindiesd RoPE uses pre-decomposed cos/sin.
    
    x: [batch, seq, heads, dim]
    freqs_cos: [seq, dim]
    freqs_sin: [seq, dim]
    
    NOTE: Uses rotated_mode= (string kwarg), NOT rotated_interleaved= (bool).
    Also requires fused=True for the fused kernel path.
    """
    return rotary_position_embedding(
        x, freqs_cos, freqs_sin,
        rotated_mode="rotated_interleaved",
        fused=True
    )
```

### Ascend Replacement Option B: torch_npu
```python
import torch_npu

def apply_rotary_emb(x, freqs_cos, freqs_sin):
    """
    torch_npu native RoPE.
    
    x: [batch, seq, heads, dim]
    freqs_cos: [1, seq, 1, dim]  
    freqs_sin: [1, seq, 1, dim]
    """
    return torch_npu.npu_apply_rotary_pos_emb(
        x, freqs_cos, freqs_sin
    )
```

### Pre-Computing cos/sin Decomposition

The original code often stores RoPE frequencies as complex numbers. For Ascend, pre-decompose:

```python
class RoPEFreqs:
    def __init__(self, dim, max_seq_len=8192, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len)
        angles = torch.outer(t, freqs)
        
        # Store cos/sin separately instead of complex
        self.cos = angles.cos()  # [max_seq, dim//2]
        self.sin = angles.sin()  # [max_seq, dim//2]
    
    def get(self, seq_len):
        cos = self.cos[:seq_len]  # [seq, dim//2]
        sin = self.sin[:seq_len]  # [seq, dim//2]
        # Expand to full dim by repeating (interleaved pattern)
        cos = cos.repeat(1, 2)  # [seq, dim]
        sin = sin.repeat(1, 2)  # [seq, dim]
        return cos, sin
```

### Environment Variable Dispatch
```python
ROPE_OPT = int(os.environ.get("ROPE_OPT", "0"))

def apply_rotary_emb(x, freqs_cos, freqs_sin):
    if ROPE_OPT == 0 and HAS_MINDIESD:
        return rotary_position_embedding(x, freqs_cos, freqs_sin, rotated_mode="rotated_interleaved", fused=True)
    elif ROPE_OPT == 1 and HAS_TORCH_NPU:
        return torch_npu.npu_apply_rotary_pos_emb(x, freqs_cos.unsqueeze(0).unsqueeze(2), freqs_sin.unsqueeze(0).unsqueeze(2))
    else:
        # Fallback: manual application
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([x1 * freqs_cos - x2 * freqs_sin, x2 * freqs_cos + x1 * freqs_sin], dim=-1)
```

---

## 4. CausalConv3d Padding

For video diffusion models using 3D causal convolutions. The optimization moves spatial padding into the Conv3d kernel itself.

### Original
```python
class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # All padding done manually via F.pad
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=(0, 0, 0))  # No built-in padding
        self.temporal_padding = kernel_size[0] - 1
        self.spatial_padding = (padding, padding, padding, padding)
    
    def forward(self, x):
        # Pad: (spatial_w, spatial_w, spatial_h, spatial_h, temporal, 0)
        x = F.pad(x, (*self.spatial_padding, self.temporal_padding, 0))
        return self.conv(x)
```

### Ascend Optimized
```python
class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # Spatial padding handled by Conv3d kernel (fused, faster)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=(0, padding, padding))
        # Only temporal padding via F.pad
        self.temporal_padding = kernel_size[0] - 1
        self._pad = (0, 0, 0, 0, self.temporal_padding, 0)
    
    def forward(self, x):
        # Only temporal dimension needs explicit padding
        x = F.pad(x, self._pad)
        return self.conv(x)
```

**Why this helps:** NPU convolution kernels are optimized for standard padding patterns. When spatial padding is part of the Conv3d spec, the kernel fuses padding with the convolution computation, eliminating a separate memory copy operation. Temporal padding (causal = pad only the past) cannot be expressed as standard Conv3d padding, so it remains as `F.pad`.

---

## 5. SiLU/GELU Activation

Standard activations generally work fine on NPU without replacement. However, if profiling shows activation functions as bottlenecks:

```python
# torch_npu has optimized paths for standard activations
# These are used automatically when tensors are on NPU
x = F.silu(x)   # Automatically uses NPU-optimized kernel
x = F.gelu(x)   # Automatically uses NPU-optimized kernel
```

No explicit replacement needed — `torch_npu` registers optimized kernels for standard PyTorch operations when tensors are on NPU device.

---

## 6. Summary Table

| Operator | Original | Ascend Replacement | Env Var | Speedup |
|----------|----------|-------------------|---------|---------|
| RMSNorm | `torch.rsqrt(x.pow(2).mean(...))` | `torch_npu.npu_rms_norm(x, w)` | — | ~2x |
| LayerNorm | `nn.LayerNorm(x)` | `fast_layernorm(norm_module, x)` | `FAST_LAYERNORM` | ~1.5x |
| RoPE | Complex number multiplication | `rotary_position_embedding(x, cos, sin, rotated_mode=..., fused=True)` | `ROPE_OPT=0` | ~2x |
| RoPE | Complex number multiplication | `torch_npu.npu_apply_rotary_pos_emb(...)` | `ROPE_OPT=1` | ~1.8x |
| CausalConv3d | All padding via `F.pad` | Spatial padding in Conv3d kernel | — | ~1.3x |
| Attention | `flash_attn_func(...)` | `attention_forward(q, k, v, opt_mode=..., op_type=..., layout=...)` | `ALGO` | ~1.5-3x |
