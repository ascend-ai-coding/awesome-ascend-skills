# Normalization Pattern

Optimize layer normalization with NPU-specific operators.

## RMSNorm Optimization

### Manual Implementation (Original)
```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight
```

### NPU Optimized Version
```python
import torch_npu

class NPURMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return torch_npu.npu_rms_norm(x, self.weight, eps=self.eps)
```

## File Modifications

### wan/modules/model.py
```python
# Add import
import torch_npu

# Replace manual RMSNorm
# from wan.modules.normalization import RMSNorm  # REMOVE

# Use NPU-optimized version
import torch_npu
def rms_norm(x, weight, eps=1e-6):
    return torch_npu.npu_rms_norm(x, weight, eps=eps)
```

## FAST_LAYERNORM Optimization

### Enable Fast LayerNorm
```bash
export FAST_LAYERNORM=1  # Enable NPU-optimized layer norm
```

### Code Integration
```python
import mindiesd.fast_layernorm

def get_fast_layernorm():
    if os.environ.get("FAST_LAYERNORM", "0") == "1":
        fast_layernorm = mindiesd.fast_layernorm
        fast_layernorm.apply()
        return True
    return False

# Call during model initialization
fast_ln_enabled = get_fast_layernorm()
```

### Apply to Model
```python
if fast_ln_enabled:
    # Fast layernorm automatically applies to all LayerNorm modules
    print("FAST_LAYERNORM enabled")
else:
    # Fallback to manual implementation
    print("Using manual layer normalization")
```

## Performance Comparison

| Method | Speed | Memory | Precision |
|--------|-------|--------|-----------|
| Manual RMSNorm | Baseline | Baseline | FP32 |
| npu_rms_norm | 2-3x faster | Same | FP32/FP16 |
| FAST_LAYERNORM | 3-5x faster | Lower | FP16/BF16 |

## Mixed Precision Context

### Precision Control
```python
import torch.amp

class NPURMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        # Maintain precision for stable computation
        with torch.amp.autocast('npu', enabled=False):
            output = torch_npu.npu_rms_norm(x.float(), self.weight.float(), eps=self.eps)
        return output.to(x.dtype)
```

## Environment Variables

```bash
# Fast layer normalization
export FAST_LAYERNORM=1          # 0: disabled, 1: enabled

# Precision control
export NORM_PRECISION=float32     # float32, float16, bfloat16

# Performance tuning
export NORM_USE_FAST_KERNEL=1
```

## Troubleshooting

### Issue: NaN values after normalization
**Cause**: Precision overflow/underflow
**Solution**:
```python
# Use higher precision
x = x.float()  # FP32 before normalization
output = torch_npu.npu_rms_norm(x, weight, eps=eps)
```

### Issue: Slow performance
**Cause**: FAST_LAYERNORM not enabled
**Solution**:
```bash
export FAST_LAYERNORM=1
```

### Issue: Out of memory
**Cause**: Normalization creating intermediate tensors
**Solution**:
```python
# In-place normalization
torch_npu.npu_rms_norm(x, weight, eps=eps, out=x)
```

## Configuration Templates

### Production Configuration
```yaml
normalization:
  type: "npu_rms_norm"
  eps: 1e-6
  fast_layernorm: true
  precision: "float16"

environment:
  FAST_LAYERNORM: "1"
  NORM_PRECISION: "float16"
```

### Debugging Configuration
```yaml
normalization:
  type: "manual_rms_norm"  # Use manual for debugging
  eps: 1e-6
  fast_layernorm: false

environment:
  FAST_LAYERNORM: "0"
```

## References

- [torch_npu RMSNorm API](https://github.com/Ascend/pytorch)
- [MindIE Fast LayerNorm](https://github.com/Ascend/mindie)
- [LayerNorm Optimization Guide](https://www.hiascend.com/document)
