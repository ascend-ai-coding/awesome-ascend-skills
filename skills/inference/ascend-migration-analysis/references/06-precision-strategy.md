# Domain 6: Precision & Numerics Analysis

## Overview

分析项目中的精度策略，识别需要调整的数值精度模式。

## Scan Commands

```bash
# Type conversions (potential precision loss on NPU)
rg "\.float\(\)" --include="*.py" -n
rg "\.double\(\)" --include="*.py" -n
rg "\.half\(\)" --include="*.py" -n
rg "\.bfloat16\(\)" --include="*.py" -n

# Dtype specifications
rg "torch\.bfloat16" --include="*.py" -n
rg "torch\.float16|torch\.float32|torch\.float64" --include="*.py" -n
rg "torch\.complex128|torch\.complex64" --include="*.py" -n

# Dtype assertions
rg "assert.*dtype.*float" --include="*.py" -n
rg "dtype == torch\.float32" --include="*.py" -n

# Type casting patterns
rg "\.type_as\(" --include="*.py" -n
rg "\.to\(dtype" --include="*.py" -n

# autocast usage
rg "autocast|torch\.amp" --include="*.py" -n

# Sinusoidal/embedding precision
rg "float64|torch\.float64|\.double\(\)" --include="*.py" -n
rg "complex128|torch\.polar|torch\.view_as_complex" --include="*.py" -n
```

## Common Patterns & Solutions

### Pattern 1: RMSNorm with float32 casting

```python
# Original: cast to float32 for numerical stability
class RMSNorm(nn.Module):
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

# Ascend: NPU fused operator, native bfloat16
import torch_npu

class RMSNorm(nn.Module):
    def forward(self, x):
        return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]
```

**Why**: NPU 的 `npu_rms_norm` 融合算子原生支持 bfloat16，无需 float32 中间转换。

### Pattern 2: LayerNorm with float32 casting

```python
# Original
class LayerNorm_FP32(nn.LayerNorm):
    def forward(self, x):
        return F.layer_norm(x.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps).to(x.dtype)

# Ascend: remove float32 casting, NPU handles BF16 natively
class LayerNorm_FP32(nn.LayerNorm):
    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
```

### Pattern 3: Sinusoidal Embedding with float64

```python
# Original: float64 for precision
freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
t = torch.arange(max_seq_len)
freqs = torch.polar(torch.ones_like(freqs), torch.outer(t, freqs))  # complex128

# Ascend: lower to float32 / complex64
freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
t = torch.arange(max_seq_len)
freqs = torch.polar(torch.ones_like(freqs), torch.outer(t, freqs))
freqs = freqs.to(torch.complex64)  # complex128 → complex64
```

**Why**: NPU 对 float64/complex128 的支持有限，float32/complex64 提供足够的精度。

### Pattern 4: RoPE (Rotary Position Embedding)

```python
# Original: complex multiplication
q_, k_ = q.float(), k.float()
cos, sin = freqs.cos(), freqs.sin()
q_ = (q_ * cos) + (rotate_half(q_) * sin)
k_ = (k_ * cos) + (rotate_half(k_) * sin)
return q_.type_as(q), k_.type_as(k)

# Ascend Option A: keep as-is (pure PyTorch, works on NPU)
# No change needed, just remove .float() casting if desired

# Ascend Option B: fused operator (if using mindiesd)
from mindiesd import rotary_position_embedding
output = rotary_position_embedding(x, cos, sin, rotated_mode="rotated_interleaved", fused=True)
```

### Pattern 5: AdaLN with float32 protection + dtype assert

```python
# Original
with amp.autocast(device_type='cuda', dtype=torch.float32):
    shift, scale = self.adaLN_modulation(t).chunk(2, dim=-1)
assert shift.dtype == torch.float32

# Ascend: change device_type, remove assert
with amp.autocast(device_type='npu', dtype=torch.float32):
    shift, scale = self.adaLN_modulation(t).chunk(2, dim=-1)
# assert shift.dtype == torch.float32  # remove or comment out
```

### Pattern 6: Random Number Reproducibility

```python
# Ascend: use CPU generator for cross-platform reproducibility
precision_cpu = int(os.getenv('PRECISION', 0))
gen_device = torch.device("cpu") if precision_cpu else target_device
seed_g = torch.Generator(device=gen_device)
seed_g.manual_seed(seed)
noise = torch.randn(shape, dtype=torch.float32, device=gen_device, generator=seed_g).to(target_device)
```

## Precision Summary Table

| Component | Typical Original | Ascend Recommendation | Reason |
|-----------|-----------------|----------------------|--------|
| Model weights | bfloat16 | bfloat16 | NPU 原生支持 |
| RMSNorm | float32 (via .float()) | native bfloat16 | npu_rms_norm 融合算子 |
| LayerNorm | float32 (via .float()) | native bfloat16 | 移除 .float() |
| Sinusoidal embedding | float64 | float32 | NPU float64 限制 |
| RoPE frequencies | complex128 | complex64 | NPU complex128 限制 |
| AdaLN modulation | float32 (autocast) | float32 | 保持不变 |
| Attention Q/K/V | original dtype | bfloat16 (forced) | NPU FA 算子要求 |
| Timestep embedding | float32 | float32 | 保持不变 |
| Random numbers | device-native | CPU (optional) | 跨平台可复现 |

## Output Template

```markdown
### Precision Strategy Analysis

| # | File | Line | Pattern | Original | Ascend | Effort |
|---|------|------|---------|----------|--------|--------|
| 1 | blocks.py | 49 | RMSNorm .float() | float32 cast | npu_rms_norm | Low |
| 2 | blocks.py | 62 | LayerNorm .float() | float32 cast | remove .float() | Low |
| 3 | rope.py | 113 | RoPE .float() | float32 cast | keep/remove | Low |
| 4 | dit.py | 82 | autocast('cuda') | cuda+fp32 | npu+fp32 | Low |
```
