# Domain 3: Precision Strategy

## Overview

Ascend NPU has different precision characteristics compared to NVIDIA GPU. This domain covers systematic precision adjustments to ensure both correctness and performance on NPU hardware.

## 3.1 Core Principle

**Lower precision wherever NPU supports it natively; maintain precision only where numerical stability requires it.**

The overall precision shift: `float64 → float32`, `float32 → bfloat16`, `complex128 → complex64`.

## 3.2 Sinusoidal Embedding Precision

```python
# Original: float64
def sinusoidal_embedding_1d(dim, position):
    half = dim // 2
    sinusoid = torch.outer(
        position.type(torch.float64),  # <-- float64
        torch.pow(10000, -torch.arange(half).to(torch.float64).div(half))
    )
    ...

# Ascend: float32
def sinusoidal_embedding_1d(dim, position):
    half = dim // 2
    sinusoid = torch.outer(
        position.type(torch.float32),  # <-- float32
        torch.pow(10000, -torch.arange(half).to(torch.float32).div(half))
    )
    ...
```

**Reason:** NPU has limited float64 support. Float32 provides sufficient precision for sinusoidal position encoding.

## 3.3 RoPE Frequency Precision

```python
# Original
@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.polar(torch.ones_like(freqs), torch.outer(t, freqs))
    return freqs  # complex128

# Ascend
@torch.amp.autocast('npu', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.polar(torch.ones_like(freqs), torch.outer(t, freqs))
    return freqs.to(torch.complex64)  # complex128 → complex64
```

## 3.4 autocast dtype: float32 → bfloat16

All `autocast` contexts within the model change their target dtype:

```python
# Original (in WanAttentionBlock, Head, WanModel)
with torch.amp.autocast('cuda', dtype=torch.float32):
    e = ...  # time embedding, modulation parameters

# Ascend
with torch.amp.autocast('cuda', dtype=torch.bfloat16):  # note: 'cuda' kept for transfer_to_npu
    e = ...
```

**Affected locations:**
- `WanAttentionBlock.forward()` — time embedding modulation
- `Head.forward()` — output head modulation
- `WanModel.forward()` — global time embedding computation
- `sp_dit_forward()` — sequence parallel time embedding

## 3.5 Remove dtype Assertions

All float32 dtype assertions are commented out:

```python
# Original
assert e.dtype == torch.float32
assert e.dtype == torch.float32 and e0.dtype == torch.float32

# Ascend
# assert e.dtype == torch.float32  (commented out)
```

This is a natural consequence of switching autocast to bfloat16.

## 3.6 Remove `.float()` in Normalization

```python
# Original (WanLayerNorm)
def forward(self, x):
    return super().forward(x.float()).type_as(x)

# Ascend
def forward(self, x):
    return torch.nn.functional.layer_norm(x, ...)  # No float() conversion

# Original (in WanAttentionBlock)
y = self.self_attn(self.norm1(x).float() * (1 + e[1]) + e[0], ...)

# Ascend
norm1_out = self.norm1(x)  # No .float()
y = self.self_attn(norm1_out * (1 + e[1]) + e[0], ...)
```

## 3.7 Random Number Reproducibility (PRECISION env var)

NPU random number generators may produce different sequences than CUDA. To ensure cross-platform reproducibility:

```python
# Ascend: controlled by PRECISION environment variable
precision_cpu = int(os.getenv('PRECISION', 0))
generator_device = torch.device("cpu") if precision_cpu else self.device

seed_g = torch.Generator(device=generator_device)
seed_g.manual_seed(seed)
noise = torch.randn(
    noise_shape,
    dtype=torch.float32,
    device=generator_device,
    generator=seed_g
).to(self.device)
```

When `PRECISION=1`:
- Random numbers are generated on CPU for deterministic behavior
- Then transferred to NPU device
- Ensures output matches CUDA version for validation

## 3.8 Attention Input Enforcement

All NPU attention paths force bfloat16:

```python
def _attention_op(self, q, k, v, **kwargs):
    if torch.npu.is_available():
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        ...
```

## Summary Table

| Component | Original Precision | Ascend Precision | Reason |
|-----------|-------------------|------------------|--------|
| Sinusoidal embedding | float64 | float32 | NPU float64 limitation |
| RoPE frequencies | complex128 | complex64 | NPU complex128 limitation |
| autocast dtype | float32 | bfloat16 | Performance optimization |
| RMSNorm forward | float32 (via `.float()`) | native dtype (bfloat16) | Fused NPU operator |
| LayerNorm forward | float32 (via `.float()`) | native dtype (bfloat16) | Direct F.layer_norm |
| Attention Q/K/V | original dtype | bfloat16 (forced) | NPU FA requirement |
| Random numbers | device-native | CPU (optional via PRECISION) | Cross-platform reproducibility |
