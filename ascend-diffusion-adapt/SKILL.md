---
name: ascend-diffusion-adapt
description: Comprehensive guide for adapting HuggingFace Diffusers and custom diffusion models (Stable Diffusion, SDXL, FLUX, Wan2.1/2.2, CogVideoX, HunyuanVideo) from CUDA/GPU to Huawei Ascend NPU. Covers device migration, attention replacement (flash_attn/xformers → torch_npu/mindiesd), operator swaps (RMSNorm, RoPE, LayerNorm, Conv3d), distributed parallelism (CFG/Ulysses/Ring/TP/FSDP), VAE optimization, quantization (W8A8/W4A8), attention caching, and runtime tuning. Use when porting diffusion/generative models to Ascend NPU, replacing CUDA attention, setting up multi-NPU inference, quantizing DiT/UNet/VAE, or debugging NPU-specific issues in diffusion workloads.
keywords:
    - ascend
    - NPU
    - torch_npu
    - mindiesd
    - diffusion
    - CUDA迁移
    - 模型适配
    - 910B
    - 910C
    - attention替换
    - 分布式推理
    - 量化部署
---

# Ascend Diffusion Model Adaptation Guide

This skill captures battle-tested patterns for migrating diffusion models from CUDA to Huawei Ascend NPU, derived from real-world adaptation of the Wan2.2 video generation model. The patterns generalize to any diffusion architecture using attention, convolutions, and iterative denoising.

## Quick-Start: Adaptation Checklist

For any diffusion model migration, work through these layers in order:

1. **Device & Backend** — Swap CUDA references to NPU, switch NCCL to HCCL
2. **Operator Replacement** — Replace flash_attn/xformers with torch_npu/mindiesd fused ops
3. **Normalization & Embedding** — Swap RMSNorm/LayerNorm/RoPE to fused NPU kernels
4. **Convolution Optimization** — Restructure padding for NPU-friendly Conv3d/Conv2d
5. **Distributed Parallelism** — Configure FSDP + sequence/tensor/CFG parallelism
6. **VAE Optimization** — Patch parallel decode, tiling, slicing
7. **Quantization** (optional) — W8A8/W4A8 via mindiesd
8. **Attention Caching** (optional) — Skip redundant attention in denoising steps
9. **Runtime Tuning** — Warmup, compilation flags, environment variables

Each layer has a dedicated reference file with code examples. Read them as needed — don't load everything upfront.

---

## 1. Device & Backend Migration

The foundation of any Ascend port. Two approaches exist:

### Approach A: Monkey-Patch (Recommended for Quick Ports)

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu  # Intercepts cuda calls → NPU

# After this import, torch.device("cuda:0") transparently maps to npu:0
# torch.cuda.set_device() → torch.npu.set_device(), etc.
```

This is the fastest path — many CUDA codebases work with just this import. However, you still need to manually fix:
- Distributed backend: `nccl` → `hccl`
- AMP autocast device: `torch.amp.autocast('cuda')` → `torch.amp.autocast('npu')`
- Explicit device checks: `if device.type == 'cuda'` → `if device.type == 'npu'`

### Approach B: Explicit Migration (Recommended for Production)

Replace all CUDA references directly. More work upfront but clearer code.

**See:** `references/device-migration-checklist.md` for the exhaustive find-and-replace list.

### Compilation & Internal Format Flags

Always set these early in your entry point:

```python
import torch_npu
torch_npu.npu.set_compile_mode(jit_compile=False)  # Disable JIT compilation
torch.npu.config.allow_internal_format = False       # Force standard tensor layouts
```

`jit_compile=False` avoids graph compilation overhead that hurts iterative denoising workloads. `allow_internal_format=False` ensures tensor layouts remain interoperable across operators — critical when mixing torch_npu native ops with mindiesd ops.

---

## 2. Attention Replacement

This is the highest-impact change. CUDA diffusion models typically use `flash_attn` or `xformers`. On Ascend, replace with a tiered fallback:

```
Priority 1: attention_forward (from mindiesd) — fused, fastest
Priority 2: torch_npu.npu_fused_infer_attention_score (native NPU op)
Priority 3: torch.nn.functional.scaled_dot_product_attention (portable fallback)
```

The attention replacement is non-trivial because diffusion models use both:
- **Standard self/cross-attention** (fixed sequence length)
- **Variable-length attention** (flash_attn_varlen_func) for packed sequences

Both need replacement. The Ascend approach also enables runtime algorithm selection via environment variables, which is valuable for benchmarking different attention backends without code changes.

**See:** `references/attention-patterns.md` for complete replacement code with all algorithm paths.

### Sparse Attention (Advanced Optimization)

For video diffusion models with long sequences (>16K tokens), sparse attention patterns can skip computation on less-important tokens:
- **Grid-based sparsity** — Sample tokens on a regular grid for early denoising steps
- **Blockwise sparsity** — Skip entire spatial blocks based on attention score thresholds

This is model-specific and requires tuning the sparsity ratio per model architecture.

---

## 3. Operator Replacement (Normalization, Embedding, Activation)

Beyond attention, several operators have fused NPU equivalents that provide 1.2-3x speedup:

| Original | Ascend Replacement | Speedup |
|----------|--------------------|---------|
| Manual RMSNorm (`torch.rsqrt`) | `torch_npu.npu_rms_norm(x, weight)` | ~2x |
| `nn.LayerNorm(x)` | `fast_layernorm(norm_module, x)` from mindiesd | ~1.5x |
| Complex RoPE application | `rotary_position_embedding(x, cos, sin, rotated_mode=..., fused=True)` | ~2x |
| Complex RoPE application | `torch_npu.npu_apply_rotary_pos_emb(...)` | ~1.8x |

**See:** `references/operator-replacement.md` for before/after code for each operator.

### Key Pattern: Environment-Variable Dispatch

The Ascend adaptation uses env vars to select operator implementations at runtime:

```python
ALGO = int(os.environ.get("ALGO", "0"))        # Attention algorithm
ROPE_OPT = int(os.environ.get("ROPE_OPT", "0"))  # RoPE implementation
FAST_LN = int(os.environ.get("FAST_LAYERNORM", "0"))  # LayerNorm implementation
```

This pattern is worth adopting — it lets you benchmark different operator paths without redeploying code, and gives users knobs to tune for their specific hardware generation (910B vs 910C have different optimal paths).

---

## 4. Convolution Optimization

For video models using `CausalConv3d`, there's a subtle but important padding restructure:

**Original:** All padding (spatial + temporal) applied via `F.pad` before convolution
**Ascend:** Spatial padding moved INTO the Conv3d kernel, only temporal padding via `F.pad`

```python
# Original
self.padding = (0, 0, 0)  # No padding in Conv3d
x = F.pad(x, (spatial_w, spatial_w, spatial_h, spatial_h, temporal, 0))

# Ascend — let Conv3d handle spatial padding natively
self.padding = (0, pad_h, pad_w)  # Spatial padding in Conv3d kernel
self._padding = (0, 0, 0, 0, temporal, 0)  # Only temporal via F.pad
x = F.pad(x, self._padding)
```

Why this matters: NPU convolution kernels are optimized for standard padding patterns. Moving spatial padding into the kernel lets the NPU fuse the padding with the convolution, avoiding a separate memory-bound pad operation.

---

## 5. Distributed Parallelism

Ascend supports richer parallelism than typical CUDA setups. For diffusion inference, the key strategies are:

| Strategy | What It Parallelizes | When to Use |
|----------|---------------------|-------------|
| **CFG Parallel** | Conditional/unconditional branches | Always for CFG-based inference (2 devices) |
| **Ulysses SP** | Sequence dimension (attention) | Long sequences, video models |
| **Ring Attention** | Sequence dimension (ring-style) | Very long sequences, memory-limited |
| **Tensor Parallel** | Weight matrices | Large models that don't fit single device |
| **FSDP** | Full model sharding | Training and large-model inference |
| **VAE Patch Parallel** | VAE decode patches | High-resolution output |

These compose orthogonally. Total devices = `cfg_size × ulysses_size × ring_size × tp_size`.

**See:** `references/parallelism-guide.md` for setup code, process group configuration, and composition examples.

### CFG Parallelism (Unique to Ascend)

Standard diffusion inference runs the denoising UNet/DiT twice per step — once with the prompt (conditional) and once without (unconditional) for classifier-free guidance. CFG parallelism splits these across 2 devices:

```python
if cfg_parallel:
    # Each device runs ONE forward pass instead of TWO
    noise_pred = model(x, t, context)  # conditional OR unconditional
    noise_pred = all_gather(noise_pred)  # Combine results
    noise_pred = unconditional + guidance_scale * (conditional - unconditional)
```

This halves per-device memory and nearly doubles throughput for CFG-based inference.

---

## 6. VAE Optimization

VAE decode is often the memory bottleneck for high-resolution outputs. Key optimizations:

- **Patch Parallel Decode** — Split the latent spatially across devices, decode in parallel, stitch results
- **Tiling/Slicing** — For single-device: decode in spatial tiles with overlap to avoid seam artifacts
- **CPU Offload** — Move VAE to CPU during denoising, back to NPU for final decode

**See:** `references/parallelism-guide.md` (VAE section) for patch parallel implementation.

---

## 7. Quantization

mindiesd provides turnkey quantization for DiT/UNet models:

```python
from mindiesd import quantize

# find_quant_config_file returns (path, use_nz) tuple
quant_des_path, use_nz = find_quant_config_file(quant_dir)
quantize(model=model, quant_des_path=quant_des_path, use_nz=use_nz)
```

Supported schemes: W8A8 (dynamic), W4A4 (MXFP4 dual-scale), W8A8 (MXFP8).

**Key API details:**
- Parameter is `quant_des_path` (description path), NOT `quant_config_path`
- `use_nz` controls NZ compression format (True for w8a8_dynamic/w4a4, False for mxfp8)
- Config filenames follow pattern: `quant_model_description_<scheme>.json`

**See:** `references/quantization-guide.md` for config file format and FSDP integration patterns.

---

## 8. Attention Caching

For iterative denoising, adjacent timesteps produce similar attention patterns. Attention caching skips recomputation when the change is below a threshold:

```python
from mindiesd import CacheConfig, CacheAgent

config = CacheConfig(
    method="attention_cache",
    blocks_count=len(model.blocks) * 2 // cfg_size,  # Total cacheable blocks
    steps_count=sample_steps,                          # Total denoising steps
    step_start=2,                                      # Start caching at step 2
    step_interval=5,                                   # Recompute every 5 steps
    step_end=45,                                       # Stop caching at step 45
)
for block in model.blocks:
    block.cache = CacheAgent(config)
    # During inference: block.cache.apply(block.self_attn, hidden_states, ...)
```

Parameters to tune: `step_start`/`step_end` (caching window), `step_interval` (recompute frequency), `blocks_count` (number of cacheable blocks). Caching too aggressively degrades quality; start with a wide interval (step_interval=5) and narrow down.

---

## 9. Runtime Tuning

### Warmup Steps

NPU graph compilation happens on first execution. Always run 2 warmup steps before timing or actual generation:

```python
# Warmup with dummy inputs matching real shapes
for _ in range(2):
    with torch.no_grad():
        _ = model(dummy_input, dummy_timestep, dummy_context)
torch.npu.synchronize()
```

### Timing

Use NPU stream synchronization for accurate timing:

```python
torch.npu.synchronize()
start = time.time()
# ... generation ...
torch.npu.synchronize()
elapsed = time.time() - start
```

### Numerical Precision

For reproducible results across runs, generate random noise on CPU:

```python
if os.environ.get("PRECISION", "0") == "1":
    noise = torch.randn(shape, dtype=dtype, device="cpu").to("npu")
else:
    noise = torch.randn(shape, dtype=dtype, device="npu")
```

NPU random number generation may produce different sequences across runs even with the same seed. CPU-generated noise is deterministic.

### Environment Variables Summary

| Variable | Values | Default | Purpose |
|----------|--------|---------|---------|
| `ALGO` | 0, 1, 3 | 0 | Attention: 0=fused_attn, 1=laser_attn, 3=npu_fused_infer |
| `ROPE_OPT` | 0, 1 | 0 | 0=mindiesd RoPE, 1=torch_npu RoPE |
| `FAST_LAYERNORM` | 0, 1 | 0 | 0=standard, 1=mindiesd fast layernorm |
| `USE_SUB_HEAD` | 0, N | 0 | Split attention into N sub-head groups (memory saving) |
| `PRECISION` | 0, 1 | 0 | 1=CPU-based RNG for reproducibility |
| `T5_LOAD_CPU` | 0, 1 | 0 | 1=Load T5 text encoder on CPU first |

---

## 10. Dependencies

### Required
- `torch_npu` (matching your CANN version)
- `torch` (2.1.0 or 2.5.0/2.6.0 depending on CANN)

### Recommended for Acceleration
- `mindiesd` (MindIE SD — fused attention, RoPE, LayerNorm, quantization, caching)
- `yunchang==0.6.0` (ring attention support)

### Removed from CUDA Version
- `flash_attn` — replaced by torch_npu/mindiesd attention
- `xformers` — replaced by torch_npu/mindiesd attention
- `torchaudio` — typically unused in Ascend inference pipelines

---

## 11. Common Pitfalls

1. **Forgetting `allow_internal_format=False`** — Causes silent data corruption when mixing operator backends
2. **Using `nccl` backend** — Will crash. Must use `hccl` for Ascend distributed
3. **Flash attention imports at module level** — Guard with `try/except` or remove entirely
4. **FSDP `use_orig_params`** — Remove this parameter on Ascend; it's not supported the same way
5. **Timing without `npu.synchronize()`** — NPU ops are async; times will be wrong without sync
6. **Large batch VAE decode** — Will OOM. Use tiling, slicing, or patch parallel
7. **Skipping warmup** — First 1-2 iterations are 10-100x slower due to graph compilation
8. **bf16 for attention** — Always cast attention inputs to `torch.bfloat16` on Ascend for best performance; fp16 attention has known precision issues on 910B

---

## Reference Files

Read these for detailed code examples when implementing each adaptation layer:

| File | Contents | When to Read |
|------|----------|-------------|
| `references/device-migration-checklist.md` | Exhaustive find-and-replace list for CUDA→NPU | Starting any migration |
| `references/attention-patterns.md` | Full attention replacement with all algorithm paths | Replacing flash_attn/xformers |
| `references/operator-replacement.md` | Before/after for RMSNorm, RoPE, LayerNorm, Conv3d | Swapping individual operators |
| `references/parallelism-guide.md` | CFG/Ulysses/Ring/TP/FSDP/VAE parallel setup | Multi-device inference or training |
| `references/quantization-guide.md` | mindiesd quantization config and FSDP integration | Deploying quantized models |
