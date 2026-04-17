---
name: wan-ascend-adaptation
description: This skill provides comprehensive guidance for adapting Wan-series video generation models (Wan2.1/Wan2.2) from NVIDIA CUDA to Huawei Ascend NPU. It should be used when performing NPU migration of DiT-based video diffusion models, including device layer adaptation, operator replacement, distributed parallelism refactoring, attention optimization, VAE parallelization, and model quantization. This skill covers 9 major adaptation domains derived from real-world Wan2.2 CUDA-to-Ascend porting experience.
---

# Wan-Series Model Ascend NPU Adaptation Skill

## Purpose

Provide a systematic, step-by-step guide for adapting Wan-series (and similar DiT-based) video generation models from NVIDIA CUDA/GPU to Huawei Ascend NPU. The skill encodes 9 major adaptation domains covering every layer of the inference stack, from device initialization to distributed parallelism.

## When to Use

- Porting a Wan-series (Wan2.1 / Wan2.2) model from CUDA to Ascend NPU
- Adapting any DiT-based video diffusion model for Ascend hardware
- Optimizing inference performance on Ascend NPU (attention, quantization, VAE parallel)
- Setting up multi-card distributed inference on Atlas 800 series hardware
- Integrating MindIE SD acceleration library into a PyTorch video generation pipeline

## Adaptation Domains Overview

The adaptation work is organized into 9 domains. Each domain has a dedicated reference file under `references/` with detailed instructions, code patterns, and pitfalls.

| # | Domain | Reference File | Priority |
|---|--------|---------------|----------|
| 1 | Device Layer Adaptation | `references/01-device-layer.md` | **P0 — Must** |
| 2 | Operator Replacement | `references/02-operator-replacement.md` | **P0 — Must** |
| 3 | Precision Strategy | `references/03-precision-strategy.md` | **P0 — Must** |
| 4 | Attention Mechanism | `references/04-attention-mechanism.md` | **P1 — Critical** |
| 5 | Distributed Parallelism | `references/05-distributed-parallelism.md` | **P1 — Critical** |
| 6 | VAE Patch Parallel | `references/06-vae-patch-parallel.md` | **P2 — Important** |
| 7 | Model Quantization | `references/07-model-quantization.md` | **P2 — Important** |
| 8 | Sparse Attention (RainFusion) | `references/08-sparse-attention.md` | **P2 — Important** |
| 9 | Inference Pipeline Integration | `references/09-pipeline-integration.md` | **P1 — Critical** |

## Workflow

To adapt a Wan-series model to Ascend, follow these steps in order:

### Step 1: Device Layer Adaptation (Domain 1)

Read `references/01-device-layer.md` for complete guidance.

Key actions:
- Import `torch_npu` and `transfer_to_npu` at the entry point
- Configure NPU compile mode and internal format settings
- Replace `dist.init_process_group(backend="nccl")` with `backend="hccl"`
- Replace all `torch.amp.autocast('cuda', ...)` with `autocast('npu', ...)`
- Replace device type checks from `'cuda'` to `'npu'`

### Step 2: Operator Replacement (Domain 2)

Read `references/02-operator-replacement.md` for complete guidance.

Key actions:
- Replace RMSNorm with `torch_npu.npu_rms_norm()`
- Replace LayerNorm forward to remove `.float()` type casting
- Replace RoPE with `mindiesd.rotary_position_embedding()` fused operator
- Optionally enable `mindiesd.fast_layernorm` via `FAST_LAYERNORM` env var
- Replace Flash Attention with `mindiesd.attention_forward()` multi-backend dispatch

### Step 3: Precision Strategy (Domain 3)

Read `references/03-precision-strategy.md` for complete guidance.

Key actions:
- Lower sinusoidal embedding from float64 to float32
- Lower RoPE frequency from complex128 to complex64
- Change autocast dtype from float32 to bfloat16
- Remove `.float()` type conversions in normalization layers
- Use `PRECISION` env var to control random number device for cross-platform reproducibility

### Step 4: Attention Mechanism Adaptation (Domain 4)

Read `references/04-attention-mechanism.md` for complete guidance.

Key actions:
- Implement multi-backend attention dispatch via `ALGO` env var (0/1/3)
- Create `xFuserLongContextAttention` combining Ulysses + Ring Attention
- Integrate Attention Cache via `mindiesd.CacheAgent`
- Add sub-head splitting support via `USE_SUB_HEAD` env var

### Step 5: Distributed Parallelism Refactoring (Domain 5)

Read `references/05-distributed-parallelism.md` for complete guidance.

Key actions:
- Implement `ParallelConfig` with 4D parallelism: TP × SP × CFG
- Create `RankGenerator` for orthogonal process group assignment
- Create `GroupCoordinator` with dual-channel communication (HCCL + Gloo)
- Implement `TensorParallelApplicator` for automatic model sharding
- Implement CFG parallel to halve sampling loop forward passes

### Step 6: VAE Patch Parallel (Domain 6)

Read `references/06-vae-patch-parallel.md` for complete guidance.

Key actions:
- Implement spatial H×W slicing across NPUs
- Monkey-patch `F.conv3d`, `F.conv2d`, `F.interpolate`, `F.pad` for boundary exchange
- Use P2P communication for neighbor boundary data exchange
- Adjust VAE CausalConv3d padding strategy for compatibility

### Step 7: Model Quantization (Domain 7)

Read `references/07-model-quantization.md` for complete guidance.

Key actions:
- Use `msmodelslim` for W8A8 dynamic quantization
- Integrate `mindiesd.quantize()` for runtime quantization loading
- Handle FSDP + float8 compatibility via `patch_cast_buffers_for_float8()`

### Step 8: Sparse Attention — RainFusion (Domain 8)

Read `references/08-sparse-attention.md` for complete guidance.

Key actions:
- Implement RainFusion v1 (window-based Local/Global adaptive)
- Implement RainFusion v2 (blockwise Top-K sparse)
- Configure skip_timesteps for quality-speed tradeoff

### Step 9: Pipeline Integration (Domain 9)

Read `references/09-pipeline-integration.md` for complete guidance.

Key actions:
- Add warm-up generation steps for NPU operator compilation
- Configure `T5_LOAD_CPU` for flexible T5 loading strategy
- Add RoPE frequency cache (`freqs_list`) lifecycle management
- Implement multi-resolution VAE decode condition (`rank < 8`)
- Add performance timing with `stream.synchronize()`

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALGO` | `0` | Attention algorithm: 0=fused_attn_score, 1=ascend_laser_attention, 3=npu_fused_infer |
| `FAST_LAYERNORM` | `0` | Enable mindiesd fast LayerNorm |
| `USE_SUB_HEAD` | `0` | Sub-head group size for attention splitting |
| `T5_LOAD_CPU` | `0` | Load T5 model on CPU to save NPU memory |
| `PRECISION` | `0` | Generate random numbers on CPU for cross-platform reproducibility |
| `OVERLAP` | `0` | Enable FA-AllToAll communication overlap |
| `PYTORCH_NPU_ALLOC_CONF` | - | NPU memory allocation strategy |
| `TASK_QUEUE_ENABLE` | - | NPU task queue optimization |
| `CPU_AFFINITY_CONF` | - | CPU affinity configuration |

## Key Dependencies

| Library | Purpose |
|---------|---------|
| `torch_npu` | PyTorch Ascend NPU backend |
| `mindiesd` | MindIE Stable Diffusion acceleration (FA, RoPE, LayerNorm, quantize) |
| `msmodelslim` | Huawei model compression toolkit (W8A8 quantization) |
| `yunchang` | Sequence parallel framework (Ulysses + Ring Attention) |
| `torch_atb` | Ascend Transformer Boost operators |
| `atb_ops` | ATB fused matmul-allreduce operators |

## Notes

- This skill is derived from comparing Wan2.2-Original (CUDA) and Wan2.2-Ascend (NPU) codebases
- The Ascend version removes S2V (Speech-to-Video) and Animate tasks, focusing on T2V, I2V, and TI2V
- Hardware target: Atlas 800I A2 / Atlas 800T A2 with 8×64G NPU
- All adaptation patterns are applicable to similar DiT-based video diffusion architectures
