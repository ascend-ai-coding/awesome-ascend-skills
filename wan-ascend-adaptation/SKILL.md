---
name: wan-ascend-adaptation
description: Comprehensive guide for adapting Wan series video generation models (Wan2.2/2.1/1.3) and similar diffusion-based video frameworks to Huawei Ascend NPU. Covers 9 key adaptation areas: device migration (CUDA to torch_npu), attention mechanism replacement (FlashAttention to MindIE), normalization optimization (RMSNorm to npu_rms_norm), rotary position embedding (manual to mindiesd.rpe), distributed training (NCCL to HCCL), sparse attention (Rainfusion), NPU stream/event management, W8A8 quantization, and performance tuning via environment variables. Use when migrating video generation models to Ascend, optimizing DiT/VAE architectures for NPU, implementing sparse attention, setting up HCCL, or applying quantization.
keywords:
    - wan
    - wan-ascend-adaptation
    - video generation
    - diffusion-based video frameworks
    - mindiesd
---

# Wan-Ascend Adaptation Skill

> Adapt Wan series video generation models to Huawei Ascend NPU (torch_npu, MindIE, HCCL)

## When to Use

- Migrating Wan2.2/2.1/1.3 models from CUDA to Ascend NPU
- Optimizing video generation models (DiT, VAE, attention mechanisms) for NPU
- Implementing sparse attention (Rainfusion) on Ascend
- Setting up distributed training with HCCL backend
- Applying W8A8 quantization to video generation models
- Similar diffusion-based video generation frameworks

## Quick Overview

Wan2.2 is a diffusion-based video generation framework with 5 variants:

- T2V-A14B (Text-to-Video, 14B params)
- I2V-A14B (Image-to-Video, 14B params)
- TI2V-5B (Text+Image-to-Video, 5B params)
- S2V-14B (Sketch-to-Video, 14B params)
- Animate-14B (Animation, 14B params)

**Key Adaptation Areas** (9 categories):

1. **Device Management** - CUDA → torch_npu compatibility layer
2. **Attention Mechanism** - FlashAttention → MindIE (3 algorithms)
3. **Normalization** - Manual RMSNorm → npu_rms_norm
4. **Rotary Position Embedding** - Manual → mindiesd.rope
5. **Distributed Training** - NCCL → HCCL + enhanced parallelism
6. **Sparse Attention** - Rainfusion v1/v2 (NEW)
7. **Stream/Event Mgmt** - NPU synchronization primitives
8. **Quantization** - W8A8 via msmodelslim (NEW)
9. **Performance Tuning** - Environment variables

## Prerequisites

- **Hardware**: Huawei Ascend 910B/910C NPU
- **Software**:
  - torch >= 2.1.0 with torch_npu extension
  - CANN >= 8.3.RC3
  - MindIE Toolkit (attention kernels)
  - msmodelslim (quantization, optional)
  - HCCL backend (distributed)
- **Codebase**: Original Wan2.2 CUDA implementation

## File Structure

```
wan-ascend-adaptation/
├── SKILL.md                    # This file
└── references/
    ├── patterns/               # Adaptation patterns
    │   ├── device_migration.md
    │   ├── attention_mechanism.md
    │   ├── normalization.md
    │   ├── rope.md
    │   ├── distributed_training.md
    │   ├── rainfusion_sparse_attn.md
    │   ├── stream_events.md
    │   ├── quantization.md
    │   └── env_vars.md
    ├── examples/               # Code examples
    │   ├── minimal_migration.py
    │   ├── attention_comparison.py
    │   ├── distributed_setup.py
    │   └── quantization_pipeline.py
    ├── configs/                # Configuration templates
    │   ├── env_setup.sh
    │   ├── mindie_config.yaml
    │   └── quantization_config.json
    └── troubleshooting.md
```

## Adaptation Workflow

### Step 1: Environment Setup

1. Install torch_npu and dependencies
2. Configure environment variables (`[env_vars.md](references/patterns/env_vars.md)`)
3. Set up HCCL for distributed training

### Step 2: Device Migration

1. Replace `torch.cuda` with `torch_npu` (`[device_migration.md](references/patterns/device_migration.md)`)
2. Update device strings and initialization
3. Add torch_npu compatibility checks

### Step 3: Core Module Adaptation

1. **Attention**: Replace FlashAttention with MindIE (`[attention_mechanism.md](references/patterns/attention_mechanism.md)`)
2. **Normalization**: Use npu_rms_norm (`[normalization.md](references/patterns/normalization.md)`)
3. **RoPE**: Integrate mindiesd.rpe (`[rope.md](references/patterns/rope.md)`)

### Step 4: Distributed Training

1. Switch NCCL → HCCL (`[distributed_training.md](references/patterns/distributed_training.md)`)
2. Configure tensor/sequence parallel
3. Add VAE patch parallelism

### Step 5: Performance Optimization

1. Implement Rainfusion sparse attention (`[rainfusion_sparse_attn.md](references/patterns/rainfusion_sparse_attn.md)`)
2. Add stream/event overlap (`[stream_events.md](references/patterns/stream_events.md)`)
3. Tune environment variables

### Step 6: Quantization (Optional)

1. Apply W8A8 quantization (`[quantization.md](references/patterns/quantization.md)`)
2. Run calibration pipeline
3. Verify accuracy/performance trade-off

### Step 7: Testing & Verification

1. Run single-device inference
2. Test multi-device distributed training
3. Benchmark performance
4. Validate output quality

## Key Files to Modify


| File                               | Changes                      | Pattern Reference                                                            |
| ---------------------------------- | ---------------------------- | ---------------------------------------------------------------------------- |
| `generate.py`                      | torch_npu init, device setup | `[device_migration.md](references/patterns/device_migration.md)`             |
| `wan/modules/model.py`             | RMSNorm, attention, RoPE     | `[normalization.md](references/patterns/normalization.md)`                   |
| `wan/modules/attn_layer.py`        | NPU attention algorithms     | `[attention_mechanism.md](references/patterns/attention_mechanism.md)`       |
| `wan/modules/t5.py`                | torch_npu import             | `[device_migration.md](references/patterns/device_migration.md)`             |
| `wan/distributed/parallel_mgr.py`  | HCCL backend, NPU mgmt       | `[distributed_training.md](references/patterns/distributed_training.md)`     |
| `wan/distributed/tp_applicator.py` | Tensor parallel NPU ops      | `[distributed_training.md](references/patterns/distributed_training.md)`     |
| `wan/utils/rainfusion.py`          | Sparse attention (NEW)       | `[rainfusion_sparse_attn.md](references/patterns/rainfusion_sparse_attn.md)` |
| `wan/vae_patch_parallel.py`        | VAE spatial parallelism      | `[distributed_training.md](references/patterns/distributed_training.md)`     |
| `quant_wan22.py`                   | Quantization pipeline (NEW)  | `[quantization.md](references/patterns/quantization.md)`                     |


## Environment Variables

```bash
# Attention algorithm
export ALGO=1                    # 0: fused_attn, 1: laser_attn (default), 3: infer_attn

# Layer normalization optimization
export FAST_LAYERNORM=1          # 0: disabled, 1: enabled

# Communication overlap
export OVERLAP=1                 # Enable AllToAll overlap with FA computation

# NPU memory allocation
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512
export TASK_QUEUE_ENABLE=1

# CPU affinity
export CPU_AFFINITY_CONF=0-23:24-47:48-71:72-95
```

## Quick Start Examples

### Minimal Device Migration

```python
# Original CUDA code
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# NPU adaptation
import torch_npu
device = torch.device("cuda:0" if torch_npu.is_available() else "cpu")
model = model.to(device)
torch_npu.npu.set_device(local_rank)  # For distributed
```

### Attention Mechanism Replacement

```python
# Original FlashAttention
from flash_attn import flash_attn_func
output = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)

# NPU MindIE attention
import mindiesd
output = mindiesd.ops.attention_forward(
    q, k, v,
    attention_mask=None,
    dropout_p=0.0,
    causal=False,
    algo=int(os.environ.get("ALGO", "1"))  # Default: laser_attention
)
```

### RMSNorm Optimization

```python
# Manual RMSNorm
def rms_norm(x, weight, eps=1e-5):
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight

# NPU optimized
import torch_npu
output = torch_npu.npu_rms_norm(x, weight, eps=1e-5)
```

## Performance Optimization Checklist

- Enable `ALGO=1` (laser_attention) for best performance
- Set `FAST_LAYERNORM=1` for faster layer normalization
- Enable `OVERLAP=1` to overlap AllToAll with FA computation
- Configure CPU affinity for NUMA-aware scheduling
- Use Rainfusion sparse attention for long sequences (>1024 tokens)
- Apply W8A8 quantization for memory reduction
- Enable VAE patch parallelism for large resolutions
- Tune `PYTORCH_NPU_ALLOC_CONF` for optimal memory fragmentation
- Use NPU streams/events for async compute/comm overlap
- Profile with `torch.profiler` + NPU-specific metrics

## Common Issues & Solutions

### Issue: Out of Memory on NPU

**Solution**:

- Enable gradient checkpointing
- Reduce batch size or sequence length
- Use Rainfusion sparse attention
- Apply W8A8 quantization

### Issue: Slow attention computation

**Solution**:

- Switch to `ALGO=1` (laser_attention)
- Enable `OVERLAP=1` for AllToAll overlap
- Use sequence parallel for long sequences

### Issue: Distributed training hangs

**Solution**:

- Verify HCCL backend configuration
- Check NPU device mapping: `torch_npu.npu.set_device(local_rank)`
- Ensure torch_npu.synchronize() after communication ops

See `[troubleshooting.md](references/troubleshooting.md)` for detailed solutions.

## References

- **Pattern Guides**: See `[references/patterns/](references/patterns/)` for detailed adaptation patterns
- **Code Examples**: See `[references/examples/](references/examples/)` for working code snippets
- **Config Templates**: See `[references/configs/](references/configs/)` for environment setup
- **Troubleshooting**: See `[references/troubleshooting.md](references/troubleshooting.md)` for common issues

## Related Skills

- [@ascendc](../../ascendc) - AscendC custom operator development
- [@torch_npu](../../torch_npu) - torch_npu environment setup and API reference
- [@msmodelslim](../../msmodelslim) - Model quantization and compression
- [@hccl-test](../../hccl-test) - HCCL performance testing

## Version Compatibility


| Component   | Min Version | Recommended |
| ----------- | ----------- | ----------- |
| torch       | 2.1.0       | 2.2.0+      |
| torch_npu   | 2.1.0       | 2.2.0+      |
| CANN        | 8.3.RC3     | 8.5.0+      |
| MindIE      | 0.2.0       | 0.3.0+      |
| msmodelslim | 1.0.0       | 1.1.0+      |


## Contributing

When extending this Skill:

1. Add new patterns to `[references/patterns/](references/patterns/)`
2. Provide working examples in `[references/examples/](references/examples/)`
3. Update this SKILL.md with summary
4. Reference related patterns with relative links

---

**Next Steps**: Start with `[references/patterns/device_migration.md](references/patterns/device_migration.md)` for Step 1 of the workflow.