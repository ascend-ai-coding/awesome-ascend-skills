# Distributed Parallelism Guide for Ascend Diffusion Models

How to configure multi-NPU inference and training for diffusion models on Ascend.

## Table of Contents
1. [Parallelism Strategies Overview](#1-parallelism-strategies-overview)
2. [CFG Parallelism](#2-cfg-parallelism)
3. [Sequence Parallelism (Ulysses)](#3-sequence-parallelism-ulysses)
4. [Ring Attention](#4-ring-attention)
5. [Tensor Parallelism](#5-tensor-parallelism)
6. [FSDP on Ascend](#6-fsdp-on-ascend)
7. [VAE Patch Parallelism](#7-vae-patch-parallelism)
8. [Composing Strategies](#8-composing-strategies)
9. [Process Group Setup](#9-process-group-setup)

---

## 1. Parallelism Strategies Overview

| Strategy | Parallelizes | Devices | Best For |
|----------|-------------|---------|----------|
| CFG Parallel | CFG branches | 2 | Any CFG inference (always use if available) |
| Ulysses SP | Sequence dim | 2-8 | Video/long-sequence models |
| Ring Attention | Sequence dim | 2-8 | Very long sequences, memory-limited |
| Tensor Parallel | Weight matrices | 2-8 | Large models (>10B params) |
| FSDP | Full model | any | Training, large-model inference |
| VAE Patch | VAE spatial | 2-8 | High-resolution decode |

**Composition rule:** Strategies are orthogonal.
```
total_devices = cfg_size × ulysses_size × ring_size × tp_size
```

---

## 2. CFG Parallelism

Classifier-Free Guidance runs the model twice per denoising step (conditional + unconditional). CFG parallelism splits these across 2 devices.

### Standard CFG (Single Device)
```python
# Standard: two forward passes on one device
noise_uncond = model(x, t, null_context)
noise_cond = model(x, t, context)
noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
```

### CFG Parallel (2 Devices)
```python
import torch.distributed as dist

def cfg_parallel_forward(model, x, t, context, null_context, guidance_scale, cfg_group):
    """
    Each device runs ONE forward pass. Results combined via all_gather.
    """
    rank_in_cfg = dist.get_rank(cfg_group)
    
    if rank_in_cfg == 0:
        # Device 0: unconditional
        noise = model(x, t, null_context)
    else:
        # Device 1: conditional
        noise = model(x, t, context)
    
    # Gather both results to all devices
    gathered = [torch.zeros_like(noise) for _ in range(2)]
    dist.all_gather(gathered, noise, group=cfg_group)
    
    noise_uncond, noise_cond = gathered[0], gathered[1]
    return noise_uncond + guidance_scale * (noise_cond - noise_uncond)
```

**Benefits:** Halves per-device memory, nearly doubles throughput. The all_gather communication cost is small relative to the full model forward pass.

---

## 3. Sequence Parallelism (Ulysses)

Splits the sequence dimension across devices. Each device holds a slice of the sequence and runs full attention within its slice, then communicates.

### Setup
```python
from yunchang import set_seq_parallel_pg, UlyssesAttention

# Create sequence parallel process group
sp_group = dist.new_group(ranks=[0, 1, 2, 3])  # 4-way SP
set_seq_parallel_pg(sp_group)

# Wrap attention
ulysses_attn = UlyssesAttention()
```

### Usage in Transformer Block
```python
class DiTBlock(nn.Module):
    def __init__(self, ...):
        self.ulysses_attn = UlyssesAttention()
    
    def forward(self, x, context=None):
        # x is already split along sequence dim by the SP framework
        q, k, v = self.to_qkv(x)
        
        if context is not None:
            # Cross attention: gather full context
            attn_out = self.ulysses_attn(
                q, k, v,
                joint_tensor_key=context_k,
                joint_tensor_value=context_v,
            )
        else:
            attn_out = self.ulysses_attn(q, k, v)
        
        return self.proj(attn_out)
```

### Input Splitting
```python
# Before the model forward, split input along sequence dim
def split_sequence(x, sp_group):
    """
    x: [batch, seq_len, dim]
    Returns: [batch, seq_len // sp_size, dim]
    """
    sp_size = dist.get_world_size(sp_group)
    sp_rank = dist.get_rank(sp_group)
    seq_len = x.shape[1]
    chunk_size = seq_len // sp_size
    return x[:, sp_rank * chunk_size:(sp_rank + 1) * chunk_size]
```

---

## 4. Ring Attention

Alternative to Ulysses for very long sequences. Each device holds a chunk of KV and rotates it ring-style.

```python
from yunchang import LongContextAttention

ring_attn = LongContextAttention(ring_impl_type="zigzag")

# In transformer block
output = ring_attn(q, k, v)
```

Ring attention has higher communication cost than Ulysses but better memory scaling for very long sequences (>64K tokens).

---

## 5. Tensor Parallelism

Splits weight matrices across devices. Each device computes a slice of the output, then all-reduces.

### TensorParallelApplicator Pattern
```python
class TensorParallelApplicator:
    """Applies tensor parallelism to model layers."""
    
    def __init__(self, tp_group):
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)
    
    def parallelize_linear(self, module, name, style="column"):
        """
        Split a linear layer across TP group.
        style="column": split output dim (for QKV projections)
        style="row": split input dim (for output projections)
        """
        linear = getattr(module, name)
        
        if style == "column":
            # Split weight along output dim
            chunk_size = linear.out_features // self.tp_size
            weight = linear.weight[self.tp_rank * chunk_size:(self.tp_rank + 1) * chunk_size]
            bias = linear.bias[self.tp_rank * chunk_size:(self.tp_rank + 1) * chunk_size] if linear.bias is not None else None
        elif style == "row":
            # Split weight along input dim
            chunk_size = linear.in_features // self.tp_size
            weight = linear.weight[:, self.tp_rank * chunk_size:(self.tp_rank + 1) * chunk_size]
            bias = linear.bias if self.tp_rank == 0 else None  # Only rank 0 adds bias
        
        new_linear = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        new_linear.weight = nn.Parameter(weight)
        if bias is not None:
            new_linear.bias = nn.Parameter(bias)
        
        setattr(module, name, new_linear)
```

### Typical Application
```python
tp_applicator = TensorParallelApplicator(tp_group)

for block in model.transformer_blocks:
    # Column parallel for QKV
    tp_applicator.parallelize_linear(block.attn, "to_q", "column")
    tp_applicator.parallelize_linear(block.attn, "to_k", "column")
    tp_applicator.parallelize_linear(block.attn, "to_v", "column")
    # Row parallel for output projection
    tp_applicator.parallelize_linear(block.attn, "to_out", "row")
    # Column parallel for FFN first layer
    tp_applicator.parallelize_linear(block.ffn, "fc1", "column")
    # Row parallel for FFN second layer
    tp_applicator.parallelize_linear(block.ffn, "fc2", "row")
```

---

## 6. FSDP on Ascend

FSDP (Fully Sharded Data Parallelism) works on Ascend with the HCCL backend. Key differences from CUDA:

### Setup
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

def shard_model(model, device_id, param_dtype=torch.bfloat16):
    """
    FSDP setup for Ascend NPU.
    Note: use_orig_params is NOT used on Ascend.
    """
    mp_policy = MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        device_id=device_id,
        # Do NOT pass use_orig_params on Ascend
        # Do NOT pass use_lora on Ascend
    )
    
    return model
```

### FSDP + Quantization Compatibility

When using quantized models with FSDP, float8 buffer types need patching:

```python
def patch_cast_buffers_for_float8(model):
    """
    FSDP's _cast_buffers doesn't handle float8 types.
    Patch to skip casting for float8 buffers.
    """
    original_cast = FSDP._cast_buffers
    
    def patched_cast(self, *args, **kwargs):
        # Filter out float8 buffers before casting
        float8_buffers = {}
        for name, buf in self.named_buffers():
            if buf.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                float8_buffers[name] = buf
        
        original_cast(self, *args, **kwargs)
        
        # Restore float8 buffers
        for name, buf in float8_buffers.items():
            # Re-assign the original float8 buffer
            parts = name.split('.')
            module = self
            for part in parts[:-1]:
                module = getattr(module, part)
            setattr(module, parts[-1], buf)
    
    FSDP._cast_buffers = patched_cast
```

---

## 7. VAE Patch Parallelism

For high-resolution output (e.g., 1080p video), VAE decode can OOM on a single device. Patch parallelism splits the latent spatially:

```python
class VAEPatchParallel:
    """
    Context manager that patches VAE forward to decode in spatial patches
    distributed across devices.
    """
    
    def __init__(self, vae, parallel_group, overlap=4):
        self.vae = vae
        self.group = parallel_group
        self.world_size = dist.get_world_size(parallel_group)
        self.rank = dist.get_rank(parallel_group)
        self.overlap = overlap  # Overlap pixels to avoid seam artifacts
    
    def decode(self, latent):
        """
        latent: [batch, channels, frames, height, width] (for video)
        or [batch, channels, height, width] (for image)
        """
        is_video = latent.dim() == 5
        spatial_dim = -1  # Split along width
        
        total_size = latent.shape[spatial_dim]
        chunk_size = total_size // self.world_size
        
        # Each device decodes its spatial chunk (with overlap)
        start = max(0, self.rank * chunk_size - self.overlap)
        end = min(total_size, (self.rank + 1) * chunk_size + self.overlap)
        
        if is_video:
            chunk = latent[..., start:end]
        else:
            chunk = latent[..., start:end]
        
        # Decode locally
        decoded_chunk = self.vae.decode(chunk)
        
        # Trim overlap
        trim_start = self.overlap if self.rank > 0 else 0
        trim_end = -self.overlap if self.rank < self.world_size - 1 else None
        decoded_chunk = decoded_chunk[..., trim_start:trim_end]
        
        # Gather all chunks
        all_chunks = [torch.zeros_like(decoded_chunk) for _ in range(self.world_size)]
        dist.all_gather(all_chunks, decoded_chunk, group=self.group)
        
        return torch.cat(all_chunks, dim=spatial_dim)
```

---

## 8. Composing Strategies

### Example: 8-device Setup for Video Generation
```
8 NPUs total:
- CFG parallel: 2 groups (conditional / unconditional)
- Within each CFG group: 4-way Ulysses SP

cfg_size = 2
ulysses_size = 4
total = 2 × 4 = 8 devices
```

### Process Group Creation
```python
def create_parallel_groups(world_size, cfg_size=1, ulysses_size=1, ring_size=1, tp_size=1):
    """
    Create orthogonal process groups for composed parallelism.
    """
    assert world_size == cfg_size * ulysses_size * ring_size * tp_size
    
    groups = {}
    
    # CFG groups: consecutive pairs
    if cfg_size > 1:
        for i in range(0, world_size, cfg_size):
            ranks = list(range(i, i + cfg_size))
            group = dist.new_group(ranks)
            if dist.get_rank() in ranks:
                groups['cfg'] = group
    
    # SP groups: within each CFG group
    if ulysses_size > 1:
        for cfg_id in range(world_size // (cfg_size * ulysses_size)):
            base = cfg_id * cfg_size * ulysses_size
            for cfg_rank in range(cfg_size):
                ranks = [base + cfg_rank + i * cfg_size for i in range(ulysses_size)]
                group = dist.new_group(ranks)
                if dist.get_rank() in ranks:
                    groups['sp'] = group
    
    return groups
```

---

## 9. Process Group Setup

### Full Initialization Pattern
```python
import torch
import torch.distributed as dist
import torch_npu

def init_parallel_env(cfg_size=1, ulysses_size=1, ring_size=1, tp_size=1):
    """
    Initialize distributed environment for Ascend.
    """
    # Basic distributed setup
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device
    local_rank = rank % torch.npu.device_count()
    torch.npu.set_device(local_rank)
    
    # Create parallel groups
    groups = create_parallel_groups(world_size, cfg_size, ulysses_size, ring_size, tp_size)
    
    return rank, world_size, groups

def finalize_parallel_env():
    """Clean up distributed environment."""
    dist.destroy_process_group()
```

### Launch Command
```bash
# 8-NPU launch
torchrun --nproc_per_node=8 --master_port=29500 \
    generate.py \
    --cfg_size 2 \
    --ulysses_size 4 \
    --prompt "..."
```
