# Domain 1: Device Layer Adaptation

## Overview

The foundational adaptation layer that enables PyTorch code to run on Huawei Ascend NPU instead of NVIDIA CUDA GPU. This is the first and most critical step — without it, no other adaptation can function.

## 1.1 NPU Runtime Initialization

At the entry point of the application (e.g., `generate.py`), add the following initialization block **before any other torch operations**:

```python
import torch
import torch_npu

# Disable JIT compilation for stable NPU execution
torch_npu.npu.set_compile_mode(jit_compile=False)

# Disable internal format optimization (required for some operators)
torch.npu.config.allow_internal_format = False

# Auto-redirect CUDA calls to NPU (convenience wrapper)
from torch_npu.contrib import transfer_to_npu
```

### Why Each Setting Matters

| Setting | Purpose |
|---------|---------|
| `jit_compile=False` | Prevents unstable JIT graph compilation on NPU; uses precompiled binary mode |
| `allow_internal_format=False` | Ensures tensors use standard memory layout; some NPU-specific formats cause issues with certain operators |
| `transfer_to_npu` | Monkey-patches `torch.cuda.*` calls to automatically redirect to `torch.npu.*`, reducing manual migration effort |

## 1.2 Distributed Communication Backend

Replace NCCL with HCCL (Huawei Collective Communication Library):

```python
# Original (CUDA)
import torch.distributed as dist
dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)

# Ascend Adaptation
import torch.distributed as dist
import torch_npu
dist.init_process_group(backend="hccl", init_method="env://", world_size=world_size, rank=rank)
torch_npu.npu.set_device(local_rank)  # Must explicitly set NPU device
```

### HCCL vs NCCL

- HCCL is Huawei's equivalent of NCCL, optimized for Ascend NPU interconnects
- Supports all standard collective operations: all_reduce, all_gather, all_to_all, broadcast, etc.
- For CPU-side metadata communication, use a separate Gloo backend group (dual-channel pattern)

## 1.3 autocast Device String Replacement

Every occurrence of `torch.amp.autocast('cuda', ...)` must be changed to `'npu'`:

```python
# Original
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    output = model(input)

# Ascend
with torch.amp.autocast('npu', dtype=torch.bfloat16):
    output = model(input)
```

**Search pattern for finding all occurrences:**
```
grep -rn "autocast.*cuda" --include="*.py"
```

## 1.4 Device Type Checks

Replace all device type string comparisons:

```python
# Original
if next(model.parameters()).device.type == 'cuda':
    model.cpu()

# Ascend
if next(model.parameters()).device.type == 'npu':
    model.cpu()
```

## 1.5 NPU Memory Management

```python
# Original
torch.cuda.empty_cache()
torch.cuda.synchronize()
seed_g = torch.Generator(device='cuda')

# Ascend (note: transfer_to_npu may handle some of these automatically)
torch.npu.empty_cache()
torch.npu.synchronize()  # or stream.synchronize() for specific streams
seed_g = torch.Generator(device='npu')
```

## 1.6 NPU Device Placement

```python
# Original
model.to('cuda')
tensor = tensor.to('cuda:0')

# Ascend
model.to('npu')
tensor = tensor.to(f'npu:{local_rank}')
```

## 1.7 Environment Variables for NPU Configuration

Set these environment variables before launching:

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True  # Memory allocation strategy
export TASK_QUEUE_ENABLE=2                                # NPU task queue optimization
export CPU_AFFINITY_CONF=1                                # CPU affinity for NPU processes
```

## Pitfalls

1. **`transfer_to_npu` does NOT cover everything**: While it redirects many CUDA calls, explicit device strings in `autocast`, `dist.init_process_group`, and `Generator` still need manual replacement.
2. **HCCL initialization order**: `torch_npu.npu.set_device(local_rank)` must be called AFTER `dist.init_process_group` but BEFORE any tensor operations.
3. **`torch.cuda.empty_cache()` in FSDP**: The `fsdp.py` `free_model()` function may still contain `torch.cuda.empty_cache()`. With `transfer_to_npu`, this will be auto-redirected, but be aware of it during debugging.
