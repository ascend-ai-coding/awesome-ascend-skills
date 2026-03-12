# Device Migration Checklist: CUDA → Ascend NPU

Exhaustive find-and-replace patterns for migrating PyTorch code from CUDA to Ascend NPU.

## Table of Contents
1. [Imports](#1-imports)
2. [Device Strings](#2-device-strings)
3. [Distributed Backend](#3-distributed-backend)
4. [AMP / Autocast](#4-amp--autocast)
5. [Device Checks](#5-device-checks)
6. [Compilation & Format](#6-compilation--format)
7. [Stream & Synchronization](#7-stream--synchronization)
8. [Memory Management](#8-memory-management)
9. [Device Properties](#9-device-properties)

---

## 1. Imports

Add at the top of every entry point (train.py, generate.py, etc.):

```python
import torch
import torch_npu

# Option A: Monkey-patch (quick migration)
from torch_npu.contrib import transfer_to_npu
# After this, most torch.cuda.* calls route to NPU automatically

# Option B: Explicit (production)
# No monkey-patch — replace all references manually
```

If using monkey-patch, you still need to fix items in sections 3-6 below.

---

## 2. Device Strings

### Simple replacements
```python
# Before                              # After
torch.device("cuda")                  torch.device("npu")
torch.device("cuda:0")               torch.device("npu:0")
torch.device(f"cuda:{rank}")         torch.device(f"npu:{rank}")
.to("cuda")                          .to("npu")
.cuda()                              .npu()
```

### With transfer_to_npu
If using monkey-patch, most of these are handled automatically. But explicit `torch.device("cuda:N")` in configs or hardcoded device maps still need fixing.

### Device map patterns (HuggingFace models)
```python
# Before
model = AutoModel.from_pretrained(path, device_map="auto")
# The "auto" device map works with transfer_to_npu

# Before — explicit cuda device map
device_map = {"encoder": "cuda:0", "decoder": "cuda:1"}
# After
device_map = {"encoder": "npu:0", "decoder": "npu:1"}
```

---

## 3. Distributed Backend

This is NOT handled by transfer_to_npu — always fix manually.

```python
# Before
dist.init_process_group(backend="nccl")
torch.distributed.init_process_group("nccl")

# After
dist.init_process_group(backend="hccl")
torch.distributed.init_process_group("hccl")
```

Environment variables:
```bash
# Before
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0

# After
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
# HCCL uses different env vars — check CANN docs for your version
```

---

## 4. AMP / Autocast

```python
# Before
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
with torch.cuda.amp.autocast():
with torch.autocast(device_type='cuda'):

# After
with torch.amp.autocast('npu', dtype=torch.bfloat16):
with torch.npu.amp.autocast():
with torch.autocast(device_type='npu'):
```

GradScaler:
```python
# Before
scaler = torch.cuda.amp.GradScaler()
scaler = torch.amp.GradScaler('cuda')

# After
scaler = torch.npu.amp.GradScaler()
scaler = torch.amp.GradScaler('npu')
```

---

## 5. Device Checks

```python
# Before
if tensor.device.type == 'cuda':
if str(device) == 'cuda':
if torch.cuda.is_available():

# After
if tensor.device.type == 'npu':
if str(device) == 'npu':
if torch.npu.is_available():
```

### CPU offload patterns (Diffusers)
```python
# Diffusers enable_model_cpu_offload checks device type internally
# If using transfer_to_npu, patch the check:
if hasattr(self, '_offload_gpu_id'):
    # Diffusers checks for 'cuda' — may need patching
    pass
```

---

## 6. Compilation & Format

Add near the top of your entry point, BEFORE model creation:

```python
import torch_npu

# Disable JIT compilation — critical for iterative denoising
torch_npu.npu.set_compile_mode(jit_compile=False)

# Force standard tensor format — prevents operator incompatibility
torch.npu.config.allow_internal_format = False
```

Why `jit_compile=False`:
- NPU JIT compiles operator graphs on first execution
- Diffusion denoising loops have varying shapes/conditions per step
- JIT recompilation per step destroys performance
- Disabling JIT uses pre-compiled operator kernels instead

Why `allow_internal_format=False`:
- NPU has internal tensor formats optimized for specific operators
- When mixing torch_npu ops with mindiesd ops, format mismatches cause silent errors
- Forcing standard format ensures all operators see the same memory layout

---

## 7. Stream & Synchronization

```python
# Before
torch.cuda.synchronize()
torch.cuda.synchronize(device)
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
torch.cuda.current_stream()

# After
torch.npu.synchronize()
torch.npu.synchronize(device)
stream = torch.npu.Stream()
with torch.npu.stream(stream):
torch.npu.current_stream()
```

### Timing pattern
```python
# Before
torch.cuda.synchronize()
start = time.time()
# ... work ...
torch.cuda.synchronize()
elapsed = time.time() - start

# After (identical structure, different API)
torch.npu.synchronize()
start = time.time()
# ... work ...
torch.npu.synchronize()
elapsed = time.time() - start
```

---

## 8. Memory Management

```python
# Before
torch.cuda.empty_cache()
torch.cuda.memory_allocated()
torch.cuda.max_memory_allocated()
torch.cuda.reset_peak_memory_stats()
torch.cuda.memory_reserved()

# After
torch.npu.empty_cache()
torch.npu.memory_allocated()
torch.npu.max_memory_allocated()
torch.npu.reset_peak_memory_stats()
torch.npu.memory_reserved()
```

---

## 9. Device Properties

```python
# Before
torch.cuda.device_count()
torch.cuda.get_device_name(0)
torch.cuda.get_device_properties(0)
torch.cuda.set_device(rank)

# After
torch.npu.device_count()
torch.npu.get_device_name(0)
torch.npu.get_device_properties(0)
torch.npu.set_device(rank)
```

### Hardware detection pattern
```python
# Detect specific Ascend chip generation
device_name = torch_npu.npu.get_device_name(0)
is_910b = '95' in device_name   # Ascend 910B contains "95" in device name
is_910c = '910C' in device_name or '100' in device_name  # Check your CANN docs
```

This is useful for enabling/disabling features that work only on certain generations.

---

## Checklist Summary

Use this as a grep/sed list:

| Find | Replace | Auto-handled by transfer_to_npu? |
|------|---------|----------------------------------|
| `"cuda"` (device string) | `"npu"` | Yes (mostly) |
| `.cuda()` | `.npu()` | Yes |
| `"nccl"` (dist backend) | `"hccl"` | **No** |
| `torch.cuda.amp` | `torch.npu.amp` | Partial |
| `torch.amp.autocast('cuda')` | `torch.amp.autocast('npu')` | **No** |
| `device.type == 'cuda'` | `device.type == 'npu'` | **No** |
| `torch.cuda.synchronize` | `torch.npu.synchronize` | Yes |
| `torch.cuda.Stream` | `torch.npu.Stream` | Yes |
| `torch.cuda.empty_cache` | `torch.npu.empty_cache` | Yes |
| `torch.cuda.device_count` | `torch.npu.device_count` | Yes |
| `flash_attn` imports | Remove / replace | **No** |
| `xformers` imports | Remove / replace | **No** |
