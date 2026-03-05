# Device Migration Pattern

Migrate from CUDA to torch_npu with compatibility layer.

## Key Changes

### Import Statements
```python
# Original
import torch

# NPU adaptation
import torch
import torch_npu  # NPU-specific operations
```

### Device Detection
```python
# Original
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# NPU adaptation (torch_npu provides compatibility layer)
device = torch.device("cuda:0" if torch_npu.is_available() else "cpu")
# Note: "cuda:" strings work via torch_npu compatibility
```

### Distributed Device Setup
```python
# Original (NCCL)
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

# NPU adaptation (HCCL)
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch_npu.npu.set_device(local_rank)  # Explicit NPU device setting
```

### Model Device Transfer
```python
# Both work identically
model = model.to(device)
model = model.half()  # FP16 conversion
```

## File Modifications

### generate.py
```python
# Add at top
import torch_npu
import torch_npu.contrib  # Optional for contrib modules

# Replace device initialization
def setup_device():
    if torch_npu.is_available():
        torch_npu.npu.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")

# Add NPU-specific settings
torch_npu.set_option('ACL_GRAPH_COMPILE_WITH_FP16', 1)  # FP16 graph compilation
```

### wan/modules/t5.py
```python
# Add NPU import
import torch_npu

# No other changes needed if using "cuda:" device strings
```

## Compatibility Notes

### Device String Compatibility
torch_npu provides compatibility layer for device strings:
- `"cuda:0"` → NPU device 0
- `"cuda"` → Default NPU device
- `"cpu"` → CPU

### Automatic Casting
```python
# Mixed precision works with torch.amp.autocast
with torch.amp.autocast('npu', dtype=torch.bfloat16):
    output = model(input)
```

## Common Patterns

### Pattern 1: Safe Device Detection
```python
def get_device():
    """Get best available device (NPU > CUDA > CPU)"""
    if torch_npu.is_available():
        return torch.device("cuda")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
```

### Pattern 2: Distributed Initialization
```python
import os
import torch_npu
import torch.distributed as dist

def setup_distributed():
    if not dist.is_initialized():
        # HCCL backend for NPU
        backend = "hccl" if torch_npu.is_available() else "nccl"
        dist.init_process_group(backend=backend)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch_npu.npu.set_device(local_rank)

    return local_rank, dist.get_rank(), dist.get_world_size()
```

### Pattern 3: Memory Pinning
```python
# DataLoader with pinned memory (works on both CUDA and NPU)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    pin_memory=torch_npu.is_available(),  # Pin for NPU
    num_workers=4
)
```

## Environment Variables

```bash
# NPU device selection
export ASCEND_VISIBLE_DEVICES=0,1,2,3

# Memory allocation strategy
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512

# Enable NPU allocator debugging
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

## Troubleshooting

### Issue: "cuda:0" device not found
**Cause**: torch_npu compatibility layer not working
**Solution**:
```python
# Use explicit NPU device name
device = torch.device(f"npu:{local_rank}")
```

### Issue: Device mismatch in distributed training
**Cause**: torch_npu.npu.set_device() not called
**Solution**: Always call after dist.init_process_group()

### Issue: Out of memory on NPU
**Solutions**:
1. Reduce batch size
2. Enable gradient checkpointing
3. Use mixed precision: `torch.amp.autocast('npu', dtype=torch.bfloat16)`
4. Configure memory allocator: `export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256`

## References

- [torch_npu Documentation](https://github.com/Ascend/pytorch)
- MindIE Operator Guide
- HCCL Programming Guide

## Examples

See [`examples/minimal_migration.py`](../examples/minimal_migration.py) for complete working example.
