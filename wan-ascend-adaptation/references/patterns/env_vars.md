# Ascend NPU Environment Variables

Environment variables are the first line of optimization for Ascend NPU workloads. Setting them correctly before training or inference begins can yield 10-30% performance improvements without any code changes.

## Overview

Ascend NPU environment variables control:

- **Attention algorithms** - Select optimized kernel implementations
- **Memory allocation** - Reduce fragmentation and OOM errors  
- **Communication overlap** - Hide communication latency behind computation
- **CPU affinity** - Minimize NUMA crossing overhead
- **Operator fusion** - Enable fused kernels for better throughput

These variables should be set in your shell environment before launching Python processes. For distributed training, set them consistently across all nodes.

## Critical Variables

### ALGO - Attention Algorithm Selection

Controls which attention kernel implementation to use.

```bash
export ALGO=1    # 0=fused_attn, 1=laser_attn (default), 3=infer_attn
```

**Values**:
- `0` - fused_attn: General purpose fused attention, stable baseline
- `1` - laser_attn: Optimized for training throughput, **recommended for most cases**
- `3` - infer_attn: Optimized for inference latency, better for generation

**When to use**:
- **Training**: `ALGO=1` typically gives best throughput
- **Inference**: `ALGO=3` may reduce latency for autoregressive generation
- **Debugging**: `ALGO=0` if you encounter numerical issues

**Impact**: 15-25% performance difference between algorithms depending on sequence length.

### FAST_LAYERNORM - Layer Normalization Optimization

Enables NPU-optimized layer normalization kernels.

```bash
export FAST_LAYERNORM=1    # 0=disabled, 1=enabled
```

**Behavior**:
- When enabled: Uses fused NPU kernels that combine multiple operations
- When disabled: Falls back to PyTorch composite operations

**Impact**: 5-10% speedup for transformer models where layer norm is frequent.

**Note**: Should be safe to enable for most models. Disable only if you encounter numerical convergence issues during training.

### OVERLAP - Communication/Computation Overlap

Enables overlapping of AllToAll collective operations with FlashAttention computation.

```bash
export OVERLAP=1    # Enable overlap (unset or 0 to disable)
```

**Effect**:
- AllToAll communication happens concurrently with FA forward/backward passes
- Reduces effective communication time in distributed training
- Most beneficial for tensor/sequence parallel configurations

**Impact**: 10-20% end-to-end training speedup in distributed setups with frequent communication.

**Requirements**: Only effective when using distributed training with tensor or sequence parallelism.

## Memory Management Variables

### PYTORCH_NPU_ALLOC_CONF - Memory Allocator Configuration

Fine-tunes NPU memory allocation behavior to reduce fragmentation.

```bash
# Recommended configuration
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"

# Alternative for high-memory workloads
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:1024,expandable_segments:True"
```

**Options**:
- `max_split_size_mb:N` - Maximum block size before forcing a new allocation
- `expandable_segments:True` - Allow segments to grow dynamically
- `garbage_collection_threshold:0.8` - Trigger GC when 80% of memory is allocated

**When to tune**:
- **OOM errors despite available memory**: Reduce max_split_size_mb
- **Memory fragmentation issues**: Enable expandable_segments
- **Long-running training**: Add garbage_collection_threshold

**Example configurations**:

| Scenario | Configuration |
|----------|---------------|
| Standard training | `max_split_size_mb:512,expandable_segments:True` |
| Large batch sizes | `max_split_size_mb:1024,expandable_segments:True` |
| Memory-constrained | `max_split_size_mb:256,garbage_collection_threshold:0.8` |
| Inference only | `max_split_size_mb:512` |

### TASK_QUEUE_ENABLE - Task Queue Optimization

Enables asynchronous task queue for NPU operations.

```bash
export TASK_QUEUE_ENABLE=1    # 0=disabled, 1=enabled
```

**Behavior**:
- When enabled: Operations are queued and dispatched asynchronously
- When disabled: Synchronous execution, easier debugging but slower

**Impact**: 5-15% throughput improvement by better utilizing NPU compute units.

**Debug tip**: Set to `0` when debugging to get clearer stack traces.

## System Optimization Variables

### CPU_AFFINITY_CONF - CPU Affinity Binding

Binds NPU devices to specific CPU cores for NUMA-aware scheduling.

```bash
# Format: device0_cores:device1_cores:device2_cores:device3_cores
export CPU_AFFINITY_CONF="0-23:24-47:48-71:72-95"
```

**Configuration for different setups**:

**4-device server (96 cores)**:
```bash
export CPU_AFFINITY_CONF="0-23:24-47:48-71:72-95"
```

**8-device server (192 cores)**:
```bash
export CPU_AFFINITY_CONF="0-23:24-47:48-71:72-95:96-119:120-143:144-167:168-191"
```

**2-device workstation (64 cores)**:
```bash
export CPU_AFFINITY_CONF="0-31:32-63"
```

**Benefits**:
- Reduces memory access latency through NUMA locality
- Prevents CPU core contention between devices
- Improves data transfer bandwidth between CPU and NPU

**To determine your core count**:
```bash
nproc                    # Total cores
lscpu | grep NUMA        # NUMA node layout
```

### TOKENIZERS_PARALLELISM - Tokenizer Parallelism Control

Controls whether HuggingFace tokenizers use parallel processing.

```bash
export TOKENIZERS_PARALLELISM=false    # true or false
```

**Why disable**:
- Tokenizer parallelism can conflict with PyTorch NPU operations
- Prevents fork-safety warnings and potential deadlocks
- NPU workloads are already highly parallelized

**Set this especially when**:
- Using data loading with multiple workers
- Running distributed training
- Seeing "The current process just got forked" warnings

### Additional Variables

```bash
# Disable Python hash randomization for reproducibility
export PYTHONHASHSEED=0

# NCCL/HCCL debugging (only when debugging distributed issues)
# export HCCL_DEBUG=INFO

# NPU-specific profiling
# export NPU_PROFILING=1
```

## Complete Setup Script

Create `setup_npu_env.sh`:

```bash
#!/bin/bash
# Ascend NPU Environment Setup Script
# Source this file before running training/inference: source setup_npu_env.sh

# ============================================
# Critical Performance Variables
# ============================================

# Attention algorithm: 1=laser_attn (best for training)
export ALGO=1

# Enable fast layer normalization
export FAST_LAYERNORM=1

# Enable AllToAll overlap with FlashAttention
export OVERLAP=1

# ============================================
# Memory Management
# ============================================

# Memory allocator configuration
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"

# Enable task queue for async execution
export TASK_QUEUE_ENABLE=1

# ============================================
# System Optimization
# ============================================

# CPU affinity - adjust based on your hardware
# Detect core count and set affinity automatically
CORES=$(nproc)
if [ $CORES -ge 96 ]; then
    # 4-device server (96+ cores)
    export CPU_AFFINITY_CONF="0-23:24-47:48-71:72-95"
elif [ $CORES -ge 64 ]; then
    # 2-device workstation (64+ cores)
    export CPU_AFFINITY_CONF="0-31:32-63"
else
    # Single device or small system
    export CPU_AFFINITY_CONF="0-$((CORES/2-1)):$((CORES/2))-$((CORES-1))"
fi

# Disable tokenizer parallelism to avoid conflicts
export TOKENIZERS_PARALLELISM=false

# Reproducibility
export PYTHONHASHSEED=0

# ============================================
# Verification
# ============================================

echo "NPU Environment Variables Set:"
echo "  ALGO=$ALGO"
echo "  FAST_LAYERNORM=$FAST_LAYERNORM"
echo "  OVERLAP=$OVERLAP"
echo "  PYTORCH_NPU_ALLOC_CONF=$PYTORCH_NPU_ALLOC_CONF"
echo "  TASK_QUEUE_ENABLE=$TASK_QUEUE_ENABLE"
echo "  CPU_AFFINITY_CONF=$CPU_AFFINITY_CONF"
echo "  TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"
echo ""
echo "To verify NPU availability, run: python -c 'import torch_npu; print(torch_npu.npu.device_count())'"
```

**Usage**:
```bash
# Source the script
source setup_npu_env.sh

# Run your training
python train.py
```

## Tuning Guidelines

### Scenario-Based Recommendations

#### Single-Device Inference

```bash
export ALGO=3                    # infer_attn for low latency
export FAST_LAYERNORM=1          # Enable optimization
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:512"
export TOKENIZERS_PARALLELISM=false
```

#### Single-Device Training

```bash
export ALGO=1                    # laser_attn for throughput
export FAST_LAYERNORM=1
export OVERLAP=0                 # No benefit without distributed
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export TASK_QUEUE_ENABLE=1
export TOKENIZERS_PARALLELISM=false
```

#### Distributed Data Parallel (DDP)

```bash
export ALGO=1
export FAST_LAYERNORM=1
export OVERLAP=0                 # DDP uses AllReduce, not AllToAll
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF="0-23:24-47:48-71:72-95"  # Adjust for your hardware
export TOKENIZERS_PARALLELISM=false
```

#### Tensor/Sequence Parallel (Megatron-style)

```bash
export ALGO=1
export FAST_LAYERNORM=1
export OVERLAP=1                 # Critical for TP/SP with AllToAll
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:1024,expandable_segments:True"
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF="0-23:24-47:48-71:72-95"
export TOKENIZERS_PARALLELISM=false
```

#### Memory-Constrained Training

```bash
export ALGO=1
export FAST_LAYERNORM=1
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True,garbage_collection_threshold:0.8"
export TASK_QUEUE_ENABLE=1
export TOKENIZERS_PARALLELISM=false
```

### Iterative Tuning Process

1. **Start with defaults** - Use the complete setup script
2. **Profile baseline** - Run 10-20 iterations, record throughput
3. **Tune ALGO** - Try 0, 1, 3; measure impact
4. **Adjust memory** - If OOM, reduce max_split_size_mb; if fragmentation, enable expandable_segments
5. **Enable OVERLAP** - Only for distributed with AllToAll patterns
6. **Set CPU affinity** - Based on your hardware topology

## Validation

### Verify Environment Variables

Check all variables are set correctly:

```bash
#!/bin/bash
# validate_env.sh

echo "=== NPU Environment Validation ==="
echo ""

# Check critical variables
check_var() {
    local var_name=$1
    local var_value=$(eval echo \$$var_name)
    if [ -z "$var_value" ]; then
        echo "[MISSING] $var_name is not set"
        return 1
    else
        echo "[OK] $var_name=$var_value"
        return 0
    fi
}

CRITICAL_MISSING=0
check_var "ALGO" || CRITICAL_MISSING=$((CRITICAL_MISSING + 1))
check_var "FAST_LAYERNORM" || CRITICAL_MISSING=$((CRITICAL_MISSING + 1))
check_var "PYTORCH_NPU_ALLOC_CONF" || CRITICAL_MISSING=$((CRITICAL_MISSING + 1))
check_var "TASK_QUEUE_ENABLE" || CRITICAL_MISSING=$((CRITICAL_MISSING + 1))
check_var "TOKENIZERS_PARALLELISM"
check_var "CPU_AFFINITY_CONF"
check_var "OVERLAP"

echo ""
if [ $CRITICAL_MISSING -eq 0 ]; then
    echo "All critical variables are set"
else
    echo "WARNING: $CRITICAL_MISSING critical variables are missing"
fi
```

### Verify NPU Availability

```python
import torch
import torch_npu

print(f"NPU available: {torch_npu.is_available()}")
print(f"NPU device count: {torch_npu.npu.device_count()}")
print(f"NPU name: {torch_npu.npu.get_device_name(0)}")

# Check current device
if torch_npu.is_available():
    x = torch.randn(2, 3).npu()
    print(f"Test tensor device: {x.device}")
    print("NPU is working correctly!")
```

### Quick Performance Test

```python
import torch
import torch_npu
import time
import os

# Print environment
print("Environment Variables:")
print(f"  ALGO={os.environ.get('ALGO', 'NOT SET')}")
print(f"  FAST_LAYERNORM={os.environ.get('FAST_LAYERNORM', 'NOT SET')}")
print(f"  OVERLAP={os.environ.get('OVERLAP', 'NOT SET')}")
print(f"  PYTORCH_NPU_ALLOC_CONF={os.environ.get('PYTORCH_NPU_ALLOC_CONF', 'NOT SET')}")
print()

# Simple benchmark
device = torch.device("npu:0")
x = torch.randn(1024, 1024, 1024, device=device)

# Warmup
for _ in range(10):
    y = torch.matmul(x, x)
torch_npu.npu.synchronize()

# Benchmark
start = time.time()
for _ in range(100):
    y = torch.matmul(x, x)
torch_npu.npu.synchronize()
elapsed = time.time() - start

print(f"Matmul benchmark: {elapsed:.3f}s for 100 iterations")
print(f"Throughput: {100/elapsed:.1f} ops/sec")
```

### Check CPU Affinity

```bash
# After starting your training script, verify affinity from another terminal
ps aux | grep python                    # Find PID
taskset -pc <PID>                       # Check CPU affinity

# Or check all python processes
for pid in $(pgrep python); do
    echo "PID $pid: $(taskset -pc $pid 2>/dev/null)"
done
```

## Common Issues

### Variables Not Taking Effect

**Problem**: Environment variables set but no performance change.

**Solution**: 
- Variables must be set **before** importing torch_npu
- Set in shell environment, not inside Python
- For distributed training, ensure all processes inherit the environment

### OOM Despite Correct Settings

**Problem**: Still getting OOM with memory settings.

**Solution**:
```bash
# More aggressive memory configuration
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True,garbage_collection_threshold:0.7"

# Also enable gradient checkpointing in code
```

### CPU Affinity Not Working

**Problem**: Affinity settings not applied.

**Solution**:
- Verify core count matches your hardware
- Check if another process is setting different affinity
- Use `taskset` manually to verify: `taskset -c 0-23 python train.py`

## References

- [torch_npu Documentation](https://gitee.com/ascend/pytorch)
- [CANN Environment Variables Guide](https://www.hiascend.com/document)
- [HCCL Tuning Guide](../hccl-test)
- Related: [Distributed Training](distributed_training.md), [Attention Mechanisms](attention_mechanism.md)
