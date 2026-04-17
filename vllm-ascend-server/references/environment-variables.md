# Environment Variables Reference

## Overview

This document lists all environment variables used by vLLM-Ascend for performance tuning, multi-NPU communication, memory management, and feature flags.

## Core Performance Variables

| Variable | Description | Default | Recommended |
|----------|-------------|---------|-------------|
| `TASK_QUEUE_ENABLE` | Enable task queue for operator dispatch pipeline | `0` | `1` (always enable) |
| `VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE` | Enable dense model optimizations | `0` | `1` (for dense models) |
| `VLLM_ASCEND_ENABLE_PREFETCH_MLP` | Enable MLP prefetching for better performance | `0` | `1` (recommended) |

### Usage

```bash
# Core performance (almost always needed)
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
```

## Multi-NPU Communication (HCCL)

For tensor parallelism (TP > 1), configure HCCL (Huawei Collective Communication Library):

| Variable | Description | Default | Recommended |
|----------|-------------|---------|-------------|
| `HCCL_BUFFSIZE` | HCCL buffer size in MB | `120` | `1024` for large models |
| `HCCL_CONNECT_TIMEOUT` | Connection timeout in seconds | `120` | `600` or higher |
| `HCCL_EXEC_TIMEOUT` | Execution timeout in seconds | `120` | `600` or higher |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | Enable FlashComm optimization | `0` | `1` for specific models |

### Usage

```bash
# Multi-card communication
export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=600
```

## Device Selection

| Variable | Description | Example |
|----------|-------------|---------|
| `ASCEND_RT_VISIBLE_DEVICES` | Which NPU devices to use | `0` or `0,1,2,3` |
| `ASCEND_DEVICE_ID` | Single device ID | `0` |

### Usage

```bash
# Single card
export ASCEND_RT_VISIBLE_DEVICES=0

# Multi-card TP2
export ASCEND_RT_VISIBLE_DEVICES=0,1

# Multi-card TP4
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
```

## Memory Management

| Variable | Description | Default | Recommended |
|----------|-------------|---------|-------------|
| `PYTORCH_NPU_ALLOC_CONF` | PyTorch NPU memory allocator config | - | `expandable_segments:True` |
| `OMP_PROC_BIND` | OpenMP thread binding | - | `false` |
| `OMP_NUM_THREADS` | Number of OpenMP threads | - | `10` for multi-card |
| `CPU_AFFINITY_CONF` | CPU affinity configuration | - | `2` |

### Usage

```bash
# Memory management
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export CPU_AFFINITY_CONF=2
```

## vLLM V1 Engine

| Variable | Description | Default | Recommended |
|----------|-------------|---------|-------------|
| `VLLM_USE_V1` | Enable vLLM V1 engine | `0` | `1` for multi-card |

### Usage

```bash
# Enable V1 engine for better multi-card performance
export VLLM_USE_V1=1
```

## Profiling Flags

| Variable | Description | Default | Usage |
|----------|-------------|---------|-------|
| `VLLM_TORCH_PROFILER_DIR` | Output directory for profiling data | - | `/path/to/profiling` |
| `VLLM_TORCH_PROFILER_WITH_STACK` | Include stack traces | `0` | `0` or `1` |

### Usage

```bash
# Enable profiling
export VLLM_TORCH_PROFILER_DIR=/home/data1/profiling
export VLLM_TORCH_PROFILER_WITH_STACK=0
```

## Feature Flags

| Variable | Description | When to Use |
|----------|-------------|-------------|
| `VLLM_ASCEND_ENABLE_MOE_OPTIMIZE` | Enable MoE optimizations | MoE models |
| `VLLM_ASCEND_ENABLE_SPECULATIVE` | Enable speculative decoding | With Eagle |
| `VLLM_ATTENTION_BACKEND` | Attention backend selection | Advanced tuning |

### MoE-Specific

```bash
# For MoE models
export VLLM_ASCEND_ENABLE_MOE_OPTIMIZE=1
```

## Common Configurations by Scenario

### Single-Card Dense Model

```bash
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export ASCEND_RT_VISIBLE_DEVICES=0
```

### Multi-Card TP2

```bash
export VLLM_USE_V1=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export CPU_AFFINITY_CONF=2
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=600
```

### MoE Model

```bash
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_MOE_OPTIMIZE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=2048
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200
```

### Offline Batch Inference

```bash
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CPU_AFFINITY_CONF=2
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0
```

## Debugging Variables

| Variable | Description | When to Use |
|----------|-------------|-------------|
| `VLLM_LOGGING_LEVEL` | Logging verbosity | Debug: `DEBUG` |
| `VLLM_TRACE_FUNCTION` | Trace function calls | Debugging |

```bash
# Debug mode
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_TRACE_FUNCTION=1
```

## Notes

1. **Always set `TASK_QUEUE_ENABLE=1`** for best performance
2. **HCCL timeouts** may need to be increased for large models or slow networks
3. **Memory settings** are crucial for avoiding OOM errors
4. **Profiling** should be disabled in production
