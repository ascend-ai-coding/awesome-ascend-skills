# Profiling Guide

## Overview

This guide covers performance profiling for vLLM-Ascend deployments. Use these tools to identify bottlenecks and optimize throughput.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_TORCH_PROFILER_DIR` | Output directory for profiling data | None |
| `VLLM_TORCH_PROFILER_WITH_STACK` | Include stack traces (0/1) | 0 |

## Quick Start

### Enable Profiling

```bash
# Set profiling output directory
export VLLM_TORCH_PROFILER_DIR=/home/data1/profiling
export VLLM_TORCH_PROFILER_WITH_STACK=0
```

### Profile vLLM Server

```bash
# Start server with profiling enabled
export VLLM_TORCH_PROFILER_DIR=/home/data1/profiling

vllm serve /path/to/model \
  --host 0.0.0.0 \
  --port 8000 \
  # ... other args
```

### Profile Offline Inference

```python
import os
from vllm import LLM, SamplingParams

os.environ["VLLM_TORCH_PROFILER_DIR"] = "./profiling_data"

llm = LLM(model="/path/to/model", ...)

# Warm-up
_ = llm.generate(["warm-up"], SamplingParams(max_tokens=10))

# Profile
llm.start_profile()
outputs = llm.generate(prompts, sampling_params)
llm.stop_profile()
```

## Analysis Tools

### torch_npu.profiler

```python
from torch_npu.profiler.profiler import analyse

# Analyze profiling data
analyse("/path/to/profiling/data")
```

### MSProbe

For detailed debugging and operator-level analysis.

```bash
# Install MSProbe
pip install msprobe

# Analyze profiling data
msprobe analyze /path/to/profiling/data
```

## Performance Metrics

### Key Metrics to Monitor

| Metric | Description | Target |
|--------|-------------|--------|
| Time to First Token (TTFT) | Latency for first token | < 500ms |
| Inter-Token Latency (ITL) | Time between tokens | < 50ms |
| Throughput | Tokens per second | Model-dependent |
| NPU Utilization | Device usage | > 80% |
| Memory Utilization | Memory usage | < 95% |

### Monitoring Commands

```bash
# Monitor NPU utilization
watch -n 1 npu-smi info

# Check memory usage
npu-smi info | grep -A 5 "Memory-Usage"
```

## Common Bottlenecks

### Memory Bottlenecks

**Symptoms:** OOM errors, low throughput

**Solutions:**
- Reduce `max_num_seqs`
- Reduce `max_model_len`
- Lower `gpu_memory_utilization`
- Enable chunked prefill

### Communication Bottlenecks

**Symptoms:** Low NPU utilization in multi-card setup

**Solutions:**
- Increase `HCCL_BUFFSIZE`
- Increase `HCCL_CONNECT_TIMEOUT`
- Check network bandwidth

### Compute Bottlenecks

**Symptoms:** High NPU utilization, low throughput

**Solutions:**
- Enable graph mode
- Check quantization
- Profile for hot operators

## Profiling Best Practices

1. **Always warm-up** before profiling to compile graphs
2. **Disable profiling in production** - adds overhead
3. **Profile representative workloads** - typical batch sizes, sequence lengths
4. **Compare before/after** changes to measure impact
5. **Document baseline metrics** for regression detection
