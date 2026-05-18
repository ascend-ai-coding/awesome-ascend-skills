# Graph Mode vs Eager Mode Guide

## Overview

vLLM-Ascend supports two execution modes: **Graph Mode** (AclGraph) and **Eager Mode**. Choosing the right mode is critical for performance and debugging.

## Quick Selection

| Scenario | Recommended Mode | Reason |
|----------|------------------|--------|
| Production deployment | Graph Mode | Best performance |
| Development/Testing | Eager Mode | Easier debugging |
| Encountering graph errors | Eager Mode | Compatibility |
| First-time deployment | Eager Mode → Graph Mode | Test first, optimize later |

## Graph Mode (AclGraph)

### What It Does

Graph mode compiles the model computation graph for optimized execution on NPU, similar to CUDA graphs on GPU.

### Benefits

- **Higher throughput**: Reduced kernel launch overhead
- **Lower latency**: Optimized execution path
- **Better memory efficiency**: Pre-allocated buffers

### Configuration

```bash
# Recommended for production
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
--no-enforce-eager
```

### Graph Mode Options

| Mode | Description | Use Case |
|------|-------------|----------|
| `FULL_DECODE_ONLY` | Full graph for decode phase | Production (recommended) |
| `PARTIAL` | Partial graph compilation | Some ops unsupported |
| `NONE` | No graph compilation | Debugging |

### When to Use Graph Mode

- ✅ Production deployments
- ✅ Model is stable and tested
- ✅ Maximum performance needed
- ✅ Throughput-oriented workloads

### Common Graph Mode Issues

| Error | Cause | Solution |
|-------|-------|----------|
| Graph capture failed | Unsupported operator | Use eager mode or PARTIAL |
| Shape mismatch | Dynamic shapes | Reduce max_num_seqs or use eager |
| Memory error | Graph memory overhead | Lower gpu_memory_utilization |

## Eager Mode

### What It Does

Eager mode executes operations one by one without graph compilation, similar to PyTorch eager execution.

### Benefits

- **Easier debugging**: Full stack traces, clear errors
- **Better compatibility**: Works with all operators
- **Flexible**: Dynamic shapes supported

### Configuration

```bash
# Enable eager mode
--enforce-eager
```

Or remove `--compilation-config` entirely.

### When to Use Eager Mode

- ✅ Development and testing
- ✅ Debugging errors
- ✅ First-time model deployment
- ✅ Models with unsupported operators
- ✅ Dynamic input shapes

## Transition Workflow

### Recommended Progression

```
1. First Deployment → Eager Mode (test functionality)
        ↓
2. Verify works → Try Graph Mode (test performance)
        ↓
3. If graph fails → Debug or use PARTIAL mode
        ↓
4. Production → Graph Mode (optimized)
```

### Step-by-Step Transition

**Step 1: Test with Eager Mode**

```bash
vllm serve /model \
  --enforce-eager \
  ...other params...
```

Verify:
- Service starts correctly
- Inference works
- No errors in logs

**Step 2: Try Graph Mode**

```bash
vllm serve /model \
  --no-enforce-eager \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  ...other params...
```

If successful, use for production.

**Step 3: Handle Graph Errors**

If graph mode fails:

```bash
# Try PARTIAL mode
--compilation-config '{"cudagraph_mode": "PARTIAL"}'

# Or fall back to eager
--enforce-eager
```

## Performance Comparison

| Mode | Throughput | Latency | Debugging | Compatibility |
|------|------------|---------|-----------|---------------|
| Graph | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Eager | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

Typical performance difference: Graph mode can be 20-50% faster.

## Decision Tree

```
Starting deployment?
├── First time with this model?
│   └── Yes → Use Eager Mode
│       └── Works? → Try Graph Mode
│           ├── Works → Use Graph Mode (production)
│           └── Fails → Debug or use PARTIAL
│
└── Known working model?
    └── Production? → Graph Mode
    └── Development? → Eager Mode
```

## Configuration Examples

### Production Configuration (Graph Mode)

```bash
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export ASCEND_RT_VISIBLE_DEVICES=0

vllm serve /model \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --no-enforce-eager \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --quantization ascend \
  --gpu-memory-utilization 0.9 \
  --async-scheduling \
  --additional-config '{"enable_cpu_binding":true}'
```

### Development Configuration (Eager Mode)

```bash
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0

vllm serve /model \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --enforce-eager \
  --gpu-memory-utilization 0.9
```

### Debugging Configuration

```bash
export TASK_QUEUE_ENABLE=1
export VLLM_LOGGING_LEVEL=DEBUG
export ASCEND_RT_VISIBLE_DEVICES=0

vllm serve /model \
  --enforce-eager \
  --gpu-memory-utilization 0.8 \
  ...
```

## Troubleshooting

### Graph Mode Fails to Start

1. **Check error message** for unsupported operator
2. **Try PARTIAL mode** instead of FULL_DECODE_ONLY
3. **Fall back to eager mode** for debugging
4. **Report issue** if operator should be supported

### Graph Mode Causes OOM

1. **Lower gpu_memory_utilization** (e.g., 0.85)
2. **Reduce max_num_seqs** (e.g., 128)
3. **Graph memory overhead**: Can be 10-20% extra

### Inconsistent Results Between Modes

1. **This is a bug** - results should be identical
2. **Report issue** with reproduction steps
3. **Use eager mode** until fixed

## Related Configuration

Graph mode works with:

- ✅ Quantization (`--quantization ascend`)
- ✅ Tensor parallelism
- ✅ Async scheduling
- ✅ CPU binding

Graph mode may have issues with:

- ⚠️ Dynamic shapes (varying sequence lengths)
- ⚠️ Some custom operators
- ⚠️ Speculative decoding (Eagle)
