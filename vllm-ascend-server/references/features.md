# Feature Configuration Guide

## Overview

This document describes supported features in vLLM-Ascend and how to configure them.

## Feature Support Matrix

| Feature | Status | Notes |
|---------|--------|-------|
| Tensor Parallelism | ✅ Supported | Multi-card scaling |
| Chunked Prefill | ✅ Supported | Better memory efficiency |
| Automatic Prefix Caching | ✅ Supported | Cache repeated prompts |
| Async Scheduling | ✅ Supported | Performance optimization |
| Speculative Decoding (Eagle) | ✅ Supported | Latency reduction |
| Quantization (Ascend) | ✅ Supported | W8A8, W4A8, MXFP8 |
| Graph Mode (AclGraph) | ✅ Supported | Performance optimization |
| LoRA | 🟡 Partial | Limited model support |
| Context Parallel | 🟡 Partial | For very long contexts |
| Multi-Node | 🟡 Partial | Via Ray |

Legend: ✅ Full support | 🟡 Partial support | 🔵 Experimental

## Tensor Parallelism

Distribute model across multiple NPU cards.

### Configuration

```bash
# TP2 deployment
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=600

vllm serve /path/to/model \
  --tensor-parallel-size 2 \
  --distributed-executor-backend mp
```

### Recommendations

| Model Size | Recommended TP |
|------------|----------------|
| ≤14B | 1 |
| 14B-70B | 2-4 |
| >70B | 4-8 |

## Chunked Prefill

Split long prefill into smaller chunks for better memory efficiency.

### Configuration

```bash
--max-num-batched-tokens 4096
--async-scheduling
```

### Notes

- Always use with `--async-scheduling`
- Adjust `--max-num-batched-tokens` based on memory

## Automatic Prefix Caching

Cache KV cache for repeated prompt prefixes.

### When to Enable

- Chatbots with system prompts
- RAG applications with similar queries
- Template-based generation

### Configuration

```bash
# Enable prefix caching
--enable-prefix-caching

# Disable prefix caching (default for best throughput)
--no-enable-prefix-caching
```

### Notes

- Increases memory usage
- Disable if prompts are mostly unique

## Speculative Decoding (Eagle)

Use Eagle draft model for latency reduction.

### How It Works

1. Draft model generates N tokens quickly
2. Target model verifies tokens in parallel
3. Accept correct tokens, reject and regenerate incorrect ones

### Configuration

```bash
# Environment
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export ASCEND_RT_VISIBLE_DEVICES=0

# Launch with Eagle
vllm serve /path/to/Qwen3-8B \
  --quantization ascend \
  --speculative-config '{
    "method": "eagle3",
    "model": "/path/to/Eagle3-Qwen3-8B/",
    "draft_model_parallel_size": 1,
    "num_speculative_tokens": 2
  }'
```

### Supported Models

| Base Model | Eagle Model |
|------------|-------------|
| Qwen3-8B | Eagle3-Qwen3-8B |
| Qwen2.5-7B | Eagle-Qwen2.5-7B |

### Parameters

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `num_speculative_tokens` | 2-4 | More tokens = higher risk |
| `draft_model_parallel_size` | 1 | Usually 1 |

## Quantization

Reduce model memory and improve throughput.

### Ascend Quantization Types

| Type | Description | Use Case |
|------|-------------|----------|
| W8A8 | 8-bit weights, 8-bit activations | General purpose |
| W4A8 | 4-bit weights, 8-bit activations | Memory constrained |
| MXFP8 | MX floating point 8-bit | High accuracy |

### Configuration

```bash
--quantization ascend
```

### Notes

- Model must be quantized with Ascend tools
- Check model README for quantization type

## Graph Mode (AclGraph)

Compile model to graph for better performance.

### Configuration

```bash
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
--no-enforce-eager
```

### Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| `FULL_DECODE_ONLY` | Full graph for decode phase | Production |
| `PARTIAL` | Partial graph | Some ops unsupported |
| `NONE` | No graph | Debugging |

### Troubleshooting

If graph mode fails:
1. Try `PARTIAL` mode
2. Fall back to eager mode: `--enforce-eager`
3. Check for unsupported operators

## LoRA (Low-Rank Adaptation)

Fine-tune models with minimal parameters.

### Status

Limited support. Check model compatibility.

### Configuration

```bash
vllm serve /path/to/base-model \
  --enable-lora \
  --lora-modules lora1=/path/to/lora-adapter
```

### Notes

- Not all models support LoRA
- May have performance impact

## Context Parallel

For very long context (>64K tokens).

### When to Use

- Context length > 64K
- Memory constraints with long context

### Configuration

Set via environment variables and additional config.

### Status

Experimental. Check documentation for latest support.

## Feature Combinations

### Production High-Throughput

```bash
--async-scheduling \
--no-enable-prefix-caching \
--quantization ascend \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"enable_cpu_binding":true}'
```

### Low Latency (with Eagle)

```bash
--async-scheduling \
--quantization ascend \
--speculative-config '{"method": "eagle3", "model": "/path/to/eagle", "num_speculative_tokens": 2}'
```

### Memory Efficient

```bash
--async-scheduling \
--enable-prefix-caching \
--gpu-memory-utilization 0.85 \
--max-num-seqs 128
```

### Debugging Mode

```bash
--enforce-eager \
--enable-prefix-caching
# Plus profiling env vars
```
