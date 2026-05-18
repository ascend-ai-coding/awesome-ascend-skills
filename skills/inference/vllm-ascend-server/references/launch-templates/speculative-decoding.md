# Speculative Decoding (Eagle)

## Overview

Eagle speculative decoding reduces latency by using a draft model to speculate tokens, then verifying with the target model.

## Quick Start

```bash
#!/bin/bash

export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export ASCEND_RT_VISIBLE_DEVICES=0

vllm serve /home/data1/Qwen3-8B-mxfp8 \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name Qwen3-8B-Eagle \
    --trust-remote-code \
    --quantization ascend \
    --speculative-config '{
        "method": "eagle3",
        "model": "/home/data1/Eagle3-Qwen3-8B/",
        "draft_model_parallel_size": 1,
        "num_speculative_tokens": 2
    }'
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `method` | string | - | "eagle" or "eagle3" |
| `model` | string | - | Path to Eagle draft model |
| `draft_model_parallel_size` | int | 1 | TP size for draft model |
| `num_speculative_tokens` | int | 2 | Number of speculative tokens |

## Supported Model Pairs

| Base Model | Eagle Model |
|------------|-------------|
| Qwen3-8B | Eagle3-Qwen3-8B |
| Qwen2.5-7B | Eagle-Qwen2.5-7B |

## Full Configuration

```bash
#!/bin/bash

export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export ASCEND_RT_VISIBLE_DEVICES=0

MODEL_PATH="/home/data1/Qwen3-8B-mxfp8"
EAGLE_PATH="/home/data1/Eagle3-Qwen3-8B/"
PORT=8000

vllm serve ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --served-model-name Qwen3-8B-Eagle \
    --trust-remote-code \
    --max-num-seqs 256 \
    --max-model-len 32768 \
    --quantization ascend \
    --gpu-memory-utilization 0.9 \
    --async-scheduling \
    --speculative-config "{
        \"method\": \"eagle3\",
        \"model\": \"${EAGLE_PATH}\",
        \"draft_model_parallel_size\": 1,
        \"num_speculative_tokens\": 2
    }"
```

## Performance Tuning

### num_speculative_tokens

| Value | Acceptance Rate | Latency | Recommendation |
|-------|-----------------|---------|----------------|
| 1 | Highest | Good | Conservative |
| 2 | High | Better | Recommended |
| 4 | Medium | Best if accepted | Aggressive |

### When to Use

- ✅ Latency-sensitive applications
- ✅ Streaming responses
- ❌ Batch processing (throughput-focused)
