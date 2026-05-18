# Launch Templates

## Overview

Template scripts for vLLM-Ascend deployment. Customize based on your requirements.

## Quick Reference

| Template | File | Use Case |
|----------|------|----------|
| Single-Card | [online-serving.md](launch-templates/online-serving.md) | Dense ≤14B, TP=1 |
| Multi-Card | [online-serving.md](launch-templates/online-serving.md) | Dense 14B-70B, TP=2-4 |
| Eagle Speculative | [speculative-decoding.md](launch-templates/speculative-decoding.md) | Latency-sensitive |
| Offline Inference | [offline-inference.md](launch-templates/offline-inference.md) | Batch processing |
| Docker Deployment | [docker.md](launch-templates/docker.md) | Containerized |

## Common Environment Variables

### Single Card

```bash
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export ASCEND_RT_VISIBLE_DEVICES=0
```

### Multi-Card (TP > 1)

```bash
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

## Common vLLM Arguments

### Required

```bash
--host 0.0.0.0
--port 8000
--trust-remote-code
--tensor-parallel-size <tp>
```

### Performance

```bash
--max-num-seqs 256
--max-model-len 32768
--max-num-batched-tokens 4096
--gpu-memory-utilization 0.9
--async-scheduling
--additional-config '{"enable_cpu_binding":true}'
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

### Quantization (if applicable)

```bash
--quantization ascend
```

## Detailed Templates

See subdirectories for complete scripts:

- **[online-serving.md](launch-templates/online-serving.md)** - Online serving scripts
- **[speculative-decoding.md](launch-templates/speculative-decoding.md)** - Eagle speculative decoding
- **[offline-inference.md](launch-templates/offline-inference.md)** - Offline batch inference
- **[docker.md](launch-templates/docker.md)** - Docker deployment
- **[health-check.md](launch-templates/health-check.md)** - Verification commands
