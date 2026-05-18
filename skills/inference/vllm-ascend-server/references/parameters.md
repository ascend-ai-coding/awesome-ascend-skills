# vLLM Parameters Reference

## Overview

This document lists vLLM command-line parameters and Python API arguments relevant to Ascend NPU deployment.

## Required Parameters

| Parameter | CLI | Python | Description |
|-----------|-----|--------|-------------|
| Model path | `--model /path/to/model` | `model="/path/to/model"` | Path to model weights |
| Host | `--host 0.0.0.0` | N/A | Server bind address |
| Port | `--port 8000` | N/A | Server port |

## Performance Parameters

### Memory and Batching

| Parameter | CLI | Python | Default | Description |
|-----------|-----|--------|---------|-------------|
| `max_model_len` | `--max-model-len 32768` | `max_model_len=32768` | Model default | Maximum sequence length |
| `max_num_seqs` | `--max-num-seqs 256` | `max_num_seqs=256` | 256 | Maximum concurrent sequences |
| `max_num_batched_tokens` | `--max-num-batched-tokens 4096` | `max_num_batched_tokens=4096` | Auto | Maximum batched tokens |
| `block_size` | `--block-size 128` | `block_size=128` | 16 | KV cache block size |
| `gpu_memory_utilization` | `--gpu-memory-utilization 0.9` | `gpu_memory_utilization=0.9` | 0.9 | GPU/NPU memory fraction |

### Recommendations

| Scenario | max_model_len | max_num_seqs | max_num_batched_tokens |
|----------|--------------|--------------|------------------------|
| Standard | 32768 | 256 | 4096 |
| Long context | 65536 | 128 | 4096 |
| High throughput | 16384 | 512 | 8192 |
| Memory limited | 16384 | 128 | 2048 |

## Parallelism Parameters

### Tensor Parallelism

| Parameter | CLI | Python | Description |
|-----------|-----|--------|-------------|
| `tensor_parallel_size` | `--tensor-parallel-size 2` | `tensor_parallel_size=2` | Number of NPU cards |
| `distributed_executor_backend` | `--distributed-executor-backend mp` | N/A | Backend: `mp` or `ray` |

### Backend Selection

| Backend | When to Use |
|---------|-------------|
| `mp` (multiprocessing) | TP ≤ 4, single node |
| `ray` | TP > 4, multi-node, advanced features |

## Quantization Parameters

| Parameter | CLI | Python | Description |
|-----------|-----|--------|-------------|
| `quantization` | `--quantization ascend` | `quantization="ascend"` | Ascend quantization |

### Ascend Quantization Types

- **W8A8**: 8-bit weights, 8-bit activations
- **W4A8**: 4-bit weights, 8-bit activations
- **MXFP8**: MX floating point 8-bit

```bash
# For quantized models
--quantization ascend
```

## Scheduling Parameters

| Parameter | CLI | Python | Description |
|-----------|-----|--------|-------------|
| `async_scheduling` | `--async-scheduling` | `async_scheduling=True` | Enable async scheduling |
| `enable_prefix_caching` | `--enable-prefix-caching` | `enable_prefix_caching=True` | Cache prefixes |
| `no_enable_prefix_caching` | `--no-enable-prefix-caching` | `enable_prefix_caching=False` | Disable prefix caching |
| `enforce_eager` | `--enforce-eager` | `enforce_eager=True` | Disable graph mode |
| `no_enforce_eager` | `--no-enforce-eager` | `enforce_eager=False` | Enable graph mode |

### Recommendations

| Scenario | async_scheduling | prefix_caching | enforce_eager |
|----------|-----------------|----------------|---------------|
| Standard | true | false | false |
| Repeated prompts | true | true | false |
| Debugging | true | false | true |

## Additional Configuration

### --additional-config

JSON configuration for advanced settings:

```bash
--additional-config '{"enable_cpu_binding":true}'
```

### Supported Options

| Option | Type | Description |
|--------|------|-------------|
| `enable_cpu_binding` | bool | Enable CPU core binding |
| `tensor_parallel_size` | int | Override TP size |
| `enable_chunked_prefill` | bool | Enable chunked prefill |

### Python API

```python
additional_config={
    "enable_cpu_binding": True,
    "enable_chunked_prefill": True
}
```

## Compilation Configuration

### --compilation-config

JSON configuration for graph compilation:

```bash
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

### cudagraph_mode Options

| Mode | Description | When to Use |
|------|-------------|-------------|
| `FULL_DECODE_ONLY` | Full graph for decode only | Production (recommended) |
| `PARTIAL` | Partial graph mode | Some ops unsupported |
| `NONE` | No graph mode | Debugging |

### Python API

```python
compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"}
```

## Speculative Configuration

### --speculative-config

For Eagle speculative decoding:

```bash
--speculative-config '{"method": "eagle3", "model": "/path/to/eagle-model/", "draft_model_parallel_size": 1, "num_speculative_tokens": 2}'
```

### Options

| Option | Type | Description |
|--------|------|-------------|
| `method` | string | `"eagle"` or `"eagle3"` |
| `model` | string | Path to Eagle draft model |
| `draft_model_parallel_size` | int | TP size for draft model |
| `num_speculative_tokens` | int | Number of speculative tokens (2-4) |

## Multimodal Parameters

| Parameter | CLI | Description |
|-----------|-----|-------------|
| `limit-mm-per-prompt` | `--limit-mm-per-prompt "image=5,video=1"` | Limit multimedia inputs per prompt |

```bash
# For vision-language models
--limit-mm-per-prompt "image=5,video=1"
```

## Server Parameters

| Parameter | CLI | Default | Description |
|-----------|-----|---------|-------------|
| `served_model_name` | `--served-model-name Qwen3-8B` | Model path | Model name in API |
| `trust_remote_code` | `--trust-remote-code` | false | Trust remote code |
| `chat_template` | `--chat-template path` | Auto | Custom chat template |

## Complete Example

### CLI (Online Serving)

```bash
vllm serve /path/to/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name Qwen3-8B \
  --trust-remote-code \
  --max-num-seqs 256 \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --tensor-parallel-size 1 \
  --quantization ascend \
  --gpu-memory-utilization 0.9 \
  --block-size 128 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --additional-config '{"enable_cpu_binding":true}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

### Python (Offline Inference)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/Qwen3-8B",
    max_num_seqs=256,
    max_model_len=32768,
    max_num_batched_tokens=4096,
    tensor_parallel_size=1,
    enable_prefix_caching=False,
    async_scheduling=True,
    quantization="ascend",
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
    additional_config={"enable_cpu_binding": True},
    compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = llm.generate(["Hello, world!"], sampling_params)
```
