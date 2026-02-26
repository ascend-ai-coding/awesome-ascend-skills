# vLLM-Ascend Deployment Guide

Complete deployment patterns for vLLM-Ascend inference serving on Huawei Ascend NPUs.

---

## OpenAI-Compatible Server

vLLM-Ascend provides an OpenAI-compatible API server that supports standard endpoints for model serving.

### Basic Server Command

```bash
vllm serve <model_path> \
    --device npu \
    --host 0.0.0.0 \
    --port 8000
```

### Server Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | 0.0.0.0 | Server bind address |
| `--port` | 8000 | Server port |
| `--served-model-name` | Model path | Name exposed in API responses |
| `--api-key` | None | API key for authentication |
| `--tensor-parallel-size` | 1 | Number of NPUs for tensor parallelism |
| `--pipeline-parallel-size` | 1 | Pipeline parallelism stages |
| `--max-model-len` | Model max | Maximum sequence length |
| `--max-num-seqs` | 256 | Maximum concurrent requests |
| `--gpu-memory-utilization` | 0.9 | NPU memory usage fraction |
| `--dtype` | auto | Data type (float16, bfloat16, float32) |
| `--quantization` | None | Quantization method (ascend) |

### Production Deployment Example

```bash
vllm serve /path/to/Qwen2.5-72B-Instruct \
    --device npu \
    --served-model-name "qwen2.5-72b" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --api-key "your-secret-api-key"
```

### Using Python Module

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --device npu \
    --max-model-len 4096 \
    --max-num-seqs 256
```

---

## API Endpoints

### GET /v1/models

List available models on the server.

```bash
curl http://localhost:8000/v1/models
```

Response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen2.5-7b",
      "object": "model",
      "created": 1700000000,
      "owned_by": "vllm"
    }
  ]
}
```

### POST /v1/completions

Text completion endpoint.

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer your-api-key" \
    -d '{
        "model": "qwen2.5-7b",
        "prompt": "Once upon a time",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9
    }'
```

### POST /v1/chat/completions

Chat completion endpoint.

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer your-api-key" \
    -d '{
        "model": "qwen2.5-7b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

### GET /health

Health check endpoint.

```bash
curl http://localhost:8000/health
```

Response: `200 OK` when server is ready.

---

## Python API

### LLM Class

The `LLM` class provides high-level batch inference capabilities.

```python
from vllm import LLM, SamplingParams

# Initialize LLM
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    max_model_len=4096,
    dtype="bfloat16",
    tensor_parallel_size=1
)

# Prepare prompts
prompts = [
    "Hello, how are you?",
    "Explain quantum computing.",
    "Write a Python function to sort a list."
]

# Configure sampling
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# Generate
outputs = llm.generate(prompts, sampling_params)

# Process results
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Output: {output.outputs[0].text}\n")
```

### SamplingParams

Configure text generation behavior.

```python
from vllm import SamplingParams

params = SamplingParams(
    temperature=0.7,        # Randomness (0.0 = deterministic)
    top_p=0.9,             # Nucleus sampling
    top_k=50,              # Top-k sampling
    max_tokens=512,        # Maximum output length
    stop=["\n"],           # Stop sequences
    presence_penalty=0.0,  # Penalty for token presence
    frequency_penalty=0.0  # Penalty for token frequency
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 1.0 | Sampling temperature |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `top_k` | int | -1 | Top-k sampling limit |
| `max_tokens` | int | 16 | Maximum tokens to generate |
| `stop` | List[str] | None | Stop sequences |
| `presence_penalty` | float | 0.0 | Presence penalty |
| `frequency_penalty` | float | 0.0 | Frequency penalty |

### LLM Engine (Advanced)

For fine-grained control over the inference loop.

```python
from vllm import LLMEngine, EngineArgs, SamplingParams
from vllm.inputs import TokensPrompt

# Configure engine
engine_args = EngineArgs(
    model="Qwen/Qwen2.5-7B-Instruct",
    max_model_len=4096,
    dtype="bfloat16"
)

# Create engine
engine = LLMEngine.from_engine_args(engine_args)

# Add requests
request_id = "req-001"
prompt = "Hello, world!"
params = SamplingParams(max_tokens=50)

engine.add_request(request_id, prompt, params)

# Process generation loop
while engine.has_unfinished_requests():
    outputs = engine.step()
    for output in outputs:
        if output.finished:
            print(f"{output.request_id}: {output.outputs[0].text}")
```

---

## Configuration Parameters Table

### Core Parameters

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `--device` | cuda | npu | Device type (must be npu for Ascend) |
| `--model` | required | - | Model path or HuggingFace ID |
| `--dtype` | auto | bfloat16 | Data type for weights/activations |
| `--max-model-len` | Model max | 4096-8192 | Maximum context length |
| `--max-num-seqs` | 256 | 128-512 | Max concurrent sequences |
| `--gpu-memory-utilization` | 0.9 | 0.85-0.95 | NPU memory fraction |

### Parallelism Parameters

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `--tensor-parallel-size` | 1 | 2,4,8 | Tensor parallelism degree |
| `--pipeline-parallel-size` | 1 | 2 | Pipeline parallelism stages |
| `--data-parallel-size` | 1 | 1 | Data parallelism (limited support) |

### Quantization Parameters

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `--quantization` | None | ascend | Quantization method |
| `--load-format` | auto | auto | Model loading format |

### Server Parameters

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `--host` | 0.0.0.0 | 0.0.0.0 | Server bind address |
| `--port` | 8000 | 8000 | Server port |
| `--served-model-name` | Model path | Custom name | API model name |
| `--api-key` | None | Secure key | API authentication |
| `--chat-template` | None | Template path | Custom chat template |

### Performance Parameters

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `--block-size` | 16 | 16,32,64 | KV cache block size |
| `--enable-prefix-caching` | False | True | Enable prefix caching |
| `--num-lookahead-slots` | 0 | 1-4 | Speculative decoding slots |
| `--max-num-batched-tokens` | - | Auto | Max tokens per batch |

---

## Multi-Node Deployment

### Network Requirements

- **Inter-node bandwidth**: 100 Gbps or higher recommended
- **Latency**: Sub-millisecond between nodes
- **Network type**: RDMA over RoCE or InfiniBand preferred
- **HCCL**: Compatible CANN version across all nodes

### Environment Setup

1. **Configure SSH passwordless access** between nodes
2. **Synchronize CANN versions** on all nodes
3. **Verify network connectivity**:
   ```bash
   # Test connectivity
   ping node1
   ping node2
   
   # Test HCCL (if hccl-test available)
   mpirun -n 2 -H node0:1,node1:1 ./all_reduce_test
   ```

### Docker Configuration for Multi-Node

**Node 0 (Master)**:
```bash
docker run -it --rm --network host \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
    -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
    -e HCCL_CONNECT_TIMEOUT=600 \
    ascendai/vllm-ascend:latest
```

**Node 1 (Worker)**:
```bash
docker run -it --rm --network host \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
    -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
    -e HCCL_CONNECT_TIMEOUT=600 \
    ascendai/vllm-ascend:latest
```

### HCCL Setup

1. **Create HCCL configuration file** (`hccl_config.json`):
```json
{
  "version": "1.0",
  "server_count": "2",
  "server_list": [
    {
      "server_id": "node0",
      "device": [
        {"device_id": "0", "device_ip": "192.168.1.10"},
        {"device_id": "1", "device_ip": "192.168.1.10"},
        {"device_id": "2", "device_ip": "192.168.1.10"},
        {"device_id": "3", "device_ip": "192.168.1.10"}
      ]
    },
    {
      "server_id": "node1",
      "device": [
        {"device_id": "0", "device_ip": "192.168.1.11"},
        {"device_id": "1", "device_ip": "192.168.1.11"},
        {"device_id": "2", "device_ip": "192.168.1.11"},
        {"device_id": "3", "device_ip": "192.168.1.11"}
      ]
    }
  ]
}
```

2. **Set environment variables** on all nodes:
```bash
export HCCL_IF_IP=192.168.1.10  # Node-specific IP
export HCCL_CONNECT_TIMEOUT=600
export HCCL_INTRA_ROCE_ENABLE=1
```

### Multi-Node Launch Commands

**Node 0 (Rank 0)**:
```bash
vllm serve /path/to/Qwen2.5-72B-Instruct \
    --device npu \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-init-method "tcp://192.168.1.10:29500" \
    --distributed-rank 0 \
    --max-model-len 8192 \
    --max-num-seqs 256
```

**Node 1 (Rank 1)**:
```bash
vllm serve /path/to/Qwen2.5-72B-Instruct \
    --device npu \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-init-method "tcp://192.168.1.10:29500" \
    --distributed-rank 1 \
    --max-model-len 8192 \
    --max-num-seqs 256
```

### Using MPI for Multi-Node

```bash
# Create hostfile
cat > hostfile << EOF
node0 slots=4
node1 slots=4
EOF

# Launch with MPI
mpirun -np 8 \
    --hostfile hostfile \
    --bind-to none \
    -x ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
    -x HCCL_CONNECT_TIMEOUT=600 \
    python -m vllm.entrypoints.openai.api_server \
        --model /path/to/model \
        --device npu \
        --tensor-parallel-size 8 \
        --pipeline-parallel-size 2 \
        --max-model-len 8192
```

---

## Deployment Examples by Model Size

### Small Models (7B-13B)

```bash
# Single NPU deployment
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --device npu \
    --max-model-len 4096 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.9
```

### Medium Models (32B-70B)

```bash
# Multi-NPU deployment
vllm serve Qwen/Qwen2.5-72B-Instruct \
    --device npu \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --max-num-seqs 128 \
    --dtype bfloat16
```

### Large Models (100B+)

```bash
# Multi-node deployment
# Node 0
vllm serve DeepSeek-V3 \
    --device npu \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-init-method "tcp://192.168.1.10:29500" \
    --distributed-rank 0 \
    --max-model-len 16384
```

---

## Monitoring and Logging

### Enable Debug Logging

```bash
export VLLM_LOGGING_LEVEL=DEBUG
vllm serve <model> --device npu
```

### Health Check Script

```bash
#!/bin/bash
while true; do
    if curl -sf http://localhost:8000/health > /dev/null; then
        echo "$(date): Server healthy"
    else
        echo "$(date): Server unhealthy!"
    fi
    sleep 5
done
```

### Resource Monitoring

```bash
# Monitor NPU usage
watch -n 1 npu-smi info

# Monitor memory
watch -n 1 'cat /proc/meminfo | grep MemAvailable'
```
