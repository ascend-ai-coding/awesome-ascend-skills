# Health Check Commands

## Basic Health

```bash
# Service health
curl http://localhost:8000/health

# Expected: {"status": "ok"}
```

## Model Information

```bash
# List models
curl http://localhost:8000/v1/models

# Expected: JSON with model list
```

## Test Inference

### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3-8B",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "max_tokens": 50
    }'
```

### Text Completion

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3-8B",
        "prompt": "The quick brown fox",
        "max_tokens": 50
    }'
```

### Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3-8B",
        "messages": [{"role": "user", "content": "Count to 10"}],
        "stream": true,
        "max_tokens": 100
    }'
```

## Server Status

```bash
# NPU usage
npu-smi info

# vLLM process
ps aux | grep vllm

# GPU memory (in container)
docker exec <container> npu-smi info -t board

# Server logs
tail -f /tmp/vllm.log
# or
docker exec <container> tail -f /tmp/vllm.log
```

## Remote Server

```bash
# Health check
curl http://<server-ip>:8000/health

# Via SSH
ssh user@host "curl -s localhost:8000/health"

# NPU status
ssh user@host "npu-smi info"
```

## Python Check

```python
import requests

response = requests.get("http://localhost:8000/health")
if response.status_code == 200:
    print("✅ Server healthy")
else:
    print("❌ Server unhealthy")
```
