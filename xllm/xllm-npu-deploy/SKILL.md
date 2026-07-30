---
name: xllm-npu-deploy
description: NPU deployment assistant for xllm framework - automatically detects model type and provides deployment guidance. Use this skill whenever the user wants to deploy LLM, VLM, or DiT models on NPU with xllm framework, needs help with xllm model deployment, or asks about NPU inference setup.
version: 1.0.0
triggers:
  - "部署模型"
  - "xllm部署"
  - "npu部署"
  - "如何部署"
  - "model deploy"
  - "xllm deploy"
  - "deploy *.pt"
  - "deploy *.safetensors"
  - "在线服务"
  - "api server"
  - "openai compatible"
---

# XLLM NPU Deployment Assistant

Automatically detect model type and provide deployment guidance for xllm framework on NPU.

## Prerequisites

**Important: Before using this skill, ensure:**

1. **You are in the xllm project directory** (the directory containing `build/xllm/core/server/xllm`)
2. **xllm has been compiled** (`build` directory exists and contains executable files)
3. **NPU drivers and environment variables are properly configured**

### Environment Check

```bash
# Verify you are in the xllm directory
pwd  # Should show /path/to/xllm

# Verify xllm is compiled
ls build/xllm/core/server/xllm  # Should exist

# Verify NPU is available
npu-smi info
```

## Directory Structure

```
xllm-npu-deploy/
├── SKILL.md                  # Main skill file
├── scripts/                  # Executable scripts
│   ├── auto_deploy.py        # Auto-deployment (detect + deploy)
│   ├── detect_model.py       # Model type detection
│   ├── deploy_llm.py         # LLM deployment
│   ├── deploy_vlm.py         # VLM deployment
│   ├── deploy_dit.py         # DiT deployment
│   ├── deploy_server.py      # Online service deployment
│   └── model_detector.py     # Model detection library
├── assets/                   # Example files
│   └── qwen2-vl-example.sh
└── references/               # Reference documentation (loaded on demand)
    └── BUILD.md              # Build guide for A2/A3 NPU (see below)
```

### Reference Documents

- **references/BUILD.md** - Detailed build instructions for A2/A3 architecture with logging

## Quick Start

```bash
# Auto detect and deploy
python ~/.claude/skills/xllm-npu-deploy/scripts/auto_deploy.py \
    --model /path/to/model \
    --devices npu:0

# Or detect first, then deploy
python ~/.claude/skills/xllm-npu-deploy/scripts/detect_model.py /path/to/model --recommend

# Start online service
python ~/.claude/skills/xllm-npu-deploy/scripts/deploy_server.py \
    --model /path/to/model \
    --port 8080
```

## Model Types

| Type | Description | Backend | Script |
|------|-------------|---------|--------|
| **LLM** | Language Model (text generation) | `llm` | `deploy_llm.py` |
| **VLM** | Vision-Language Model (multimodal understanding) | `vlm` | `deploy_vlm.py` |
| **DiT** | Diffusion Transformer (multimodal generation) | `dit` | `deploy_dit.py` |

## Online Service (API Server)

xllm supports OpenAI-compatible HTTP API server.

### Foreground Mode (for debugging)

```bash
# Start server (foreground)
python ~/.claude/skills/xllm-npu-deploy/scripts/deploy_server.py \
    --model /path/to/model \
    --port 8080 \
    --devices npu:0 \
    --xllm_binary ./build/xllm/core/server/xllm
```

### Daemon Mode (for production, recommended)

```bash
# Daemon mode with log file
python ~/.claude/skills/xllm-npu-deploy/scripts/deploy_server.py \
    --model /path/to/model \
    --port 8080 \
    --devices npu:0 \
    --xllm_binary ./build/xllm/core/server/xllm \
    --daemon \
    --log_file ./logs/xllm_8080.log

# Output:
# Server started as daemon (PID: 12345)
# Logs will be written to ./logs/xllm_8080.log
# To stop: kill 12345
# To view logs: tail -f ./logs/xllm_8080.log
```

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completion (OpenAI compatible) |
| `POST /v1/completions` | Text completion |
| `POST /v1/embeddings` | Text embedding |
| `GET /v1/models` | List available models |

### Example Client Request

```bash
# Chat completion
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
print(response.choices[0].message.content)
```

## Model-Specific Configurations

### VLM (e.g., Qwen2.5-VL)

Required flags:
- `--disable_prefix_cache`
- `--disable_chunked_prefill`
- `--enable_shm` (recommended)

```bash
python scripts/deploy_vlm.py \
    --model /path/to/Qwen2.5-VL-7B-Instruct \
    --devices npu:0 \
    --enable_shm \
    --max_seqs_per_batch 4
```

### MoE LLM (e.g., DeepSeek-V3)

```bash
python scripts/deploy_llm.py \
    --model /path/to/deepseek-v3 \
    --devices npu:0,npu:1,npu:2,npu:3 \
    --enable_mla \
    --dp_size 4 \
    --disable_chunked_prefill
```

## Common Issues

| Issue | Solution |
|-------|----------|
| OOM | Reduce `--max_memory_utilization` or `--max_tokens_per_batch` |
| VLM slow | Add `--enable_shm` flag |
| Prefix cache error | Add `--disable_prefix_cache` |
| Chunked prefill error | Add `--disable_chunked_prefill` |
| HCCL error | Specify `--rank_tablefile` |
| xllm not found | Use `--xllm_binary ./build/xllm/core/server/xllm` |
| Not in xllm directory | `cd /path/to/xllm` before running |
| xllm not compiled | Read **references/BUILD.md** for detailed A2/A3 build instructions |
| Build fails / compilation errors | Check **references/BUILD.md** - Troubleshooting section |

### Build Issues

When encountering compilation or build-related issues, **always refer to references/BUILD.md** for:

- Correct build commands for A2/A3 architecture
- Build logging best practices
- Troubleshooting common build errors
- First-time build notes

**Quick build reference:**
```bash
# A2 build
python3 setup.py build 2>&1 | tee logs/build_a2.log

# A3 build
python3 setup.py build --device a3 2>&1 | tee logs/build_a3.log
```

For detailed instructions, see: `~/.claude/skills/xllm-npu-deploy/references/BUILD.md`

### Server Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--daemon` / `-d` | Run in daemon mode | `--daemon` |
| `--log_file` | Log file path | `--log_file ./server.log` |
| `--xllm_binary` | Path to xllm executable | `--xllm_binary ./build/xllm/core/server/xllm` |

## Programmatic Detection

```python
import sys
sys.path.insert(0, '/root/.claude/skills/xllm-npu-deploy/scripts')

from model_detector import detect_model_type, get_deployment_recommendations

result = detect_model_type("/path/to/model")
print(result["type"])        # "llm", "vlm", "dit", or "unknown"
print(result["confidence"])  # "high", "medium", or "low"

recs = get_deployment_recommendations(result["type"])
print(recs["script"])        # Which deploy script to use
```
