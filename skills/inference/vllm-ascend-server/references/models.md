# Model Support Matrix

## Overview

This document describes how to detect and categorize models for vLLM-Ascend deployment, and maps each model to its best-practice configuration file.

## Model Categories

| Category | Description | Examples | Config Pattern |
|----------|-------------|----------|----------------|
| **Dense** | Standard transformer models | Qwen3-8B, Qwen3-30B, GLM-4 | TP based on size |
| **MoE** | Mixture of Experts | DeepSeek-V3, Qwen3-235B-A22B | Expert parallel |
| **VL** | Vision-Language multimodal | Qwen2.5-VL, Qwen3-VL | Multimodal config |
| **Embedding** | Text embedding models | Qwen3-Embedding | Embedding mode |
| **Reranker** | Reranking models | Qwen3-Reranker | Reranker mode |

## Model Detection

### Detection via config.json

Read the model's `config.json` file and check these fields:

```json
{
  "architectures": ["Qwen2ForCausalLM"],
  "model_type": "qwen2",
  "hidden_size": 8192,
  "num_hidden_layers": 32,
  "num_attention_heads": 64,
  "num_key_value_heads": 8,
  "intermediate_size": 22016,
  "num_experts": 64,           // Present for MoE
  "vision_config": {...}       // Present for VL
}
```

### Detection Logic

```python
def detect_model_category(config):
    """Detect model category from config.json"""

    # Check for MoE
    if "num_experts" in config or "router_hidden_size" in config:
        return "moe"

    # Check for Vision-Language
    if "vision_config" in config and config["vision_config"] is not None:
        return "vl"

    # Check for Embedding
    model_type = config.get("model_type", "").lower()
    if "embedding" in model_type or "embed" in model_type:
        return "embedding"

    # Check for Reranker
    if "reranker" in model_type:
        return "reranker"

    # Default to dense
    return "dense"
```

### Architecture to Model Mapping

| Architecture | Model Family | Config File |
|--------------|--------------|-------------|
| `Qwen2ForCausalLM` | Qwen2/Qwen2.5 | `qwen2.5-*.yaml` |
| `Qwen3ForCausalLM` | Qwen3 | `qwen3-*.yaml` |
| `Qwen2VLForConditionalGeneration` | Qwen2-VL | `qwen2-vl.yaml` |
| `Qwen2_5_VLForConditionalGeneration` | Qwen2.5-VL | `qwen2.5-vl.yaml` |
| `DeepseekV3ForCausalLM` | DeepSeek-V3 | `deepseek-v3.yaml` |
| `DeepseekV2ForCausalLM` | DeepSeek-V2 | `deepseek-v2.yaml` |
| `Glm4ForCausalLM` | GLM-4 | `glm-4.x.yaml` |
| `ChatGLMModel` | GLM | `glm-4.x.yaml` |

### Model Size Detection

Estimate model size from `hidden_size`:

| hidden_size | Approximate Parameters | TP Recommendation |
|-------------|------------------------|-------------------|
| 4096 | ~7B | TP=1 |
| 5120 | ~13B | TP=1 |
| 6144 | ~20B | TP=1-2 |
| 8192 | ~34B | TP=2 |
| 10240 | ~70B | TP=2-4 |
| 12288 | ~100B+ | TP=4-8 |

## Supported Models

### Dense Models

| Model | Config File | Recommended TP | Notes |
|-------|-------------|----------------|-------|
| Qwen3-8B | `qwen3-8b.yaml` | 1 | Single card |
| Qwen3-14B | `qwen3-14b.yaml` | 1 | Single card |
| Qwen3-30B | `qwen3-30b.yaml` | 2 | TP2 |
| Qwen2.5-7B | `qwen2.5-7b.yaml` | 1 | Single card |
| Qwen2.5-14B | `qwen2.5-14b.yaml` | 1-2 | Depending on memory |
| Qwen2.5-32B | `qwen2.5-32b.yaml` | 2 | TP2 |
| Qwen2.5-72B | `qwen2.5-72b.yaml` | 4 | TP4 |
| GLM-4-9B | `glm-4.x.yaml` | 1 | Single card |

### MoE Models

| Model | Config File | Recommended TP | Notes |
|-------|-------------|----------------|-------|
| Qwen3-235B-A22B | `qwen3-235b-a22b.yaml` | 4-8 | Expert parallel |
| DeepSeek-V3 | `deepseek-v3.yaml` | 8+ | Large scale, CP |
| DeepSeek-V2 | `deepseek-v2.yaml` | 4-8 | Expert parallel |

### Multimodal Models (VL)

| Model | Config File | Recommended TP | Notes |
|-------|-------------|----------------|-------|
| Qwen2.5-VL-7B | `qwen2.5-vl.yaml` | 1 | Multimodal |
| Qwen3-VL | `qwen3-vl.yaml` | 1-2 | Multimodal |

### Embedding/Reranker Models

| Model | Config File | Recommended TP | Notes |
|-------|-------------|----------------|-------|
| Qwen3-Embedding | `qwen3-embedding.yaml` | 1 | Embedding mode |
| Qwen3-Reranker | `qwen3-reranker.yaml` | 1 | Reranker mode |

## Model Config File Structure

Each model config file (`references/model_configs/<model>.yaml`) contains:

```yaml
model:
  name: "Model Name"
  category: dense | moe | vl | embedding | reranker
  description: "Brief description"
  hf_path: "org/model-name"

deployment:
  recommended_tp: 1
  min_npu_count: 1
  recommended_memory_gb: 32

performance:
  max_model_len: 32768
  max_num_seqs: 256
  max_num_batched_tokens: 4096
  block_size: 128
  gpu_memory_utilization: 0.9

env_vars:
  TASK_QUEUE_ENABLE: 1
  # ... other env vars

vllm_args:
  trust_remote_code: true
  quantization: ascend
  # ... other args

additional_config:
  enable_cpu_binding: true

compilation_config:
  cudagraph_mode: "FULL_DECODE_ONLY"

features:
  chunked_prefill: true
  automatic_prefix_cache: true
  speculative_decoding: false
  lora: false
```

## Detection Example

Given a model path `/home/data1/Qwen3-8B/`:

1. Read `/home/data1/Qwen3-8B/config.json`
2. Check `architectures`: `["Qwen3ForCausalLM"]` → Qwen3 family
3. Check `hidden_size`: 4096 → ~8B parameters
4. No `num_experts` → Dense model
5. No `vision_config` → Not multimodal
6. Result: Load `references/model_configs/qwen3-8b.yaml`
