# Model Support Matrix

Complete list of models supported by msModelSlim.

---

## Large Language Models (LLM)

### Qwen Series

| Model | One-Click | V0 Script | W8A8 | W4A8 | W8A8C8 | W8A8S | Dependencies |
|-------|-----------|-----------|------|------|--------|-------|--------------|
| **Qwen3-8B** | ✓ | `example/Qwen/` | - | - | - | ✓ | transformers==4.51.0 |
| **Qwen3-14B** | ✓ | `example/Qwen/` | ✓ | - | - | ✓ | transformers==4.51.0 |
| **Qwen3-32B** | ✓ | `example/Qwen/` | ✓ | ✓ | ✓ | ✓ | transformers==4.51.0 |
| **Qwen2.5-7B-Instruct** | ✓ | `example/Qwen/` | ✓ | - | - | ✓ | - |
| **Qwen2.5-32B-Instruct** | ✓ | `example/Qwen/` | ✓ | - | - | - | - |
| **Qwen2.5-72B-Instruct** | ✓ | `example/Qwen/` | - | ✓ | ✓ | - | - |
| **Qwen2.5-Coder-7B** | ✓ | `example/Qwen/` | - | - | - | ✓ | - |
| **Qwen2-7B** | - | `example/Qwen/` | ✓ | - | - | ✓ | - |
| **Qwen2-72B** | - | `example/Qwen/` | ✓ | - | ✓ | ✓ | - |
| **QwQ-32B** | ✓ | `example/Qwen/` | ✓ | - | - | ✓ | - |

### DeepSeek Series

| Model | One-Click | V0 Script | W8A8 | W4A8 | W8A8C8 | Dependencies |
|-------|-----------|-----------|------|------|--------|--------------|
| **DeepSeek-V3** | - | `example/DeepSeek/` | ✓ | - | - | transformers==4.48.2 |
| **DeepSeek-V3.1** | - | `example/DeepSeek/` | ✓ | ✓ | ✓ | transformers==4.48.2 |
| **DeepSeek-V3.2** | ✓ | `example/DeepSeek/` | ✓ | - | - | transformers==4.48.2 |
| **DeepSeek-V3.2-Exp** | ✓ | `example/DeepSeek/` | ✓ | ✓ | - | transformers==4.48.2 |
| **DeepSeek-R1** | - | `example/DeepSeek/` | ✓ | ✓ | ✓ | transformers==4.48.2 |
| **DeepSeek-R1-0528** | - | `example/DeepSeek/` | ✓ | ✓ | ✓ | transformers==4.48.2 |
| **DeepSeek-Coder-33B** | - | `example/DeepSeek/` | ✓ | - | ✓ | - |

### DeepSeek-R1-Distill Series

| Model | One-Click | V0 Script | W8A8 | W8A8S |
|-------|-----------|-----------|------|-------|
| **DeepSeek-R1-Distill-Llama-8B** | - | `example/DeepSeek/` | ✓ | ✓ |
| **DeepSeek-R1-Distill-Llama-70B** | - | `example/DeepSeek/` | ✓ | - |
| **DeepSeek-R1-Distill-Qwen-1.5B** | - | `example/DeepSeek/` | ✓ | ✓ |
| **DeepSeek-R1-Distill-Qwen-7B** | - | `example/DeepSeek/` | ✓ | ✓ |
| **DeepSeek-R1-Distill-Qwen-14B** | - | `example/DeepSeek/` | ✓ | ✓ |
| **DeepSeek-R1-Distill-Qwen-32B** | - | `example/DeepSeek/` | ✓ | ✓ |

### LLaMA Series

| Model | One-Click | V0 Script | W8A8 | W8A8S |
|-------|-----------|-----------|------|-------|
| **LLaMA2-7B** | - | `example/Llama/` | ✓ | ✓ |
| **LLaMA2-13B** | - | `example/Llama/` | ✓ | ✓ |
| **LLaMA2-70B** | - | `example/Llama/` | ✓ | - |
| **LLaMA3-70B** | - | `example/Llama/` | ✓ | - |
| **LLaMA3.1-8B** | - | `example/Llama/` | ✓ | - |
| **LLaMA3.1-70B** | - | `example/Llama/` | ✓ | ✓ |

### GLM Series

| Model | One-Click | V0 Script | W8A8 | W8A8C8 | W8A8S |
|-------|-----------|-----------|------|--------|-------|
| **GLM-4-9B** | - | `example/GLM/` | ✓ | ✓ | ✓ |

### InternLM Series

| Model | One-Click | V0 Script | W8A8 | W8A8C8 |
|-------|-----------|-----------|------|--------|
| **InternLM2-20B** | - | `example/InternLM2/` | ✓ | ✓ |

### HunYuan Series

| Model | One-Click | V0 Script | W8A8 |
|-------|-----------|-----------|------|
| **Hunyuan-A52B-Instruct** | - | `example/HunYuan/` | ✓ |

### Kimi Series

| Model | One-Click | V0 Script |
|-------|-----------|-----------|
| **Kimi K2** | - | `example/Kimi/` |

---

## MoE (Mixture of Experts) Models

msModelSlim supports MoE architecture models from various model families.

| Model | One-Click | V0 Script | W8A8 | W4A8 | Dependencies |
|-------|-----------|-----------|------|------|--------------|
| **Qwen3-30B-A3B** | - | `example/Qwen3-MOE/` | ✓ | ✓ | transformers==4.51.0 |
| **Qwen3-235B-A22B** | ✓ | `example/Qwen3-MOE/` | ✓ | ✓ | transformers==4.51.0 |
| **Qwen3-Coder-480B-A35B** | ✓ | `example/Qwen3-MOE/` | - | ✓ | transformers==4.51.0 |
| **Qwen3-Next-80B-A3B-Instruct** | ✓ | `example/Qwen3-Next/` | ✓ | - | transformers>=4.57.0 |
| **DeepSeek MoE Series** | ✓ | `example/DeepSeek/` | ✓ | ✓ | transformers==4.48.2 |

---

## Multimodal Vision-Language Models (VLM)

| Model | One-Click | V0 Script | W8A8 | W8A8S | Dependencies |
|-------|-----------|-----------|------|-------|--------------|
| **Qwen3-VL-4B-Instruct** | - | `example/multimodal_vlm/Qwen3-VL/` | ✓ | - | transformers==4.57.1 |
| **Qwen3-VL-8B-Instruct** | - | `example/multimodal_vlm/Qwen3-VL/` | - | ✓ | transformers==4.57.1 |
| **Qwen3-VL-32B-Instruct** | - | `example/multimodal_vlm/Qwen3-VL/` | ✓ | - | transformers==4.57.1 |
| **Qwen3-VL-235B-A22B** | ✓ | `example/multimodal_vlm/Qwen3-VL-MoE/` | ✓ | - | transformers==4.57.1, flax |
| **Qwen2.5-VL-7B** | - | `example/multimodal_vlm/Qwen2.5-VL/` | ✓ | - | transformers==4.49.0, qwen_vl_utils |
| **Qwen2.5-VL-72B** | - | `example/multimodal_vlm/Qwen2.5-VL/` | ✓ | - | transformers==4.49.0, qwen_vl_utils |
| **Qwen2-VL-7B** | - | `example/multimodal_vlm/Qwen2-VL/` | ✓ | - | transformers==4.46.0, qwen_vl_utils |
| **Qwen2-VL-72B** | - | `example/multimodal_vlm/Qwen2-VL/` | ✓ | - | transformers==4.46.0, qwen_vl_utils |
| **Qwen-VL** | - | `example/multimodal_vlm/Qwen-VL/` | ✓ | - | transformers-stream-generator |
| **InternVL2-8B** | - | `example/multimodal_vlm/InternVL2/` | ✓ | - | transformers==4.46.0, timm, fastchat |
| **InternVL2-40B** | - | `example/multimodal_vlm/InternVL2/` | ✓ | - | transformers==4.46.0, timm, fastchat |
| **LLaVA-1.5-7B** | - | `example/multimodal_vlm/LLaVA/` | ✓ | - | transformers==4.37.2 |
| **GLM-4.1V-9B-Thinking** | - | `example/multimodal_vlm/GLM-4.1V/` | - | ✓ | transformers==4.53.0 |

---

## Multimodal Generation Models

| Model | One-Click | V0 Script | W8A8 | W8A8C8 | Dependencies |
|-------|-----------|-----------|------|--------|--------------|
| **SD3-Medium** | - | `example/multimodal_sd/` | ✓ | - | diffusers |
| **Open-Sora-Plan v1.2** | - | `example/multimodal_sd/` | ✓ | - | huggingface_hub==0.25.2 |
| **FLUX.1-dev** | - | `example/multimodal_sd/` | ✓ | ✓ | - |
| **HunyuanVideo** | - | `example/multimodal_sd/` | ✓ | ✓ | - |
| **Wan2.1** | ✓ | `example/multimodal_sd/` | ✓ | - | - |

---

## Quantization Type Notes

| Symbol | Meaning |
|--------|---------|
| ✓ | Supported and verified |
| - | Not supported or not verified |
| **One-Click** | Supports one-click quantization (V1) |
| **V0 Script** | Supports traditional quantization via example scripts |

### Special Notes

1. **W8A16**: MindIE only
2. **W8A8C8 / W4A8C8**: KVCache + FA3 quantization, MindIE only
3. **W8A8S / W16A16S**: Sparse quantization, Atlas 300I Duo optimized, MindIE only

---

## Best Practice Configs

Best practice YAML configurations are available in the repository:

```bash
# View best practice configs
ls msmodelslim/lab_practice/

# Example locations:
# lab_practice/qwen2/qwen2-72b-w8a8.yaml
# lab_practice/qwen2_5/qwen2.5-32b-w8a8.yaml
# lab_practice/deepseek_v3_2/deepseek_w8a8_quarot.yaml
```

---

## References

- [Official Model Support Matrix](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/model_support/foundation_model_support_matrix/)
- [Example Scripts](example-scripts.md)
