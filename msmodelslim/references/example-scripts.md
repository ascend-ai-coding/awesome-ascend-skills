# Example Scripts Guide

Guide for using example scripts in the msModelSlim repository.

---

## Directory Structure

```
msmodelslim/example/
├── common/              # Common utilities
├── DeepSeek/            # DeepSeek model examples
├── GLM/                 # GLM model examples
├── GPT-NeoX/            # GPT-NeoX model examples
├── HunYuan/             # HunYuan model examples
├── InternLM2/           # InternLM2 model examples
├── Kimi/                # Kimi model examples
├── Llama/               # Llama model examples
├── Qwen/                # Qwen model examples
├── Qwen3-MOE/           # Qwen3 MoE model examples
├── Qwen3-Next/          # Qwen3-Next model examples
├── multimodal_sd/       # Multimodal generation models
├── multimodal_vlm/      # Vision-language models
└── osp1_2/              # Open-Sora examples
```

---

## Common Utilities (example/common/)

### Calibration Data

- `boolq.jsonl` - BoolQ dataset for calibration
- `teacher_qualification.jsonl` - Teacher qualification dataset

### Utility Scripts

| Script | Description |
|--------|-------------|
| `convert_fp8_to_bf16.py` | Convert FP8 weights to BF16 |
| `ms_to_vllm.py` | Convert msModelSlim output to vLLM format |
| `add_safetensors.py` | Add safetensors support |
| `copy_config_files.py` | Copy configuration utilities |

#### ms_to_vllm.py - Weight Conversion

```bash
# Convert to vLLM format
python3 example/common/ms_to_vllm.py \
    --input /path/to/msmodelslim-output \
    --output /path/to/vllm-model
```

#### convert_fp8_to_bf16.py - Format Conversion

```bash
# Convert FP8 to BF16
python3 example/common/convert_fp8_to_bf16.py \
    --input /path/to/fp8-model \
    --output /path/to/bf16-model
```

---

## Qwen Examples (example/Qwen/)

### quant_qwen.py - Standard Quantization

```bash
python3 example/Qwen/quant_qwen.py \
    --model_path /path/to/Qwen2.5-7B-Instruct \
    --save_directory /path/to/output \
    --calib_file example/common/boolq.jsonl \
    --w_bit 8 \
    --a_bit 8 \
    --device_type npu \
    --trust_remote_code True
```

**Parameters**:
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_path` | Path to original model | Required |
| `--save_directory` | Output directory | Required |
| `--calib_file` | Calibration data file | `teacher_qualification.jsonl` |
| `--w_bit` | Weight bits | 8 |
| `--a_bit` | Activation bits | 8 |
| `--device_type` | Device type | `cpu` |
| `--act_method` | Activation method | 1 (minmax) |
| `--anti_method` | Outlier suppression | None |
| `--model_type` | Qwen version | `qwen2` |
| `--trust_remote_code` | Trust remote code | False |

### quant_qwen_pdmix.py - PDMIX Quantization

```bash
python3 example/Qwen/quant_qwen_pdmix.py \
    --model_path /path/to/Qwen2.5-72B-Instruct \
    --save_directory /path/to/output \
    --device_type npu \
    --trust_remote_code True
```

### w4a4.py - W4A4 Quantization

```bash
python3 example/Qwen/w4a4.py \
    --model_path /path/to/Qwen3-32B \
    --save_directory /path/to/output \
    --device_type npu
```

---

## DeepSeek Examples (example/DeepSeek/)

### quant_deepseek.py - Standard Quantization

```bash
python3 example/DeepSeek/quant_deepseek.py \
    --model_path /path/to/DeepSeek-V3 \
    --save_directory /path/to/output \
    --device_type npu
```

### quant_deepseek_w4a8.py - W4A8 Quantization

```bash
python3 example/DeepSeek/quant_deepseek_w4a8.py \
    --model_path /path/to/DeepSeek-V3.2-Exp \
    --save_directory /path/to/output \
    --device_type npu
```

### quant_deepseek_w8a8.py - W8A8 Quantization

```bash
python3 example/DeepSeek/quant_deepseek_w8a8.py \
    --model_path /path/to/DeepSeek-V3.2 \
    --save_directory /path/to/output \
    --device_type npu
```

---

## Llama Examples (example/Llama/)

### quant_llama.py

```bash
python3 example/Llama/quant_llama.py \
    --model_path /path/to/LLaMA3.1-8B \
    --save_directory /path/to/output \
    --calib_file example/common/boolq.jsonl \
    --w_bit 8 \
    --a_bit 8 \
    --device_type npu
```

---

## GLM Examples (example/GLM/)

### quant_glm.py

```bash
python3 example/GLM/quant_glm.py \
    --model_path /path/to/GLM-4-9B \
    --save_directory /path/to/output \
    --device_type npu
```

---

## InternLM2 Examples (example/InternLM2/)

### quant_internlm2.py

```bash
python3 example/InternLM2/quant_internlm2.py \
    --model_path /path/to/InternLM2-20B \
    --save_directory /path/to/output \
    --device_type npu
```

---

## Kimi Examples (example/Kimi/)

### kimi_quant.py

```bash
python3 example/Kimi/kimi_quant.py \
    --model_path /path/to/Kimi-K2 \
    --save_directory /path/to/output \
    --device_type npu
```

---

## HunYuan Examples (example/HunYuan/)

### quant_hunyuan.py

```bash
python3 example/HunYuan/quant_hunyuan.py \
    --model_path /path/to/Hunyuan-A52B-Instruct \
    --save_directory /path/to/output \
    --device_type npu
```

---

## Qwen3-MoE Examples (example/Qwen3-MOE/)

### quant_qwen_moe_w8a8.py

```bash
python3 example/Qwen3-MOE/quant_qwen_moe_w8a8.py \
    --model_path /path/to/Qwen3-235B-A22B \
    --save_directory /path/to/output \
    --device_type npu
```

---

## Multimodal VLM Examples (example/multimodal_vlm/)

### Qwen-VL

```bash
python3 example/multimodal_vlm/Qwen-VL/quant_qwenvl.py \
    --model_path /path/to/Qwen-VL \
    --save_directory /path/to/output \
    --device_type npu
```

### Qwen2-VL

```bash
python3 example/multimodal_vlm/Qwen2-VL/quant_qwen2vl.py \
    --model_path /path/to/Qwen2-VL-7B \
    --save_directory /path/to/output \
    --device_type npu
```

### Qwen3-VL

```bash
python3 example/multimodal_vlm/Qwen3-VL/quant_qwen3vl.py \
    --model_path /path/to/Qwen3-VL-4B-Instruct \
    --save_directory /path/to/output \
    --device_type npu
```

### InternVL2

```bash
python3 example/multimodal_vlm/InternVL2/quant_internvl2.py \
    --model_path /path/to/InternVL2-8B \
    --save_directory /path/to/output \
    --device_type npu
```

### LLaVA

```bash
python3 example/multimodal_vlm/LLaVA/quant_llava.py \
    --model_path /path/to/LLaVA-1.5-7B \
    --save_directory /path/to/output \
    --device_type npu
```

### GLM-4.1V

```bash
python3 example/multimodal_vlm/GLM-4.1V/quant_glm4v.py \
    --model_path /path/to/GLM-4.1V-9B-Thinking \
    --save_directory /path/to/output \
    --device_type npu
```

---

## Multimodal Generation Examples (example/multimodal_sd/)

### SD3

```bash
python3 example/multimodal_sd/sd3_inference.py \
    --model_path /path/to/SD3-Medium \
    --output_path /path/to/output
```

### FLUX

```bash
python3 example/multimodal_sd/inference_flux.py \
    --model_path /path/to/FLUX.1-dev \
    --output_path /path/to/output
```

---

## Multi-Device Quantization

For multi-device quantization, set environment variables:

```bash
# Set visible devices
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# Memory allocation config
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False

# Run quantization
python3 example/Qwen/quant_qwen.py \
    --model_path /path/to/large-model \
    --save_directory /path/to/output \
    --device_type npu
```

---

## References

- [Example Directory](https://gitcode.com/Ascend/msmodelslim/tree/master/example)
- [Model Support Matrix](model-support.md)
- [Quantization Algorithms](quantization-algorithms.md)
