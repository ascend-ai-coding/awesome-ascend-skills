# Deployment Guide

Guide for deploying quantized models with vLLM-Ascend and MindIE.

---

## vLLM-Ascend Deployment

### Prerequisites

- vLLM-Ascend installed
- Quantized model from msModelSlim

### Installation

```bash
# Pull Docker image
docker pull quay.io/ascend/vllm-ascend:latest

# Or install via pip
pip install vllm-ascend
```

### Online Service Deployment

```bash
# Start vLLM server with quantized model
vllm serve /path/to/quantized-model \
    --served-model-name "Qwen2.5-7B-w8a8" \
    --max-model-len 4096 \
    --quantization ascend \
    --host 0.0.0.0 \
    --port 8000
```

**Key Parameters**:
- `--quantization ascend`: Use Ascend quantization backend
- `--max-model-len`: Maximum sequence length
- `--served-model-name`: Model name for API

### API Usage

```bash
# Completion API
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen2.5-7B-w8a8",
        "prompt": "What is large language model?",
        "max_tokens": 128,
        "top_p": 0.95,
        "temperature": 0.7
    }'

# Chat API
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen2.5-7B-w8a8",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 128
    }'
```

### Offline Inference (Python)

```python
from vllm import LLM, SamplingParams

# Initialize LLM with quantized model
llm = LLM(
    model="/path/to/quantized-model",
    max_model_len=4096,
    quantization="ascend",  # Required for Ascend quantized models
    tensor_parallel_size=1,  # Adjust for multi-GPU
)

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    top_k=40,
    max_tokens=128,
)

# Generate text
prompts = [
    "Hello, my name is",
    "The future of AI is",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}")
    print()
```

---

## MindIE Deployment

### Prerequisites

- MindIE installed
- Quantized model in compatible format

### Basic Usage

```bash
# Deploy with MindIE
# Refer to MindIE documentation for detailed commands
```

> **Note**: Some quantization types are MindIE-only:
> - W8A16 (weight-only)
> - W8A8C8 / W4A8C8 (KVCache quantization)
> - W8A8S / W16A16S (sparse quantization)

---

## Model Directory Structure

After msModelSlim quantization:

```
quantized-model/
├── config.json                      # Original model config
├── generation_config.json           # Generation config
├── quant_model_description.json     # Quantization metadata
├── quant_model_weight_w8a8.safetensors  # Quantized weights
├── tokenizer_config.json            # Tokenizer config
├── tokenizer.json                   # Tokenizer vocabulary
└── vocab.json                       # Vocabulary (if applicable)
```

### quant_model_description.json

Contains quantization metadata:

```json
{
    "quantization_type": "w8a8",
    "weight_bit": 8,
    "activation_bit": 8,
    "layers": {
        "model.layers.0.self_attn.q_proj": {"weight": "int8", "act": "int8"},
        ...
    }
}
```

---

## Weight Conversion

### Convert to AutoAWQ / AutoGPTQ Format

```bash
# Use conversion script
python3 example/common/ms_to_vllm.py \
    --input /path/to/msmodelslim-output \
    --output /path/to/converted \
    --format awq  # or gptq
```

### FP8 to BF16 Conversion

```bash
# Convert FP8 weights to BF16
python3 example/common/convert_fp8_to_bf16.py \
    --input /path/to/model \
    --output /path/to/converted
```

---

## Multi-Device Deployment

### vLLM Tensor Parallelism

```python
from vllm import LLM

# Multi-GPU deployment
llm = LLM(
    model="/path/to/quantized-model",
    quantization="ascend",
    tensor_parallel_size=4,  # Use 4 GPUs
    max_model_len=8192,
)
```

### vLLM Server with Multi-GPU

```bash
vllm serve /path/to/quantized-model \
    --quantization ascend \
    --tensor-parallel-size 4 \
    --max-model-len 8192
```

---

## Performance Optimization

### Memory Optimization

```bash
# Reduce KV Cache memory
vllm serve /path/to/model \
    --quantization ascend \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096
```

### Throughput Optimization

```bash
# Increase batch size
vllm serve /path/to/model \
    --quantization ascend \
    --max-num-seqs 256 \
    --max-model-len 2048
```

---

## Troubleshooting

### Model Loading Fails

**Problem**: `quantization="ascend"` not recognized.

**Solution**: Ensure vLLM-Ascend is properly installed.

```bash
pip install vllm-ascend
```

### Out of Memory

**Problem**: OOM during inference.

**Solution**:
1. Reduce `--max-model-len`
2. Reduce `--gpu-memory-utilization`
3. Use smaller batch size

### Precision Issues

**Problem**: Quantized model output differs significantly.

**Solution**:
1. Check quantization config
2. Verify calibration data quality
3. Try fallback sensitive layers

---

## References

- [vLLM-Ascend Documentation](https://docs.vllm.ai/projects/ascend/en/latest/)
- [Qwen3-32B W4A4 Tutorial](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/Qwen3-32B-W4A4.html)
- [Weight Use Cases](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/case_studies/quantization_weight_use_cases_in_acceleration_and_mindie_torch/)
- [AutoAWQ/AutoGPTQ Conversion](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/case_studies/msmodelslim_quantized_weight_to_autoawq_autogptq/)
