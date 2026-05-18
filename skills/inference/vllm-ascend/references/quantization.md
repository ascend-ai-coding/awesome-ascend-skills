# Quantization Support in vLLM-Ascend

vLLM-Ascend supports quantized models for improved inference performance and reduced memory usage on Ascend NPU devices. This guide covers the quantization types supported and how to deploy quantized models.

## Overview of Quantization Support

Quantization reduces model size and memory bandwidth requirements by representing weights and activations with lower precision integers. vLLM-Ascend supports Ascend-native quantization formats produced by the msmodelslim toolkit.

**Key Benefits:**
- Reduced memory footprint for large models
- Improved inference throughput
- Lower power consumption
- Support for larger batch sizes

**Supported Quantization Workflow:**
1. Quantize model using msmodelslim toolkit
2. Deploy with vLLM-Ascend using `--quantization ascend`

## Quantization Types

### W8A8 (INT8 Weights + INT8 Activations)

W8A8 quantization uses 8-bit integers for both model weights and activations.

| Aspect | Details |
|--------|---------|
| Weight Precision | INT8 |
| Activation Precision | INT8 |
| Memory Reduction | ~50% |
| Accuracy Impact | Minimal for most models |
| Recommended Use | General purpose, balanced performance |

**Best for:** Production deployments where accuracy is critical and moderate compression is acceptable.

### W4A8 (INT4 Weights + INT8 Activations)

W4A8 uses 4-bit weights with 8-bit activations, providing higher compression.

| Aspect | Details |
|--------|---------|
| Weight Precision | INT4 |
| Activation Precision | INT8 |
| Memory Reduction | ~75% |
| Accuracy Impact | Slight for most models |
| Recommended Use | Memory-constrained environments |

**Best for:** Deploying large models on devices with limited memory.

### W4A4 (INT4 Weights + INT4 Activations)

W4A4 uses 4-bit precision for both weights and activations, offering maximum compression.

| Aspect | Details |
|--------|---------|
| Weight Precision | INT4 |
| Activation Precision | INT4 |
| Memory Reduction | ~87.5% |
| Accuracy Impact | Moderate - requires evaluation |
| Recommended Use | Maximum compression scenarios |

**Best for:** Edge deployments and extreme memory constraints where some accuracy trade-off is acceptable.

### W8A8C8 (INT8 + KV Cache Quantization)

W8A8C8 extends W8A8 with additional KV cache quantization for long sequence handling.

| Aspect | Details |
|--------|---------|
| Weight Precision | INT8 |
| Activation Precision | INT8 |
| KV Cache Precision | INT8 |
| Memory Reduction | ~50% model + reduced KV cache |
| Recommended Use | Long context applications |

**Best for:** Applications with long input sequences or large context windows.

## Quantization Process

The quantization process is handled by the msmodelslim toolkit. For detailed instructions on quantizing models, see:

**[msmodelslim Skill - Quantization Guide](../msmodelslim/SKILL.md)**

The general workflow:
1. Prepare calibration data
2. Run msmodelslim quantization tool
3. Validate quantized model accuracy
4. Export to Ascend-compatible format

## Deploying Quantized Models

### Basic Deployment

Deploy a quantized model using the `--quantization ascend` flag:

```bash
vllm serve /path/to/quantized-model \
    --quantization ascend
```

### With Additional Options

```bash
vllm serve /path/to/quantized-model \
    --quantization ascend \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --dtype float16
```

### Configuration Examples

**Single NPU deployment:**
```bash
vllm serve /path/to/w8a8-model \
    --quantization ascend \
    --device npu \
    --max-model-len 8192
```

**Multi-NPU deployment with tensor parallelism:**
```bash
vllm serve /path/to/w4a8-model \
    --quantization ascend \
    --tensor-parallel-size 4 \
    --device npu
```

**With specific dtype:**
```bash
vllm serve /path/to/quantized-model \
    --quantization ascend \
    --dtype float16 \
    --device npu
```

### OpenAI-Compatible Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/quantized-model \
    --quantization ascend \
    --device npu \
    --port 8000
```

## Important Notes

- Quantized models must be produced using msmodelslim toolkit
- The `--quantization ascend` flag is required for proper inference
- Quantization type (W8A8, W4A8, etc.) is embedded in the model during conversion
- Performance gains vary based on model size and hardware configuration

## References

- [vLLM-Ascend Quantization Documentation](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html)
- [msmodelslim Quantization Guide](../msmodelslim/SKILL.md)
