# Quantization Guide

## Overview

Quantization reduces model memory footprint and improves throughput by using lower-precision weights and activations.

## Quick Reference

| Model Type | Detection | Parameter Required |
|------------|-----------|-------------------|
| Quantized (W8A8, W4A8, MXFP8) | `quant_model_description.json` exists | `--quantization ascend` |
| Non-quantized (FP16/BF16) | No `quant_model_description.json` | No quantization param |

## Detection

### Automatic Detection

```bash
# Check if model is quantized
if [ -f "<model-path>/quant_model_description.json" ]; then
    echo "Quantized model - use --quantization ascend"
else
    echo "Non-quantized model - no quantization param needed"
fi
```

### Quantization File Example

`quant_model_description.json`:
```json
{
    "quant_method": "ascend",
    "quant_level": "W8A8",
    "model_type": "llama"
}
```

## Quantization Types

### W8A8 (8-bit Weights, 8-bit Activations)

- **Memory reduction**: ~50% vs FP16
- **Accuracy**: Minimal loss
- **Speed**: Fast
- **Best for**: General deployment

### W4A8 (4-bit Weights, 8-bit Activations)

- **Memory reduction**: ~75% vs FP16
- **Accuracy**: Slight degradation
- **Speed**: Fast
- **Best for**: Memory-constrained scenarios

### MXFP8 (MX Floating Point 8-bit)

- **Memory reduction**: ~50% vs FP16
- **Accuracy**: Better than W8A8
- **Speed**: Optimized
- **Best for**: High accuracy requirements

## Configuration

### For Quantized Models (REQUIRED)

```bash
vllm serve /model/Qwen3-8B-mxfp8 \
  --quantization ascend \
  ...other params...
```

**Important**: The `--quantization ascend` parameter is **REQUIRED** for quantized models. Without it, you'll get errors.

### For Non-Quantized Models

```bash
vllm serve /model/Qwen3-8B \
  # NO --quantization parameter!
  ...other params...
```

**Important**: Do NOT add `--quantization ascend` for non-quantized models. It will cause errors.

## Decision Tree

```
Is quant_model_description.json present?
├── Yes → Model is quantized
│   └── Add --quantization ascend (REQUIRED)
│
└── No → Model is non-quantized
    └── Do NOT add quantization parameter
```

## Common Errors

### Error: Quantization parameter on non-quantized model

```
ValueError: quantization ascend is not supported for this model
```

**Solution**: Remove `--quantization ascend` parameter

### Error: Missing quantization on quantized model

```
RuntimeError: Expected quantized weights but found FP16
```

**Solution**: Add `--quantization ascend` parameter

### Error: Quantization file not found

```
FileNotFoundError: quant_model_description.json not found
```

**Solution**: Ensure model is properly quantized, or remove quantization parameter

## Quantized Model Naming Conventions

Common suffixes indicating quantized models:

| Suffix | Meaning |
|--------|---------|
| `-mxfp8` | MXFP8 quantized |
| `-w8a8` | W8A8 quantized |
| `-w4a8` | W4A8 quantized |
| `-int8` | INT8 quantized |
| `-quant` | Quantized (general) |

Examples:
- `Qwen3-8B-mxfp8` → Quantized
- `Qwen3-30B-w8a8` → Quantized
- `Qwen3-8B` → Non-quantized (FP16/BF16)

## Performance Impact

| Quantization | Memory | Throughput | Latency | Accuracy |
|--------------|--------|------------|---------|----------|
| None (FP16) | 100% | Baseline | Baseline | 100% |
| W8A8 | ~50% | +20-40% | -10-20% | ~99% |
| W4A8 | ~25% | +30-50% | -15-25% | ~97% |
| MXFP8 | ~50% | +25-45% | -10-20% | ~99.5% |

## Python API

### Quantized Model

```python
from vllm import LLM

llm = LLM(
    model="/model/Qwen3-8B-mxfp8",
    quantization="ascend",  # REQUIRED
    ...
)
```

### Non-Quantized Model

```python
from vllm import LLM

llm = LLM(
    model="/model/Qwen3-8B",
    # NO quantization parameter
    ...
)
```

## Best Practices

1. **Always check for `quant_model_description.json`** before deciding on quantization parameter
2. **Never guess** - let detection determine the configuration
3. **Quantized models**: `--quantization ascend` is mandatory
4. **Non-quantized models**: No quantization parameter needed
5. **When in doubt**: Check model documentation or README

## Troubleshooting Checklist

- [ ] Checked for `quant_model_description.json` in model directory
- [ ] Applied correct quantization parameter (or none)
- [ ] Verified model naming matches quantization status
- [ ] Tested with small request before production deployment
