# Precision Tuning Guide

Guide for tuning quantization precision in msModelSlim.

---

## Overview

Quantization precision tuning involves:
1. **Sensitive Layer Analysis** - Identify layers sensitive to quantization
2. **Auto-Tuning Strategies** - Automatically optimize quantization config
3. **Manual Fallback** - Manually exclude sensitive layers

---

## Sensitive Layer Analysis

### Usage

```bash
# Analyze model sensitivity
msmodelslim analyze \
    --model_path /path/to/model \
    --model_type Qwen2.5-7B-Instruct \
    --device npu
```

### Analysis Algorithms

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **std** | Standard deviation based | General quantization, baseline evaluation |
| **quantile** | Quantile/IQR based | Long-tail distributions, extreme outliers |
| **kurtosis** | Kurtosis based | Extreme value detection, precision-critical |

### Algorithm Details

#### std (Standard Deviation)

Evaluates data variation and range relationship.

```yaml
# Config
analysis:
  algorithm: std
  threshold: 1.0
```

**When to use**: First choice for general quantization scenarios.

#### quantile (Quantile-based)

Evaluates distribution robustness using quantiles and Interquartile Range (IQR).

```yaml
# Config
analysis:
  algorithm: quantile
  lower_quantile: 0.25
  upper_quantile: 0.75
```

**When to use**: Data has long-tail distribution or extreme outliers.

#### kurtosis (Kurtosis-based)

Measures distribution peak sharpness and tail thickness.

```yaml
# Config
analysis:
  algorithm: kurtosis
  threshold: 3.0
```

**When to use**: Need to identify extreme value impact for precision-critical scenarios.

---

## Auto-Tuning Strategies

### Standing High

**Core Idea**: Binary search to minimize fallback layers while meeting precision requirements.

**Workflow**:
1. Start with all layers quantized
2. Evaluate precision
3. If precision drops, identify and fallback sensitive layers
4. Binary search for minimal fallback set

```yaml
# Config example
apiversion: modelslim_v1
spec:
  auto_tuning:
    strategy: "standing_high"
    precision_threshold: 0.95
    max_iterations: 10
```

### Standing High With Experience

**Core Idea**: Expert experience-based automatic config generation.

**When to use**: Familiar with model structure, no need to provide complete config.

```bash
# Use experience-based auto-tuning
msmodelslim quant \
    --model_path /path/to/model \
    --save_path /path/to/output \
    --model_type Qwen2.5-7B-Instruct \
    --quant_type w8a8 \
    --auto_tuning experience
```

---

## W8A8 Precision Tuning

### Strategy

1. **Start Simple**: Use default config without outlier suppression
2. **Add SmoothQuant**: If precision drops, enable SmoothQuant
3. **Use QuaRot**: For higher precision requirements
4. **Fallback Layers**: Identify and exclude sensitive layers

### Recommended Config

```yaml
apiversion: modelslim_v1
spec:
  process:
    - type: "smooth_quant"  # Optional, add if precision drops
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_token"
          dtype: "int8"
          symmetric: true
          method: "minmax"
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: true
          method: "minmax"
        include: ["*"]
        exclude: []  # Add sensitive layers here
  save:
    - type: "ascendv1_saver"
```

### Common Sensitive Layers

For transformer models, common sensitive layers include:
- First and last layers
- Attention projection layers (sometimes)
- Down projection layers (`*down_proj*`)

---

## W8A16 Precision Tuning

### Strategy

W8A16 (weight-only quantization) is simpler with less precision loss.

1. **Use Default Config**: W8A16 typically works well
2. **Check Attention Layers**: Sometimes need fallback

### Recommended Config

```yaml
apiversion: modelslim_v1
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          dtype: "float16"  # No activation quantization
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: true
          method: "minmax"
        include: ["*"]
        exclude: []
```

---

## Sparse Quantization Tuning

### Strategy

Sparse quantization (W8A8S) requires additional tuning for sparsity ratio.

1. **Determine Sparsity Ratio**: Typically 2:4 or 4:8 pattern
2. **Outlier Suppression**: Essential for maintaining precision
3. **Calibration Data**: Use representative calibration dataset

### Recommended Config

```yaml
apiversion: modelslim_v1
spec:
  process:
    - type: "smooth_quant"
    - type: "float_sparse"
      sparsity_ratio: 0.5  # 2:4 sparsity
    - type: "linear_quant"
      qconfig:
        # Standard W8A8 config
```

---

## Case Studies

### Qwen3-32B W8A8 Tuning

**Problem**: Initial W8A8 quantization shows precision drop.

**Solution**:
1. Enable SmoothQuant for outlier suppression
2. Analyze sensitive layers
3. Fallback `down_proj` layers if needed

**Config**: See `lab_practice/qwen3/qwen3-32b-w8a8.yaml` in repository.

### Large Model Memory Optimization

**Problem**: Out of memory during quantization.

**Solution**:
1. Use layer-by-layer quantization (default in V1)
2. Use CPU quantization: `--device cpu`
3. Multi-device distributed quantization: `--device npu:0,1`

---

## Validation

### Quantization Evaluation

```bash
# Compare with original model
python3 -c "
from vllm import LLM, SamplingParams

# Original model
orig_llm = LLM(model='/path/to/original')

# Quantized model
quant_llm = LLM(model='/path/to/quantized', quantization='ascend')

# Compare outputs
prompts = ['Test prompt 1', 'Test prompt 2']
# ...
"
```

### Dataset Evaluation

Use benchmark datasets for evaluation:
- **MMLU**: General knowledge
- **GSM8K**: Math reasoning
- **HumanEval**: Code generation

---

## Best Practices

1. **Start with best practice configs**: Check `lab_practice/` directory
2. **Use representative calibration data**: Match your use case
3. **Compare with baseline**: Always validate against original model
4. **Iterate gradually**: Start simple, add complexity as needed
5. **Document changes**: Keep track of what works

---

## References

- [Quantization Precision Tuning Guide](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/case_studies/quantization_precision_tuning_guide/)
- [W8A8 Tuning Strategy](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/case_studies/w8a8_accuracy_tuning_policy/)
- [W8A16 Tuning Strategy](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/case_studies/w8a16_accuracy_tuning_policy/)
- [Qwen3-32B W8A8 Case](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/case_studies/qwen3-32b_w8a8_precision_tuning_case/)
- [Sparse Quantization Cases](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/case_studies/sparse_quantization_accuracy_tuning_cases/)
