# Quantization Algorithms

Complete guide to quantization algorithms supported by msModelSlim.

---

## Algorithm Overview

msModelSlim supports multiple advanced quantization algorithms across three categories:
1. **Outlier Suppression** - Smooth activation distributions
2. **Quantization** - Core quantization methods
3. **Auto-Tuning** - Automatic optimization strategies

---

## Outlier Suppression Algorithms

Outlier suppression algorithms smooth activation distributions to reduce quantization precision loss.

### SmoothQuant

**Core Idea**: Co-scale activation and weights to smooth outliers.

**When to Use**: Standard outlier suppression for most models.

**Algorithm**: Applies scaling factor to balance activation and weight quantization difficulty.

```yaml
# Config example
anti_method: m1  # SmoothQuant
```

**Reference**: [SmoothQuant Documentation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/outlier_suppression_algorithms/smooth_quant/)

### QuaRot

**Core Idea**: Apply orthogonal rotation matrix to smooth activation distribution.

**When to Use**: High precision requirements, better outlier suppression.

**Algorithm**: Uses random orthogonal matrices to rotate activation distributions.

```yaml
# Config example
process:
  - type: "quarot"
    # QuaRot configuration
```

**Reference**: [QuaRot Documentation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/outlier_suppression_algorithms/quarot/)

### Iterative Smooth

**Core Idea**: Iterative smoothing for finer distribution adjustment.

**When to Use**: Complex distributions, when single-pass smooth is insufficient.

**Reference**: [Iterative Smooth Documentation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/outlier_suppression_algorithms/iterative_smooth/)

### Flex Smooth Quant

**Core Idea**: Two-stage grid search for optimal alpha/beta parameters.

**When to Use**: Different architectures, need automatic parameter tuning.

**Reference**: [Flex Smooth Documentation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/outlier_suppression_algorithms/flex_smooth_quant/)

### Flex AWQ SSZ

**Core Idea**: Combines AWQ with SSZ, uses real quantizer for error evaluation.

**When to Use**: Automatic optimal smoothing parameter search.

**Reference**: [Flex AWQ SSZ Documentation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/outlier_suppression_algorithms/flex_awq_ssz/)

### KV Smooth

**Core Idea**: Smoothing algorithm for KV Cache.

**When to Use**: KV Cache quantization scenarios, long sequence inference.

**Reference**: [KV Smooth Documentation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/outlier_suppression_algorithms/kv_smooth/)

---

## Quantization Algorithms

### AutoRound

**Core Idea**: SignSGD optimization for rounding offsets, minimizing reconstruction error.

**When to Use**: 4-bit ultra-low quantization, W4A8 scenarios.

**Algorithm**: Optimizes rounding decisions instead of simple nearest-integer rounding.

```yaml
# Config example
qconfig:
  weight:
    method: "autoround"
    dtype: "int4"
```

**Reference**: [AutoRound Documentation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/quantization_algorithms/autoround/)

### GPTQ

**Core Idea**: Column-wise optimization with error compensation.

**When to Use**: High precision weight quantization requirements.

**Algorithm**: Minimizes quantization error through layer-by-layer optimization.

**Reference**: [GPTQ Documentation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/quantization_algorithms/gptq/)

### SSZ

**Core Idea**: Iterative search for optimal scaling factor and offset.

**When to Use**: Uneven weight distributions, precision optimization.

**Algorithm**: Iteratively optimizes scale and zero-point for better quantization.

**Reference**: [SSZ Documentation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/quantization_algorithms/ssz/)

### PDMIX

**Core Idea**: Prefilling uses dynamic quantization, Decoding uses static quantization.

**When to Use**: Large model inference acceleration, balance precision and performance.

**Algorithm**: Hybrid approach leveraging benefits of both dynamic and static quantization.

```yaml
# Config example
process:
  - type: "smooth_quant"
  - type: "linear_quant"
    qconfig:
      # PDMIX specific configuration
```

**Reference**: [PDMIX Documentation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/quantization_algorithms/pdmix/)

### FA3 Quant

**Core Idea**: Per-head INT8 quantization for attention activations.

**When to Use**: Long sequence inference, MLA architecture models.

**Algorithm**: Quantizes attention activations at per-head granularity.

**Reference**: [FA3 Documentation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/quantization_algorithms/fa3_quant/)

### KVCache Quant

**Core Idea**: Quantization for KV Cache.

**When to Use**: Long sequence inference, memory optimization.

**Algorithm**: Quantizes K and V caches in attention layers.

```yaml
# Config example
qconfig:
  kv_cache:
    dtype: "int8"
```

**Reference**: [KVCache Quant Documentation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/quantization_algorithms/kvcache_quant/)

### Linear Quant

**Core Idea**: Basic linear layer weight and activation quantization.

**When to Use**: Basic quantization scenarios.

**Reference**: [Linear Quant Documentation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/quantization_algorithms/linear_quant/)

### MinMax

**Core Idea**: Statistics of min/max values to determine quantization range.

**When to Use**: Basic quantization, low computational overhead.

**Algorithm**: Simple min-max range calculation.

```yaml
# Config example
qconfig:
  act:
    method: "minmax"
  weight:
    method: "minmax"
```

**Reference**: [MinMax Documentation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/quantization_algorithms/minmax/)

### Histogram

**Core Idea**: Analyze histogram distribution to find optimal truncation interval.

**When to Use**: Filter outliers, improve precision.

**Algorithm**: Histogram-based percentile truncation.

**Reference**: [Histogram Documentation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/quantization_algorithms/histogram_activation_quantization/)

### LAOS (W4A4)

**Core Idea**: Optimization for W4A4 extreme low-bit scenarios.

**When to Use**: Extreme compression requirements.

**Reference**: [LAOS Documentation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/quantization_algorithms/laos/)

### Float Sparse

**Core Idea**: ADMM-based floating-point sparsification.

**When to Use**: High compression rate requirements, Atlas 300I Duo.

**Reference**: [Float Sparse Documentation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/quantization_algorithms/float_sparse/)

---

## Auto-Tuning Strategies

### Standing High

**Core Idea**: Binary search to minimize fallback layers while meeting precision requirements.

**When to Use**: Need fine-grained control over fallback strategy, provide complete quantization config.

**Algorithm**: 
1. Start with all layers quantized
2. If precision drops, fallback sensitive layers
3. Binary search for minimal fallback set

```yaml
# Config example
auto_tuning:
  strategy: "standing_high"
  precision_threshold: 0.95
```

**Reference**: [Standing High Documentation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/auto_tuning_strategies/standing_high/)

### Standing High With Experience

**Core Idea**: Expert experience-based automatic config generation.

**When to Use**: Familiar with model structure, no need to provide complete config.

**Reference**: [Standing High Documentation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/auto_tuning_strategies/standing_high/)

---

## Sensitive Layer Analysis

Evaluate layer sensitivity to quantization for precision protection strategies.

### std (Standard Deviation)

**Core Idea**: Evaluate data variation and range relationship.

**When to Use**: General quantization, baseline evaluation method.

### quantile (Quantile-based)

**Core Idea**: Evaluate distribution robustness using quantiles and IQR.

**When to Use**: Long-tail distributions or extreme outliers.

### kurtosis (Kurtosis-based)

**Core Idea**: Measure distribution peak sharpness and tail thickness.

**When to Use**: Identify extreme value impact, precision-critical scenarios.

**Reference**: [Sensitive Layer Analysis Documentation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/sensitive_layer_analysis/algorithms/)

---

## Algorithm Selection Guide

| Priority | Recommended Algorithm Combination |
|----------|----------------------------------|
| **Beginners** | One-click quantization (auto-configured) |
| **Precision first** | QuaRot + AutoRound |
| **Long sequence** | FA3 + KVCache Quant |
| **High compression** | SSZ + Sparse Quant |
| **Custom model** | SmoothQuant + MinMax (start simple) |

---

## References

- [Algorithm Overview](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/)
- [Best Practice Configs](https://gitcode.com/Ascend/msmodelslim/tree/master/lab_practice)
