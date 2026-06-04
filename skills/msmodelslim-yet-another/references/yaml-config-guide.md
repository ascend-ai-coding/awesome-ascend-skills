# msmodelslim YAML Configuration Guide

> **Purpose**: Authoritative reference for writing msmodelslim quantization YAML configs. Read this when building a custom YAML for a model that has no lab_practice config, or when the one-click result needs tuning.

## Table of Contents

- [YAML Top-Level Structure](#yaml-top-level-structure)
- [Processor Pipeline](#processor-pipeline)
- [Outlier Suppression Processors](#outlier-suppression-processors)
- [Quantization Processors](#quantization-processors)
- [Group Meta-Processor](#group-meta-processor)
- [YAML Anchors](#yaml-anchors)
- [include / exclude Wildcard Rules](#include--exclude-wildcard-rules)
- [Save Configuration](#save-configuration)
- [Complete Templates](#complete-templates)
- [From Requirements to YAML (Decision Path)](#from-requirements-to-yaml-decision-path)
- [Common Errors & Troubleshooting](#common-errors--troubleshooting)

______________________________________________________________________

## YAML Top-Level Structure

```yaml
apiversion: modelslim_v1        # required

metadata:                       # optional, for best-practice management
  config_id: my_config          # unique config identifier
  score: 90                     # accuracy score (0-100, for recording)
  verified_model_types:         # verified model list
    - Qwen3-32B-Instruct
  label:                        # quick-search tags
    w_bit: 8
    a_bit: 8
    is_sparse: False
    kv_cache: False

spec:                           # required
  process: [...]                # ordered processor list
  save: [...]                   # save configuration
```

**Minimal YAML** (only `apiversion` and `spec.process`):

```yaml
apiversion: modelslim_v1
spec:
  process:
    - type: "linear_quant"
      qconfig:
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: true
          method: "minmax"
      include: ["*"]
```

______________________________________________________________________

## Processor Pipeline

`spec.process` is an **ordered list** — processors execute in sequence:

```
[outlier suppression] → [quantization]
     or
[group (multiple quantizers with different include/exclude)]
```

### Processor Type Quick Reference

| type | category | purpose | key fields |
|------|----------|---------|------------|
| `smooth_quant` | outlier suppression | Classic SmoothQuant, norm-linear only | `alpha`, `symmetric` |
| `iter_smooth` | outlier suppression | **Recommended default**, iterative smoothing | `alpha`, `symmetric`, `enable_subgraph_type` |
| `flex_smooth_quant` | outlier suppression | Two-stage grid search α/β, high accuracy | auto-searches, no α needed |
| `flex_awq_ssz` | outlier suppression | AWQ+SSZ combined, for INT4/W4A8 | `n_grid` |
| `awq` | outlier suppression | Activation-aware weight protection | `weight_qconfig`, `n_grid` |
| `quarot` | outlier suppression | Orthogonal rotation to smooth activations | — |
| `linear_quant` | quantization | Linear layer quantization (weight+act), most通用 | `qconfig` (weight/act) |
| `autoround_quant` | quantization | signSGD optimization for rounding, INT4 peak accuracy | `iters`, `strategies` |
| `fa3_quant` | quantization | DeepSeek MLA per-head activation quantization | — |
| `kvcache_quant` | quantization | KV Cache quantization | — |
| `group` | meta-processor | Group multiple quantizers with distinct include/exclude | `configs` |

______________________________________________________________________

## Outlier Suppression Processors

Outlier suppression must run **before** `linear_quant`. These do equivalent mathematical transforms, not low-bit data production.

### Iterative Smooth (Recommended Default)

```yaml
- type: "iter_smooth"
  alpha: 0.5                       # migration strength, 0~1. 0=no migration, 1=all to weights
  symmetric: True
  enable_subgraph_type:            # at least one required
    - "norm-linear"                #   RMSNorm → Linear (classic SmoothQuant)
    - "linear-linear"              #   Linear → Linear
    - "ov"                         #   v_proj → o_proj (attention internal)
    - "up-down"                    #   up_proj → down_proj (MLP gating)
  include: ["*"]
  exclude: []
```

**Alpha selection guide**:

| alpha | effect | when to use |
|-------|--------|-------------|
| 0.5 | Equal difficulty split between activation and weight | W8A8 default |
| 0.7–0.9 | More difficulty shifted to weights (offline quantizable) | W4A8, severe activation outliers |
| 0.2–0.3 | More difficulty kept on activation side | Weight-hard-to-quantize (extreme low-bit) |

### SmoothQuant (Classic)

```yaml
- type: "smooth_quant"
  alpha: 0.5
  symmetric: True
  include: ["*"]
  exclude: ["*self_attn*"]          # only supports norm-linear, not ov/up-down
```

Difference from Iterative Smooth: SmoothQuant computes scale once and applies immediately. Iterative Smooth runs multiple rounds for finer adjustment. **New projects should use Iterative Smooth.**

### Flex Smooth Quant (High Accuracy)

```yaml
- type: "flex_smooth_quant"
  enable_subgraph_type:
    - "norm-linear"
  include: ["*"]
```

No manual `alpha` needed — algorithm auto-searches [0, 1] grid. Higher accuracy than Iterative Smooth but 3–4× slower quantization time.

### AWQ (Activation-Aware Weight Protection)

```yaml
- type: "awq"
  weight_qconfig:                   # search-phase weight quant config
    scope: "per_channel"
    dtype: "int4"
    symmetric: true
    method: "minmax"
  n_grid: 20                        # grid search steps (higher = finer)
  enable_subgraph_type:
    - "norm-linear"
    - "linear-linear"
    - "ov"
    - "up-down"
  include: ["*"]
```

AWQ is positioned as an **outlier suppression** processor (not weight quantization). It finds optimal scale factors and fuses them into weights — a subsequent `linear_quant` is still needed for actual quantization.

### QuaRot (Orthogonal Rotation)

```yaml
- type: "quarot"
  include: ["*"]
```

Use when activations have stubborn outliers that SmoothQuant methods can't handle. Applies orthogonal rotation matrices to activations and weights to均匀化 distributions. Often paired with AutoRound for extreme low-bit scenarios.

______________________________________________________________________

## Quantization Processors

### linear_quant (Most General)

```yaml
- type: "linear_quant"
  qconfig:
    weight:                          # weight quantization (omit for act-only)
      scope: "per_channel"           # per_tensor / per_channel / per_group
      dtype: "int8"                  # int8 / int4
      symmetric: true                # true=symmetric, false=asymmetric
      method: "minmax"               # minmax / ssz / gptq / histogram
      ext:                           # extension params (needed for ssz/gptq)
        group_size: 128              #   per_group group size
        percdamp: 0.01               #   gptq damping coefficient
        block_size: 128              #   gptq block size
    act:                             # activation quantization (omit for weight-only)
      scope: "per_tensor"            # per_tensor(static) / per_token(dynamic) / pd_mix(hybrid)
      dtype: "int8"                  # int8 / int4
      symmetric: false               # activations usually asymmetric (non-zero-centered)
      method: "minmax"               # minmax / histogram
  include: ["*"]
  exclude: []
```

**scope selection**:

| scope | object | meaning | when |
|-------|--------|---------|------|
| `per_tensor` | weight/act | One scale for entire tensor | Weight: not recommended. Act: throughput-first |
| `per_channel` | weight | One scale per output channel | Weight default, good accuracy |
| `per_group` | weight | One scale per N elements (needs `group_size`) | INT4 low-bit, better local distribution fit |
| `per_token` | act | One scale per token (dynamic) | Act accuracy-first |
| `pd_mix` | act | prefill=per_token, decode=per_tensor | Balance accuracy and throughput |

**method selection**:

| method | target | bit-width | speed | accuracy | notes |
|--------|--------|-----------|-------|----------|-------|
| `minmax` | weight/act | INT8 | fastest | ★★ | RTN baseline, INT8首选 |
| `ssz` | weight | INT4/INT8 | fast | ★★ | Iterative MSE optimization, INT4首选 |
| `gptq` | weight | INT8 | slow | ★★★ | Hessian compensation, no INT4 support |
| `histogram` | act | INT8 | slower | ★★★ | Histogram truncation, filters tail outliers |

**Symmetric selection**:

| target | symmetric | reason |
|--------|-----------|--------|
| weight | `true` (default) | Weight distributions usually symmetric; hardware optimized for symmetric |
| activation | `false` (default) | Activations (esp. post-ReLU/SiLU) are asymmetric, need Z≠0 |

### autoround_quant (Low-Bit Peak Accuracy)

```yaml
- type: "autoround_quant"
  iters: 400                        # optimization iterations
  enable_minmax_tuning: True
  enable_round_tuning: True
  strategies:                        # supports multi-strategy mixed quantization
    - qconfig:
        weight:
          scope: "per_group"
          dtype: "int4"
          symmetric: True
          method: "autoround"        # fixed value
          ext:
            group_size: 256
            scale_dtype: "bfloat16"
        act:
          scope: "per_token"
          dtype: "int8"
          symmetric: True
          method: "minmax"
      include: ["*"]
      exclude: ["*.down_proj"]       # W4A8: exclude sensitive layers
    - qconfig:
        weight:                      # sensitive layers → INT8
          scope: "per_channel"
          dtype: "int8"
          symmetric: True
          method: "autoround"
        act:
          scope: "per_token"
          dtype: "int8"
          symmetric: True
          method: "minmax"
      include: ["*.down_proj"]
```

**Important**: AutoRound strongly benefits from outlier suppression (QuaRot or Iterative Smooth) run before it. Standalone INT4 AutoRound may have significantly degraded accuracy.

______________________________________________________________________

## Group Meta-Processor

`group` bundles multiple processors, each with its own `include`/`exclude`. This is the core mechanism for **per-structure mixed quantization**.

```yaml
- type: "group"
  configs:
    # Config 1: self-attention quantization (exclude o_proj)
    - type: "linear_quant"
      qconfig: *default_w8a8          # YAML anchor reference
      include: ["*self_attn*"]
      exclude: ["*self_attn.o_proj*"]

    # Config 2: MoE expert quantization
    - type: "linear_quant"
      qconfig: *default_w8a8
      include: ["*mlp.experts*"]

    # Config 3: everything else → implicitly kept FP16 (not in any include)
```

**Group matching rules**:

- Each layer is handled by the **first matching** sub-config only
- Sub-configs are tried in `configs` list order
- Layers not matched by any sub-config stay FP16 (implicit fallback)

______________________________________________________________________

## YAML Anchors

Use anchors to avoid repeating shared quantization parameters:

```yaml
# Define anchor
default_w8a8_dynamic: &default_w8a8_dynamic
  act:
    scope: "per_token"
    dtype: "int8"
    symmetric: True
    method: "minmax"
  weight:
    scope: "per_channel"
    dtype: "int8"
    symmetric: True
    method: "minmax"

# Use anchor
spec:
  process:
    - type: "linear_quant"
      qconfig: *default_w8a8_dynamic
      include: ["*self_attn*"]
    - type: "linear_quant"
      qconfig: *default_w8a8_dynamic
      include: ["*mlp*"]
```

______________________________________________________________________

## include / exclude Wildcard Rules

| wildcard | meaning | example |
|----------|---------|---------|
| `*` | Match any character sequence | `"*"` matches all layers |
| `*self_attn*` | Contains `self_attn` | Matches `model.layers.0.self_attn.q_proj` |
| `*.down_proj` | Ends with `.down_proj` | Matches all layers' `down_proj` |
| `model.layers.3.*` | Everything under `model.layers.3` | Matches layer 3 all submodules |

**Priority**: `exclude` overrides `include`. If both match, the layer is excluded.

**Debugging**: If `include`/`exclude` match nothing, msmodelslim prints a warning. Use `model.named_modules()` to verify actual module names.

______________________________________________________________________

## Save Configuration

```yaml
spec:
  process: [...]
  save:
    - type: "ascendv1_saver"
      part_file_size: 4              # shard size in GB
```

______________________________________________________________________

## Complete Templates

### Template 1: W8A8 General (Most Models)

```yaml
apiversion: modelslim_v1

default_w8a8: &default_w8a8
  act:
    scope: "per_tensor"              # static quantization for throughput
    dtype: "int8"
    symmetric: false
    method: "minmax"
  weight:
    scope: "per_channel"
    dtype: "int8"
    symmetric: true
    method: "minmax"

spec:
  process:
    - type: "iter_smooth"
      alpha: 0.5
      symmetric: True
      enable_subgraph_type:
        - "norm-linear"
        - "linear-linear"
        - "ov"
        - "up-down"
      include: ["*"]

    - type: "linear_quant"
      qconfig: *default_w8a8
      include: ["*"]
      exclude:
        - "*lm_head*"

  save:
    - type: "ascendv1_saver"
      part_file_size: 4
```

### Template 2: W4A8 Extreme Compression

```yaml
apiversion: modelslim_v1

default_w4a8: &default_w4a8
  act:
    scope: "per_token"               # dynamic quantization for accuracy
    dtype: "int8"
    symmetric: false
    method: "minmax"
  weight:
    scope: "per_group"
    dtype: "int4"
    symmetric: true
    method: "ssz"                    # INT4首选 ssz
    ext:
      group_size: 128

spec:
  process:
    - type: "iter_smooth"
      alpha: 0.7                     # more difficulty shifted to weights
      symmetric: True
      enable_subgraph_type:
        - "norm-linear"
        - "linear-linear"
        - "ov"
        - "up-down"
      include: ["*"]

    - type: "linear_quant"
      qconfig: *default_w4a8
      include: ["*"]
      exclude:
        - "*lm_head*"
        - "*embed_tokens*"           # embedding extremely sensitive to INT4

  save:
    - type: "ascendv1_saver"
      part_file_size: 4
```

### Template 3: W4A16 Weight-Only (Consumer GPU)

```yaml
apiversion: modelslim_v1

default_w4a16: &default_w4a16
  weight:                            # weight-only: no act block
    scope: "per_group"
    dtype: "int4"
    symmetric: true
    method: "ssz"
    ext:
      group_size: 128

spec:
  process:
    - type: "awq"                    # AWQ to protect important channels
      weight_qconfig:
        scope: "per_channel"
        dtype: "int4"
        symmetric: true
        method: "minmax"
      n_grid: 20
      enable_subgraph_type:
        - "norm-linear"
        - "linear-linear"
        - "ov"
        - "up-down"
      include: ["*"]

    - type: "linear_quant"
      qconfig: *default_w4a16
      include: ["*"]
      exclude:
        - "*lm_head*"
        - "*embed_tokens*"

  save:
    - type: "ascendv1_saver"
      part_file_size: 4
```

### Template 4: MoE Mixed Precision (Qwen3-Next Style)

```yaml
apiversion: modelslim_v1

default_w8a8_dynamic: &default_w8a8_dynamic
  act:
    scope: "per_token"
    dtype: "int8"
    symmetric: True
    method: "minmax"
  weight:
    scope: "per_channel"
    dtype: "int8"
    symmetric: True
    method: "minmax"

default_w4a8_dynamic: &default_w4a8_dynamic
  act:
    scope: "per_token"
    dtype: "int8"
    symmetric: True
    method: "minmax"
  weight:
    scope: "per_channel"
    dtype: "int4"
    symmetric: True
    method: "ssz"

spec:
  process:
    - type: "flex_smooth_quant"
      enable_subgraph_type:
        - "norm-linear"
      include: ["*"]

    - type: "group"
      configs:
        # Self-Attention (exclude o_proj)
        - type: "linear_quant"
          qconfig: *default_w8a8_dynamic
          include: ["*self_attn*"]
          exclude: ["*self_attn.o_proj*"]

        # MoE experts → aggressive compression
        - type: "linear_quant"
          qconfig: *default_w4a8_dynamic
          include: ["*mlp.experts*"]

        # Shared MLP (exclude gate)
        - type: "linear_quant"
          qconfig: *default_w8a8_dynamic
          include: ["*mlp*"]
          exclude: ["*gate", "*mlp.experts*"]

        # Linear Attention QKVZ projection
        - type: "linear_quant"
          qconfig: *default_w8a8_dynamic
          include: ["*linear_attn.in_proj_qkvz*"]

        # Embedding / LM Head / Norm / Router / o_proj / Shared Expert
        # not in any include → implicitly kept FP16

  save:
    - type: "ascendv1_saver"
      part_file_size: 4
```

### Template 5: VLM (Vision-Language Model)

```yaml
apiversion: modelslim_v1

default_w8a8_dynamic: &default_w8a8_dynamic
  act: { scope: "per_token", dtype: "int8", symmetric: true, method: "minmax" }
  weight: { scope: "per_channel", dtype: "int8", symmetric: true, method: "minmax" }

default_w4a8_dynamic: &default_w4a8_dynamic
  act: { scope: "per_token", dtype: "int8", symmetric: true, method: "minmax" }
  weight: { scope: "per_channel", dtype: "int4", symmetric: true, method: "ssz" }

default_w8a16: &default_w8a16
  weight: { scope: "per_channel", dtype: "int8", symmetric: true, method: "minmax" }

spec:
  process:
    - type: "flex_smooth_quant"
      enable_subgraph_type: ['norm-linear']
      include:
        - '*language_model*'         # suppress outliers only in LM backbone

    - type: "group"
      configs:
        # Vision projection — light quantization
        - type: "linear_quant"
          qconfig: *default_w8a16
          include: ["*mm_projector*", "*visual_projection*"]

        # LM attention — W8A8 for accuracy
        - type: "linear_quant"
          qconfig: *default_w8a8_dynamic
          include: ["*language_model*self_attn*"]

        # LM MLP — W4A8 for memory savings
        - type: "linear_quant"
          qconfig: *default_w4a8_dynamic
          include: ["*language_model*mlp*"]
          exclude: ["*gate"]

  # VLM requires multimodal calibration dataset
  dataset: /path/to/multimodal_calib.jsonl

  save:
    - type: "ascendv1_saver"
      part_file_size: 4
```

**VLM notes**:

- Vision encoder body (`*visual_model*`, `*image_encoder*`) should be fully excluded — do not add to any `include`
- Built-in datasets (mix_calib, qwen3_cot, etc.) are text-only and cannot calibrate vision components
- Calibration dataset format: JSONL with `messages` field in standard chat schema, images as base64 data URIs
- 64–256 samples covering the modality mix of the intended workload is sufficient

______________________________________________________________________

## From Requirements to YAML (Decision Path)

The decision path for choosing parameters:

```
Model, target bit-width, hardware constraints
        ↓
1. Choose outlier suppression (needed?)
   W8A8 → iter_smooth(alpha=0.5)     ← general default
   W4A8 → iter_smooth(alpha=0.7)     ← stronger migration
   W4A16 → awq                        ← weight protection
   No activation quantization → skip outlier suppression
        ↓
2. Choose weight quantization method
   INT8 → minmax                      ← RTN sufficient
   INT4 → ssz (default) / autoround (peak accuracy)
        ↓
3. Choose activation quantization scope
   Throughput-first → per_tensor (static)
   Accuracy-first → per_token (dynamic)
   Balanced → pd_mix
        ↓
4. Determine include/exclude scope
   - Embedding / LM Head → always exclude
   - o_proj / down_proj → suspicious, analyze first
   - MoE experts → full quantization
        ↓
5. (Optional) Run msmodelslim analyze to verify sensitive layers
   Add excludes based on results
        ↓
6. Quantize → evaluate → iterate
```

______________________________________________________________________

## Common qconfig Patterns (Quick Copy)

```yaml
# W8A8 Static (throughput-first, e.g. attention layers)
act:    {scope: "per_tensor",  dtype: "int8", symmetric: false, method: "minmax"}
weight: {scope: "per_channel", dtype: "int8", symmetric: true,  method: "minmax"}

# W8A8 Dynamic (accuracy-first, standard for LLMs)
act:    {scope: "per_token",   dtype: "int8", symmetric: true,  method: "minmax"}
weight: {scope: "per_channel", dtype: "int8", symmetric: true,  method: "minmax"}

# W4A8 per_channel+ssz (standard for MoE experts)
act:    {scope: "per_token",   dtype: "int8", symmetric: true,  method: "minmax"}
weight: {scope: "per_channel", dtype: "int4", symmetric: true,  method: "ssz"}

# W4A8 per_group+autoround (W4A4 only, or when per_group explicitly needed)
act:    {scope: "per_token",   dtype: "int8", symmetric: true,  method: "minmax"}
weight: {scope: "per_group",   dtype: "int4", symmetric: true,  method: "autoround", ext: {group_size: 256}}

# W8A8 pd_mix (only with KV cache quantization / w8a8c8)
act:    {scope: "pd_mix",      dtype: "int8", symmetric: false, method: "minmax"}
weight: {scope: "per_channel", dtype: "int8", symmetric: true,  method: "minmax"}

# W8A16 (weight-only, activation kept FP16)
weight: {scope: "per_channel", dtype: "int8", symmetric: true,  method: "minmax"}

# W4A16 (weight-only, consumer GPU style)
weight: {scope: "per_group",   dtype: "int4", symmetric: true,  method: "ssz", ext: {group_size: 128}}
```

______________________________________________________________________

## Common Errors & Troubleshooting

| error | cause | fix |
|-------|-------|-----|
| Garbled output after quantization | Insufficient outlier suppression | Switch to `iter_smooth`(alpha=0.7~0.9) or `flex_smooth_quant` |
| Accuracy OK but slow | Activation uses `per_token` | Switch to `per_tensor` (static) or `pd_mix` (hybrid) |
| Layer quantized twice in `group` | Multiple sub-configs match same layer | Check include/exclude ordering; ensure single match per layer |
| `exclude` has no effect | Wildcard doesn't match actual module name | Verify with `model.named_modules()` |
| GPTQ shape error | `group_size` or `block_size` doesn't divide layer dims | Check `input_features`/`output_features`, use 128/256 |
| AutoRound accuracy worse than baseline | No outlier suppression before it | Add `quarot` or `iter_smooth` before `autoround_quant` |
| `msmodelslim analyze` mid-run error | Model adapter doesn't implement analysis interface | Use adapter-free metrics (kurtosis/std/quantile/mse_layer_wise) |
| Sensitivity analysis results inconsistent | Calibration dataset randomness | Fix calibration set and `--device` parameter, re-run to confirm |
| VLM vision quality poor after quantization | Text-only calibration dataset used | Supply multimodal calibration dataset with base64 image URIs |
