# Sensitive Layer Analysis

> **Purpose**: Identify layers that degrade under quantization, so they can be excluded or promoted to higher bit-width. Run this **after** evaluation shows accuracy drops — not proactively.

## Workflow Position

```
Quantize → Serve → Evaluate
                      │
                 ┌────┴────┐
                 │ PASS     │ FAIL
                 ▼          ▼
               Done    Run analysis → add excludes → re-quantize → re-evaluate
```

## Running the Analysis

```bash
msmodelslim analyze \
    --model_path ${MODEL_PATH} \
    --device npu \
    --model_type <ModelName> \
    --metrics <METRIC> \
    --topk 20 \
    --trust_remote_code True 2>&1 | tee analyze_<model>.log
```

## Metric Selection

| metric | what it measures | when to use |
|--------|-----------------|-------------|
| `kurtosis` | Tail heaviness of activation distribution | Default — identifies outlier-heavy layers |
| `std` | Activation range / standard deviation | Layers with unusually wide activation spread |
| `quantile` | Activation value at given percentile | Finding layers with extreme outlier values |
| `mse_layer_wise` | MSE between FP16 and quantized output per layer | Most direct measure of quantization damage |

- **Start with `kurtosis`** — it's fast, doesn't need adapter support, and correlates well with quantization sensitivity
- **Use `mse_layer_wise`** when you need the most direct signal — but it may require model adapter support
- If a metric fails with "model adapter not implemented", try `kurtosis`, `std`, or `quantile` which are adapter-free

## Interpreting Output

The analysis ranks layers by sensitivity score. The top-k layers are the most problematic.

**Typical patterns**:

- First/last 3–5 transformer layers frequently appear in top-k
- `o_proj` and `down_proj` often show higher sensitivity than other linear layers
- Gate/router layers may appear but should not be excluded (they're tiny — quantizing them saves negligible memory)

## Acting on Results

Add sensitive layers to `exclude` in the quantization YAML, or create a separate qconfig entry to bump them to higher bit-width:

```yaml
# Option A: Exclude sensitive layers entirely (keep FP16)
- type: "linear_quant"
  qconfig: *default_w4a8
  include: ["*"]
  exclude:
    - "model.layers.0.*"
    - "model.layers.1.*"
    - "model.layers.58.*"
    - "model.layers.59.*"
    - "*.down_proj"

# Option B: Bump sensitive layers to W8A8 instead of W4A8
- type: "linear_quant"
  qconfig: *default_w8a8
  include:
    - "model.layers.{0,1,2,3,4}.*"
    - "model.layers.{55,56,57,58,59}.*"
    - "*.down_proj"
```

Option B (bumping to higher bit-width) usually recovers more accuracy than full exclusion, while still saving memory vs FP16.

## VLM Considerations

When analyzing a Vision-Language Model, pass the **same multimodal calibration dataset** used during quantization:

```bash
msmodelslim analyze \
    --model_path ${MODEL_PATH} \
    --device npu \
    --model_type <VLM_ModelName> \
    --metrics kurtosis \
    --topk 20 \
    --calib_dataset /path/to/multimodal_calib.jsonl \
    --trust_remote_code True 2>&1 | tee analyze_<model>.log
```

Using a text-only dataset for VLM analysis produces artificially uniform sensitivity scores for vision-adjacent layers, making the output unreliable.

## Iteration

After adding excludes, re-run the full E2E loop (quantize → serve → evaluate). If accuracy still doesn't meet the threshold, consider:

1. Adding more layers to exclude (increase top-k)
1. Switching outlier suppression: `iter_smooth` → `flex_smooth_quant`, or adding `quarot`
1. Raising bit-width for the most sensitive layers (W4A8 → W8A8)
1. Changing weight method: `ssz` → `autoround` (slower but more accurate)

If accuracy still fails after multiple iterations, stop and discuss with the user — the target dtype may not be viable for this model.
