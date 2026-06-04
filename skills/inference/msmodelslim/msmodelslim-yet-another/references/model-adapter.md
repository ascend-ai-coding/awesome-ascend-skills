# Adding a New Model Adapter

> **Purpose**: Register a new model (or a new dtype for an existing model) under `third-party/msmodelslim/` when no `lab_practice` config exists.

## Directory Layout

```
third-party/msmodelslim/
└── <model_name_or_family>/
    ├── <purpose>_w4a8.yaml          # one YAML per target dtype
    ├── <purpose>_w8a8.yaml
    ├── <purpose>_w4a4.yaml
    └── README.md                    # optional: HF repo link, notes
```

- One YAML per target dtype
- File names follow `<purpose>_<dtype>.yaml` — same convention as `lab_practice`
- For a variant of an existing model family (e.g., Qwen3-14B from Qwen3-32B), copy the closest existing YAML as a starting point

## Decision Flow

```
Need to quantize a model with no lab_practice config?
        │
        ├─ Sibling model in same family exists?
        │   └─ YES → copy sibling YAML, update metadata.config_id and verified_model_types
        │
        ├─ VLM (Vision-Language Model)?
        │   └─ YES → use VLM template + prepare multimodal calibration dataset
        │
        └─ Otherwise → write fresh YAML from the templates in the config guide
```

## Minimum Required YAML Fields

```yaml
apiversion: modelslim_v1
metadata:
  config_id: <model_name>_<dtype>    # unique slug, no spaces
  score: 0                            # 0 = unverified; update after evaluation
  verified_model_types:
    - <ModelName>                     # exact transformers model_type string
  label:
    w_bit: <4|8>
    a_bit: <4|8|16>
    is_sparse: False
    kv_cache: False

# Inline qconfig anchors
w8a8: &w8a8
  act: { scope: "per_token", dtype: "int8", symmetric: true, method: "minmax" }
  weight: { scope: "per_channel", dtype: "int8", symmetric: true, method: "minmax" }

w4a8: &w4a8
  act: { scope: "per_token", dtype: "int8", symmetric: true, method: "minmax" }
  weight: { scope: "per_channel", dtype: "int4", symmetric: true, method: "ssz" }

spec:
  process:
    - type: "flex_smooth_quant"
      enable_subgraph_type: ["norm-linear"]
      include: ["*"]

    - type: "group"
      configs:
        - type: "linear_quant"
          qconfig: *w8a8
          include: ["*self_attn*"]

        - type: "linear_quant"
          qconfig: *w8a8
          include: ["*mlp*"]
          exclude: ["*gate"]

        - type: "linear_quant"
          qconfig: *w4a8
          include: ["*mlp.experts*"]  # MoE only; omit for dense models

  save:
    - type: "ascendv1_saver"
      part_file_size: 4
```

**Required metadata fields**:

| field | value |
|-------|-------|
| `metadata.config_id` | `<model_name>_<dtype>` — unique slug, no spaces |
| `metadata.verified_model_types` | Exact `<ModelName>` string as passed to `--model_type` |
| `metadata.label.w_bit / a_bit` | Match the dominant `dtype` in the qconfig group |
| `dataset` | VLM only: absolute path to multimodal JSONL. Omit for text-only models |

## Finding the Model Type String

If unsure of the exact `model_type` string:

```bash
python -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained('<HF_REPO_OR_LOCAL_PATH>', trust_remote_code=True); print(c.model_type)"
```

## VLM Adapter Checklist

If the new model is a Vision-Language Model:

- [ ] Prepare multimodal calibration dataset (64–256 samples, base64 image URIs in chat `messages` format)
- [ ] Add `dataset: /absolute/path/to/multimodal_calib.jsonl` under `spec:`
- [ ] Exclude vision encoder body: `*visual_model*`, `*image_encoder*` — do not add to any `include`
- [ ] Light-quantize vision projection: `*mm_projector*`, `*visual_projection*` → W8A16 or keep FP16
- [ ] Run the curl probe to verify the served model accepts image inputs before evaluating
- [ ] Pass `--calib_dataset /path/to/multimodal_calib.jsonl` to `msmodelslim analyze` if accuracy recovery is needed

## Validation After Adding

1. **Parse check**: `python -c "import yaml; yaml.safe_load(open('path/to.yaml'))"`
1. **Dry-run quantization** with a short calibration subset to catch config errors early
1. **Full E2E**: quantize → serve → evaluate per the main workflow
1. **Update score**: after successful evaluation, update `metadata.score` with the accuracy result
