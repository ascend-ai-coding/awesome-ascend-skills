# Accuracy & Performance Methodology + Model Interface

## Table of contents
- Pluggable model interface (BaseModelLoader)
- Generic input synthesis (how the scripts infer inputs)
- Accuracy methodology
- Performance methodology
- Gated / unavailable weights: config-init stand-ins
- Expected output shapes

## Pluggable model interface (BaseModelLoader)

The scripts are **config-driven** so no model name is hard-coded. The contract an
adapter must satisfy (the default implementation lives inline in the scripts):

```python
class BaseModelLoader:
    def load_model(self, path, dtype, device) -> torch.nn.Module:
        """eval()-ed module on `device` with `dtype` weights.
        Default: transformers.AutoModel.from_pretrained(path, torch_dtype=dtype)."""

    def build_inputs(self, config, batch_size, device, dtype) -> dict:
        """kwargs consumed by model(**inputs). Default inspects config:
          vision_config OR (image_size & patch_size) -> pixel_values [B,C,H,W]
          text_config   OR vocab_size                -> input_ids/attention_mask [B,L]
        Override only for exotic signatures (NaFlex pixel_attention_mask+spatial_shapes,
        decoder_input_ids, etc.) or to feed real data via AutoProcessor."""

    def extract_outputs(self, output) -> dict:
        """Named float tensors to compare. Default recursively flattens
        dict/list/ModelOutput; comparator auto-picks a primary key
        (last_hidden_state > logits > pooler_output)."""
```

To support a new model you usually change **nothing**. Override `build_inputs`
only when the forward needs inputs beyond `pixel_values` / `input_ids`.

## Generic input synthesis (how the scripts infer inputs)

`detect_modalities(config)` decides which dummy inputs to create:
- `vision_config` present, or top-level `image_size` + `patch_size` ⇒ vision ⇒ `pixel_values`.
- `text_config` present, or `vocab_size` on a non-vision model ⇒ text ⇒ `input_ids` + `attention_mask`.
- image-text models (both sub-configs) get both.

Inputs are seeded once; every backend `clone()`s + casts the *same* data so the
comparison is apples-to-apples. If inference needs real data, swap the synthetic
builder for `transformers.AutoProcessor`.

## Accuracy methodology

- CPU **fp32** output is the golden reference.
- Metrics per output tensor: **cosine similarity**, **max absolute diff**, **max relative error**.
- Recursively reduce dict/list outputs; compare a primary tensor (`last_hidden_state` / `logits` / `pooler_output`) plus every common key.
- Status tiers:
  - `PASS` — `max_abs_diff ≤ atol` (default `1e-3`) **and** `cosine ≥ threshold` (default `0.999`).
  - `PASS_COSINE` — cosine passes but abs diff exceeds atol. **Expected for fp16**, and for embeddings with near-zero entries (relative error explodes while direction is identical).
  - `FAIL` — neither.
- Always add the **torchair-vs-npu_eager** cross-check (same device + dtype): it isolates *graph-vs-eager* divergence from *fp16-vs-fp32* rounding.

Rule of thumb (validated on Ascend 910B3):
- fp32 NPU↔CPU: `max_abs_diff ~1e-4..1e-3` ⇒ `PASS`.
- fp16 NPU↔CPU: `~1e-2` abs but `cosine ≥ 0.9999` ⇒ `PASS_COSINE` (functionally equal). Projection heads / logits often still pass strict `<1e-3` in fp16.

## Performance methodology

- Warmup ≥ 10 iterations; **the first iteration is the compile/build cost** in graph mode — record it separately, never fold it into steady-state stats.
- Time ≥ 100 steady iterations; **call `torch.npu.synchronize()` before stopping each timer** (NPU is async — without sync you measure launch latency, not compute).
- Report `first_iter_ms`, `p50/p95/p99`, throughput (`batch×1000/p50`), and peak HBM (`torch.npu.max_memory_allocated()`).
- On shared NPUs, latency variance rises — prefer p50 and note co-tenant load.

## Gated / unavailable weights: config-init stand-ins

When real weights are gated (no HF token) or undownloadable, build a
**random-init** model from the real architecture/config:

```python
# generic (architecture inferred from a real config)
from transformers import AutoConfig, AutoModel
m = AutoModel.from_config(AutoConfig.from_pretrained(local_config_dir))

# or an explicit architecture when only the class is known
from transformers import DINOv3ViTModel, DINOv3ViTConfig
m = DINOv3ViTModel(DINOv3ViTConfig())   # defaults == ViT-S/16
m.save_pretrained(out_dir)
```

This is **valid** for CPU-vs-NPU / graph-vs-eager numerical comparison and for
exercising the compile path (identical weights on both sides). It is **not**
valid for downstream task accuracy. Always state in the report when a stand-in
was used.

## Expected output shapes

```
input[pixel_values  ] shape=(B, 3, H, W) dtype=<compute dtype>
backend mode   : torchair(graph)
output[last_hidden_state] shape=(B, N, D) dtype=<compute dtype>
first iter (compile+run): <tens of seconds>      # one-time graph build
steady latency  : p50=<small ms>  p95=...  p99=...
throughput      : <samples/sec> @ p50
```

`benchmark_results.md` columns: backend · accuracy (cosine / max_abs_diff /
rel_err / status) · perf (first_iter, p50/p95/p99, throughput, peak_mem). A
failed graph compile appears as a `failed` row with the truncated error; other
backends still complete.
