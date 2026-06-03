---
name: npu-torchair-infer
description: Migrate any HuggingFace model to Ascend NPU torchair graph mode (torch.compile) and benchmark it for accuracy and performance against NPU eager and CPU eager. Use when running, compiling, or benchmarking HF models (vision, text, image-text encoders such as SigLIP2, DINOv3, ViT, CLIP, Qwen-VL, SAM) on Ascend 910B/CANN with torch_npu and torchair; when a torch.compile graph-mode run on NPU fails (Dynamo TorchRuntimeError, unsupported op, interpolate/contiguous errors); or when comparing torchair vs npu_eager vs cpu with cosine similarity, max abs diff, and p50/p95/p99 latency.
---

# NPU TorchAir Inference & Migration (主流程)

## Overview

Bring an arbitrary HuggingFace model up on **Ascend NPU torchair graph mode**,
verify it numerically against CPU/NPU-eager, and measure its speedup. The bundled
scripts are model-agnostic (config-driven inputs, recursive output comparison),
so the same flow applies to the next model with only a changed
`--model_name_or_path`.

Bundled assets:
- `scripts/torch_air_infer.py` — minimal single-backend inference + latency stats.
- `scripts/benchmark.py` — 3-backend (torchair / npu_eager / cpu) accuracy + perf → JSON + Markdown.
- `scripts/run_benchmark.sh` — generic driver: prepare model → sanity infer → benchmark.

## The one rule you cannot break

`torch_npu` MUST be imported **before** `torchair`, or graph mode does not engage.

```python
import torch
import torch_npu        # before torchair
import torchair
config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)
model = torch.compile(model, backend=npu_backend, dynamic=False)
```

Details, version matrix, and CompilerConfig knobs: see `references/environment.md`.

## Main flow (run in order)

```bash
source $ASCEND_HOME/ascend-toolkit/set_env.sh   # CANN runtime first
export HF_ENDPOINT=https://hf-mirror.com         # China-friendly mirror

# 0) Sanity on CPU (fast) — confirms inputs/outputs resolve for this model
python scripts/torch_air_infer.py --model_name_or_path <path-or-id> --device cpu --backend eager

# 1) NPU eager — confirms the model runs on NPU at all
python scripts/torch_air_infer.py --model_name_or_path <path-or-id> --device npu --backend eager --dtype float16

# 2) NPU torchair graph mode — the migration target
python scripts/torch_air_infer.py --model_name_or_path <path-or-id> --device npu --backend torchair --dtype float16

# 3) Full 3-backend accuracy + performance comparison (writes JSON + Markdown)
python scripts/benchmark.py --model_name_or_path <path-or-id> \
    --backends torchair,npu_eager,cpu --device npu \
    --dtype float16 --batch_size 1 --warmup 10 --iterations 100 \
    --output_dir benchmark_results/<model>

# strict <1e-3 parity check (fits-in-memory models): re-run step 3 with --dtype float32
```

Or do steps 2+3 in one shot: `bash scripts/run_benchmark.sh <path-or-id> [out_subdir]`.

## Decision tree

```
Step 0 (CPU) fails to build inputs?  → model needs a custom input builder (see references/methodology.md, BaseModelLoader.build_inputs).
Step 1 (NPU eager) fails?            → environment problem: re-check import order + set_env.sh (references/environment.md).
Step 2 (torchair) fails to compile?  → walk references/troubleshooting.md; fall back to npu_eager and keep going.
Step 3 accuracy not within gate?     → compare torchair vs npu_eager first (references/methodology.md): same ⇒ fp16 rounding (PASS_COSINE); differ ⇒ graph bug.
Weights gated / undownloadable?      → build a config-init random stand-in (references/methodology.md); valid for backend comparison, not task accuracy.
```

## Accuracy & performance gates (summary)

- Golden = CPU fp32. Metrics: cosine sim, max abs diff, max rel err.
- `PASS` = `max_abs_diff ≤ 1e-3` and `cosine ≥ 0.999`; `PASS_COSINE` = cosine passes (expected in fp16); else `FAIL`.
- Perf: first iteration = compile cost (reported separately); steady p50/p95/p99 over ≥100 iters with `torch.npu.synchronize()` before each timer stop; report throughput + peak HBM.

Full methodology, the model interface, gated-weights handling, and expected
output: `references/methodology.md`.

## References (load as needed)

- `references/environment.md` — import order, CANN/torch_npu/torchair/transformers compatibility matrix, pre-flight checklist, CompilerConfig.
- `references/troubleshooting.md` — compile-failure decision tree, known op failures (interpolate/contiguous), fallback order, bug localization.
- `references/methodology.md` — `BaseModelLoader` interface, input synthesis, accuracy/perf methodology, config-init stand-ins, expected output.

## Reuse checklist for the next model

1. Step 0 on CPU — inputs/outputs resolve (fast).
2. Step 1 NPU eager works.
3. Step 2 torchair — compiles ⇒ done; else walk `references/troubleshooting.md`.
4. Step 3 benchmark — gate on cosine (fp16) or `<1e-3` (fp32).
5. If inputs could not be inferred, add a small `build_inputs` override (`references/methodology.md`).
