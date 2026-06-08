# Troubleshooting Decision Tree

## Table of contents
- Decision tree
- Known op-level failures (catalog)
- Delivery fallback order
- How to localize a graph-mode numerical bug

## Decision tree

```
torch.compile(model, backend=npu_backend) FAILS or errors at runtime
│
├─ ImportError / the backend has no effect (runs but no graph)
│    → fix import order (torch_npu BEFORE torchair); source set_env.sh;
│      verify torch↔torch_npu minor versions match.
│
├─ Dynamo TorchRuntimeError during compile   (most common)
│    │  Read the "from user code" frame — it names the offending op + the exact module line.
│    ├─ "NPU contiguous operator only supported contiguous memory format"
│    │   / upsample_bicubic2d / interpolate    (e.g. ViT interpolate_pos_encoding)
│    │     → run at the model's NATIVE resolution so position-embedding interpolation is skipped,
│    │       OR pass interpolate_pos_encoding=False,
│    │       OR keep that submodule eager (wrap with torch.compiler.disable),
│    │       OR fall back to npu_eager for this model.
│    ├─ unsupported op / aten fallback
│    │     → try a different config.mode; mark/report the op to CANN; use npu_eager meanwhile.
│    └─ dynamic-shape recompiles / guard cache blowup
│          → pass dynamic=False; fix input shapes; raise torch._dynamo.config.cache_size_limit.
│
├─ Compiles but OOM (HBM)
│    → smaller batch; fp16 instead of fp32; pick a less-loaded NPU (npu-smi info);
│      torch.npu.empty_cache() between runs.
│
└─ Compiles & runs but PRECISION looks wrong
     → compare torchair vs npu_eager FIRST (same device + dtype):
       ├─ they MATCH (cosine ≈ 1)  → divergence is fp16↔fp32 rounding, not a graph bug
       │                              ⇒ accept PASS_COSINE (see methodology.md).
       └─ they DIFFER               → genuine graph-mode numerical bug:
              bisect by comparing intermediate hidden_states; disable fusion on the
              suspect submodule; re-run that submodule in eager to localize the op;
              file a minimal repro.
```

## Known op-level failures (catalog)

| Symptom in traceback | Root cause | Fix |
|---|---|---|
| `NPU contiguous operator only supported contiguous memory format` after `upsample_bicubic2d` / `interpolate` | bicubic `interpolate` in ViT `interpolate_pos_encoding` decomposes to a `contiguous(memory_format=...)` NPU does not support in graph | run at native resolution / `interpolate_pos_encoding=False` / keep submodule eager / use npu_eager |
| backend imported but graph never builds | wrong import order or missing `set_env.sh` | fix import order; source CANN env |
| repeated recompilation, slow steady state | dynamic shapes | `dynamic=False`, fixed input shapes |

Models using **RoPE** position embeddings (instead of interpolated learned
position embeddings) typically avoid the interpolate/contiguous failure and
compile cleanly.

## Delivery fallback order

Always degrade gracefully and **capture the full compile traceback** to an
`.err.txt` so the failing op is recoverable later:

```
torchair graph  →  (on compile failure) npu_eager  →  (still off) cpu fp32
```

`benchmark.py` already implements this: a backend that fails is recorded as a
`failed` row with the truncated error, and the other backends still run.

## How to localize a graph-mode numerical bug

1. Set `TORCHDYNAMO_VERBOSE=1` (and optionally `TORCH_LOGS="+dynamo"`) and re-run to get the internal stack.
2. Add `output_hidden_states=True` and compare per-layer hidden states (torchair vs npu_eager) to find the first diverging layer.
3. Wrap the suspect submodule with `torch.compiler.disable` to confirm it is the culprit.
4. Reduce to a minimal module + input that reproduces, then report to the CANN/torch_npu team.
