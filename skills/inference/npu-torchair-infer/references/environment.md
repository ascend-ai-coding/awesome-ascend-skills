# Environment, Import Order & Compatibility

## Table of contents
- The one non-negotiable rule: import order
- Compatibility matrix
- Pre-flight checklist
- CompilerConfig knobs

## The one non-negotiable rule: import order

Graph mode silently degrades to eager (or errors) if `torch_npu` is not imported
**before** `torchair`. `torch_npu` registers the `npu` device/backend that
torchair depends on.

```python
import torch
import torch_npu        # MUST come before torchair
import torchair
from torchair import patch_for_hcom
patch_for_hcom()        # required only if the model has HCCL collectives; harmless otherwise

config = torchair.CompilerConfig()
# config.mode = "reduce-overhead"          # optional; guard with hasattr(config, "mode")
npu_backend = torchair.get_npu_backend(compiler_config=config)
model = torch.compile(model, backend=npu_backend, dynamic=False)
```

`dynamic=False` is recommended for fixed-shape encoders: it avoids dynamic-shape
recompilation and produces the most stable graph. Use dynamic shapes only when
the input shape genuinely varies per call.

## Compatibility matrix

`benchmark.py` records all of these automatically into `meta.env` of the JSON.
Known-good baseline this skill was authored and validated against:

| Component | Known-good | How to check |
|---|---|---|
| CANN toolkit | `8.5.1` | `cat $ASCEND_TOOLKIT_HOME/<arch>-linux/ascend_toolkit_install.info` |
| Python | `3.11` | `python --version` |
| torch | `2.9.0` (cpu wheel + torch_npu) | `python -c "import torch;print(torch.__version__)"` |
| torch_npu | `2.9.0.post1` | `python -c "import torch_npu;print(torch_npu.__version__)"` |
| torchair | bundled **inside** torch_npu (no standalone `__version__`) | `python -c "import torch_npu,torchair;print(hasattr(torchair,'get_npu_backend'))"` |
| transformers | `4.57+` (must contain the model class) | `python -c "import transformers;print(transformers.__version__)"` |
| device | 8× `Ascend910B3` healthy | `npu-smi info` |

Hard rules:
- torch and torch_npu **minor versions must match** (e.g. torch `2.9.x` ↔ torch_npu `2.9.x.postN`).
- torchair ships with torch_npu — never `pip install torchair` separately.
- `source $ASCEND_HOME/ascend-toolkit/set_env.sh` first, or torchair cannot find ATC/ACL and compile fails.

## Pre-flight checklist

- [ ] `source set_env.sh` done; `echo $ASCEND_TOOLKIT_HOME` is non-empty.
- [ ] `import torch_npu` then `import torchair` both succeed (in that order).
- [ ] `torch.npu.is_available()` is `True`; pick an NPU with enough free HBM (`npu-smi info`).
- [ ] `transformers` actually contains the model class (`AutoModel` resolves it, or `hasattr(transformers, "<Class>")`).
- [ ] Weights present locally; gated repos need an HF token, else build a config-init stand-in (see methodology.md) to validate mechanics.
- [ ] dtype chosen: **fp16** for deployment-representative perf; **fp32** for strict (<1e-3) accuracy parity.

## CompilerConfig knobs

`torchair.CompilerConfig()` is the entry point. Common adjustments:
- `config.mode` — optional execution mode (e.g. `"reduce-overhead"`); set only if `hasattr(config, "mode")`.
- Keep defaults for the first attempt; only tune after a clean baseline compiles.

On a busy/shared NPU (co-tenant workloads), expect higher latency variance and
tighter free HBM — prefer fp16, small batch, and report p50.
