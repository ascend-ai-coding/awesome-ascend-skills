# Phase 5 — Compile, install, test, debug

Build the project, install the wheel, generate a basic test, run it, and debug failures.

Precondition: op_host + op_kernel generated; `ops.h`, `register.cpp`,
`csrc/CMakeLists.txt` updated.

## Step 0 — Environment (every shell command)

```bash
source ${CANN_PATH}/*/set_env.sh
conda activate <env_name>
```

> Re-source before **every** command. Missing env → no AscendC compiler, wrong python,
> or `torch_npu` import error.

## Step 1 — Compile

```bash
cd ascend-kernel
chmod +x build.sh        # template copies may drop the exec bit
bash build.sh
ls ./output/ascend_kernel*.whl   # success if a .whl exists
```

Failure → enter the debug loop (max 3 attempts).

## Step 2 — Install

```bash
pip install output/ascend_kernel*.whl --force-reinstall --no-deps
```

## Step 3 — Generate the basic test

If `tests/test_<op>.py` already exists, skip. Otherwise create it (functional + basic
precision). Skeleton:

```python
import torch, torch_npu, pytest
try:
    import ascend_kernel
except ImportError:
    import os, glob
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    libs = glob.glob(os.path.join(root, "python/ascend_kernel/ascend_kernel/lib/*.so"))
    if libs: torch.ops.load_library(libs[0])
    else: raise ImportError("ascend_kernel library not found")

def is_npu():
    try: return torch.npu.is_available()
    except Exception: return False

class Test<OpName>:
    @pytest.mark.parametrize("shape", [ ... ])     # typical shapes
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_basic(self, shape, dtype):
        if not is_npu(): pytest.skip("NPU not available")
        dev = torch.device("npu:0")
        x = torch.randn(shape, dtype=dtype, device=dev)
        npu = torch.ops.npu.<op>(x)
        ref = <pytorch_reference>(x.cpu().float()).to(dtype)
        rtol, atol = (1e-3, 1e-3) if dtype != torch.float32 else (1e-5, 1e-5)
        assert torch.allclose(npu.cpu(), ref, rtol=rtol, atol=atol), \
            f"max diff = {(npu.cpu()-ref).abs().max().item()}"
```

Test-data rules: respect the operator domain (`acosh` x>=1, `log` x>0, `sqrt` x>=0);
generate with `torch.rand()*scale+offset`; include boundary cases; fp16/bf16 reference
computed in fp32 then cast back.

## Step 4 — Run tests

```bash
cd ascend-kernel && python tests/test_<op>.py        # functional (must exit 0)
cd ascend-kernel && pytest tests/test_<op>.py -v      # precision (should pass)
```

## Debug loop (max 3 attempts)

```
fail → read log → locate file:line → fix → rebuild/retest
  ↓ still failing
2nd → recheck missing include / declaration / CMakeLists
  ↓ still failing
3rd → deep-check API usage / type match / CMake config
  ↓ still failing
STOP → report detailed error + attempted fixes to the user
```

### Compile-error decision tree

| Symptom | Fix location | Direction |
|---|---|---|
| `undeclared identifier` | `ops.h` / op_host | missing declaration or include |
| `no matching function` | op_host | arg type/order vs kernel entry |
| `undefined reference` / linker | `csrc/CMakeLists.txt` | source not in OP_SRCS / ascendc_library |
| `redefinition` | ops.h / register.cpp | duplicate definition |
| AscendC kernel compile error | op_kernel | unsupported API usage / type |

### Test-error decision tree

| Symptom | Direction |
|---|---|
| `ImportError: ascend_kernel` | wheel not installed / .so missing |
| `RuntimeError: ... not found` | register.cpp name vs call name mismatch |
| `allclose failed` | compute logic / precision → check Compute |
| `shape mismatch` | op_host output shape computation |
| `SIGABRT` / device error | tiling param error or UB out-of-bounds |

On `allclose` failure / large deviation / all-zero / NaN that thresholds cannot fix,
go to precision root-cause debugging: [`07b-precision-debug.md`](07b-precision-debug.md).

## Editable files

`csrc/ops/<op>/op_host/*.cpp`, `op_kernel/*.cpp`, `csrc/ops.h`, `csrc/register.cpp`,
`csrc/CMakeLists.txt`, `tests/test_<op>.py`.

## Gate (must pass before Phase 6)

- [ ] `.whl` built and installed.
- [ ] functional test exits 0.
- [ ] precision pytest green.

## Anti-patterns (NEVER)

- NEVER hardcode/guess CANN or conda paths.
- NEVER modify `cmake/` or `csrc/utils/`.
- NEVER delete other operators' registration during debugging.
- NEVER skip the functional test and go straight to precision.
- NEVER ignore compile warnings (especially type truncation).
- NEVER pass r-values to `EXEC_KERNEL_CMD`.
- NEVER keep retrying past 3 attempts — report to the user.
