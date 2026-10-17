# NPU Kernel Launch Failures (DEBUG_GUIDEBOOK.md §9.20)

*Verified lessons from delivered operators: `scale_add`, `softmax`, `relu`, `gelu`, `sigmoid` (fwd+bwd), `rmsnorm`, `matmul` (2D GEMM — first cube-pipe), and `attention` (non-causal SDPA — first mixed cube+vec, verified at T up to 4096). Each issue below cost at least one failed Stage check cycle. Last update: 2026-04-22.*

Related reference material:
- Kernel code format and per-file invariants: skill `pypto-op-develop`'s `references/pypto-kernel-design-format.md`
- Pipe-class conventions (vector vs cube vs mixed): skill `pypto-op-design`'s `references/quick_ref.md`

## Issue: `Errcode: FFFFFF! launch aicpu failed: 107003`

**Symptom:** Kernel compiles but launch fails with opaque FFFFFF error.

**Root Causes (check in order):**

1. **`stitch_function_*` JIT options on simple vector ops.** These corrupt the stream context. For simple ops, use ONLY `runtime_options={"run_mode": pypto.RunMode.NPU}`.

2. **Tile shapes don't divide tensor dimensions.** `pypto.set_vec_tile_shapes(8, 8, 8, 8)` fails when B=1 or H=4. Every tile value must divide the corresponding dimension.

3. **Missing `torch.npu.set_device(device_id)`.** Just `.to("npu:8")` doesn't activate the device. Call `torch.npu.set_device(id)` at module load time.

## Issue: `RuntimeError: Non-tensor parameter must not be a torch.Tensor`

**Root Cause:** JIT function has no `pypto.Tensor` annotations — JIT classifies all params as non-tensor.

**Fix:** Add `pypto.Tensor([], pypto.DT_FP32)` annotations. See `jit-signature.md`.

## Issue: `ValueError: Return annotation is not allowed`

**Root Cause:** `-> None` on JIT function signature.

**Fix:** Remove `-> None` from JIT-decorated functions.

## Issue: `TypeError: Tensor is not iterable` during JIT recording

**Root Cause:** `B, T, H, K = input_tensor.shape` inside JIT — the shape proxy is not iterable.

**Fix:** Extract shape on the host side, pass as concrete `int` params to the JIT function.

## Issue: `max_diff=0.0` but kernel didn't actually run

**Root Cause:** Try/except block catches JIT failure and falls back to golden, producing golden-vs-golden comparison.

**Fix:** Remove all golden fallback paths. Let JIT failures crash visibly. Real NPU diffs are small but non-zero (e.g. 5.96e-08 for softmax).

## Issue: SIM mode shows precision failure but NPU works fine

**Root Cause:** SIM mode cannot validate precision — only structure. SIM mode memory operations (view/assemble) may produce incorrect data addressing.

**Fix:** Use SIM mode only for structural validation (kernel compiles, no crashes). Always validate precision on real NPU hardware. See `sim-mode.md`.

## Proven minimal kernel pattern for simple vector ops

```python
import os, torch, pypto, torch_npu

DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
torch.npu.set_device(DEVICE_ID)
DEVICE = f"npu:{DEVICE_ID}"

def _impl(inp, out, B_t, T_t, H_t, K_t):
    pypto.set_vec_tile_shapes(B_t, T_t, H_t, K_t)
    for _ in pypto.loop(1):  # required by layout check CI
        out[:] = pypto.some_op(inp, ...)

def jit_entry(
    inp: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
    B_t: int, T_t: int, H_t: int, K_t: int,
):  # NO -> None
    _impl(inp, out, B_t, T_t, H_t, K_t)

kernel = pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)(jit_entry)

def host_wrapper(x):
    B, T, H, K = x.shape
    x_npu = x.to(DEVICE)
    y_npu = torch.empty_like(x_npu)
    kernel(x_npu, y_npu, 1, min(T,16), min(H,4), min(K,16))
    return y_npu.cpu()
```
