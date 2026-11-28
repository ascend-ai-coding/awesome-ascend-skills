# fixpipe Dual-Destination Mode + `relu_pre_mode` Triggers an AICore Exception

| | |
| --- | --- |
| Status | The hardware combination is invalid. Official PyPTO `bca90a6c6e69a1b1111cfe8a43b975ba2bfcf28a` has no matching frontend guard; the current example uses parse-only rejection probes and exposes this missing check without launching the invalid combination. |
| Impact | **A runtime AICore exception**; after the exception, all NPU operations in the same process also fail. |
| Toolchain | Ascend950PR_957b / CANN 9.2.0 / pypto 0.2.1 / torch 2.10.0 |
| Source | Current frontend: `python/pypto_pro/ir/op/block_ops.py::_ir_move`; A5 ISA: `pto/npu/a5/TMov.hpp`, `common.hpp`. The device-fault observation uses the toolchain in the preceding row, not a new launch on the current installation. |
| Reproduction | `examples/basic_api/fixpipe/fixpipe_dual_dst/kernel.py::dual_with_relu`; the current test `test_a_dual_mode_with_relu_is_rejected` checks parsing only. |

## 1. Symptom

```python
pl.move(ub_half, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
        relu_pre_mode=pl.ReluPreMode.NormalRelu)
```

Parsing, compilation, and launch all succeed, but `torch.npu.synchronize()` raises:

```
NPU function error: device error type 3, error code is 507015
  Failed to check task send status, ... reason=aicore exception
```

**The entire process is unusable after the exception**—even a later `torch.zeros(..., device="npu:0")` fails with `aclnnInplaceZero`. Failures after the first AICore fault in that process are cascades and do not independently establish another defective combination.

Control: the same kernel without `relu_pre_mode` (`dual_split_m`) is bitwise correct. The difference is therefore the relu option.

## 2. Neither Layer Rejected It

The **Python side** (`_ir_store_tile` in `block_ops.py`, in the `_dual_modes` block at the time) checked only `scale`:

```python
_dual_modes = {AccToVecMode.DualModeSplitM, AccToVecMode.DualModeSplitN}
if is_quant and acc_to_vec_mode in _dual_modes:
    raise NotSupported("scale cannot be combined with dual-mode acc_to_vec_mode ...")
```

On the **C++ side**, `GetDualDstCtl` (`pto/npu/a5/common.hpp:183`) had only two `static_assert`s in the dual branch:

```cpp
static_assert(quantPre == QuantMode_t::NoQuant, "Quant is not support in dual Dst Mode.");
static_assert(!(!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox),
              "Dual Dst Mode is not support in nz2dn.");
```

**There was no relu check.** `TMovCcToUb` (`TMov.hpp:233`) passed both control fields to the instruction **side by side**, without any cross-check:

```cpp
pto_copy_matrix_cc_to_ub(
    dstAddr, srcData, 0, validCol, validRow, dstStride, srcStride,
    dualDstCtl, subBlockId, 0, unitFlagCtrl,
    quantPre, static_cast<uint8_t>(reluMode), channelSplitEnable, ...);
```

This generated an instruction with `dualDstCtl != 0` and `reluMode != NoRelu`, which the hardware does not accept.

**Note that `unitFlagCtrl` (that is, `phase`) is also passed alongside it without a cross-check.** It has exactly the same shape as the relu case and likely has the same issue. It has not been measured: an unbalanced UF chain for `phase` may **hang** the AI Core (requiring a card reset), which is much more costly than raising an exception—not worth testing for a fact that can already be read from source.

## 3. Inspect the Executing Transfer Path

Acc→Vec move uses `TMovCcToUb`, which receives `reluMode` and passes it through unchanged. The absence of a relu template parameter from the separate `TExtractL0CToUBDualDst` helper does not describe this move path. Check the actual generated call, source/destination types and control fields before inferring whether an option is applied, ignored or rejected.

## 4. Minimal Reproduction

See `examples/basic_api/fixpipe/fixpipe_dual_dst/kernel.py::dual_with_relu`. The key setup is a half-height UB tile `[M/2, N]`, `acc_to_vec_mode=DualModeSplitM`, and `relu_pre_mode=pl.ReluPreMode.NormalRelu`; everything else is identical to `dual_split_m`.

Routine regression must not launch this known-invalid combination. A dedicated device-fault investigation needs a separate process and the recovery checks from [board-run fault isolation](../../verification/board-runs.md#isolating-device-faults).

## 5. Workarounds

Do not use relu in dual mode. If relu is needed:

1. Use `SingleModeVec0` / `SingleModeVec1` (the single destination supports relu; the `fixpipe_riders` example verified relu itself on `store`); or
2. Apply relu after dual mode on the vector side using `vf.max(x, 0)`.

## 6. Fix Direction

Ordered by impact, **first prevent generation of an invalid instruction**:

1. **Reject it on the Python side**: a relu/dual cross-check belongs beside the scale/dual check in `_ir_move`. This guard is absent from the inspected official commit. Keep `phase` separate because its supported conditions are not identical to relu.
2. **Add a C++ `static_assert`** (the complete fix): `GetDualDstCtl` is already the natural location—it receives `quantPre`, and can reject the combination at compile time by also receiving `reluMode`.
3. Document the relu exclusion and the separate phase restrictions in the official `AccToVecMode.md`.

With both items 1 and 2, Python gives a clear error and C++ protects callers that invoke the ISA directly.
