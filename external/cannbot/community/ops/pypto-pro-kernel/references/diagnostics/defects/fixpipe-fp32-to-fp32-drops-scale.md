# fixpipe Silently Drops `scale` on the **L0C→UB** `FP32 → FP32` Path

| | |
| --- | --- |
| Status | The root cause is in the CANN PTO header and is **not being fixed for now** (user decision, 2026-09-27); **pypto now rejects it during parsing**. See “Why protection is at parse time” in [`diagnostics/defects/fixpipe-overview.md`](fixpipe-overview.md). |
| Impact | **Silently returns a wrong answer**: parsing, compilation, and execution succeed, but the output is the unscaled accumulation result. |
| Scope | Only the `pl.move` (Acc→**Vec**) path; the same case in `pl.store` (Acc→**GM**) is correct. |
| Toolchain | Ascend950PR_957b / CANN 9.2.0 / pypto 0.2.1 / torch 2.10.0 |
| Source | `pypto` @ `integration/pro-surface`, `arch35/copy_l0c_to_ub_impl.h` |
| Reproduction | `examples/basic_api/fixpipe/fixpipe_quant_matrix`, `test_fp32_to_fp32_silently_drops_the_scale`; for the per-channel path see `fixpipe_scaling_buffer::test_per_channel_fp32_to_fp32_also_drops_the_scale`. |

## 1. Symptom

`pl.move(ub_tile, acc_tile, scale=s)`, where L0C is `DT_FP32` and UB is also `DT_FP32`:

```python
pl.move(t, ac, scale=0.5)          # expected: 0.5 * acc
```

The output is `acc`, **not `0.5 * acc`**. This is the same for `scale = 0.5 / 2.0 / 0.125`, so the result is not “accidentally equal to 1.”

There is no error or warning.

## 2. It Is Not Simply “Quantization Is Unsupported for Same-Type Copies”

The most natural explanation is “the source and destination types are the same, so it takes a plain copy path.” But measurements from the same batch show:

| src → dst | Legal without scale? | Result with scale |
| --- | --- | --- |
| `FP32 → FP32` | ✓ | **scale is dropped** |
| `FP32 → FP16` | ✓ | `scale * acc` is **bitwise exact** |
| `FP32 → BF16` | ✓ | `scale * acc` is **bitwise exact** |

All three cases are in the non-quantized table (`_A5_MOVE_COMBOS`); only `FP32 → FP32` is broken. **So the rule is not that “all combinations in the non-quantized table drop scale.”**

The other eight destination types (`INT8`, `UINT8`, `HF8`, `FP8E4M3FN`, plus `INT8`, `UINT8`, `FP16`, and `BF16` on the `INT32` side) are **reachable only with scale**, and scale works for all of them.

## 2.2 The Same `store` (Acc→GM) Case Is **Correct**—the Path Is Broken, Not the Combination

Additional measurement on 2026-09-27 (`fixpipe_quant_matrix`, all 22 cases in `_A5_STORE_QUANT_COMBOS`):

```python
pl.store(out, acc, [0, 0], scale=0.5)     # L0C(FP32) → GM(FP32): scale works, bitwise exact
pl.move(ub, acc, scale=0.5)               # L0C(FP32) → UB(FP32): scale is dropped
```

Scale is preserved whether the destination Tensor is `ND` or `NZ`. **The same dtype combination, scale value, and L0C produce the correct result when sent to a different destination.**

The issue is therefore in **L0C→UB dispatch**, not in the dtype combination or scale encoding. This matches the file name `arch35/copy_l0c_to_ub_impl.h`—**it handles only the UB path.**

## 2.5 The Per-Channel Path Also Drops It—Both Scale Encodings Fail

Additional measurement on 2026-09-27 (`fixpipe_scaling_buffer`): replacing the per-tensor scalar `scale` with a **per-channel Scaling Tile** still makes `FP32 → FP32` **drop scale** (the result is bitwise equal to the unscaled `acc`).

This narrows the scope considerably. The two paths use **completely different scale encodings**:

| | How scale reaches hardware |
| --- | --- |
| per-tensor | The `float32` bit pattern is encoded as an immediate in the instruction (`_encode_deq_scalar`). |
| per-channel | A `[1, N] INT64` bit-field table is moved through L1 into FB, where fixpipe reads it. |

**Both paths drop it, so the problem is not how scale is encoded.** In the same batch, per-channel `FP32 → FP16` / `FP32 → INT8` were bitwise exact; `FP32 → FP32` was the only broken case, consistent with the per-tensor result.

Together with §2.2, these two directions narrow the scope to one case: **the destination is UB and its dtype is FP32.**

## 3. Root Cause: The On-Chip Path in the ISA Header Is Missing an `else if`

**The hardware supports this**—A5's QuantMode enum includes `QF322F32_PRE = 27` (scalar) and `VQF322F32_PRE = 38` (per-channel), and the store path uses them (measured in §2.2).

The problem is that **two mapping functions in the PTO ISA header are asymmetric**:

| Function | File | `dst = float` branch |
| --- | --- | --- |
| `GetScalarPreQuantModeGm` | `pto/npu/a5/TStore.hpp:33` | `QF322F32_PRE` ✓ |
| `GetVectorPreQuantModeGm` | `pto/npu/a5/TStore.hpp:63` | `VQF322F32_PRE` ✓ |
| `GetScalarPreQuantMode` | `pto/npu/a5/common.hpp:85` | **No such branch** → falls back to `NoQuant` |
| `GetVectorPreQuantMode` | `pto/npu/a5/common.hpp:112` | **No such branch** → falls back to `NoQuant` |

The two `common.hpp` functions (on-chip paths: Acc→UB and Acc→L1) have this chain:

```cpp
if (IsInt8) … else if (IsHF8) … else if (IsFP16) … else if (IsBF16) … else if (IsFP8E4M3) …
// ← missing: else if (IsFP32) { quantPre = QuantMode_t::QF322F32_PRE; }
```

The `...Gm` versions in `TStore.hpp` contain exactly this branch. The two sides also use different styles (`Gm` uses `caps::IsXxx<T>()`; the on-chip path uses bare `std::is_same`), **suggesting that F32 was added while refactoring the store path and the on-chip path was not updated.**

Thus, for `FP32 → FP32`, `pl.move(ub, acc, scale=s)` receives `quantPre = NoQuant`; the instruction is emitted and the scale field is populated, **but hardware follows `NoQuant` and ignores scale**. This is the entire mechanism behind the silent wrong answer.

**The fix belongs in the CANN PTO header, not pypto.** The instantiated dispatch uses the mapping function in `common.hpp`; predicates from a different path in `copy_l0c_to_ub_impl.h` cannot establish that this call reaches a `static_assert`. Confirm the selected template path before predicting rejection.

### 3.1 Tables on Both Sides (Original Record)



The Python-side quantization table `_A5_MOVE_QUANT_COMBOS` in `block_ops.py` allows this case and matches the table in the official `docs/zh/api/pro_api/SIMD-API/memory_data_movement/move.md`, which explicitly states that “L0C Buffer → UB (with scale configured) … supports DT_FP32 → …/DT_FP32.”

**The documentation promises this combination and Python allows it; only the actual behavior is wrong.**

The two predicates on the C++ side in `arch35/copy_l0c_to_ub_impl.h`:

```cpp
IsSupportedQuantMode = (int32 -> half) || (-> int8)
IsSupportedBasicMode = (f32 -> half) || (f32 -> bf16)
```

Neither predicate matches `FP32 → FP32`, so reading this code suggests it should hit a `static_assert`. **Measurement showed that it does not**—actual dispatch uses the mapping function in §3, not these predicates. This remains as a record: **finding a set of apparently related predicates does not mean you found the actual dispatch path.**

## 4. Minimal Reproduction

```python
import pypto_pro.language as pl

M = N = K = 64

@pl.jit(auto_mutex=True)
def drops_scale(a: pl.Tensor[[M, K], pl.DT_FP16], b: pl.Tensor[[K, N], pl.DT_FP16],
                out: pl.Tensor[[M, N], pl.DT_FP32]):
    ub = pl.make_tile_group(
        type=pl.TileType(shape=[M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids="auto", depth=1)
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000, mutex_ids="auto", depth=1)
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x2000, mutex_ids="auto", depth=1)
    l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids="auto", depth=1)
    l0b = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids="auto", depth=1)
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[M, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids="auto", depth=1)
    t = ub.current()
    with pl.section_cube():
        pl.load(a_l1.current(), a, [0, 0])
        pl.load(b_l1.current(), b, [0, 0])
        pl.gemm(acc.current(), a_l1.current(), b_l1.current(), l0a=l0a, l0b=l0b)
        pl.move(t, acc.current(), scale=0.5)        # ← scale is dropped here
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    with pl.section_vector():
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if pl.get_subblock_idx() == 0:
            pl.store(out, t, [0, 0])
        pl.system.bar_all()
```

The output equals `a @ b`, not `0.5 * (a @ b)`.

## 5. Reproducibility

The issue does not depend on kernel-object caching, the scale value, or the input. All three scale values (0.5 / 2.0 / 0.125) with the same input produce an unscaled result.

## 6. Workarounds

**Switching to per-channel does not help** (see §2.5). Two options remain:

**The cheapest option, when the result only needs to go to GM, is to call `pl.store(out, acc, scale=)` directly** instead of moving it to UB and storing it; the same case works on the store path (§2.2).

If computation must continue in UB, there are two options:

- **Scale outside fixpipe:** move without scale, then multiply by it in UB using `vf.muls`. This costs an additional vector operation.
- **Change the destination type:** scale works for `FP16` / `BF16`, at the cost of precision.

## 7. Fix Direction

The root cause is in the CANN PTO header, so **this defect cannot be closed with a single repository change**; work must be split across both sides:

### CANN / PTO Side (Complete Fix, 4 Lines)

Add a branch to each of `GetScalarPreQuantMode` and `GetVectorPreQuantMode` in `pto/npu/a5/common.hpp`, matching the `...Gm` versions in `TStore.hpp`:

```cpp
} else if constexpr (caps::IsFP32<DstType>()) {
    quantPre = QuantMode_t::QF322F32_PRE;      // vector version uses VQF322F32_PRE
}
```

The enum values already exist, the store path already uses them, and the hardware has been verified, so this branch **completes existing support rather than adding a capability**.

### pypto Side (Containment, Needed Now)

**Do not wait for the ISA fix.** The current behavior silently returns a wrong answer, the worst of the three outcome classes:

1. In the quantization branch of `_check_layout_dtype`, reject **Acc→Vec / Acc→Mat `FP32 → FP32` with scale**. Explain in the error that this case works with store but not move, and why; provide the two workarounds (store directly / move to UB and then use `vf.muls`). **Do not reject Acc→GM**—that case works.
2. Remove `DT_FP32 → DT_FP32` from the table in the official `docs/zh/api/pro_api/SIMD-API/memory_data_movement/move.md`, or annotate it as ineffective when used with scale. Leave the table in `store.md` unchanged.
3. After the ISA is fixed, remove the rejection and restore the documentation together—**keep a test that pins down the current behavior** (`fixpipe_quant_matrix::test_the_store_path_keeps_the_scale_that_the_move_path_drops` is that test; it asserts both that store works and move is wrong, so a change to either side makes it fail).

The `INT32 → INT32` case is **a separate issue**: the enum has no corresponding mode (there is no `QS322S32_PRE`, for example), and the Python side already rejects it during parsing, which is correct.
