# Measured Fixpipe Findings

Mat→Scaling and its quantized consumer use FIX. Establish the MTE2→FIX dependency before moving a scale table from Mat into Scaling. Verify the installed framework’s pipe assignment when composing synchronization; an MTE1→FIX handover does not establish this dependency.

The five examples are in [`examples/basic_api/fixpipe/`](../../examples/basic_api/fixpipe/). Their current API target is official PyPTO `bca90a6c` on Ascend950PR / CANN 9.3.0. Known invalid routes remain explicit rejection checks. Missing guards use strict expected-failure markers, which retain the checks and expose a repaired guard as strict XPASS; they do not establish support for the unsafe route.

Defect and gap dispositions for fixpipe are in [`diagnostics/defects/fixpipe-overview.md`](../diagnostics/defects/fixpipe-overview.md) (status table, where each item is blocked, and why some are not fixed); this page records only “what measurement showed.” The contract table is in [`tooling/example-coverage.md`](../tooling/example-coverage.md).

<!-- kernel-query: {"id":"acc-final-output","aliases":["acc-final-output","final output dtype","final FP16 output","store output conversion","最终输出转换"],"symbols":["pl.store","pl.move"],"when":"A GEMM accumulator dtype differs from final storage dtype; inspect Acc-to-GM conversion before adding a cast kernel.","missing_facts":["Verify the actual accumulator/output dtypes, rounding contract, path and matching installed capability; unrelated Vec stores need their own contract."],"term_groups":[["acc","accumulator","累加器","accumulation"],["fp16","bf16","output","输出","gm"]],"discovery_terms":[["acc","accumulator","accumulation","gemm","cube","final"],["output","storage","gm","destination"],["fp16","dtype","half","conversion","convert"]],"discovery_aliases":["FP32 to FP16","FP32 narrowing","dense FP16 store","move"],"facets":{"source_memory":["Acc"],"destination_memory":["GM"]}} -->
## Choose the final output path before adding a conversion kernel

Separate accumulation dtype from storage dtype. When a completed GEMM result is in FP32 Acc and its only remaining operation is conversion to FP16 or BF16, declare the GM output in that final dtype and probe `pl.store(out, acc, offsets)` directly. The A5 Acc→GM path infers conversion from the destination dtype; it does not require an intermediate FP32 GM tensor, a VF cast, or a `scale=1.0` argument. The official store table (`pypto/docs/zh/api/pro_api/SIMD-API/memory_data_movement/store.md`) and installed `pypto_pro.ir.op.block_ops::_A5_STORE_COMBOS` own the accepted layout/dtype combinations. `scale=None` does not imply that source and destination must have the same dtype.

`query acc-final-output` also returns `selected_frontend_store`: the Acc→GM rows read from the selected Python source's non-quantized table, with its source hash, declaration line and matching `row_lines`. The reference starts at the first matching row; use the corresponding row line when inspecting a particular layout/dtype combination. It checks the literal `store/a5` table binding without importing the framework. These are source declarations, not an installation or native-execution probe; an unfamiliar or unavailable declaration remains unverified.

Choose the path from the last unfinished operation:

| Remaining work after accumulation | Candidate to validate |
| --- | --- |
| Only the supported storage conversion | Direct Acc→GM store in the public output dtype |
| A nonlinear or row-dependent epilogue | Acc→Vec, VF arithmetic and conversion, then Vec→GM within the same output owner's kernel when resources and synchronization permit |
| Another matrix consumer needs a narrower operand | The supported Acc→Mat path and the next GEMM, preserving required intermediate rounding |

Do not apply the Acc rule to Vec→GM: that transfer does not provide the same numerical conversions. Likewise, a dual-destination Acc→Vec handover has its own dtype restrictions; see §9. Finish all required accumulation before narrowing, and retain any higher-precision value still needed by another consumer. A direct store cannot perform a row-dependent normalization merely because it can convert dtype. Optional Fixpipe scaling has the precision boundary in §17; moving arbitrary arithmetic onto that path needs a separate error analysis.

Validate the complete result at the original tolerance, including output rounding and tails. When replacing a standalone cast, compare against the existing path on the same inputs and inspect the native task sequence: eliminating that launch and its intermediate traffic is the hypothesis, while the measured complete-call difference determines its value. This output-path decision applies to every terminal contraction, independently of the surrounding algorithm.

### Pair accumulator phases with the actual drain

When enabling the hardware unit-flag protocol, match the last matrix write's `AccPhase.Final` to the last drain's `STPhase.Final`; earlier writes or drains use their documented partial phases. The selected official `AccPhase.md` and `STPhase.md` contracts under `pypto/docs/zh/api/pro_api/SIMD-API/basic_data_structures/` include `store`, `store_tile`, `move` and `insert` as possible drains. Confirm the exact source/destination, dtype, layout and phase combination on the target; an older deployment lacking `move(phase=...)` does not establish a permanent API restriction. The official `memory_data_movement/move.md` also warns about `DualModeSplitN` with phase and an N extent not aligned to 32.

Do not enable the producer half while leaving the drain unspecified. When phase is unused, preserve the ordinary M→FIX dependency instead; increasing buffer depth does not repair an unbalanced flag protocol. Inspect generated code and existing fault evidence before attempting another device reproduction, following the [board discipline](../verification/board-runs.md).

## 1. Why It Has Its Own Chapter

Fixpipe controls are keyword arguments on `pl.store` / `pl.store_tile` / `pl.move` / `pl.insert`. API-name coverage alone does not establish coverage of their combinations.

Its bugs are in **combinations**: dual destination excludes quantization and relu; `offset` excludes `scale`; per-channel requires the caller to provide a Tile in the Fixpipe Buffer and establish MTE2→FIX before Mat→Scaling; `phase` must pair with the producer's `AccPhase`; INT8 output requires bit46. The parameter surface is 3 destinations × 3 layouts × 11 dtype combinations × {scalar, per-channel} × {relu, none} × {phase, none} × {atomic, none}; sampling needs 4–6 examples.

Home: `examples/basic_api/fixpipe/`, with a minimal built-in `pl.gemm` producer.

## 2. The Actual Fixpipe Parameter Surface (PTO ISA)

`pto/common/fixpipe.hpp` (CANN 9.2.0):

```cpp
template <LayoutMode_t layoutMode = NZ2ND, QuantMode_t quantMode = NoQuant,
          ReluPreMode reluMode = NoRelu, STPhase phase = Unspecified,
          uint8_t subBlockId = 0, AtomicType atomicT = AtomicNone,
          ClipReluMode_t clipReluMode = NOCLIP_RELU, bool isChannelSplit = false>
struct FixpipeParams
```

Distinguish layout selection from dual-destination selection:

| Group | Values | Python side |
| --- | --- | --- |
| `LayoutMode_t` (layout) | `NZ2NZ` / `NZ2ND` / `NZ2DN` | No explicit parameter; inferred from the destination Tile's `layout=`. |
| `AccToVecMode` (dual-destination control) | `SingleModeVec0` / `SingleModeVec1` / `DualModeSplitM` / `DualModeSplitN` | `acc_to_vec_mode=`; all four values are exposed. |

One item is **entirely absent from the Python side**: `ClipReluMode_t` (clamp relu).

`isChannelSplit` is not a gap, bit46 is ignored on some routes, and `ReluPreMode` is complete—these three “measured but not defects” findings are grouped in [`diagnostics/defects/fixpipe-overview.md`](../diagnostics/defects/fixpipe-overview.md) §4, with a note that they are there to avoid reopening the same questions.

## 3. The Eleven On-path Quantization Cases: Complete Measured Table

Measured each cell on board on 2026-09-27 (`fixpipe_quant_matrix`, 30 cases):

| src → dst (with scale) | Measured behavior |
| --- | --- |
| FP32 → INT8 | `round_half_even(scale*acc)`, saturate to `[-128, 127]`. |
| FP32 → UINT8 | Same, saturate to `[0, 255]`, **clamp negative values to 0**. |
| FP32 → FP16 | `scale*acc` is **bitwise exact**. |
| FP32 → BF16 | `scale*acc` is **bitwise exact**. |
| FP32 → FP8E4M3FN | After decoding = `round_half_even(scale*acc)`. |
| FP32 → HF8 | Runs (byte carrier; values in §15). |
| **FP32 → FP32** | **Scale is silently dropped** (output == acc). |
| INT32 → INT8 | `round_half_even(scale*acc)`. |
| INT32 → UINT8 | Same; clamp negative values to 0. |
| INT32 → FP16 | `scale*acc` is **bitwise exact**. |
| INT32 → BF16 | `scale*acc` is **bitwise exact**. |

**All 11 cases ran, 10 were correct, and 1 silently dropped scale** (a [defect](../diagnostics/defects/fixpipe-fp32-to-fp32-drops-scale.md) is open). The same 11 per-channel cases had **identical outcomes, cell for cell** (§13); only the `FP32 → FP32` cell differs for Acc→GM—**store does not drop scale** (§14).

Official PyPTO `bca90a6c` does not reject FP32→FP32 Acc→Vec with scale. On CANN 9.3.0, both scalar and per-channel `scale=2` still return the unscaled accumulator. The rejection checks therefore report strict XFAIL on this target, without launching the unsafe route. The root cause is a missing branch in the CANN A5 on-chip quantization mapping; do not predict this route using a predicate template from another path. See [fixpipe disposition](../diagnostics/defects/fixpipe-overview.md) for current handling and API scope.

⚠️ **The comment block at the top of `fixpipe.hpp` is outdated**; do not use it as a checklist: it lists `QF162*` (half source), `REQ4`/`QF162S4_PRE` (int4), `DEQS16`/`QF162S16_PRE` (int16), none of which exist in the actual enum, while omitting `QF322F32_PRE` / `QF322F16_PRE` / `QS322BF16_PRE` from the real enum—the comment and the 20 lines of code below it disagree. The real list with hardware encoding values is in §16.

### 3.1 Three Counterintuitive Findings (All Measured)

**1. Eight of the eleven destination types are reachable only with scale.** Without `scale`, the non-quantized table `_A5_MOVE_COMBOS` is used; it contains only `FP32 → FP32/FP16/BF16` and `INT32 → INT32`. To get `FP32 → INT8`, you **must** pass scale—use `scale=1.0` if no scaling is desired. **`scale` is not an optional optimization; it is the only route to those eight destination types.**

**2. But the combinations reachable with scale do not include every combination reachable without it.** `INT32 → INT32` is the reverse case: legal without scale, but rejected during parsing **with** scale (`unsupported quantized layout/dtype combination`). These are **two sets, neither containing the other**; the first finding describes only one side.

**3. The broken cell is in the three cells that are also legal without scale, while the other two in that group are correct.** `FP32→FP32` drops scale, but `FP32→FP16` / `FP32→BF16` are bitwise exact—the pattern is not “all combinations in the non-quantized table drop scale”; **the simplest accurate description is “only this one cell is broken.”**

### 3.2 Not Yet Measured / Cannot Be Measured

The dtype axis is represented in all three quantization tables (Acc→Vec / Acc→Mat / Acc→GM), with distinct numerical, execution-only and rejection criteria. The remaining measurement boundaries are in [`diagnostics/defects/fixpipe-overview.md`](../diagnostics/defects/fixpipe-overview.md). An instrument limitation does not establish a defect: the two UINT8 Acc→Mat cells have no supported numerical readback in this example. HF8 likewise needs its stated evidence-based criterion rather than a general bitwise assertion.

## 4. Additional Fixpipe Payload Controls

Check on-path quantization payload fields separately from public move/store keywords. The source searches and probes below have their own version scope; a missing parameter does not establish that a payload field is unusable.

| Rider | pypto_pro | Search evidence |
| --- | --- | --- |
| `offset` (requantization bias term) | No public `quant_offset` keyword on official PyPTO `bca90a6c`; the field is usable through runtime INT64 `scale`. | The move/store signatures and implementation accept a packed scalar configuration word. Acc→Vec and Acc→GM controls verify signed nine-bit offsets and saturation; see §18. |
| `hif8_hybrid` | **Absent**. | Searched `hif8_hybrid` `hifHybrid` `HIF8Hybrid` `hybrid`; only the FA hybrid mask and `CAST_HYBRID` rounding mode matched. |

Do not confuse this with `pl.quant` / `pl.dequant`, which are **vector-side** tile ops (with `QuantMode.SYM/ASYM` and their own `offset` Tile); they are separate from on-path fixpipe quantization. The two pages under the official `SIMD-API/quantization/` discuss the former.

## 5. Dual Supports Only Trivial Same-type Copies

The restrictions and their actual rejection layers are maintained in [`diagnostics/defects/fixpipe-overview.md`](../diagnostics/defects/fixpipe-overview.md). Check scale, relu, dtype conversion and `DN` destination separately; the current official parser guards only the scale combination.

A broad `RuntimeError` match containing `507015` cannot distinguish an AICore fault from a parser guard: `NotSupported` inherits from `RuntimeError`, and a guard diagnostic may quote that code. Assert the expected layer and diagnostic body; see [`verification/evidence.md`](../verification/evidence.md) #14.

## 6. Gaps in Official Documentation

There is **no fixpipe page** under `docs/zh/api/pro_api/`. `basic_data_structures/` has `AccToVecMode.md` / `ReluPreMode.md` / `STPhase.md` / `AtomicType.md` / `AccPhase.md`; other rules are scattered in parameter tables in `memory_data_movement/move.md` and `store.md`. There are no corresponding pages for `LayoutMode_t` or `ClipReluMode_t` in official Pro documentation.

## 7. Responsibilities of the Five Examples

| Example | Coverage | Criterion |
| --- | --- | --- |
| `fixpipe_paths` | One built-in minimal `pl.gemm` + three destinations (UB / L1 / GM) × three layouts, **baseline with no riders**. | Compare with NumPy matmul; assert each layout's output shape and arrangement. |
| `fixpipe_quant_matrix` | Eleven dtype combinations × {per-tensor scalar, per-channel Tile}; **record per cell: parsed? compiled? did scale take effect?** | Three states per cell, not pass/fail; gaps go in the table in §3 and the defect file. |
| `fixpipe_riders` | `relu_pre_mode`, `atomic`, `phase` (Partial/Final) pairing with producer `AccPhase`, and `order`. | Use negative input to test relu; run atomic twice; give each rider its own control group. |
| `fixpipe_dual_dst` | Four `AccToVecMode` values; three exclusion rules for “dual + rider” (§5). | Each dual Vec subblock receives half; each rule has a compile-time rejection check. |
| `fixpipe_scaling_buffer` | Caller-managed per-channel Scaling Tile: GM→L1→FB with MTE2→FIX before the FB move, six shape/type constraints, bit46 semantics. | Use different scale per column and compare output by column; add a “change only column 0” case to rule out broadcasting. |

## 8. `fixpipe_riders`: Axis Selection and Launch Multiplicity

The 2026-09-27 board probes cover `relu_pre_mode`, `phase`, axis selection and atomic launch multiplicity. The following controls distinguish axis mapping from transpose and a single launch group from automatic budgeting.

**1. `order` is not transpose.** Using `order=[1, 0]` for “transpose on store” raises:

```
InvalidVal [block_ops.py:348]: store: order must be ascending, got [1, 0]
```

`order` answers **“which two axes of a higher-rank Tensor should receive a 2D Tile”**—it requires two elements, distinct, within rank, and in **ascending order** (`_validate_tile_dims` plus the ascending-order check at line 348). If omitted, it defaults to the final two axes, `list(range(ndim - 2, ndim))`. Ascending order is a hard constraint, so store cannot be used for transpose this way.

This example pins three related facts:

| | Result | Evidence |
| --- | --- | --- |
| `store(..., order=[0, 2])` to axes 0/2 of a 3D ND Tensor | Accepted; offset on the omitted Tile axis is treated as an index. | Measured; page 0 remains 0. |
| `store(..., order=[2, 0])` descending | Rejected during parsing. | `block_ops.py:347`. |
| NZ destination + non-final two axes | Rejected during parsing (a separate rule). | `block_ops.py::_validate_nz_transfer_axes`. |

**There is no ascending-order check on the `load` side**—official ST `test_cube_datacopy_api_basic.py:416` uses `pl.load(cur_k, k, [0, 0], order=[1, 0])`. So transpose may be available on load but not store. This example did not test load; it records only this asymmetry.

**2. The default `block_dim` uses all cores, exposed by `AtomicAdd`.** The initial direct call to `store_atomic_add(...)` first produced `28 × scale × acc`. The root cause was not fixpipe: direct `kernel(...)` uses `_AUTO_BLOCK_DIM = 0` (`runtime/jit.py::_AUTO_BLOCK_DIM`), resolved by native code to the full budget of the current stream; these kernels do not partition work by `get_block_idx()`, so every block computes the same value and adds it into the same GM region.

**This is invisible in overwrite tests**—every block writes identical bytes, so the 39 tests in `fixpipe_paths` / `fixpipe_quant_matrix` did not reveal it. `AtomicAdd` is the only case where the output reveals the block count.

On this machine, `get_platform_info()` reported `core_num=28 / cube_core_num=28 / vector_core_num=56`; a pure cube section launched **28** blocks by default (consistent across three runs), while `support/debug_probe` reported `get_block_num()==56` for a pure vector section. Both values are measured; why cube gets 28 and vector gets 56 was not investigated and is out of scope for this example.

One related FP16 pitfall: `28 × 0.5 × acc` can exceed the “half-integer exactly representable” range (FP16 spacing in `[1024, 2048)` is **1.0**), so the bitwise comparison in the default-launch test covers only positions where `|k × want| < 1024`; exactness is handled by the `[None, 1]` / `[None, 2]` cases.

## 9. `fixpipe_dual_dst` Measurements: One Rule Has Three Different Outcomes at Three Layers

Board run on 2026-09-27, 14 checks passed. All four `AccToVecMode` values work, and `DualModeSplitM/N` are bitwise correct. Four additional findings:

**1. “Dual does not carry riders” has three different outcomes in this version.**

| Combination | Where blocked | What happens |
| --- | --- | --- |
| dual + `scale` | Python parsing | `NotSupported`: scale cannot be combined with dual-mode... |
| dual + lower-precision dtype (FP32→FP16, no scale) | **C++ compilation** | `static_assert(quantPre == NoQuant, "Quant is not support in dual Dst Mode.")` |
| dual + `relu_pre_mode` | **Not blocked at any layer** | Runtime **AICore exception** (`507015`), leaving the device state unusable for the process. |

The second row follows from the dtype-selected mode: **FP32 → FP16 is itself a QuantMode in the ISA (`F322F16`)**, so even without `deqScalar` it hits the same “dual requires `NoQuant`” check. The Python guard checks only the `scale` argument and cannot see the QuantMode inferred from dtype, so it reaches C++ compilation.

The third row is a real gap, documented in the [defect](../diagnostics/defects/fixpipe-dual-dst-plus-relu-faults-aicore.md): `TMov.hpp:233` passes `dualDstCtl` and `reluMode` side by side to `pto_copy_matrix_cc_to_ub`; neither layer cross-checks them. **`unitFlagCtrl` (`phase`) is beside it and equally unchecked**—same shape, but not measured because an unbalanced UF chain “hangs the AI Core” instead of throwing, and the already-readable source evidence did not justify occupying a shared card for another cell.

Acc→Vec move uses `TMovCcToUb`, which accepts and forwards `reluMode` unchanged; `TExtractL0CToUBDualDst` is a different route. Reading the latter's predicates does not establish that relu is silently dropped on the former. Resolve the instantiated path before predicting behavior.

**2. A move without `acc_to_vec_mode` defaults to `SingleModeVec0`.** The test seeds both UBs with 1×, then writes 2× with no mode; all four possible outcomes are distinguishable and no assumption is made about “uninitialized UB” contents. The ISA agrees: `subBlockId = (mode == SingleModeVec1)` in `TMov.hpp`, while `pl.move` defaults to `SingleModeVec0` when mode is omitted.

**This is an implicit assumption in the 39 tests in `fixpipe_paths` / `fixpipe_quant_matrix`**—they all use “move without mode + `if sub_id == 0`” stores and depend on this answer; it simply had not been recorded before.

**3. `block_num` / `subblock_num` depend on kernel shape.**

| Kernel shape | `get_block_num()` | `get_subblock_num()` |
| --- | --- | --- |
| Pure vector section | 56 | 1 |
| Cube + vector mixed | **28** | **2** |

`get_platform_info()` reports `core_num=28 / cube=28 / vector=56`. For the mixed shape, 28 × 2 = 56 and 56 × 1 for pure vector are consistent. This also explains the “default block_dim multiplier = 28” in §8—it was a mixed kernel.

**4. `isChannelSplit` is not “unexposed”; it is inferred from the destination Tile type.** `TMov.hpp:199`: `channelSplitEnable = !isRowMajor && SFractal == RowMajor && float && SFractalSize == 512`. So it should not remain on the gap list—**correction to §2**.

## 10. Toolchain Identity in Child Processes

Check subprocess import precedence explicitly. A child can import an older conda `pypto_pro` when its `PYTHONPATH` or interpreter differs from the parent. Compare `pypto_pro.__file__` and loaded native-library paths with the selected environment before interpreting a subprocess failure.

Bind every evidence item to a toolchain identity. See [`verification/board-runs.md`](../verification/board-runs.md) §5.5 for operating instructions.

## 11. `fixpipe_scaling_buffer` Measurements: bit46 Changes the Saturation Range

**1. Per-channel numerics.** FP32 → FP16 / FP32 → INT8 are bitwise exact on the controlled power-of-two scales and integer products used by the example. FP32 → FP32 with scale on Acc→Vec silently ignores scaling on the current target, and its required parser rejection is absent; see the [defect](../diagnostics/defects/fixpipe-fp32-to-fp32-drops-scale.md). The FB bitfield encoding is distinct from a per-tensor instruction immediate.

**2. bit46 (called `FP_SCALE_SIGNED_INT8_FLAG` in the official ST) selects the saturation range.** With the bit clear, INT8 output is **bitwise equal** to:

```python
(np.clip(np.rint(acc * scale), 0, 255) + 128) % 256 - 128
```

That means saturation to `[0, 255]`, then reading the bytes as int8. Negative values become zero and values above 127 wrap to negative signed payloads. For floating-point destinations, setting or clearing this bit produces bitwise-identical output. The flag tells Fixpipe that the destination is signed.

**3. All six shape/type constraints are checked during parsing**, with distinct diagnostics from the scale consumer in `block_ops.py`. With the example's explicit Tile groups, an INT64 table `[1, 1024]` reaches the consumer's `N must be <= 512` check and raises `OutOfRange` before launch. The limit follows from the 4 KiB Fixpipe Buffer.

**4. Per-channel + `atomic` is rejected** (`block_ops.py::_ir_store` / `_ir_store_tile`, after `_resolve_scale_param`), while per-tensor + atomic works (§8). This is a per-channel restriction, not an atomic restriction.

## 12. Tile Types and Closure Values

Tuple-unpacking Tiles that differ only in memory space is supported; see [constraints](constraints.md). Whether an enum may come from a closure depends on its argument position. Op kwargs such as `bin_type=` accept closure values; inspect the native type resolver before applying the same assumption to parameter annotations.

## 13. Completing the Three Quantization Tables: Per-channel, Acc→Mat, and a More Accurate Instrument

`fixpipe_quant_matrix` covers scalar and per-channel scales, Acc→Mat readback, direct Acc→GM stores, HF8 execution and decoding, scale precision and packed payload fields. Each route retains its own numerical or rejection criterion.

**1. The same 11 per-channel cells have outcomes identical to per-tensor, cell for cell.** This includes the dropped scale in `FP32 → FP32`. So this is a table of “dtype combinations,” not “dtype combinations × scale forms”—the test reuses `outcome` from `MATRIX` to run the second scale route and **asserts equivalence between the routes**.

**One asymmetry: different mechanisms determine the saturation range for integer destinations.**

| | What determines the saturation range |
| --- | --- |
| Per-tensor scalar | **Destination Tile dtype** |
| Per-channel table | **bit46 in the table**, regardless of destination dtype |

Measured: UINT8 destination + bit46 set → saturates to `[-128, 127]`. When they disagree, **bit46 wins**, and no layer checks it (the table is host data, invisible to the compiler). The caller must keep them consistent; the mapping is in `reference.py::SIGNED_FLAG_FOR`.

**2. Ten Acc→Mat cells.** Exactly the eleven Acc→Vec cells minus `FP32→FP32`. The hard part was not running the operation but **reading the result back**—L1 output has only one exit (`Mat→Left/Right` → multiply again). We use **`q @ q`**: use the quantized result as both operands, so no external operand has to be created (an identity matrix would need encoding in the destination dtype, and there is no HF8 encoder).

| `verify` | Number of cells | Result |
| --- | ---: | --- |
| `q_at_q` | **7** | `q @ q` is bitwise equal to `q_ref @ q_ref`. |
| `runs_only` | 1 | HF8 runs and is nonzero; values not checked. |
| `no_readback` | 2 | **UINT8 cannot be read back**: `Mat(NZ,UINT8) → Left(NZ,UINT8)` is rejected during parsing. |

The two `no_readback` cells deserve one sentence: **“cannot read back” does not mean “this cell is broken.”** `_A5_MOVE_QUANT_COMBOS` allows `Acc(FP32) → Mat(UINT8)`, so the quantization step may be correct, but UINT8 in L1 has no exit route. Recording an instrument limitation as a finding about the measured operation is one of the easiest mistakes in these matrices; this table uses three states rather than pass/fail.

**3. `FP8E4M3FN` has an exact encoding criterion.** `quantize_to_fp8_e4m3` enumerates the 253 representable values and selects the nearest, enabling `atol=0` instead of an approximate decoded-value comparison against `scale*acc`. Deriving the encoder from the decoder avoids relying on a separately handwritten encoding algorithm; the reference still needs the format's rounding and saturation rules.

Thus FP8E4M3FN now passes bitwise checks on both paths. **The same data supports a stronger conclusion when the instrument improves**—more valuable than running one additional cell. A companion test, `test_the_fp8_encoder_is_built_from_the_decoder`, checks all 253 values, the ±448 range, and identity for representable values. **The criterion itself must also be validated.**

## 14. The Fourth Table Isolates the Defect to One Cell: L0C→UB Is Broken, Not This Dtype Combination

`_A5_STORE_QUANT_COMBOS` has 22 rows = the same 11 dtype combinations × destination Tensor layout `ND` / `NZ`. After measuring each cell, **only one differs from Acc→Vec**:

```python
pl.store(out, acc, [0, 0], scale=0.5)    # L0C(FP32) → GM(FP32): bitwise exact
pl.move(ub, acc, scale=0.5)              # L0C(FP32) → UB(FP32): scale is dropped
```

Neither `ND` nor `NZ` destinations drop scale. **Same dtype combination, same scale value, same L0C; changing only the destination makes it correct.**

Along with the per-channel result from §13, the two directions isolate the defect:

| Dimension | Result |
| --- | --- |
| Scale encoding (scalar immediate / FB bitfield) | Both drop it → **not an encoding problem**. |
| Destination (UB / GM) | UB drops it, GM does not → **destination-specific**. |
| Destination layout (ND / NZ) | Neither drops it. |

So the defect location narrows from “this dtype combination is broken” to “**the destination is UB and its dtype is FP32**.” This matches the filename: `arch35/copy_l0c_to_ub_impl.h` handles only the UB path.

There is now a correct reference path when fixing it: `store` works, so compare the dispatch on both sides to see which branch the UB route takes when “destination dtype == source dtype.” This comparison was impossible with only one failing case.

**A better workaround when the result only needs to reach GM:** call `pl.store(scale=)` directly; do not move to UB and store afterward.

**The table's shape is itself a finding.** `reference.py` has no fourth 11-row table; it has only `STORE_OUTCOME_OVERRIDES = {("FP32", "FP32"): "exact"}` as a one-line override. “Almost everywhere the same” is part of the result; copying an almost identical table would bury it. Another test pins the override table to exactly the cells that differ from Acc→Vec, preventing the two records from drifting apart.

## 15. The `HF8` Cell: Neither of Two Failures Was an HF8 Problem

The HF8 criterion in these quantization examples uses the device's `vf.astype(HF8 → FP32)` decoder. It therefore establishes cross-unit consistency, not an independent Host format oracle. Isolate transport and register conversion before interpreting the decoded values.

**Control the slot before comparing layouts.** A probe combining four `astype` calls, output-tile `depth=4` and four `.next()` calls returned `[-256.0, nan, 0.0, 0.0]`; a single conversion using one `current()` produced usable data. That comparison changes both layout coverage and slot handling, so it cannot isolate a CastLayout defect. Start with one initialized source and one observed destination, then vary one selector.

**Wait on the consumer's first pipe.** A VF consumer uses V; waiting on MTE3 instead passed alone but failed alongside other examples (maximum relative difference 7.6, periodic wrong positions). Changing only the wait to `pipe=pl.PipeType.V` passed three consecutive runs in the recorded control.

**The official documentation identifies the blocked pipe**: the parameter table of `wait_cross_core.md` says it need not match the paired producer's `set_cross_core` pipe. Copying MTE3 from a `pl.store` consumer does not establish the wait needed by VF arithmetic.

This is recorded in [`api/constraints.md`](constraints.md): `pipe` blocks the pipeline you name, so it must be the consumer's pipeline. All four examples in this chapter use `pl.store` as the consumer, hence the many `MTE3` values—the most copyable pattern is precisely the one easiest to cargo-cult.

For “passes alone, fails in combination,” inspect synchronization and prior state using [numerics and races](../verification/board-runs.md#numerics-and-races). That symptom prioritizes a check; it does not prove a unique cause.

**Historical conversion-probe outputs, not a layout support matrix.** The recorded integer probes returned:

| Conversion | Values read with `ZERO` | `ONE` / `TWO` / `THREE` |
| --- | --- | --- |
| `INT8 → INT32` (4× widening) | lanes `0, 4, 8, …` | **All zero** |
| `INT16 → INT32` (2× widening) | lanes `0, 2, 4, …` | **All zero** |

The INT8→INT32 position check used 256 distinct values `-128..127`, so each decoded value identified its source position. This summary does not retain the exact predicate construction, source-register arrangement or source hash for those probes; their zero outputs cannot establish that nonzero selectors are unsupported or that other source lanes are inaccessible. The current [conversion-position and predicate contract](vf.md#213-the-fp32-path-for-fp16-source-data-order-preserving-widening-with-unpk-required-reading) includes legal `ONE` consumers and explains why a predicate for even halfwords does not enable odd ones. Recheck each dtype pair against the official conversion table, then vary layout with compatible masks and coordinate-distinct input; do not reuse the old table as a rejection rule.

**What this final cell verifies—and what it does not.** The criterion is “decoded value differs from `scale*acc` by at most 1/8 relatively” (measured maximum 0.111; hifloat8 is a tapered format whose mantissa shrinks as the exponent grows, so the bound is looser than 1/16), plus two anti-vacuity checks: include bitwise-equal values (to rule out systematic offset), and multiple distinct values (to rule out a constant output).

**It proves “the bytes written by fixpipe agree with the vector unit's decoder”—two different hardware units interpret this batch the same way. It does not prove the format matches an external specification**, because both sides of the criterion come from this chip. Bitwise comparison would require an encoder that selects the nearest representable value; the value table can only be inferred from the device decoder (one measurement: 253 finite values, ±32768, sign-symmetric, and **not monotonic by byte order**—a property of tapered formats). That table is also device-derived, so it is no more independent than the current check.

## 16. Full A5 QuantMode Table and Reachability from pypto_pro

**Three mutually corroborating evidence sources:**

| Source | Provides |
| --- | --- |
| `pto/costmodel/a2a3/cce_costmodel/cce_costmodel_core.hpp:17` | Enum definition itself, **with hardware encoding values** (24 members). |
| `pto/npu/a5/common.hpp:73-138` | Three on-chip mappings: `GetCastPreQuantMode` / `GetScalarPreQuantMode` / `GetVectorPreQuantMode`. |
| `pto/npu/a5/TStore.hpp:18-88` | Three GM mappings: same names with the `Gm` suffix. |

The mapping functions are `if constexpr` chains selecting by `(SrcType, DstType)`, so “which modes are reachable” is equivalent to “which dtype combinations can reach them”—exactly what the four tables in this chapter measure.

| # | QuantMode | Semantics | Reachable from pypto_pro |
| --- | --- | --- | --- |
| 0 | `NoQuant` | No quantization | ✓ Without scale. |
| 1 | `F322F16` | fp32→fp16, **without scale** | ✓ |
| 16 | `F322BF16` | fp32→bf16, without scale | ✓ |
| 24 / 23 | `QF322B8_PRE` / `VQF322B8_PRE` | fp32→int8/uint8 | ✓ Measured. |
| 25 / 28 | `QF322HIF8_PRE` / `VQF322HIF8_PRE` | fp32→hifloat8 | ✓ Measured. |
| 26 / 37 | `QF322FP8_PRE` / `VQF322FP8_PRE` | fp32→float8_e4m3 | ✓ Measured (bitwise). |
| 32 / 33 | `QF322F16_PRE` / `VQF322F16_PRE` | fp32→fp16, **with scale** | ✓ Measured. |
| 34 / 36 | `QF322BF16_PRE` / `VQF322BF16_PRE` | fp32→bf16, with scale | ✓ Measured. |
| 3 / 2 | `REQ8` / `VREQ8` | int32→int8/uint8 | ✓ Measured. |
| 5 / 4 | `DEQF16` / `VDEQF16` | int32→fp16 | ✓ Measured. |
| 35 / 39 | `QS322BF16_PRE` / `VQS322BF16_PRE` | int32→bf16 | ✓ Measured. |
| **27 / 38** | **`QF322F32_PRE` / `VQF322F32_PRE`** | **fp32→fp32, with scale** | **Reachable only through `store` (Acc→GM).** |
| **13 / 12** | **`SHIFTS322S16` / `VSHIFTS322S16`** | **int32→int16, shifted** | **Completely unreachable.** |

**Of 24 modes, 20 are normally reachable, 2 partially, and 2 are not connected at all.** Details for the latter groups:

**1. `QF322F32_PRE` / `VQF322F32_PRE` (27 / 38)—hardware supports them, but the on-chip path is not wired.** Both `...Gm` functions in `TStore.hpp` have an `IsFP32<DstType>` branch; the two on-chip functions in `common.hpp` **do not**, so they fall through to `NoQuant`. **This is the complete root cause of the dropped scale in `FP32 → FP32`**, recorded in the [defect](../diagnostics/defects/fixpipe-fp32-to-fp32-drops-scale.md): the fix belongs in the CANN PTO headers (add one `else if` to each); the PyPTO-side guard is a mitigation until then.

Measurement confirms both routes: per-tensor and per-channel `FP32 → FP32` scales both **work through store** (`VQF322F32_PRE` is also correct) and both **drop scale through move**.

**2. `SHIFTS322S16` / `VSHIFTS322S16` (13 / 12)—int32→int16, with no path at all in the A5 PTO layer.** The enum contains them, but none of the six mapping functions in `common.hpp` and `TStore.hpp` returns them (`IsSInt32` has only int8 / fp16 / bf16 cases). So this is not merely an API missing from pypto_pro; **the A5 PTO C++ API cannot reach it either.**

Expansion of this path is outside the current scope for three reasons:

1. It is “shift-based quantization,” not the “float-scale quantization” used by other modes (`SHIFT` rather than `deqScalar`), so adding it creates a new parameter surface.
2. The headers do not establish whether A5 truly supports int16 fixpipe output—the enum is shared across architectures (the path is under `costmodel/a2a3/`) and may be for A2/A3.
3. **There is no use case for it.** The purpose of int16 output on this path is unclear, and there is no user to validate it.

Before adding it, resolve reason 2 first: **that needs ISA documentation or a hardware manual; rereading the headers cannot settle it.**

## 17. Precision of `scale`: fp19, Truncated Toward Zero

**Power-of-two scales do not expose mantissa truncation.** They can make `scale * acc` exact in the destination type while leaving scale precision untested; include non-power-of-two values that separate the candidate rounding rules.

Measured in `fixpipe_quant_matrix`, `store_with_runtime_scale`, destination FP32:

| Property | Result | Measurement method |
| --- | --- | --- |
| Explicit mantissa | **10 bits** | `scale = 1 + 2^-k`: k≤10 changes the result bitwise; k≥11 is entirely dropped (hardware uses 1.0). |
| Exponent | **At least 8 bits** | `2^±100` remains accurate; 7 signed bits would reach only ±63. |
| Sign | Effective | Negative scale works normally. |
| Rounding | **Truncate toward zero** | `±1.00146484375 → ±1.0009765625`; round-to-nearest-even would choose the value above it. |

So **1 + 8 + 10 = 19 bits: fp19**, not fp20.

The measurement method is worth recording: **use all ones as input so every `acc` element equals K (a power of two); dividing the output by K gives the actual scale used by hardware**—no inverse reconstruction and no destination-precision blurring. The destination must be `FP32`, and `FP32 → FP32` with scale **works only through store**, so the test can only use `pl.store` (another application of the §14 finding).

The reference is `reference.py::truncate_scale_to_hardware`: on the float32 bit pattern, **clear the low 13 bits (23 - 10)**—clearing bits truncates toward zero for both signs.

**Independent second evidence: the ISA-side payload packing code contains this mask.** `asc/impl/c_api/instr_impl/npu_arch_3510/cube_datamove_impl/asc_set_l0c_copy_prequant_impl.h:32`:

```c
config = (is_signed << 46) | ((offset & 0x1FF) << 37) | (float_bits(scale) & 0xFFFFE000u)
```

`& 0xFFFFE000u` masks the low 13 bits → leaves 10 explicit mantissa bits, and **masking them is truncation toward zero**. Both measured facts (10 bits and truncation) can be read from this line. `test_the_payload_layout_agrees_with_the_measured_mantissa_width` pins both sources of evidence together; if either changes, the test fails.

The cited `scale` parameter descriptions in `move.md:43` / `store.md:43` cover float / Scalar / Tile forms, per-channel preparation and bit46, but do not state the measured 19-bit scale truncation. Keep that measured precision boundary distinct from the documented input forms.

**Practical effect:** `scale = 0.1` is truncated to `0.0999755859375`, a relative difference of 2.4e-5. This error is **introduced by fixpipe, not by accumulation**—keep it in mind first when debugging numerical differences. For higher-precision scaling, do not use on-path quantization; move to UB and use `vf.muls` (full FP32 precision).

## 18. `offset` Is Also Encoded in the uint64—**An Interface Gap, Not a Capability Gap**

The C++ line in §17 also reveals two other fields. Since pypto writes an **integer** passed as a runtime INT64 `scale` value directly into the configuration register, we can manually construct the whole payload and ask whether hardware recognizes it.

Measured in `fixpipe_quant_matrix`, `store_with_runtime_payload_int8`, `scale=0.5`, all-ones input → `acc = K = 64` → output is 32 without offset:

| `offset` bits | Interpreted as signed 9-bit two's complement | Output | Finding |
| --- | ---: | ---: | --- |
| `0` | 0 | 32 | Baseline. |
| `5` | 5 | **37** | Added **after scaling**; otherwise `(64+5)*0.5 = 34.5`. |
| `100` | 100 | **127** | `132` is out of range and clamps → added **before saturation**. |
| `0x100` | −256 | **−128** | High bit is the **sign bit**; unsigned interpretation would be `+256`, clamped to 127. |
| `0x1FF` | −1 | **31** | All bits set means −1. |

With an **FP32** destination, the same payload has no effect (control case)—offset applies only to integer destinations.

Official PyPTO `bca90a6c` exposes this capability through the runtime INT64 `scale` payload, with no `quant_offset` keyword. Validate the signed offset range before encoding it, pack the FP32 scale bits with `& 0xFFFFE000`, and put `(offset & 0x1FF) << 37` into the same word. Pass that word through a `pl.DT_INT64` kernel parameter. The [example](../../examples/basic_api/fixpipe/fixpipe_quant_matrix/kernel.py) performs only metadata encoding on the host; the original GEMM and Acc→Vec move perform the computation. Independent reference encoding and GM-store controls check the same five offset settings.

Do not confuse this field with `offsets` in `store.md:41` or `offset` in `move.md:40`: those are Tensor/Tile coordinates, unrelated to quantization bias.

**Also pinned down bit46.** It behaves **differently** on the two routes:

| Route | `bit46` | Measurement |
| --- | --- | --- |
| Per-tensor (float / Scalar form) | Set / clear | **Same either way**; both interpret as signed (`acc=-64 → -32`)—**the bit is ignored**. |
| Per-channel (Scaling Tile form) | Set / clear | **Determines the saturation range and overrides destination dtype** (see `fixpipe_scaling_buffer` §3). |

Official `store.md:43` / `move.md:43` scope this bit to **Tile form**, consistent with measurement. Their INT8 example does not exclude UINT8: the measured Tile-form selector also affects a `UINT8` destination. Preserve the route distinction when interpreting the parameter description.

## 19. Guard disposition and validation boundary

The authoritative disposition is in [`diagnostics/defects/fixpipe-overview.md`](../diagnostics/defects/fixpipe-overview.md), including dual + dtype conversion and dual + DN. It owns the cause, rejection layer and safe alternatives for each route.

Keep rejection tests separate from numerical controls. On official PyPTO `bca90a6c`, the FP32→FP32 scaled-move checks and three dual-destination checks expose absent parser guards through strict XFAIL markers. Only the missing-rejection outcome is expected; other errors remain failures, and a passing required guard becomes strict XPASS. Do not launch dual + relu to test that absence, or accept an unrelated compiler error as a parser diagnostic. A direct numerical probe of scaled FP32 move must compare against `scale * acc`, since successful compilation and a nonzero result do not establish that scaling occurred.

For a framework guard implementation, check dual-mode restrictions before the FP32 scaled-move restriction. A dual + scale diagnostic should identify the dual restriction even when FP32→FP32 is also invalid, so changing only the output dtype cannot appear to make that route legal. Keep Acc→GM scaled-store positive controls, since that route preserves scaling.
