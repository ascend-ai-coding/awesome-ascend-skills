# Measured Support Findings

**All eight examples were run on board** (2026-09-28, Ascend950PR). The examples are in [`examples/basic_api/support/`](../../examples/basic_api/support/). **At the close of this chapter, all 40 examples in the example plan had been implemented.**

`debug_probe` and `dtype_carriers` were delivered in batch 1 (you need `printf` and a low-precision carrier table on the first day of writing VF); their findings are in §13 / §14 at the end. Sections 2–12 cover the six examples from batch 6.

The contract table is in [`tooling/example-coverage.md`](../tooling/example-coverage.md).

## 1. Responsibilities of the Eight Examples

The six examples in batch 6 are `spr_ctrl` (CTRL bitfields), `saturation` (saturation flags), `block_grid` (core/subcore indices), `scalar_const` (scalars and compile-time constants), `entry_and_loop` (`pl.jit` / `pl.pipeline` / `pl.range`), and `layout_table` (legal layout combinations). The two examples delivered in batch 1 are at the end: `debug_probe` (§13) and `dtype_carriers` (§14).

**None of the `spr_ctrl` and `saturation` APIs has an ST**—those two examples probe behavior from scratch rather than copying an ST. See [`tooling/example-coverage.md`](../tooling/example-coverage.md) for the contract table; use `pytest examples/basic_api/support -q` for the current test count.

## 2. CTRL Register: Entry Value Differs from the Documented Default

| | Value | bit 48 |
| --- | --- | --- |
| Official `reset_ctrl_spr.md` and C++ `defaultCtrl` | `0x1000000000000008` | 0 (saturated) |
| **Measured kernel entry value** | `0x1001000000000008` | **1 (not saturated)** |

The difference is exactly `{48}`; the other writable bits match the documentation. The consequence is more serious than it first appears:

> **`reset_ctrl_spr` restores the hardware default, not the value at kernel entry.**

Therefore, the pattern “set → work → reset” for “restoring the original state” **changes bit 48 from 1 to 0**, silently switching floating-point conversion from non-saturating to saturating. To actually restore the prior value, first save it with `get_ctrl_spr` and then write it back.

**CTRL does not persist across launches** (three consecutive launches each read bit 48 and clear it; all three entries read 1). This is good news, but it must be measured before relying on it.

All three guards run at **compile time** and give precise messages: unwritable bits (lists writable bits), `CTRL[8:6]=0b111` (lists legal values 0–6), and `CTRL[10:9]=0b11` (lists ADD/MAX/MIN). **They apply only to literals**—the C++ side first tries `ir::As<ir::ConstInt>` and skips the check if it fails. According to the comment, bypassing them can cause a device fault; **that case was deliberately not tried**.

## 3. Saturation-Flag Polarity Is Reversed for Three Modes

| Mode | CTRL bit | Bit value for `enable=True` | Polarity | Saturated at entry |
| --- | ---: | ---: | --- | --- |
| `FLOAT` | 48 | **0** | Inverted | **Off** |
| `FLOAT8` | 50 | **0** | Inverted | On |
| `INT` | 53 | 1 | Normal | Off |
| `CAST` | 59 | **0** | Inverted | On |

`IsInvertedPolarity` in `backend_cce_ops.cpp` states this explicitly.

**Mixing the two interfaces can be misleading:** after `set_saturation_flag(mode=CAST, enable=True)`, `get_ctrl_spr(59, 59)` returns 0, making it look as though the setting had no effect. `get_saturation_flag` itself is symmetric.

Keep polarity and entry state in separate columns: **the two “off” states have different causes**. FLOAT is “bit is 1 + inverted polarity,” while INT is “bit is 0 + normal polarity.” Collapsing those properties predicts the wrong behavior for both cases.

## 4. CTRL[60] Determines Which Setting Controls Saturation (Six-Line Measurement)

| CTRL[60] | `set_saturation_flag(CAST)` | `astype(saturate=)` | Result |
| ---: | --- | --- | --- |
| 1 | `True` | `OFF` | Saturating |
| 1 | `False` | `OFF` | Truncating |
| 1 | `True` | `ON` | Saturating |
| 1 | `False` | **`ON`** | **Truncating** ← `saturate` is ignored |
| 0 | `False` | **`ON`** | **Saturating** ← `saturate` takes effect |
| 0 | **`True`** | `OFF` | **Truncating** ← flag is ignored |

The final three rows are the discriminating cases; rows 4 and 5 use the same parameters except for CTRL[60], and produce opposite results. And **CTRL[60] is 1 at kernel entry**, so:

> By default, `vf.astype(saturate=...)` **has no effect**; saturation is controlled by `set_saturation_flag`.

If passing `saturate=ON` to `astype` has no effect, the first instinct is to think “the argument was passed incorrectly”; the setting that needs to change is CTRL[60].

## 5. `get_subblock_num` Depends on Whether the Kernel Has a Cube Section

| Kernel | `subblock_num` | `subblock_idx` (4 blocks) |
| --- | ---: | --- |
| Only `section_vector` | 1 | 0, 0, 0, 0 |
| Only `section_cube` | 1 | 0, 0, 0, 0 |
| **Both sections present** | **2** | **0, 1, 0, 1** |

The docstring says “Returns 1 on AIC binaries, `get_subblockdim()` on AIV binaries”—**that wording does not reveal that adding a cube section changes the result**, because the sentence describes binary type, while “which binary is generated” depends on the sections in the kernel.

Consequence: code that partitions work using `subblock_idx` changes behavior the day a cube section is added to the kernel. For a pure vector kernel, `subblock_idx` is always 0; code that “splits work by subcore” makes every block take the same half, **without an error**.

Recheck of the current installation on 2026-10-06 (behavior version `872df763f`, CANN 9.2.0): pure Vector launches `kernel[None, W]` at W=1/3/8 recorded `get_block_idx()` as 0..W-1, `get_block_num()==W`, `get_subblock_idx()==0`, and `get_subblock_num()==1`. The existence of two AIVs in hardware does not mean every pure Vector launch yields twice as many workers.

For a pure Vector loop that processes one row per iteration, use `pl.range(pl.get_block_idx(), rows, pl.get_block_num())` to match the launch; writing the stride as `2*W` skips an entire range of rows. See [row_gather](../../examples/algorithms/row_gather/) for a runnable consumer and boundary tests. This does not overturn the `block_idx // subblock_num` grouping used by mixed kernels, nor does it mean every custom mapping with stride `2*W` is wrong.

## 6. Block Index APIs Share Values but Have Different Call Scopes

| | Kernel side | SIMT side | Measured |
| --- | --- | --- | --- |
| Block index | `pl.get_block_idx()` | `pl.simt.block_idx().x` | **Same** |
| Grid size | `pl.get_block_num()` | `pl.simt.grid_dim().x` | **Same** |

The distinction is **where each can be called**: `pl.get_block_idx()` is rejected during parsing inside a SIMT body (`parse_op_call` in `_call_parser.py`), while `pl.simt.block_idx()` is the SIMT-side entry point. For the measured block index and grid size above, their values agree across the two scopes; do not infer different ownership merely from the namespace.

## 7. Small GM Scalar Writes from Multiple Blocks

The following historical observation applies only to the GM scalar path using `pl.setval`; eight blocks each write four INT32 values to their own segment. Do not generalize it into a rule requiring gaps between all GM writes.

| Per-block spacing | Blocks preserved out of 8 | Blocks preserved out of 16 |
| ---: | --- | --- |
| 16 bytes | **2–3**, varying by run | — |
| 64 bytes | **All 8, in 5 consecutive runs** | **All 16, in 5 runs** |

For a block whose write was lost, the **entire segment** remains at its initial value, not a partial write—it was overwritten by another block's stale-data write. **`pl.system.dcci` does not recover the writes** (only 2–3 blocks still survive, varying by run, even with dcci).

**The mechanism is not settled:** 64 bytes resembles a cache line, but “which blocks survive” does not follow “at most one winner per line” (observed `[0, 3, 7]`, where 0 and 3 share a line at 16-byte spacing). So report only the facts: 64-byte spacing is stable, smaller spacing loses writes, and dcci has no effect.

On 2026-10-06, the small per-row-result path was rechecked under behavior version `872df763f`, CANN 9.2.0: same VF computation, same row owner and inputs, with only the writeback route changed. Each run had 65 FP32 and INT32 results, used 1 or 4 workers, and rotated through three seeds.

| Writeback route | 1 worker | 4 workers |
| --- | --- | --- |
| Write adjacent values one by one with `pl.setval` to a dense GM array | 3/3 exact | 3/3 failed; each run left 48 values at the initial NaN |
| First store results in UB, narrow the valid region, then use `pl.store` to the same dense GM array | 3/3 exact | 3/3 exact |
| Diagnostic only: change scalar GM writes to 64-byte spacing | 3/3 exact | 3/3 exact |

Therefore, prefer forming small reduction results in a UB result tile and DMA-writing the actual valid region; pair a one-dimensional GM array with a two-dimensional UB tile whose physical size is aligned. The dense-output contract is preserved; callers do not need to leave 64 bytes for each result. If GM scalar writes are necessary, first verify physical write regions and visibility across workers; distinct logical indices alone do not prove that the path is safe in parallel.

The evidence is the official scalar semantics in `setval.md`, installed Python Tile/Tensor dispatch, the ordinary `__gm__` pointer assignment emitted by C++ `MakeSetValCodegenCCE`, and the board comparison of the scalar and DMA paths. The exact hardware mechanism remains unproven; do not infer a block-indexing error from periodic missing writes or assume that adding `dcci` will fix it.

This also fills the gap in `mutex_dcci` §5, which called for a construction using “another observer”; **in this construction, `dcci` also had no observable effect**.

## 8. `pl.get_spr()` Cannot Read the Squeeze Byte Count, and Its Value Is Not Constant

Official documentation and the docstring both say that AR stores the total valid-byte count from `vf.squeeze`. Measured:

| Variable | AR value |
| --- | --- |
| Valid element count 0/1/7/8/16/30/31/32/63/64 | All the same value |
| FP32 (64 lanes) / FP16 (128 lanes) | Same value |
| Before / after the section containing squeeze | Same value |
| **No VF instructions at all** | Same value ← control case |

Yet that value **depends on what ran before**: **128** when this example runs alone, and **0** after the whole chapter runs.

An assertion of `== 128` accepts the isolated process but rejects the combined run, exceeding the measured invariant. The criterion checks that “all readings are equal” within each process, including the no-VF control; it does not infer a fixed value or a squeeze byte count. This illustrates why [combined-run diagnosis](../verification/board-runs.md#numerics-and-races) must inspect the oracle as well as synchronization.

Why this route has no way forward: `pl.get_spr()` is **rejected during parsing** inside `@pl.vector_function`, while `vf.squeeze` can only be written in a VF body. **The place that can write squeeze cannot read AR; the place that can read AR cannot see the value written by squeeze.**

## 9. `dim=` Names the Preserved Axis, Opposite to NumPy

| | pypto_pro | NumPy equivalent | Output shape |
| --- | --- | --- | --- |
| Row reduction | `dim=0` | `axis=1` | `[M, 1]` |
| Column reduction | `dim=1` | `axis=0` | `[1, K]` |

Corroboration: a `dim=1` reduction is called `block.col_max` in IR.

**The third parameter means different things in the two forms**: in an elementwise call it is the right operand (Tile or Scalar); with `dim=` it is workspace (Tile only, same shape as src).

For this shape, adding or omitting `layout=pl.DN` (present on the row-reduction output Tile in the ST) makes no difference. The check runs both variants and asserts equality; it does not establish when layout is required.

## 10. `pl.const` Checks Syntax Only and Rejects Even Constant Names

```python
if not isinstance(value_node, ast.Constant) or not isinstance(value_node.value, (int, float)):
    raise InvalidType("pl.const() first argument must be a numeric literal", ...)
```

`scalar_ops.py:214`. Therefore `pl.const(CLAMP_LOW, FP32)` fails even if `CLAMP_LOW = -0.5` is defined at module scope. **Consequence: the clamp bound cannot share one name between the kernel and reference.**

**This is the third time in this repository that a check accepted only AST literals:**

| Location | Symptom | Type |
| --- | --- | --- |
| `pl.gemm_mx` `k % 64` guard | A constant name **bypasses** the check. | False negative |
| First argument to `pl.const` | A constant name is **rejected**. | False positive |
| `target_memory` for a cross-core buffer | An alias is **rejected** (`_cross_core_scanner.py::_tile_type_memory`). | False positive |

The same implementation habit (reading AST rather than evaluating it) produces two opposite outcomes.

One related rule: **`TileType` cannot be a closure variable** (`_expr_evaluator.py:266`); construct it inside the kernel body.

## 11. Changing `preload` Must Not Change the Result

For preload values 0/1/2/3, all four outputs were **pairwise bitwise identical** and also matched the reference. The docstring in `config.py` says 0 is the starting point to “first verify the serial version”; the check turns that statement into a test that fails if violated.

This check is not about an API return value; it verifies that **the pipeline transform preserves semantics**, a prerequisite for `pl.pipeline` as a whole.

Each of the four invalid `PipelineConfig` values (`-1` / `1.5` / `True` / `()`) has its own exception and useful message. `True` is checked specially (`isinstance(True, int)` is true; without an explicit check, it would be accepted as 1).

## 12. Layout Legality Is Checked at **Three Layers**

| Layer | Location | Question | Message quality |
| --- | --- | --- | --- |
| 1. Operator operand | `_ir_load` / `_ir_store` and operand checks in `block_ops.py` | Which space contains the dst of `load` or src of `store`? | Good; it says where it belongs. |
| 2. Tile declaration | `_validate_tile_type_params` / `_validate_tile_layout_alignments` / `_apply_default_layout` in `block_ops.py` | Can this layout/shape be declared? | Good; includes allowed sets and defaults. |
| 3. Data path | Three tables such as `_A5_LOAD_COMBOS` | Can this `(space, layout, dtype)` combination be transferred? | **Only says “unsupported.”** |

Check the layer that rejects the actual form. A data-path table omitting dtype cannot establish a legal transfer; a layout such as `Mat(NN)` can fail at declaration before that path is tested; an invalid operand space can fail even earlier. Match each negative control to its own diagnostic layer.

Tile declaration table (A5, FP16):

| Space | Default | Declarable layouts |
| --- | --- | --- |
| `Mat` (L1) | `NZ` | `NZ` `ZN` `ND` `DN`, **plus `ZZ` `NN` only for `dtype=FP8E8M0`** |
| `Left` (L0A) | `NZ` | `NZ` |
| `Right` (L0B) | `ZN` | `ZN` |
| `Acc` (L0C) | `NZ` | `NZ` |
| `Bias` | `ND` | `ND`, and **exactly one row is required** |
| `Scaling` | `ND` | `ND` |
| `ScaleLeft` | `ZZ` | `ZZ` |
| `ScaleRight` | `NN` | `NN` |

**Collapsing away the dtype axis gives only necessary conditions:** for FP16, `GM → Mat` supports only `DN→NZ`, `DN→ZN`, `ND→NZ`, and `NZ→NZ`; the extra combinations `ND→ND`, `ND→NN`, `ND→ZZ` seen after collapsing all come from 8-bit / 4-bit rows. Keep both tables; each answers a different question—**combining them would repeat the same mistake**.

`ZZ` / `NN` on `Left` / `Right` / `Acc` are rejected by **a different rule** and with **a different message** (`do not support ZZ layout on 'a5'` rather than `require layout in {...}`), because the allowed set for `Left` includes A3's default `ZZ`. This is a trace of two hardware generations sharing one code path.

`set_mm_layout_transform` was checked only to show that it is accepted and does not change the result of one ordinary transfer; only a matmul that accumulates along K reveals the transposed direction. **No layer rejects it inside a vector section.**

## 13. `debug_probe`: Printing, Assertions, and Their Boundaries

**These four constraints determine how the tests in later examples are written**, so they are collected here. See [`examples/basic_api/support/debug_probe/`](../../examples/basic_api/support/debug_probe/test.py) for reproductions and a table of four pitfalls.

| Fact | Consequence |
| --- | --- |
| Printing cannot be used inside an `@pl.vector_function` body (`_call_parser.py:1159`, rejected during parsing). | To inspect a VF register, `store_align` it to a Tile, then call `dump_data` **outside** the body. |
| Device printing is not flushed by `torch.npu.synchronize()`. | To **assert** printed content, read the child process's stdout. |
| Each print point is multiplied by the number of cores (56 copies for one point). | Two points in a probe produce more than 400 lines, burying pytest's summary line (see [verification/board-runs.md](../verification/board-runs.md) §1). |
| `pl.pto_assert` only logs and **does not abort**. | It is not a safety net. The source docstring says “abort if condition is false”; official documentation and measurement both show logging only—when the three evidence sources disagree, state which one is authoritative. |

One related point: only `pl.trap` unconditionally terminates execution; **give it its own test shape**, not the same process as other cases.

## 14. `dtype_carriers`: Dtype Carriers and Transfer Table

Bit-width groups for each dtype, formulas for 69 combinations, the `HF8→uint8` mapping, and three independent confirmations for 4-bit formats are all in [`examples/basic_api/support/dtype_carriers/`](../../examples/basic_api/support/dtype_carriers/test.py). This page keeps only cross-example findings—**four constraints that affect how other chapters declare tensors**:

| Fact | Affected |
| --- | --- |
| **Host tensor dtype is not checked at all** (`jit.py:2326` compares only byte count): declarations can disagree completely with the actual dtype for 11 dtypes, silently. | All chapters. An 8-bit parameter may receive a tensor with 4 bytes per element → 64 bytes are transferred while the following 192 bytes remain untouched, with no error. |
| Any same-bit-width pairing is allowed, so declaring GM as `INT8` and Tile as `HF8` is legal. | The HF8 section of `fixpipe` relies on this to read the data back. |
| The static last axis of sub-byte dtypes counts logical elements; Host uses packed width. | A hot/cold inconsistency was fixed; see [contract](../diagnostics/defects/subbyte-static-dim-order-dependent.md). |
| `torch_npu` cannot create tensors for three dtypes (this is not a pypto limitation). | Host-side setup in all chapters. |

## 15. In a Mixed Kernel's Vector Section, `get_block_idx` Is an AIV Thread Index; Divide by `get_subblock_num` for the Core Index

Board evidence from 2026-09-29 (Ascend950PR, pypto board build `8faa853`). In a mixed cube+vector kernel, `pl.get_block_idx()` read in the vector section is not the physical core index; it is the **flattened AIV thread index**. Each physical core corresponds to two consecutive values, and `get_subblock_idx()` is exactly that index modulo 2. Compute the core index as:

```python
core_id = pl.get_block_idx() // pl.get_subblock_num()
```

The official `cube_vector_fusion.md` example `matmul_softmax_kernel` contains this line, but no documentation states it as a rule. Diagnostic signature when division is omitted (highly useful for diagnosis): roughly 70% of elements are wrong, large parts of output retain their initial values (even threads write only the first half of each core's range, odd threads write only the second half), error magnitude looks random from tens to 1e33, and **every kernel variant fails in exactly the same way** (ruling out a synchronization bug unique to one variant). This is in the same family as §5 (`get_subblock_num` depends on whether there is a cube section) and §6 (the two index APIs).

**Launch has different semantics; do not extrapolate from this rule:** for a mixed kernel, `X` in `kernel[None, X]` is the **number of execution groups** (one AIC + two AIVs per group; official `kernel_function.md` “Meaning of blockDim”, `LimitLaunchBlocks` in `pypto_launch.h` does not divide the request by 2; all three evidence sources agree, board-verified 2026-09-29). “Vector-side index is thread-based” ⇏ “launch count is thread-based”—passing `2×core_count` to match thread count really launches twice as many groups, with extra ghost groups whose `core_id ≥ core_count` writes past `out` into GM rows `[M, 2M)`. The first call may still return correct output (real groups filled `out`), while an adjacent tensor is silently changed or the device raises 507015 (the location drifts with `out` size). See [api/crosscore-sync.md](crosscore-sync.md) “Mix-kernel block_dim is a group count” for the full diagnosis and address accounting.
