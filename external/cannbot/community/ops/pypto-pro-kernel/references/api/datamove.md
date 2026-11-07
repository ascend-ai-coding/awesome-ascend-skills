# Measured datamove behavior

**All 8 examples ran on hardware** (2026-09-28, Ascend950PR). Examples are in [`examples/basic_api/datamove/`](../../examples/basic_api/datamove/).

The contract table is in [`tooling/example-coverage.md`](../tooling/example-coverage.md).


## 1. Roles of the eight examples

`gm_ub_roundtrip` (two transfer API families), `dma_pad` (padding regions), `init_and_insert` (in-place initialization and insertion), `move_sync` (cross-pipe and inter-core synchronization), `mutex_dcci` (manual locking and cache cleanup), `strided_view` (strided views and pointer arithmetic), `tile_view` (three shape layers), and `ssbuf_struct` (passing structured metadata to cube). The sections below follow this order.

For API coverage per example, see [`tooling/example-coverage.md`](../tooling/example-coverage.md); for the reverse index, run `python scripts/check.py surface`. Current counts come from `pytest examples/basic_api/datamove -q`.

## 2. The two offset APIs differ in units, not age

`pl.load` / `pl.store` offsets are **element** coordinates; `pl.load_tile` / `pl.store_tile` offsets are **tile-block indices**, multiplied internally by tile shape (the `load_tile` docstring states: “Offsets are in tile-block units, internally multiplied by tile shape”).

A block index can express only the start of the nth full block. When a tensor does not divide evenly, the last block's start is not a multiple expressible by any block index; use absolute offsets with `set_validshape` there. **Both families must be understood**, which is why `gm_ub_roundtrip` compares them bit for bit: if the framework's multiplication is wrong, the difference is a whole block rather than a few elements, giving an appropriately coarse criterion.

### Higher-rank GM to a two-dimensional Tile: map axes first

For ordinary GM Tensors, `pl.load/store` `offsets` has length equal to Tensor rank, and each entry is an absolute element coordinate on its axis; the Tile remains two-dimensional. By default, GM's last two axes become Tile rows and columns, while offsets fix the other axes. Specify a plane from other axes explicitly with `order=[row_axis, col_axis]`. First map “GM axis → Tile axis → compute axis”; there is no need to invent a three-dimensional Tile for a batch axis or flatten GM and recompute strides. The MX scale phase axis has its own default rules; this ordinary-Tensor rule does not apply to it.

Sources: official `memory_data_movement/load.md`, `store.md`, and `_api.py::load/store`; a runnable example at behavior version `872df763f` is [batched_transposed_gemm](../../examples/algorithms/batched_transposed_gemm/). Default axis mapping and where transposition occurs are separate decisions: load `order` and gemm `trans_a/trans_b` govern their own boundaries; avoid accidentally transposing twice.

### Use actual extents at every level of nested tiling

Physical capacity, nominal group length, and current actual length are different quantities. On entering each logical group or outer block, first compute `extent = min(nominal_extent, total_extent - origin)`. Bound the inner register loop by this extent, then compute `active = min(register_width, extent - local_offset)`. Every emitted fragment then satisfies `0 < active <= register_width`. If actual extent is nonpositive, skip that fragment.

Do not run the inner loop to the nominal group length while merely taking min with remaining length inside: the final partial group can keep producing zero or negative active counts alongside out-of-range GM starts. A negative value is not an empty task; do not pass it to valid_shape or a mask. The last store's mask cannot repair earlier illegal transfers. Validation should include an outer group with only a few remaining elements while its nominal length still spans several register blocks.

## 3. `fillpad` requires three pieces in three places; omitting one leaves valid data surrounded by garbage

| | Where to specify it | Responsibility |
| --- | --- | --- |
| `pad=pl.TilePad.zero / max / min / null` | On the **destination `TileType`**, at declaration | Fill value |
| `pl.set_validshape(src, [r, c])` | At runtime, on the **source** | Which region is valid |
| `pl.fillpad(dst, src, mode=...)` | One instruction | Actually write outside the valid region |

`TilePad` alone **does nothing**. The example includes `pad_declared_but_not_filled` as a control: it first fills all UB with known nonzero values, then omits `fillpad`. Without preinitialization, residue that happens to be 0 would match `zero`, letting the test pass for the wrong reason.

**`pad=null` with `fillpad` is rejected at parse time** (`out_memory.cpp:66`), consistent with the ISA `static_assert` (`pto/npu/a5/TFillPad.hpp:118`). pypto rejects it earlier and includes a source line. **This is not a defect.**

**Version boundary for INPLACE padding with fixed valid extents**: at behavior version `872df763f`, CANN 9.2.0, an INPLACE fill from a source with `pad=zero` and fixed `valid_shape=[1,65]` generates `__block_fillpad_src_alias_type_*` retaining static valid dimensions but invokes a runtime constructor with `GetValidRow()/GetValidCol()`. BiSheng reports `requirement '1 == DYNAMIC && 65 == DYNAMIC' was not satisfied`. Python `_ir_fillpad` selects `fillpad_inplace`, and C++ `BuildNullPadSourceAliasCCE` creates this alias. This is a measured failure of that generation path, not general invalidity of fixed valid extents. For the same error, make only the padding source's valid dimensions dynamic and assign them before transfer; other fixed-valid-region outputs in the same kernel may remain fixed. The dynamic-source revision passed tail-block and multicore hardware tests; recheck the generation path when changing versions.

## 4. The three `FillPadMode` values differ in the destination, not how the source is narrowed

| | `dst` shape | `dst` address |
| --- | --- | --- |
| `NORMAL` (default) | Same as `src` | Different from `src` |
| `EXPAND` | **Larger than `src`** | Different from `src` |
| `INPLACE` | Same as `src` | **Same address as `src`** |

All three modes **narrow the source identically**, using `set_validshape`. Treating `EXPAND` as “smaller source tile shape, no `set_validshape` needed” leaves the source without valid-region bounds and reads residue. ST `test_pro_fillpad.py::fillpad_expand_dynamic_cce_kernel` uses an `8 × 8` source with `set_validshape(src, [5, 7])` and an `8 × 16` destination: **the destination is what changes**.

The `EXPAND` / `INPLACE` examples require both correct valid-region bounds and the synchronization in §5. A failure with either condition missing does not establish an unsupported padding mode.

## 5. `auto_mutex=True` does not manage bare `make_tile`

**A repeatable enum-like pattern can come from missing synchronization.** The `dma_pad` probe produced `zero`→`-2147483648`, `max`→`0`, and `min`→`2147483647` across four same-state runs. Python, C++ and ISA enum mappings agreed; the kernel paired `@pl.jit(auto_mutex=True)` with bare `pl.make_tile`:

> `auto_mutex` infers dependencies from `pl.make_tile_group` `mutex_ids`. Bare `make_tile` provides no mutex id, so **no synchronization is inserted and no error is raised** between the MTE2 load, V fill, and MTE3 store.

**This can be established without hardware**, directly in parsed IR:

| Form | Synchronization ops |
| --- | --- |
| `auto_mutex=True` + `make_tile_group(mutex_ids=[0, 1])` | **6** (`mutex_lock_dyn` / `mutex_unlock_dyn`, 3 pairs each) |
| `auto_mutex=True` + bare `make_tile` | **0** |
| `@pl.jit()` + manual `sync_src` / `sync_dst` | **4** |

Inspect parsed IR before attributing a repeated value pattern to the framework. Repetition without changing preceding state is not independent verification; see [`verification/evidence.md`](../verification/evidence.md) #16 and [`diagnostics/validation-traps.md`](../diagnostics/validation-traps.md) #7. The deterministic source gate is `check.py`'s `auto-mutex-needs-tile-group`; its scope and limitations remain in [tool usage](../tooling/checks.md).

## 6. Row-major Tile alignment

A `BLayout::RowMajor` tile requires `Cols * sizeof(DType) % 32 == 0`. A `4 × 4` INT32 tile (16 bytes) is rejected **during construction** by `_validate_tile_layout_alignments`, before op-level checks.

## 7. `move_sync` / `mutex_dcci`: two synchronization mechanisms

| Example | Coverage | On hardware |
| --- | --- | --- |
| `move_sync` | `pl.system.sync_src` / `sync_dst` / `sync_all`, including four parse-time rejection categories |
| `mutex_dcci` | `pl.system.mutex_lock` / `mutex_unlock` / `dcci` (four parameter forms) |

The public `pl.system.*` surface has 15 APIs, of which 9 are covered. **The 6 uncovered APIs fall into two categories**: five per-pipe `bar_*` APIs (the next section explains their exclusion) and `set_mm_layout_transform` (layout behavior, owned by `layout_table`).

**Two synchronization mechanisms are equivalent but distinct:**

| Form | Parsed ops | How direction is expressed |
| --- | --- | --- |
| Manual flags | `sync_src_dyn` / `sync_dst_dyn` | In `set_pipe` / `wait_pipe` |
| Manual mutex | `mutex_lock_dyn` / `mutex_unlock_dyn` | Order of ownership of the same `mutex_id`; **direction is not visible in the arguments** |
| `auto_mutex=True` + groups | Same as above, inserted by the framework | Inferred by the framework from `mutex_ids` |

Both examples use IR comparisons to pin which form follows which path. Numerical criteria cannot establish this because both chains compute correctly.

**The three id pools are independent**. Their sole definitions are `MAX_EVENT_ID` / `MAX_FLAG_EVENT_ID` / `MAX_MUTEX_ID` in `system_ops.py`: flag `event_id` `[0, 7]`, cross-core `event_id` `[0, 15]`, and `mutex_id` `[0, 31]`.

**Legal `sync_src` / `sync_dst` paths are separate for AIC / AIV** (`_A5_AIC_SYNC_WAIT_PIPES` / `_A5_AIV_SYNC_WAIT_PIPES` in `system_ops.py`). This yields an unusual criterion: **the same `MTE2 → V` line is valid in `section_vector()` but rejected in `section_cube()`**, because cube has no V pipe. The message says `on AIC` and lists the AIC table. Validity depends on which side the code occupies.

**Two explicit skips, not claims of successful verification**: `sync_all(MIX)` / `sync_all(AIC_ONLY)` are not run on hardware in a pure vector kernel; neither is `mutex_lock` without `mutex_unlock`. Both can leave pipe queues blocked, making experiments on a shared machine disproportionate in cost to benefit. Both **parse-time** facts are pinned.

## 8. Why `bar_*` is excluded: it is an intra-pipe barrier, not a cross-pipe barrier

`bar_mte2()` alone does not order a later V consumer. In the 2026-09-28 measurement, the same load → add → store chain, with fixed preceding state immediately following a kernel that dirties UB, ran three times per form:

| Synchronization between operations | Immediately after dirtying UB | Alone in a cold process |
| --- | --- | --- |
| Two `sync_src` / `sync_dst` pairs | **3/3 correct** | Correct |
| Two `bar_all()` calls | 3/3 correct | Correct |
| `bar_mte2()` + `bar_mte3()` | **3/3 wrong** | Correct |
| No synchronization | **3/3 wrong** | Correct |

Per-pipe `bar_*` behaves like no synchronization; `bar_all` works because it drains **all** pipes. **Every entry in the last column is correct**: even no synchronization succeeds in a cold process. That is why this chapter's criteria do not rely on numerical output.

Real usage appears in `easyasc` (the preceding generation on the same hardware): `bar_mte3()` at the end of a loop body serializes MTE3 behind itself; `bar_all()` is used where automatic synchronization analysis cannot see a dependency, with the comment **“The barrier drains it”**. Both drain pipes rather than establish cross-pipe ordering.

**Reason for exclusion**: an example called “synchronize with `bar_*`” would teach the wrong behavior. Explaining the correct use, serialization within a pipe, requires a scenario where automatic synchronization cannot see a dependency; that is outside this chapter's subject. Facts are in [validation traps](../diagnostics/validation-traps.md) §8. Cube-specific barriers inside Vector are now rejected at parse time; see the [diagnostic entry point](../diagnostics/messages.md#parse-error-localization-and-regressions).

## 9. `init_and_insert`: `init_output` occupies UB `[0, 8K)`

The two APIs use different offset units: `init_output` uses a **one-dimensional element range**, flattened row-major with no shape concept; `insert` uses **two-dimensional element coordinates**. A table of three conditions documented in docstrings but unchecked at parse time, and two explicit reasons for skipping hardware runs, are in [`examples/basic_api/datamove/init_and_insert/`](../../examples/basic_api/datamove/init_and_insert/test.py). For the coordinate system when the destination is declared ZN, see §16: **using destination coordinates compiles but silently computes wrong values**.

The cross-example concern is that **this affects every kernel placing tiles at low UB addresses**:

| Fact | Consequence |
| --- | --- |
| The temporary `init_output` tile is hardcoded at UB `0x0`, occupying `(256*32)/sizeof(T)` bytes (8 KB for b32), and **fills the entire width regardless of `size=`** | Boundary verified on hardware: `0x1000` is overwritten; **`0x2000` is the first safe address** |
| **No barrier** follows its `TSTORE` | Overlapping addresses silently write input data to GM (6 failures in 12 runs) |
| `make_tile_group` fixes it by emitting `get_buf`; **`auto_mutex` does not** | Toggling only `auto_mutex` produced byte-for-byte identical C++; the group changes the synchronization |

Gate: `scripts/_checks/init_output.py`. Mechanism, seven-row measurement table, and framework-side locations: [`diagnostics/validation-traps.md`](../diagnostics/validation-traps.md) #11.

## 10. `init_output` requires a vector section, even an empty one

It internally fills temporary UB via V, then transfers through MTE3. Without `pl.section_vector()` in the kernel, **compilation fails**, reporting a C++ `PIPE_V` range error and `copy_ubuf_to_gm_align_v2 does not support the given target feature`. Both say the core has no V pipe, but **do not reveal the connection to sections**.

| Form | Result |
| --- | --- |
| Only `init_output` | **Compilation failure** |
| `init_output` + empty `section_vector` | Passes |
| `init_output` + `make_tile` + nonempty section | Passes |

Parsing passes completely, so `parse_target_program` cannot detect it. Both rules are included in `references/api/constraints.md`.

## 11. `strided_view`: offsets follow strides too, and the sole justified use of `bar_*`

| Example | Coverage | On hardware |
| --- | --- | --- |
| `strided_view` | `pl.make_tensor`, `pl.make_ptr`, `pl.addptr` |

**A view's stride applies to every access through it, including `pl.load` / `pl.store` offsets.** Two forms measured with the same stride (`2 * COLS`):

| Form | Rows read |
| --- | --- |
| `pl.load(tile, view, [1, 0])` | Rows 2, 4, 6, …: **the offset is multiplied by stride** |
| `pl.make_tensor(pl.addptr(pl.make_ptr(x), 1 * COLS), …, [s, 1])` | Rows 1, 3, 5, … |

**Read both criteria together**; either one alone can give a false sense of understanding. The ST comments document this at the source (`test_strided_gather_view.py:105-108`). `addptr` counts **elements**.

`dtype=` performs **reinterpretation**, not conversion. The criterion uses numpy `x.view(np.int32)`, not `astype`: the former reads `1.0` as `1065353216`, while the latter reads it as `1`. Choosing the wrong reference function fails for the wrong reason and invites misplaced suspicion of the framework.

### Host Tensor strides and boundary guards

An ordinary ND `pl.Tensor[[...], dtype]` parameter does not automatically inherit the row stride of an incoming PyTorch view. The current launcher passes `arg.data_ptr()` and required shape parameters; without an explicit TensorView stride, C++ `CCECodegen::BuildTensorStrideExpressions` computes contiguous row-major strides from shape. Thus, `storage[:, left:right]` views with identical shapes may have different physical row strides yet pass shape checks. If noncontiguous inputs/outputs are part of the task contract, express actual strides explicitly using `pl.make_tensor(ptr_or_tensor, shape, stride)` and validate them separately; the PyTorch view object alone does not convey this information.

The view-stride explanation above applies to DMA transfers. Scalar `A[i,j]` has a separate linearization path; correct `pl.load` behavior does not establish correct SIMT subscripting. For noncontiguous scalar accesses, first read [current subscript limits and explicit addressing](simt.md#13-scalar-indices-and-custom-physical-strides).

To add boundary sentinels to ordinary contiguous output, a test can take a contiguous slice from a larger one-dimensional backing allocation, preserving the kernel's original shape, strides, and zero-based logical coordinates:

```python
# FP32 example: the 64-element prefix preserves a 256-byte aligned offset.
numel = math.prod(shape)
guard = 64
storage = torch.full((numel + 2 * guard,), sentinel, dtype=torch.float32, device=device)
out = storage[guard:guard + numel].view(shape)
assert out.is_contiguous()
# Launch the original kernel with out, then compare its values and both guard slices.
```

`data_ptr()` already points to the slice's first element; the kernel must not add the guard offset again. Do not enlarge the kernel output shape, introduce border parameters, or rewrite GM offsets just to pass this test: those change the contract under validation. A two-dimensional view with a halo on four sides is a separate stride use case and cannot replace validation of ordinary contiguous output. Prefix/suffix guards detect out-of-range writes, while the mathematical reference detects missing or displaced writes inside the range; both are required.

Evidence: the explicit-stride contract of official make_tensor (`pypto/docs/zh/api/pro_api/SIMD-API/resource_management/make_tensor.md`), argument packing in Python `runtime/jit.py`, and the C++ stride-generation path above. Minimal transfer tests at behavior version `872df763f`, CANN 9.2.0, Ascend950PR verified three cases: ordinary parameters do not inherit host row strides; explicit strides work; contiguous slices with nonzero storage offsets work. This does not mean all noncontiguous views are unsupported.

## 12. Overlapping writes: program order is not guaranteed even within one pipe

Two `pl.store` calls using MTE3 do not by themselves establish a deterministic winner for an overlapping GM region. The framework **does not check aliasing** between those views; the recorded control shows both completion orders:

| Between the two stores | Which write owns the overlap (20 runs = 4 processes × 5) |
| --- | --- |
| Nothing | **Later write 14 times, earlier write 6 times**; even consecutive runs in one process vary |
| One `pl.system.bar_mte3()` | **Later write in all 20 runs**, consistent cold/hot/with warmup |

The unbarriered “later write wins” criterion fails in 6 of 20 runs, so a few passing repetitions cannot justify that assertion. Vary preceding state and process boundaries as in the synchronization controls in §5.

Resolution: **exclude the nondeterministic overlap from the criterion**, asserting only A's exclusive segment, B's exclusive segment, and the tail sentinel. Use a separate kernel with a barrier to assert that the later write wins.

**This also identified the sole justified use of `bar_*` here.** Flags cannot help: `sync_src` / `sync_dst` require `set_pipe != wait_pipe`, so MTE3→MTE3 is rejected during parsing (a dedicated kernel in the example pins this). Thus, `bar_mte3()` is the only choice here, and the requirement is exactly “no parallelism within the pipe.” This agrees with §8: **`bar_*` drains one pipe; flags establish order between two pipes**. Recorded in [validation traps](../diagnostics/validation-traps.md) §9.

## 13. `tile_view`: three shape layers and a shared cursor

| Example | Coverage |
| --- | --- |
| [tile_view](../../examples/basic_api/datamove/tile_view/) | `pl.make_tile`, `pl.make_tile_group`, `pl.set_validshape`, `pl.reinterpret`; shape behavior and VF/FIX consumer probes |

**The three layers coexist**: physical shape (`TileType(shape=)`, compile time), valid region (`set_validshape`, runtime), and reinterpreted view (`reinterpret`, declaration time, emitting no instruction).

### Verify a VF consumer's address map after reinterpretation

`pl.reinterpret` changes a declaration over existing bytes; it performs no transpose or format conversion. Its validity depends on the actual consumer. A layout reinterpretation used by a Cube operand does not establish an equivalent VF access: the ordinary two-coordinate `vf.load_align(tile, [r, c])` offset is flattened as `r * tile.shape[1] + c` before the contiguous load. Declaring `layout=pl.ZN` does not turn that load into a column gather from ND storage. This lowering is in `backend_cce_vf_ops.cpp::ResolveOffsetArg` and was checked with coordinate-distinct data on `75a6c2600`, CANN 9.2.0.

For ND storage `[R, C]`, column `c` at row `r0 + lane` lives at `(r0 + lane) * C + c`; compare that expression with the address actually generated by the chosen load. Use a physical packing/transpose producer or an appropriate indexed/strided load when the desired elements are not contiguous. Prove the conversion through its consumer on a nonsquare panel with distinct row and column values, including the last block. An unchanged first row, uniform inputs, normalized row sums or successful compilation can all hide a wrong axis map.

Keep this layout probe separate from coefficient generation and reduction arithmetic. If VF writes an intermediate tile and then consumes it in the same call, also establish the [vector-memory ordering](vf.md#212-multiple-memory-accesses-to-the-same-tile-inside-vf-have-no-ordering-guarantee); fixing the address map does not establish completion of the preceding stores.

The [tile-view consumer probes](../../examples/basic_api/datamove/tile_view/kernel.py) (`load_reinterpreted_layout` and `make_receive_view`) and their [tests](../../examples/basic_api/datamove/tile_view/test.py) cover both the VF address map and source-driven FIX extent without undersizing physical storage.

### Fixed and dynamic valid regions

When `pl.set_validshape` is needed, leave `TileType.valid_shape` unspecified or declare `[-1, -1]`, then assign it before any transfer depending on that region. If fixed `[r, c]` is declared, use that region directly without a setter; both dimensions must be dynamic template parameters to use the SDK's `SetValidShape`. Physical `shape` and valid region remain independent pieces of information.

`fixed-validshape-mutation` in `check.py rules` prechecks this misuse on traceable direct declarations. When the fixed valid region already meets the need, simply remove the redundant setter; do not change to dynamic templates just to eliminate the error.

Three evidence sources: official `SIMD-API/basic_data_structures/TileType.md` descriptions of fixed, default, and dynamic forms; Python declarations and valid-shape template parameters in C++ `type_converter.cpp`; compilation and hardware tests on 2026-10-06 at behavior version `872df763f`, CANN 9.2.0. The original error from a setter after a fixed shape is `Only Dynamic Valid Shape Support Set Value.` A revision retaining the fixed valid region and removing the redundant setter passed hardware tests. This is not a requirement to make every tile dynamic.

**`set_validshape` limits transfer volume for both load and store** (measured by counting remaining GM sentinels across 128 positions):

| Form | Sentinels remaining | Positions written |
| --- | ---: | ---: |
| No narrowing | 0 | 128 |
| Narrow before `store` | 96 | 32 |
| Narrow before `load` | 96 | 32 |
| Narrow, then `reinterpret` to the same shape | **0** | **128 (narrowing lost)** |

The last row is this example's trap and precisely confirms the docstring's “valid_shape is NOT inherited.” **Rows 2 and 3 have identical outputs but different processes**: the distinction exists in UB and is invisible in GM. Those two criteria are therefore parameterized together to avoid implying two independent sources of evidence.

`set_validshape` and `reinterpret` **both accept an entire `make_tile_group`**: the former expands in IR into two calls, `set_validshape(_tg_group_tiles[0], …)` and `[1]`; the latter returns a new group and **retains mutex_ids** (locks in IR still reference the original tuple).

**The `.next()` rotation cursor is shared between the original group and its reinterpreted group.** After `group.next()` takes slot 0, `wide.next()` takes slot 1 (`_bufidx_0` / `_bufidx_1` in IR). Transferring through the latter does not read the former's freshly loaded slot. In the hardware counterexample, every 16 elements had 8 correct values followed by 8 zeros; the correct 8 were **UB residue** from the preceding kernel's `4 × 8` write to slot 1. For this view-declaration test, **compare IR**, asserting changed shape and unchanged mutex tuple rather than treating partially matching bytes as a valid transport result.

**Vec cannot reach the `reinterpret` base-address alignment check**: a counterexample would need `addr=0x0004`, but tile creation first rejects it with `Tile address 0x00004 is not 32-byte aligned for memory space Vec` (`_validate_tile_addr_alignment` in `block_ops.py`). Vec's 32-byte alignment is a multiple of every element width (up to 8 bytes), so **every valid Vec address inherently satisfies the new dtype's alignment**. Record it as unreachable, not nonexistent.

## 14. `ssbuf_struct`: SSBUF slot width measured and checked through predictions

| Example | Coverage | On hardware |
| --- | --- | --- |
| `ssbuf_struct` | `pl.struct`, `pl.struct_array`, `pl.make_tuple`, `pl.ssbuf_store`, `pl.ssbuf_load` |

**This completes the chapter: all 8 / 8 datamove examples ran on hardware.**

**Neither `offset` units nor slot width are documented** (the docstring says only “offset: SSBUF offset,” and ST uses only `0`). Measured conclusion:

> **`offset` counts bytes, but each SSBUF field occupies 8 bytes**, regardless of its dtype width. A struct of N INT32 fields occupies `8 * N` bytes; the second struct's offset must be `>= 8 * N`.

Sweeping 5 offsets (4 / 8 / 12 / 16 / 24…) for a three-field struct produced this model, **then four fields were used for prediction**:

| Four-field `offset` | Prediction | Measurement |
| ---: | --- | --- |
| 16 | `[1, 2, 11, 12]` (overwrites slots 3 and 4) | Match ✓ |
| 24 | `[1, 2, 3, 11]` (overwrites only slot 4) | Match ✓ |
| 32 | Correct | Correct ✓ |

**Seven configurations matched bit for bit; four predictions were written before measurement.** This justifies asserting the corruption pattern in this example: at offset 16, `first.z` must read `second.x`. It will fail under two changes: a different slot width, or the framework beginning to reject this overlap.

**Two API boundaries:**

- `pl.make_tuple` fields **cannot be assigned** (“the target is not a struct”): it only names existing IR values and does not generate a C++ struct. **The SSBUF path therefore requires `pl.struct`.**
- The `pl.struct` docstring listed “Legacy: anonymous form,” which is **unconditionally rejected at parse time**. The docstring was fixed and two UTs added in `pypto/` `eccb0e7c3`.

**Three explicit skips**: total SSBUF capacity (would require increasing offsets until failure, with unknown consequences of out-of-bounds SSBUF writes), slot width for non-INT32 fields (8 bytes can hold INT64, but it was not verified), and mixed-width structs. The model is established only for INT32.

**Isolate potentially faulting offsets in separate processes.** A device error at one offset can invalidate later cases in that process without executing them. Follow [device-fault isolation](../verification/board-runs.md#isolating-device-faults) and check recovery before continuing; cascade failures are not independent offset results.

## 15. Transfer-op pipe ownership: the complete table for layer C synchronization

When hand-writing `set/wait_cross_core` at layer C, choose the pipe of the op actually accessing the data. Transfer-op pipe ownership **is determined by framework mapping functions and is not always intuitive** (hardware counterexample: Mat→Scaling move uses FIX, not MTE1; choosing the wrong pipe causes “not ready on the first launch, later passing accidentally on residue from the previous launch”; see api/quantization.md).

**Cross-check two source owners**: legal combinations are owned by `ir/op/block_ops.py::_supported_move_paths`, the parse-time allowlist; combinations outside it immediately raise `NotSupported`. Pipe ownership is owned by `language/parser/_op_pipeline.py` (`get_move_pipe` / `get_store_pipe` / `get_op_pipe`, with the latter querying the CCE backend registry). **`get_move_pipe` has a broader domain than the allowlist**: it returns pipes even for illegal combinations (for example V for Mat→Vec). Those branches are dead code and do not establish that a combination is available.

**All legal `pl.move` combinations and pipes** (ten allowlisted paths × `get_move_pipe`):

| Source | Destination | pipe |
| --- | --- | --- |
| Acc | Vec / Mat | **FIX** |
| Mat | Left / Right / Bias / ScaleLeft / ScaleRight | **MTE1** |
| Mat | Scaling (FB) | **FIX** (`get_move_pipe` fallback; Scaling is not explicitly listed and lands here) |
| Vec | Mat | **MTE3** |
| Vec | Vec | **V** |

**There is no direct Mat→Vec (L1→UB) transfer**: it is absent from the allowlist (`NotSupported` at parse time, with all ten supported paths listed in the message). A5 hardware lacks this path, and the PTO instruction library's `pto/npu/a5/TMov.hpp::TMOV_TILE_IMPL` Mat-source dispatch likewise has only six destinations, Bias/Scaling/Left/Right/ScaleLeft/ScaleRight, with no Vec branch. An unmatched constexpr dispatch leaves an **empty function body**; the a5 version lacks the a2a3 version's entry static_assert fallback. Bypassing the parser to force generation therefore compiles without error and transfers nothing. Returning L1 data to UB requires a GM detour (store→load) or rearranging the dataflow.

**`pl.store` / `store_tile`** (`get_store_pipe`): Acc source → **FIX**; all others → **MTE3**.

**Other common ops** (`get_op_pipe` queries the registry; checked/hardware-verified): `pl.load` (GM→L1/UB) → **MTE2**; `pl.insert` (UB→L1) → **MTE3**.

**No pipe used**: `set_validshape` is descriptor-only, changing metadata without touching data (`_DESCRIPTOR_ONLY_OPS`). Both auto_mutex and synchronization insertion skip it, and layer C needs no synchronization for it.

The table follows framework versions. After a version change, cross-check `_supported_move_paths` (legal combinations) and `get_move_pipe` (pipe ownership), trusting pipe values only for allowlisted combinations. The common combinations in crosscore-sync's “Pipes at both endpoints” section are a subset of this table.

## 16. `pl.insert` into a ZN-declared destination: offsets use NZ source coordinates (hardware evidence 2026-09-30)

When `pl.insert` writes an NZ source tile into a Mat destination declared **ZN** (the usual return path through a transpose view of the same bytes, such as flash_attention's `P^T` handoff), offsets use **the source's NZ coordinate system**, not the destination declaration's ZN coordinates. To assemble `[2K, Q]` in two halves, use `[0, off]` / `[KH, off]` (NZ source frame). Using destination-frame `[off, 0]` / `[off, KH]` **compiles but silently computes wrong values on hardware** (err/bound 1287, no diagnostic). The official insert documentation says only “destination coordinates [row, col]” and does not define ZN destinations. The PTO library interprets offsets; neither Python nor C++ source establishes this behavior here. This rule has only numerical evidence, from a controlled hardware comparison of wrong and correct frames. Upstream FA ST uses a square shape (TS=TKV=128) and cannot distinguish the frames. Pin this return path with a wrong-frame variant whose negative control must exceed tolerance. Example: [flash_attention](../../examples/algorithms/flash_attention/).

### Rectangular NZ-to-ZN handover

Keep three shapes distinct: GEMM consumes a ZN Mat operand declared `[M, K]`; its bytes can be assembled as an NZ panel `[K, M]`; for the even, aligned M used here, each of two AIVs can own an NZ source `[K, M // 2]`. Changing the Mat declaration to `[K, M]` also changes the dimensions GEMM sees, so swapping that declaration alone cannot repair an insert-offset rejection.

On framework `51d7dab69`, Python `_ir_insert` calls `_validate_offset_bounds` with the destination's declared shape and does not swap the axes for ZN. A constant row offset `64` therefore fails for declared `[32, 128]` or `[64, 128]`, even though the producer is addressing the K axis of the NZ assembly. The diagnostic says `source shape`, but the checked shape is the destination's. This limits that split-write form; it is not an `M == K` requirement.

For the tested FP16/BF16 thin panels (`M < K`), a full-K panel fits in UB: use one `[K, M // 2]` NZ source per AIV and insert at `[0, sub_id * (M // 2)]`. The [make_rectangular_handover example](../../examples/matmul/transpose_reinterpret/kernel.py) keeps Mat declared `[M, K]`, uses B-layer synchronization at `preload=0`, and passes the result to `pl.gemm(..., splitk=64)`. Its producer uses ND→NZ `pl.move`; a fused VF producer must separately establish its register packing, predicates and full physical store footprint before shrinking the allocation. One register's touched bytes do not shrink with logical M.

The [tests](../../examples/matmul/transpose_reinterpret/test.py) cover `M=32/64`, `K=128`, `N=64`, FP16/BF16, using non-square, position-distinct values and both halves of K in the consumer. On Ascend950PR with CANN 9.2.0 and framework `51d7dab69`, the controlled outputs were exact. This establishes a legal rectangular transport/consumer path, not an arbitrary-size guarantee or a performance result for a complete algorithm. If the full panel does not fit, rederive packing and transfer decomposition rather than inferring a universal dimension limit from this one form.

---
