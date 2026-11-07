# VF contracts and measured behavior

For parameters and defaults, search the [generated API reference](pypto_pro_language.pyi) and verify its [source identity](../tooling/checks.md#generated-api-reference). This page owns the contracts that signatures cannot express: [physical conversion maps](#map-physical-containers-before-composing-conversions), [FP16 widening and predicate reuse](#213-the-fp32-path-for-fp16-source-data-order-preserving-widening-with-unpk-required-reading), [implicit input destinations](#destination-registers-used-as-inputs) and [register bindings/copies](#vf-register-aliasing-and-copying), along with the operation-specific observations below.

Hardware observations retain their individual toolchain and test scopes. Historical results are not current acceptance; record source, toolchain, target and actual output when revalidating. Runnable VF examples are in [`examples/basic_api/vf/`](../../examples/basic_api/vf/).

The contract table (which `vf.*` APIs each example should cover) is in [`tooling/example-coverage.md`](../tooling/example-coverage.md) §8. For the reverse index (which examples use an API), run `python scripts/check.py surface`.

## Keep runtime dispatch out of VF loops

For a performance-sensitive SIMD loop, distinguish a Python factory constant, a runtime scalar uniform across the invocation, and a per-lane vector predicate. They can describe similar logical conditions but compile differently. A loop-invariant runtime scalar is not automatically a compile-time constant.

On Ascend950PR with CANN 9.2.0 and PyPTO `fd1badd7c`, adding first/last-block conditions inside a recurrent VF row/chunk loop changed native code from `VLOOPV2` hardware loops and `VAG` address generation to repeated scalar compares/jumps (`SCMP`, `SCBZI`, `SJUMPI`) and scalar-addressed vector accesses. Paired constant/runtime probes and complete-kernel ablations isolated this effect. This is a measured compiler behavior for those loop shapes, not a promise that every branch is slow or that every branch-free loop receives hardware-loop lowering. Recheck another compiler, target or body in generated native code; a CCE `for` loop alone does not establish its final form.

Use the condition's meaning to choose a comparison:

| Condition | Preferred comparison |
| --- | --- |
| First, middle or last reduction block; a mode uniform for the entire VF call | Dispatch in JIT/stage scalar control and call a VF whose factory captures the mode as a Python constant. Retain separate first+last handling for a one-block reduction |
| A contiguous prefix of active rows | Compare `pl.range(active_rows)` with the per-row `if`. Define inactive output/padding separately when a downstream consumer still reads it; clamp and validate the bound outside the hot loop |
| Each vector lane has different validity | Use VF predicates and selection with the correct lane granularity. Predicating arithmetic does not authorize out-of-range loads or stores |
| A condition is genuinely sequential or cannot be moved without changing the recurrence | Preserve it, inspect the compiled loop and measure a valid alternative. Correctness takes precedence over loop recognition |

For example, a factory capture can select the update body before VF compilation:

```python
def make_update(first):
    @pl.vector_function
    def update(dst, src, rows):
        mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
        for row in pl.range(rows):
            value = vf.load_align(src, [row, 0])
            if not first:  # Python factory constant; not a VF runtime parameter.
                previous = vf.load_align(dst, [row, 0])
                value = vf.add(value, previous, mask)
            vf.store_align(dst, value, mask, [row, 0])
    return update

initial_update = make_update(True)
later_update = make_update(False)
# In a JIT/stage, outside VF: choose initial_update or later_update once per block.
```

The [sparse-attention implementation](../../examples/algorithms/sparse_attention/kernel.py) applies this to normalization and output accumulation. Verify resolved factory constants and loop bounds in generated code as well as branch elimination; a specialization must preserve iteration counts and memory footprints. Cover first, middle, last and single-block execution, fully inactive rows, repeated addresses and changed inputs. When only dispatch moves, compare the same inputs directly as well as using the original evaluator; do not change the summation order, rounding, stores or synchronization to make the comparison pass.

For attribution, hold tile sizes, layout, work counts, slots and `mutex_id` assignments fixed. Inspect the generated VF body and native disassembly, and compare affected Vector time with complete-call time. Unrolling, a worker sweep, or a wholesale pipeline rewrite can hide this cause. The [runtime-control advisory](../tooling/checks.md#static-rules) identifies source sites to inspect; it does not model call-site constants, final compiler optimization or a speedup.

This performance issue does not establish a synchronization defect. `auto_mutex` requires correctly associated tile-group IDs, cross-core handovers require their own completion/reuse proof, and [ordering within VF](#212-multiple-memory-accesses-to-the-same-tile-inside-vf-have-no-ordering-guarantee) still needs appropriate dependencies. Keep those checks separate.

## 1. How squeeze and unalign work together

This section grew from a question: the unalign family is complicated. It proceeds from the problem it solves, through the difference between two chains and a real use case, to four constraints, then three measurements.

### 1.1. The problem unalign solves: writes not aligned to block boundaries

When storing a register to UB, hardware writes an **aligned block** at a time. If the destination is not at a block boundary, or the **length is not a whole number of blocks**, two issues arise: the bytes crossing a boundary span two blocks, and the next write starts at an unaligned address.

The `unalign` approach is to **keep leftover bytes in a register and join them with the next write**. This register is the alignment tracker, created by `vf.unalign_reg_for_store()` (called `ureg` or `align_reg` in the official documentation). It holds bytes that do not yet form a full block; it is not an independent squeeze write position, which also depends on the shared AR state explained below.

One `store_unalign` does two things:

1. Writes the portion that **forms aligned blocks** into the Tile.
2. Stores the **trailing partial block** in `ureg`.

The next `store_unalign` first **concatenates** the leftover bytes in `ureg` with the start of the new register, then writes them. The boundary is joined rather than overwritten. When the loop finishes, the final tail in `ureg` has no later data to join it: **flushing that tail is the sole purpose of `_post`**. Omitting `_post` loses the last few output bytes.

```
Iteration 1: reg ──┬─→ Write aligned portion to Tile
                   └─→ Keep leftover bytes in ureg
Iteration 2: ureg tail ⊕ reg head ──→ Write to Tile
                                    reg tail ──→ Keep in ureg
...
Finalize:    _post: final ureg tail ──→ Write to Tile
```

### 1.2. The difference between the two chains: who knows this write's length

This is the most easily confused part: **two nearly identical call chains differ only in who computes how many bytes to advance this time**.

| | Caller-driven | Data-driven (squeeze) |
| --- | --- | --- |
| Write | `store_unalign(tile, src, align_reg, **stride**, post_update)` | `squeeze_store_unalign(tile, src, align_reg)` |
| Finalize | `store_unalign_post(tile, align_reg, **stride**)` | `squeeze_store_unalign_post(tile, align_reg)` |
| Source of this write's length | **The `stride` you pass** | The preceding `vf.squeeze` produces the valid byte count; **the AR special register also holds the cumulative destination byte offset** |
| Instructions | `vstus`/`vstu` + `vstas`/`vsta` | `vstur` + `vstar` |
| When to use | Each write length is computable **at compile time or from loop variables** | Each write length is known **only at runtime** |

The complete `squeeze` mechanism:

```python
sq = vf.squeeze(idx, hit, gather_mode=pl.SqueezeMode.STORE_REG)
#    ① Pack elements selected by hit into sq's low lanes; zero the rest
#    ② STORE_REG produces the valid byte count for the immediate store and participates in AR chain state
vf.squeeze_store_unalign(out_tile, sq, areg)
#    ③ No explicit offset: the store uses this byte count and AR's cumulative destination displacement
```

**`squeeze_store_unalign` has no `stride` parameter because its advance is not determined by the caller**, rather than merely for API simplicity. Explicit `gather_mode=NO_STORE_REG` does not write AR; in that case, `squeeze` is simply a compaction operator unrelated to unalign.

### 1.3. A real use case: densely packing variable-length filter results

Only one ST in the repository uses this combination: `lightning_indexer/test_quant_lightning_indexer_vf.py`, with 5 squeezes and 5 squeeze_store_unalign calls. It performs **Top-K index collection** for a sparse-attention indexer:

```python
# Pass 1: Collect indices whose scores are strictly greater than the kth-largest score
for i in pl.range(0, vf_loop):
    idx_c = vf.arange(begin_idx + i * 128, dtype=pl.DT_INT16)   # Indices in this batch
    vreg_input = vf.load_align(score_u16_tile, i * 128)          # Scores in this batch
    pout_gt = vf.gt(vreg_input, kth_value, preg_b16)             # Matching elements
    sqz = vf.squeeze(idx_c, pout_gt, gather_mode=pl.SqueezeMode.STORE_REG, dtype=pl.DT_UINT16)
    vf.squeeze_store_unalign(output_idx_tile, sqz, align_idx)    # Store immediately; the cursor advances itself

# Pass 2: Append indices equal to the kth-largest score, sharing the same align_idx
for i in pl.range(0, vf_loop):
    ...
    pout_eq = vf.eq(vreg_input, kth_value, preg_b16)
    sqz = vf.squeeze(idx_c, pout_eq, gather_mode=pl.SqueezeMode.STORE_REG, dtype=pl.DT_UINT16)
    vf.squeeze_store_unalign(output_idx_tile, sqz, align_idx)

vf.squeeze_store_unalign_post(output_idx_tile, align_idx)        # Finalize once, flushing the tail
```

**Why it fits this use case**: the number of matches in each batch depends on a runtime predicate, while output must be dense; the squeeze chain passes the valid length directly. An explicit-stride store chain requires obtaining each batch's length and managing offsets separately. Do not mix the two chains' state mechanisms.

**Pass 1 / Pass 2 share one `align_idx`, with no finalization between them**: the second pass continues at the byte position where the first stopped. Both the UnalignReg cache and AR cumulative offset continue along the same store chain, and are finalized together at the end.

**What these use cases have in common**, beyond top-k:

- **Stream compaction / filtering**: select qualifying elements from a batch and write them densely. Examples include top-k indices, sparse-attention candidates, NMS-style retained boxes, and nonzero collection for sparsification.
- If each write length is already available, prefer considering explicit `store_unalign(stride=…)`; `store_align` is also an option when the destination and valid region satisfy alignment requirements.

`lightning_indexer` has three further uses with the same pattern: finding the first histogram bin `>= bottom_k` during sorting (the number of matching bins is unknown), low-byte histograms after MSB filtering, and fixed-length collection with a `-1` sentinel.

### 1.4. Four constraints, one unchecked by the toolchain

| Constraint | Source | Consequence of violation |
| --- | --- | --- |
| A `vf.squeeze` must write AR before `squeeze_store_unalign` | Official documentation | Reads the previous AR value |
| Pair with `squeeze_store_unalign_post`, placed afterward | Official documentation | Output loses the final few bytes |
| **Alternate `squeeze` and `squeeze_store_unalign`**; never issue two consecutive squeezes | Official documentation: **“otherwise hardware may hang”** | **Device hangs** |
| vf operators must be inside `@pl.vector_function` | Measured (see §9.5) | Explicit parse-time rejection |

The third rule protects production and consumption of STORE_REG notifications. Clearing AR does not consume an unmatched notification and cannot replace its store.

### Multiple compacted output streams share AR

AR also participates in cumulative destination offsets; **it is not maintained independently per destination Tile or UnalignReg**. Creating another UnalignReg only creates another tail-byte cache. Changing destination Tile, changing VF, or executing `_post` does not clear AR automatically. Two independent arrays therefore cannot interleave appends in one AR chain merely by using two UnalignRegs. Even with a store immediately after every squeeze and a passing static alternation check, writes can land at wrong positions.

First define each logical output stream's start, written length, and end. A simple organization completes streams sequentially: `clear_spr → (squeeze → store)* → post`. Fully flush the preceding stream, then clear again for the next and use its own tail cache. When continuing to append to the same stream across loops, retain the state; do not clear it midway. When multiple fields share a filter condition, use the same logical predicate and ordering; choose whether to cache or recompute the predicate based on memory and measurements.

Validation must check all outputs and their unused tails, covering zero matches, all matches, register-block crossings, byte counts that are not whole blocks, and repeated calls. A passing single-stream example does not establish independent multi-stream state. Formal contracts: clear_spr (`pypto/docs/zh/api/pro_api/SIMD-API/reg_computation/data_movement/clear_spr.md`) and squeeze_store_unalign_post (`pypto/docs/zh/api/pro_api/SIMD-API/reg_computation/data_movement/squeeze_store_unalign_post.md`). At behavior version `872df763f`, CANN 9.2.0, Ascend950PR, interleaved dual streams missed writes at block crossings; completing and initializing streams separately passed.

Runnable comparisons: [vf_unalign](../../examples/basic_api/vf/vf_unalign/) initializes independent squeeze streams separately. [stable_filter](../../examples/algorithms/stable_filter/) demonstrates another choice: predicate prefix ranks construct unique destination indices, which multiple fields then share. The latter must satisfy scatter address and index-uniqueness contracts and does not depend on AR's cumulative displacement.

### 1.5. Three measurements (2026-09-26, parsing only, no hardware run)

A minimal kernel remotely parsed four forms without using a card:

| Probe | Result |
| --- | --- |
| `vf.squeeze(..., dtype=pl.DT_UINT16)` | **Parsing passes**. `dtype` specifies the destination register type; the backend reinterprets the source register as that type rather than numerically truncating it. Omitting it infers the source type. Public declarations and official documentation have been completed |
| Omit `squeeze_store_unalign_post` | **Parsing passes**, with no error. Missing bytes are a silent runtime error |
| **Two consecutive `squeeze` calls without a store between them** | **Parsing passes**. The documentation says this can **hang hardware**, yet the toolchain does not block it |

**The third is a silent-failure class**: a pattern capable of hanging the device passes parsing without a warning. Inside one `@pl.vector_function` body, `vf.squeeze(gather_mode=STORE_REG)` and `vf.squeeze_store_unalign` must strictly alternate. The `squeeze-alternation` rule in `check.py` checks this condition.

A5 generated code defaults to `MODE_STORED`: omitting `gather_mode` writes AR. Pure register compaction should explicitly use NO_STORE_REG. API documentation and defaults were aligned at framework `aff3b5a70`; `test_squeeze_contract.py` pins the generated code, and the gate uses the same default. Validation: 2026-09-28, CANN 9.2.0 / Ascend950PR; 34 lane-layout and unalign cases plus 11 SPR cases passed.

### 1.6. Entry scope and example coverage

1. **VF instructions require `@pl.vector_function`.** Putting them directly in `pl.section_vector()` raises *"Operation 'vf.xxx' can only be used inside @pl.vector_function"*. The enclosing section calls VF procedures; the [scope contract](constraints.md#simd-vf-functions-communicate-through-tile-parameters) identifies parser behavior and version evidence.
2. **`vf_align` must cover parameter behavior, not just two API names.** Both `load_align` / `store_align` have three modes: single contiguous transfer, dual transfer (de-interleave), and noncontiguous DataBlock transfers, plus `dist=` broadcast / upsampling / downsampling / decompression. There are only two API names, but much more behavior than an ordinary API; estimate this example's scope by parameter behavior, not a count of 2.
3. **Explain `vf_unalign` as two separate chains**, and include a negative case with two consecutive squeezes to pin the fact that parsing does not block it. **Never run that case on hardware**, because it hangs the device; assert parsing behavior only.

---

## 2. Masks, reductions, transcendental functions, and lane layouts (8 examples)


### 2.1. Two facts about `reduce_*`

**1. `datablock=` performs within-group reduction.** A group is **8 FP32 elements (32 bytes)**: for input 1..64, `datablock=True` produces `[36, 100, 164, 228, 292, 356, 420, 484, 0, ...]`. Within-group reduction **does not write indices**.

**2. `reduce_max` / `reduce_min` write the extremum's index to `dst[1]`.**

| | lane 0 | lane 1 |
| --- | --- | --- |
| `reduce_sum` | Sum | 0 |
| `reduce_max` | Maximum | **argmax index** |
| `reduce_min` | Minimum | **argmin index** |
| `reduce_*(datablock=True)` | First group result | Second group result, no index |

A non-`datablock` reduction still returns a register; **not every lane contains the same scalar**. If the next consumer needs the same value or the same within-register index, first broadcast lane 0 with `vf.full(reduced, mask)`. `vf.select` chooses lane by lane and does not broadcast. In particular, do not directly use the entire `reduce_min` result as a reg→reg gather index vector: lane 1 may hold the reduction's argmin, and the other lanes are not the desired position either. If only lane 0 is consumed, satisfy that single-lane operation's contract. Source position, dtype, and mask for broadcasting: full (`pypto/docs/zh/api/pro_api/SIMD-API/reg_computation/data_movement/full.md`). Gather address units and ranges: §3.1 below.

The index is an integer written into an FP32 register. Reading it as a float gives a subnormal (index 63 reads as `8.83e-44`), **looking like neither an index nor an error**. The return-value description in the official `reduction/reduce_max.md` documents this fully, plus two details: tied extrema use the first index; with NaN, `dst[0]` is NaN and `dst[1]` is the first NaN's index. Tests were added and passed for both.

**The `_vf_api.py` docstring saying "The remaining lanes are zeroed" is wrong** (the `reduce_sum` version is correct). It was fixed in `pypto` (`1ea35c7ec` on `integration/pro-surface`).

Read official return-value descriptions and call examples as well as source docstrings. `addc` destination declarations, `pto_assert` reach limits and reduction-result lanes illustrate distinct contract details; the evidence rule is in [skill constraints](../../SKILL.md#constraints-and-workspace-policy).

### 2.2. Match the reference to deterministic binary-tree reduction

The official `reduce_sum.md` specifies binary-tree accumulation and **a deterministic result**. Noninteger input alone does not justify relaxing `rtol` under an “unspecified order” assumption.

The distinction matters: the order is fixed, so **noninteger inputs can also be compared bit for bit**, provided the reference also merges pairs in a binary tree. Sequential accumulation via `numpy.sum` can differ in the last bit; then change **the reference's summation order**, not the tolerance. “Unspecified order” would otherwise be treated as a valid reason to relax tolerance, which it is not here.

### 2.3. `precision=` reaches IR, but results are bitwise identical

`ln` / `log` / `log2` / `log10` / `exp` / `sqrt` all have a `precision: Optional[bool]` parameter. Measurements (FP32, inputs `[0.5, 8]`, `ln` and `exp`):

- **It reaches IR**: `@vf.ln(..., precision="precision": True)`; the frontend did not discard it.
- **Results are bitwise identical**; both modes have maximum relative errors of `ln` 5.4e-08 / `exp` 7.2e-08 (about 1 ulp).

Identical values do not establish that the backend ignores the parameter: C++ codegen emits a subnormal-handling path, and the ordinary positive inputs used here do not distinguish the modes. Frontend IR and numerical validation answer whether the parameter propagates and what outputs occur for the selected inputs, respectively; neither replaces checking the mechanism.

### 2.4. `pl.const` requires a literal first argument, but adjacent parameters do not

`vf.shift_left(a, SHIFT, preg)` accepts the module-level constant name `SHIFT`; `vf.shift_right(a, pl.const(SHIFT, I32), preg)` is rejected with `pl.const() first argument must be a numeric literal` (`_parse_typed_constant` in `scalar_ops.py`). **The two positions define “constant” differently**, so the same line can contain both an accepted and a rejected use. `vf_arith` provides the executable positive and negative forms.

### 2.5. Combining masks with existing comparisons and selection

The current `and_` / `or_` / `xor` / `not_` APIs support both RegTensor and MaskReg overloads. Mask operations need not start with `mask_` or appear only in the “mask operations” directory. First determine whether the source is a data or predicate register, then choose the operation and participation mask.

| Goal | Form |
| --- | --- |
| **Intersection / union / exclusive-or** of two masks | `vf.and_(m1, m2, preg)` / `vf.or_(m1, m2, preg)` / `vf.xor(m1, m2, preg)`; the result remains MaskReg |
| **Intersection** of two comparison conditions | Chaining also works: `m2 = vf.lt(a, HI, m1)`; the second comparison uses the first mask as its participation domain |
| **Invert** a mask | `inverse = vf.not_(m, preg)`; if only reverse data selection is needed, swapping the two `vf.select` sources also works |
| Compare with a **scalar** | Pass the scalar directly; `gt.md` says the API dispatches to the corresponding hardware instruction |
| Select between a register and a **constant** | First broadcast the constant into a register of the same dtype with `vf.full`, then use `vf.select`; do not generalize comparison operators' scalar overloads to select's two sources |

The same operation names perform data-bit operations on RegTensor and predicate logic on MaskReg; a data register containing 0/1 does not automatically become a mask. Evidence: the dual overloads on the official and_ (`pypto/docs/zh/api/pro_api/SIMD-API/reg_computation/logical_computation/and_.md`) and related pages, installed `_call_parser.py::_VF_UNIFIED_OPS`, and `pand/por/pxor/pnot` dispatch in C++ `backend_cce_vf_ops.cpp`. At behavior version `872df763f`, CANN 9.2.0, Ascend950PR, [vf_bitwise](../../examples/basic_api/vf/vf_bitwise/) verified all four boolean input combinations for b32 full and tail participation masks, observing results through `vf.select(ones, zeros, mask)`.

To exclude lanes from a reduction, apply `vf.select(values, neutral, valid)` to the loaded register, with the reduction's neutral value first broadcast into a register. Reuse that validity for later consumers. If editing a tile instead, verify the [complete store address and physical container](#distinguish-numerical-conversion-from-port-bit-packing-first) and the [store-to-load ordering](#212-multiple-memory-accesses-to-the-same-tile-inside-vf-have-no-ordering-guarantee) before reading it again in the same VF; register selection avoids that new memory dependency.

```python
values = vf.load_align(score_tile, offset)
neutral = vf.full(NEG_INF, all_lanes, dtype=FP32)  # NEG_INF is a Host constant
masked = vf.select(values, neutral, valid_lanes)
# Reduce/transform masked; score_tile remains unchanged.
```

Two additional measurements:

- **mask_reg has a fixed total width of 256 bits; bits per element depend on dtype**: FP32 uses 4 bits/element (64 × 4 = 256), FP16 uses 2, and INT8 uses 1. This is why mask dtype must match data dtype.
- **`mask_gen_with_reg_tensor` `offset` counts 8-byte DataBlocks, not elements.** For FP32, 256B contains 32 blocks; `offset=0` takes the 64 bits of lanes 0..1.

### 2.6. An unreliable observation method produced four wrong conclusions

A probe used `vf.move(a, mask)` to observe masks, assuming inactive lanes would be zeroed. They actually contained another input's values: the compiler allocates the destination register, so inactive contents depend on which register it reuses. Thus, all four conclusions, including “`and_` intersection is all true,” “`get_mask_spr` is all true,” and “`not_` has no effect,” were wrong.

The signal that exposed this was **contradictory conclusions**: mutually exclusive masks intersected to all true. The correct observation method is `select(ones, zeros, mask)`; `select.md` specifies its semantics without allocation dependence.

Recorded in [verification/evidence.md](../verification/evidence.md), item 7.

### 2.7. Two mask-state modes; `get_mask_spr` understands only one

The four `vf_mask_state` APIs (`pl.set_vec_mask` / `set_mask_norm` / `set_mask_count` / `reset_mask`) have **no dedicated pages** under `docs/zh/api/pro_api/`; only `get_mask_spr.md` mentions `set_vec_mask` in passing. The only primary source is ST `element_wise/test_vector_mask.py`, so behavior comes entirely from measurements.

| Mode | Switch | Meaning of `low` in `set_vec_mask(high, low)` |
| --- | --- | --- |
| norm (bitmap, default) | `pl.set_mask_norm()` | Bit i controls element i |
| count | `pl.set_mask_count()` | Number of valid elements |

**Trap: `vf.get_mask_spr()` reads only as a bitmap.** With count mode set to 16, a vf body reads “only lane 4 is valid” (`16 = 0b10000`). **No error is raised.**

| Setting | Tile-level operators interpret it as | `get_mask_spr` reads it as |
| --- | --- | --- |
| `set_mask_count(); set_vec_mask(0, 16)` | First 16 elements valid | Only element 4 valid |

Thus, **count mode is intended for tile-level vector operators** (the ST pairs it with `pl.add`). For a count mask in the vf path, use `vf.update_mask(count, dtype=)`. This qualifies the decision to exclude tile computation: the excluded part is tile-level **computation**, while state APIs such as `set_mask_count` remain on the `pl.` side. **Their semantics were designed for the tile path**, so take extra care on the vf path.

Two further measurements: `reset_mask()` restores all 64 lanes to valid; mask state **persists across `@vf` boundaries**. Set it once, and two consecutive vf calls both observe it, so failing to reset leaks state into later code.

### 2.8. `vf_lane_layout`: two points spanning examples

The `pack` / `unpack` dtype-pair table, `interleave` ordering, the `astype` default `CAST_RINT`, and the FP16-only restriction of `muls_cast` are all covered by five rules in [`examples/basic_api/vf/vf_lane_layout/`](../../examples/basic_api/vf/vf_lane_layout/test.py). Two points are not apparent there:

1. **The official `unsqueeze` formula was wrong and has been fixed in pypto** (`a0656978d`). Together with `reduce` and `arange`, it makes three incorrect official formulas. **An incorrect formula is more dangerous than a missing one**: following it produces a self-consistent but wrong reference.
2. **Distinguish this layer from `vf_align`**: this example rearranges reg↔reg data; `vf_align` transfers Tile↔reg data. Both use the word “alignment,” but one concerns register lane arrangement and the other memory addresses.

### 2.9. Hardware findings for `vf_unalign`

The core mechanism **works**: two variable-length filtering passes share one `areg`, with no finalization between them, and the second continues at the **byte position** where the first stopped. For input `1..64`, the first pass selects `>48` and the second `<=16`; the first 32 output elements are exactly `49..64, 1..16`, with correct order and positions. This is the skeleton of Top-K index collection in the `lightning_indexer` ST.

Three silent traps were encountered during this hardware run:

| Trap | Where it is blocked | Rule |
| --- | --- | --- |
| Integer `stride` without `post_update=True` | **bisheng**: `static assertion failed ... The last argument can only be 'POST_UPDATE'` | A scalar stride requires `post_update=True` |
| Omit `vf.clear_spr()` before `squeeze(STORE_REG)` | **Nowhere** | AR is a special register and is not guaranteed empty on entry to `@pl.vector_function` |
| Variable-length writes cover only part of a Tile, but unwritten positions are assumed to be 0 | **Nowhere** | UB is not zero-initialized; first zero variable-length output with `store_align` |

The first exposes a contradiction between documentation and the default: the parameter table says “effective only with `post_update = True`,” but the default is `False`. Supplying a scalar stride without advancing the cursor is inherently meaningless; this combination should not have a default.

The last two share a pattern: **check initialization before attributing incorrect results to transfer semantics**. In the `1..64` counterexample, uncleared AR prevented any bytes from being written and the apparent output was entirely Tile residue. The third appears only with variable-length writes; fixed-length writes cover the entire block and hide it.

### 2.10. Narrowing `astype` occupies only half the width: pair two chunks and `de_interleave` before storing

After narrowing such as FP32→FP16, results occupy only the even half positions of the physical register (`CastLayout.ZERO` semantics; see “Widening / narrowing `vf.astype`” in [constraints](constraints.md) and the half-position measurements in fixpipe §15). A standalone `astype` followed by a full-width store also writes garbage from the odd half positions. **Symptom signature**: columns alternate between correct values and nearly constant wrong values, without an error.

Full-panel recipe (hardware evidence 2026-09-29, `_softmax_cast_band` in `examples/algorithms/mm_softmax_mm`; upstream precedent: `pypto/python/tests/st/pypto_pro/frontend/auto_pipeline/test_fa_serial_dn_auto.py`): apply `vf.astype(..., dtype=FP16, layout=pl.CastLayout.ZERO)` separately to two adjacent 64-element chunks. Take the first return value from `vf.de_interleave(h_lo, h_hi)` to obtain a dense 128-element FP16 register, then issue one `store_align` with a full-width FP16 mask. For **ND destinations**, a cheaper alternative is available: narrow one chunk, then use `store_align(..., dist=pl.StoreDist.PACK_B32)` to write densely (hardware evidence; see §2.14). **Explicit `PACK_B32` is required**, because the generic name selects the wrong granularity. For vsstb cursor scatter, compact the useful FP16 values with `de_interleave` and derive the store predicate from their resulting positions as below; a smaller panel does not require all 128 positions to be valid.

#### Recompute the store predicate after packing or changing tile width

Packing changes which physical positions hold useful elements. Two full 64-element FP32 chunks narrowed and de-interleaved produce 128 dense FP16 values, so an FP16 `H` mask writes only half the intended result. Conversely, a dense prefix of only 32 useful FP16 values needs a 32-element destination predicate; carrying `H` from a 64-element example can write beyond a narrower tile. Derive the actual valid destination positions and touched bytes after packing, using `update_mask(valid_count, dtype=FP16)` only when they form a dense prefix. Do not infer this count from the source register's mask or retain an example's `H`/`Q` pattern after changing the tile shape. Two partially valid source registers do not automatically pack into one prefix: `de_interleave` places the second source's even lanes after all 64 even-lane positions of the first. Check both unwritten holes and extra writes with nonzero data and guards through the real consumer.

### 2.11. Mask pattern families and separate responsibilities: masks select prefixes; stores handle scattering

**Every `MaskPattern` is anchored at the low end** (official `MaskPattern.md`, specific to the 950 family): `VL1..VL128` (lowest N), `H` (low half), `Q` (low quarter), `M3`/`M4` (first of every 3/4), and `ALL`/`ALLF`. **There is no arbitrary-window pattern.** For a compile-time half/prefix pattern, use one `create_mask(pattern=)` instead of assembling an `arange` + `ge/lt` chain. Repeated runtime predicates have a separate [construction-cost decision](#repeated-runtime-masks-in-vf-loops).

The three construction paths have distinct roles. `create_mask(pattern=)` produces a static low-anchored pattern. `vf.update_mask(scalar)` has **count semantics, verified on hardware** (FP16: `update_mask(3)`→lanes {0,1,2}, `update_mask(0x14)`→lanes {0..19}, `update_mask(128)`→all 128 lanes): `update_mask(N)` means the first N elements are valid; N ≥ element count selects the full width (`update_mask(128, FP16)` is equivalent to `create_mask(ALL, FP16)`; the official example's `0xFFFFFFFF` saturates to full width). Follow the official functional description, “generate a valid-bit mask of the corresponding length.” The parameter-table phrase “its bits define the new mask pattern” refers to hardware expansion of per-element mask bits (2 bits per FP16 element, 256 bits total); interpreting it as scalar bits selecting lanes is wrong. Use it for a prefix mask of runtime scalar length (`create_mask` patterns are compile-time constants). `vf.mask_gen_with_reg_tensor` generates a mask at runtime from bits in a specified RegTensor DataBlock (data-driven mask; offset counts DataBlocks rather than elements, as at the end of §2.5).

**Separate-responsibility idiom (ascriptor precedent)**: masks determine only which lanes are valid, selecting a **contiguous prefix** (Q/H/VL/count); store-side stride/dist/block_stride determines where they go. Both ascriptor's `reg_to_ub(dst, src, FRAC_STRIDE4, mask=LOWQUAT)` (low-quarter mask + strided scatter) and pto FA's vsstb (`block_stride` block copy + full-width mask) follow this split. The usable vsstb form is **cursor-based**, with no explicit offset and with `post_update`; the mechanism and explicit-offset gap are in crosscore-sync's “row+1 pitch convention” section. Window-mask segment selection (8 16-lane windows × full-width stores) was tested and rejected: values were correct, but each store used only 1/8 of its lanes, causing 8× VF port overhead and doubling measured device time.

`CastLayout.ZERO` outputs are **even-interleaved** (lanes 0,2,4,…; see “Widening / narrowing `vf.astype`” in [constraints](constraints.md) and fixpipe's half-position measurements), not the contiguous low half selected by `H`. Packing narrowed values densely still requires `de_interleave`; an `H` mask cannot replace it.

#### Repeated runtime masks in VF loops

First check [VF control flow and realized loop form](#keep-runtime-dispatch-out-of-vf-loops). Then count the complete predicate expression at its loop frequency: scalar coordinate arithmetic, `pl.min` / `pl.max`, count conversion, mask generation and selection. A compact expression such as `vf.update_mask(pl.max(0, pl.min(width, count)), dtype=...)` can expose substantial scalar-control cost inside VF. Use [VF cost decomposition](../diagnostics/performance.md#decompose-a-busy-vf-stage) before attributing the remaining cost to mask generation or nonlinear arithmetic.

Under the count semantics above, `update_mask` already saturates counts at the register width. When `width` is the full register element count for that dtype, an upper `min(width, count)` can be redundant; a smaller logical tail bound still needs enforcement. Preserve the lower bound unless its nonnegativity is proved: the scalar is converted to `uint32`, and a negative count does not mean an empty mask. Also prove that the count fits the conversion range. This is a reason to simplify a particular expression, not a blanket claim that `update_mask` is slow.

Choose a comparison from the predicate's structure:

| Predicate structure | Implementation to compare |
| --- | --- |
| Whole blocks are fully valid, fully invalid or boundary blocks | Classify outside the VF row loop and select a specialized body. Fully valid blocks can omit masking; skipping invalid work still needs the original output, memory-access and state-update contract |
| The boundary changes affinely across rows | Seed lane coordinates and a boundary register once per VF call, compare them with `vf.lt` / `vf.ge`, and advance the boundary with vector arithmetic. Avoid rebuilding it from a runtime scalar for every row |
| A fixed mask repeats across many blocks | Compare a retained predicate or precomputed bias with its storage, initialization and reuse costs. [MHA's `_init_tri` and `_softmax_block`](../../examples/algorithms/flash_attention_mha/kernel.py) show a triangular bias; exponential masking still needs a numerical proof of zero excluded weights |

For example, inside a VF with a full b32 `preg`, a signed prefix whose count advances by `ROW_STEP` can be constructed as follows. The body consumes `valid` before advancing `limit`:

```python
lanes = vf.arange(0, dtype=pl.DT_INT32)
limit = vf.full(first_count, preg, dtype=pl.DT_INT32)
for r in pl.range(ROWS):
    valid = vf.lt(lanes, limit, preg)
    # Consume valid for this row.
    limit = vf.adds(limit, ROW_STEP, preg)
```

Use each register chunk's logical column origin, rather than restarting every chunk at zero. Prove signed-coordinate bounds, reset state at the VF invocation boundary, and retain the physical predicate granularity required by each consumer. Test empty/full prefixes, both sides of register boundaries, tails and each participating AIV owner. When only predicate construction changes, compare outputs bitwise against the accepted path as well as checking the original numerical contract. Then time the complete call; one successful shape does not select a universal mask implementation.

### 2.12. Multiple memory accesses to the same tile inside VF have no ordering guarantee

Inside one `@pl.vector_function`, two accesses to **the same tile region** have no automatic dependency synchronization and may complete out of program order. Two hardware observations:

- **Write then read**: vsstb (`DATA_BLOCK_COPY` scatter) immediately followed by `load_align` from the same tile read stale pre-write data. Five consecutive runs of the same probe produced 2 FAIL and 3 PASS; wrong values changed each time (random residue), while the number of wrong positions stayed fixed (stable race window).
- **Write then write**: a full-width constant store followed by a masked data store at the same address let the full-width store overwrite the masked store, appearing as though the masked store wrote no lanes. This illusion directly caused a wrong conclusion about update_mask semantics (§2.11); the artifact was convincing.

When ordering is required, enumerate source accesses, destination accesses, and shared address ranges, including loop backedges. Native `vf.mem_bar(mode=...)` can establish the required ordering within VF: `VST_VLD` handles vector write→read, `VLD_VST` read→write, `VST_VST` write→write, and `VV_ALL` all vector-memory directions. It is a statement with no return value, using enum `pl.MemBarMode`. Choose the mode covering the actual dependency; a single RAW barrier does not synchronize every direction. The official mem_bar contract (`pypto/docs/zh/api/pro_api/SIMD-API/reg_computation/data_movement/mem_bar.md`), installed `_vf_api.py`, and [vf_spr_misc](../../examples/basic_api/vf/vf_spr_misc/) are independent sources of evidence.

Another option is to split dependent phases into two VF calls so the framework inserts synchronization across calls based on tile-group read/write dependencies, or use nonoverlapping storage. First check whether loop state can remain in registers and avoid the memory round trip. `auto_mutex` does not mean all accesses inside VF automatically complete in program order; fixing only the final merge loop does not prove the earlier candidate-construction loop safe.

At behavior version `872df763f`, CANN 9.2.0, Ascend950PR, a streaming candidate-update implementation passed the three given shapes but emitted an index sentinel on an additional valid short row. Adding only `VV_ALL` on the update loop's backedge made 112 independent combinations pass, including repeated calls, extrema, and bit patterns. This verifies that barrier for the tested dependencies, not every toolchain or every access form. The early vsstb/store race observations remain diagnostic clues: they neither prohibit all same-VF reads/writes nor explain every error as missing barriers. If errors remain with a barrier, inspect addresses, register aliasing, and generated code. Scalar Tile/Tensor accesses inside VF also have the default `-O3` contract restriction, which a barrier cannot lift.

### 2.13. The FP32 path for FP16 source data: order-preserving widening with UNPK (required reading)

Basic facts (hardware evidence, examples/algorithms/rmsnorm_gemm):

- **One ordinary `vf.load_align` reads “256-byte physical register ÷ dtype byte width” native elements**: **128** for an FP16 tile, not 64. “One register = 64 lanes / `VL=64`” applies specifically to FP32 sources. The `load_align` `dtype=` parameter does not change this.
- **Widening `astype` (FP16→FP32) on a compact FP16 register makes `layout` act as source subsampling**: `ZERO` selects even-indexed source elements, `ONE` selects odd. Fundamentally, `ZERO/ONE` selects the low/high 16-bit position of each b32 container; in a compact register, each container's low half happens to hold the even-indexed element. Omitting `layout=` defaults to `ZERO` and **silently drops half the data**, with output approximately equal to the source's even columns.

**Order-preserving FP32 path (correct, bit-exact hardware evidence)**: `load_align(tile, off, dist=pl.LoadDist.UNPK)` expands 64 consecutive FP16 values into the **low halves of 64 b32 containers, preserving source order**. Then `astype(dtype=FP32, layout=ZERO)` numerically converts the low halves. Together, these two instructions are equivalent to one load from an FP32 source; standard FP32 row-reduction/elementwise recipes apply unchanged, with FP32 throughout and only one narrowing at output. In rmsnorm_gemm, compared with the compromise that retained FP16 elementwise values and rounded twice, this tightened numerical tolerance back to atol 2e-2 while keeping device time equal (4.36 vs 4.38us). Even/odd split reduction (widen each half, reduce each, then add; sum unchanged) remains an alternative for order-independent steps **without introducing UNPK**.

**Derive the predicate from container positions, not the compact source count.** After FP16 `UNPK`, logical element `i` occupies FP16 position `2*i` in a b32 container. For `n` valid logical elements (at most 64), `astype(..., layout=ZERO)` therefore needs a predicate spanning those b32 containers, such as `update_mask(2*n, dtype=FP16)`; `update_mask(n, dtype=FP16)` only converts roughly half the values. Later FP32 arithmetic and `PACK_B32` stores use a b32 predicate for `n` containers. The `dtype` in `update_mask` chooses physical predicate granularity, not the desired output type; derive each conversion edge separately rather than carrying a source-count mask through a width change.

**Reuse a predicate when its consumed physical bits match.** `update_mask(n, dtype=FP32)` sets representative bits `4*i` for `0 <= i < n`. After FP16 `UNPK`, `ZERO` conversion selects source halfword `2*i`, whose representative predicate bit is also `4*i`; subsequent FP32 arithmetic reads those same bits. The existing b32 compute predicate therefore suffices for both operations. `update_mask(2*n, dtype=FP16)` is another valid conversion mask, but it additionally sets bits `4*i+2` that this ZERO conversion ignores. These predicates are equivalent for this consumer, not bitwise identical. Reusing the b32 predicate avoids an extra count expression and mask-generation instruction in each VF iteration.

The native input sequence for contiguous FP16 data can share one predicate:

```python
compute_mask = vf.update_mask(valid, dtype=pl.DT_FP32)
expanded = vf.load_align(src, offset, dist=pl.LoadDist.UNPK)
values = vf.astype(expanded, compute_mask, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
# Reuse compute_mask for the following FP32 operations and stores.
```

Here `0 <= valid <= 64` and `offset` counts source FP16 elements. Changing the source dtype in an FP32 anchor without changing its load distribution is not this sequence. A compact FP16 load followed by `ZERO` conversion is appropriate only when selecting its even half-elements is intentional. Do not reuse the b32 predicate blindly for `ONE` or another conversion edge: odd source halfwords consult bits `4*i+2`, which this predicate leaves zero. Derive the selected source positions and their representative bits first.

Evidence: the official `LoadDist.UNPK`, `CastLayout.ZERO` and `update_mask` descriptions define the expansion, selected half and mask granularity; `_vf_api.py` preserves the source register dtype and exposes these selectors. At behavior `872df763f`, CANN 9.2.0, a FP16→FP32→FP16/INT8 consumer lost values beginning at column 32 with the compact-count mask; changing only the widening predicate restored its 65-column test and subsequent boundary matrix. Full-width predicates and a one-element probe do not distinguish this failure. Test below/at/above the half-register and full-register boundaries with lane-distinguishing values.

On the same behavior/toolchain, the installed `update_mask` docstring and C++ `EmitVFUpdateMask` distinguish b16/b32 predicate generation, while `EmitVFCast` forwards the predicate with the selected layout. A hardware control replacing only a separate b16 conversion predicate with the existing b32 predicate preserved the mixed-precision update's boundary, cancellation and state checks and reduced task time. Predicate compatibility is a physical-bit proof, not a general rule that a source or destination dtype always determines the right mask.

### 2.14. Load/store port distribution modes: prefer built-in port rearrangement over compute emulation

`vf.load_align(dist=pl.LoadDist.*)` / `vf.store_align(dist=pl.StoreDist.*)` let the load/store **ports** rearrange data during transfer. General rule: prefer built-in port distribution over compute instructions emulating the same rearrangement (counterexample from hardware: masked-window scatter emulating vsstb was 2× slower, §2.11). The complete tables are in the official `LoadDist.md`/`StoreDist.md`. Hardware usage precedents: `pypto/python/tests/st/pypto_pro/frontend/lightning_indexer/` (BRC scalar broadcasting inside loops, DINTLV splitting bytes into two return values, INTLV interleaving two registers into one write).

Modes verified on hardware in this repository (2026-09-29):

- **`LoadDist.UNPK`**: expands 64 consecutive 16-bit elements into low halves of b32 containers, preserving source order; the entry path for order-preserving FP16→FP32 widening (§2.13, bit-exact). **The b8 member of this family also works**: INT8 widened with `UNPK_B8` + `astype(FP16, layout=ZERO)` is bit-exact (hardware evidence 2026-09-29, W_q widening in examples/algorithms/quant_gemm_w8a16). `UNPK_B8` and `UNPK4` have different container widths and cannot be interchanged in composite conversions; see below.
- **The official `LoadDist.E2B` description is wrong**: it says “16-bit→32-bit expansion,” but measurement shows **element-to-block broadcasting**, duplicating each source element to fill a 32-byte DataBlock, consistent with ascriptor interp's `e2b` implementation. Use UNPK for 16→32 expansion.
- **`StoreDist.PACK_B32`**: writes the low 16 bits of each b32 element contiguously. After narrowing `astype(FP16, ZERO)`, whose results occupy container low halves, **one store directly emits dense FP16** (bit-exact). This replaces the §2.10 `de_interleave` recipe and saves one rearrangement for ND destinations. Continue using de_interleave for vsstb cursor scatter, which requires a dense 128-wide register.
- **The generic `PACK` name is a trap**: it chooses granularity automatically from source dtype. An FP16 register selects `PK_B16`, taking “the low **8 bits** of each **b16** element,” which is pure bit truncation and emits garbage for FP16 values (measured all 0, because 1.0=0x3C00 has low 8 bits of 0). The FP32→FP16 narrow-and-store path here needs **explicit** `PACK_B32`; do not copy it to other widths. The official description “low-half bits” is literally correct at the selected granularity, but automatic granularity can choose the wrong semantics.
- **`vf.interleave`/`de_interleave` (register layer) versus INTLV/DINTLV (port layer)**: the first return from `de_interleave(a, b)` concatenates even lanes from both sources. A port `DINTLV` load sends memory elements 2i/2i+1 to lane i of two destination registers. A port `INTLV` store interleaves the two sources lane by lane in memory; **A5 ignores the predicate and writes all 2L elements** (ascriptor RFC-0001 hardware note: the mask is ineffective for this instruction, so the entire footprint must be in bounds).

#### Map physical containers before composing conversions

The ability to specify dtype in `astype` does not mean every source/destination pair can be emitted. First check the official astype conversion table (`pypto/docs/zh/api/pro_api/SIMD-API/reg_computation/type_conversion/astype.md`) and installed `_vf_api.py`, then C++ `EmitVFCast` if needed. At behavior version `872df763f`, direct UINT8→FP32 fails Bisheng compilation, while UINT8→FP16 is a supported step. After splitting an unsupported conversion into several steps, prove the layout at every step again; checking only the numerical type chain is insufficient.

For every step, state which bits of which physical container hold logical element i, which positions the conversion selects, and how many physical bits the predicate covers. `UNPK_B8` places bytes into b16 containers; `UNPK4` uses b32 containers. `dist` changes arrangement, not the returned register's element type: a UINT8 source still returns UINT8 and cannot directly participate in bitwise operations with UINT32/INT32 registers. First create the corresponding-width register view according to the bit-pattern contract, then match operand types for bitwise operations. Further widening may subsample even/odd positions again. `update_mask(n, dtype=UINT8)` covers n compact-byte positions, not n expanded logical elements; a final output mask cannot recover values zeroed earlier. Full-mask examples do not expose this tail error.

A verified route from unsigned bytes to FP32 computation is `load_align(..., dist=UNPK4)` → `bit_cast(..., dtype=INT32)` → `astype(..., dtype=FP32)`, with b32 predicates for later operations on 32-bit containers. Here `bit_cast` preserves numerical values because each container's high 24 bits are zero and its range is only 0..255. It is not a general numerical UINT32→INT32 conversion and cannot be reused as signed-byte sign extension. When storing `astype(FP16, ZERO)` results, `PACK_B32` takes each b32 container's low 16 bits; partial masks still cover b32 containers, not merely the compact FP16 output element count.

Runnable chain: `make_u8_float_chain` in [vf_align_widths/kernel.py](../../examples/basic_api/vf/vf_align_widths/kernel.py). Tests cover 0..255, boundaries 31/32/33 and 63/64/65, cross-register offsets, two output dtypes, and guards in unwritten regions. Before incorporating new composite conversions into quantization formulas or matrix computation, verify the whole chain with an order-preserving, small-integer, partial-mask probe like this. Do not diagnose lane displacement through loose tolerances on large matrices. Hardware identity: behavior version `872df763f`, CANN 9.2.0, Ascend950PR.

#### Distinguish numerical conversion from port bit packing first

When choosing a store distribution, specify the bits per physical register container, which bits hold the desired result, and the bits per destination element. Do not choose solely by result dtype or an FP16 example. Register conversion can leave gaps, and composite conversions may rearrange several times. Check the actual layout with recognizable bit patterns and partial masks before attaching the full computation.

| Explicit store distribution | Extracted from each source container | Compact output |
| --- | --- | --- |
| `PACK_B16` | Low 8 bits of each 16-bit container | One byte |
| `PACK_B32` | Low 16 bits of each 32-bit container | Two bytes |
| `PACK4_B32` | Low 8 bits of each 32-bit container | One byte |

All are bit extraction, with no float-to-integer conversion, rounding, or saturation. For integer numerical outputs, first complete the numerical conversion and range restrictions required by the contract. INT32 values already within the target range can remain in b32 registers and use `PACK4_B32` to write INT8/UINT8 tiles; `astype(INT8)` is not required merely to match the output declaration. This path's predicate is built for b32 source lanes; offsets count destination tile elements, and one active source lane writes one destination byte. If data is already compact bytes, applying the same packing discards data. Packing is not mandatory for every 8-bit output.

**Recompute every access's byte address after changing bit width or start coordinates.** Alignment applies to `tile_base + element_offset * tile_element_bytes`, not just the tile base or source address. A logical group stride can keep an FP32 source 32-byte aligned while misaligning a byte destination. A masked store limits participating lanes but does not repair an unaligned start: ordinary `StoreDist.NORM` still needs 32-byte alignment with a one-lane mask. The separate `FIRST_ELEMENT_B32` form permits 4-byte alignment but ignores the mask, so it is a different operation. `PACK4_B32` destination starts also obey the official 32-byte alignment requirement; violating it can produce `The address for VEC to access UB is not aligned` on the current card, rather than an ordinary numerical error.

Logical groups need not equal physical memory-access blocks. Keep register blocks physically aligned and process different logical groups within each block; alternatively, first write an aligned local tile, then use valid transfers to write the logical position. If using unalign or another rearrangement path, recheck that path's bit-width, state, range, and finalization contracts. Do not unilaterally restrict the task's allowed group sizes or change output layout to avoid this issue. In addition to full blocks and tails, tests should include parameter combinations where the same logical stride yields different source and destination alignment. Compute addresses before launch; an output mask cannot replace this.

Minimal executable fragment: `make_pack4_fragment` in [vf_align_widths/kernel.py](../../examples/basic_api/vf/vf_align_widths/kernel.py). VF always packs at local UB offset zero, while `pl.store` writes the actual valid length to the logical GM start; keep these addresses distinct. Tests cover GM starts that are not 32-byte aligned and the unwritten regions on either side. When using this transfer skeleton for segmented computation, check input-fragment addresses under their own access paths too; fragment length must not exceed local capacity. The task still determines numerical computation and logical fragment boundaries.

Evidence: official StoreDist (`pypto/docs/zh/api/pro_api/SIMD-API/reg_computation/basic_data_structures/StoreDist.md`) / store_align (`pypto/docs/zh/api/pro_api/SIMD-API/reg_computation/data_movement/store_align.md`), installed `_vf_api.py::store_align` enums, and C++ `EmitVFStoreAlign` emission of `PK_B16/PK_B32/PK4_B32`. Direct b32→byte hardware evidence at behavior version `872df763f`, CANN 9.2.0 is in [vf_align_widths](../../examples/basic_api/vf/vf_align_widths/). It verifies bit truncation, signed/unsigned destinations, partial masks, cross-register offsets, and guards; it does not replace the user's rounding and saturation contract.

### 2.15. Extreme-value `exp` semantics and `mul_dst_add` as true FMA (hardware evidence 2026-09-30)

Kernels with recurrences such as `exp(x - running_max)` in online softmax depend on two numerical facts:

- **`vf.exp` returns exactly 0 for `x <= -88.0` through `-inf`**, without NaN / Inf / subnormal noise (128-point probe, including -88 / -1e30 / -inf). Maximum relative error in the normal domain is 8.84e-8 (≈1 ulp, consistent with the existing `exp` entry in the §3 series). Thus, initializing running max with a finite sentinel such as `-1e30` makes `exp(sentinel - m)` cleanly zero before the first block, without an explicit mask. The +88 upper side was not measured: the running-max recurrence guarantees arguments ≤ 0 and does not depend on it.
- **`dst = vf.mul_dst_add(a, b, preg)` is bitwise equal to single rounding of `round(dst * a + b)`** (512 points with 0 differences; comparison with separate mul-then-add double rounding gives relative errors of 1.3e-6 versus 5.8e-8). It is true FMA, the numerically sound default for online rescaling `O = alpha * O + PV`; [initialize its implicit input destination](#destination-registers-used-as-inputs) first. `vf.exp_sub` is bitwise identical to `sub` + `exp`, with no fused-precision difference.

---

## 3. gather / scatter, SPR, and softmax (3 examples)


### 3.1. `gather` has two forms: out-of-bounds behavior is defined for only one

The backend dispatches by the type of `src`:

| | `src` | `preg` | `data_copy_mode` | Out-of-bounds indices |
| --- | --- | --- | --- | --- |
| Tile→reg | Tile (32B-aligned base) | **Required** | Supported | **No guarantee**; documentation only says indices must be in the valid range |
| reg→reg | Register | **Must not be passed** | Unsupported | **Wraps** `i % (VL/sizeof(T))` |

Passing `preg` incorrectly raises a signature error rather than failing silently. But **the asymmetry in out-of-bounds behavior appears only in documentation**. Code relying on wrapping will read unrelated UB data if the same logic switches to Tile→reg.

Index dtype is fixed by element width: an FP32 source requires `DT_UINT32` (the official constraint table contains 17 triples).

`DATA_BLOCK_LOAD` has two silent pitfalls: **indices are bytes and must be 32B aligned**, not element indices; **only indices in the first VL/32 lanes are read**.

`scatter` indices **must be unique**; for duplicates, “the system retains only one,” with no guarantee which. `gather` explicitly allows duplicates: multiple lanes reading one source perform a broadcast. Reads can broadcast while writes collide, a sensible asymmetry that **is not checked at runtime**.

A full consumer with multiple rows, negative indices, and tail blocks is [row_gather](../../examples/algorithms/row_gather/). Input keeps a compact GM layout; indices are first normalized in the signed domain, then converted to the matching index dtype; predicates constrain both the actual gather and output writes. Physical UB width is separate from logical column count, and the pure-Vector row loop is driven by the runtime worker count.

### Grouped state and boundaries in lane exchange

In shuffle/gather, scan, or reductions carrying indices, treat fields that jointly describe a logical result as one state, such as “value + index,” “accumulator + reset flag,” or “candidate + validity.” For each exchange round, specify source lanes, destination lanes, and merge conditions, then apply the same boundary condition to every field. If a numerical value retains its old state, its flag or index must not still update from an invalid source. The final store mask cannot repair state corrupted earlier.

A field's logical meaning does not determine its register type. An INT32/BOOL data register holding 0/1 remains RegTensor and cannot directly serve as `preg` for `vf.select` and similar operations; first generate MaskReg with `vf.eq/ne` or another comparison. Annotate both what each field represents and which carrier the API accepts. Do not infer types from flag/head/mask in variable names. Signatures and bit-width constraints: select (`pypto/docs/zh/api/pro_api/SIMD-API/reg_computation/comparison_and_selection/select.md`).

Scalar constants must also fit the operand dtype's range. For a U32 index register, a negative literal passed to `vf.adds` can be rejected during parsing; hardware wraparound semantics do not imply automatic conversion to unsigned two's complement. Choose an equivalent expression representable in that dtype, or explicitly use signed arithmetic then convert. This is separate from checking the logical participation domain.

Define within-block results and already-merged results as distinct states, making clear when old state has already contributed. When forming the next state from a merged result, do not add the same carry or scaling again. Derive all new fields from the same old state, then update them together. Multi-block inputs without resets expose repeated merges; a single block or one handoff is insufficient.

reg→reg gather wrapping guarantees defined instruction indices, not that the algorithm permits crossing logical boundaries. When expressing shifts with wrapped indices, boundary lanes must retain old state or select the algorithm's identity element; do not mix fields from the opposite end or tail padding into a prefix. Distinguish the current register's active range, the whole row/segment range, and cross-block carry state, and check consistent handling of every state component across these ranges.

Tests should include no state resets, reset at every element, sparse resets, and boundaries `VL-1`, `VL`, `VL+1`, and one remaining element after multiple registers. Dense random flags alone can hide incorrect carry loss behind the next reset. This is a method for checking state propagation, not a requirement for a particular scan network or thread assignment. Full consumer: [segmented_scan](../../examples/algorithms/segmented_scan/). Stable value/index selection and preservation of original bit patterns: [row_top2](../../examples/algorithms/row_top2/).

The [streaming segmented scan](../../examples/algorithms/streaming_segmented_scan/) extends this mechanism beyond UB capacity with three native stages; its kernel/test headers own the local and incoming state domains, scratch bounds and complete-call performance tradeoffs.

### 3.2. `arange` `DECREASE_ORDER`: the documented formula was wrong

Measured `start=100`, VL=64 → `163, 162, ..., 100`: **`start` is the last lane's value**, with `dst[i] = start + (VL - 1 - i)`.

| Source | Claim | Measurement |
| --- | --- | --- |
| Official prose: supplied start is the **final value** | Correct | ✓ |
| `_vf_api.py` docstring `dst[i] = start - i` | Incorrect | ✗ |

**The apparently more precise formula was the wrong source.** Fixed in pypto (`f1a9282ca`).

Step is fixed at `±1` (a hardware property of `vci`); follow it with `vf.muls` for other steps. **There is no step parameter; failing to find it is not an oversight.**

### 3.3. AddrReg: loop addressing and validation limits

`create_addr_reg` supplies cumulative offsets per loop axis. It is neither a per-lane address index vector nor the UnalignReg created by `unalign_reg_for_store` to cache an unaligned tail. The limit is 8 registers and at most 4 loop axes. This library's `addr_reg_walk` verified offset advancement through two rounds of aligned transfers.

**AddrReg + `store_unalign` is an existing path, not a capability awaiting implementation.** Checked against framework `8faa85389`:

- The official `store_unalign.md` stride parameter table and AddrReg example explicitly include this combination; `store_unalign_post.md` also explains pairing. The “only supports load_align/store_align” statement in `create_addr_reg.md` conflicts with them and cannot be an unconditional rejection rule.
- `_vf_api.py` documentation and C++ `EmitVFStoreUnAlign` / `EmitVFStoreUnAlignPost` already have AddrReg branches, emitting `vstu` / `vsta`, respectively; `test_backend_cce_vf_ops.cpp` has corresponding code-generation tests.
- This library's `test_an_addr_reg_reaches_store_unalign_at_parse_time` guarantees only parser acceptance. The official example asserts output shape alone, so it is not evidence of numerical correctness either.

This library has not validated the combination's actual write range or values. Current unaligned-transfer needs are demonstrated by the scalar-stride and squeeze chains in `vf_unalign`. This records the validation boundary without creating a standing task to extend measurements or inferring unpredictable writes from lack of testing.

### 3.4. `histograms` `ACCUMULATE` is global accumulation; `bin_type` only selects a window

```
FREQUENCY:  dst[n] = |{v == n + 128*bin}|
ACCUMULATE: dst[n] = |{v <= n + 128*bin}|      ← Accumulation covers all of 0..255
```

Thus, **`BIN1` continues accumulating from where `BIN0` ended**. The halves form one continuous cumulative distribution over 0..255, and `BIN1`'s last bin is the total element count. Measurement (values `{0,32,128,192}`, each repeated 64 times): `BIN0` has `dst[0]=64`, while `BIN1` has `dst[0]=**192**` (= 128 low values + 64 occurrences of value 128).

The official wording said the nth element counts occurrences from 0 through n “within the corresponding interval,” suggesting interval-local accumulation. The resulting reference failure was revealing: **`BIN0` happened to pass; only `BIN1` mismatched**, because interval-local and global prefix sums coincide in the low half. Fixed in pypto (`afc5e0a49`).

Three further rules: src must be `DT_UINT8`, `preg` must have b8 granularity, and dst is `DT_UINT16` (128 bins, overflowing beyond 65535). **Initialize dst with `vf.full(0, ...)` first**: `ACCUMULATE` adds in place; without initialization, it adds to garbage **without any error**.

UINT16 has no host torch mapping (`_PL_DTYPE_TO_TORCH` has only 11 entries); `bit_cast` to `DT_INT16` before storing.

### 3.5. `clear_spr` does not clear `create_addr_reg` offsets

| Object | Does `clear_spr()` affect it? | Evidence |
| --- | --- | --- |
| AR state for `squeeze(STORE_REG)` / unaligned transfers | **Yes** | `vf_unalign`: omitting it silently fails |
| Loop offsets from `create_addr_reg` | **No** | `vf_spr_misc`: bitwise identical with and without it |

**Both are called “AddrReg” in documentation, but they are different objects.** A plausible explanation is that the latter's offsets are tied to loop axes and fixed at codegen, leaving nothing for `clear_spr` to clear. Fixed in pypto.

Do not use `clear_spr` to reset a loop address offset; the with/without control above does not show such a reset.

### 3.6. `mem_bar` has no numerical criterion; this is a finding, not skipped work

Barriers guarantee order, but **correct code may happen to compute correctly without them**. The official documentation also says hardware orders reads/writes of the same register, needing no synchronization instruction. Thus, a counterexample requiring failure when the barrier is removed is neither stable nor informative.

This example asserts only two defensible properties: all 12 combinations (`MemBarMode.__members__` measured exactly 12, matching the documented table) compile and run, and **adding a barrier does not change values**.

**Assert observable behavior, not the test's own prose.** A source-text search for “no numerical criterion” cannot validate memory ordering or replace the execution controls above.

### 3.7. Two additional gaps discovered

- **`MaskPattern` has 15 members** (measured via `__members__`): `ALL` `ALLF` `H` `Q` `M3` `M4` `VL1` `VL2` `VL3` `VL4` `VL8` `VL16` `VL32` `VL64` `VL128`. `vf_predicate` uses only `ALL`; **the other 14 have no example coverage of their semantics**. Owner: batch 6's `layout_table`, or an additional section in `vf_predicate`.
- **Inline helpers called inside kernel bodies accept only positional arguments**: `_tile_type(cols=2*N)` is rejected during parsing (`_bind_inline_arguments` in `_call_parser.py`: “only supports plain positional arguments”). Like “`pl.const`'s first argument must be literal” and “no list comprehensions,” this follows from **the kernel body being a DSL**. Conversely, **enum values can enter the kernel body from a closure** (measured IR: `bin_type="bin_type": 1`), so one factory creating N kernels differing only by enum is valid. The four single-mode baselines in `vf_spr_misc` use this pattern.

### 3.8. Measure a complete VF computation with real dependencies

[VF softmax](../../examples/basic_api/vf/vf_softmax/) demonstrates the register reduction/broadcast chain. Its benchmark carries each softmax result into the next iteration and orders the UB write/read dependency, so repeated work cannot be lifted out as an invariant. It validates each complete specialization before [native-task timing](../verification/native-timing.md). Host launch time, a short stream interval and an isolated instruction count are different measurements; none establishes a universal advantage for an API family. The workspace's VF requirement is an authoring policy.

## 4. Entry API destination registers and temporary resources

Historical measurements: 2026-09-27, Ascend950PR; documentation/signature fixes in pypto `f8f172bdc`, `59d8629f6`. Rerunnable entry point: [vf_entry](../../examples/basic_api/vf/vf_entry/test.py).

| API | Confirmed boundary |
| --- | --- |
| `vf.store` | count is optional and defaults to one full register; signature and official documentation are aligned |
| `vf.move` | Defaults to MERGING and supports only MERGING; with a partial mask, reuse an initialized destination name to verify retention of unselected lanes; unselected lanes of a new name are undefined |
| `vf.full` | Defaults to ZEROING and supports only ZEROING; Tensor-mode HIGHEST/LOWEST means last/first element, not bit order |
| UnalignReg in `vf.load/store` | Generated temporaries are consumed immediately after declaration; consecutive calls do not proportionally increase the live count |

Four simultaneously live UnalignRegs passed; five simultaneously live once caused a bisheng register-allocation segfault. Five nonoverlapping declarations, and one `vf.load` among four live registers, both passed. The gate therefore counts lifetimes, not total calls/declarations; see [tools](../tooling/checks.md).

A vf op producing a destination register cannot be nested inside another call's arguments; assign it first, then use it. Applicability is determined by `_VF_OP_DST_COUNT` and must not be generalized to every returning vf call, such as create_mask. The nested-call diagnostic was fixed in `f8f172bdc`.

### Destination registers used as inputs

The assignment target is an implicit operand of some VF operations. `dst = vf.mul_dst_add(a, b, preg)` computes `dst = old_dst * a + b`; `dst = vf.mul_add_dst(a, b, preg)` computes `dst = a * b + old_dst`. Initialize the **same target name** with the required multiplicand or addend before calling either operation. ZEROING mode clears inactive lanes; it does not initialize the active lanes read by the FMA.

For an online denominator update `l = alpha * l + rowsum(P)`, use:

```python
l_state = vf.load_align(l_prev, [r, 0])
l_state = vf.mul_dst_add(alpha, block_sum, preg)
vf.store_align(l_cur, l_state, preg, [r, 0])
```

Loading into `l_old` and assigning the FMA to a fresh `l_new` leaves `l_new` uninitialized. A mathematical `l_new = alpha * l_old + block_sum` formula does not make `l_old` an operand of that call. Inspect the emitted destination when adapting or renaming recurrence code; [ordinary aliases](#vf-register-aliasing-and-copying) do not create independent copies either. [vf_activation](../../examples/basic_api/vf/vf_activation/kernel.py) shows both accumulator positions; [decode_attention_gqa](../../examples/algorithms/decode_attention_gqa/kernel.py) carries the denominator through the same destination name.

The official mul_dst_add (`pypto/docs/zh/api/pro_api/SIMD-API/reg_computation/composite_computation/mul_dst_add.md`) / mul_add_dst (`pypto/docs/zh/api/pro_api/SIMD-API/reg_computation/composite_computation/mul_add_dst.md`) contracts and `_vf_api.py` specify the destination read/write. C++ `EmitVFMulDstAdd` emits `vmadd(dst, src0, src1, mask, ...)`; declaring a fresh `RegTensor` does not supply its old value. The [static rule](../tooling/checks.md#static-rules) rejects visibly unseeded targets but does not prove initialization on every control-flow path or active lane.

### VF register aliasing and copying

Use different names for a runtime scalar and the register that broadcasts it: `indices = vf.full(index, mask, dtype=UINT32)`, not `index = vf.full(index, mask, dtype=UINT32)`. A VF assignment writes its bound destination; a `dtype=` keyword does not turn an existing scalar binding into fresh register storage. This also matters when the scalar is a VF parameter. Trace the destination's first binding when an index has an unexpected dtype or a broadcast/gather returns a plausible but wrong value; casting the scalar alone can hide a type error while retaining the name collision.

At behavior version `872df763f`, a minimal compile comparison generated `RegTensor<uint32_t> index; vdup(index, index, ..., POS_LOWEST)` for the self-rebinding form, shadowing the incoming scalar and reading an uninitialized register. A distinct destination generated scalar-mode `vdup(indices, index, ..., MODE_ZEROING)` and passed dynamic gather indices 0, 1, 3, 17, 31 and 63 on Ascend950PR with CANN 9.2.0. The official full contract (`pypto/docs/zh/api/pro_api/SIMD-API/reg_computation/data_movement/full.md`) requires matching source/destination dtypes; it does not describe this binding defect. Installed `_assignment_parser.py::_parse_vf_assignment` reuses an existing destination, and C++ `vf.full` deduces its type from that destination. Inspect emitted operands before attributing such a failure to unsupported dynamic indexing. The [broadcast advisory](../tooling/checks.md#static-rules) flags possible scalar self-rebinding without forbidding an already initialized register's legal in-place broadcast.

States requiring independent updates must own separate registers. `second = first` creates an alias, not a snapshot of first's data; subsequent VF writes targeting second may also change first. The same applies to sentinels, identity elements, and old-state snapshots: do not assign one seed name to several accumulators and assume the seed remains unchanged. Initialize each with `vf.full`, or copy with a full-mask `vf.move` into a **fresh, unbound name**. A name already bound to a shared register does not automatically detach when assigned another VF result.

```python
first = vf.full(0.0, all_lanes, dtype=FP32)
second = vf.move(first, all_lanes)  # fresh destination, independent storage
saved = vf.move(first, all_lanes)   # snapshot before later in-place updates
```

This concerns storage identity, not just Python variable names. If independent states produce identical results, first inspect each write's destination register in generated IR/CCE, then inspect the formula. At behavior version `872df763f`, `_assignment_parser.py::_parse_vf_assignment` reuses registers for existing destination names; ordinary name assignment does not insert a vector copy. Native copy semantics: official move (`pypto/docs/zh/api/pro_api/SIMD-API/reg_computation/data_movement/move.md`).

`vf-shared-register-write` in `check.py rules` locates traceable writes to shared registers as a review hint. It does not prohibit intentional reuse: sharing storage may be intended when the old value is no longer needed; if both states must remain independent, create them separately at their first bindings.

Every LHS of a multi-output VF instruction is also a destination register. `_` remains an ordinary name, not a discarded output. Reusing `_` for results of different dtypes reuses the declared register; use different names for temporaries of different types.

Conversion results must also respect destination type. For example, creating a destination with `u = vf.unpack(..., dtype=UINT32)` then writing `u = vf.bit_cast(u, dtype=INT32)` is not array-style Python rebinding to a new type. At behavior version `872df763f`, CANN 9.2.0, it generates assignment of `RegTensor<int32_t>` to the existing `RegTensor<uint32_t>` and raises `no viable overloaded '='`. Use `i = vf.bit_cast(u, dtype=INT32)` instead; this does not make bit_cast a numerical copy into independent state. `vf-register-dtype-change` in `check.py rules` can preempt statically confirmable explicit type conflicts; see the [check contract](../tooling/checks.md) for scope.

#### Runtime branches and literal helper flags

A helper called with a literal flag may specialize away a branch. Replacing that flag with a runtime scalar introduces a control-flow merge that the original call did not test; preserve the literal form while adapting another contract feature, or verify both runtime arms before expanding the kernel.

When updating a VF register inside runtime branches or loops for later use outside the branch, express the update with a VF operation. `dst = vf.move(src, all_lanes)` copies a full register; ordinary Python assignment `dst = src` rebinds a name and must not be assumed to emit the same instruction. Use VF writes in every branch arm. Partial-mask `move` still follows the MERGING constraints above; unwritten lanes are not zero.

On 2026-10-06, at behavior version `872df763f`, CANN 9.2.0, a branch with `result = vf.select(...)` in one arm and `result = zeros` in the other generated a float-typed merge variable when stored after the branch. BiSheng reported `assigning to 'float' from incompatible type 'RegTensor<float>'`. Initializing `result` with `vf.full` before the branch alone still failed; changing the other arm to `vf.full` or full-mask `vf.move` made both variants pass 13 hardware tests each. This confirms failure for ordinary aliasing participating in this control-flow merge, not a blanket lack of VF branch support.

Evidence: copy semantics in official `SIMD-API/reg_computation/data_movement/move.md` and installed `_vf_api.py::move`; different handling of VF writes and ordinary assignment by Python `_control_flow_parser.py::collect_written_vars`; original generated CCE and hardware logs. Validation scope: behavior version `872df763f`, CANN 9.2.0.

### Numerical truncation is not integer type conversion

`vf.truncate` changes floating-point values while retaining FP16/BF16/FP32 type. Numerical conversion to integers uses an explicit destination type such as `vf.astype(..., dtype=pl.DT_INT32)`. `vf.bit_cast` is separate: it reinterprets the same bits rather than converting values. Ordinary `NORM` stores should match numerical types across the register, UB tile, and GM Tensor; do not expect implicit numerical conversion during storage. Explicit packing stores follow the container-width and bit-pattern contract in [port bit packing](#distinguish-numerical-conversion-from-port-bit-packing-first), allowing different register and destination element widths.

In a probe on 2026-10-06 at behavior version `872df763f`, CANN 9.2.0, the FP32 value 10.0 produced by `truncate` and written into INT32 output read back as `1092616192` (0x41200000), not 10. The official `type_conversion/truncate.md` explicitly says it retains the original data type; installed `_vf_api.py` defines it mathematically as `trunc(src[i])`, and C++ `EmitVFTruncate` checks that source and destination types match. Such errors can compile and launch; absence of an error does not replace type and numerical validation.
