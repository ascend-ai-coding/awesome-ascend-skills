# Tuple Unpacking Flattens the Memory Spaces of Two Tiles, with the Error Pointing to Generated `.cpp`

| | |
| --- | --- |
| Status | **Fixed** (2026-09-27) in `framework/src/interface/ir/structural_equal.cpp`. |
| Impact | Before the fix, compilation failed and **the error could not be traced back to source**—the line number in the error referred to generated `kernel.cpp`. |
| Toolchain | Ascend950PR_957b / CANN 9.2.0 / pypto 0.2.1 |
| Root cause | The `TileType` branch of `EqualType` did not compare `memref_`, so memory space was excluded from type equality. |
| Regression | UT `tests/ut/pypto_pro/ir/test_tile_type_memory_space_equality.py`; ST `tests/st/pypto_pro/frontend/datacopy/test_tuple_unpack_memory_space.py`. |

## 1. Symptom

```python
fp_mat, fp_tile = scale_mat.current(), scale_fb.current()
```

`scale_mat` is `[1, 64] INT64` in `MemorySpace.Mat`; `scale_fb` is the **same `[1, 64] INT64`** in `MemorySpace.Scaling`. bisheng reports:

```
./build/store_fp16__a5__d0/tk_none/kernel.cpp:89:164: error: no viable conversion from
  'Tile<TileType::Scaling aka 6, [9 * ...]>' to 'Tile<TileType::Mat aka 1, [9 * ...]>'
```

This is followed by `no matching function for call to 'TMOV'` / `'TSTORE'`; both are downstream of the same cause. The Python side reports only `Failed to compile kernel 'store_fp16'` (`jit.py:2465`).

**All line numbers in the errors refer to generated files**; the corresponding line in the kernel source appears in none of the messages.

## 2. Generated Code

```cpp
Tile<TileType::Mat, int64_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
     PadValue::Null> _tuple_tmp_3_0[] = {_tg___inline_1_scale_mat_tiles_0[0],
                                         _tg___inline_1_scale_fb_tiles_0[0]};
```

Tuple unpacking introduces a `_tuple_tmp_N` during parsing (`_assignment_parser.py:303`); codegen lowers it to a **C array** whose element type is taken from the **first element**. Apart from `TileType`, both tiles have identical template parameters, so the “same type” check—which compares dtype / rows and columns / layout / fractal—allows them in one array.

The template arguments for the later `TMOV<Tile<Scaling>, Tile<Mat>>(_tuple_tmp_3_0[1], _tuple_tmp_3_0[0])` are **correct** (the types remain intact in IR); only the array element type is wrong. **The type-equality check at lowering omitted memory space; the IR is not at fault.**

Control: `cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()` (Mat / Mat / Acc, with different dtypes) does not trigger it; no `_tuple_tmp` is generated for this assignment in the same file.

## 3. Minimal Reproduction

```python
@pl.jit(auto_mutex=True)
def repro(a: pl.Tensor[[64, 64], pl.DT_FP16], scales: pl.Tensor[[1, 64], pl.DT_INT64],
          out: pl.Tensor[[64, 64], pl.DT_FP16]):
    mat = pl.make_tile_group(
        type=pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Mat,
                         layout=pl.TensorLayout.ND),
        addrs=0x0000, mutex_ids="auto", depth=1)
    fb = pl.make_tile_group(
        type=pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling),
        addrs=0x0000, mutex_ids="auto", depth=1)
    with pl.section_cube():
        x, y = mat.current(), fb.current()      # ← this line
        pl.load(x, scales, [0, 0])
        pl.move(y, x)
```

Splitting this into two lines, `x = mat.current()` / `y = fb.current()`, succeeds.

## 4. Workaround

**Do not unpack two Tiles that differ only in memory space in the same tuple assignment.** Split them into two lines at no cost. This is documented in [`api/constraints.md`](../../api/constraints.md).

## 5. Root Cause and Fix (Located and Fixed on 2026-09-27)

**The initial guess was half right: type equality did omit memory space, but the bug was in `structural_equal`, not codegen.**

`CCECodegen::IsHomogeneousTuple` in `cce_codegen.cpp` compares element types using `ir::structural_equal`, while the `TileType` branch of `EqualType` is **handwritten**: it compares dtype, shape, `tileView_`, and `hardwareInfo_`, but **not `memref_`**—even though `memref_` is one of `ShapedType`'s reflected fields and `MemRef::GetFieldDescriptors` includes `memory_space`. The handwritten comparison omitted it.

First, isolate this with a probe (before the fix):

```
Mat vs Scaling    : True     ← this is the bug
Mat@0 vs Mat@4096 : True     ← this is correct and must not change
```

The fix adds one comparison in that branch, **deliberately narrowly scoped**:

- **Compare only memory space, not addr / size.** Space is part of the type (codegen lowers it to the `TileType::` template argument), while addr / size belong to an allocation instance. Including them in equality would make adjacent tiles in the same tile group different types.
- **Compare only when both sides have a memref.** Tiles without a memref (IfStmt return values and intermediate tiles) do not yet have a decided buffer, so preserve the old behavior for them.

Probe after the fix:

```
Mat vs Scaling    : False    ← fixed
Mat@0 vs Mat@4096 : True     ← unchanged
Mat vs bare       : True     ← unchanged
```

`structural_hash` was **not changed**: equality is stricter now, while the hash is unchanged. Two tiles in the same space may still have the same hash and compare unequal; this is a valid hash collision, not an inconsistency.

Regression: the full `ut/pypto_pro` suite had 2518 passed on board; see §6 below for `st/pypto_pro/frontend`.

## 6. The Two Options Originally Listed

Ordered by cost (option 1 is the one adopted above):

1. **Include memory space in type equality during lowering** (the complete fix). Two tiles whose only difference is `TileType` should not share an array; lower each element separately, as heterogeneous tuples already do.
2. **Fallback: reject during parsing.** If `_parse_tuple_unpacking` finds two Tiles in a tuple with the same dtype / shape / layout but different memory spaces, raise an error with a source span and add “split into two lines” as a hint. This is inexpensive and **immediately fixes the “error cannot be traced to source” problem**—under [skill constraints](../../../SKILL.md#constraints-and-workspace-policy), error-message quality itself is a foundation issue.

Option 2 would still be useful after option 1: it would reject before compilation something that can be identified during parsing. **But after option 1, it no longer has a valid target**—this combination is now legal and must not be rejected.
