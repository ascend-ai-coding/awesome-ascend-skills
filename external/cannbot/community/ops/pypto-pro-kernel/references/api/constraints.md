# Rejected patterns when writing pypto_pro kernels

This document routes rejected forms and behavior boundaries to their owners. Each entry distinguishes parser diagnostics from silent-failure observations and their conditions; [diagnostics/validation-traps.md](../diagnostics/validation-traps.md) explains additional cases that pass silently.

These restrictions are easy to work around, but each encounter costs a compilation cycle. Original error messages are included to help identify the cause directly.

---

## Type aliases cannot be used as annotations

```python
T_A = pl.Tensor[[M, K], pl.DT_FP16]

@pl.jit
def kernel(a: T_A):          # InvalidType: Unsupported type annotation
```

Annotations must spell out `pl.Tensor[[...], ...]` literally. **Module-level int constants can** appear in shapes (`[M, 2 * K]` works); the alias itself is what gets rejected.

## Inline helpers accept only positional arguments

The parser inlines ordinary Python functions called inside a kernel body. **Their call sites cannot use keyword arguments**:

```
NotSupported: Call to inline function '_mat' only supports plain positional arguments
```

```python
_mat([1, N], [2], 0x20000, dtype=pl.DT_FP32)   # Rejected
_mat([1, N], [2], 0x20000, pl.DT_FP32)         # Accepted
```

**Defaults on the def are unaffected**: `def _mat(shape, ids, addr, dtype=pl.DT_FP16)` works; only `dtype=` at the call site is prohibited.

The definition itself also accepts only positional parameters with optional defaults. Keyword-only parameters such as `def _tile(dtype, cols, *, padded=False)` are rejected, so changing only the call site is insufficient. The current error is `Inline function '_tile' only supports positional parameters with optional defaults`; `_call_parser.py::_bind_inline_arguments` checks `kwonlyargs`, variadic parameters, and keyword calls separately. This applies only to ordinary Python inline helpers, not keyword arguments of native APIs such as `pl.TileType(shape=...)`. Verified at behavior version `872df763f`.

Similarly, forms such as `dict(**kw)` do not work (`Unsupported function call: dict(...)`); write the declaration out explicitly.

## `for` must use `pl.range()`

```
NotSupported: For loop must use pl.range()
```

Built-in `range()` is allowed only in the Python layer **outside** the kernel, such as in a factory function. The rejection is at `_control_flow_parser.py:988`; the three-argument form `pl.range(start, stop, step)` is available. **Iterating over a static Python list (`for off in [0, 64, 128, 192]`) is also rejected**, even when its contents are compile-time constants. Every `for` in a kernel body must use `pl.range` (measured on 2026-09-28; also applies inside `@pl.vector_function`; nested `pl.range` has hardware evidence).

Keep VF loop counters separate from logical coordinates. The official range contract (`pypto/docs/zh/api/pro_api/SIMD-API/controlflow/range.md`) requires VF start/stop values in `[0,65535]`, a positive step, and no overflow even on the final increment; ranges outside VF differ. The installed parser's `_control_flow_parser.py::_check_range_bound` and `_check_range_overflow` at behavior version `872df763f` match this restriction. A negative start produced `pl.range() start must be in [0,65535]`. For negative neighborhood offsets, enumerate nonnegative indices and derive signed logical offsets from them; do not use mathematical coordinates directly as VF loop counters. Callers must also guarantee the range of dynamic bounds; absence of a static rejection does not establish validity.

## `pl.load` destinations must be UB or L1

```
InvalidOperation: load: dst tile must be in Vec (UB) or Mat (L1) memory, got Acc
```

Thus, **an accumulator cannot be initialized from global memory**. To test `init=False`, use the first call to write and the second to accumulate.

## Move VF scalar division to the outer scope

Writing `1.0 / width` inside `@pl.vector_function` goes through DSL parsing. Even when both operands are constants, Python is not guaranteed to fold it first. The current `_expression_parser.py::_check_vf_scalar_division` explicitly rejects scalar `/`, and `//` or `%` with floating-point operands, with `Scalar division ('/', or float '//', '%') is not supported inside a vector function`.

Precompute reciprocals or normalization coefficients for static dimensions in the Python factory, for example `inv_width = 1.0 / width`, and use them inside VF via `vf.muls(value, inv_width, mask)`. Evaluate runtime scalars in the outer kernel as directed by the diagnostic. This does not restrict integer scalar `//` or `%`, or the register form `vf.div`. The official `reg_computation.md` says that VF supports scalar expressions; verify the exact boundaries against the installed version. The rejection and the static-coefficient rewrite above were verified at behavior version `872df763f`.

## SIMD VF functions communicate through tile parameters

Keep register initialization, arithmetic, reductions and register load/store inside the default SIMD `@pl.vector_function`. The enclosing `@pl.jit` function handles tiles, transfers and VF calls; `section_vector()` alone does not establish VF scope. A SIMD VF procedure writes through its tile parameters and contains no `return`, even a bare return. Keep dependent register calculations within the procedure, or explicitly store state to a tile for the next VF call. Returning a `RegTensor` to JIT code is not a way to carry reduction state across those scopes.

The official Reg computation guide (`pypto/docs/zh/guide/programming_guide/pro/development/vector_computation/reg_computation.md`) defines the scope and procedure-shaped examples. `_control_flow_parser.py::parse_return` rejects a return while `inline_vf_depth` is nonzero; `_call_parser.py::_implicit_func_call` inlines VF procedures without ordinary-helper return lowering. The parser rejection was reproduced on behavior `872df763f`. This boundary is specific to SIMD VF: ordinary inline helpers and `vector_function(mode="simt")` have separate return contracts and must not inherit this restriction.

## Do not pass `compact=None` explicitly

`pl.TileType(..., compact=None)` is rejected (`InvalidType`), even though omitting this kwarg is equivalent to `None`. Consequently, **a helper cannot simply forward `compact` as a parameter**: write two versions (`_left` / `_left_c`), or inline the declaration in the kernel.

## NZ Acc tile column counts must be multiples of 16

```
InvalidVal: Tile with NZ layout (fractal=1024) requires Cols (24) to be a multiple of
inner box cols (16), got remainder 8. Pad Cols to a multiple of 16.
```

When slicing an accumulator into strips by address, **round the last strip's width up to 16** rather than declaring the exact remaining column count. Extra columns outside the valid region do not affect the result: store reads the narrowed parent handle.

## Annotate scalar parameters directly with a dtype

Use a concrete dtype for `@pl.jit` scalar parameters, such as `scale: pl.DT_FP32` or `count: pl.DT_INT64`. `pl.Scalar` is the DSL scalar value class; using it alone as a parameter annotation raises `Incomplete type annotation: pl.Scalar`. Do not change it to `pl.Scalar[...]` either. Tensor and Ptr parameters use complete declarations of `pl.Tensor[[shape], dtype]` and `pl.Ptr[dtype]`, respectively.

Distinguish the complete parameter annotation from its dtype subexpression: even if `FP32 = pl.DT_FP32` is defined, `epsilon: FP32` still raises `Unsupported type annotation: FP32`; write `epsilon: pl.DT_FP32`. In contrast, the alias in `pl.Tensor[[M,N], FP32]` is resolved by `resolve_dtype` and is supported. At behavior version `872df763f`, `resolve_type` requires a dtype attribute expression for a complete scalar annotation; the two entry points are not equivalent. Named TilingData classes follow a separate path, so this rule must not reject all name annotations.

The official kernel parameter documentation (`pypto/docs/zh/guide/programming_guide/pro/development/kernel_function.md`) shows the dtype form. The installed parser's `_type_resolver.py::resolve_type` at behavior version `872df763f` agrees with actual parameter calls. Scalar values produced inside a kernel by `getval`, shape, or block APIs are a different use case and do not replace parameter declarations for caller-supplied scalars.

## Bind `pl.gemm` operands to variables first

```
NotSupported: pl.gemm() needs the operand 'splitn' works on bound to a variable
```

```python
pl.gemm(acc.next(), ...)              # Rejected
cur_acc = acc.next(); pl.gemm(cur_acc, ...)   # Accepted
```

Lowering needs the operands' declared shapes, and **evaluating the expression again would advance the TileGroup by another slot**. Which operands need binding depends on the parameters used: `splitn` reads the accumulator; `trans_a` / `trans_b` read the corresponding L1 operand; `m` / `n` / `k` read all three.

## `pl.gemm` `init` / `trans_a` / `trans_b` cannot be runtime values

```python
for ki in pl.range(0, kt, TILE):
    pl.gemm(acc, a, b, l0a=l0a, l0b=l0b, init=(ki == 0))
# InvalidType: pl.gemm() 'init' must be a compile-time bool, got 'ki == 0'
#   hint: It selects which instructions are emitted, so it cannot be a runtime value
```

`init` selects the emitted instruction (`True` → `matmul`, `False` → `matmul_acc`), so use a two-branch `if ki == 0:` inside the K loop. Keep transfers identical in both branches to preserve balanced ping-pong rotation. Template: [`dynamic_shape`](../../examples/matmul/fundamental_usage/dynamic_shape/kernel.py).

## `pl.gemm` `splitk` / `splitn` make the corresponding `k` / `n` compile-time values

Thus, **dynamic shapes cannot use split**: `splitk` turns `k` into a loop bound, which must be known at compile time. Write a dynamic K loop explicitly.

The three **narrowing** parameters `m` / `n` / `k` do not have this restriction and explicitly accept runtime values (`pl.gemm` docstring: *"An extent may be a runtime value, unlike a split size"*). Also, `splitn` and `m` are mutually exclusive; the rejection hint includes measured results.

## Tiles narrowed by `pl.gemm` require `compact=1`

```
InvalidVal: pl.gemm() needs 'l0a' declared with compact=1 to take an effective extent
```

The roles are asymmetric; do not infer them from intuition: `m` narrows `l0a` and `acc`, `n` narrows `l0b` and `acc`, and `k` narrows `l0a` and `l0b` (K is reduced away and does not enter the output shape, so it does not affect `acc`). gemm deliberately leaves L1 operands untouched; narrowing them is `pl.load`'s responsibility.

This is an explicit **rejection**, but the error it prevents is silent. The `_gemm_require_compact` docstring records measurements (Ascend950PR, 48×40 in a 64×64 Tile): error about 50 with neither compact, about 47 with only the accumulator compact, about 62 with only the L0 temporaries compact, and 1.7e-05 with both compact. **Every case respected the bounds and wrote nothing outside the region**; values were merely placed incorrectly inside it, so no later check could catch this.

**`pl.gemm_mx` planes do not require `compact=1`**, but that does not make `valid_shape` ineffective: narrowing K must also narrow the planes' group axes. The lack of a measured effect along M cannot be generalized to K. See “Effective plane dimensions” in [api/gemm-mx.md](gemm-mx.md).

## Naming sanitizer variants (framework-conflict workaround, hardware evidence 2026-09-29)

Passing `pl.jit(sanitizer=True)` together with `name=` fails: the instrumentation pass identifies the entry point by “function name == program name” (`name=` changes the program name while the function name remains the `def` name). A mismatch **silently skips instrumentation**. The launch wrapper checks only the sanitizer boolean and unconditionally appends 2 hidden arguments, producing a signature/argument mismatch and `Expected N args, got N+2` at launch. The library's factory convention (all inner functions named `kernel` + a unique `name=` for each variant) always triggers this. Starting at `b1c683973` on the pypto dev branch, this combination is rejected during decoration with `InvalidArgument` rather than failing silently; the upstream version still has the silent form. The workarounds remain the same, and neither modifies the framework:

1. **Minimal workaround**: omit `name=` for the sanitizer variant (add a `sanitizer=` factory parameter: `_make_kernel(preload=0, name=None, sanitizer=True)`). Secondary trap: the program name falls back to the function name `kernel`, and **the build directory is also `kernel__<arch>__<dev>`**. Sanitizer variants of multiple kernels therefore share the on-disk artifact directory, overwriting sequentially or interfering concurrently; this is harmless for a single kernel.
2. **Complete workaround**: bypass the factory for the sanitizer variant and write a top-level `def` with a unique function name, such as `def my_kernel_san(...)`, without `name=`. The program name automatically matches the unique function name, making pass matching, the build directory, and the cache key unique together.

Example: the end of [examples/algorithms/rmsnorm_gemm_tail](../../examples/algorithms/rmsnorm_gemm_tail/kernel.py) (workaround 1). If the installed pass accepts an invalid launch form, report it with a minimal reproducer; validation should reuse the launch entry-point criterion.

## Multiple cores + nondivisible M: propagate runtime row counts through the full chain (teaching recipe)

The [rmsnorm_gemm_tail example](../../examples/algorithms/rmsnorm_gemm_tail/kernel.py) uses M=1000: eight groups each own 125 rows, split into 64/61-row bands and 32/32 or 32/29-row AIV regions. Each stage computes runtime scalars (`band_rows = pl.min(BAND_ROWS, rows_per_core - band_off)` and the current AIV row count `rows`), then **propagates them through six steps**: `pl.set_validshape(x_tile, [rows, K])` bounds the GM→UB transfer → `pl.load` → the VF row loop uses `pl.range(rows)` → `set_validshape(nz, [rows, ...])` followed by `pl.insert` → `pl.gemm(..., m=band_rows)` (with `compact=1` as above) → `pl.store` writes the accumulator's valid rows. AIV insert offsets remain multiples of the 16-row fractal. Physical shapes and explicit slot addresses reserve full capacity; effective extents bound the work. Padding the logical input/output tensors to alignment without actual backing storage would access outside them; see the distinct guard, poison and sanitizer criteria in [check coverage](../diagnostics/check-coverage.md#5-three-distinct-criteria-for-out-of-bounds-access-each-with-its-own-blind-spot-board-evidence-2026-09-29).

## `pl.gemm_mx` rejects `bias` / `splitn` / `splitk` by name

```
NotSupported: pl.gemm_mx() does not split: a K step would have to cut both planes
              along their group axis, and a group is indivisible
```

These are not “unrecognized keyword” errors: all three provide reasons and alternatives. `bias` is rejected because `pl.matmul_mx` has no such operand; both split parameters are rejected because splitting must cut along the planes' group axes, and groups are indivisible. Write the loop explicitly and call `pl.gemm_mx` once per step instead.

## `pl.gemm_mx` `k` must be a multiple of 64

```
InvalidVal: pl.gemm_mx() 'k' is 32, which is not a multiple of 64
```

**This is a measured hardware requirement**: `k=32` and `k=96` (both **whole numbers of groups**) each caused AICore trap **507015**, whereas `k=64` / `k=128` matched bit for bit. The unit is therefore **64**, not 32. The explanation that “the remainder has no group to occupy” was wrong; whole groups still crash.

**Declared K** is always checked. **Supplied `k`** is checked when it is a literal **or a name** (a module constant such as `k=TILE`, the usual form in this repository). More complex expressions are not checked: parsing `pl.min(TILE, mt - mi)` would emit its IR once, and the expanded `set_validshape` would parse it again, emitting it twice. This check therefore cannot establish the validity of runtime `k`.

When narrowing `k`, the sugar **narrows both planes' group axes** to `k // 32`. Leaving the group axes at full size produced an error of 183.2 (narrowing them on the same card restored bitwise equality), so this is mandatory.

The five positional parameters are **positional-only**: `pl.gemm_mx(acc, l1a, l1b, l1a_scale, l1b_scale, l0a=..., l0b=..., scale_a=..., scale_b=...)`. Positional arguments are caller-loaded tiles; keyword arguments are temporaries written by the sugar. Template: [`gemm_mx`](../../examples/matmul/gemm_mx/kernel.py).

## MX Scale Planes Require Explicit Paired Addresses

For every slot, declare a ScaleLeft/ScaleRight address equal to the paired Left/Right data address `>> 4`, with matching depth. L1 scale views use `layout=pl.ZZ` / `pl.NN`; their Mat space alone does not determine the required layout. The parser checks each paired slot in `_check_mx_plane_slots`. See [MX scale planes](memory-allocation.md#mx-scale-planes).

---

## `wait_cross_core` `pipe` must be the **consumer's** pipe (no error is raised)

The first entry in the official `wait_cross_core.md` parameter table defines `pipe` as “the hardware pipe blocked during the wait” and says it “need not match the paired `set_cross_core` pipe.” It therefore blocks **the pipe you specify**: MTE3 for a `pl.store` consumer, V for `vf`.

**A wrong value fails silently**, with the symptom “passes alone, fails when run together” (measured maximum relative difference 7.6, with periodic displacement). See “Three rules for handwritten layer C,” §1 in [api/crosscore-sync.md](crosscore-sync.md), for both endpoints, and [api/fixpipe.md](fixpipe.md) §15 for the hardware investigation. The `MTE3` waits in examples with `pl.store` consumers cannot be copied unchanged to a VF consumer; choose the first consuming pipe from the actual dataflow.

## Match widening / narrowing `vf.astype` layout to source positions and predicate bits

`CastLayout.ONE` is not generally unsupported. For compact FP16→FP32 widening, `ZERO` selects even source halfwords and `ONE` selects odd ones. A b32 predicate enables representative bits `4*i`; the odd halfwords instead consult `4*i+2`, so blindly reusing that predicate for `ONE` can yield zeros. With `UNPK_B16`, consecutive source values occupy the low halves of b32 containers; `ZERO` with the matching b32 predicate preserves their order. The [physical-position and predicate contract](vf.md#213-the-fp32-path-for-fp16-source-data-order-preserving-widening-with-unpk-required-reading) owns these cases and their version scope.

The [INT8/INT16 probe in Fixpipe §15](fixpipe.md#15-the-hf8-cell-neither-of-two-failures-was-an-hf8-problem) observed zeros for nonzero layouts, but its recorded conditions do not establish a general layout restriction. Check the supported conversion pair, loaded register arrangement and selected predicate bits before diagnosing unavailable lanes.

For narrowed outputs, choose the store from its physical container map: [port packing](vf.md#distinguish-numerical-conversion-from-port-bit-packing-first) can emit dense ND output directly, while [de-interleave packing](vf.md#210-narrowing-astype-occupies-only-half-the-width-pair-two-chunks-and-de_interleave-before-storing) serves consumers needing a dense register. A valid layout selector alone does not establish a dense write.

## Kernel definition modules cannot use `from __future__ import annotations`

```python
from __future__ import annotations   # ✗ Turns every annotation into a string
...
kernel[None, 1](a, b, out)           # TypeError: 'function' object is not subscriptable
```

`pl.jit` identifies Tile kernels through **actual objects** in parameter annotations (`_is_tile_kernel`'s isinstance check). Once future annotations turn them into strings, jit **silently returns an ordinary function wrapper** through the legacy path; only bracket launch fails. The symptom and root cause are one layer apart; hardware evidence dated 2026-09-29. It is fine in test/bench files, but **not in modules defining `@pl.jit` kernels**.

## Variable names inside `@pl.vector_function` have a single type environment; masks and data cannot share names

```python
m0 = vf.lt(idx, 16.0, allm)      # m0: MaskReg(FP16)
...
m0 = vf.reduce_max(c0, preg)     # ✗ At codegen: vf.reduce_max requires src and dst
                                 #   to have the same type, got dst=DT_FP16 src=DT_FP32
```

Assigning to the same name is not Python rebinding: the parser's type environment records types by name, so a later assignment conflicts with the type already associated with that name (hardware evidence 2026-09-29; the codegen error looks like a different type error, one layer removed from the actual cause). Give masks a separate naming prefix, such as `sm0`.

## Layer B `pl.insert` accessing a crosscore buffer requires a registered tile group parameter as its source

The result of `pl.reinterpret` is absent from the crosscore scanner's declaration registry: `InvalidOperation: cannot determine the pipe of pl.insert ... its source tile does not resolve to a declared tile group`. To use a different layout view for insert, **declare the buffer directly with the target layout** (the write side can still scatter using physical linear offsets; linear offsets into an NZ tile follow physical order), or pass the group as a parameter of the stage itself, as the diagnostic suggests (hardware evidence 2026-09-29).

## `pl.reinterpret` rejects tiles from dynamic slots; use group-level reinterpret

Applying `pl.reinterpret` to an **individual tile** obtained from `group.next()/current()/previous()` (a multi-slot group) or a runtime subscript `group[i]` is rejected at parse time (`NotSupported`, starting at pypto `d4dadf181`; the hint shows the group-level form). **Earlier builds had a silent defect for the same form**: the result always bound to **slot 0**, with no diagnostic; generated code was `auto& v1_0 = v0_0;` (hardware evidence 2026-09-29: the second double-buffer slot read the first slot's data; both views had address 0 in IR, whereas constant subscripts / single-slot groups / group-level forms each retained their respective addresses). Use **group-level** `pl.reinterpret(group, ...)` to obtain a view group, then `.next()` (this is what “Redeclare Tile views” in the official tile_creation_and_operations.md means). Constant-index slots `group[i]`, whose addresses resolve at compile time, are unrestricted.

**`pl.gemm` `trans_a` / `trans_b` have the same cause and rejection**: the transpose alias is a single-tile reinterpret, so dynamic-slot operands are also rejected (the diagnostic uses gemm terminology, starting at `d4dadf181`; earlier builds silently read slot 0 through the transpose view, causing wrong values during double-buffer rotation). Choose one alternative: use constant-index slots as operands, or create a **group-level ZN view** instead of `trans_x`: after `gv = pl.reinterpret(group, shape=[cols, rows], layout=pl.ZN)`, pass `gv.next()` without `trans_x` (NZ `[r,c]` and ZN `[c,r]` use the same bytes; the alias is the transpose). Group-level example: [quant_gemm_w8a16](../../examples/algorithms/quant_gemm_w8a16/kernel.py). Regression entry points: rejection cases in pypto `ut/pypto_pro/ir/op/test_reinterpret.py` and `ut/pypto_pro/language/parser/test_gemm.py`.

## Feeding VF results to cube (v→c return path): cursor vsstb scatter to NZ with row+1 pitch

Teaching recipe (hardware evidence, example [examples/algorithms/mm_softmax_mm](../../examples/algorithms/mm_softmax_mm/kernel.py)): register an NZ declaration of `[rows+1, 128]` for each 128 columns. Inside the VF row loop, issue one `vf.store_align(nz, reg, full_width_mask, data_copy_mode=DATA_BLOCK_COPY, block_stride=rows+1, repeat_stride=1, post_update=True)` for each dense FP16 register. **Do not supply an explicit offset**: that combination is a framework gap causing compilation errors or silently incorrect instructions. In the stage, call `pl.set_validshape(nz, [rows, 128])` and then `pl.insert` with column offsets. Compared with the three-step “ND store → `pl.move` → insert” path, this saves one intermediate group and an entire move; the odd pitch separates UB banks. Cursor semantics, gap details, and tested-and-rejected alternatives are in the “row+1 pitch convention” section of [api/crosscore-sync.md](crosscore-sync.md), which owns the mechanism explanation.

## `vf.astype` requires `dtype` as a keyword argument

```python
h = vf.astype(p, preg, FP16)          # ✗ vf.astype requires explicit dtype kwarg
h = vf.astype(p, preg, dtype=FP16)    # ✓
```

The stub signature `astype(src, preg, dtype: DType, ...)` does not reveal this constraint: it shows an ordinary parameter, but the parser is stricter (hardware evidence 2026-09-29).

## `pl.move` does not narrow dtype for same-space Vec→Vec transfers

```python
pl.move(vec_fp16, vec_fp32)   # ✗ InvalidType: move: unsupported layout/dtype combination
```

A transfer with the same Vec space and ND layout but a different dtype is rejected (hardware evidence 2026-09-29). Narrow at the VF layer (recipe in [api/vf.md](vf.md) §2.10); do not expect a transfer to convert the type as a side effect.

## Tuple-unpacking two Tiles differing only in memory space: fixed (2026-09-27)

Fixed in `framework/src/interface/ir/structural_equal.cpp`. This line is now valid:

```python
fp_mat, fp_tile = scale_mat.current(), scale_fb.current()   # One in Mat, one in Scaling
```

Older versions failed compilation, with **line numbers pointing into generated `.cpp` rather than kernel source**; splitting it into two lines was the workaround. See the [defect](../diagnostics/defects/tuple-unpack-collapses-tile-memory-space.md) for the original error, generated C array, and root cause. The `fixpipe_scaling_buffer` example still uses two lines to work on older versions.

## Parameter Type Annotations Must Keep Their Native Syntax

The complete annotation cannot be a variable, while dtype and layout inside a native Tensor annotation may come from a closure:

```python
ann = pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dt, lay]
def kernel(out: ann):                                       # unsupported annotation
def kernel(out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dt, lay]):  # dtype/layout from closure
```

Whether an enum may come from a closure depends on its position. Inspect the relevant parser or type resolver; do not infer a blanket restriction from a different argument.


---

## FP4 tensor shapes count logical elements; the host's last axis counts packed storage units

A fixed declaration of `[64, 64]` corresponds to a host byte carrier of `[64, 32]`. Both static and dynamic last axes are supported; `172d4d915` fixed older versions' inconsistent cold/hot call behavior. See the [contract and regression](../diagnostics/defects/subbyte-static-dim-order-dependent.md). Explicit `static_shapes` use logical shapes and must not be divided by two again. Tiles also use logical shapes, with byte counts computed by the framework from bit width.

---

## Key-dependent parameter types require a pointer and an in-body Tensor declaration

```python
class MatmulKey:
    Dtype = TilingKeyField(bits=1, values=[0, 1])

@pl.jit(tiling_key=MatmulKey)
def kernel(a: pl.Tensor[[M, K], io_dtype]):    # Cannot express this: io_dtype depends on the key
```

`Dtype` is used as a **free variable** in the function body and folded into a constant by the parser. Its value is unavailable before binding, so a dtype depending on it cannot appear in an annotation. For that key-dependent interface, declare the parameter as `pl.Ptr[pl.DT_UINT8]` and supply the type in the body through `pl.make_tensor(ptr, shape, dtype=io_dtype)`.

A key that changes only internal tile capacity or algorithm selection does not require pointer parameters. Keep ordinary `pl.Tensor[[pl.DYNAMIC, ...], dtype]` annotations when their type can be expressed independently of the key. The framework's multi-field tiling-key test (`pypto/python/tests/st/pypto_pro/frontend/system/tiling_key/test_tk_08_multi_field_combination.py`) demonstrates that interface; its tile arithmetic is not an authoring-policy exception. Separate the public type contract from internal specialization using the [generalization criteria](../workflows/authoring.md#generalization-and-specialization).

Two related points:

- Put `# ruff: noqa: F821` at the top of the file: `Dtype` is undefined from the linter's perspective.
- A key value can be used directly as a number (`tile = TileBlock`); it need not be written as `if TileBlock == 0: tile = 64`. Both forms were measured to fold into IR literals.

## Omitting the key dictionary at launch does not select defaults

```python
kernel[None, 1](a, b, out)
# InvalidArgument: Expected kernel[stream, block_dim, tilingkey]
```

The third subscript is a concrete key dictionary: `kernel[None, 1, {"Dtype": 0, "TileBlock": 64}](a, b, out)`. Note that calling `to_kernel_def()` without a key **is allowed**; the key is mandatory only at launch.

## `is_valid` cannot read attributes from `self`

```python
class Key:
    Dtype = TilingKeyField(bits=1, values=[0, 1])
    def __init__(self):
        self.allowed = {0}
    def is_valid(self, key):
        return key[0] in self.allowed      # AttributeError: no attribute 'allowed'
```

`TilingKeySchema` creates an instance with `object.__new__(cls)` and **does not call `__init__`** (`tilingkey.py:90`), so attributes set in `__init__` do not exist. The method must be a pure predicate of `key`.

Schema construction **does not raise an error**: it binds the method without calling it. Failure occurs when the method is actually invoked (`enumerate_valid()` / `validate_concrete()`).

## On-the-fly quantization `scale` is fp19, truncated toward zero (no error)

For `scale=0.1`, hardware actually uses `0.0999755859375`, a relative difference of 2.4e-5. **This error is introduced by fixpipe, not accumulation**. Choosing powers of 2 for scale in test criteria permits `atol=0`; for high-precision scaling, move to UB and use `vf.muls`. See [api/fixpipe.md](fixpipe.md) §17 for two independent sources of evidence (a hardware sweep of `1 + 2^-k` and the ISA's `& 0xFFFFE000u`) and the measurement method.

## Besides `scale`, `quant_offset` adds an offset before saturation

```python
pl.store(out, ac, [0, 0], scale=0.5, quant_offset=5)    # Scaled value 32 becomes 37
pl.store(out, ac, [0, 0], scale=0.5, quant_offset=100)  # 132 exceeds range → 127
```

It shares the configuration word with scale (bit37..45, **9-bit two's complement**), so:

- Its range is `[-256, 255]`; out-of-range values raise `InvalidVal`.
- **It works only with a compile-time `float` / `int` scale**. For a runtime Scalar or per-channel Tile, OR the offset into the integer yourself, or write it into every Tile element. The framework rejects unsupported combinations instead of silently ignoring them.
- **It affects integer destinations only**; floating-point destinations raise an error.

Do not confuse it with `pl.store` `offsets` or `pl.move` `offset`: those are **Tensor/Tile coordinates**.

Before 2026-09-27, this field had no API; callers had to construct the entire uint64 payload and pass it as a runtime scalar. It now has a parameter; measurements are in `pypto/python/tests/st/pypto_pro/frontend/datacopy/test_quant_offset.py`.

## Dual-destination `acc_to_vec_mode` supports only trivial copies of the same type

```python
pl.move(ub, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)               # ✓ FP32 → FP32
pl.move(ub, acc, acc_to_vec_mode=..., scale=0.5)                               # ✗ scale
pl.move(ub, acc, acc_to_vec_mode=..., relu_pre_mode=pl.ReluPreMode.NormalRelu) # ✗ relu
pl.move(ub_fp16, acc_fp32, acc_to_vec_mode=...)                                # ✗ Narrowing
```

The hardware rule is simple: **dual-destination mode supports neither requant nor relu**. It has three forms; the third is easy to miss: **`FP32 → FP16` is itself a QuantMode (`F322F16`) in the ISA**, so “only changing dtype, without scale” also counts as requant. All three are now rejected at parse time:

| Combination | Error |
| --- | --- |
| dual + `scale` (scalar or Scaling Tile) | `scale cannot be combined with dual-mode acc_to_vec_mode` |
| dual + `relu_pre_mode` | `relu_pre_mode cannot be combined with dual-mode acc_to_vec_mode` |
| dual + narrowing / any dtype conversion | `dual-mode acc_to_vec_mode cannot convert ... -> ...` |
| dual + `DN` destination layout | `does not support a DN destination layout (nz2dn)` |

When several apply, they are reported in this order: fixing one reveals the next, providing step-by-step guidance. The first three are forms of the same ISA rule (`quantPre == NoQuant`); the fourth is a separate `GetDualDstCtl` static_assert.

**Only `FP32 → FP32` and `INT32 → INT32` are valid, and the destination layout must be `ND` or `NZ`**. For conversion or `DN`, use `SingleModeVec0` / `SingleModeVec1`, or transfer in the source dtype and apply `vf.astype` afterward.

**`phase` is outside this rule**: the official `move.md` says `DualModeSplitN` + `phase` **may hang** when N is not 32-aligned, meaning conditional support rather than no support. Not measured: an unbalanced UF chain can hang the AI Core and require a card reset.

Template and measurements at three layers: [`fixpipe_dual_dst`](../../examples/basic_api/fixpipe/fixpipe_dual_dst/test.py).

## `atomic=AtomicAdd` supports only six destination dtypes

```python
pl.store(out_fp16, ac, [0, 0], scale=0.5, atomic=pl.AtomicType.AtomicAdd)   # ✓
pl.store(out_uint8, ac, [0, 0], scale=0.5, atomic=pl.AtomicType.AtomicAdd)  # ✗ UINT8 does not support atomic add
```

The hardware atomic-add unit accepts only `FP16` / `FP32` / `BF16` / `INT8` / `INT16` / `INT32` (`SetAtomicAdd` in `a5/TStore.hpp:92`). **`UINT8` / `HF8` / `FP8E4M3FN` are unsupported**: they are destinations supported by on-the-fly quantization, but not by atomic add. They are now rejected at parse time.

To accumulate into these types, finish accumulation in a wider dtype and then quantize; do not rely on atomic.

## A move with an offset cannot have a `DN` destination

```python
pl.move(ub_dn, acc)            # ✓ Whole-tile transfer; TMOV supports nz2dn
pl.move(ub_dn, acc, [16, 0])   # ✗ An offset lowers to TEXTRACT, which does not accept DN
```

**Adding an offset changes validity for the same destination Tile**, because the forms lower to different ISA instructions. With an offset, the path uses `TEXTRACT` (`MakeBlockOutMoveCodegenCCE` in `backend_cce_block_out_ops.cpp`), whose Acc branch in `TEXTRACT_TILE_IMPL` requires an NZ or ND destination.

## `vf.move` and `vf.full` have opposite MergeMode defaults, and neither supports the other mode

| | Default | Unselected lanes | Other mode |
| --- | --- | --- | --- |
| `vf.move` | **MERGING** | Preserve the destination register's **original values** | `ZEROING` **unsupported** |
| `vf.full` | **ZEROING** | **Set to 0** | `MERGING` **unsupported** |

Both unsupported forms **pass parsing** and are rejected only at codegen (clear messages with source lines):

```
vf.move only supports MERGING mode on current device   # backend_cce_vf_ops.cpp:6614
vf.full only supports ZEROING mode on current device   # backend_cce_vf_ops.cpp:162
```

**Whether a partially masked `vf.move` is defined depends on whether the LHS is a new name.**

```python
dst = vf.load_align(keep, 0)       # dst holds known data
dst = vf.move(src_reg, half)       # ✓ Reuses the register → preserves keep in the second half; defined

fresh = vf.move(src_reg, half)     # ✗ New name → new destination register; second half is undefined
```

The parser declares a register only on its **first definition** (`_assignment_parser.py` Step 2: “For existing variables, reuse them directly”), so reusing an already-declared name is the intended MERGING pattern. Measurements (2026-09-27, `keep=1001..1064`, `src=1..64`, half mask): reuse gave `[1..32, 1033..1064]`, while a new name gave `[1..32, 33..64]`. The latter's unselected lanes contained **the source register's own values**, so **the output looked like a complete copy**; an assertion that “move copied the whole block” would pass for the wrong reason.

For a trivial copy, use a full mask (`MaskPattern.ALL`); to zero unselected lanes, use `vf.full`.

## Result-producing `vf.*` calls cannot be nested inside another call's arguments

```python
vf.store_align(dst, vf.full(x, preg), preg, 0)     # ✗ Rejected at parse time
dup = vf.full(x, preg)                             # ✓ Assign first
vf.store_align(dst, dup, preg, 0)
```

Error message (starting at `pypto` `f8f172bdc`):

```
vf.full produces a result and cannot be nested inside another call's arguments.
Bind it to a variable first.
hint: Use: dst = vf.full(...)  # then pass that variable
```

**Before that commit, it described a different problem**: “The statement form `vf.full(dst, ...)` is no longer supported,” even though no statement form was used. Do not let an older version's message mislead you; split it into two statements as the hint suggests.

**Not every returning `vf.*` call has this restriction.** The criterion is the `_VF_OP_DST_COUNT` table in `_call_parser.py`: `vf.create_mask` is absent, so `vf.store_align(tile, reg, vf.create_mask(), 0)` is valid.

`check.py rules` checks nested result-producing calls in recognizable SIMD VF bodies against the selected parser's table and rejection form. It preserves the parser's `bit_cast` exception and reports an advisory when that source cannot be recognized; no framework import or board compilation is needed.

## `vf.store` `count` can be omitted

```python
vf.store(dst, reg)          # ✓ One full register (256 bytes / sizeof(dtype))
vf.store(dst, reg, 64)      # ✓ Explicit element count
```

Three sources of evidence originally disagreed: the official `store.md` **prototype and parameter description** made `count` mandatory, and the Python signature had no default; yet that page's **call example omitted it**, and the framework docstring said “optional.” Measurements followed the call example. **The two incorrect sources have been fixed** (`pypto` `f8f172bdc`: the signature now uses `count=None`, and the documentation prototype and parameter row match). As with `addc`, the actual rule once appeared only in the call example.

## Alternate `vf.squeeze(STORE_REG)` and `vf.squeeze_store_unalign`, or hardware hangs

```python
vf.clear_spr()                                   # ← Required; see the next rule
areg = vf.unalign_reg_for_store()
a = vf.squeeze(src, hit_a, gather_mode=pl.SqueezeMode.STORE_REG)
vf.squeeze_store_unalign(dst, a, areg)           # ✓ Consume AR immediately
b = vf.squeeze(src, hit_b, gather_mode=pl.SqueezeMode.STORE_REG)
vf.squeeze_store_unalign(dst, b, areg)
vf.squeeze_store_unalign_post(dst, areg)         # Finalize once; not part of the alternation
```

`squeeze(STORE_REG)` writes the valid byte count into the **AR special register**, and `squeeze_store_unalign` reads AR as its advance amount. There is only one AR: **two consecutive squeezes without an intervening store overwrite the unconsumed byte count**. The official `squeeze_store_unalign.md` says this hangs hardware, requiring a card reset. **Neither parsing nor compilation blocks it** (measured; `references/api/vf.md` §1.5).

This also applies to loops: a squeeze in a `for` body that is not consumed in the same iteration is overwritten by the next iteration.

**A tool now blocks this**: `python scripts/check.py rules <path>`, using AST only and no hardware time.

## `vf.clear_spr()` must precede the first `vf.squeeze(STORE_REG)`

AR is a special register and **is not guaranteed empty on entry to `@pl.vector_function`**; it contains values left by previous code. Omitting `clear_spr()` neither raises an error nor hangs: `squeeze_store_unalign` uses a stale byte count as its advance and **silently writes to the wrong position**.

The official documentation body does not state this requirement; **`vf.clear_spr()` appears only in its call example** (`squeeze_store_unalign.md:61`), just like `addc`'s predeclaration requirement. The evidence is a hardware measurement (§12.9). This is also checked by `check.py`.

## No more than 4 `UnalignReg` objects may be live simultaneously

```python
a0 = vf.unalign_reg_for_store()
a1 = vf.unalign_reg_for_store()
a2 = vf.unalign_reg_for_store()
a3 = vf.unalign_reg_for_store()
a4 = vf.unalign_reg_for_store()        # ✗ All five remain live → bisheng crashes
vf.store_unalign(dst, reg, a0, 4, post_update=True)
...
```

**The limit applies to simultaneous liveness, not the number of declarations.** Declaring five but consuming each immediately, with no overlapping lifetimes, compiles normally. `vf.load` / `vf.store` each declare one internally but consume it immediately, so they do not count toward persistent liveness.

**Exceeding the limit produces no diagnostic about register limits**, only a compiler crash:

```
Running pass 'Greedy Register Allocator' on function '@..._vector.thread'
bisheng: error: unable to execute command: Segmentation fault
bisheng: error: clang frontend command failed due to signal
```

Check this rule first for that signature. The four measured forms are in `references/tooling/example-coverage.md` §8.6. **A tool blocks this**: `unalign-reg-budget` in `python scripts/check.py rules <path>`.

## `pl.init_output` requires a vector section in the kernel, even an empty one

```python
@pl.jit()
def k(out: pl.Tensor[[8, 16], pl.DT_FP32]):
    pl.init_output(out, offset=0, size=128, value=-1.0)     # ✗ This statement alone → compilation failure
```

Internally, it uses the V pipe to fill temporary UB, then MTE3 to transfer it to GM. Without `pl.section_vector()` in the kernel, those two ops land on a core with no V pipe. The two C++ errors **do not reveal the connection to sections**:

```
kernel.cpp:23:13: error: the ranges of 1st parameter must be [0, 0], [2, 5], [10, 10]
    get_buf(PIPE_V, 0, 0);
arch_cce_intrinsic.hpp:306:5: error: function type '...' of 'copy_ubuf_to_gm_align_v2'
    does not support the given target feature
```

Adding an empty section is sufficient:

```python
    pl.init_output(out, offset=0, size=128, value=-1.0)
    with pl.section_vector():
        pass
```

**Parsing succeeds completely**, so `parse_target_program` cannot detect this. See `examples/basic_api/datamove/init_and_insert/test.py` §3 for four measured forms.

## UB tiles beside `pl.init_output`: use `make_tile_group` or add `bar_all()`

```python
@pl.jit()
def k(x, out):
    pl.init_output(out, offset=0, size=N, value=-1.0)
    src = pl.make_tile(tt, addr=0x0)         # ✗ Received x elements instead of -1.0 in 6 of 12 runs
    with pl.section_vector():
        pl.load(src, x, [0, 0])
```

The temporary UB tile in `init_output` is **hardcoded at `0x0` and occupies `(256*32)/sizeof(T)` bytes** (8 KB for b32), and **no barrier follows its `TSTORE`**. While MTE3 is still transferring UB `0x0` to GM, a `TLOAD` (MTE2) writing the same UB can run concurrently: **a WAR race, with no error reported anywhere**.

Two fixes have different costs:

```python
    pl.init_output(out, offset=0, size=N, value=-1.0)
    pl.system.bar_all()                      # ✓ 0/12; costs a full-pipeline drain
    src = pl.make_tile(tt, addr=0x0)
```

```python
    pl.init_output(out, offset=0, size=N, value=-1.0)
    g = pl.make_tile_group(type=tt, addrs=0x0, mutex_ids=[0])   # ✓ 0/12; serializes only the same buffer id
```

Tiles declared with `make_tile_group` acquire `get_buf(PIPE_MTE2, 0, 0)`, using **the same buffer id** as `init_output`'s `get_buf(PIPE_MTE3, 0, 0)`, which serializes them. The cpp generated for bare `make_tile` contains **no `get_buf` at all**.

**`auto_mutex=True` is not a fix**: adding it to bare tiles produces byte-for-byte identical cpp and still fails 6/12 on hardware. `bar_mte3()` does not work either (an **intra-pipe** barrier; see `references/diagnostics/validation-traps.md` §8).

**Ordering is not the issue.** The docstring requires “Must be called before Tile buffer allocation,” yet following that still fails 6/12. What matters is whether addresses overlap and which declaration path the tile uses.

Gate: `init-output-needs-a-drain` in `python scripts/check.py rules`. Full measurements and framework-side locations are in `references/diagnostics/validation-traps.md` #11.

## `pl.init_output` `value` must not change value during conversion

```python
pl.init_output(int32_tensor, offset=0, size=128, value=1.5)      # ✗ Compilation failure
pl.init_output(int32_tensor, offset=0, size=128, value=3.0)      # ✓ Passes; fills 3
pl.init_output(int32_tensor, offset=0, size=128, value=5)        # ✓
```

The docstring says value “must be type-compatible with tensor's dtype”; **the actual rule is broader and more precise**: the literal in generated C++ must pass `-Werror,-Wliteral-conversion`, which reports an error only when **the value changes**.

```
implicit conversion from 'float' to '...::DType' (aka 'int') changes value from 1.5 to 1
```

**This message is unavailable from the exception**: `_compile_for_key` in `jit.py` raises an exception containing only `Failed to compile kernel '<name>'`; `_run_bisheng` in the same file logs the detailed compiler output above. To pin it in a test, capture logs with `caplog`. Matching the exception message only asserts that something failed.

## SIMT atomic operand dtype must exactly match the target

```python
tid = pl.simt.linear_thread_idx()
pl.simt.atomic_add(state[0, 0], tid + 1)                    # ✗ tid + 1 is INT64
pl.simt.atomic_add(state[0, 0], pl.simt.cast(tid + 1, pl.DT_INT32))     # ✓
pl.simt.atomic_inc(counter[0, 0], 6)                        # ✗ The literal is INT64
```

```
simt.atomic_add operand 0 dtype must match target dtype int32, but got int64
```

`simt_ops.py:578`. **Arithmetic on `tid` and Python literals are both INT64 in SIMT**, so any scalar operand not sourced from a tensor needs an explicit `pl.simt.cast`. Values read from INT32 / UINT32 tensors need no further cast.

The same applies to `atomic_inc` / `atomic_dec` `limit`: it is operand 0, and the target is usually a UINT32 counter, so read `limit` from a UINT32 tensor or cast it.

Remember the symptom: **the error points to the first atomic call**, while the actual cause is `+ 1`. All 10 criteria failed together when writing `examples/basic_api/simt/simt_atomic`, each for this reason.

## `pl.simt.syncthreads()` cannot appear in runtime control flow (**no check blocks it; the symptom is a hang**)

```python
if tid > 0:
    pl.simt.syncthreads()          # ✗ Some threads skip the barrier → hang
```

The docstring explicitly prohibits this (runtime `if` / `for` / `while`, and use as an expression), but **neither parsing nor compilation blocks it** (measured 2026-09-28). **Fences are not subject to this restriction**: `threadfence_block()` / `threadfence()` are valid in runtime control flow. Do not conflate the two; banning both would reject correct kernels.

Gate: `syncthreads-must-be-uniform` in `python scripts/check.py rules` (scope limits in [`tooling/checks.md`](../tooling/checks.md)). **Neither framework layer blocks it, so this gate is the only one**.

## `pl.const` accepts only **literal tokens**; constant names are rejected

```python
CLAMP = -0.5
pl.const(CLAMP, pl.DT_FP32)     # InvalidType: must be a numeric literal
pl.const(-0.5, pl.DT_FP32)      # Accepted
```

`scalar_ops.py:214` checks the AST node (`isinstance(value_node, ast.Constant)`), not the value. **Constants such as clamp bounds therefore cannot share one name between the kernel and reference**: the two numbers in `support/scalar_const` are written twice, with the reason documented at both locations.

For runtime scalars, use `pl.astype`. The boundary is **one-way**: `astype` also accepts literals (its docstring says only “runtime scalar expression,” but literals were measured to work); the reverse does not hold.

## `TileType` cannot be a closure variable

```python
TT = pl.TileType(shape=[16, 64], dtype=FP16, target_memory=pl.MemorySpace.Vec)

@pl.jit()
def kernel(...):
    g = pl.make_tile_group(type=TT, ...)     # InvalidType: Unsupported closure variable type
```

`_expr_evaluator.py:266`. **Construct TileType inside the kernel body**. Calling a helper from the body is valid (`support/dtype_carriers` does so). The distinction is whether TileType is constructed while parsing the kernel body or already exists before parsing starts.

## Cross-core buffer `target_memory` must be the literal `pl.MemorySpace.X`

```python
MAT = pl.MemorySpace.Mat
sh = pl.make_tile_group(type=pl.TileType(..., target_memory=MAT), fwd_ids=[...], bwd_ids=[...])
# InvalidVal: pipeline: cross-core buffer 'sh' has no resolvable memory space.
#             ... must set target_memory to a literal pl.MemorySpace.<X>.
```

`_cross_core_scanner.py:775`. **Aliases are a natural pattern** (every example in this repository has such aliases at the top), yet they fail specifically in this cross-core buffer position.

This is the same kind of check as `pl.const` above: it reads AST rather than evaluating values. **This repository has encountered it three times**. The third was a constant name bypassing the `pl.gemm_mx` `k % 64` guard: that caused a false negative, whereas these two caused false positives.

## A `Bias` tile must have exactly one row

`_validate_tile_type_params` in `block_ops.py` enforces the hardware requirement `Rows==1`; declare shape `[1, N]`. The historical diagnostic labeled `Bias` as “Bias (L0B),” **but L0B is `Right`**: that old space label was incorrect.

## Layout validity is checked at three layers, in this order

In `block_ops.py`: operator operands → tile declarations → data paths (three tables including `_A5_LOAD_COMBOS`). **The first two layers say exactly what to write; the third says only `unsupported layout/dtype combination`**, requiring you to inspect the source tables.

See [api/support.md](support.md) §12 for the complete three-layer table, diagnostic quality at each layer, and declarable layouts for all eight spaces. The most common rule: **FP16 from an `ND` GM tensor can land only in `NZ` in L1**; off-chip to L0 always takes two hops.

## `set_saturation_flag` CTRL bits differ from flag values

The `FLOAT` / `FLOAT8` / `CAST` modes have **inverted polarity**: `enable=True` writes **0** to the bit. Only `INT` has positive polarity. Thus, after `set_saturation_flag(mode=CAST, enable=True)`, `get_ctrl_spr(59, 59)` reads 0 and appears ineffective. `get_saturation_flag` is symmetric and works correctly when used by itself.

**CTRL[60] chooses which setting is obeyed**: its entry value is 1 (global mode), so **`vf.astype(saturate=...)` has no effect by default**. First call `set_ctrl_spr(60, 60, 0)` to make it effective.

## `reset_ctrl_spr` restores hardware defaults, not the kernel entry value

These differ for bit 48 (entry `0x1001000000000008`, documented default `0x1000000000000008`), so the “restore original state” pattern `set → work → reset` changes bit 48 from 1 to 0. To truly restore it, first save it with `get_ctrl_spr` and write it back. Details: `support/spr_ctrl` §3.

## Avoid adjacent multi-worker GM scalar writes with `pl.setval`

The measured loss concerns the GM scalar-write path, not every GM output: eight blocks writing 16-byte segments through `pl.setval` retained only 2–3 segments, while 64-byte spacing preserved all segments in the recorded probes. `pl.system.dcci` did not fix the loss; its hardware mechanism remains unresolved.

Preserve the task's dense output layout. Prefer collecting results in UB and writing the valid region with `pl.store`; the [scalar/DMA comparison](support.md#7-small-gm-scalar-writes-from-multiple-blocks) documents current evidence, conditions and toolchain identity. Treat 64-byte spacing as a diagnostic control for that scalar path, not a requirement to pad the public output. The [small-result pattern](../patterns/vf-small-results.md) gives the aligned UB declaration and writeback form.

## `pl.minimum` / `pl.maximum` `dim=` is the **retained** axis

`dim=0` → `out[i, 0] = reduce_j(...)`, output `[M, 1]` (numpy `axis=1`); `dim=1` → `out[0, j] = reduce_i(...)`, output `[1, K]` (numpy `axis=0`). **This is the opposite of numpy / torch `axis=`.**

The third parameter has different roles in the two forms: for elementwise operations, it is the right operand (Tile or Scalar); with `dim=`, it is workspace (Tile only, same shape as src).

## `.next()` inside `@pl.pipeline.stage` accepts only simple assignment

```
NotSupported: pipeline: cross-core buffer group 'a_l1' slot accessor `.next()`
must be assigned to a simple variable (`slot = a_l1.next()`);
inline/chained/tuple-unpack forms are not supported.
```

```python
cur_a, cur_b = a_l1.next(), b_l1.next()   # Rejected inside a stage function body
cur_a = a_l1.next()                        # Accepted: one per statement
cur_b = b_l1.next()
```

The cross-core scanner validates this (`_cross_core_scanner.py::_validate_slot_accessors`, rule L10), **only inside `@pl.pipeline.stage` function bodies**. The same tuple unpacking works in an ordinary `@pl.jit` body (`examples/matmul/fundamental_usage/bias_fusion` uses it). Valid stage-function patterns are therefore a subset of kernel-body patterns; check this rule before moving a form from an ordinary example into a stage. Hardware measurements on 2026-09-28, build corresponding to pypto `8faa85389`.

## Layer B slot-selection diagnostics (F00007 / F00005)

- **F00007 outside the pipeline loop:** `slot 'qn' is taken from tile group ... outside the pipeline loop, but ... mutex_ids could not be resolved statically` concerns an outer slot consumed by a stage whose slot count the transform cannot resolve. It is not a blanket ban on named outer slots. Follow [stage slot selection](crosscore-sync.md#keep-slot-selection-visible-to-the-stage-scanner) for the supported form and version scope.
- **F00005 inside a stage:** a traced accessor such as `_vf_fn(st_a[tick % ST_SLOTS], ...)` needs a separate simple binding before the call. The same [scanner contract](crosscore-sync.md#keep-slot-selection-visible-to-the-stage-scanner) covers `.next()`, `.current()`, `.previous()` and indexed slots; ordinary kernel-body forms are not interchangeable with stage-body forms.

---

## Pipeline topology errors after adding output panels

The diagnostics `stage ... is called more than once`, `cross-core buffer ... is used by 3 stages`, and `with ... stage(s) in between` concern the synchronized graph, not tile capacity. If they appear after splitting one product's output into local panels, first check whether that panel loop belongs inside one consumer stage. Use [compute tiling versus synchronization stages](crosscore-sync.md#compute-tiling-is-not-a-new-synchronization-stage); renaming stage functions or duplicating a shared buffer alone does not establish a valid graph.

A `stage/buffer dependency cycle` introduced by a mask or postprocess on a received buffer has a different first check: inspect new writer pipes and use [read-only shared inputs with register postprocessing](crosscore-sync.md#postprocess-a-shared-input-without-changing-its-writer). Changing preload does not change the read/write topology.

## ST cases require a `soc` marker

Set `pytestmark = pytest.mark.soc("950")` at module level. Without it, the root `conftest.py` treats cases as 910 and **silently drops them** on A5, counting them as neither skipped nor deselected; details in ../diagnostics/validation-traps.md §2.

## Do not use `match=` to assert error messages **with source spans**

Use `excinfo.value.message`. `str()` embeds the test's own source line, and the `match=` regex can match itself; details in ../diagnostics/validation-traps.md §1.

**This rule is conditional; measurements on 2026-09-26 narrowed its scope.** It occurs only when the exception **includes a source span** (`str(exc)` contains a `--> file:line:col` line pointing to a line of the test file's `@pl.jit` kernel) and the ±2-line excerpt window includes the `with pytest.raises(...)` line.

For exceptions without spans (`str(exc)` is one line with no `-->`), `match=` is safe. To distinguish them, deliberately use an impossible pattern and check whether it still passes:

```python
# str() is one line for this exception type; match= works (a wrong pattern correctly fails)
with pytest.raises(InvalidVal, match="did not pass MatmulKey.is_valid"):
    TilingKeySchema(MatmulKey).validate_concrete({"Dtype": 1, "TileBlock": 64})
```

Validation errors in `tilingkey.py` / the schema layer belong to this category; [`tilingkey_specialize`](../../examples/matmul/tilingkey_specialize/test.py) uses `match=` directly.

## An exception's **base class**, not its name, defines the `except` contract

Every error class in `_errors.py` is created by `_make(name, code, base, doc)`. The source says: “The table below is both the class list and the code -> builtin-base mapping.” Whether `except` / `pytest.raises` catches it depends on that table's third column:

| Built-in base class | Classes |
| --- | --- |
| `ValueError` | `InvalidArgument`, `InvalidOperation`, `InvalidShape`, `InvalidVal`, `OutOfRange`, `InvalidFormat`, `InvalidTile`, `CommonExternal` |
| `RuntimeError` | `RuntimeFailure`, `CommonInner`, `PassInner`, `CodegenInner` |
| `NotImplementedError` | `NotSupported`, `DynamicShapeUnsupported` |
| `TypeError` | `InvalidType` |
| `KeyError` | `KeyNotFound` |
| `NameError` | `NameNotFound` |
| `OSError` | `BadFd` |

**Three counterintuitive points affect different tests**:

- **`except RuntimeError` catches `NotSupported`**, because `NotImplementedError` is itself a subclass of `RuntimeError`. A test for parse-time rejection that uses `except RuntimeError` as a subprocess fallback will report the parse rejection as a runtime exception, so **it will not fail when the rejection layer changes** (an actual incident; see [verification/evidence.md](../verification/evidence.md) §14).
- **`except ValueError` does not catch `InvalidType` / `KeyNotFound` / `BadFd`.** Most argument errors are `ValueError`, making it tempting to catch them all with it; these three are outside that family.
- **`OutOfRange` is a `ValueError`, not an `IndexError`.** A source comment explains the choice: it reports values outside an API's accepted range (dtype range, thread count, mutex id), not sequence indices.

Every class also inherits from `PyptoProError` → `PyptoError` → `Exception`, so **`except PyptoError` catches them all**. Use it, rather than a built-in class, for tests that accept any framework rejection.

Choose the criterion by **using the class belonging to the layer being verified**, not a shared ancestor or built-in class: `pytest.raises(NotSupported)` fails if the rejection layer changes; `pytest.raises(RuntimeError)` does not.

> Source: `_make` and the following table in `pypto_pro/_errors.py` (**derived from source inspection; `pypto_pro` was not installed locally, so no probe was run**). Python inheritance itself was measured: `issubclass(NotImplementedError,
> RuntimeError)` is `True`; `issubclass(KeyError, ValueError)` and `issubclass(OSError,
> RuntimeError)` are `False`.

## Kernels created with `exec` cannot be parsed: the parser needs real source

`@pl.jit` uses `inspect` to retrieve source and then parse its AST. **A function object is insufficient; source text is required**. A kernel created by `exec` from a string (generated tests or parameterized kernel creation) therefore lacks retrievable source.

`_get_source_info` in `_call_parser.py` has four fallback steps:

1. `inspect.getsourcelines()`: the normal case.
2. `linecache` fallback: **IPython / Jupyter work**, because they cache cell source.
3. Find `-c` in `sys.orig_argv`: **`python -c '...'` works**.
4. Only if all fail, raise `Cannot retrieve source code for function '<name>'`, with the hint “Define the callable in a Python source file so its AST can be inspected.”

Thus, **absence from a file does not necessarily mean failure**: success depends on whether something has put the source into `linecache`. Bare `exec(code_str)` does not, so it reaches step 4. Supply that missing source yourself to work around it: write and import a temporary `.py` file, or populate `linecache.cache` manually.

Step 4 raises `BadFd`, which the table above identifies as **`OSError`**; neither `pytest.raises(RuntimeError)` nor `pytest.raises(ValueError)` catches it.

> Source: `_get_source_info` and its three `_source_from_*` helpers in `pypto_pro/language/parser/_call_parser.py` (**derived from source inspection, not measured**). There is also an **unverified discrepancy**: `extract_func_source_info` in `runtime/kernel.py` calls `inspect.getsourcelines(f)` directly, without the fallback chain above, and should fail with Python's native `OSError: could not get source code`. Which entry point runs first was not verified locally; do not use this as a test criterion.

## `pl.insert` into a ZN-declared destination: use NZ source coordinates for offsets (wrong coordinates raise no error)

When inserting NZ source blocks into a ZN-declared Mat slot (the usual transpose-view return path), offsets use the **source's NZ coordinate system**. Using destination ZN coordinates **compiles but silently computes wrong values on hardware**. The mechanism, correct/incorrect comparison, and evidence limits belong to [api/datamove.md](datamove.md) §16; example: [flash_attention](../../examples/algorithms/flash_attention/). Include a wrong-coordinate variant as a negative control for this return path.

For a rectangular destination, `insert: offset[0] (64) exceeds source shape[0] (32)` can instead be a front-end bounds rejection of a split write. See the [rectangular NZ-to-ZN handover](datamove.md#rectangular-nz-to-zn-handover) for the checked axes and a working full-K write; this diagnostic does not impose a square GEMM operand.

## Insufficient stage-buffer slots: errors and warning coverage

`pipeline: cross-core buffer ... has ... tile(s) ... so ... tiles are live at once` is an insufficient-depth rejection. Same-core handovers have version-scoped `SlotShortage` and `SlotCheckIncomplete` warnings; absence of a warning does not prove safe reuse. The [slot-lifetime contract](crosscore-sync.md#slot-counts-identity-and-check-coverage) owns the required depth, selector identity, warning coverage and serial/shipping comparison. Include the required depth in the full [capacity plan](memory-limits.md); a capacity failure describes the current allocation, not a universal limit for an operator or head dimension.

## Tile capacity failures after increasing slot depth

`Tile addr ... + size ... exceeds Left/Right/Acc capacity ...` identifies the generated allocation that exceeded a space's bound. Count every slot, alignment and physical padding, using the accumulator's actual dtype. L1 depth and L0 depth are independent choices. The [capacity owner](memory-limits.md) describes the installed checks; one rejected layout does not establish the widest feasible GEMM shape.

## Tile slices require a UB ND/DN view

In a JIT body, a Tile slice such as `tile[r0:r1, c0:c1]` requires Vec memory and ND/DN layout; being in UB alone is insufficient. The parser rejects other memory spaces or fractal layouts with `InvalidFormat: Tile slice is only supported on UB (Vec) tiles with ND/DN layout`. In a VF body, Tile slices are unsupported even for UB ND/DN tiles; pass the tile and coordinates to `vf.load_align` / `vf.store_align` instead. These checks are in `_expression_parser.py` (`pypto/python/pypto_pro/language/parser/_expression_parser.py`).

For a partial fractal-tile transfer, choose a supported `pl.load`, `pl.move` or `pl.insert` operation with explicit offsets and valid extents, following the [data-movement contracts](datamove.md). A reinterpretation changes the view, not the bytes: an ND staging buffer and an NZ destination need a supported layout conversion, not just a new layout label. Whole-tile transfers still obey the load format combinations (`pypto/docs/zh/api/pro_api/SIMD-API/memory_data_movement/load.md`).

## `NameNotFound` inside a pipeline stage

Check where the referenced name is bound and whether that binding is available in the transformed stage. Pass kernel-local Tile groups, Tensors and views explicitly as stage parameters. Explicit tile declarations live in the kernel scope; stage argument binding still determines their availability after transformation. See [allocation scope](memory-allocation.md#scope-and-responsibilities).

For delayed stages, passing an argument alone does not establish its lifetime: non-scalar arguments must remain invariant across the delayed calls, and selected slots need the [pipeline capture and slot checks](crosscore-sync.md). A failure involving a loop-local view is distinct from an allocation-scope error.

## Unhashable dtype in a Python cache key

At behavior version `fd1badd7c`, using `pl.DT_FP16` as an `lru_cache` argument raises `TypeError: unhashable type: 'pypto.pypto_impl.ir.DataType'` before kernel parsing. For an existing cached Python helper, use a validated dtype name as the key and resolve the DataType inside the helper. Do not generalize this to all framework objects: `pl.MemorySpace.Vec` and `pl.NZ` are hashable on the same installation. Check `hash(argument)` when diagnosing a different type or version.
