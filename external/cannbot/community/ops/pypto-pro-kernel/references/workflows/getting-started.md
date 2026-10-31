# PyPTO Pro primer — the resident basics

Before the first kernel-authoring task in a session, read this once unless its basics are already in context. Debugging or optimizing an existing kernel needs only the sections resolving a current uncertainty, as specified by [stage selection](../../SKILL.md#choose-the-current-stage). This is a **cached digest, not an owner**: every section names its owner, and on any conflict the owner wins. The common digest was verified against pypto `b13f8cfaf` on 2026-09-28; individual sections identify later evidence separately. Recheck a detail with its owner when the target toolchain differs. Keep this foundational cache within 210 lines while retaining the launch, type, memory and synchronization distinctions needed for a first kernel.

## Execution model

Owner: `../pypto/docs/zh/guide/programming_guide/pro/development/kernel_function.md`; runnable: [examples/basic_api/support/entry_and_loop](../../examples/basic_api/support/entry_and_loop/).

- A kernel is a plain Python function under `@pl.jit(...)`, with `import pypto_pro.language as pl`. Tensors appear as typed parameters.
- The body is sectioned: `with pl.section_vector():` (AIV) / `with pl.section_cube():` (AIC). A kernel with both is a mixed kernel.
- Loops inside the body use `pl.range(...)` — the builtin `range` is rejected at parse.
- Key `@pl.jit` kwargs: `auto_mutex` (default True), `pipeline=`, `tiling_key=`, `name=`, `sanitizer=`. Launch is eager `kernel(x, out)`; a bracketed grid form `kernel[grid, cores]` called like a normal function exists (owner: `runtime/jit.py`).

## Type surface

Owner: `../pypto/python/pypto_pro/language/typing/` and `_api.py`.

- Signature params: `x: pl.Tensor[[64, 200], pl.DT_FP32]`; scalar parameters spell the dtype as `scale: pl.DT_FP32` or `count: pl.DT_INT64`, not a bare `FP32` alias or `pl.Scalar` ([annotation contract](../api/constraints.md#annotate-scalar-parameters-directly-with-a-dtype)). Dtype aliases inside `Tensor`/`Ptr` are a different case. Pointer parameters use `pl.Ptr[dtype]`; `Input`/`Output` provide direction marks. Dims are literal ints or `pl.DYNAMIC` (runtime); compile-time variants go through `tiling_key`/`datatype` specialization.
- `pl.TileType(shape=[r, c], ...)` describes a **physical 2-D buffer**, not the logical GM shape. For the ordinary FP32/INT32 row-major Vec tiles used here, physical columns must be a multiple of 8 (32 bytes). Carry the true dimensions in `valid_shape`; it cannot repair an invalid physical declaration. Other layouts/dtypes follow their own TileType alignment contract (`pypto/docs/zh/api/pro_api/SIMD-API/basic_data_structures/TileType.md`).
- Construct `TileType` during parsing, inside the kernel body or a helper invoked there; do not capture a preconstructed `TileType` in a Python factory closure. Capture shape/count values instead ([rejected form and replacement](../api/constraints.md#tiletype-cannot-be-a-closure-variable)).
- [Host dtype carriers](../../examples/basic_api/support/dtype_carriers/): the host side does not check dtypes for you.

## Common language — hardware term ↔ DSL spelling

Task descriptions and Ascend docs speak hardware; the DSL spells it differently. This table is the translation — mechanisms stay with their owners.

| You hear | You write |
|---|---|
| L1 | `MemorySpace.Mat` |
| L0A / L0B | `MemorySpace.Left` / `Right` (65536 B each — plan capacity first) |
| L0C (accumulator) | `MemorySpace.Acc` |
| UB | `MemorySpace.Vec` |
| BT / bias table | `MemorySpace.Bias` — FP32 tile, fed GM→L1(Mat)→Bias by `pl.move` |
| MX scale plane | `MemorySpace.ScaleLeft` / `ScaleRight` (addr = data addr >> 4) |
| AIC / AIV | `with pl.section_cube():` / `with pl.section_vector():` |
| the two AIVs of a **mixed cube+vector** block | `pl.get_subblock_idx()` (0/1) splits the rows; this is not the pure Vector launch mapping |
| pure Vector row ownership | `kernel[None, W]` has W vector workers. Use `pl.range(pl.get_block_idx(), rows, pl.get_block_num())` for one row per iteration; `get_subblock_num()==1` here. Do not multiply the stride by two just because the hardware has two AIVs ([current probe](../api/support.md#5-get_subblock_num-depends-on-whether-the-kernel-has-a-cube-section)) |
| multi-core split (mix kernel) | core id = `pl.get_block_idx() // pl.get_subblock_num()` — on the vector side `get_block_idx()` alone is the flattened AIV-thread number, not the core. Launch is the OTHER semantics: `kernel[None, X]` takes the number of **execution groups** (1 cube + 2 vec each) = physical cores, never the thread count — passing 2× launches ghost groups that write GM out of bounds (../api/support.md §15) |
| double buffer / ping-pong | tile group with `depth=2` (two slots + ids); `.next()` rotates |
| fixpipe (L0C drain path) | `pl.move(dst, acc, acc_to_vec_mode=/scale=/relu_pre_mode=)` — Acc→Vec/Mat on FIX; Dual modes split to both AIVs but carry same-dtype copies only |
| VF result → cube operand (v→c handback) | taught recipe: per 128-col group one `[rows+1, 128]` NZ tile, cursor-vsstb `vf.store_align(..., DATA_BLOCK_COPY, block_stride=rows+1, repeat_stride=1, post_update=True)` with NO explicit offset, then `set_validshape` + `pl.insert` — ../api/constraints.md "Feeding VF results to Cube", worked example examples/algorithms/mm_softmax_mm |
| unitFlag | `phase=` (`AccPhase`/`STPhase`) — perf mechanism, not accumulation control |
| cross-core event | C layer: `pl.system.set/wait_cross_core(pipe=, event_id=)` |
| software pipeline | B layer: `pl.jit(pipeline=…)` + `@pl.pipeline.stage` |
| tail block | `m=/n=/k=` extents on `pl.gemm` (narrowed tiles need `compact=1`), or `set_validshape`/`fillpad`; the full multi-core non-divisible recipe (runtime row count through load→VF→insert→gemm→store, plus the 3-layer out-of-bounds evidence) — ../api/constraints.md "Multicore execution with non-divisible M", worked example examples/algorithms/rmsnorm_gemm_tail |
| GM workspace | For a one-shot C-layer Cube→Vector relay followed by VF computation, use separate scratch: reusing the output failed on the recorded toolchain. This does not prohibit reuse across separate ordered launches; see [output reuse boundaries](../api/crosscore-sync.md#output-reuse-across-handover-boundaries) |
| KV cache layout BNSD / BSND | `order=` on `pl.load`/`pl.store` names the tensor axes that become the tile's (rows, cols): an `[Sblk, D]` KV tile is `order=[2, 3]` from BNSD `[B,H,S,D]`, `order=[1, 3]` from BSND `[B,S,H,D]` (worked: decode_attention_gqa / flash_attention_mha) |
| NZ / ND / ZN / DN | `layout=` on `pl.TileType` / `pl.reinterpret` |

## Memory spaces and addressing

Owner: [api/memory-allocation.md](../api/memory-allocation.md), [api/memory-limits.md](../api/memory-limits.md).

- On-chip addresses are byte offsets you assign (`addrs=0x0000`); plan every slot and alias explicitly. Use `mutex_ids="auto", depth=N` to plan mutex IDs.
- Capacity and alignment are checked by the framework from your tile declarations — plan against the owner's rules, do not hardcode limits in your head.

## Tile lifecycle — and the trap that matters most

Owner: [diagnostics/validation-traps.md](../diagnostics/validation-traps.md) §7; handle methods: `_api.py` `make_tile_group` docstring.

- Under `auto_mutex`, tiles come from `pl.make_tile_group(type=..., addrs=[...], mutex_ids="auto", depth=N)`. The handle supports `g.next()` (rotate slots), `g.current()` (peek), `g[i]` (subscript).
- A bare `pl.make_tile` under `auto_mutex=True` gets **zero** synchronization, silently. `check.py` gates this (`auto-mutex-needs-tile-group`).
- Multiple slots + distinct `mutex_ids` is the double-buffer idiom; one slot serializes.

## Data movement

Owner: [api/datamove.md](../api/datamove.md); runnable: [examples/basic_api/datamove/](../../examples/basic_api/datamove/).

- Two unit systems: `pl.load`/`pl.store` take absolute **element** offsets; `pl.load_tile`/`pl.store_tile` take **tile-block** indices. Don't mix them up.
- A tile narrower than the tensor, loaded at a nonzero offset, is addressed against the tensor's own stride (precedent: matmul STs tiling over K).
- Tails: an honestly-sized smaller tile, or declared-shape + `set_validshape`/`fillpad` ([dma_pad](../../examples/basic_api/datamove/dma_pad/) shows the second).
- `pl.move` transfers on-chip. `pl.init_output` has a fixed UB footprint that can clobber your tiles (traps §11) — check before using it.

## Compute forms

Vector computation uses `@pl.vector_function` and the current `Vf` surface. Tile Vector computation is prohibited except for the three [sorting APIs](../../SKILL.md#constraints-and-workspace-policy). A method name from `_api.py` is not evidence for a same-named VF operation: query `_vf_api.py` with `check.py api vf`, including the return contract. For example, the current VF surface has `sqrt` and `div`; do not invent `vf.rsqrt` from `pl.rsqrt`. Reductions return registers, not a Python value/index pair; [reduction lanes and broadcasting](../api/vf.md#21-two-facts-about-reduce_) describe their consumers. Named float constants may be scalar operands; only `pl.const()` itself requires a literal.

Cube compute starts at the [short Cube route and sugar map](../patterns/gemm-desugar.md#the-mapping): select one runnable sugar entry and write the axis plan before searching more examples. Reuse a shared L0A/L0B base under synchronization rather than allocating one slice per lane. Vector compute uses the VF anchor below. Precompute scalar reciprocals such as `1.0 / width` outside the VF body and pass the coefficient in; scalar `/` is rejected there even for constants, while integer `//`/`%` and register `vf.div` are distinct supported forms ([scope and replacement](../api/constraints.md#move-vf-scalar-division-to-the-outer-scope)).

## API surface at a glance

Choose the namespace by responsibility, then query exact signatures with `python scripts/check.py api [pl|vf|pl.simt]` (from the skill root (`pypto-pro-kernel/`)). This routing table lists allowed authoring forms, not every framework capability. The parser dispatch table can be wider than declaration stubs (absence note below).

| Family | Names |
|---|---|
| Data movement / padding (`pl.*`) | `load load_tile store store_tile move insert init_output ssbuf_load ssbuf_store fillpad` |
| Vector arithmetic / conversion (`vf.*`) | Query the VF surface: e.g. `add mul muls sqrt div astype`; never infer names from tile APIs |
| Vector reduction / broadcast (`vf.*`) | `reduce_sum reduce_max reduce_min full`; read the register-result contract |
| Indexed or rearranged Vector data (`vf.*`) | `gather scatter interleave de_interleave`; SIMT is another route for thread-style indexing |
| Cube (`pl.*`) | `gemm gemm_mx` first; manual matrix operations only under the closed GEMM exception list |
| Sorting exception (`pl.*`) | `sort32 mrgsort mrgsort2` |
| Element / shape | `getval setval set_validshape reinterpret` |
| Mask state (`pl.*`) | `set_mask_count set_mask_norm set_vec_mask reset_mask`; these do not authorize tile computation |
| Debug probes | `pto_assert printf dump_data trap` (reach limits: support.md §13) |
| Scalar / const | `min max const astype range` (`pl.const` is literal-only) |
| Scope / alloc | `section_vector section_cube make_tile make_tile_group` |
| Block ids | `get_block_idx get_subblock_idx get_block_num get_subblock_num` |
| VF masks | `MaskPattern` is LOW-ANCHORED only (`VL1..VL128`/`H`/`Q`/`M3`/`M4`) — a mask picks a valid PREFIX; scattering lanes belongs to the store's stride/dist, never to window masks (../api/vf.md §2.11) |
| SPR / flags | `get_spr set_saturation_flag get_saturation_flag set_ctrl_spr get_ctrl_spr reset_ctrl_spr` |
| Cross-core | `bind_crosscore_sync` plus `pl.system.*` sync ops (../api/crosscore-sync.md) |

VF and SIMT have distinct surfaces. `check.py rules` verifies resolved VF names against selected Python/C++ source; missing matching source is an explicit incomplete check. Read the API owner for argument and dtype/layout support.

## Synchronization in one breath

Owner: [api/crosscore-sync.md](../api/crosscore-sync.md) (decision tree).

- Layer A `auto_mutex`: intra-core, derived from tile groups. Layer B `pipeline=` + `@pl.pipeline.stage`: cross-core, auto-inserted, verified by "results must be identical at every preload". Layer C manual `set/wait_cross_core`: zero framework checks.
- Per-pipe barriers such as `bar_mte2` drain only that pipe; `bar_all` drains all pipes on the executing core. Cross-core handover still needs the owner's synchronization protocol ([barrier scope](../api/datamove.md#8-why-bar_-is-excluded-it-is-an-intra-pipe-barrier-not-a-cross-pipe-barrier)).

## VF is the shipping form for vector compute

Owners: [api/vf.md](../api/vf.md), [api/simt.md](../api/simt.md); policy: SKILL.md.

- Vector-side computation belongs inside `@pl.vector_function`; a `pl.section_vector()` body orchestrates tile allocation, transfers and calls to those VF procedures. The [Vector policy](../../SKILL.md#constraints-and-workspace-policy) applies equally to examples.
- `pl.simt.*` = thread-style kernels with their own two-bracket launch.
- Both have example groups under `examples/basic_api/{vf,simt}/`.

Before writing a VF expression, keep three distinctions explicit: result-producing VF instructions need their own assignment before another call consumes them; `vf.select` takes two register sources, so broadcast a scalar with `vf.full` first; and `b = a` aliases a register rather than copying its contents. Give independently updated states fresh destinations with `vf.full` or full-mask `vf.move`. See the [nested-call boundary](../api/constraints.md#result-producing-vf-calls-cannot-be-nested-inside-another-calls-arguments), [selection contract](../api/vf.md#25-combining-masks-with-existing-comparisons-and-selection), and [register identity](../api/vf.md#vf-register-aliasing-and-copying); mask constructors and other return-value APIs are not all subject to the instruction-assignment rule.

A SIMD VF helper is a procedure: keep register values inside the VF and write results through tile parameters. It cannot use `return`, including a bare return, or export a register for JIT code to accumulate; [VF scope and results](../api/constraints.md#simd-vf-functions-communicate-through-tile-parameters) owns this boundary. Several dependent reductions can remain in one VF call, or communicate through explicit UB state between VF calls in one native kernel. A VF helper boundary does not imply another native kernel or GM scratch.

Keep runtime scalar names separate from VF destination names: use `indices = vf.full(index, mask, dtype=UINT32)`. Self-rebinding a scalar or scalar parameter can generate an uninitialized register self-broadcast even with an explicit dtype; see [binding identity and the diagnostic](../api/vf.md#vf-register-aliasing-and-copying).

## Anchor kernel (VF, masked tails, runtime worker count)

Use this structure directly instead of re-reading examples for imports, closures, tile lifecycle, tails or row ownership. This **pointwise** anchor keeps dense GM tensors, pads the physical UB width, masks each VF block and uses the runtime worker count. Reverified 2026-10-06 on behavior implementation `872df763f`: 45 combinations of row count, aligned/tail width and worker count, exact FP32 results. Other sections retain their own historical digest boundaries.

```python
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
FP32 = pl.DT_FP32

@pl.vector_function
def _add_one(tile, cols):
    for col in pl.range(0, cols, 64):
        active = pl.min(64, cols - col)
        mask = vf.update_mask(active, dtype=FP32)
        value = vf.load_align(tile, [0, col])
        result = vf.adds(value, 1.0, mask)
        vf.store_align(tile, result, mask, [0, col])

def make_add_one(cols):
    padded_cols = ((cols + 63) // 64) * 64
    @pl.jit(auto_mutex=True)
    def kernel(x: pl.Tensor[[pl.DYNAMIC, cols], FP32],
               out: pl.Tensor[[pl.DYNAMIC, cols], FP32]):
        with pl.section_vector():
            group = pl.make_tile_group(
                type=pl.TileType(shape=[1, padded_cols], dtype=FP32,
                                 target_memory=pl.MemorySpace.Vec),
                addrs=[0], mutex_ids=[0])
            for row in pl.range(pl.get_block_idx(), x.shape[0], pl.get_block_num()):
                tile = group.next()
                pl.set_validshape(tile, [1, cols])
                pl.load(tile, x, [row, 0])
                _add_one(tile, cols)
                pl.store(out, tile, [row, 0])
    return kernel
```

Standalone host test: `import torch_npu` before creating any `npu` device; create `k = make_add_one(cols)` and call `k[None, workers](x_npu, out_npu)`, then `torch.npu.synchronize()` before comparison. Start with one small dense case. A memory-gather/scatter mask belongs on that memory operation too; masking only the final store does not prevent earlier invalid indexed accesses.

This factory is a teaching form. For a final operator package, follow [factory-to-entry migration](delivery.md#adapt-a-teaching-factory): separate runtime extents from physical capacities and reuse the linked module-level JIT example.

## Verify and escalate

- After every edit: `python scripts/check.py rules <path>` from the skill root (`pypto-pro-kernel/`) (static AST gate; exit 0 ≠ numerically correct).
- Signature questions end at two owners: `_api.py` stubs and the per-API pages under `../pypto/docs/zh/api/pro_api/` (SIMD-API / SIMT-API / Utils-API). Absence from both is **not** absence from the DSL — the parser's dispatch table is wider than the stubs (case: `pl.muls` was ST-proven yet undeclared until pypto `b13f8cfaf`). A verified example or ST using a name outranks a missing stub; note the gap and move on.
- Board runs: [verification/board-runs.md](../verification/board-runs.md). Failures: [workflows/debugging.md](debugging.md) triage. Tuning: [workflows/optimization.md](optimization.md).
