# Pattern: gemm sugar ↔ manual chain, the two-way map

Distilled from the lowering source (`pypto/python/pypto_pro/language/parser/_gemm_lowering.py`, pypto `b13f8cfaf`) and board-verified examples. Read this page when a search lands you on a hand-written `move`+`matmul` example, before deciding the sugar "cannot do" something: common chains map to the parameters below; retained L0 operands have a scoped exception. It is also the debug map when a `pl.gemm` call fails — the error points at what the sugar expanded to. Budget: ≤ 120 lines.

## Applies when

Any cube-side matmul. Workspace policy (SKILL.md): `pl.gemm` / `pl.gemm_mx` first; a hand-written chain needs a stated reason — the closed list below, nothing looser.

## The mapping

Use this short route for unresolved Cube mechanics after [selecting the nearest complete computation](../workflows/authoring.md#selecting-an-example):

1. Write a three-column axis plan: GM storage axes → L1 tile axes → mathematical M/K/N. This settles batch addressing and which boundary performs any transpose. Ordinary GM offsets cover every tensor axis; the tile stays 2-D ([axis mapping](../api/datamove.md#higher-rank-gm-to-a-two-dimensional-tile-map-axes-first)).
2. Before adding an epilogue stage, inspect fusion at the producer and drain. For ordinary GEMM, check `bias=` and its initialization/broadcast contract in [api/gemm.md](../api/gemm.md). For Acc transfers, check options such as `relu_pre_mode` and conversion/quantization in [api/fixpipe.md](../api/fixpipe.md), including its destination/dtype/dual-mode combination matrix. API names alone do not establish a legal combination. Match the requested axes, precision and rounding before choosing a fused path; fixed-function transfer features do not require a separate VF stage merely because the workspace uses VF for vector computation.
3. Start from `_groups` and **only `matmul_by_gemm`** in [gemm_vs_matmul/kernel.py](../../examples/matmul/gemm_vs_matmul/kernel.py): five single-slot groups, two loads, one gemm, one store. Read the manual variant only to answer a lowering question. Keep the file declaration literal, e.g. `写法：语法糖。` ([example convention](../tooling/example-coverage.md)).
4. Add the feature you need using the table below. For transpose, inspect [transpose_reinterpret/kernel.py](../../examples/matmul/transpose_reinterpret/kernel.py); for extents, [tail_extent/_kernels/m.py](../../examples/matmul/tail_extent/_kernels/m.py). These are alternatives for resolving an uncertainty, not a required reading sequence. Use VF for the remaining epilogue when the available fused operations do not express its contract.
5. Check capacity before adding K/N splitting. If the unsplit operand fits, its simpler chain is a useful first board slice. If splitting is required, identify the first accumulator write and later accumulation before coding the loop. Count the [complete M/N output grid](../api/gemm.md#count-the-complete-output-grid-before-assigning-workers) before capping workers by one axis; for runtime groups, compare [tile ownership across groups](../api/gemm.md#distribute-tiles-across-runtime-groups). Then [choose M/N/K blocks together](../api/gemm.md#from-capacity-planning-to-performance-candidates): a full K can force narrow N strips and more GM rereads; fit alone does not choose the fastest block or loop order. For a transformed operand, first plan its [consumer format, independent Vec/L1 blocks and producer reuse](../api/gemm.md#transformed-operands-format-residency-and-repeated-work); the GEMM grid alone does not describe conversion work or available parallelism.
6. Run the smallest contractual case, then expand batch/transpose/tail/worker coverage with sentinel outputs. Do not wait until the entire generalized kernel is written to discover its first parse error. Compare alternatives at the same shape and worker count, including all native tasks and full-output GM passes. Fusion can change accumulation order, so keep the numerical reference and tolerance fixed during that comparison.

`pl.gemm(acc, a, b, l0a=la, l0b=lb)` rewrites **at AST level** into exactly the chain the older examples hand-write, and the rewritten statements go through the normal parse path — so auto_mutex locks each `move`/`matmul` on its real pipe. The sugar is not a sync bypass and adds no IR op.

| Hand-written step (what the old examples show) | The sugar spells it |
| --- | --- |
| `pl.move(la, a); pl.move(lb, b); pl.matmul(acc, la, lb)` | the bare call, `init=True` (default) |
| First K step `pl.matmul`, later steps `pl.matmul_acc(acc, acc, ...)` | `init=True` / `init=False`; under `splitk=` the first-step `if` is generated (statically, not a runtime flag) |
| K loop: `move` with K offsets slicing the already-loaded L1, L0 group `.next()` per step | `splitk=STEP` — acc bound once above the loop, L0 staging rotates inside it; no GM reload |
| Bias on the **first** K segment only: `pl.matmul(acc, la, lb, bl)` | `bias=bl` — refused with `init=False` (a bias is folded into the accumulator *write*) |
| Tail block: `set_validshape` on l0a / l0b / acc | `m=` / `n=` / `k=` — narrows only the tiles carrying that axis; put `compact=1` on those tile declarations, never on the `gemm` call; L1 deliberately untouched; runtime values OK (except the split loop bound) |
| N-strip loop over the accumulator | `splitn=STEP` (16-column granularity; refused together with `m=` — measured pitch corruption) |
| Transpose via `pl.reinterpret(..., layout=pl.ZN)` alias | `trans_a=` / `trans_b=` (named Tile, parse-time dims) |
| MX: four paired moves (data + its plane) + `pl.matmul_mx` | `pl.gemm_mx(acc, a, b, sa, sb, l0a=, l0b=, scale_a=, scale_b=)` |
| Reusing one L0/acc set across **shape-varying** matmuls | declare the whole chain (L1, L0, acc) at the *largest* shape, `compact=1` on narrowed tiles, pass `m=`/`k=`/`n=` on the smaller calls — board-proven; a *small* L1 tile against a large L0 is refused. Shape-specialized factories must test the entire axis shorter than their chosen split, separately from a partial last split; see [split extent boundaries](../api/gemm.md#effective-dimensions-and-splitting) |

`l0a=` / `l0b=` accept a Tile **or a TileGroup** — pass the group when splitting, so its slots actually rotate. Full narrowing/refusal contracts live in [api/gemm.md](../api/gemm.md) (owner).

Use the [named-operand form](../api/gemm.md#division-of-responsibilities-between-the-caller-and-the-sugar) for transpose and splitting; selecting a group slot inline can prevent shape-dependent lowering.

## What stays yours either way

The sugar starts at L1 and ends at the accumulator. You still own:

- GM→L1 `pl.load`s — and deciding the L1 valid region *before* loading;
- L0/acc declarations: address, mutex ids, depth — **shared single L0 base** convention ([api/memory-allocation.md](../api/memory-allocation.md)), double buffering stays a choice;
- draining: `pl.store(out, acc, ...)`;
- the bias staging chain GM→L1(Mat)→L0(Bias, FP32) — `bias=` takes the L0 Bias tile;
- cross-core sync layer choice (B-layer pipeline vs manual events).

## The only stated reasons to hand-write (closed list, source-verified)

1. **`phase=` (unitFlag)** — the sugar never emits AccPhase/STPhase; that perf mechanism requires the manual chain. Teaching owner: `examples/matmul/fundamental_usage/acc_phase`.
2. **`gemm_mx` + bias** — `matmul_mx` has no bias operand. Official fallback: drain, then add on the vector side.
3. **`gemm_mx` + split** — a scale plane's group axis is indivisible. Official fallback: write the outer loop yourself, call `gemm_mx` per step (`init=True` on the first).
4. **`splitn` and `splitk` together** — the sugar refuses; write the outer loop, keep the sugar inside it.
5. **Runtime-variable step count** — split sizes are parse-time constants.
6. **An unchanged L0 operand reused by several matmuls** — inspect the expansion first. In the lowering checked at behavior `75a6c2600`, ordinary `gemm` stages both operands on each call; repeated calls sharing a Right tile do not by themselves retain its contents. A native `move` once followed by `matmul` / `matmul_acc` calls may express that reuse. State which operand remains live, its physical extent and slot, the independent accumulator lifetimes, and the overwrite boundary. Keep the normal tile-group mutex analysis; this is not permission to drop synchronization. Validate an independent reference and compare against sugar at matched geometry before claiming a gain. [L0 operand reuse](../../examples/matmul/l0_operand_reuse/) supplies the minimal numerical and generated-code control. Fewer moves do not guarantee lower latency.

Note the shape of 2–5: the escape is an **outer hand-written loop around a sugar call per step**, not a return to the full manual chain. Cite the failing row before hand-writing.

## Failure signatures

- Error mentions a `_pl_gemm*` name → that is a lowering temp; read it against the mapping table above.
- `cannot combine 'm' with 'splitn'` → measured fractal-pitch corruption; restructure.
- Wrong values, no error, after passing an extent → the narrowed tile was not `compact=1` (matmul reads compact; a non-compact tile keeps full physical layout).
- `gemm_mx` K refused → K must be a multiple of 64 (two 32-element scale groups pair up).

## Runnable references

- [examples/algorithms/ragged_grouped_gemm](../../examples/algorithms/ragged_grouped_gemm/) — device-resident group offsets, short-group weight reuse and long-group partitioning in one Cube task; tests include all supported K/N pairs, changing runtime metadata and offset guards. Its conditional ownership and block sizes have a measured scope, not a universal dispatch rule.
- [examples/algorithms/gemm_rmsnorm](../../examples/algorithms/gemm_rmsnorm/) — independently blocked Cube and row-reduction stages, using the output as an intermediate and batched VF row processing; guarded tests include simultaneous domain maxima and cancellation with an unchanged FP64 reference tolerance.
- [examples/algorithms/broadcast_gemm_relu](../../examples/algorithms/broadcast_gemm_relu/) — broadcast weights, M tails, native bias initialization and a fixed-function ReLU drain in one Cube task; tests hold the FP64 reference and tolerance fixed.
- [examples/algorithms/batched_transposed_gemm](../../examples/algorithms/batched_transposed_gemm/) — high-rank GM offsets, transpose at the L1 boundary and M narrowing in one single-slot kernel; N/K alignment requirements are explicit. Run its `test.py` for shape, guard and worker-count checks.
- [examples/matmul/gemm_vs_matmul](../../examples/matmul/gemm_vs_matmul/) — the same multiply hand-written and via sugar, minimal case, on-board.
- [examples/matmul/split_k](../../examples/matmul/split_k/), [split_n](../../examples/matmul/split_n/), [gemm_mx](../../examples/matmul/gemm_mx/) — sugar forms of the split/MX rows.
- Mechanism owners on the manual side (their docstrings state why they hand-write): [fundamental_usage/acc_phase](../../examples/matmul/fundamental_usage/acc_phase/), [bias_fusion](../../examples/matmul/fundamental_usage/bias_fusion/), [matmul_mx](../../examples/matmul/fundamental_usage/matmul_mx/).
- Upstream STs: `pypto/python/tests/st/pypto_pro/frontend/matmul/test_gemm_st.py`, `test_gemm_matrix_st.py`.

## Do not use when

- Vector-side compute — that is VF territory ([patterns/vf-row-reduce.md](vf-row-reduce.md)).
- You need C-layer manual cross-core events around the matmul — the sugar composes with them, but the sync design is owned by [api/crosscore-sync.md](../api/crosscore-sync.md).

## Source escape

Signature: `python scripts/check.py api pl`, grep `gemm`. Contracts: [api/gemm.md](../api/gemm.md) and [api/gemm-mx.md](../api/gemm-mx.md) (owners — if this page disagrees, they win; fix this page). Official page: `../pypto/docs/zh/api/pro_api/SIMD-API/cube_computation/gemm.md`. Implementation: `pypto/python/pypto_pro/language/parser/_gemm_lowering.py` @ `b13f8cfaf`.
