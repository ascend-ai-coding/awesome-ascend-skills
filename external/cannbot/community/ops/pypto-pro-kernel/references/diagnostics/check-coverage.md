# Boundaries of Framework Checks

Baseline reviewed on 2026-09-28, pypto `13d4573c064d`. Source inspection shows where checks run; historical probes show what they triggered at the time. Neither replaces verification against the currently installed environment. This page does not count `raise` statements.

## Four mechanisms

### 1. Automatic mutex insertion does not prove that a program is race-free

`auto_mutex` inserts mutexes based on tile-group mutex metadata; a bare `make_tile` does not receive the same protection. Overlap checks now include manually declared slots, but they are still not a general RAW/WAR/WAW or deadlock analysis for arbitrary programs. See [validation traps](validation-traps.md) §7.1 and [tools](../tooling/checks.md) for the actual blind spots and tool scope.

### 2. `pipeline=` is the entry point for transformation and validation

`runtime/jit.py` calls `transform_pipeline` only when `_pipeline is not None`. Checks for loops, distances, slot counts, and event IDs inside a pipeline must not be treated as protection available by default to every kernel.

Analysis also depends on recognized cross-core buffers: `_analyzer.py::_scan_kernel_buffers` collects tile groups, placed groups, and bound GM workspaces. Without cross-core objects, the related access analysis has no input. Providing only `fwd` and omitting `bwd` explicitly gives up reverse-reuse protection; the framework does not fill it in automatically. See [crosscore-sync](../api/crosscore-sync.md) for route selection.

### 3. Non-constant values may skip a check layer

In `block_ops.py`, `_static_dim`, `_tile_shape_ints`, `_const_shape_ints`, `_static_last_axis`, and `_try_get_const_offset` may return None when static evaluation is not possible. To determine whether protection is lost, follow the callers: the DSL may already have rejected the form, the next layer may report an error, or a dynamic `valid_shape` may be allowed by design.

The actionable case found by historical UT probes was a **loop-variable offset**: static negative-offset and out-of-bounds checks could not complete. This also occurs in kernels with static shapes, so it cannot be generalized as “dynamic shapes greatly reduce checking.” Existing `_validate_offsets` emits `CheckSkipped` when appropriate; see [diagnostic messages](messages.md) for usage. Runtime sanitizer can check accesses actually reached, but cannot prove safety for every input.

### 4. A multi-slot check cannot use slot 0 as a proxy for the address

A tile extracted from a group may expose only the slot-0 address in its type's memref. This once let a case with a “correct base address but incorrect spacing between slots” pass `_check_mx_scale_tile`. In pypto `a5c48ce63`, the parser added a per-slot check in `_check_mx_plane_slots`; this must no longer be listed as an unresolved issue.

All slots in a group share shape/dtype/layout/compact; **address is the dimension that varies by slot**. Capacity and alignment are checked per slot at declaration time, so the historical MX blind spot does not mean that all double-buffer checks fail. Explicit scale-plane address lists are covered by the per-slot check.

### 5. Three distinct criteria for out-of-bounds access, each with its own blind spot (board evidence, 2026-09-29)

Numerical comparison, guard rows/poisoned inputs, and runtime sanitizer are three distinct levels of evidence. Each was verified on board using a single-defect variant developed in `examples/algorithms/rmsnorm_gemm_tail`:

| Defect form | Numerical comparison | Guard rows/poisoning | Sanitizer |
| --- | --- | --- | --- |
| Out-of-bounds **write** (padding strategy writes past the end, or store forgets to narrow) | **Invisible** (the out-of-bounds row is not in the comparison range) or **timing-dependent** (writes into a neighboring core's row; symptoms depend on which core runs first—the same variant damaged 18–21 rows in some runs and 0 rows in others) | ✅ Guard rows expose it | ✅ Reports `GM_OUT_OF_BOUNDS` and the exact window |
| Out-of-bounds **read whose result is discarded** (vector forgets to narrow, or extra rows are discarded by Cube's `m=`) | Invisible | Invisible (poisoned input does not leak into the result) | **✅ The only layer that can catch it** |
| Misaligned rows / missing writes / stale values from a prior tile | ✅ (NaN prefill and per-row localization identify it directly) | — | — |

Therefore, **numerically correct output is not evidence that there are no out-of-bounds accesses**; validate non-aligned kernels at all three layers. See the sanitizer activation trap (conflict with `pl.jit(name=)`) in the naming section of [api/constraints.md](../api/constraints.md): starting with pypto dev `b1c683973`, compilation rejects this combination; upstream still silently accepts it.

### 6. Random inputs are blind to recurrent-state defects: structured inputs are what catch them (board evidence, 2026-09-30)

For kernels with state carried across iterations (such as the running maximum and row sum in online softmax), **seven classes of random inputs all failed to catch the real defect where the running maximum was not propagated and only the block-local maximum was used**. With random Gaussian scores, the maxima across blocks differed too little, so the incorrect rescale factor was numerically almost correct. Only one-hot structured input exposed it by forcing `exp(+90)` to overflow; only `all_negative` input caught the defect where `m` was initialized to 0. The criterion is: **construct a dedicated input for every path through recurrent state**—rescaling on every block / only the first block / only the final block / all-negative / one-hot / uniform—and use mutation testing to verify that each input kills its corresponding defect (the methodology with 19 mutations and none surviving is in [flash_attention](../../examples/algorithms/flash_attention/) `test.py`). A larger random-input matrix still is not evidence of recurrence correctness.

## Consumable diagnostics

`PyptoProError` already has `error_code`, `span`, `module`, and `origin`; actual/expected values and parameter names do not yet have a general field contract. DSL spans are propagated through public dispatch, while ordinary runtime calls are usually located through the traceback. Catch using the exception's built-in base class; see [writing constraints](../api/constraints.md) for the specific contract.

## Python and C++ review scope

The following applies only to the mechanisms already inspected and must not be generalized into a capability inventory for the entire framework.

| Question | Conclusion and source basis |
| --- | --- |
| Is there a static proof for manually paired mutexes or arbitrary ordering? | Python `ir/op/system_ops.py` validates parameters; automatic mutexes go through `_op_pipeline.py`. Mutex hits in C++ `ir/` and `pypto_pro/` refer respectively to op registration, codegen, and sanitizer instrumentation. No general static proof was found within this search scope; runtime replay is a separate layer. |
| Does a token pass verify token accounting? | `infer_token_pass.cpp` and `remove_redundant_token_pass.cpp` perform inference/optimization. Historical searches for `PRO_IR_CHECK/PRO_CHECK/throw/CHECK` found no check point, so these passes must not be treated as token-balance validators. |
| Does an IR verifier exist? | TypeCheck, SSA, and NoNestedCall exist and are bound in `python/src/bindings/ir/pass.cpp`; it is not true that there is “no verifier.” |
| Does the verifier run automatically? | Historical searches in `framework/src`, `framework/include`, and `python/src` for `RunVerifier/IRVerifier/Verify` found no Pro-pipeline `AddPass` integration. A callable binding does not mean checks run automatically after each pass. |
| Are `@pl.pipeline.stage` functions rejected in `@pl.jit` when `pipeline=` is absent? | Starting with pypto dev `3b0311911`, compilation rejects this with `InvalidOperation` (a stage function is called by bare name with no pipeline configuration). Upstream versions neither reject nor warn: `stage` is a marker-only decorator (`_stage.py` only sets the `pipeline_stage=True` attribute); without `pipeline=`, the transformation does not run, the stage call is inlined as an ordinary function, and no synchronization is inserted for a `crosscore=True` group. The kernel compiles and runs normally, producing almost entirely zero or incorrect output (board evidence, 2026-09-29). This is in the same family as the crosscore-sync statement “silent until `bind`”: missing configuration is treated as no requirement. |
| Are launch group count and kernel-internal indexing checked against each other? | The native launcher rejects an explicit `block_dim` that exceeds the Stream budget; it does not silently truncate it (rechecked 2026-10-06, see [SIMT work coverage](../api/simt.md#10-first-determine-thread-ownership-and-total-work)). There is still no guard that checks whether an in-budget group count matches the kernel's logical partitioning; incorrectly derived GM offsets may be out of bounds. See the historical mixed-kernel case from 2026-09-29 in [api/crosscore-sync.md](../api/crosscore-sync.md), “`block_dim` is the group count for a mix kernel”; do not treat that case's old launch boundary as the current contract. |

When reviewing a missing capability, search both Python and C++ sides and name variants, then inspect the actual call path; update this table when pipeline integration is added. Mutex domains are shared across pipes by ID, but remain separate across cores; evidence is in [validation traps](validation-traps.md) §7.1. Do not list unverified comparison capabilities as defects or leave them as numbered TODOs.
