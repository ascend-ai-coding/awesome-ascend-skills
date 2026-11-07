# Measured SIMT Findings

The native examples in [`examples/basic_api/simt/`](../../examples/basic_api/simt/) own runnable evidence and toolchain scope in their `kernel.py` / `test.py` headers. This page keeps API-family findings; historical passes do not establish acceptance on the current installation.

The contract table (which APIs each example should cover) is in [`tooling/example-coverage.md`](../tooling/example-coverage.md).

## 1. Responsibilities of the Five Examples

`simt_launch` covers thread/block indices and syncthreads; `simt_math` covers transcendental functions and fma; `simt_round_pred` covers rounding and predicates; `simt_atomic` covers atomic families; `simt_ring_fence` covers ring increments/decrements and fences.

See [`tooling/example-coverage.md`](../tooling/example-coverage.md) for the API coverage of each example; use `pytest examples/basic_api/simt -q` for the current test count.

## 2. The Two SIMT Kernel Shapes

| Form | When to use |
| --- | --- |
| Pass a GM tensor **directly** to a SIMT function (no tile, no synchronization). | Each thread touches only its own data. |
| First use `make_tile` + `pl.load`, then pass the tile in. | Threads need to exchange data through UB. |

The first form is used by `simt_round_pred` / `simt_atomic` / `simt_ring_fence`; the second by `simt_launch` / `simt_math`. **This distinction is worth calling out:** both forms appear in SIMT STs, so “must move into UB first” is a misconception.

SIMT scalar operations use the `pl.simt` family. In particular, outer-DSL `pl.min` is not interchangeable with `pl.simt.min` (`pypto/docs/zh/api/pro_api/SIMT-API/scalar_compute/min.md`); the latter takes same-dtype integer or floating Scalars, as confirmed by `_simt_api.py::Simt.min`. After a namespace rejection, check that family's signature before changing the work domain. On behavior version `872df763f`, an INT32 `pl.simt.min` feeding a runtime `pl.range(begin, end)` passed empty, truncated and full-range board probes. Match the index dtypes explicitly where needed; a failed `pl.min` call does not establish that dynamic bounded traversal is unavailable.

## 3. Two Launch Brackets: Omitting the Second Does Not Default It to 1

| Brackets | Controls |
| --- | --- |
| `body[x, y, z](...)` | How threads are arranged within one block (inside the kernel body). |
| `kernel[None, 1](...)` | **How many blocks run** (on the Host side). |

The same kernel was called both ways in a probe: with `[None, 1]`, `grid_dim.x == 1` and `block_idx.x == 0`; when called directly, `grid_dim.x == 56` (the number of cores on that machine), while `block_idx.x` identified **which block wrote last and won**—the probe once observed 9, a number with no semantic meaning.

`warp_size()` = **32** (measured).

`linear_thread_idx` is defined as `x + y * bdim.x + z * bdim.x * bdim.y`. The check folds the measured values of all three axes and compares the result with the measured linear index—**it does not compare against any expected constant**, so it verifies internal consistency in the framework.

## 4. Precision of SIMT Scalar Math (Measured Calibration)

| Group | Operation | Maximum relative error | Tolerance |
| --- | --- | ---: | --- |
| **Bitwise equal** | `fma` `abs` `min` `max` | **0** | `rtol=0, atol=0` |
| About 1 ulp | `sqrt` `rsqrt` `exp` `exp2` `log` `log2` `log1p` `sin` `cos` | ≤ 1.18e-07 | `rtol=2e-07` |
| About 3 ulp | `tanh` | 3.48e-07 | `rtol=5e-07` |

FP32 eps is 1.19e-07, so the second group differs at roughly one ulp on these inputs. **`sqrt` is not in the measured bitwise-equal group**: its maximum relative difference was 1.12e-07. Do not infer exact agreement with the Host reference solely from the operation name; the table describes this calibration workload, not all floating-point inputs.

**Bitwise-equal `fma` is itself a finding:** computing `a*b+a` in float64 in the reference and then converting to float32 exactly matched hardware, showing that it rounds only once.

## 5. `rint` and `round` Differ at `.5`; `cast` Truncates

| | Behavior at `.5` | `-2.5` | `-0.5` | `0.5` | `2.5` |
| --- | --- | ---: | ---: | ---: | ---: |
| `pl.simt.rint` | **Round to nearest, ties to even** | `-2` | `0` | `0` | `2` |
| `pl.simt.round` | **Round away from zero** | `-3` | `-1` | `1` | `3` |

In 32 measured inputs, the two differed **at exactly four positions**, all `.5` values (`±1.5` were not among them because `2` is even). The check directly asserts “these four positions differ”—**stronger than checking each against a reference**, because it makes the distinction between the two operations itself the criterion.

**`pl.simt.cast(FP32 → INT32)` truncates toward zero** (default `RoundMode.CAST_NONE`); it was bitwise equal to `trunc` and differed from `rint`. An interface called `cast` paired with a default called `CAST_NONE` does not reveal “truncation.”

## 6. Ring Counter Period Is `limit + 1`

| | Rule | Old value on the kth call (zero-based) |
| --- | --- | --- |
| `atomic_inc(t, limit)` | `old >= limit ? 0 : old + 1` | `k mod (limit + 1)` |
| `atomic_dec(t, limit)` | `old == 0 ? limit : old - 1` | `(-k) mod (limit + 1)` |

Measured with `limit=6` and 32 threads: final `inc` value was 4 = `32 mod 7`, and final `dec` value was 3. **These parameters distinguish the candidate periods `limit` and `limit+1`** (`limit` would produce 2).

The criterion uses the **multiset of return values** (thread order is unknown, but the number of appearances of each old value is deterministic); both count rows match the measurements exactly.

## 7. Order Independence in the INT32 Example

For this non-overflowing INT32 example, the final values of `add` / `sub` / `max` / `min` / `and` / `or` / `xor` do not depend on thread completion order; the criterion is the reduction value. `exch` / `cas` **are order-dependent**; the check asserts only order-independent properties (the final value belongs to the candidate set and there is exactly one winner).

Do not generalize the INT32 finding to floating-point accumulation. Each FP32 `atomic_add` update rounds, so different update orders may produce different bit patterns; atomicity prevents an update from being overwritten but does not guarantee a fixed summation order. Keep the tolerance given by the task, validate each run against an independent high-precision reference, and cover concentrated hotspots, dispersed addresses, and cancellation. Do not avoid order effects by widening tolerance or requiring bitwise equality across calls. If the contract requires determinism, choose a [fixed-order or exact accumulation strategy](numerics.md#6-reproducible-reductions) and verify both accuracy and reproducibility.

**The strongest check is the return values of `atomic_add`:** when 32 threads each add 1, the 32 returned old values must be **exactly one permutation of `0..31`**. A missing value means an update was lost; a duplicate means two threads saw the same old value. **The final value alone cannot prove this**—it could equal 32 even if 32 additions were merged into one.

## 8. Two New Rules, Both Cases of “Documented but Not Enforced”

| Rule | Symptom | Where to enforce |
| --- | --- | --- |
| Atomic operand dtype must match the target | Rejected during parsing; **the message is precise**. | [Writing constraints](constraints.md) |
| `syncthreads` cannot appear in runtime control flow | **Hangs**; parsing and compilation do not reject it. | `syncthreads-must-be-uniform` in `check.py`; see [writing constraints](constraints.md) for the form. |

For the first rule, `tid + 1` and Python integer literals are both INT64 in SIMT; they are rejected when the target is INT32. The diagnostic can point at the atomic call even when the mismatched dtype originates in the preceding `+ 1`; trace the producer as well as the target declaration.

The docstring for the second rule says “cannot be placed inside runtime if/for/while,” while the recorded parser/compiler probes did not reject it. Threads reaching a barrier without all peers can wait indefinitely. **The example verifies parsing only and is not run on the shared board**; the conservative source check is `syncthreads-must-be-uniform`, with its scope documented in [tool usage](../tooling/checks.md#scope-and-known-limits). **Fences have a different contract**: `threadfence_block` / `threadfence` are allowed inside runtime control flow according to their docstring and passed the example's measurements.

## 9. Inner Tensor Annotations and Factory Closures

The outer `@pl.jit` entry retains the Tensor contract; parameter types for an inner SIMT function may be inferred from call arguments and need not be repeated. The official `development/vector_computation/simt_computation.md` explicitly allows omitted annotations as well as Tensor/Scalar annotations (Tile parameters cannot be annotated). Python `_instantiate_simt_function` passes argument types to `_parse_function_param`: with an annotation, it additionally checks compatibility; without one, it uses the argument type directly.

When factory-local sizes **appear only in an inner annotation**, Python may not preserve them in that function's `__closure__`. `_build_function_closure` collects globals and actual free variables; it does not restore every local variable from the function's creation scope. Thus `x: pl.Tensor[[rows, cols], I32]` may raise `Cannot resolve shape policy for tensor parameter 'x'` even though the same annotation works on the outer entry. When this occurs, first check whether the variable was actually captured; the simplest form is `def body(x, out): ...`, letting the Tensor types passed by the outer function determine the shape.

This does not mean “SIMT does not support Tensor annotations” or “factory closures never work.” On 2026-10-06, behavior version `872df763f`, a 2D copy was compared in the same environment: it failed to parse when only the annotation referenced local sizes; three variants each passed elementwise on board when the function body also used the sizes, when dimensions were changed to literals, or when the inner annotation was omitted. The failure is in Python parsing and does not reach device execution.

## 10. First Determine Thread Ownership and Total Work

For each candidate mapping, list the data unit owned by one thread, whether threads share state, work per thread, total work across all threads, and Host grid coverage conditions. Do not assume one output element must map to one thread: different decompositions may reread or recompute the same inputs. Compare work and synchronization needs first, then decide whether UB sharing, cooperative reduction, or finer parallelism is needed; fewer total operations do not automatically mean faster hardware execution.

For a one-dimensional mapping without a grid-stride loop, `linear = block_idx.x * block_dim.x + linear_thread_idx()` requires the Host block count to cover the full work set, and tail threads still need a bounds check. **The number of data blocks is not the same as the logical-core budget.** The current native launcher resolves the budget using resource limits of the current Stream; an explicit over-budget request is rejected before device execution. It is neither automatically scheduled over an arbitrary number of rounds nor silently truncated. A direct `kernel(...)` call uses the automatic budget. Evidence: official `kernel_function.md`, `runtime/jit.py::__getitem__`, C++ `include/pypto_launch.h::ResolveLaunchBlockDim`, and a probe in this task: requests for 68/134 blocks were rejected by the current Stream budget of 56 blocks; 56 was specific to that environment, not a universal constant.

A general coverage pattern is `start = block_idx.x * block_dim.x + linear_thread_idx()`, followed by `pl.range(start, work_items, grid_dim.x * block_dim.x)` so each thread strides through later work. The Host may choose a legal small grid or use automatic budgeting; it must not merely shrink the former ceil-div grid without changing the work mapping. Use the [SIMT starter example](../patterns/simt-first.md) to check coverage for a small grid, automatic budgeting, and work beyond one grid. Direct-GM forms do not add bounds checks for you; continue to validate the two launch brackets separately as described in §3.

That global grid partitions a shared work domain. If an outer MIX loop already owns a work item and SIMT fills its private Vec tile, each invocation must cover its entire local extent: start from the local thread ID and stride by the local thread count. A global block offset/stride can leave private tiles partially unwritten. Keep this inner coverage separate from the [common AIC/AIV work-item mapping](crosscore-sync.md). Validate coordinate-distinct data across multiple items, both AIV regions and worker counts, checking that each private tile is fully initialized before publication.

### Reuse Metadata Across Contiguous Payloads

When several contiguous elements share coordinates, weights, indices, or a destination address, first read [SIMT payload mapping and validation](../patterns/simt-payload.md). That pattern separately covers logical thread-group mapping, per-thread state reuse, joint measurement of block/group width, and data-validity rate; this page retains SIMT API and type constraints.

### Work Assignment for Unequal-Length Segment Reductions

Use [runtime-reduction partitioning](../patterns/multi-output-reductions.md#partition-unequal-runtime-reductions) for direct owners, owned partials, empty work, merge ordering and workload comparisons. This API page owns the SIMT launch, index-type and synchronization constraints used to implement that mapping.

## 11. Argument Physical Shapes Must Be Static; Input Sizes Can Be Specialized Per Call

The current SIMT launch requires GM Tensor arguments to have nonzero rank, static shape, and ND layout. A dynamic dimension in an outer `pl.Tensor[[pl.DYNAMIC], dtype]` is rejected when passed to SIMT as `simt.launch Tensor argument must have a static shape`. This matches the data-access table in the official `simt_computation.md`, Python's delayed instantiation and argument-type forwarding, and the C++ check in `ir/op/simt_ops.cpp` requiring every dimension to be a `ConstInt`.

When size changes across calls, use `pl.Tensor[[pl.STATIC], dtype]` to create a compile variant for the actual size, or use an ordinary Python factory to generate a fixed-shape entry point. Inner parameter annotations may still be omitted and inferred from the argument type. `STATIC` passed direct-GM copy for multiple lengths under behavior version `872df763f`; it does not mean one binary is reused for every length. Runtime indexing, loop bounds, and logical valid counts over a static physical shape are separate matters; do not ban them together.

## 12. When an Existing Variable Reports “Potentially Undefined,” Check Branch Types

In addition to genuine missing assignments, the current parser may produce `Use of potentially undefined variable` when state types differ across branch arms. For example, a local read from an INT32 Tensor may become INT64 after adding a Python integer literal on one arm, while remaining INT32 on the other. `_expression_parser.py::parse_name` binds the merged `NoneType` as this name error; do not infer that SIMT lacks branch support or that atomics cannot use variable indices.

Choose a state type based on the value-range contract and keep the incoming value and every branch update consistent. Explicitly widen the initial value to INT64, or convert the update back to INT32 after proving it cannot overflow; both passed separate scalar read/write controls under behavior version `872df763f`. Conversion and atomic operand rules remain as described in §8; do not silently change the numeric range to remove the error.

## 13. Scalar Indices and Custom Physical Strides

Under behavior version `872df763f`, scalar indexing `A[i,j,...]` is linearized using the shape in contiguous row-major order; it does not read a TensorView's custom stride. Passing a 2D view declared with `pl.make_tensor` strides into SIMT does not establish that multi-index reads are correct; this differs from the view-transfer path used by `pl.load/store`. Installed Python `_expression_parser.py::_parse_scalar_subscript_index` explicitly expands indices using products of later dimensions, and C++ `backend_cce_ops.cpp::MakeGetValCodegenCCE` then reads the pointer at that element offset.

For scalar access with positive strides, derive addresses from the actual shape/stride: `offset = sum(index[d] * stride[d])`; the physical span from the logical first element is `span = 1 + sum((shape[d]-1) * stride[d])`. You can create a native contiguous 1D view `pl.make_tensor(x, [span], [1])`, access it on device with explicit physical offsets, and prove that every access stays inside `span`. Host reads metadata only and the external input/output contract remains unchanged; do not add a storage offset a second time when it is already included in the pointer. See [Host stride and guards](datamove.md#host-tensor-strides-and-boundary-guards).

This method was validated on CANN 9.2.0 / Ascend950PR for non-unit column stride, row padding, nonzero storage offset, multiple worker counts, and full backing guards; the complete consumer is [strided_transpose](../../examples/algorithms/strided_transpose/). It is an adaptation for the current scalar-indexing path, not a claim that “the framework does not support stride.” After an upgrade, check the installed source and a minimal readback before deciding whether explicit linearization is still needed.

---
