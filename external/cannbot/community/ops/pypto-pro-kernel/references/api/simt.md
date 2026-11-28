# Measured SIMT Findings

The native examples in [`examples/basic_api/simt/`](../../examples/basic_api/simt/) own runnable evidence and toolchain scope in their `kernel.py` / `test.py` headers. This page keeps API-family findings; historical passes do not establish acceptance on the current installation.

The contract table (which APIs each example should cover) is in [`tooling/example-coverage.md`](../tooling/example-coverage.md).

For a SIMT helper created in a Python factory, or `Cannot resolve shape policy for tensor parameter`, read [inner annotations and captured dimensions](#9-inner-tensor-annotations-and-factory-closures). For a dynamic Tensor argument rejected at launch, read [physical shapes and per-call specialization](#11-argument-physical-shapes-must-be-static-input-sizes-can-be-specialized-per-call).

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

### Compile-time launch contract and mechanism availability

On source `c46548eac0ce40350a5d19293368552c2fae3dc5`, Python `parser/decorator.py` validates `max_threads`, `ir/op/simt_ops.py` parses the launch and C++ `ir/op/simt_ops.cpp::DeduceSimtLaunchType` requires positive constant dimensions with product at most `max_threads` and 2048. `CCECodegen::BuildSimtFunctionSignature` emits `max_threads` as `__launch_bounds__`. A smaller actual launch is legal; runtime-dependent thread dimensions are not established by Ascend C's `Dim3` examples. The official Pro SIMT guide (`pypto/docs/zh/guide/programming_guide/pro/development/vector_computation/simt_computation.md` in the matching source checkout) agrees with this contract. Resource and performance consequences belong to [launch selection](../patterns/simt-first.md#launch-variants), not a new hard equality rule.

That guide also explicitly excludes warp shuffle/vote/reduce, thread-private arrays and explicit cached-GM interfaces. The corresponding names and alternate spellings were checked in the Python language/parser/IR and C++ Pro IR/backend sources at the same revision. Keep upstream warp/cooperative-group/cache recipes as capability leads until the installed version exposes the required primitive and a focused probe verifies it. `warp_size` alone does not expose warp communication; a scalar lane-local variable is not a broadcast. An ordinary scalar load is not an explicit cache-policy selector. Recheck these source boundaries after upgrades rather than inferring general hardware absence.

## 4. Precision of SIMT Scalar Math (Measured Calibration)

| Group | Operation | Maximum relative error | Tolerance |
| --- | --- | ---: | --- |
| **Bitwise equal** | `fma` `abs` `min` `max` | **0** | `rtol=0, atol=0` |
| About 1 ulp | `sqrt` `rsqrt` `exp` `exp2` `log` `log2` `log1p` `sin` `cos` | ≤ 1.18e-07 | `rtol=2e-07` |
| About 3 ulp | `tanh` | 3.48e-07 | `rtol=5e-07` |

FP32 eps is 1.19e-07, so the second group differs at roughly one ulp on these inputs. **`sqrt` is not in the measured bitwise-equal group**: its maximum relative difference was 1.12e-07. Do not infer exact agreement with the Host reference solely from the operation name; the table describes this calibration workload, not all floating-point inputs.

**Bitwise-equal `fma` is itself a finding:** computing `a*b+a` in float64 in the reference and then converting to float32 exactly matched hardware, showing that it rounds only once.

<!-- kernel-query: {"id":"simt-math-lowering","aliases":["SIMT math cost","SIMT transcendental lowering","SIMT log1p accuracy","SIMT 数学接口成本","SIMT 数学精度边界"],"term_groups":[["simt"],["math cost","transcendental","log1p accuracy","数学接口成本","数学精度边界"]],"when":"Choose a scalar math implementation or reuse nonlinear intermediates under the original numerical contract.","missing_facts":["Match dtype and installed Python/C++ lowering, test the task's actual input range including sensitive values, and compare emitted code plus complete native calls; names and source expression counts do not establish instructions, accuracy or speed."]} -->
### Actual FP32 lowering and sensitive inputs

The calibration above does not establish the accuracy of every input to a named API. On source `c46548eac0ce40350a5d19293368552c2fae3dc5`, `_simt_api.py` (`pypto/python/pypto_pro/language/_simt_api.py` in the matching source checkout) declares the interfaces while `backend_cce_simt_ops.cpp` (`pypto/framework/src/interface/pypto_pro/backend/backend_cce_simt_ops.cpp` in the matching source checkout) emits the following FP32 expressions. These are compiler input expressions, not final ISA counts or throughput guarantees; FP16/BF16 paths have separate conversions and support boundaries.

| Pro operation | FP32 CCE expression or composition |
| --- | --- |
| `sqrt(x)` | `__sqrtf(x)` |
| `rsqrt(x)` | `1.0f / __sqrtf(x)` |
| `exp(x)` | `__expf(x)` |
| `exp2(x)` | `__expf(x * 0.6931471805599453f)` |
| `log(x)` | `__logf(x)`, with a scale/log/unscale path for positive subnormals |
| `log2(x)` | The guarded natural log divided by `__logf(2.0f)` |
| `log1p(x)` | `__logf(1.0f + x)`; this API accepts FP32 only |
| `tanh(x)` | `1.0f - 2.0f / (__expf(2.0f*x) + 1.0f)` |

The official `log1p` precision contract (`pypto/docs/zh/api/pro_api/SIMT-API/scalar_compute/log1p.md` in the matching source checkout) explicitly equates its accuracy to FP32 addition followed by natural log. For example, current native FP32 probes at `x=±1e-8` produce zero because `1+x` rounds to one. A task requiring accurate tiny `log1p(x)` or `log1p(x)/x` cannot assume this name provides a cancellation-resistant implementation. The ratio also needs its continuous limit at zero. Preserve a correct stable formulation, or compare a range-proved series/polynomial with a justified truncation error; do not widen the task's tolerance or replace it with the native API as the oracle.

The emitted tanh formula likewise subtracts nearly equal values near zero. At large negative arguments it can round to -1 while an algebraically transformed exponential can overflow elsewhere. Rewrites, shared intermediates and FMA can change rounding, domains and exceptional behavior. Use an independent higher-precision reference formed from the actual quantized input, the original tolerances and each contractual zero, subnormal or extreme-value regime. A bounded finite-input optimization does not establish an all-input replacement.

When outputs share nonlinear expressions, compare computing an exponential or reciprocal once and reusing it. Confirm whether the installed compiler already performs the sharing and whether the proposed form adds live state or changes accuracy; one source-level exponential is not proof of one issued instruction or a speedup. The [performance evidence rules](../diagnostics/performance.md#simt-memory-paths-and-cache-evidence) exclude substituting upstream single-instruction tables for matched complete-call measurements.

<!-- kernel-query: {"id":"simt-low-precision","aliases":["SIMT FP16 precision","SIMT BF16 accumulator","SIMT low precision","SIMT 低精度","SIMT 低精度累加"],"term_groups":[["simt"],["fp16","bf16","bfloat16","low precision","低精度"]],"when":"Choose intermediate precision or preserve rounding points in a FP16/BF16 compound computation or sequential accumulation.","missing_facts":["Separate input/output storage dtype from intermediate and loop-carried dtype; identify whether the contract rounds once or after each update, and verify the installed conversion/lowering path."]} -->
### FP16/BF16 compound arithmetic and rounding contracts

Storage dtype does not specify the precision of a whole expression. On source `c46548eac0ce40350a5d19293368552c2fae3dc5`, BF16 `rsqrt` converts its argument to FP32, computes, then converts back to BF16. That internal FP32 carrier does not keep a chain of BF16 operations in FP32. Single-operation coverage likewise does not establish the error of a compound expression.

For `x * rsqrt(s + eps)` under a mathematical accuracy contract, widen the inputs and epsilon **before** the sensitive addition and retain the chosen compute dtype through the expression, then convert the output. Converting FP32 `1e-8` to FP16 first yields zero on the verified path; widening that zero later cannot recover epsilon. A BF16 chain can also accumulate more rounding error than its individual operations. Use the actual quantized inputs for the independent reference, and retain the task's output tolerance rather than borrowing the FP32 calibration above. The cast contract (`pypto/docs/zh/api/pro_api/SIMT-API/scalar_compute/cast.md` in the matching source checkout) specifies modes by source/target pair: FP16/BF16 to FP32 uses the default mode; FP32 to either low-precision type supports explicit `CAST_RINT`. Do not apply that explicit mode indiscriminately to identity conversions.

Initialize a loop accumulator at its intended compute dtype and convert loaded contributions before arithmetic. In this revision, a low-precision loop-carried Scalar is emitted as `half` or `bfloat16_t`, while an FP32 carry is emitted as `float`. Widening only the final result cannot restore contributions lost during earlier updates. Conversely, a contract requiring nearest-even low-precision rounding after **every** addition differs from a mathematical sum rounded once. Preserve the low-precision carry, or explicitly round each FP32 update to the required dtype before continuing; keeping an unrounded FP32 sum can violate the former contract even when it is closer to the mathematical sum.

Compare legal alternatives with the same public storage dtype, addresses, work and launch geometry. Added conversions or a changed dependency chain can offset an accuracy benefit. Widening is a numerical choice, not a general latency optimization; use the existing [complete-call timer](../verification/native-timing.md) and [compiler evidence](../diagnostics/performance.md#bind-resources-to-the-device-function) for the selected consumer.

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

<!-- kernel-query: {"id":"simt-entry-annotations","aliases":["SIMT entry annotations","SIMT outer wrapper","SIMT inner annotations","SIMT factory closure","SIMT 内外层注解"],"symbols":["pl.simt.atomic_add"],"example":"basic_api/simt/simt_atomic/kernel.py","when":"Inspect outer-entry classification or an inner SIMT helper's annotation and captured dimensions; an unresolved shape diagnostic does not by itself establish a closure failure.","missing_facts":["Identify the outer JIT entry and inner SIMT helper; check the actual annotations, argument types and captured factory-local dimensions before changing the implementation paradigm."],"term_groups":[["Cannot resolve shape policy for tensor parameter"]],"review_symbols":[["pl.simt.atomic_add"]],"discovery_terms":[["simt"],["entry","outer","wrapper","factory","closure","内外层"],["annotation","annotations","type","typing","closure","注解"]]} -->
## 9. Inner Tensor Annotations and Factory Closures

The outer `@pl.jit` entry retains the Tensor contract; parameter types for an inner SIMT function may be inferred from call arguments and need not be repeated. The official `development/vector_computation/simt_computation.md` explicitly allows omitted annotations as well as Tensor/Scalar annotations (Tile parameters cannot be annotated). Python `_instantiate_simt_function` passes argument types to `_parse_function_param`: with an annotation, it additionally checks compatibility; without one, it uses the argument type directly.

When the outer entry's native classification is uncertain, first inspect [the imported callable and its actual annotations](constraints.md#check-the-imported-entry-before-changing-its-layout). A downstream parameter-parser branch does not establish that this callable reaches native parsing.

When factory-local sizes **appear only in an inner annotation**, Python may not preserve them in that function's `__closure__`. `_build_function_closure` collects globals and actual free variables; it does not restore every local variable from the function's creation scope. Thus `x: pl.Tensor[[rows, cols], I32]` may raise `Cannot resolve shape policy for tensor parameter 'x'` even though the same annotation works on the outer entry. When this occurs, first check whether the variable was actually captured; the simplest form is `def body(x, out): ...`, letting the Tensor types passed by the outer function determine the shape.

This does not mean “SIMT does not support Tensor annotations” or “factory closures never work.” On 2026-10-06, behavior version `872df763f`, a 2D copy was compared in the same environment: it failed to parse when only the annotation referenced local sizes; three variants each passed elementwise on board when the function body also used the sizes, when dimensions were changed to literals, or when the inner annotation was omitted. The failure is in Python parsing and does not reach device execution.

<!-- kernel-query: {"id":"simt-worker-coverage","aliases":["SIMT worker grid","SIMT single block slow","SIMT large input slow","SIMT 单 block 性能","SIMT 大输入慢"],"term_groups":[["kernel[None, 1]","kernel[None,1]"]],"when":"A correct independent SIMT kernel uses a small explicit Host grid, or large work scales poorly despite a grid-stride loop.","missing_facts":["Separate actual threads from Host workers, establish the current Stream budget and coverage mapping, and compare worker counts at fixed threads on small and large workloads; one worker is legal and may be appropriate."],"discovery_terms":[["simt"],["single block","one block","worker grid","large input slow","单 block","大输入慢"]]} -->
## 10. First Determine Thread Ownership and Total Work

For each candidate mapping, list the data unit owned by one thread, whether threads share state, work per thread, total work across all threads, and Host grid coverage conditions. Do not assume one output element must map to one thread: different decompositions may reread or recompute the same inputs. Compare work and synchronization needs first, then decide whether UB sharing, cooperative reduction, or finer parallelism is needed; fewer total operations do not automatically mean faster hardware execution.

For a one-dimensional mapping without a grid-stride loop, `linear = block_idx.x * block_dim.x + linear_thread_idx()` requires the Host block count to cover the full work set, and tail threads still need a bounds check. **The number of data blocks is not the same as the logical-core budget.** The current native launcher resolves the budget using resource limits of the current Stream; an explicit over-budget request is rejected before device execution. It is neither automatically scheduled over an arbitrary number of rounds nor silently truncated. A direct `kernel(...)` call uses the automatic budget. Evidence: official `kernel_function.md`, `runtime/jit.py::__getitem__`, C++ `include/pypto_launch.h::ResolveLaunchBlockDim`, and a probe in this task: requests for 68/134 blocks were rejected by the current Stream budget of 56 blocks; 56 was specific to that environment, not a universal constant.

A general coverage pattern is `start = block_idx.x * block_dim.x + linear_thread_idx()`, followed by `pl.range(start, work_items, grid_dim.x * block_dim.x)` so each thread strides through later work. The Host may choose a legal small grid or use automatic budgeting; it must not merely shrink the former ceil-div grid without changing the work mapping. Use the [SIMT starter example](../patterns/simt-first.md) to check coverage for a small grid, automatic budgeting, and work beyond one grid. Direct-GM forms do not add bounds checks for you; continue to validate the two launch brackets separately as described in §3.

A grid-stride loop can make one worker numerically correct while serializing a large independent work set. Compare that control with more legal workers at fixed thread dimensions before attributing scaling to memory or arithmetic. For a flat independent-output mapping with `T` threads, `min(stream_budget, ceil(work_items / T))` is a bounded starting candidate for nonempty work, not an optimal selector. Preserve empty-work dispatch and test small workloads: opening the full budget can add inactive workers without useful parallelism. This formula does not apply to private-tile or cooperative ownership without deriving their work units. Validate the retained selector with the [paired complete-call procedure](../verification/native-timing.md#compare-paired-implementations); retune thread width separately.

That global grid partitions a shared work domain. If an outer MIX loop already owns a work item and SIMT fills its private Vec tile, each invocation must cover its entire local extent: start from the local thread ID and stride by the local thread count. A global block offset/stride can leave private tiles partially unwritten. Keep this inner coverage separate from the [common AIC/AIV work-item mapping](crosscore-sync.md). Validate coordinate-distinct data across multiple items, both AIV regions and worker counts, checking that each private tile is fully initialized before publication.

### Reuse Metadata Across Contiguous Payloads

When several contiguous elements share coordinates, weights, indices, or a destination address, first read [SIMT payload mapping and validation](../patterns/simt-payload.md). That pattern separately covers logical thread-group mapping, per-thread state reuse, joint measurement of block/group width, and data-validity rate; this page retains SIMT API and type constraints.

### Work Assignment for Unequal-Length Segment Reductions

Use [runtime-reduction partitioning](../patterns/multi-output-reductions.md#partition-unequal-runtime-reductions) for direct owners, owned partials, empty work, merge ordering and workload comparisons. This API page owns the SIMT launch, index-type and synchronization constraints used to implement that mapping.

<!-- kernel-query: {"id":"simt-bounded-index-arithmetic","aliases":["UINT32 scalar indexing","unsigned index arithmetic","有界无符号索引","uint32 规避标量冗余"],"term_groups":[["simt","scalar","标量"],["uint32","unsigned","无符号","int32"]],"when":"A nonnegative local index or offset may retain signed division/remainder correction or wide arithmetic.","missing_facts":["Prove the range of every intermediate, subtraction and loop update; preserve wide GM pointers and compare generated code plus the complete call."]} -->
## Bound local integer arithmetic before choosing signedness

A scalar annotation does not establish the type of its actual argument or every expression using it. Trace the caller, induction variable, constants and intermediate products. In a one-dimensional SIMT launch, `pl.simt.thread_idx().x` expresses the actual coordinate directly; a multidimensional launch still needs its full coordinate mapping. Use `pl.simt.cast` inside SIMT, and type the initial value, bounds and updates consistently. A Python integer literal can widen an otherwise narrow expression.

The same range/signedness argument applies to JIT-side scalar scheduling, using that scope's `pl.astype` conversion. It does not establish support for a scalar cast inside VF; verify that separate frontend boundary before moving a conversion into a vector function.

For a local row/word offset whose complete range is in `[0, 2**32)`, UINT32 can expose nonnegative division and remainder to the compiler and avoid correction needed for signed Python semantics. This is conditional on the whole expression, not just its final result: prove that `base + row * pitch + column` fits, that no subtraction underflows, and that a loop's final increment cannot wrap. Keep the actual GM pointer and any wider global product at their required width. Negative indices, negative strides and expressions such as `remainder - worker` do not satisfy this proof automatically. Casting an INT32 payload or negative sentinel to UINT32 is a semantic change.

```python
# Inside a one-dimensional SIMT body; caller bounds rows, pitch and the full span.
lane = pl.simt.cast(pl.simt.thread_idx().x, pl.DT_UINT32)
pitch = pl.simt.cast(row_pitch, pl.DT_UINT32)
row = pl.simt.cast(selected_row, pl.DT_UINT32)
offset = row * pitch + lane
```

Inspect the emitted expression to check that later constants or operands did not restore signed/wide arithmetic. A bounded positive-step `while` can retain a chosen SIMT induction type when `pl.range` lowers to a wider induction variable, but that is an implementation alternative to validate, not a general loop rule. Preserve VF hardware-loop structure separately; adding a narrow loop-carried counter can replace a hardware loop with scalar branches. Fewer static instructions or spills alone do not establish a faster complete call.

<!-- kernel-query: {"id":"simt-physical-shapes","aliases":["SIMT physical shapes","SIMT static shape","simt.launch Tensor argument must have a static shape"],"term_groups":[["simt.launch Tensor argument must have a static shape"]],"when":"A GM Tensor descriptor passed to SIMT has a dynamic physical dimension, or a per-call shape specialization is being chosen.","missing_facts":["Separate physical descriptor dimensions and accessed storage extent from runtime indices, loop bounds and logical valid counts; a rejected dynamic physical shape does not reject runtime traversal."]} -->
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
