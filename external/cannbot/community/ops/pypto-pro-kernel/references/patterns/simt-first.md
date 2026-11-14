# SIMT: start with ownership and a direct-GM slice

API contract: [SIMT findings](../api/simt.md), especially the two launch levels, optional inner annotations and grid coverage. For an independent direct-GM kernel, this page supplies the first runnable slice: write and run it before opening more example groups or an entire API listing. Look up another document only for a named operation or unresolved failure.

1. Name the independent work unit and its input/output region. Estimate both per-thread and total work for that choice; an element-per-thread decomposition can repeat input work. For runtime segments, also estimate the busiest worker's work and compare the [direct-owner and split-reduction choices](multi-output-reductions.md#partition-unequal-runtime-reductions). Compare alternatives before adding shared storage or synchronization. This is an algorithm check, not a hardware speed claim.
2. If threads only access their own outputs, pass GM tensors directly. SIMT Tensor arguments need a static physical shape: use fixed dimensions or `pl.STATIC` for per-call specialization, not `pl.DYNAMIC` ([shape contract](../api/simt.md#11-argument-physical-shapes-must-be-static-input-sizes-can-be-specialized-per-call)). For noncontiguous inputs, derive physical offsets from the real strides; a multidimensional view does not establish the scalar-subscript behavior ([current lowering and portable addressing](../api/simt.md#13-scalar-indices-and-custom-physical-strides)). Keep the external contract on `@pl.jit`; let the inner SIMT function infer its types. For a factory annotation failure, see [the exact closure boundary](../api/simt.md#9-inner-tensor-annotations-and-factory-closures).
3. Keep the two launch levels explicit: `body[T](...)` selects threads per block, host `kernel[None, B](...)` selects worker blocks within the Stream budget. A ceil-div data-block count can exceed that budget. Use the grid-stride slice below for large work sets; reducing the host grid alone drops work. `kernel(...)` uses the automatic budget and is safe for this coverage scheme ([launch and coverage contract](../api/simt.md#10-first-determine-thread-ownership-and-total-work)).

This **copy** slice verifies entry, factory, shapes, indexing and launch; it does not prescribe a reduction or recurrence algorithm. It was checked on behavior version `872df763f` with small and automatic grids, including workloads larger than the grid. The annotation boundary is explained in the owner above.

When runtime coordinates, weights or addresses are shared across a continuous element axis, the element-per-thread slice is only the starting ownership. Use the [record/lane mapping and metadata lifetime](simt-payload.md) to derive another candidate: keep per-record work outside each lane's element loop, prove unique coverage, then compare complete calls at fixed worker counts. Merely increasing threads per block leaves the repeated metadata work in place.

```python
import pypto_pro.language as pl
I32 = pl.DT_INT32
THREADS = 64

def make_copy(rows, cols):
    @pl.vector_function(mode="simt", max_threads=THREADS)
    def body(x, out):
        start = pl.simt.block_idx().x * THREADS + pl.simt.linear_thread_idx()
        stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
        for linear in pl.range(start, x.shape[0] * x.shape[1], stride):
            row = linear // x.shape[1]
            col = linear % x.shape[1]
            out[row, col] = x[row, col]

    @pl.jit()
    def kernel(x: pl.Tensor[[rows, cols], I32],
               out: pl.Tensor[[rows, cols], I32]):
        with pl.section_vector():
            body[THREADS](x, out)
    return kernel
```

Host: initialize `torch_npu`, create `k = make_copy(rows, cols)`, first run `k[None, 1](x_npu, out_npu)`, then test a permitted multi-block count and `k(x_npu, out_npu)` for the automatic budget. Synchronize before exact comparison. Include work sets larger than the grid, plus output sentinels. A file named `kernel.py` uses a first-line module docstring such as `写法：原始。`.

## Launch variants

The anchor's `THREADS=64` is illustrative, not a tuned default or VF register-width requirement. Make varied thread counts explicit factory/cache parameters; `max_threads`, `body[T]`, block offsets and grid strides must agree. Do not mutate a global while retaining an old specialization. After a work-unit or phase change, compare at least two legal block sizes at fixed workers on the new mapping. Worker sweeps do not test this choice; grouped payloads use the [coupled T/G experiment](simt-payload.md#compare-flat-and-grouped-mappings).

Reduction chunk size and thread-block width can be coupled. Count the active thread work created by the live chunks and their payload ownership, then compare it with the worker/thread budget. A wider block may pack several chunks into one worker; too few chunks can leave other workers idle. Smaller chunks expose more independent work but add partial/merge costs. Compare paired chunk/thread configurations at fixed workers on both high-reuse and dispersed-address workloads. Payload width alone, or a launch copied from the correctness anchor, does not identify the fastest configuration.

Inside SIMT, ordinary scalar `if`/`while` and `pl.range(start, stop, step)` express control flow. Fix running-state dtype before a branch or loop and preserve it in every update. Tensor reads retain their dtype while integer literals and shape arithmetic can widen it; inconsistent branch types may misleadingly report a variable as “potentially undefined” ([diagnosis and repair](../api/simt.md#12-when-an-existing-variable-reports-potentially-undefined-check-branch-types)). Use `pl.simt.cast` according to the required value range. These basics do not require another example lookup.

If threads must exchange values, use the shared-UB route in [simt_launch](../../examples/basic_api/simt/simt_launch/). Read its synchronization example only for that need. `syncthreads` must be reached uniformly; the local checker rejects runtime control-flow placements. Atomic operations and fences have different contracts, so select their finding instead of treating them as interchangeable barriers.

For an operation with initialization followed by conflicting updates, [simt_scatter_add](../../examples/algorithms/simt_scatter_add/) demonstrates two native kernels ordered on one Stream, empty-input dispatch and repeated-call tests. Atomicity of each update is not a global initialization barrier.
