---
priority: high
---

# Grid Flattening And UB Buffering Pattern

## Summary

Flatten logical work items onto physical cores and batch small row-wise memory transfers into wider UB stores to reduce launch overhead and improve per-core work density.

## Use When

- The logical grid is much larger than the physical AICore or VectorCore count.
- Work is partitioned by batch or sequence buckets with visible load imbalance.
- Each program processes many tiny rows after grid-to-physical-core mapping.
- Gather-like code has continuous destination rows but still stores one row at a time.
- Scatter-weight-gradient-like code has repeated row loads that can be batched from continuous source rows.

## Repairs

### Flatten logical tasks

Treat all independent logical work items as one continuous stream and split that stream evenly across physical cores:

```python
pid = tl.program_id(0)
task_start = pid * TASKS_PER_CORE
offs = task_start + tl.arange(0, TASKS_PER_CORE)
mask = offs < TOTAL_TASKS
```

Compute `TASKS_PER_CORE` on the host and pass it as `tl.constexpr` when it controls tensor shapes or loop bounds. Prefer uniform per-core loop structure and masks over early returns that create uneven control flow.

### Map logical grid to physical grid

When `logical_tasks > num_cores`, launch `num_cores` programs and loop over logical tasks inside the kernel. Keep `NUM_TASKS` and `NUM_CORES` as explicit constants so the compiler can simplify loop bounds.

This helps only when each physical program has enough work to amortize the internal loop. If the original grid is near the physical core count, the extra loop can be overhead.

### Collapse a lightweight batch/head grid axis into the program

For chunked attention or recurrence kernels, a grid like `(NT, B * HV)` can create many small programs when each `(batch, head)` lane does modest work. If the per-lane work is independent and writes disjoint output regions, consider launching only over chunks and looping over `B * HV` inside each program:

```python
pid_t = tl.program_id(0)

for i_bh in range(BH):
    i_b = i_bh // HV
    i_h = i_bh % HV
    # Process one batch/head lane for this chunk.
```

Wrapper launch:

```python
kernel[(NT,)](..., BH=B * HV)
```

instead of:

```python
kernel[(NT, B * HV)](...)
```

This is a work-density variant of grid flattening. It is not a universal replacement for parallel batch/head execution. Use it when the original per-head programs are too thin and `BH` is small enough that serializing the axis does not underutilize the device.

### Discover core counts and choose grid by task kind

Use a best-effort runtime query before hardcoding one grid size:

```python
import torch

print(torch.npu.device_count())
device = torch.npu.current_device()
props = torch.npu.get_device_properties(device)
print(props)
```

If this query fails, if `torch.npu` is unavailable in the current environment, or if `props` does not expose explicit per-engine counts, fall back to:

- cube cores: `24`
- vector cores: `48`

Do not turn one observed physical-core count into a universal launch rule. Pick the initial flattened grid from the operator task kind:

- `cube`-like operators: start from the discovered Cube-core count and verify that the kernel really stays Cube-dominant.
- `vector`-like operators: start from the discovered Vector-core count and retest after any major tiling or buffering rewrite.
- `mix` operators: choose the starting point from the dominant bottleneck side, or test both cube-first and vector-first launch counts when the profile is ambiguous.

Keep `NUM_CORES` aligned with the chosen task kind rather than hardcoding one global constant across unrelated operators.

### UB aggregate writes

After physical-core mapping, if each program produces several rows with continuous destination addresses, stage a small group of rows in UB and issue a wider 2D store. Choose `SUB_BLOCK_SIZE` so the staged rows plus live compute tensors fit UB.

Do not use aggregate writes for genuinely scattered destinations.

### UB bulk reads

For row-wise reductions or gradient accumulation where several consecutive source rows are loaded one at a time, load a small 2D slab into UB and select rows from the slab. Keep a fallback path if row indices are not continuous enough for a bulk read to be correct.

## Dependencies

- Apply `grid-flatten-and-ub-buffering` only after the kernel's basic access semantics are clear.
- UB aggregate write and UB bulk read usually depend on a prior grid-to-physical-core or flattening rewrite that gives each program multiple rows.
- Combine with `autotune` only after the structural rewrite is correct; tune `TASKS_PER_CORE`, `BLOCK`, and `SUB_BLOCK_SIZE` separately enough to explain the result.

## Risks

- Flattening can erase useful multidimensional locality if offsets are decoded poorly.
- Physical-grid loops can hurt small workloads.
- Serializing a batch/head axis can under-parallelize large `B * HV` or Cube-heavy workloads.
- Large compile-time `BH` values can increase code size or make one program too long.
- UB staging can exceed UB capacity or reduce occupancy.
- Gather and scatter have different address-continuity requirements; do not reuse aggregate-write logic for scattered stores.

## What To Verify After Applying

- Record the physical core assumption and how it is discovered or passed.
- Validate edge cases where `TOTAL_TASKS` is not divisible by `NUM_CORES`.
- Benchmark small and large shapes if the operator supports both.
- If the win depends on row continuity, add that condition to the round summary.
- For batch/head-in-program loops, test several `B * HV` regimes and verify outputs are disjoint across the serial loop.

## Related Patterns

- Complements `program-multiple-rows`: that pattern widens row-wise work inside a program, while this pattern flattens logical work onto physical cores and batches memory movement inside each core.
- Combine with `autotune` only after the structural rewrite is correct; tune `TASKS_PER_CORE`, `BLOCK`, and `SUB_BLOCK_SIZE` with enough separation to explain the result.
