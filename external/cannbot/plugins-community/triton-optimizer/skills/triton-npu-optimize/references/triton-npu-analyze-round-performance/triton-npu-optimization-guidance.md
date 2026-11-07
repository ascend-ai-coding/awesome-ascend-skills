# Ascend NPU Optimization Guidance

Use this reference when `triton-npu-analyze-round-performance` needs help turning profiling evidence and IR evidence into concrete potential optimization points.

This document is for guiding the agent, not for preserving historical lessons. The goal is to answer:

- what profiling symptom is dominant
- what IR evidence explains that symptom
- what optimization point the evidence supports

## Core Method

Use profiling and IR as two linked analysis paths:

1. use profiling to locate the hotspot and dominant bottleneck
2. use IR to explain why the bottleneck appears in the current lowering
3. infer one or more potential optimization points from both

Do not start from generic optimization ideas. Start from evidence.

## Step 1: Choose The Real Target

Focus on the hottest operator first.

Practical rule:

- if one operator dominates `op_statistic`, optimize that operator before spending effort on secondary operators

Start with:

- `Ratio(%)`
- `Total Time(us)`
- `Avg Time(us)`
- core-type distribution

If the operator is not a meaningful hotspot, do not generate speculative optimization points for it.

## Step 2: Convert Profiling Symptoms Into Bottleneck Hypotheses

Read the profiling data and write down the dominant symptom in one sentence.

Examples:

- scalar time is unexpectedly high for a vector-like operator
- transfer pressure is dominating the round
- cube utilization is too low
- task waits and overlap look weak
- host launch or sync overhead is too high
- parallelism is too low because `Block Dim` is too small
- scheduling overhead may be too high because `Block Dim` is too large for the useful per-block work

Useful first checks:

- dominant ratio fields in `op_summary`
- `Block Dim`
- `Task Wait Time(us)`
- transfer-like hotspots
- timeline gaps in `task_time`
- concurrency clues in `msprof` JSON
- bandwidth, L2, and wait signals in `.bin`
- whether measured transfer or compute time is far above a simple lower-bound estimate
- optional code-mapping outputs when the profiler already suggests a scalar problem that still needs source-level attribution

## Step 3: Map Symptoms To IR Checks And Potential Optimization Points

Once the profiling symptom is clear, inspect IR to explain why it happens.

Do not use IR as a replacement for profiling. Use it to answer “what in the current implementation or lowering is causing this?”

Use the following symptom blocks as a reading guide. Start from the symptom that best matches the profiling data instead of trying to understand every branch at once.

### Symptom A: Scalar ratio is high

How it looks in profiling:

- a vector-like or cube-like operator still spends too much time in scalar work
- `aic_scalar_ratio` or `aiv_scalar_ratio` looks unexpectedly high
- code-mapping outputs show scalar-heavy execution even in regions that should mostly feed vector or transfer hardware

Check in IR:

- excessive branching such as `scf.if`
- heavy boundary handling
- indexing or masking patterns that may block vectorization
- scalar fix-up logic that dominates the steady-state path

Potential optimization points:

- reduce conservative masking
- separate hot steady-state tiles from edge handling
- simplify index math on the hot path
- restructure loops so vector-friendly work stays vectorized

### Symptom B: Transfer or MTE ratio is high

How it looks in profiling:

- transfer pressure dominates the operator
- `aic_mte*` or `aiv_mte*` ratios stay high
- transfer-like hotspots appear too often
- measured movement time is far above a rough moved-bytes / bandwidth lower bound

Check in IR:

- too many `load`, `store`, `copy`, or layout-conversion operations
- repeated `nd2nz` or similar conversion patterns
- tiling that looks too small to amortize movement
- weak reuse of on-chip buffers

Potential optimization points:

- improve tiling for better reuse
- reduce redundant movement or conversion
- fuse adjacent work to keep data on chip longer
- reorganize layout so the kernel does not bounce between formats unnecessarily

Extra interpretation:

- If the data volume is modest enough that the working set should fit comfortably on chip, a very large gap above the transfer lower bound often strengthens the hypothesis that the kernel is issuing too many small transfers or repeating movement unnecessarily.

### Symptom C: Cube utilization is low

How it looks in profiling:

- `aic_mac_ratio` is weaker than expected
- `cube_utilization(%)` stays low
- the operator should be cube-heavy but does not behave that way

Check in IR:

- tile shapes that underfeed cube compute
- extra scalar or vector-side work around the cube path
- lowering that introduces avoidable data feeding delays
- block structure that limits parallel cube work

Potential optimization points:

- retune tiling to fit hardware buffers better
- reduce side work around the cube path
- fuse pre/post work when it helps feed the compute path
- increase useful parallelism instead of only tuning instruction-level details

### Symptom D: Vector wait or pipeline wait is high

How it looks in profiling:

- `Task Wait Time(us)` is high
- `.bin` shows vector wait or poor pipe utilization
- `task_time` or `msprof` JSON suggests weak overlap

Check in IR:

- serial load-compute-store ordering
- synchronization-heavy structure
- weak overlap between producer and consumer stages
- staging decisions that prevent pipeline fill

Potential optimization points:

- restructure the schedule for better overlap
- reduce unnecessary synchronization
- pipeline movement and compute more effectively
- rebalance stages so one path does not starve another

### Symptom E: Host API overhead is high

How it looks in profiling:

- `api_statistic` shows large launch or synchronization cost
- benchmark time looks dominated by host-side work instead of device execution

Check in IR and round context:

- whether the kernel shape or specialization strategy causes too many launches
- whether setup work is repeated unnecessarily
- whether the benchmark is dominated by launch/sync rather than device work

Potential optimization points:

- reduce launch count through batching or fusion
- simplify specialization behavior
- avoid expensive host-side setup on the hot path

### Symptom F: `Block Dim` is too small

How it looks in profiling:

- `Block Dim` is much smaller than the available parallel machine width
- the round looks under-parallelized before any micro-level bottleneck dominates

Check in IR and launch structure:

- whether loop partitioning collapses available parallel work
- whether tile choices or specialization reduce usable blocks
- whether small shape handling is forcing the kernel into low parallel occupancy

Potential optimization points:

- increase block-level parallel decomposition
- separate small-shape and regular-shape paths when needed
- avoid tiling changes that starve available AI cores

### Symptom G: `Block Dim` is too large for the useful work

How it looks in profiling:

- `Block Dim` exceeds the effective hardware width by a large margin for a light vector kernel
- host or scheduling overhead looks high relative to the device work per block
- the kernel launches many tiny blocks even though the operator is not meaningfully increasing useful concurrency

Check in IR and launch structure:

- whether tiling split the work into very small blocks that do not amortize launch and scheduling cost
- whether the kernel is vector-heavy and shape-regular enough that fewer, fuller blocks would likely be better
- whether shape-specialized edge handling accidentally multiplies block count

Potential optimization points:

- reduce over-partitioning by increasing per-block useful work
- retune tiling so the block count matches the operator's effective parallel width more closely
- split small-shape and regular-shape behavior if one path is forcing excess blocks for the other

### Symptom H: `.bin` shows poor L2 or suspicious memory paths

How it looks in profiling:

- L2 hit ratio is weak
- memory path or bandwidth signals look imbalanced
- block or lane behavior looks uneven

Check in IR and movement structure:

- repeated data reloads without reuse
- layout or staging decisions that hurt locality
- imbalance across blocks or lanes

Potential optimization points:

- increase on-chip reuse
- tune tile shape for locality
- reduce unnecessary movement across memory hierarchy boundaries
- rebalance work distribution

## Step 4: Use Architecture Constraints To Rank The Ideas

Not every theoretically valid optimization point is equally useful on the target chip.

Before recommending a change, ask:

- does the chip reward larger tiles because buffer capacity allows it?
- is the problem really a layout issue that the architecture can or cannot optimize away?
- is the current parallelism limit shape-driven or implementation-driven?
- is the measured gap from the rough lower bound large enough to justify structural changes, or is the operator already near a hardware-facing limit?

Use architecture knowledge to rank ideas, not to replace evidence.

## Step 5: Write Potential Optimization Points

For each suggested optimization point, tie it back to both kinds of evidence:

1. profiling symptom
2. IR explanation
3. likely implementation problem
4. candidate optimization point

Good example shape:

- Observed: vector-like operator but scalar ratio is high
- IR explanation: hot path still contains masking and branch-heavy edge handling
- Implementation issue: vectorization-friendly structure is degraded on the steady-state path
- Potential optimization point: split steady-state tiles from edge tiles and reduce masking on the hot path

### Suggested Output Template

Use this shape when writing candidate ideas:

- Observed profiling symptom:
  vector-like operator but scalar ratio is high
- IR explanation:
  masking and branch-heavy edge handling remain on the hot path
- Likely implementation issue:
  steady-state vectorization is degraded
- Potential optimization point:
  split steady-state and edge tiles; reduce masking on the hot path

## Cross-Round Comparison Guidance

When comparing rounds, start small:

- did runtime improve
- did the dominant bad ratio improve
- did parallelism or utilization improve

If a round becomes slower, consider whether the change caused:

- worse tiling
- worse `Block Dim`
- more data movement
- degraded vectorization

Do not assume the optimization idea was wrong before checking whether the new lowering shape became worse.

## Final Rule

The output of this analysis should not be “memory-bound” or “scalar ratio is high.”

The output should be a small set of plausible optimization points that are explicitly grounded in:

- profiling evidence
- IR evidence
- current operator implementation structure
