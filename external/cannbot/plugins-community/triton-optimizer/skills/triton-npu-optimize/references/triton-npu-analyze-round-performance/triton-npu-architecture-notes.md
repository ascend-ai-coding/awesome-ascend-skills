# Ascend NPU Architecture Notes

Use this reference only after profiling and IR have already exposed a likely bottleneck.

The purpose is simple:

- confirm whether the current chip changes the value of a candidate optimization point
- avoid recommending an A5-friendly idea on A3, or vice versa

## When To Read This

Read this note when the likely issue involves:

- tiling
- layout conversion
- cube to vector handoff
- feature availability that may differ across chips

## First Rule

Confirm the target chip from profiler metadata such as `info.json.0`.

Do not assume an optimization idea learned on A5 automatically applies to A3.

## A3 vs A5: What Usually Matters

### 1. L0C capacity affects tiling

- A3: L0C is around 128 KB
- A5: L0C is around 256 KB

Why this matters:

- A5 can often sustain larger `M x N` tiles than A3
- if A5 shows no cube-side gain over A3, the tiling may not be using the larger buffer well

Use this to refine ideas such as:

- retune tile sizes for the actual chip
- avoid carrying A3-constrained tiling assumptions onto A5

### 2. Layout optimization behavior may differ

- A5 may do better layout optimization
- A3 may leave more repeated layout conversion cost visible

Why this matters:

- on A3, repeated ND/NZ conversion may remain a real optimization target
- on A5, the same issue may be partially mitigated already

Use this when profiling and IR suggest:

- repeated `nd2nz`
- movement-heavy layout bouncing

### 3. Cube result handoff may be more expensive on A3

- A3 behaves more like a memory-based handoff path
- A5 behaves more like a register-friendly handoff path

Why this matters:

- on A3, cube results may pay more writeback or transfer cost before downstream use
- on A5, the same post-cube path may be cheaper

Use this when profiling and IR suggest:

- high `aic_mte3_ratio`
- cube-heavy work followed by vector-side post-processing

Potential refinement:

- reduce post-cube materialization
- fuse downstream work when that avoids extra handoff cost

### 4. Feature support can change whether an idea is realistic

Some optimization points depend on chip-specific support such as:

- layout optimization quality
- newer fixpipe behavior
- staging or copy capabilities

Use this note to rank ideas, not to invent them from nothing.

## How To Use This

Use this order:

1. profiling finds the dominant symptom
2. IR explains why it happens
3. architecture notes refine or re-rank the optimization point

Example:

- Profiling says cube utilization is low and transfer pressure is high
- IR says tiling is conservative and post-processing adds extra movement
- Architecture notes say A5 may reward larger tiles more, while A3 may suffer more from cube-result handoff cost
- Refined optimization point: retune tiling for the actual chip and reduce post-cube materialization, especially on A3

## Final Rule

Architecture notes are a decision modifier.

They refine optimization points that already came from profiling and IR. They do not replace hotspot analysis, profiler diagnosis, or IR attribution.
