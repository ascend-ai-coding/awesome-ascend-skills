# npusim Performance Metrics Reference

Data source: `summary.json` (located at `sim_output/npusim_*/report/summary.json`)

Read and interpret all non-empty sections before drawing conclusions.

---

## Sections Overview

| Section | Purpose | Always present? |
|---------|---------|-----------------|
| `kernel_info` | Kernel duration and hardware configuration | yes |
| `pipe_utilization.pipeline_util_summary` | Per-pipeline utilization as fraction [0, 1] | yes |
| `top_level_diagnosis` | Immediate bottleneck orientation | yes |
| `scalar_instructions` | Register spill detection per subcore | yes |
| `aiv_vector_instructions` | Vector compute efficiency | yes |
| `cache` | Instruction and data cache misses | **may be absent** |
| `bandwidth` | DMA transfer efficiency per transport | **may be absent** |
| `pipeline_overlap` | Parallelism between pipeline pairs | yes |
| `simd_vf_metrics` | Per-VF code-shape metrics (granularity, timing, IPC, overhead, work, stalls, pattern_signals) | yes (empty `{}` when kernel has no SIMD VFs) |

> **Missing-section note**: `cache` and `bandwidth` may be absent from `summary.json`. If a section is absent, just don't factor it into the analysis — skip steps that depend on it and use the fallback signals (`pipe_utilization` + `pipeline_overlap`); each issue page documents the fallback when applicable. The `simd_vf_metrics` block follows a third pattern: always present when the kernel has SIMD VFs (`__simd_vf__` + `asc_vf_call`), empty `{}` otherwise — if `{}`, §6 in `performance-issues-aiv.md` is a no-op.

---

## 1. `kernel_info`

| Field | Description |
|-------|-------------|
| `ai_core_active` | Number of AI cores whose profiling data is available for analysis (see note below) |
| `kernel_total_clocks` | Wall-clock duration of the kernel in ticks — **primary optimization target: minimize** |
| `kernel_instructions_executed` | Total instructions across all cores — secondary optimization signal |

**Optimization targets**:
- `kernel_total_clocks` — the direct measure of kernel latency. Every optimization should be validated against this number. Lower is always better.
- `kernel_instructions_executed` — secondary signal. A decrease confirms redundant work was eliminated. An increase alongside a decrease in clocks is acceptable (e.g., loop unrolling trades more instructions for better pipelining).

> **Note on `ai_core_active`**:
> - `ai_core_active = 1` can mean either (a) the kernel genuinely ran on one core, or (b) profiling data was captured for only one core even if multiple were active. In case (b), `imbalance_ratio` will be 1.0 and `per_core` will not appear, but real multi-core imbalance may still exist.
> - `ai_core_active > 1` but `per_core` arrays contain fewer entries (often just one): the kernel was launched with `blockDim = ai_core_active` cores but the simulator only traced a subset. `imbalance_ratio` is still meaningful (it reflects the launch's actual span), but `per_core` won't let you locate the slow core — use the structural check from `performance-issues-general.md §1` / `§2` (compare `(m × n) / (baseM × baseN)` against `blockDim`).
>
> **Implication for inter-core optimizations**: when `per_core.length < ai_core_active`, cross-core hardware effects (L2 sharing between cores in a scheduling wave, HBM contention) are only partially modelled. Optimizations whose primary benefit is *inter-core* spatial locality (SWAT scheduling, cross-core scale preload) will show a smaller measured win on npusim than on real silicon. Treat single-digit-percent improvements from these optimizations as confirmation that the direction is right, not as the full impact.

---

## 2. `pipe_utilization.pipeline_util_summary`

Per-pipeline activity as a fraction [0, 1] of `kernel_total_clocks`. Only non-zero pipelines appear.

Each entry: `{SUBCORE}_{PIPE}: { mean, min, max }` — always present.

When `ai_core_active > 1`, each entry also contains a `per_core` array:

```json
"AIC_CUBE": {
    "mean": 0.8934,
    "max":  0.9712,
    "min":  0.7201,
    "per_core": [0.9712, 0.9120, 0.7201, 0.8703]
}
```

`per_core[i]` is the utilization of pipeline `{SUBCORE}_{PIPE}` on core `i`. The array length is ≤ `ai_core_active` (typically equals it, but may be shorter when the simulator traces only a subset of cores — see the note on `ai_core_active` above). Use it to identify which specific core is the outlier when `imbalance_ratio > 1.3`.

> **Skip `per_core` unless imbalance is flagged.** When `imbalance_ratio ≤ 1.3`, `mean` / `max` / `min` are sufficient — do not read the (potentially dozens of) `per_core` entries. Estimate per-pipe spread cheaply from `max − min`; only drill into `per_core` to locate the outlier core index once `imbalance_ratio > 1.3` has already justified it.

### Subcore types

| Subcore | Pipelines | Role |
|---------|-----------|------|
| `AIC` | `SCALAR`, `CUBE`, `MTE1`, `MTE2`, `FIXP` | AI Cube core |
| `AIV0`, `AIV1`, … | `SIMD`, `SIMT`, `SCALAR`, `MTE2`, `MTE3` | AI Vector cores |

> **`AIVx_SIMT` appears only when the kernel uses the SIMT programming model** (`Simt::VF_CALL`). It is a **separate pipeline from `AIVx_SIMD`** — SIMT work does **not** show up under `AIVx_SIMD.mean` (which can stay ≈ 0 even while SIMT is the dominant pipe). SIMT execution inflates the shared `aiv_vector_instructions.RVECEX_count` and is measured by the `SIMT_*IPC` fields (see §5).

### Utilization interpretation

| Dominant pipeline | Bound type |
|-------------------|------------|
| `AIC_CUBE` mean > 0.80 | CUBE Bound |
| `AIVx_SIMD` mean > 0.50 | VECTOR Bound |
| `AIVx_SIMT` highest | SIMT Bound (thread-parallel irregular / serial-decomposed workload — see aiv §5) |
| `AIC_MTE2` or `AIVx_MTE2` highest | MTE2 Bound |
| `AIC_MTE1` highest | MTE1 Bound |
| `AIC_FIXP` highest | FIXPIPE Bound |
| `AIC_SCALAR` or `AIVx_SCALAR` significantly above CUBE/VECTOR | SCALAR Bound |

---

## 3. `top_level_diagnosis`

| Field | Description | Thresholds |
|-------|-------------|------------|
| `dominant_pipeline` | Pipeline key with the highest `mean` utilization | — |
| `dominant_pipeline_util` | Its utilization as fraction [0, 1] | — |
| `imbalance_ratio` | `max_core_duration / min_core_duration` | ≤ 1.3 balanced · 1.3–2.0 mild · > 2.0 severe |

`imbalance_ratio > 1.3` signals uneven tiling across cores. Resolve before chasing pipeline bounds.

---

## 4. `scalar_instructions`

One entry per subcore: `AIC`, `AIV0`, `AIV1`, …

| Field | Description | Threshold |
|-------|-------------|-----------|
| `total_count` | Total scalar instructions |  |
| `load_count` | Scalar load instructions |  |
| `store_count` | Scalar store instructions |  |
| `load_store_ratio` | `(load + store) / total` | < 0.30 normal · ≥ 0.30 register spill suspected |

`load_store_ratio ≥ 0.30` is the primary indicator of register spill causing SCALAR Bound. Combine with `cache.dcache_refill_count` to gauge severity.

---

## 5. `aiv_vector_instructions`

Aggregate over all AIV subcores (they typically execute the same code).

| Field | Description | Thresholds |
|-------|-------------|------------|
| `RVECEX_count` | Compute (execute) instructions | — |
| `RVECLD_count` | UB read-back instructions | — |
| `RVECST_count` | UB write-back instructions | — |
| `ub_traffic_ratio` | `(RVECLD + RVECST) / RVECEX` | < 1.0 acceptable · ≥ 1.0 excessive UB traffic → VF RegAPI |
| `SIMD_ScalarIPC` | Scalar IPC within the SIMD part of VF | — |
| `SIMD_ExecIPC` | Compute IPC for SIMD; dual-issue max ≈ 2.0 | < 1.2 poor · 1.2–1.5 moderate · > 1.5 good |
| `SIMD_LdStIPC` | Load/store IPC within SIMD | — |
| `SIMT_ExecIPC` | Compute IPC for the SIMT part of VF | — |
| `SIMT_LdStIPC` | Load/store IPC within SIMT | — |
| `SIMT_BranchIPC` | Branch instruction IPC within SIMT | — |

`ub_traffic_ratio` is the key trigger for VF RegAPI recommendation. `SIMD_ExecIPC` rises after RegAPI is applied.

> **Note on shared instruction counters**: `RVECEX_count`, `RVECLD_count`, and `RVECST_count` are shared counters — the profiler does not split them between SIMD and SIMT. `ub_traffic_ratio` is therefore an aggregate over both sub-units. The IPC fields (`SIMD_ExecIPC`, `SIMT_ExecIPC`, etc.) are tracked independently per sub-unit.

---

## 6. `cache`

| Field | Description | Action threshold |
|-------|-------------|-----------------|
| `icache_refill_count` | Instruction cache miss count | > 0 → consider icache prefetch |
| `icache_refill_ticks` | Ticks lost to icache misses | > 0 → quantify cost |
| `dcache_refill_count` | Data cache misses (register spill reaching GM) | > 0 → HIGH PRIORITY: fix spill root cause |
| `dcache_refill_ticks` | Ticks lost to dcache misses | `null` = counter unavailable |

`dcache_refill_count > 0` means register spill is reaching Global Memory (~hundreds of cycles per miss). This is very expensive and should be fixed before any other optimization.

---

## 7. `bandwidth`

One entry per transport channel. Key name pattern: `{SUBCORE}_{PIPE}_{SRC}_TO_{DST}`.

| Field | Description | Threshold |
|-------|-------------|-----------|
| `total_gbps` | Effective bandwidth over kernel duration | — |
| `bandwidth_utilization` | `total_gbps / hardware_peak_gbps` | < 0.70 when pipeline is Bound → transport is inefficient |
| `avg_transaction_gbps` | Bandwidth per individual DMA request | Very low → small DMA granularity; double buffer may not help |
| `transaction_count` | Number of DMA transactions | `= 1` with anomalous `avg_transaction_gbps` → ignore, not statistically meaningful |

**Hardware peak bandwidth for Ascend 950 (dav-3510)**: ~1600 GB/s HBM. If `bandwidth_utilization` is missing from the summary but `total_gbps` is present, compute it as `total_gbps / 1600`. On-chip transports (L1↔L0, L0C↔UB) have much higher peaks and are typically not the limiting factor — focus on HBM-touching transports (`*_OUT_TO_*` / `*_TO_OUT`).

### Key transport channels

| Key | Data path | Relevant bound |
|-----|-----------|---------------|
| `AIC_MTE2_OUT_TO_L1` | Global Memory → L1 | CUBE Bound / MTE2 Bound (AIC) |
| `AIC_MTE1_L1_TO_L0A` | L1 → L0A (matrix A) | MTE1 Bound |
| `AIC_MTE1_L1_TO_L0B` | L1 → L0B (matrix B) | MTE1 Bound |
| `AIC_FIXPIPE_L0C_TO_L1` | L0C → L1 | FIXPIPE Bound |
| `AIC_FIXPIPE_L0C_TO_OUT` | L0C → Global Memory (direct output, no L1 staging) | FIXPIPE Bound |
| `AIC_FIXPIPE_L0C_TO_UB0` | L0C → UB0 (cross-core feed to AIV) | FIXPIPE Bound |
| `AIC_FIXPIPE_L0C_TO_UB1` | L0C → UB1 (cross-core feed to AIV) | FIXPIPE Bound |
| UB0/UB1 → L1 | UB to L1 transfer (key name varies by pipe) | Present when AIC reads AIV-side UB |
| `AIVx_MTE2_OUT_TO_UB` | Global Memory → UB | VECTOR Bound / MTE2 Bound (AIV) |
| `AIVx_MTE3_UB_TO_OUT` | UB → Global Memory | MTE3 Bound |

Only the active transports appear in `bandwidth`.

---

## 8. `pipeline_overlap`

Fraction of time two pipelines run simultaneously:
`overlap = intersect(A_busy, B_busy) / union(A_busy, B_busy)`

| Threshold | Interpretation |
|-----------|---------------|
| < 0.05 | No overlap — double buffer not working or not enabled |
| 0.05–0.30 | Partial overlap |
| > 0.30 | Effective overlap |
| > 0.60 | Good overlap |

> **`VEC` in overlap keys = the VF-function pipe**, not a fourth instruction type. On the AIV side, SIMD and SIMT instructions only ever execute **inside a VF function** (`Simt::VF_CALL` / vector VF). The profiler measures overlap at VF-function granularity — one `VEC` interval wraps that function's SIMD *or* SIMT instructions — so the overlap key operand is `VEC`, not `SIMD`/`SIMT`. Thus `AIV0_MTE2_vs_AIV0_VEC` = "MTE2 overlapped with VF compute (whichever of SIMD/SIMT ran inside)". The separate `AIVx_SIMD` / `AIVx_SIMT` entries in `pipe_utilization` (§2) still report where instructions executed; `VEC` exists only in `pipeline_overlap`.

### Key pipeline pairs

| Key | Meaning | Low overlap action |
|-----|---------|-------------------|
| `AIC_MTE2_vs_AIC_CUBE` | GM→L1 vs CUBE compute | Double buffer on L1 |
| `AIC_MTE1_vs_AIC_CUBE` | L1→L0 vs CUBE compute | Double buffer on L0A/L0B |
| `AIC_FIXP_vs_AIC_CUBE` | FIXPIPE output vs CUBE compute | Low → CUBE stalling waiting for L0C to drain; increase N-axis tile size |
| `AIC_SCALAR_vs_AIC_CUBE` | Scalar dispatch vs CUBE compute | Low → serial scalar/CUBE (backpressure); High + SCALAR dominant → scalar IS cause |
| `AIV0_MTE2_vs_AIV0_VEC` | GM→UB vs VECTOR compute | Double buffer (check bufNum=2, EnQue/DeQue pairing) |
| `AIV0_MTE3_vs_AIV0_VEC` | UB→GM vs VECTOR compute | Increase ubFactor |
| `AIV0_SCALAR_vs_AIV0_VEC` | Scalar dispatch vs VECTOR compute on AIV0 | Low → scalar overhead serial with vector; check loop scalar overhead |
| `AIV1_SCALAR_vs_AIV0_VEC` | Scalar on AIV1 vs VECTOR on AIV0 | See note below |
| `AIC_CUBE_vs_AIV0_VEC` | Cube vs Vector inter-core | `= 0.0` → Cube/Vector inter-core pipeline not used |

> **Note on `AIV1_SCALAR_vs_AIV0_VEC`**: the key name in `summary.json` crosses cores (AIV1 scalar vs AIV0 VF compute). Treat it analogously to `AIV0_SCALAR_vs_AIV0_VEC` — low overlap indicates the two subcores have serial scalar/VF phases.

> **Note on compute-light (DMA-bound) kernels — no VF compute**: the `VEC` operand is the VF-function pipe (see the note above the key-pairs table). In a kernel whose real work is scalar + DMA (gather/scatter, histogram, index remap) there is **no VF compute at all** — both `AIVx_SIMD.mean` and `AIVx_SIMT.mean` are ≈ 0, so the `VEC` pipe is idle and every `*_vs_VEC` overlap stays near zero **even when double buffering is working correctly**. The overlap that actually matters (MTE2↔MTE3) has **no field** in `summary.json`. When `AIVx_SIMD.mean < 0.05` **AND** `AIVx_SIMT.mean < 0.05`:
> - **Diagnosis** (which pipe dominates) — use `AIVx_MTE2.mean` / `AIVx_MTE3.mean`; ignore the `*_vs_VEC` overlaps.
> - **Verification** of double buffering / store batching — use **`kernel_total_clocks` ↓ only**. The MTE utilizations may stay flat **or even rise** (the same busy time packs into fewer total clocks, and `util = busy / total_clocks`), and the `*_vs_VEC` overlaps stay ~0. Flat/rising MTE utilization and unchanged overlap are **not** signs the fix failed — confirm by the clock drop.
> - **Once the kernel gains VF compute** (e.g. a SIMT rewrite, aiv §5), the `VEC` pipe becomes active and `*_vs_VEC` overlaps become meaningful again.

---

## 9. `simd_vf_metrics`

Per-VF code-shape metrics for kernels that use the SIMD VF (vector function) programming model (`__simd_vf__` + `asc_vf_call<>`). Block is **always present** when the kernel has SIMD VFs; absent (or `{}`) otherwise. Used by aiv §6 (SIMD VF Code-Shape Patterns) in `performance-issues-aiv.md` to detect code-shape patterns (broadcast fragmentation, reduce serial dependency, binary folding) — distinct from pipeline-bound diagnosis in §1–§5.

> **Always-present note**: `simd_vf_metrics` is generated by npusim and is appended after `pipeline_overlap`. If the kernel has no SIMD VF functions, the block is `{}` and all §6 aiv triggers are no-ops.

### Granularity

| Field | Description | Threshold |
|-------|-------------|-----------|
| `vf_distinct_count` | Number of distinct call sites that launch VF functions | info field |
| `vf_total_executions` | Total VF invocations — raw count, affected by multi-core and npusim profiling behavior. Use `vf_executions_per_core` for thresholds. | info field |
| `vf_active_cores` | Number of distinct AIV subcores that executed VF calls — self-consistent with `vf_total_executions` (both from the same profiling data) | info field |
| `vf_executions_per_core` | `vf_total_executions / vf_active_cores` — normalized VF calls per AIV subcore. Self-consistent: both numerator and denominator from the same profiling data, so ratio is stable regardless of how many physical cores are active. | `> 8` → aiv §6.1 secondary (fragmentation); `≤ 8` → aiv §6.2 guard; `(5, 8]` grey zone — do not re-trigger |
| `vf_max_executions_per_call_site` | Max times any single call_site fired | info field — not used in aiv §6.1 trigger |

### Timing

| Field | Description | Threshold |
|-------|-------------|-----------|
| `vf_avg_cycles` | Average VF duration in ticks | drops after aiv §6.1 fold (one VF does more work, total clocks drop more) |
| `vf_max_cycles` | Longest single VF duration | — |
| `vf_min_cycles` | Shortest single VF duration | — |
| `vf_total_cycles` | Sum of all VF durations — denominator for overhead and stall ratios | — |

### IPC

| Field | Description | Threshold |
|-------|-------------|-----------|
| `vf_min_exec_ipc` | Min `ExecIPC` across all VF invocations — worst VF (low → serial dependency, reduce) | `< 0.7` (corroborating aiv §6.2) |

> **Average IPC**: `vf_avg_exec_ipc` is **not** in `simd_vf_metrics` — it is reported as `aiv_vector_instructions.SIMD_ExecIPC` (see §5). Both derive from the same VF profiling data. The primary classifier for latency-bound in aiv §6 (SIMD VF Code-Shape Patterns) is `simd_vf_metrics.vf_stall_ratio` (see Stall below); `SIMD_ExecIPC` is the secondary corroborating metric.

### Overhead

| Field | Description | Threshold |
|-------|-------------|-----------|
| `rvecsu_count` | Aux-scalar instruction count for VF (address arithmetic feeding VF parameters) | — |
| `rvecsu_cycles` | Cycles spent in aux-scalar instructions | — |
| `pushq_count` | `PUSHQ` instruction count (proxy for parameter-buffer pushes) | — |
| `pushq_cycles` | Cycles spent in `PUSHQ` (proxy for PB-push cycles) | — |
| `pushq_overhead_ratio` | `pushq_cycles / vf_total_cycles` — share of VF time spent on PB parameter passing | info field (always > 1.0 in npusim; not discriminating) |
| `rvecsu_overhead_ratio` | `rvecsu_cycles / vf_total_cycles` — share of VF time spent on aux scalar | **`> 0.15` → aiv §6.1 primary trigger** |

### Work-per-VF

| Field | Description | Threshold |
|-------|-------------|-----------|
| `rvecex_per_vf` | `RVECEX_count / vf_total_executions` — execute instructions per VF invocation | `> 4` → long chain (possible aiv §6.3 binary foldable for `reduce_sum_ar`) |
| `rvecex_cycles_per_vf` | Execute cycles per VF | drops after aiv §6.3 binary fold (fewer Add instructions) |

> **Dedup note**: `rvecex_per_vf` may be inflated when `vf_total_executions` is deduplicated by npusim. This does not affect aiv §6.3 because §6.3 requires `rv_vcadd > 0`, which excludes broadcast patterns. For reduce kernels, `rvecex_per_vf` is accurate.

### Pattern signals

Per-instruction-name counts for code-shape pattern detection (reduce / elemwise). Exposed directly in `summary.json` so aiv §6 triggers don't depend on a sibling report file. All fields default to `0` if the instruction name is absent.

| Field | Description | Used by |
|-------|-------------|---------|
| `pattern_signals.rv_vmax` | `Max` vector instruction count | aiv §6.2 (`reduce_max_like`) |
| `pattern_signals.rv_vcmax` | Inside-vector `ReduceMax` count (ar-axis signal) | aiv §6.2 (`reduce_max_like`) |
| `pattern_signals.rv_vadd` | `Add` vector instruction count | aiv §6.2 / §6.3 (`reduce_sum_like`) |
| `pattern_signals.rv_vcadd` | Inside-vector `ReduceSum` count (ar-axis signal) | aiv §6.3 (required) |
| `pattern_signals.rv_vstui` | Unaligned scalar store count (ra-axis signal) | aiv §6.2 (`reduce_sum_like`) |
| `pattern_signals.rv_vst` | `StoreAlign` vector store count — used to distinguish broadcast from reduce: `rv_vst / (rv_vadd + rv_vmax) > 0.5` → broadcast (every result stored, no accumulator); `≤ 0.5` → reduce (accumulator, few stores) | aiv §6.2 (reduce pattern guard: `≤ 0.5` required) |
| `pattern_signals.rv_vmul` | `Mul` vector instruction count | info field (elemwise signal) |
| `pattern_signals.rv_vmul_s` | Scalar-multiply (`Muls`) count | info field (elemwise signal, AXPY) |

### Stall

Stall metrics measure ticks **strictly inside** one VF call where **no vector-pipe instruction** (RVECEX, RVECLP, RVECSU, RVECST, RVECLD) starts executing. For each tick in the VF's execution range, the reporter checks if any vector instruction starts at that tick; if none — stall tick. Non-vector instructions inside VF are ignored — they do not count as "issue". Ticks between VF calls (while main scalar prepares parameters) are **excluded** — those are captured by `rvecsu_overhead_ratio`.

> **Example**: VF spans ticks `[10001, 10526]`. At tick 10050, no vector instruction starts (data-dependency stall). `vf_total_stall_cycles += 1`. If ticks 10050–10068 all have no vector issue — 19 stall ticks accumulated, and when issue resumes at 10069 — `vf_stall_event_count += 1` (one stall run completed).

| Field | Description | Threshold |
|-------|-------------|-----------|
| `vf_stall_event_count` | Number of stall runs (transitions from issue to no-issue) across all VFs — each time a consecutive stall sequence ends, this increments. Orthogonal to `vf_stall_ratio`: measures stall **frequency** (how many times stall started), not **intensity** (how long total stall was) | corroborating: `vf_stall_event_count / vf_total_executions > 10` → real serial chain exists regardless of `vf_stall_ratio` |
| `vf_total_stall_cycles` | Total number of stall ticks across all VFs (each tick where no vector instruction was issued) | — |
| `vf_stall_ratio` | `vf_total_stall_cycles / vf_total_cycles` — share of VF ticks where no vector instruction was issued | **primary trigger for aiv §6.2**: `> 0.20` latency-bound (optimise via multi-accumulator unroll). Grey zone `[0.10, 0.20]` — do not re-apply aiv §6.2 |

> **Starting thresholds note**: `vf_stall_ratio` thresholds `0.10` / `0.20` are starting points. Verify against production shapes before treating them as final.

### Known limitations of single-threshold triggers

`vf_stall_ratio` is a ratio of stall ticks to total VF ticks. It works well for strong serial chains (long serial chains where stall easily exceeds 20% of VF time) but **normalizes away the chain-length signal**: a short chain in a long VF can produce a sub-threshold ratio even though the chain is real and optimization yields a measurable gain.

Practical implications for aiv §6.2 in `performance-issues-aiv.md`:
- **Strong chains (≥ 8 serial steps)**: reliably caught by `vf_stall_ratio > 0.20`. Apply aiv §6.2.
- **Weak chains (3–7 serial steps, e.g. `reduce_*_ar`)**: may fall into the grey zone `[0.10, 0.20]`. The agent should NOT auto-skip these — instead use `vf_stall_event_count` as a corroborating signal: if `vf_stall_event_count / vf_total_executions > 10`, a serial chain exists regardless of `vf_stall_ratio`. For `reduce_sum_like`, also check aiv §6.3 (binary folding) which has an independent trigger based on `rvecex_per_vf` and pattern, not on stall metrics.
- **Post-optimization reduce**: lands below `0.10` (correctly — kernel is optimized). Distinguish via pattern filter, not via stall metrics alone.

This is a **documented limitation**, not a bug: the boolean trigger catches "obvious" cases; weak cases require either (a) a corroborating metric (`vf_stall_event_count`), (b) an independent pattern-based trigger (aiv §6.3 binary folding), or (c) explicit source inspection for short accumulator chains.

---

## Analysis Priority

Apply this order strictly. Fixing a later step before an earlier one usually changes nothing because the upstream problem still hides the metric you'd use to verify.

1. **`imbalance_ratio`** ([general §1](performance-issues-general.md)) — fix load imbalance first; pipeline metrics are unreliable when cores are unbalanced.
2. **Kernel utilization sanity** ([general §2](performance-issues-general.md)) — if `top_level_diagnosis.dominant_pipeline_util < 0.50` **AND** all `pipeline_overlap.*` values are near zero (`< 0.10`) **AND** the compute pipe is idle — for a **Cube** kernel `AIC_CUBE.mean < 0.10`, for a **pure-vector** kernel `AIVx_SIMD.mean` **and** `AIVx_SIMT.mean` both `< 0.05` — the kernel isn't doing enough work to be analysable. `imbalance_ratio` is **not** required (a balanced too-small shape still qualifies). Common cause: `blockDim` too large for the problem shape (`(m × n) / (baseM × baseN) < blockDim`). Fix the launch or the shape; do not chase bound types. Without this check, the later steps will all "fire" on noise.
3. **SIMD VF code-shape check** ([aiv §6](performance-issues-aiv.md)) — if `simd_vf_metrics` block is non-empty (kernel uses `__simd_vf__` + `asc_vf_call`), check §6.1–§6.3 **BEFORE** pipe-bound diagnosis (steps 4+). Code-shape issues (fragmentation, serial dependency) often manifest as SCALAR/MTE2 overhead in pipe utilization — fixing the structure first may eliminate the pipe-bound symptom. If §6 triggers, apply the fix and re-profile before continuing to step 4.
4. **`dcache_refill_count`** ([aic §6.1](performance-issues-aic.md)) — if > 0, register spill is reaching GM; fix before other pipeline work.
5. **Wasted transports** ([aic §5](performance-issues-aic.md)) — scan `bandwidth` for redundant paths (e.g. `L0C_TO_OUT` + `MTE2_OUT_TO_UB` for the same payload). Or, if `bandwidth` is absent, look for `AIVx_MTE2.mean > 0` in a kernel that should have no AIV-side GM input.
6. **`dominant_pipeline` + utilization** — identify the primary bound type using the table in §2 above.
7. **`pipeline_overlap`** for that bound's key pair — determine if parallelism is working (double buffer, inter-core pipeline). When overlap is in `0.30–0.60` and double buffer is already in the code, the next-step optimization is usually a *scheduling* change (e.g. SWAT tile order for MTE2 — see aic §2.2), not more buffers.
8. **`bandwidth` for the bound's transport** — check DMA efficiency if the pipeline is still bound after overlap is good.

After each fix, re-run `npusim record --gen-report` and verify `kernel_total_clocks` decreased. If it didn't, your diagnosis was wrong — back up to the earliest step that still applies.

> **Shape-dependent effect size**: the magnitude of any fix's win on `kernel_total_clocks` is a function of the kernel shape (`m`, `n`, `k`, batch dimensions, loop counts), not just the fix itself. Two shapes where the *same* bottleneck dominates can show very different speedups for the same fix: small shapes amortize startup overhead poorly, large shapes spread per-iteration savings across more iterations. A small measured win (e.g. ~5%) does not invalidate the fix — it may just mean the test shape under-represents production. Re-verify direction (which metric moved which way) rather than magnitude. If a fix's direction is wrong, the diagnosis was wrong; if direction is right but magnitude is small, try a more representative shape before discarding the fix.

> **Optimizations are not always additive — an earlier one can block a later one.** When a fix doesn't improve (or regresses) `kernel_total_clocks`, the cause may not be the fix itself but a **prior** optimization that conflicts with it — and not necessarily the immediately preceding one; it can be any earlier step. To find it: identify what the new fix needs to **remove or restructure** to pay off (a buffer, a data layout/format, a loop or scheduling structure), then trace **which earlier step introduced that thing**. That step is the blocker, even if it sped the kernel up in isolation — a weaker optimization is often **superseded** by a stronger one that targets the same region more fundamentally (e.g. a small algebraic/scalar tweak vs. a full register-resident rewrite). Decision: **revert the conflicting earlier step** and re-derive the later fix from the clean base, rather than stacking the new fix on top of the old structure; then re-profile. This is only practical if each optimization is kept as a **separate, revertible revision** (commit / file) — so do that, and treat "roll back step M, re-apply step N" as a normal move, not a rewrite.
