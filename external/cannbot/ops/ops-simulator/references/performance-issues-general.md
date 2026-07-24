# General Performance Issues

Issues applicable to all operator types regardless of dominant pipeline. Check these **before** drilling into `performance-issues-aic.md` / `-aiv.md` — imbalance distorts every other metric.

> All detection metrics here come from sections that are always present in `summary.json` (no `bandwidth` / `cache` dependency), so this file works regardless of whether `bandwidth`/`cache` are present.

---

## Quick Reference Table

| Issue | Detection metric | Threshold | Fix | Section |
|-------|------------------|-----------|-----|---------|
| Single-core / under-launched kernel | `kernel_info.ai_core_active` | `== 1` on a multi-core chip — `imbalance_ratio` does **not** catch this | Launch all cores (priority over every pipe-bound rule) | §1.1 |
| Multi-Core Load Imbalance — block dim or remainder | `top_level_diagnosis.imbalance_ratio` | `> 1.3` (mild), `> 2.0` (severe) | Rebalance tiling across cores | §1.1, §1.2, §1.3 |
| Multi-Core Load Imbalance — tail round | `imbalance_ratio` slightly > 1.0 AND `0 <` `tail_tiles` (`total_tiles % blockDim`) `< blockDim/2` | tail round leaves cores idle | Split tail tiles in N (e.g. `baseN/2`) | §1.4 |
| Kernel Underutilization | `AIC_CUBE.mean` + overlaps + tiles/core | `AIC_CUBE.mean < 0.10` AND all overlaps `< 0.10` AND tiles/core `< 2` (imbalance optional) | Shrink blockDim, grow problem, or shrink baseM/baseN | §2 |

---

## 1. Multi-Core Load Imbalance

### Problem Description

Cores have uneven work distribution: some cores finish early and sit idle while others are still running. This inflates `kernel_total_clocks` without any pipeline being the actual bottleneck.

### Detection Criteria

| Metric | Path in summary.json | Threshold | Description |
|--------|----------------------|-----------|-------------|
| `imbalance_ratio` | `top_level_diagnosis.imbalance_ratio` | > 1.3 mild · > 2.0 severe | `max_core_duration / min_core_duration` |

### Locating the slow core (when `ai_core_active > 1`)

`imbalance_ratio` tells you imbalance exists. `per_core` in `pipe_utilization.pipeline_util_summary` tells you exactly which core and which pipeline are responsible.

**Step 1** — find the pipeline with the largest spread: compare `max - min` across all pipeline entries.

**Step 2** — inspect `per_core` for that pipeline to find the outlier core index:

```json
"AIC_CUBE": {
    "mean": 0.8934, "max": 0.9712, "min": 0.7201,
    "per_core": [0.9712, 0.9120, 0.7201, 0.8703]
}
```

Here core 2 (index 2, value 0.7201) is the slowest — it finishes CUBE work much earlier than core 0, meaning it received less work.

**Step 3** — use the core index to inspect the same index in other pipelines (e.g., `AIC_MTE2.per_core[2]`) to confirm whether the underload is uniform or pipeline-specific.

### Fix Method

#### §1.1 Block Dim Too Small

**Problem**: Kernel launches fewer cores than the chip has available — idle cores are never used. The extreme case is a single-core baseline (`blockDim = 1`).

**Detection**: `kernel_info.ai_core_active == 1` (or far below the chip's core count) on a many-core chip. This is the dominant bottleneck and **takes priority over every pipeline-bound rule** — launching all cores is the first fix, not double buffering or `ubFactor`.

> **`imbalance_ratio` does not catch this.** When only one core runs (or the simulator traces a single core), `imbalance_ratio` is `1.0` and `per_core` has one entry — the profile looks "balanced". Do not infer balance from `imbalance_ratio = 1.0` when `ai_core_active == 1`; read `ai_core_active` **first**. A single-core kernel also shows every `pipeline_overlap.*` near zero and no pipe saturated (`dominant_pipeline_util < 0.50`) — symptoms that mimic a double-buffer problem and will route an unwary agent into MTE2/MTE3 Bound. Resist that: parallelize across cores first, then re-profile and re-diagnose the *new* dominant pipe.

**Fix**: never hardcode the core count — **query the platform and launch on every core**. The count is chip-dependent (e.g. Ascend 950: 32 Cube / 64 Vector cores), so a hardcoded number like `8` leaves most of the chip idle. Pick the query by kernel type: a **vector** kernel (dominant pipe `AIVx_*` — elementwise, gather, norm, sort) uses `GetCoreNumAiv()`; a **cube** kernel (dominant pipe `AIC_*` — matmul) uses `GetCoreNumAic()`. The tiling must split work across the **same** `blockDim`, or the extra cores still sit idle.

```cpp
// ❌ Single core — 1 of N used
rms_norm_quant<<<1, 0, stream>>>(...);

// ❌ Hardcoded core count — uses only 8, the rest of the chip's cores sit idle
constexpr uint32_t blockDim = 8;                      // never hardcode this
rms_norm_quant<<<blockDim, 0, stream>>>(...);

// ✅ Vector (AIV) kernel — query AIV core count, use all of them
uint32_t blockDim = ascendcPlatform->GetCoreNumAiv();  // 64 on Ascend 950
rms_norm_quant<<<blockDim, 0, stream>>>(...);

// ✅ Cube (AIC) kernel — query AIC core count
uint32_t blockDim = ascendcPlatform->GetCoreNumAic();  // 32 on Ascend 950
matmul<<<blockDim, 0, stream>>>(...);
```

The same `blockDim` must reach the tiling so each core gets its slice (e.g. `tiling.set_coreNum(blockDim)` / split `totalElements` across `blockDim`). Hardcoding the core count in the tiling has the same effect as hardcoding it at launch — query it once and thread it through.

> **After the fix, re-profile and re-diagnose**: the kernel total clocks should drop roughly proportionally to the core count, but the dominant pipe and per-pipe utilizations can shift because per-core work is now smaller. Don't assume the pre-fix bound type still applies — read the new profile from the top.

#### §1.2 Remainder Elements Create Imbalance

**Problem**: When the total element count is not evenly divisible by `blockDim`, the last few cores process a different (often larger) amount of data.

**Fix**: Distribute the remainder evenly — first `N % blockDim` cores each process `ceil(N / blockDim)` elements, the rest process `floor(N / blockDim)`.

```cpp
// ❌ Last core may handle very different workload
int tileLen = totalLen / blockDim;  // integer division drops remainder

// ✅ Distribute remainder across first few cores
int bigTile = (totalLen + blockDim - 1) / blockDim;  // ceil
int smallTile = totalLen / blockDim;
int bigCoreCount = totalLen % blockDim;
int myTile = (blockIdx < bigCoreCount) ? bigTile : smallTile;
```

#### §1.3 Non-uniform Data Structure

**Problem**: Input has rows/columns of varying length (e.g., variable-length sequences), causing some cores to process much more data.

**Fix**: Sort inputs by size and assign to cores to equalize total work per core, or use dynamic scheduling.

#### §1.4 Tail-Round Imbalance

**Problem**: Even when `total_tiles` is well above `blockDim` (so per-core averages look fine), the **last scheduling round** can have fewer tiles than `blockDim`. The first `tail_tiles` cores get one more tile; the remaining `blockDim - tail_tiles` cores sit idle through the entire last round. `imbalance_ratio` may only show a mild bump (e.g. 1.05–1.15), but `kernel_total_clocks` carries the extra half-empty round.

**Detection**:

```
total_tiles  = CeilDiv(m, baseM) * CeilDiv(n, baseN)
rounds       = CeilDiv(total_tiles, blockDim)
tail_tiles   = total_tiles % blockDim
```

Apply this fix when `0 < tail_tiles < blockDim / 2` **AND** `rounds ≥ 2`. (If `rounds == 1`, route to §2 Kernel Underutilization instead — the whole kernel is one tail round.)

**Fix**: in the `BlockScheduler`, detect the tail-round case and emit smaller tile shapes for those specific tiles so the count of tail tiles ≥ `blockDim`. Earlier rounds keep the full `baseM × baseN` shape unchanged.

```cpp
// ❌ Before — every tile is baseM × baseN, tail round leaves cores idle
auto tileShape = AscendC::Te::MakeShape(baseM, baseN);

// ✅ After — split tail tiles in N so tail_tiles doubles
bool inTailRound = (tileIdx >= (rounds - 1) * blockDim);
auto tileN = inTailRound ? (baseN / 2) : baseN;
auto tileShape = AscendC::Te::MakeShape(baseM, tileN);
```

**Verification after the fix**: `imbalance_ratio` should drop closer to 1.0.

**When NOT to apply**:
- `tail_tiles == 0` — last round is full, no idle cores
- Smaller base tiles would push below MMAD efficiency limits

**Related Skills**:

📖 Tiling partition strategy: [ascendc-tiling-design](https://gitcode.com/cann/cannbot-skills/blob/master/ops/ascendc-tiling-design/SKILL.md)

---

## 2. Kernel Underutilization

### Problem Description

The kernel barely does any compute work. CUBE (and / or SIMD) is mostly idle, no pipeline overlap exists anywhere; `imbalance_ratio` is often high but **not required** (a balanced too-small shape qualifies too — see Step 1.5). The usual cause is a mismatch between problem size and launch parameters: too many cores launched for the work, OR `baseM × baseN` doesn't fit into `m × n` enough times to give each core ≥ 2 tiles.

### Detection Criteria

| Metric | Path | Condition |
|--------|------|-----------|
| CUBE almost idle | `pipe_utilization.pipeline_util_summary.AIC_CUBE.mean` | `< 0.10` |
| Imbalance (optional, not required) | `top_level_diagnosis.imbalance_ratio` | often `> 1.5`, but a balanced too-small shape also qualifies |
| All overlaps near zero | every `pipeline_overlap.AIC_*_vs_AIC_CUBE` | `< 0.10` |
| Scalar dispatch dominant but no spill | `scalar_instructions.AIC.load_store_ratio` | `< 0.30` AND `AIC_SCALAR.mean` is the dominant pipe |
| Tiles per core | `CeilDiv(m, baseM) * CeilDiv(n, baseN) / blockDim` (compute from kernel params) | `< 2` |

This pattern superficially looks like SCALAR Bound (dominant pipe is `AIC_SCALAR`), but `load_store_ratio < 0.30` excludes register spill. [aic §6.3](performance-issues-aic.md) routes back here when CUBE util is under 0.40 for the same reason.

### Fix Method

Pick one of these — they're alternatives, not a sequence:

#### §2.1 Shrink blockDim to match the work

```cpp
// ❌ Before — always use every available AIC, regardless of problem
uint32_t blockDim = ascendcPlatform->GetCoreNumAic();

// ✅ After — clamp to what the shape can actually fill (≥ 2 tiles per core)
uint32_t maxCores = ascendcPlatform->GetCoreNumAic();
uint32_t tiles    = CeilDiv(m, BASE_M) * CeilDiv(n, BASE_N);
uint32_t blockDim = std::min(maxCores, std::max<uint32_t>(1u, tiles / 2));
```

#### §2.2 Use a representative shape for tuning

If the small shape was only chosen for the simulator run, the profile says nothing useful about the optimization. Re-run with a shape that produces `tiles_per_core ≥ 2` and re-profile.

#### §2.3 Shrink baseM / baseN to manufacture more tiles

If both the shape and the core count are fixed external constraints (production scenario), reduce `baseM`/`baseN` so the existing problem yields more tiles. Trade-off: smaller base tiles also reduce MMAD efficiency per tile.

### Pitfalls

- **Do not "fix" CUBE Bound or SCALAR Bound first.** The §1.x CUBE and §6.x scalar rules will all "trigger" on this profile, but their fixes (double buffer, spill cleanup, inter-core staggering) cannot change anything — the kernel doesn't have enough iterations for them to act on.
- **Don't trust per-core data on this profile.** With `tiles_per_core < 1`, the `per_core` array reflects the single core that got the only tile, not a meaningful average across the launch.
