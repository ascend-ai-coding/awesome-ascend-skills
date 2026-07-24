# Cube Core (AIC) Performance Issues

AIC-side pipeline bound diagnosis and fix solutions. All metric paths refer to fields in `summary.json`.

> **Reading order**: scan the Quick Reference Table for a row matching your symptoms, then jump to the section number in the rightmost column. Section numbers `§N.M` are greppable verbatim across this file.

---

## Quick Reference Table

| Issue | Detection metric | Threshold | Fix | Section |
|-------|------------------|-----------|-----|---------|
| CUBE Bound — no Cube/Vector inter-core pipeline | `pipeline_overlap.AIC_CUBE_vs_AIV0_VEC` | `= 0.0` | Stagger mm/vec across AIC and AIV | §1.1 |
| CUBE Bound — MTE1 not overlapping | `pipeline_overlap.AIC_MTE1_vs_AIC_CUBE` | `< 0.30` | Double buffer L0A / L0B | §1.2 |
| CUBE Bound — MTE2 not overlapping | `pipeline_overlap.AIC_MTE2_vs_AIC_CUBE` | `< 0.30` | Double buffer L1 input | §1.3 |
| MTE2 Bound (AIC) — inefficient transport | `bandwidth.AIC_MTE2_OUT_TO_L1.bandwidth_utilization` | `< 0.70` | UB batching or NZ format | §2.1 |
| MTE2 Bound (AIC) — small DMA granularity | `bandwidth.AIC_MTE2_OUT_TO_L1.avg_transaction_gbps` | very low | Scale preload, UB batching | §2.1 |
| MTE2 Bound (AIC) — quant Scale reloaded per K-chunk (fallback) | `kernel_instructions_executed` / `scalar_instructions.AIC.total_count` inflated vs `AIC_CUBE.mean` | MTE2 dominant AND Scale re-loaded inside K-loop | Coalesce Scale across full K (`scaleKL1 = K`) | §2.1 |
| MTE2 Bound (AIC) — partial overlap, pingpong already on | `pipeline_overlap.AIC_MTE2_vs_AIC_CUBE` AND L1 pingpong present in code | overlap `0.30–0.60` | SWAT tile scheduling (M-window + N-zigzag) | §2.2 |
| MTE2 Bound (AIC) — redundant A re-transfer (small A, huge N) | `AIC_MTE2` dominant AND `baseM/baseN` non-trivial AND A re-loaded per N-tile | structural; verify on msprof (not cannsim) | Keep A L1-resident (A full-load) | §2.3 |
| MTE1 Bound — L0 not prefetched | `pipeline_overlap.AIC_MTE1_vs_AIC_CUBE` | `< 0.30` | Double buffer L0A / L0B | §3 |
| FIXPIPE Bound — small N tile | `pipeline_overlap.AIC_FIXP_vs_AIC_CUBE` AND `baseN` small | `< 0.30` AND `baseN < 16` | Increase N-axis tile size | §4.1 |
| FIXPIPE Bound — serial drain, L0C DB not feasible | `pipeline_overlap.AIC_FIXP_vs_AIC_CUBE` AND no L0C double-buffer | `< 0.30` AND single L0C buffer | UnitFlag (`mmadParams.unitFlag`) | §4.2 |
| Cube→Vector GM round-trip | `AIC_FIXPIPE_L0C_TO_OUT` AND `AIVx_MTE2_OUT_TO_UB` carry same payload | tile streams through GM twice | Fixpipe L0C→UB direct | §5 |
| SCALAR Bound — register spill | `scalar_instructions.AIC.load_store_ratio` | `≥ 0.30` | Local vars, drop multi-level deref | §6.1 |
| SCALAR Bound — icache miss | `cache.icache_refill_ticks` | `> 0` | Icache prefetch, reduce code size | §6.2 |
| SCALAR Bound — backpressure from CUBE | `pipeline_overlap.AIC_SCALAR_vs_AIC_CUBE` | low (`< 0.20`) AND `load_store_ratio < 0.30` | Treat as CUBE Bound | §6.3 |

> **Missing-section note**: if your `summary.json` lacks `bandwidth` / `cache` sections, the table rows keyed on those will not trigger by themselves. §2, §5 and §6.2 each include a fallback signal that uses only `pipe_utilization` + `pipeline_overlap`. The other rows already use only always-present sections.

---

## 1. CUBE Bound

### Problem Description

`AIC_CUBE` utilization is the highest (mean > 0.80). The matrix multiply unit is the bottleneck. Optimization goal: maximize CUBE utilization while overlapping data movement.

### Detection Criteria

| Metric | Path | Condition |
|--------|------|-----------|
| CUBE utilization | `pipe_utilization.pipeline_util_summary.AIC_CUBE.mean` | > 0.80 and dominant |
| Cube/Vector inter-core overlap | `pipeline_overlap.AIC_CUBE_vs_AIV0_VEC` | = 0.0 → inter-core pipeline unused |
| MTE1 vs CUBE overlap | `pipeline_overlap.AIC_MTE1_vs_AIC_CUBE` | < 0.30 → L1→L0 stalls CUBE |
| MTE2 vs CUBE overlap | `pipeline_overlap.AIC_MTE2_vs_AIC_CUBE` | < 0.30 → GM→L1 stalls CUBE |

### Fix Method

#### §1.1 Cube/Vector inter-core pipeline not used (`AIC_CUBE_vs_AIV0_VEC == 0.0`)

AIC (CUBE) and AIV (SIMD) cores are running sequentially: the kernel finishes all CUBE work on AIC, then does all VECTOR work on AIV (or vice versa). The inter-core pipeline keeps both active simultaneously.

**Fix**: Implement the staggered pipeline pattern — split the work into stages and alternate:
```
Time:  |--mm1--|--mm2--|--mm3--|
       |       |--v1---|--v2---|
AIC:   CUBE    CUBE    CUBE
AIV:   idle    SIMD    SIMD
```

The kernel structure uses two passes (mm+vec pairs), where AIC starts mm2 while AIV runs vec1:

```cpp
// Pseudo-structure of AIC/AIV interleaved pipeline
// AIC side:
MmadPipe.push(tile0);       // mm1
MmadPipe.push(tile1);       // mm2 (while AIV runs vec1)
// AIV side:
VecPipe.pop(result0);       // vec1 (while AIC runs mm2)
VecPipe.pop(result1);       // vec2
```

#### §1.2 MTE1 not overlapping with CUBE (`AIC_MTE1_vs_AIC_CUBE < 0.30`)

L1 → L0A/L0B transfers are not hidden behind CUBE compute. The CUBE unit stalls waiting for data.

**Fix**: Enable double buffering on L0A and L0B. While CUBE computes on L0A[ping], MTE1 loads L0A[pong] for the next tile.

```cpp
// InitBuffer with 2 copies for L0A and L0B
pipe.InitBuffer(l0aQue, 2, l0aBytes);
pipe.InitBuffer(l0bQue, 2, l0bBytes);
```

> **Apply ONE buffer level per edit, in this order — never all at once.** AIC ping-pong is hand-managed `set_flag/wait_flag` + EVENT_ID + `%2`; read [`ascendc-performance-best-practices`](../../ascendc-performance-best-practices/reference/matmul/pingpong_design.md) (co-located in `ops/`) for the protocol, but its example is the *finished* kernel — **do not transcribe L1 + L0 + scale + UnitFlag in one pass**, that collides EVENT_IDs and produces `execute_set_flag already has same set_flag` (deadlock). Stage it: **(1) L1 ping-pong only → compile + precision-check + re-profile; (2) then L0; (3) every further optimization (scale double-buffer, UnitFlag, anything else) is its own separate edit + verify**. Each level: its own EVENT_IDs, exactly one `wait` per `set` before the next `set` on that id, and don't pre-set an id the loop re-sets. Adding everything in one edit **hangs** (unbalanced/duplicate `set`); reusing a buffer before MMAD consumed it or `cmatrixInitVal` not `k==0` **breaks precision**.
>
> **L0-scale offset trap (recurring precision bug):** A/B **and their MX scale** must share one parity offset — `HALF_L0_SIZE * (idx & 1)` — scale rides the same half as its operand. Splitting scale into separate per-parity half-regions corrupts the MX scale (precision drifts). L1 single-L0 works fine; the bug appears exactly when L0 is doubled.

#### §1.3 MTE2 not overlapping with CUBE (`AIC_MTE2_vs_AIC_CUBE < 0.30`)

GM → L1 data loading is not overlapping with CUBE. CUBE stalls waiting for L1 to be refilled.

**Fix**: Enable double buffering on L1 input buffers. While CUBE processes L1[ping], MTE2 loads L1[pong]. Same caveat as §1.2 — the L1 ping-pong is a hand-managed `set_flag/wait_flag` handshake; implement it from [`ascendc-performance-best-practices`](../../ascendc-performance-best-practices/reference/matmul/pingpong_design.md), don't just raise `L1_BUFFER_NUM`. **One optimization per edit**: get L1 ping-pong compiling + precision-clean + re-profiled before adding L0 or any other change — never stack multiple buffer levels/optimizations in one pass (collides EVENT_IDs → `same set_flag` deadlock).

> **Decision check before applying this fix**: grep the kernel source for `L1_BUFFER_NUM`, `PINGPONG_NUM`, or any explicit `InitBuffer(buf, 2, …)` on L1 tensors. If L1 pingpong is **already enabled** but overlap is still `< 0.30`, the root cause is not the buffer count — it is the multi-core tile scheduling pattern dispersing tiles across M-N in a way that defeats L1 reuse. Skip to **§2.2**.

---

## 2. MTE2 Bound (AIC side)

### Problem Description

`AIC_MTE2` (Global Memory → L1) has the highest utilization. Data loading into L1 is the bottleneck for the CUBE pipeline.

### Detection Criteria

**Primary signal** (full summary):

| Metric | Path | Condition |
|--------|------|-----------|
| MTE2 utilization | `pipe_utilization.pipeline_util_summary.AIC_MTE2.mean` | Dominant |
| Bandwidth utilization | `bandwidth.AIC_MTE2_OUT_TO_L1.bandwidth_utilization` | `< 0.70` → transport inefficient |
| Avg transaction bw | `bandwidth.AIC_MTE2_OUT_TO_L1.avg_transaction_gbps` | very low → small DMA granularity |

**Fallback signal** (no `bandwidth` section):

| Metric | Path | Condition |
|--------|------|-----------|
| MTE2 utilization | `pipe_utilization.pipeline_util_summary.AIC_MTE2.mean` | Dominant |
| MTE2 vs CUBE overlap | `pipeline_overlap.AIC_MTE2_vs_AIC_CUBE` | `< 0.30` → double-buffer issue (see §1.3) |

Without `bandwidth`, distinguish §1.3 (no overlap) from §2 (overlap exists but transport itself is slow) is hard — start by fixing the overlap, then re-profile.

### Fix Method

#### §2.1 Small DMA granularity (`avg_transaction_gbps` very low)

Each DMA request transfers too little data. HBM startup latency dominates. Adding more buffers will not help.

**Fix A — UB batching**: Accumulate multiple tiles in L1 per DMA call.

**Fix B — Scale preload / coalescing**: In quantized matmul (MXFP4/MXFP8) the per-block Scale tensor is loaded per K-chunk (`baseK`), so each Scale DMA is tiny (often `< 20 KB`) and is re-issued once per K-iteration per tile. With a large K this becomes many small Scale transfers. Coalesce them: load Scale across the **whole K** (`scaleKL1 = K`) into L1 once, then reuse it across the K-loop — collapsing N small Scale DMAs into one large transfer.

```cpp
// ❌ Scale loaded per baseK chunk — many small DMAs
for (int kIdx = 0; kIdx < K / baseK; ++kIdx) {
    LoadScale(scaleGm[kIdx * baseK / 32], scaleL1, baseK / 32);  // tiny transfer, re-issued each kIdx
    Mmad(...);
}

// ✅ Scale coalesced — one large DMA across full K, reused
LoadScale(scaleGm, scaleL1, K / 32);            // single large transfer, scaleKL1 = K
for (int kIdx = 0; kIdx < K / baseK; ++kIdx) {
    Mmad(..., scaleL1[kIdx * baseK / 32]);      // reuse L1-resident scale
}
```

**Detection**:

| Metric | Path | Condition |
|--------|------|-----------|
| MTE2 dominant | `pipe_utilization.pipeline_util_summary.AIC_MTE2.mean` | dominant |
| Small Scale transport | `bandwidth.*scale*.avg_transaction_gbps` | very low (when `bandwidth` present) |

**Fallback signal** (`bandwidth` absent): MTE2 dominant **and** `kernel_instructions_executed` / `scalar_instructions.AIC.total_count` are inflated relative to the compute volume (`AIC_CUBE.mean`). The flood of small per-K-chunk Scale DMAs shows up as a high instruction + scalar-bookkeeping count, not as a bandwidth number. Confirm structurally: the kernel re-loads Scale inside the K-loop.

**Verification after the fix**:
- `kernel_total_clocks` drops.
- `AIC_MTE2.mean × kernel_total_clocks` (MTE2 absolute) drops.
- `kernel_instructions_executed` and `scalar_instructions.AIC.total_count` drop sharply (the per-chunk Scale-DMA issue count collapses); `AIC_CUBE.mean` stays flat (same compute).

**When NOT to apply**: the win scales with Scale's share of MTE2 traffic. On very large N (B streaming dominates HBM, e.g. `m / baseN` small and N huge) Scale is a tiny fraction and the gain disappears into noise — this fix pays off when K is large and N is moderate. Also requires L1 headroom to hold the full-K Scale.

**Fix C — NZ format**: If input is in ND format, the address mapping may cause many small non-contiguous transfers. Converting to NZ format aligns data for larger DMA requests and avoids format conversion overhead.

📖 Best practice: [scale_coalescing_design](../../ascendc-performance-best-practices/reference/matmul/scale_coalescing_design.md) (co-located in `ops/`).

#### §2.2 Partial overlap with L1 pingpong already enabled — multi-core tile scheduling

**Detection Criteria**:

| Metric | Path | Value |
|--------|------|-------|
| MTE2 dominant | `pipe_utilization.pipeline_util_summary.AIC_MTE2.mean` | dominant (highest among AIC pipes) |
| MTE2 vs CUBE overlap | `pipeline_overlap.AIC_MTE2_vs_AIC_CUBE` | `0.30–0.60` (partial, not poor) |
| L1 pingpong already on | kernel source | `L1_BUFFER_NUM ≥ 2` / `PINGPONG_NUM = 2` / explicit double `InitBuffer` |
| Load is balanced | `top_level_diagnosis.imbalance_ratio` | `≤ 1.3` |
| Cube has real work | `AIC_CUBE.mean` | `≥ 0.40` (rule out §general 2 underutilization) |
| Tile count per core | `(m × n) / (baseM × baseN) / blockDim` | `≥ 2` |

If all six rows match, this issue applies. The remaining MTE2 stall isn't about buffer count — it's about *which* tiles each core processes in each scheduling wave. With column-major tile assignment, in one wave the N cores touch N tiles spread far apart in M-N space → each tile's MTE2 pays full HBM startup → L1 reuse across consecutive tiles on the same core is poor → MTE2 and CUBE keep stepping on each other.

**Fix — SWAT (Sliding Window And Twisted) tile scheduling**:

Replace simple column-major `GetTileIdx()` with M-direction sliding window + N-direction zig-zag traversal. Tiles in one scheduling wave become spatially clustered in M-N → consecutive tiles on the same core share L1-resident A (or B, or scale) data → MTE2 fires less often → overlaps with CUBE better.

The change lives in the `BlockScheduler`, not the inner `BlockMmad`. The matmul data path (`GM → L1 → L0 → MMAD`) is unchanged.

**Related skill**: tile scheduling / tiling partition strategy — [ascendc-tiling-design](https://gitcode.com/cann/cannbot-skills/blob/master/ops/ascendc-tiling-design/SKILL.md). SWAT design (co-located in `ops/`): [swat_design](../../ascendc-performance-best-practices/reference/matmul/swat_design.md).

**When NOT to apply**:
- `imbalance_ratio > 1.3` → fix balance first (general §1); SWAT amplifies pre-existing imbalance.
- `AIC_CUBE.mean < 0.40` → kernel under-utilized (general §2). SWAT won't manufacture work.
- `AIC_MTE2_vs_AIC_CUBE > 0.60` → overlap is already good; switch focus to FIXP or other pipes.
- **SWAT already applied** (scheduler source uses M-window + N-zigzag) but MTE2 still dominant → SWAT has done what it can. Choose next step from the **post-SWAT routing** below.

**Post-SWAT routing** (when §2.2 is exhausted but MTE2 is still dominant — pick by which secondary metric is most off):

1. **Tail-round imbalance** — if `imbalance_ratio` crept above 1.0 and the tail round (last `total_tiles % blockDim` tiles) leaves cores idle → [general §1.4 Tail-Round Imbalance](performance-issues-general.md). Split tail-round tiles in the N direction so the tail tile count ≥ `blockDim`.
2. **FIXP/CUBE overlap regressed** — if `AIC_FIXP_vs_AIC_CUBE` dropped below `0.30` after SWAT, the tile reorder exposed the canonical "FIXPIPE waits for full MMAD chain" serial pattern → §4.2 UnitFlag. Set `mmadParams.unitFlag = FINAL_ACCUMULATION` on the last MMAD of each tile's k-chain, `NON_FINAL_ACCUMULATION` elsewhere — gives 512B-granular MMAD ↔ FIXPIPE overlap without needing L0C DB.
3. **L1 bank conflict on MTE1 reads** — split each L1 buffer in half and put ping/pong on **different L1 banks** (front-half = ping, back-half = pong) so MTE2 writes and MTE1 reads no longer hit the same bank. This optimization is low-impact on the current chip; apply only if `pipe_utilization.AIC_MTE1.mean` is dominant and other routes are exhausted.

**Verification after the fix**:
- `pipeline_overlap.AIC_MTE2_vs_AIC_CUBE` should rise (target ≥ 0.60).
- `AIC_CUBE.mean` should rise.
- On cannsim (single-core trace) the improvement is bounded — SWAT's larger payoff comes from inter-core L2 sharing which a single-core trace under-represents. Treat a small win on cannsim as direction-confirmed, not full impact (see [metrics-reference.md note on `ai_core_active`](performance-metrics-reference.md)).
- The tile reorder can shift load onto other pipes. If `AIC_FIXP_vs_AIC_CUBE` falls below 0.30 after applying SWAT, route to §4.2.

#### §2.3 Redundant operand re-transfer — A full-load

When one operand is small (typically A `[m, k]` with small `m`) and the other is large (B `[k, n]` with huge `n`), the streaming schedule re-loads A from GM once per N-tile. The redundant A traffic as a fraction of B traffic is `baseM / baseN` — independent of `n`. If that ratio is non-trivial (e.g. `m = baseM`, `baseN = 256` → A is ~½ of B's bytes), keeping A **resident in L1** across the N-loop ("A full-load") removes the repeated A transfers.

### Detection Criteria

This is a **structural** call, not a single metric:

| Metric / source | Path | Condition |
|-----------------|------|-----------|
| MTE2 dominant | `pipe_utilization.pipeline_util_summary.AIC_MTE2.mean` | dominant (memory-bound) |
| A re-loaded in N-loop | kernel source | A `[m,k]` tile is re-fetched from GM for every N-tile |
| Redundant-A share | `baseM / baseN` (from tiling) | non-trivial (≳ 0.3) |
| A fits L1 | `m · k · sizeof(dtype)` vs L1 size | A (or its baseM-block) fits resident alongside B ping/pong |

### Fix Method

Keep A (and its scale, for quant) L1-resident across the N-loop; stream only B / scaleB. Select the A-full-load template/policy instead of the streaming one.

**Implementation note**: select the dedicated A-full-load kernel/policy (the one that keeps A resident) over the streaming/SWAT variant — they are separate binaries of the same recipe, differing only in a full-load mode flag. A simplified "fullload" demo that does not fully implement the tiling can regress; use a complete recipe implementation.

📖 Best practice: [fullload_design](../../ascendc-performance-best-practices/reference/matmul/fullload_design.md) (co-located in `ops/`).

### Profiling caveat — measure on real hardware, not cannsim

The A-full-load payoff only appears when `n` is large (needed both to balance all cores at small `m` and for A-reuse to matter). That regime is a huge MAC volume — **cannsim, being cycle-accurate, cannot simulate it in feasible time** (it does not finish). The published win for this optimization was measured with **msprof on real hardware**, and the simplified tutorial demo even regresses on cannsim. Treat A full-load as a structural recommendation; validate it with `msprof`, not a cannsim `summary.json` delta.

### When NOT to Apply

- **MTE2 is dominated by B streaming, not A re-transfer** (`baseM / baseN` small, e.g. tiny `m`): removing A reloads saves little, and holding A resident steals L1 from B ping/pong → can **regress**.
- **A (or its baseM-block) does not fit L1** alongside B double-buffering — no room to make A resident.

---

## 3. MTE1 Bound

### Problem Description

`AIC_MTE1` (L1 → L0A/L0B) has the highest utilization. L1-to-L0 data transfer is the bottleneck.

### Detection Criteria

| Metric | Path | Condition |
|--------|------|-----------|
| MTE1 utilization | `pipe_utilization.pipeline_util_summary.AIC_MTE1.mean` | Dominant or near CUBE level |
| MTE1 vs CUBE overlap | `pipeline_overlap.AIC_MTE1_vs_AIC_CUBE` | < 0.30 |

### Fix Method

L0A and L0B should be double-buffered so MTE1 prefetches the next tile while CUBE computes the current one.

```cpp
// L0A and L0B with 2 buffer copies
pipe.InitBuffer(l0aQue, 2, l0aBufBytes);
pipe.InitBuffer(l0bQue, 2, l0bBufBytes);
```

Also verify that the L0 buffer size is large enough to hold one full tile — if L0 is undersized, MTE1 must refill mid-computation.

📖 Best practice (hand-managed L0 ping-pong handshake): [pingpong_design](../../ascendc-performance-best-practices/reference/matmul/pingpong_design.md), [mte2_preload_design](../../ascendc-performance-best-practices/reference/matmul/mte2_preload_design.md) (co-located in `ops/`).

---

## 4. FIXPIPE Bound

### Problem Description

`AIC_FIXP` (L0C → L1/GM output writeback) has unexpectedly high utilization relative to CUBE, or `AIC_FIXP_vs_AIC_CUBE` overlap is low. FIXPIPE is a post-processing unit that applies bias, quantization, and format conversion; it runs after each MMAD tile. Without explicit fine-grained sync, FIXPIPE waits for the **entire** MMAD k-accumulation chain to finish before draining L0C — that's the canonical serial pattern.

### Detection Criteria

| Metric | Path | Condition |
|--------|------|-----------|
| FIXP utilization | `pipe_utilization.pipeline_util_summary.AIC_FIXP.mean` | Dominant or `> 0.20` |
| FIXP vs CUBE overlap | `pipeline_overlap.AIC_FIXP_vs_AIC_CUBE` | `< 0.30` → CUBE stalls waiting for L0C to drain |

### Fix Method

#### §4.1 Increase N-axis tile size

FIXPIPE throughput is proportional to the output tile's N dimension. Small N tiles cause FIXPIPE to iterate many times with high per-tile setup cost.

```
// Heuristic: N tile should be at least 16 (fp16) or 32 (int8) to amortize FIXPIPE overhead
tilingN = max(tilingN, 16);
```

If FIXPIPE is applying per-element operations (e.g., quantization with per-channel scale), verify that Scale data fits in L1 — L1 cache miss during FIXPIPE can inflate its utilization.

#### §4.2 UnitFlag — fine-grained MMAD ↔ FIXPIPE sync (when L0C double-buffer not available)

**When to use this instead of §4.1**: increasing N tile may not be an option — `baseN` is already at the L0C limit, or the kernel's shape constraints prevent it. Also, L0C double-buffer (which would let CUBE start the next tile while FIXPIPE drains the previous one) requires twice the L0C budget and isn't always feasible.

**What UnitFlag does**: it tells the hardware to issue an early flush signal from MMAD to FIXPIPE at **512-byte data granularity** instead of "end of MMAD chain". MMAD writes a 512B chunk of L0C → FIXPIPE can immediately start draining that chunk while MMAD continues on the next chunk. The two pipes overlap inside a single k-accumulation chain.

**API**: set `mmadParams.unitFlag` on each MMAD call. The last MMAD in the k-chain uses `FINAL_ACCUMULATION`; all earlier MMADs use `NON_FINAL_ACCUMULATION`:

```cpp
// AscendC convention (dav-3510): last K slice = FINAL, earlier = NON_FINAL
constexpr uint32_t FINAL_ACCUMULATION     = 3;
constexpr uint32_t NON_FINAL_ACCUMULATION = 2;

// 1) per MMAD: only the very last k-iter (outer L1 AND inner L0) is FINAL
uint8_t mmadUnitFlag = (iter0 + 1 == kL1Iter && iter1 + 1 == kL0Iter)
    ? FINAL_ACCUMULATION       // last MMAD in this tile's k-chain
    : NON_FINAL_ACCUMULATION;  // intermediate MMAD
mmadParams.unitFlag       = mmadUnitFlag;
mmadParams.cmatrixInitVal = (iter0 == 0 && iter1 == 0);  // init accumulator on first k slice only
AscendC::Te::Mmad(/*…*/, mmadParams);

// 2) the L0C -> GM Fixpipe must AGREE: pass FINAL so the conclusive drain lands
AscendC::Te::Copy(CopyL0C2GM, gmC, tensorL0C, AscendC::Te::FixpipeParams{FINAL_ACCUMULATION});
```

The `iter0` / `iter1` indices come from the outer L1-iteration and inner L0-iteration of the k-loop respectively. Only the very last MMAD of the entire tile's k-accumulation gets `FINAL_ACCUMULATION` — that's what triggers the conclusive FIXPIPE flush; all earlier ones use the 512B-granular early flushes. **Two non-obvious requirements**: (a) the L0C→GM Fixpipe copy must also pass `FINAL_ACCUMULATION` — if the MMAD chain and the Fixpipe disagree on which is final, the drain is wrong; (b) `cmatrixInitVal` must be true only on the first k slice (`iter0==0 && iter1==0`), or the accumulator isn't reset and precision breaks.

**Detection of this specific case** (vs §4.1):
- `AIC_FIXP_vs_AIC_CUBE < 0.30`
- L0C double-buffer is NOT in the code (`InitBuffer` on the L0C tensor uses a single buffer)
- `baseN` is already at architectural limit or business constraint
- Often surfaces *after* a scheduling change (e.g. SWAT) that doesn't directly touch FIXPIPE but shifts the timing such that FIXPIPE's serial drain becomes visible

**Verification after the fix**:
- `pipeline_overlap.AIC_FIXP_vs_AIC_CUBE` should rise above 0.30.
- `AIC_CUBE.mean` should rise (less stalling).
- `AIC_FIXP.mean` should be roughly unchanged (same drain work, just better parallelized).

**Composes with L0C double-buffer**: if an L0C double-buffer is also available, use both — UnitFlag handles intra-tile MMAD↔FIX overlap, L0C DB handles inter-tile CUBE↔FIX overlap.

---

## 5. Cube→Vector GM Round-trip (use L0C→UB direct)

### Problem Description

A fused Cube + Vector kernel (matmul followed by an eltwise epilogue like LeakyRelu / Gelu / Cast / quantization) routes the matmul result through Global Memory:

1. AIC FIXPIPE writes L0C → GM (the matmul output tile)
2. AIV MTE2 reads the same tile GM → UB
3. AIV runs the eltwise on UB
4. AIV MTE3 writes UB → GM

Steps 1 + 2 are a wasted GM round-trip. On dav-3510 / Ascend 950, FIXPIPE has a dedicated **L0C → UB** transport (`AIC_FIXPIPE_L0C_TO_UB0` / `AIC_FIXPIPE_L0C_TO_UB1`) that delivers the tile straight into the paired AIV's UB, skipping HBM entirely.

### Detection Criteria

**Primary signal** (when `bandwidth` is present in `summary.json`):

| Metric | Path | Condition |
|--------|------|-----------|
| FIXPIPE → GM transport active | `bandwidth.AIC_FIXPIPE_L0C_TO_OUT` | Present with significant `transaction_count` |
| AIV reads same data back from GM | `bandwidth.AIV0_MTE2_OUT_TO_UB`, `bandwidth.AIV1_MTE2_OUT_TO_UB` | Present with comparable byte volume |
| GM bandwidth utilization | Both transports above | Each typically `< 0.30` — small tiles waste startup latency twice |
| AIV MTE2 has no overlap with VF compute | `pipeline_overlap.AIVx_MTE2_vs_AIVx_VEC` | Often `≈ 0` because MTE2 is gated by the cross-core "GM ready" flag from AIC |

The smoking gun: same tile size appears in both `AIC_FIXPIPE_L0C_TO_OUT` and `AIVx_MTE2_OUT_TO_UB` in the bandwidth section, and AIV MTE2 is synchronized to AIC FIXPIPE via a cross-core flag (`PIPE_FIX → AIV`).

**Fallback signal** (when `bandwidth` & `cache` sections are absent):

| Metric | Path | Condition |
|--------|------|-----------|
| AIV MTE2 pipe active | `pipe_utilization.pipeline_util_summary.AIVx_MTE2.mean` | Present and > 0 in a kernel where the only AIV input should come from AIC |
| Cube/Vector inter-core overlap absent | `pipeline_overlap.AIC_CUBE_vs_AIVx_VEC` | `= 0` (AIV cannot start until GM round-trip lands) |
| AIV MTE2 has no overlap with VF compute | `pipeline_overlap.AIVx_MTE2_vs_AIVx_VEC` | `≈ 0` or `null` |

If the kernel structurally produces its AIV input via AIC's matmul (i.e. `Process_Vec` reads back `cGM[…]` that `Process_Cube` just wrote), `AIVx_MTE2.mean > 0` alone is enough to suspect this pattern even without the bandwidth section.

### Fix Method

Replace the L0C → GM Fixpipe + GM → UB DataCopy pair with a single Fixpipe whose destination is a UB-position LocalTensor. The hardware routes via `AIC_FIXPIPE_L0C_TO_UB0` / `..._UB1` instead of going through HBM.

**Step 1 — enable the L0C→UB Fixpipe variant.** The `enableUB` flag on `FixpipeConfig` selects the L0C→UB transport instead of L0C→GM:

```cpp
static constexpr FixpipeConfig CFG_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, /*enableUB=*/true};
```

**Step 2 — allocate the UB receiver as a mirror of the full output, plus a vector scratch region.** Each tile lands at its natural GM-equivalent offset (`loop_n * tileN`) with the full row stride `N`, so AIC can address it the same way it addressed `cGM`. A separate `vTmp` half is used by AIV for the eltwise compute on contiguous data:

```cpp
// UB total: 2 * cSize floats:
//   vLocal[0..cSize)       = Fixpipe destination (m×n mirror, stride-n layout)
//   vTmp = vLocal[cSize]   = AIV scratch for the compacted eltwise input
pipe->InitBuffer(BufVEC, 2 * cSize * sizeof(float));
LocalTensor<float> vLocal = BufVEC.Get<float>();
LocalTensor<float> vTmp   = vLocal[cSize];
```

**Step 3 — point Fixpipe at the UB tensor. Keep `dstStride = N` and pick the routing mode.** Two routing options exist on `FixpipeParamsC310`:

- `dualDstCtl = 0` (default) + `subBlockId = 0 or 1` → single-destination mode, AIC explicitly picks which AIV's UB this tile goes to. Pattern: process tiles in a stripe, each tile to one AIV. Used in this kernel because successive `loop_n` values go to alternating AIVs.
- `dualDstCtl = 1` → dual-destination SPLIT_M mode, AIC writes the same tile split by M to BOTH UBs simultaneously (top `M/2` rows to AIV0, bottom `M/2` rows to AIV1; `M` must be even). Used when both AIVs cooperatively consume the same tile. This is the canonical SPLIT_M pattern for fused attention / matmul-epilogue kernels.

```cpp
// Before — L0C → GM
FixpipeParamsC310<CO2Layout::ROW_MAJOR> p(tileN, tileM, tileM, /*dstStride=*/N);
Fixpipe<float, float, CFG_ROW_MAJOR>(cGM[loop_n * tileN], c1Local, p);

// After — L0C → UB direct, single-destination (dstStride stays N: UB mirrors m×n layout)
FixpipeParamsC310<CO2Layout::ROW_MAJOR> p(tileN, tileM, tileM, /*dstStride=*/N);
p.dualDstCtl = 0;                          // single-dest mode (often the default but set explicitly)
p.subBlockId = (loop_n % 2 == 1);          // 0 → UB0, 1 → UB1
Fixpipe<float, float, CFG_ROW_MAJOR_UB>(vLocal[loop_n * tileN], c1Local, p);

// Alternative — L0C → both UBs, SPLIT_M (each AIV gets half the rows of the same tile)
FixpipeParamsC310<CO2Layout::ROW_MAJOR> p(tileN, tileM_half, srcStride, dstStride);
p.dualDstCtl = 1;                          // M / 2 * N to each UB
Fixpipe<float, float, CFG_ROW_MAJOR_UB>(vLocal_perAIV, c1Local, p);
```

**Step 4 — add the forward AIC→AIV cross-core flag.** No reverse handshake is needed: each tile occupies a distinct UB region (`vLocal[loop_n * tileN]`), so there are no UB slots to free. Two API forms work; pick by CANN version:

```cpp
// High-level templated CrossCore API:
AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX * (loop_n % 2));

// Low-level intra_block primitive:
set_intra_block(PIPE_FIX, EVENT_ID0 + 16 * (loop_n % 2));   // +16 is the AIV-lane offset
```

The two forms are equivalent at the hardware level — `+ FLAG_ID_MAX (=16)` and `+ 16 * (lane)` route the flag to the second AIV. Use whichever your CANN version exposes; the user-facing `summary.json` is identical.

**Step 5 — AIV reads its UB region (no MTE2 from GM) and processes via scratch.** The AIV waits on `PIPE_S` (the AIC→AIV cross-core event gates scalar dispatch, not MTE2 anymore). Within each AIV's lane the event is observed as `EVENT_ID0` / `SYNC_AIC_AIV_FLAG` — the `+ 16 * subIdx` from the AIC side is the lane routing, not part of the in-lane event ID:

```cpp
for (uint32_t loop_n = subIdx; loop_n < n / tileN; loop_n += 2) {
    // High-level:
    AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_S>(SYNC_AIC_AIV_FLAG);
    // Low-level equivalent:
    // wait_intra_block(PIPE_S, EVENT_ID0);

    DataCopy(vTmp, vLocal[loop_n * tileN], paramIn);     // UB → UB compact (stride-N → contiguous)
    LeakyRelu(vTmp, vTmp, 0.01f, tileN * m);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    DataCopy(cGM[loop_n * tileN], vTmp, paramOut);       // UB → GM (only memory hop left)
}
```

### Verification after the fix

If the fix landed correctly, the next profile should show:
- `bandwidth.AIC_FIXPIPE_L0C_TO_OUT` and `bandwidth.AIVx_MTE2_OUT_TO_UB` disappear; `AIC_FIXPIPE_L0C_TO_UB0` / `..._UB1` appear instead.
- `pipe_utilization.AIVx_MTE2.mean` drops (typically to ≈ 0, since the only AIV input source is now UB).
- `pipeline_overlap.AIC_CUBE_vs_AIVx_VEC` rises above 0.

If the bandwidth section is absent, rely on the second and third signals only.

### Pitfalls

- **`dstStride` stays `N`, not `tileN`.** The UB is laid out as a stride-N mirror of the GM output, so AIC and AIV use the same offset arithmetic. Setting `dstStride = tileN` is a plausible-looking mistake that writes the wrong rows.
- **Don't add a reverse AIV→AIC handshake** thinking the UB needs releasing. UB regions are disjoint per tile in single-destination mode — adding a bidirectional flag protocol (e.g., `wait_intra_block(PIPE_FIX, …)` in AIC's loop) will deadlock when the pre-arm flags don't line up with the lane routing.
- **AIV waits on `PIPE_S`, not `PIPE_V` or `PIPE_MTE2`.** The old `wait_intra_block(PIPE_MTE2, EVENT_ID0)` gated an MTE2 that no longer exists; `PIPE_V` will gate vector compute but misses the scalar dispatch that needs to issue first. `PIPE_S` is the canonical pipe for cross-core scalar gating.
- **In-lane event ID is `EVENT_ID0`, not `EVENT_ID0 + 16 * subIdx`.** The `+ 16 * (loop_n % 2)` on the AIC `set_intra_block` is the lane selector; from inside the receiving AIV the event is just `EVENT_ID0` (or `SYNC_AIC_AIV_FLAG` in the high-level API).
- **`dualDstCtl` defaults to 0 but set it explicitly** when using `subBlockId` routing. Some `FixpipeParamsC310` constructors zero-initialize the struct, but relying on the default is fragile across CANN versions where SPLIT_M may become the default.

### When Not to Apply

- The matmul output is consumed only by a downstream kernel (not by an AIV epilogue in the same kernel) — there is no AIV side to receive the UB tile.
- The full output mirror exceeds the per-AIV UB budget. Required UB per AIV ≈ `(cSize + scratchSize) × sizeof(OutDType)`. Check against `GetCoreMemSize(CoreMemType::UB, ...)` (typically 192 KB or 248 KB depending on SKU). For large outputs, switch to a tile-mirror layout (UB sized as `2 × tileM × tileN`, `dstStride = tileN`) with proper bidirectional sync — but only do this if the simple full-mirror version doesn't fit.

**Related Skills**:

📖 Full matmul + epilogue fusion template (Tensor API, `Te::Fixpipe<SPLIT_M>`): [ascendc-direct-invoke-template — matmul_fusion_guide](https://gitcode.com/cann/cannbot-skills/blob/master/ops/ascendc-direct-invoke-template/references/matmul_fusion_guide.md)

📖 dav-3510 cross-core paths (L0C↔UB, UB↔L1, SSBuffer): [npu-arch](https://gitcode.com/cann/cannbot-skills/blob/master/ops/npu-arch/references/npu-arch-guide.md)

---

## 6. SCALAR Bound (AIC side)

### Problem Description

`AIC_SCALAR` utilization is significantly higher than CUBE/MTE pipelines. Scalar overhead — either from register spill or from backpressure caused by another pipeline's full issue queue — is stalling the AIC core.

### Detection Criteria

| Metric | Path | Threshold | Description |
|--------|------|-----------|-------------|
| SCALAR utilization | `pipe_utilization.pipeline_util_summary.AIC_SCALAR.mean` | Significantly > CUBE | Scalar is dominant on AIC |
| Load/store ratio | `scalar_instructions.AIC.load_store_ratio` | ≥ 0.30 | Register spill |
| Dcache refill count | `cache.dcache_refill_count` | > 0 | Spill reaching Global Memory (very expensive) |
| Icache refill ticks | `cache.icache_refill_ticks` | > 0 | Instruction cache misses |

### Fix Method

#### §6.1 Register spill reaching GM (`dcache_refill_count > 0`) — HIGH PRIORITY

Each spilled register requires a GM access (~hundreds of cycles). Fix before any other optimization.

**Fix actions** (apply in order of impact):

1. **Replace class member variables with local variables** inside the compute function — member access requires pointer dereference, breaking the compiler's ability to keep values in registers.

   ```cpp
   // ❌ Member variable: compiler cannot keep in register across calls
   struct MyKernel {
       int count;
       void Process() { count++; }
   };

   // ✅ Local variable: compiler can allocate to register
   void Process(int& count) { count++; }
   ```

2. **Remove arrays from structs** — arrays in structs prevent constant propagation of struct fields.

3. **Eliminate multi-level pointer dereference** — replace with value-type local copies at the start of the hot loop.

   ```cpp
   // ❌ Pointer chain: each access may reload from memory
   int val = ctx->params->config->blockSize;

   // ✅ Copy to local at loop entry
   int blockSize = ctx->params->config->blockSize;
   for (...) { use(blockSize); }
   ```

4. **Move TilingData copy** — read TilingData fields directly from the kernel argument pointer instead of copying into a struct.

5. **Define variables close to their use site** — reduces live ranges and helps the register allocator.

#### §6.2 Icache misses (`icache_refill_ticks > 0`)

The instruction cache is evicting kernel code, causing refills (stalls) during execution.

**Fallback (no `cache` section)**: Use indirect signals: low overall pipeline utilization with no clear dominant pipe + many small functions / deep call chains in the kernel source. If the kernel is short-lived and runs few times, icache misses on the first run are first-touch effects and not actionable.

**Fix**:
- Add icache prefetch at loop entry or before large function calls
- Reduce code size: eliminate dead branches, merge small functions that are called in hot paths
- Move cold paths (error handling, rarely-taken branches) out of the hot section

#### §6.3 Scalar backpressure from CUBE

If `load_store_ratio < 0.30` but `AIC_SCALAR` is still elevated, the scalar issue queue is blocked because the CUBE (or MTE) pipeline has a full queue and stops accepting new instructions. Scalar instructions accumulate in the queue and inflate SCALAR utilization.

> **Precondition — apply only if CUBE is genuinely busy**: this rule assumes the CUBE queue is full enough to back up dispatch. If `AIC_CUBE.mean < 0.40`, CUBE is not busy enough to be the backpressure source — the elevated SCALAR is then from loop bookkeeping on a kernel that isn't doing real work. Go to [general §2 Kernel Underutilization](performance-issues-general.md) instead.

**Diagnosis via `AIC_SCALAR_vs_AIC_CUBE`** (assumes `AIC_CUBE.mean ≥ 0.40`):
- **High overlap (> 0.50) + SCALAR elevated**: SCALAR and CUBE are running at the same time — SCALAR IS the bottleneck (genuine spill despite low `load_store_ratio`, or high loop overhead).
- **Low overlap (< 0.20) + SCALAR elevated + `load_store_ratio < 0.30`**: SCALAR and CUBE take turns serially — CUBE queue is full, blocking scalar dispatch. Treat as CUBE Bound.

**Fix**: Identify and resolve the CUBE-side bottleneck (MTE1 overlap, L0C drain). The SCALAR metric will normalize once CUBE throughput improves.

**Related Skills**:

📖 TilingData and struct optimization: [ascendc-tiling-design](https://gitcode.com/cann/cannbot-skills/blob/master/ops/ascendc-tiling-design/SKILL.md)