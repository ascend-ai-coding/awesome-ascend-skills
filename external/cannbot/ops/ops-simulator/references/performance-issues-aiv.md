# Vector Core (AIV) Performance Issues

AIV-side pipeline bound diagnosis and fix solutions. All metric paths refer to fields in `summary.json`.

> **Reading order**: scan the Quick Reference Table for a row matching your symptoms, then jump to the section number in the rightmost column. Section numbers `§N.M` are greppable verbatim across this file.

---

## Quick Reference Table

| Issue | Detection metric | Threshold | Fix | Section |
|-------|------------------|-----------|-----|---------|
| UB-bouncing pattern (applies regardless of dominant pipe) | `ub_traffic_ratio` OR `SIMD_LdStIPC / SIMD_ExecIPC` | `≥ 1.0` OR `≥ 0.7` (with `RVECEX_count > 0`) | VF RegAPI (intermediates in registers) | §1.1 |
| VECTOR Bound — low compute IPC, independent ops | `aiv_vector_instructions.SIMD_ExecIPC` | `< 1.2` AND compute body has parallel streams | Reorder for compiler dual-issue | §1.2a |
| VECTOR Bound — low compute IPC, dependency chain | `aiv_vector_instructions.SIMD_ExecIPC` | `< 1.2` AND compute body is sequential | Route to §1.1 (RegAPI) | §1.2b → §1.1 |
| MTE2 Bound — no overlap | `pipeline_overlap.AIV0_MTE2_vs_AIV0_VEC` | `< 0.05` | Enable double buffer (`bufNum=2`) | §2.1 |
| MTE2 Bound — setup-bound per iter | `pipeline_overlap.AIV0_MTE2_vs_AIV0_VEC` AND `dominant_pipeline_util` | `0.05–0.60` AND util `< 0.70` | Increase `ubFactor` (batch rows) | §2.2 |
| MTE2 Bound — small DMA granularity | `bandwidth.AIVx_MTE2_OUT_TO_UB.avg_transaction_gbps` | very low | UB batching | §2.3 |
| MTE2/MTE3 Bound — pipeline too shallow, UB has room | `AIVx_MTE2.mean` dominant AND `bufNum == 2` in source AND UB not full | transfer per loop is not tiny | Grow buffer depth (`bufNum` 4–8) | §2.4 |
| MTE3 Bound — store-side overlap absent | `pipeline_overlap.AIV0_MTE3_vs_AIV0_VEC` | `< 0.30` | Double buffer `VECOUT`; fuse downstream consumer | §3.1, §3.2 |
| MTE3 Bound — many small stores in a gather/scatter loop | `AIVx_MTE3.mean` elevated AND per-row `DataCopy` store in source AND dst contiguous | per-row stores, contiguous destination | Accumulate rows in UB, one large store | §3.3 |
| SCALAR Bound — register spill (AIV) | `scalar_instructions.AIV0.load_store_ratio` | `≥ 0.30` | Local vars, shrink live ranges | §4.1 |
| SCALAR Bound — loop overhead | `load_store_ratio < 0.30` + low `SCALAR_vs_VEC` overlap | scalar serial with VF compute | Increase `ubFactor`, unroll | §4.2 |
| SCALAR Bound — backpressure from SIMD | `pipeline_overlap.AIV0_SCALAR_vs_AIV0_VEC` | high AND `load_store_ratio < 0.30` | Treat as VECTOR Bound | §4.3 |
| SCALAR Bound — irregular gather/scatter loop | `AIVx_SCALAR` dominant + `load_store_ratio < 0.30` + per-element address from index | scalar address arithmetic on non-contiguous access | Rewrite gather as SIMT (`VF_CALL`) | §5.1 |
| SCALAR Bound — serial-dependency scalar scan | `AIVx_SCALAR` dominant + loop-carried dependency in source | each iteration depends on previous (histogram, scan) | Decompose into independent per-position checks, run as SIMT | §5.2 |

> **Missing-section note**: if `bandwidth` is missing, the §2.3 row will not trigger by itself — `pipe_utilization.AIVx_MTE2.mean` being dominant alongside §2.1 / §2.2 overlap states is the fallback.

---

## 1. VECTOR Bound

### Problem Description

`AIVx_SIMD` has the highest utilization among all AIV pipelines, indicating the vector compute unit is the bottleneck.

> **§1.1 applies more broadly than the VECTOR Bound umbrella.** The UB-bouncing pattern (intermediates spilling to UB between compute ops) hurts throughput regardless of which pipe is formally dominant. If the `SIMD_LdStIPC / SIMD_ExecIPC ≥ 0.7` trigger fires AND the kernel has substantial vector compute (`aiv_vector_instructions.RVECEX_count > 0`), apply §1.1 (RegAPI) — even when SIMD isn't the dominant pipe. The other §1.x sub-issues (low IPC investigations) do require SIMD to be dominant.

### Detection Criteria

| Metric | Path | Condition | Description |
|--------|------|-----------|-------------|
| SIMD utilization | `pipe_utilization.pipeline_util_summary.AIV0_SIMD.mean` | Highest among AIV pipes | Vector unit is dominant |
| UB traffic ratio (total) | `aiv_vector_instructions.ub_traffic_ratio` | `≥ 1.0` | More UB read/write-backs than compute instructions |
| UB traffic per busy cycle | `aiv_vector_instructions.SIMD_LdStIPC / SIMD_ExecIPC` | `≥ 0.7` | On every busy SIMD cycle a load/store rides along compute — UB bouncing pattern |
| Compute IPC | `aiv_vector_instructions.SIMD_ExecIPC` | `< 1.2` | Poor dual-issue utilization |

### Fix Method

#### §1.1 Excessive UB traffic (`ub_traffic_ratio ≥ 1.0` OR `SIMD_LdStIPC / SIMD_ExecIPC ≥ 0.7`)

Intermediate results are written to UB between compute steps instead of staying in registers. Each UB write-back costs extra instructions and memory cycles.

> **Why two triggers**: `ub_traffic_ratio` is a kernel-wide count ratio. It can stay under 1.0 even when bouncing is heavy (e.g. when the dependency chain is shorter than the buffer-resident operands). `SIMD_LdStIPC / SIMD_ExecIPC` normalizes per busy cycle and catches the bouncing pattern earlier — if compute and load/store are firing in lockstep within busy cycles, intermediates are flowing through UB regardless of the kernel-total ratio. Apply this fix if **either** trigger fires.

**Fix**: Use VF RegAPI — keeps intermediate values in registers, writes to UB only at the end.

> **When this fix applies, go read the regbase skill now and rewrite — it's a step to do, not a blocker.** Immediately open / invoke [`ascendc-regbase-best-practice`](../../ascendc-regbase-best-practice/SKILL.md) (co-located in `ops/`): it carries the VF / RegAPI surface you need — `__simd_vf__` kernels, `RegTensor`, predicate masks (`UpdateMask` / `preg`), vector-length loops, `DataCopy` with `LoadDist` / `StoreDist`, reg-form `ReduceSum` / `Sqrt`. Study it and rewrite the compute in registers in the same pass — don't pause to ask, defer, or treat "need to read another skill" as a stopping point; that's the expected way to do this step. The snippet below only shows the *shape* of the change; get exact signatures from that skill or the CANN headers (don't invent them). Compile errors after the rewrite (`no member named MicroAPI`, `unknown type __local_mem__`) are RegAPI surface issues — resolve them in `ascendc-regbase-best-practice`.

```cpp
// ❌ LocalTensor: intermediates spill to UB between steps
LocalTensor<float> xLocal = xQue.DeQue<float>();
LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
AscendC::Mul<float>(tmpLocal, xLocal, xLocal, 32);   // write to UB
AscendC::Add<float>(yLocal, tmpLocal, xLocal, 32);   // read back from UB

// ✅ RegTensor: intermediates stay in registers
AscendC::Reg::RegTensor<float> xReg = xRegQue.DeQueReg<float>();
AscendC::Reg::RegTensor<float> tmpReg;
AscendC::Reg::Mul<float>(tmpReg, xReg, xReg, 32);   // in registers
AscendC::Reg::Add<float>(yReg, tmpReg, xReg, 32);   // in registers
```

> **Convert the whole compute chain, not just part of it.** The win comes from never materializing intermediates in UB, so the *entire* chain must stay in registers — load once, compute through, store once. Converting only a few ops while leaving the heaviest ones in membase keeps the dominant UB traffic **and** adds VF-scope entry/exit overhead, so the partial fix can run **slower** than the membase version. Don't leave membase scratch buffers behind for the steps you registerized.

**Verification after the fix**: `SIMD_LdStIPC / SIMD_ExecIPC` should drop sharply (toward ~0.2); `ub_traffic_ratio` should also drop; `SIMD_ExecIPC` should rise. If they don't, only part of the chain was registerized (see the note above) — `kernel_total_clocks` may even regress.

**Related Skills**:

📖 **Read this first** — RegTensor / RegAPI surface and exact signatures: [ascendc-regbase-best-practice](../../ascendc-regbase-best-practice/SKILL.md) (co-located in `ops/`).

#### §1.2 Low compute IPC (`SIMD_ExecIPC < 1.2`)

SIMD dual-issue requires two independent instructions (e.g., Add + Mul) to be schedulable in the same cycle. Low IPC has two structurally different causes — pick the fix by inspecting the compute body:

**§1.2a — Independent ops that the compiler isn't interleaving**: the kernel has parallel compute streams (e.g. multiple unrelated tensors processed in the same loop) but they're written sequentially so the compiler can't dual-issue. Reorder the source so independent ops are adjacent.

**§1.2b — True data-dependency chain** (each op consumes the previous result, e.g. `Mul → ReduceSum → Sqrt → Div → Mul`): interleaving is impossible — there are no independent ops to pair. The fix here is **not §1.2**, it is **§1.1 RegAPI** — keeping the chain's intermediates in registers shortens per-op latency and raises effective IPC even though the chain stays serial. Route to §1.1.

To distinguish: grep the compute body for tensor names. If the same tensor flows from output of one op into input of the next, repeatedly → §1.2b → use §1.1.

> **SIMT note**: if the kernel uses SIMT (scatter/gather, irregular access patterns), check `SIMT_ExecIPC` and `SIMT_BranchIPC` alongside `SIMD_ExecIPC`. Low `SIMT_BranchIPC` indicates branch-heavy divergence. Note that `RVECEX_count` / `RVECLD_count` / `RVECST_count` are shared counters — `ub_traffic_ratio` does not separate SIMD from SIMT contributions.

---

## 2. MTE2 Bound (AIV side)

### Problem Description

`AIVx_MTE2` (Global Memory → UB data load) has the highest AIV utilization. Data movement is the bottleneck, not compute.

### Detection Criteria

| Metric | Path | Condition |
|--------|------|-----------|
| MTE2 utilization | `pipe_utilization.pipeline_util_summary.AIV0_MTE2.mean` | Highest or near-highest among AIV pipes |
| MTE2 vs VEC overlap | `pipeline_overlap.AIV0_MTE2_vs_AIV0_VEC` | See sub-cases below |
| Avg transaction bandwidth | `bandwidth.AIV0_MTE2_OUT_TO_UB.avg_transaction_gbps` | Very low → small DMA granularity |

### Fix Method

#### §2.1 No overlap (`AIV0_MTE2_vs_AIV0_VEC < 0.05`)

Double buffer is not working or not enabled. MTE2 and SIMD execute serially.

> **Precondition check — grep the source first**: if `BUF_NUM ≥ 2` (or `pipe.InitBuffer(que, 2, …)` with `TQue<…, 1>`) is **already** in the kernel, the issue is not buffer count. The three common reasons overlap stays `< 0.05` with double buffer already enabled:
> - **Too few iterations per worker** (typically `< 3`) — pipeline can't reach steady state because warmup/drain dominates. Symptom: `dominant_pipeline_util` low (`< 0.50`) across all pipes, `kernel_instructions_executed` low relative to shape. Often a side-effect of an over-aggressive `ubFactor` (see §2.2) on a small problem shape. Fix is shape-side, not buffer-side: either reduce `ubFactor` so more iterations survive, or run on a representative production shape.
> - **EnQue/DeQue misordered** — buffer count is right but the queue protocol breaks pipelining. Verify each `AllocTensor → DataCopy → EnQue → DeQue → … → FreeTensor` chain is properly paired with no premature `FreeTensor` calls.
> - **Buffer too shallow** — `bufNum == 2` works but isn't deep enough to saturate the transfer pipeline, and UB has spare room. This is common on compute-light kernels where the `< 0.05` overlap reading is an artifact (no VF compute — SIMD/SIMT ≈ 0), not evidence the buffer count is fine. Go to **§2.4** to grow buffer depth.

> **Compute-light kernels (no VF compute — `AIVx_SIMD.mean` & `AIVx_SIMT.mean` < 0.05)**: this trigger still correctly points to "enable double buffer", but the real win is the MTE2↔MTE3 overlap, not MTE2↔VEC. After the fix `AIV0_MTE2_vs_AIV0_VEC` stays near zero (no VF compute to overlap with) — verify via `kernel_total_clocks` drop, not the overlap metric. See [metrics-reference §8](performance-metrics-reference.md).

**Fix (default case, `BUF_NUM = 1` in source)**: Enable double buffer with `bufNum=2` and ensure EnQue/DeQue are correctly paired.

```cpp
// ✅ Double buffer: load next tile while computing current tile
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECIN, 1> xQue;   // depth=1, double buffer via InitBuffer num=2
AscendC::TQue<AscendC::TPosition::VECOUT, 1> yQue;
pipe.InitBuffer(xQue, 2, tileBytes);
pipe.InitBuffer(yQue, 2, tileBytes);

for (int i = 0; i < tileNum; ++i) {
    LocalTensor<T> xBuf = xQue.AllocTensor<T>();
    DataCopy(xBuf, gm_x[i * tileLen], tileLen);
    xQue.EnQue(xBuf);

    LocalTensor<T> xIn = xQue.DeQue<T>();
    LocalTensor<T> yOut = yQue.AllocTensor<T>();
    // compute ...
    yQue.EnQue(yOut);

    LocalTensor<T> yStore = yQue.DeQue<T>();
    DataCopy(gm_y[i * tileLen], yStore, tileLen);
    yQue.FreeTensor(yStore);
    xQue.FreeTensor(xIn);
}
```

**Related Skills**:

📖 Double buffer enabling: [ascendc-api-best-practices](https://gitcode.com/cann/cannbot-skills/blob/master/ops/ascendc-api-best-practices/SKILL.md)

#### §2.2 Partial overlap, per-iteration setup dominates (`AIV0_MTE2_vs_AIV0_VEC` between 0.05 and 0.60)

Double buffer is enabled but the kernel processes too few rows per loop iteration — DMA startup latency per iteration dominates the actual byte-transfer time, and per-iteration scalar bookkeeping spreads across all pipes. Symptom: MTE2 dominant, but `dominant_pipeline_util < 0.70` and all 3–4 AIV pipes hover in a 0.30–0.60 band (no single pipe saturated).

**Sub-zones**:
- `0.05–0.30` — double buffer may not be fully effective (recently enabled, or queue depth too small). First confirm `bufNum=2` and `EnQue`/`DeQue` pairing are correct, then increase `ubFactor`.
- `0.30–0.60` — double buffer is working but each iteration is too small. The fix is `ubFactor` alone. Confirm `BUF_NUM ≥ 2` and a multi-row UB allocation pattern is **not** already in the source — otherwise see "When NOT to apply" below.

**Fix**: Increase `ubFactor` (number of rows processed per UB loop iteration) so each DMA transfer covers more data and per-iteration scalar overhead amortizes.

```
ubFactor = (ubSize - fixedSize) / linearCoef
```

`fixedSize` = UB bytes that don't scale with `ubFactor` (e.g. `gammaBuf_` if gamma is row-shared). `linearCoef` = per-row UB bytes × `BUF_NUM` summed across queues. A larger `ubFactor` means fewer, larger DMA transactions and fewer outer-loop iterations.

```cpp
// ❌ Before — 1 row per iteration
for (int64_t loop = 0; loop < tilingData->a; loop++) {
    CopyInX(loop);                          // DMA: 1 row
    Compute();                              // setup + 1-row work
    CopyOut(loop);
}

// ✅ After — ubFactor rows per iteration
for (int64_t loop = 0; loop < CeilDiv(a, ubFactor); loop++) {
    CopyInX(loop, ubFactor);                // DMA: ubFactor rows
    Compute(ubFactor);                      // setup amortized over ubFactor rows
    CopyOut(loop, ubFactor);
}
```

**When NOT to apply**:
- Source already batches multiple rows per UB iteration — `ubFactor` is at the UB capacity ceiling. Next step is structural (e.g. fuse with downstream consumer, reduce intermediate UB tensors).
- `dominant_pipeline_util ≥ 0.80` — kernel is genuinely pipe-bound, not setup-bound. Look at the dominant pipe's own bottleneck (e.g. HBM bandwidth ceiling).

> **Watch out — pipeline collapse on small shapes**: increasing `ubFactor` reduces the number of outer-loop iterations per worker. If `iterations_per_worker = ceil(rows_per_worker / ubFactor) < 3`, `BUF_NUM=2` can no longer hide MTE2 behind VF compute because warmup/drain dominates. Symptom after applying this fix: `AIV0_MTE2_vs_AIV0_VEC` crashes back to near-zero, all pipe utilizations drop, but `kernel_total_clocks` still goes down (per-iter scalar overhead saved). On large shapes this doesn't happen — pipeline stays saturated.
>
> If after this fix `kernel_total_clocks` did **not** improve, `ubFactor` was too aggressive — back it off. If clocks improved but overlap dropped to near-zero, the trade-off worked on this shape but **don't trigger §2.1 on the new overlap reading** — it's a false positive (see §2.1 precondition check). Re-verify on a production-representative shape if available.

#### §2.3 Small DMA granularity (`avg_transaction_gbps` very low)

Even with double buffer, each individual DMA request transfers very little data. Startup overhead per transaction dominates, and adding more buffers will not help.

**Fix**: UB batching — accumulate multiple rows in UB per transfer, reducing the number of DMA requests.

```cpp
// ❌ One DMA per row (many small requests)
for (int i = 0; i < rows; ++i) {
    DataCopy(ubBuf, gmBuf[i * rowLen], rowLen);
    Compute(ubBuf, rowLen);
}

// ✅ Batch multiple rows per DMA
constexpr int BATCH = 8;
for (int i = 0; i < rows; i += BATCH) {
    DataCopy(ubBuf, gmBuf[i * rowLen], BATCH * rowLen);  // one large transfer
    for (int j = 0; j < BATCH; ++j)
        Compute(ubBuf[j * rowLen], rowLen);
}
```

#### §2.4 Pipeline too shallow — grow buffer depth (`bufNum` > 2)

Double buffer (`bufNum = 2`) is enabled and working, but two in-flight buffers don't keep the transfer pipeline full: the next MTE2 load can't start until a buffer frees up, so MTE2 still dominates. When UB has spare capacity, a **deeper** queue (more in-flight buffers) lets consecutive transfers overlap each other and the opposite-direction MTE3, raising effective bandwidth. This is a distinct knob from `ubFactor` (§2.2, rows *per* transfer) and UB batching (§2.3, bytes per transfer) — here the transfer size is unchanged; only the number of buffers grows.

### Detection Criteria

| Metric | Path | Condition |
|--------|------|-----------|
| MTE2 (or MTE3) dominant | `pipe_utilization.pipeline_util_summary.AIVx_MTE2.mean` | dominant or near-highest |
| Double buffer already on | kernel source | `bufNum == 2` on the input/output queues |
| No imbalance | `top_level_diagnosis.imbalance_ratio` | `≤ 1.3` |
| No register spill | `scalar_instructions.AIVx.load_store_ratio` | `< 0.30` |
| UB has spare room | kernel source: `InitBuffer` sizes vs UB capacity | 2 buffers don't fill UB |

There is **no single metric trigger** — this is a structural call. On compute-light kernels the `*_vs_VEC` overlaps are an artifact (no VF compute — SIMD/SIMT ≈ 0, see [metrics-reference §8](performance-metrics-reference.md)) and cannot confirm or deny shallow buffering; lean on "MTE pipe dominant + UB not full + per-loop transfer is not tiny".

### Fix Method

Grow `bufNum` past 2, sizing it adaptively so each buffer still holds one loop's worth of data:

```cpp
// ❌ Fixed double buffer — pipeline only 2 deep
constexpr int BUF_NUM = 2;
pipe.InitBuffer(xQue, BUF_NUM, perLoopBytes);

// ✅ Deepen the pipeline to use spare UB
int bufNum = 2;
while (bufNum < MAX_BUF_NUM && (ubSize / (bufNum + 1)) >= perLoopBytes) {
    ++bufNum;                       // next depth still fits one loop's transfer
}
pipe.InitBuffer(xQue, bufNum, perLoopBytes);
```

### When NOT to Apply

- **Per-transfer is tiny** (small `cols`, `avg_transaction_gbps` very low) — startup latency, not transfer time, is the wall; more buffers can't hide a fixed per-DMA launch cost. Use §2.3 UB batching (fewer, larger transfers) instead.
- **UB already full at `bufNum = 2`** — no room to deepen; reduce per-loop size or batch the store side (§3).
- **Diminishing returns** — past `bufNum ≈ 6–8` overlap is near its limit and shrinking each buffer can introduce other overhead; stop when `kernel_total_clocks` flattens.

### Verification after the fix

- `kernel_total_clocks` should drop.
- Absolute MTE2 busy time (`AIVx_MTE2.mean × kernel_total_clocks`) should drop, even if the *ratio* moves less.
- The dominant pipe may shift to the next bottleneck (e.g. SCALAR) — that is **exposure**, not regression (confirm via flat absolute scalar counts). Re-diagnose from the top against the new profile.

---

## 3. MTE3 Bound (AIV side)

### Problem Description

`AIVx_MTE3` (UB → Global Memory store) has high utilization relative to compute. Output writeback is the bottleneck.

### Detection Criteria

| Metric | Path | Condition |
|--------|------|-----------|
| MTE3 utilization | `pipe_utilization.pipeline_util_summary.AIV0_MTE3.mean` | High relative to SIMD |
| MTE3 vs VEC overlap | `pipeline_overlap.AIV0_MTE3_vs_AIV0_VEC` | < 0.30 → no effective overlap |

### Fix Method

#### §3.1 Increase overlap (double buffer `VECOUT`)

MTE3 (store) can overlap with VF compute when the output queue has double buffering. If `AIV0_MTE3_vs_AIV0_VEC < 0.30`, check that `VECOUT` queue also uses `bufNum=2`.

> **Compute-light kernels (no VF compute — `AIVx_SIMD.mean` & `AIVx_SIMT.mean` < 0.05)**: `AIV0_MTE3_vs_AIV0_VEC` reads near zero regardless of double buffering — there is no VF compute to overlap with, and the relevant overlap (MTE3↔MTE2) is not exposed. Don't read low `MTE3_vs_VEC` as "no overlap" on such kernels; verify store-side fixes via `kernel_total_clocks`. See [metrics-reference §8](performance-metrics-reference.md).

#### §3.2 Reduce output size (fuse downstream consumer)

If the kernel produces a large intermediate output that is immediately consumed, consider fusing the consumer operation to eliminate the store entirely.

#### §3.3 Batch irregular-gather stores (many small stores → one large store)

In a gather/scatter kernel the **source** is non-contiguous (each row is fetched from a different `rowIdx`), so the copy-in must stay row-by-row. But the **destination** is usually contiguous (rows land in sorted/output order). When the kernel also stores per-row, MTE3 fires one small DMA per row — each pays full launch latency, and the per-row store carries scalar address/sync bookkeeping. Accumulating the gathered rows in UB and writing them out in **one large contiguous DMA** collapses both costs.

This is the store-side counterpart of §2.3 UB batching: apply it when the load must stay per-row (non-contiguous source) but the store need not.

### Detection Criteria

| Metric | Path | Condition |
|--------|------|-----------|
| MTE3 elevated | `pipe_utilization.pipeline_util_summary.AIVx_MTE3.mean` | high relative to MTE2 / others |
| Per-row store in source | kernel source | a `DataCopy(gm[...], ub, rowLen)` inside the gather loop, one call per row |
| Destination contiguous | kernel source | output rows are written in consecutive offsets |

No single metric is sufficient — confirm by reading the gather loop. On compute-light kernels the `*_vs_VEC` overlaps are an artifact (no VF compute — SIMD/SIMT ≈ 0, see [metrics-reference §8](performance-metrics-reference.md)) and cannot diagnose this.

### Fix Method

```cpp
// ❌ Gather row-by-row, store row-by-row — one small MTE3 per row
for (int i = 0; i < curLoopElements; ++i) {
    int64_t src = rowIdx[i] / k * cols;
    CopyXIn(src, ubRow, cols);                       // per-row load (source non-contiguous)
    CopyXOut(dstBase + i * cols, ubRow, cols);       // per-row store ← many small DMAs
}

// ✅ Gather row-by-row into one UB buffer, store once
LocalTensor<float> xLocal = xQue.AllocTensor<float>();
for (int i = 0; i < curLoopElements; ++i) {
    int64_t src = rowIdx[i] / k * cols;
    CopyXIn(src, xLocal[i * cols], cols);            // load still per-row
}
CopyXOut(dstBase, xLocal, curLoopElements * cols);   // one large contiguous store
```

### Verification after the fix

- `AIVx_MTE3.mean` and its absolute busy time (`mean × kernel_total_clocks`) should drop sharply.
- `scalar_instructions.AIVx.total_count` should also drop — per-row store bookkeeping is gone (this is a *real* scalar reduction, not exposure; confirm via the absolute count).
- `kernel_total_clocks` should drop.
- The dominant pipe often shifts to / stays SCALAR (the gather/histogram scalar loops). If it does, route to §4.2 — or, for irregular access patterns, consider a SIMT rewrite (see §5).

### When NOT to Apply

- **Destination is also non-contiguous** (scatter to scattered offsets) — there is no single large store to batch into.
- **UB can't hold `curLoopElements × cols`** — reduce the per-loop element count, or keep a moderate batch (`BATCH` rows per store) rather than the whole loop.

---

## 4. SCALAR Bound (AIV side)

### Problem Description

`AIVx_SCALAR` has elevated utilization. On AIV cores, this typically means the scalar control-flow overhead (loop counters, address calculations) is disproportionately large relative to vector compute, or register spill is occurring.

### Detection Criteria

| Metric | Path | Threshold | Description |
|--------|------|-----------|-------------|
| SCALAR utilization | `pipe_utilization.pipeline_util_summary.AIV0_SCALAR.mean` | Significantly above AIV0_SIMD | Scalar overhead is dominant |
| Load/store ratio | `scalar_instructions.AIV0.load_store_ratio` | ≥ 0.30 | Register spill on AIV core |
| SCALAR vs VEC overlap | `pipeline_overlap.AIV0_SCALAR_vs_AIV0_VEC` | See sub-cases below | Distinguishes backpressure from genuine scalar overhead |

### Fix Method

**§4.1 Register spill (`load_store_ratio ≥ 0.30`)**

> **Calibration check — confirm with absolute counts**: `load_store_ratio` is a denominator-sensitive metric. After a vector-side fix (e.g. §1.1 RegAPI) that reduces total scalar bookkeeping, the ratio can rise even though absolute `scalar_instructions.AIV0.load_count + store_count` dropped. Before applying §4.1, compare absolute load+store counts against the previous profile. If absolute counts are flat or falling, the elevated ratio is a reshuffling artifact and §4.1 does not apply.
- Reduce variable live ranges: define variables close to their use site
- Replace class member variables with local variables inside the compute loop
- Avoid passing large structs; use pointer/reference or unpack fields locally

**§4.2 Loop scalar overhead (`load_store_ratio < 0.30`, `AIV0_SCALAR_vs_AIV0_VEC` low)**

Scalar and SIMD are running serially — each loop iteration does scalar bookkeeping, then vector compute, with no overlap. Scalar overhead is proportionally large because tiles are small.

**Fix**: Increase `ubFactor` so each loop iteration processes more data per scalar setup cost. Unroll inner loops if the trip count is small and known at compile time.

**§4.3 SIMD backpressure (`load_store_ratio < 0.30`, `AIV0_SCALAR_vs_AIV0_VEC` high)**

SCALAR and SIMD are running simultaneously — SIMD is the actual bottleneck and its full issue queue is blocking scalar dispatch, inflating scalar cycle counts.

**Diagnosis**: If `AIV0_SCALAR_vs_AIV0_VEC > 0.50` and `AIV0_SIMD.mean > 0.50`, treat as VECTOR Bound. Fixing the VECTOR bottleneck (RegAPI, double buffer) will normalize SCALAR utilization.

> **Note**: Check both `AIV0_SCALAR_vs_AIV0_VEC` and `AIV1_SCALAR_vs_AIV0_VEC` (cross-core pair present in summary.json) to confirm the pattern is consistent across AIV cores.

---

## 5. SCALAR Bound — rewrite as SIMT (irregular access / serial dependency)

### Problem Description

The kernel is `AIVx_SCALAR` dominant with `load_store_ratio < 0.30` (no spill), and §4.2's lever (increase `ubFactor`, unroll) has hit its ceiling because the scalar hot loop is **fundamentally not vectorizable** with SIMD — either it does per-element address arithmetic on a **non-contiguous** access pattern (gather/scatter), or it carries a **loop-to-loop dependency** (each iteration reads state written by the previous one, e.g. a histogram boundary scan). SIMD needs contiguous, aligned, independent operands; neither pattern provides them.

The SIMT (Single Instruction Multiple Threads) model fits both: many threads each handle independent elements and can read **Global Memory directly** (no MTE2 staging into UB first), with thread-private registers instead of UB scratch. On dav-3510 / Ascend 950 SIMT runs on the AIV cores and appears in `summary.json` as a **separate `AIVx_SIMT` pipeline** — not under `AIVx_SIMD`.

### Detection Criteria

| Metric | Path | Condition |
|--------|------|-----------|
| SCALAR dominant | `pipe_utilization.pipeline_util_summary.AIVx_SCALAR.mean` | dominant or highest |
| Not register spill | `scalar_instructions.AIVx.load_store_ratio` | `< 0.30` (rules out §4.1) |
| §4.2 exhausted | kernel source | rows-per-iter already large / loop unrolled, scalar still dominant |
| Hot loop is irregular or serial | kernel source | per-element address from an index (§5.1), **or** loop-carried dependency (§5.2) |

This is a structural call confirmed by reading the hot loop, not a single metric. The `*_vs_VEC` overlaps do not help (no VF compute — SIMD/SIMT ≈ 0 on such kernels — see [metrics-reference §8](performance-metrics-reference.md)).

### Fix Method

#### §5.1 Irregular gather/scatter → SIMT direct GM access

Each thread computes its own source offset from the index and reads `x` straight from GM. Use a 2-D thread layout: dim-0 spreads threads across the discrete row indices, dim-1 across the contiguous `cols`.

```cpp
// ❌ Scalar loop: per-row address arithmetic + MTE2 staging
for (int i = 0; i < curLoopElements; ++i) {
    int64_t src = rowIdx[i] / k * cols;
    CopyXIn(src, ubRow, cols);          // MTE2 load
    // ... per-row scalar bookkeeping
}

// ✅ SIMT: threads read GM directly, no MTE2 stage
// dim0 = discrete row index, dim1 = contiguous cols
Simt::VF_CALL<GatherOutSimt>(Simt::Dim3{64, 32, 1},
    curLoopElements, cols, magic, shift, xGmAddr, rowIdxLocalAddr, xLocalAddr);
// inside GatherOutSimt: for (i = tid0; i < curLoopElements; i += tnum0)
//   for (j = tid1; j < cols; j += tnum1)
//     xLocalAddr[i*cols + j] = xGmAddr[ (rowIdx[i]/k)*cols + j ];
```

#### §5.2 Serial-dependency scalar scan → independent per-position SIMT checks

Break the loop-carried dependency into a test that depends only on a position and its neighbours, so every position is independent and threads can run in parallel. Example — histogram boundary detection: instead of carrying `lastExpertId` across iterations, each thread tests `expert[i] != expert[i-1]` (first occurrence) / `expert[i] != expert[i+1]` (last occurrence).

```cpp
// ❌ Serial scan: lastExpertId carried across iterations (cannot parallelize)
for (int i = 1; i < n; ++i)
    if (expert[i] != lastExpertId) { firstIdx[expert[i]] = i; lastExpertId = expert[i]; }

// ✅ SIMT: each position independent — boundary test on neighbours
Simt::VF_CALL<ComputeExpertFirstIndexSimt>(Simt::Dim3{SIMT_THREAD_NUM, 1, 1}, n, ...);
// inside: for (i = tid; i < n; i += tnum)
//   if (i == 0 || expert[i] != expert[i-1]) firstIdx[expert[i]] = i;
```

### Verification after the fix

- A new `AIVx_SIMT` pipeline appears in `pipe_utilization` and becomes the dominant pipe.
- `AIVx_SCALAR.mean` **and** `scalar_instructions.AIVx.total_count` drop sharply (the scalar loop is gone — confirm via the absolute count, this is real reduction not exposure).
- `AIVx_MTE2.mean` drops if §5.1 replaced an MTE2-staged gather with direct GM reads.
- `aiv_vector_instructions.RVECEX_count` rises sharply (SIMT execution shares this counter); `ub_traffic_ratio` typically drops (thread-private registers, not UB scratch).
- `SIMT_ExecIPC` / `SIMT_LdStIPC` / `SIMT_BranchIPC` populate. Low `SIMT_BranchIPC` flags branch divergence (the boundary `if`s) — the next tuning target.
- `AIVx_SIMD.mean` does **not** rise — SIMT is a distinct pipeline.

### When NOT to Apply

- **Large `H` / `cols` (large contiguous transfers)**: SIMT's direct GM→UB read bandwidth is **lower** than a bulk SIMD MTE2 DMA. SIMT wins for small, discrete, irregular packets; for large contiguous rows, deeper-buffered SIMD (§2.4) is faster. Verify by `kernel_total_clocks`, not by assuming SIMT is always better.
- **Access is regular and vectorizable**: use SIMD + VF RegAPI (§1.1), not SIMT — SIMD throughput on contiguous data beats SIMT.

**Related Skills**:

📖 SIMT / VF programming model details: [ascendc-regbase-best-practice](https://gitcode.com/cann/cannbot-skills/blob/master/ops/ascendc-regbase-best-practice/SKILL.md)
