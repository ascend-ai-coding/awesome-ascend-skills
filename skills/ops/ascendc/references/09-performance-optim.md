# Phase 9 — Performance optimization (closed loop)

Not just diagnosis — modify code and verify the gain. Closed loop, **max 3 rounds**.

```
1 Investigate → 2 Baseline → 3 Optimize → 4 Precision → 5 Performance → 6 Iterate
```

## Stage 1 — Investigate (read design.md + both sources first)

Review the operator across **5 dimensions**, one at a time, then emit a ranked report.

### 1. Tiling
- `blockDim` set to the hardware core count? (coupled: `GetCoreNumAiv/Aic`; separated
  vector: AIV count; separated cube: AIC count; MIX: physical core groups, never exceed
  physical cores).
- When `input + output > L2Cache`, split data into L2Cache-sized blocks processed by all
  cores together before moving on.
- Inter-core load balance: alternate the tail block across passes so the same cores don't
  always trail.

### 2. Data copy
- Each `DataCopy` moves **>= 16 KB** (smaller drops bandwidth sharply).
- GM start address 512-byte aligned (on Atlas A2, 32B vs 512B can cost up to ~30% BW).
- Use stride params (`blockCount/blockLen/srcStride/dstStride`) in one issue instead of a
  per-row for-loop.

### 3. API usage
- Create `TPipe` **outside** the kernel class and pass by pointer (class-internal TPipe
  blocks scalar constant folding, ~+17% scalar_time).
- Pure-copy operators use `TQueBind<VECIN, VECOUT>` instead of separate queues.
- Use Counter mode (`SetMaskCount`) instead of manual main/tail mask.
- Matmul `enAtomic=1` to fuse accumulation into GM matrix D (~-12% cycles).
- Combine `BlockReduceSum` + `WholeReduceSum` for buffer→scalar reductions.

### 4. Memory
- Fuse consecutive vector ops in UB (keep intermediates on-chip, avoid GM round-trips).
- Accumulate `A1*B1 + A2*B2 + ...` in L0C (CO1) in place.
- Keep the smaller matrix resident in L1; loop-copy only the larger one.
- (separated arch) put bias in BT Buffer (C2) fused via `Mmad`; quant params in FP Buffer
  (C2PIPE2GM) via `Fixpipe`.

### 5. Pipeline
- CopyIn/Compute/CopyOut three-stage pipeline synced via `TQue`.
- `BUFFER_NUM = 2` (double buffer) so copy overlaps compute (needs loop count >= 2 and
  non-negligible copy time).
- (MIX) async `Iterate<false>()`/`IterateAll<false>()` to avoid per-iteration AIC/AIV sync.

Report format:

```
## Optimization investigation report
### Issues found (ranked by expected gain)
1. [stage X.Y] <issue> — <expected gain>
### Confirmed OK
- [stage X.Y] <checked item>
### Optimization plan
<targets for this round, largest gain first>
```

## Stage 2 — Baseline

Ensure `<op>_perf_cases.jsonl` and `<op>_torch_npu_profiler_report.md` exist (run Phase 8
if missing/stale). Snapshot the current report as `<op>_baseline_report.md` in the same
`test/` directory.

## Stage 3 — Optimize

Re-read the kernel API essentials ([`04a-kernel-api.md`](04a-kernel-api.md)) before
editing. For each target: plan file(s) to change, the change, expected effect, and risk
(precision / tiling impact). Apply changes obeying the **unified anti-patterns**. Rebuild
and install; on compile failure use the debug loop (max 3).

## Stage 4 — Precision (MANDATORY before perf compare)

Run the full precision flow ([`07-precision-eval.md`](07-precision-eval.md)). All pass →
Stage 5. Partial fail → fix/revert, back to Stage 3. Many fail → revert this round.

## Stage 5 — Performance

Re-run Phase 8 with the **exact same** `perf_cases.jsonl` (no add/remove). Compare against
the baseline snapshot (baseline → optimized → reference three-way table). Improved → done.
Not improved / regressed → Stage 6.

## Stage 6 — Iterate (max 3 rounds)

Each round records: target, change summary, precision result, performance result,
decision (keep/revert/continue). After 3 rounds, stop and emit the final summary.

## Final output

Write `<op>_optim_summary.md` (investigation, baseline, iteration history, final
three-way comparison, >= 3 conclusions). **In chat MUST show**: investigation summary,
performance comparison table, iteration history, >= 3 conclusions — then file paths.

## Anti-patterns (NEVER)

- NEVER optimize without a saved baseline.
- NEVER compare performance before precision passes.
- NEVER change the perf cases between baseline and optimized runs.
- NEVER violate the unified anti-patterns while optimizing.
- NEVER iterate past 3 rounds.
- NEVER reply with only file paths.
