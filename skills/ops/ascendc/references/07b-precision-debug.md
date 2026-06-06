# Phase 7b — Precision root-cause debugging

Use when precision fails (allclose fails, large deviation, all-zero/NaN) and threshold
tuning cannot fix it. Five stages, shallow to deep: **look at the data, then the code,
then isolate by experiment, then instrument.**

```
1 Error analysis → 2 Code review → 3 Experiment isolation → 4 Instrumentation → 5 Fix + verify
```

## Stage 1 — Error analysis (look at data first)

From [`../scripts/debug_precision_template.py`](../scripts/debug_precision_template.py)
create `csrc/ops/<op>/test/debug_<op>_precision.py` and run it to get: error stats
(MaxAbsErr/MeanAbsErr/MaxRelErr), first wrong element (coords + linear index + npu vs
ref), error distribution (count/ratio, periodic spacing), special values (all-zero,
NaN/Inf), and fixed-vs-random / shrink-shape comparisons.

| Symptom | Most likely cause | Next |
|---|---|---|
| FP16 fails, FP32 passes | **no FP32 up-cast** | Stage 2: check Cast |
| all-zero output | CopyOut not run / GM offset wrong | Stage 2: check CopyOut |
| NaN/Inf | divide-by-zero / log of negative / overflow | Stage 2: check Compute |
| all values off, CosineSim≈1 | systematic precision loss | Stage 2: check up-cast |
| periodic/striped errors | tile boundary / copy offset | Stage 3 |
| only tail elements wrong | tail-tile length / alignment | Stage 2: tail tile |
| different result each run | insufficient sync | Stage 3 exp B |
| small shape ok, large fails | multi-core / tiling boundary | Stage 3 exp A |
| fixed input ok, random fails | address/stride/offset | Stage 3 exp C |

## Stage 2 — Code review (MANDATORY: read op_host, op_kernel, design.md)

Layer 1 (most frequent): FP16/BF16 not up-cast to FP32; wrong formula/API order; GM
offset unit confusion (`xGm[progress*tileLength]` is element offset — don't multiply by
`sizeof(T)`); `tileLength` (offset) vs `curTileLength` (compute/copy).

Layer 2 (copy/align): `DataCopyExtParams` copyLen is bytes = `curTileLength*sizeof(T)`;
tail-tile `alignedTailLen`; multi-input offsets when shapes differ.

Layer 3 (tiling/multi-core): host/kernel tiling symbol consistency; inter-core coverage
(`formerNum*formerLength + tailNum*tailLength == total`); `bufferCoefficient` vs UB table.

Layer 4 (API traps): Reduce modifies source (backup with `Adds`); Alloc/Free + EnQue/DeQue
pairing; vector length is element count, not bytes.

Layer 5 (boundaries): divide-by-zero / domain (Div/Reciprocal/Ln/Sqrt); int32 overflow in
tiling math → use int64_t.

Output a ranked suspect list. If root cause is clear, go to Stage 5; else Stage 3.

## Stage 3 — Experiment isolation (change one variable at a time)

- **A — blockDim → 1**: hardcode single core (optionally shrink shape). Single ok /
  multi fails → inter-core (GM overlap / tiling map / sync). Single also fails → exp B.
- **B — `PipeBarrier<PIPE_ALL>`** between CopyIn/Compute/CopyOut. Passes → intra-core sync
  missing (restore fine-grained sync to locate). Still fails → exp C. (`PIPE_ALL` is
  **experiment only, never the final fix**.)
- **C — fixed/regular input** (all-ones / `arange` / random). Ones ok, arange/random fail
  → address/offset/stride. All fail → compute or global tiling. All pass → value-range
  precision issue (check extremes).
- **D — shrink shape** `(32,)` → `(tileLength,)` → `(tileLength*2,)` → original; find the
  exact failing boundary to infer tile/core edges.

First-error index → which tile → which core → that core's GM start → expected copy bytes.
Period = tileLength → copy/offset; period = vector width → compute; aligned to core edge →
multi-core/offset.

## Stage 4 — Instrumentation (`printf` / `DumpTensor`)

Rules: print **only core 0** (`if (AscendC::GetBlockIdx()==0)`); read `LocalTensor` only
**after** `DeQue`/`PipeBarrier`; cast FP16 to float before printing; use `DumpTensor`
desc to mark stage; start with a small `dumpSize`. Instrument step-by-step after `DeQue`
and compare each step to a hand-computed Python reference — the first deviating step is the
root cause. Full per-element compare is done **on host** (read GM + Python), not in-kernel.

## Stage 5 — Fix and verify

| Root cause | Fix |
|---|---|
| FP16 not up-cast | Cast(fp16→fp32) + compute + Cast(fp32→fp16) |
| GM offset wrong | fix element-vs-byte offset |
| tail-tile length | compute/copy use curTileLength; offset uses tileLength |
| tiling param | fix host tiling |
| sync missing | correct EnQue/DeQue or PipeBarrier |
| Reduce overwrote source | Adds backup before Reduce |
| copy length | fix DataCopyExtParams copyLen |

After fixing: remove all debug instrumentation (or wrap in `#ifdef DEBUG_PRECISION`);
rebuild/install; rerun the failing case + full precision suite. Still failing → back to
Stage 1 (max 3 rounds), then report to the user.

**Output (MANDATORY)**: show problem summary, root-cause analysis, the fix, verification
result, and >= 2 key lessons in chat. Never reply with only "fixed".

## Worked examples (load on demand, only when the symptom matches)

| Symptom | Example |
|---|---|
| FP16 fails / FP32 ok, all off | [`../examples/precision-debug/fp16-no-upcast.md`](../examples/precision-debug/fp16-no-upcast.md) |
| first error at tile boundary, period=tileLength | [`../examples/precision-debug/gm-offset-error.md`](../examples/precision-debug/gm-offset-error.md) |
| only a few tail elements wrong | [`../examples/precision-debug/tail-tile-misalign.md`](../examples/precision-debug/tail-tile-misalign.md) |
| blockDim=1 ok, multi-core fails | [`../examples/precision-debug/multicore-tiling-overlap.md`](../examples/precision-debug/multicore-tiling-overlap.md) |
| different result each run | [`../examples/precision-debug/async-sync-missing.md`](../examples/precision-debug/async-sync-missing.md) |

## Anti-patterns (NEVER)

- NEVER edit code before analyzing the error distribution.
- NEVER `printf` a full tensor in a loop in-kernel — use DumpTensor or host compare.
- NEVER print from all cores at once.
- NEVER read a `LocalTensor` before sync.
- NEVER ship `PIPE_ALL` as the final fix.
- NEVER leave debug code in after the fix.
- NEVER fix only the known failing case without rerunning the full suite.
- NEVER keep trying past 3 rounds — report to the user.
