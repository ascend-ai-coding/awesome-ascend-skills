# AGENTS.md

## Project Purpose

Migrate GPU Triton kernels to Ascend 910_95 NPU and iteratively optimize performance. The agent follows a minimal-change, verify-after-each-step workflow.

## Target Hardware

| Parameter | Value |
|-----------|-------|
| Architecture | `dav-c310` (Reg-based) |
| AI Core | 1 Cube + 2 Vector |
| UB Capacity | 248 KB (256KB - 8KB reserved) |
| L0C Capacity | 256 KB |
| L1 Capacity | 512 KB |
| UB Alignment | 32B |
| L0C Alignment | 512B |

## First Action: Read Architecture Docs

Before any migration task, read these files to understand the 910_95 hardware:

1. `docs_ascendnpu_ir/00-Architecture/01-npu-hardware-overview.md`
2. `docs_ascendnpu_ir/00-Architecture/02-memory-hierarchy.md`
3. `docs_ascendnpu_ir/00-Architecture/03-pipeline-execution-model.md`
4. `docs_ascendnpu_ir/00-Architecture/04-data-layout.md`

## Migration Workflow

### Phase 1: Make It Run (minimal changes only)

Apply these 5 changes in order. Do NOT add any optimization at this stage. Even if `docs_for_triton_agent/` contains a complete migration example for the target kernel (e.g., Flash Attention, Fused MatMul), do NOT apply all the changes from the example at once — only apply the minimal changes needed to make the kernel compile and produce correct results.

1. **Device replacement**: `cuda` → `npu`, add `import torch_npu` before any triton import
2. **Remove GPU-only parameters**: delete `num_warps`, `num_stages`, `cache_modifier`, `eviction_policy`
3. **Data type replacement**: fp64 → fp32, uint series → int series
4. **Grid adjustment**: Grid size aligned to physical core count, prefer 1D
5. **Add accuracy verification**: add `if __name__ == "__main__"` block comparing Triton output against PyTorch reference

After Phase 1, invoke **ascend-triton-data-collector** subagent with the script path and target kernel names to verify correctness, extract IR, and collect baseline performance.

If accuracy check fails at this stage, the migration itself has a bug — fix it before proceeding.

### Phase 2: Iterative Optimization (one change at a time)

After baseline is established, iteratively identify and apply optimizations. Consult `docs_for_triton_agent/` to find optimization strategies matching the current bottleneck.

**Critical rule: Apply ONLY ONE optimization per iteration.** Even if `docs_for_triton_agent/` contains a complete migration example for the target kernel (e.g., Flash Attention, Fused MatMul), do NOT apply all changes at once. First apply only the minimal changes needed to pass accuracy verification, then add optimizations one at a time. If accuracy fails or performance regresses, it must be clear which change caused it.

For each iteration:

```
1. Analyze msprof data and IR from ascend-triton-data-collector to identify the biggest bottleneck
2. Consult docs_for_triton_agent/ to find ONE optimization that addresses the bottleneck
3. Apply the optimization to the kernel code
4. Invoke ascend-triton-data-collector subagent → verify accuracy + collect IR + profile performance
5. If accuracy fails OR performance regresses:
   a. Revert the change
   b. Analyze the failure — can the optimization be corrected or adjusted?
   c. If yes → apply the corrected version, re-verify
   d. If no → move on to the next optimization candidate
6. If improved → proceed to next bottleneck
```

Stop iterating when: all key metrics are in healthy range, or no remaining optimization improves performance, or accuracy cannot be maintained after correction attempts.

### Phase 3: 910_95 Specific (after Phase 2 converges)

After Phase 2 converges, consult `docs_for_triton_agent/` for 910_95-exclusive optimizations (L0C→UB direct path, FP8, SIMT mode, etc.). Apply the same one-at-a-time rule with analysis on failure.

## Accuracy Verification Template

Every migrated script must include an `if __name__ == "__main__"` block:

```python
if __name__ == "__main__":
    dtype = torch.float16
    x = torch.randn((1024, 1024), dtype=dtype, device="npu").requires_grad_()

    # --- Triton kernel ---
    y = YourTritonFunction.apply(x, ...)
    dy = torch.randn_like(y)
    y.backward(dy)
    triton_dx, x.grad = x.grad.clone(), None

    # --- PyTorch reference ---
    ref = your_pytorch_reference(x, ...)
    ref.backward(dy)
    torch_dx, x.grad = x.grad.clone(), None

    # --- Compare ---
    atol, rtol = 1e-3, 1e-3
    if torch.allclose(y, ref, atol=atol, rtol=rtol):
        print("✅ [Fwd]Triton and Torch match")
    else:
        print("❌ [Fwd]Triton and Torch differ")
    if torch.allclose(triton_dx, torch_dx, atol=atol, rtol=rtol):
        print("✅ [Bwd]Triton and Torch match")
    else:
        print("❌ [Bwd]Triton and Torch differ")
```

Key rules:
- Reference must be pure PyTorch (no Triton calls)
- Clone gradients immediately after `.backward()`, reset `x.grad` to `None`
- Use `atol=1e-3, rtol=1e-3` for fp16/bf16; `atol=1e-4, rtol=1e-4` for fp32
- The `✅`/`❌` format is important — data collector reports these

## Performance Analysis Guide

After each ascend-triton-data-collector run, **prioritize IR analysis** over msprof metrics. IR reveals the actual computation structure, tiling strategy, and pipeline overlap; msprof ratios alone can be misleading (e.g., high scalar_ratio may be hidden by pipeline overlap and not a real bottleneck).

### Step 1: IR Analysis (primary)

Examine the compiler IR output to identify:
- **Scalar-heavy patterns**: excessive `arith` ops (int64 comparisons, type conversions) that are NOT absorbed by pipeline overlap
- **Memory access patterns**: non-continuous loads/stores, redundant copies, misaligned tiling
- **Pipeline structure**: whether Vector and Cube ops are properly overlapped; whether scalar ops sit on the critical path or are hidden behind data transfers
- **Sync/barrier density**: unnecessary barriers between independent operations

### Step 2: msprof Metrics (supplementary, validate IR findings)

| Metric | Ideal | Caveat |
|--------|-------|--------|
| `aiv_vec_ratio` | > 80% | Low ratio may indicate Vector pipeline stalled OR that Cube/transfer dominates — cross-check with IR |
| `aiv_mte2_ratio` | < 50% | High ratio confirms memory bottleneck seen in IR |
| `aiv_scalar_ratio` | < 20% | **High ratio does NOT always mean scalar-bound** — if scalar ops are hidden by pipeline overlap (MTE2 wait time covers scalar execution), they are not the bottleneck. Only treat as scalar degradation when IR confirms scalar ops are on the critical path |
| `aic_cube_ratio` | > 80% | Low ratio may indicate Cube pipeline stalled — cross-check with IR tiling |
| `aic_mte1_ratio` | moderate | High ratio confirms L1→L0A/L0B transfer bottleneck seen in IR |

### Bottleneck → Action

| Bottleneck | Action |
|------------|--------|
| Compute-bound (high vec/cube ratio) | Increase tiling, CV fusion |
| Memory-bound (high mte2 ratio) | MultiBuffer, care_padding=False, continuous access |
| Scalar degradation (IR-confirmed critical-path scalar ops) | Type conversion (int64→int32, int cmp→fp32 cmp), where→get_element+insert_slice |
| Pipeline-hidden scalar (high scalar_ratio but IR shows overlap) | No action needed — scalar ops are masked by data transfer latency |
| Sync overhead | sync_solver, reduce barrier count |

## Knowledge Base Hierarchy

Consult documentation in this priority order:

### Priority 1: Agent Migration Guide (problem-oriented, code-driven)

Path: `docs_for_triton_agent/`

| Scenario | Document |
|----------|----------|
| Hardware specs / memory / alignment | `00-hardware-quick-ref.md` |
| First migration / architecture differences | `01-migration-overview.md` |
| API not working / need alternatives | `02-api-differences.md` |
| BLOCK_SIZE / Grid contraction | `03-tiling-and-grid.md` |
| Memory access optimization | `04-memory-access-patterns.md` |
| tl.dot + vector ops co-optimization | `05-cv-pipeline-optimization.md` |
| Performance far below expected | `06-scalar-degradation-avoidance.md` |
| multibuffer / enable_mixed_cv / sync_solver | `07-compile-params.md` |
| Type conversion / precision | `08-data-type-precision.md` |
| tl.make_block_ptr | `09-block-pointer-migration.md` |
| Autotune on NPU | `10-autotune-on-npu.md` |
| tl.dot + bias fusion | `11-fixpipe-and-bias-fusion.md` |
| Multiple tl.store merge | `12-store-merge.md` |
| Single-position tl.where | `13-where-optimization.md` |
| compile_hint / extension APIs | `14-compile-hint-and-extension.md` |
| Cube-Vector synchronization | `15-sync-and-barrier.md` |
| Compilation / runtime errors | `16-debugging-common-errors.md` |
| Flash Attention migration | `17-flash-attention-migration.md` |
| Fused MatMul migration | `18-fused-matmul-migration.md` |
| Fused SwiGLU migration | `19-fused-swiglu-migration.md` |
| RoPE migration | `20-rope-migration.md` |
| Softcap migration | `21-softcap-migration.md` |
| Advanced optimization | `22-advanced-optimization.md` |

### Priority 2: Triton-Ascend Developer Docs (API reference, compilation flow)

Path: `docs_triton_ascend/`

Consult when Priority 1 docs don't cover the detail needed: API parameters, compilation pipeline internals, extension usage examples.

### Priority 3: AscendNPU-IR Compiler Docs (IR semantics, pass definitions)

Path: `docs_ascendnpu_ir/`

Consult when analyzing extracted IR or understanding compiler behavior: HIVM dialect operations, pass semantics, memory management, data layout.
