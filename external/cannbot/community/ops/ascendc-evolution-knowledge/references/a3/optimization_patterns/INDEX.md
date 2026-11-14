# Optimization Patterns Index

Proven AscendC optimization patterns with code examples. Each pattern includes:
- When to apply (operator characteristics)
- Code template
- Measured impact range
- Common mistakes

## Patterns

| File | Pattern | Impact | Applicability | 算子类型标签 |
|------|---------|--------|---------------|-------------|
| `double_buffering.md` | Double buffering for pipeline overlap | 20-80% speedup | All memory-bound kernels | elementwise, reduction, softmax |
| `tiling_strategies.md` | Adaptive tiling (Split-N, Split-D, Split-ND) | 10-50% speedup | Variable-shape operators | reduction, norm, attention |
| `causal_block_skip.md` | Skip invalid blocks in causal/masked attention | 20-50% reduction | Attention operators with causal mask | attention |
| `pipeline_overlap.md` | Fine-grained pipeline stage overlap | 5-30% speedup | Already double-buffered kernels | elementwise, reduction |
| `memory_coalescing.md` | Coalesced memory access patterns | 10-40% speedup | Operators with strided access | transpose, gather, attention |
| `shape_specialization.md` | TILING_KEY-based per-shape kernel variants | Eliminates shape interference; isolates optimizations | Multi-shape operators where best strategy diverges per shape | all (esp. norm/attention) |
| `scalar_to_vector.md` | Scalar computation → Vector API (ReduceSum/Adds/Muls/Sort32) | 2-11x geomean speedup | All operators with scalar for-loops | elementwise, reduction, normalization, sort, special |

## Quick Selection Guide

```
Is the kernel memory-bound?
  ├─ Yes: Start with double_buffering.md + tiling_strategies.md
  └─ No (compute-bound or balanced):
      ├─ Has causal/mask logic? → causal_block_skip.md
      ├─ Already has double buffering? → pipeline_overlap.md
      └─ Has strided access? → memory_coalescing.md

Multi-shape mode (target_shapes / generalization_shapes provided)?
  ├─ Any target shape regressed (gating == target_regression)?
  │   → shape_specialization.md (REFINE auto-injects P-ShapeSpec-01)
  ├─ shape_divergence ≥ 0.20 across target shapes?
  │   → shape_specialization.md (agent proactively chooses; utility +1.0 bonus)
  └─ Planning a global safety change (e.g., double buffer across all shapes)?
      → modify default branch + declare "default-safe"; evaluation runs
        --shapes-mode all and supervises generalization regression
```
