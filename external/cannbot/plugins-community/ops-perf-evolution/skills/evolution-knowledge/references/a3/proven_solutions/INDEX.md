# Proven Solutions Registry

## Purpose

Record successful optimization techniques discovered during evolution runs.
Enables cross-operator knowledge transfer: if a technique worked for operator A,
it may work for a similar operator B.

## Format

Each entry should follow this template:

```markdown
### {technique_name}

**Source**: {op_name} evolution, round {r}, parallel {p}
**Speedup**: {before}x → {after}x ({improvement_pct}% improvement)
**Operator type**: {elementwise | reduction | attention | matmul | ...}
**Hardware**: {chip_model}

**Technique description**:
{1-3 sentences describing what was done}

**Key code change**:
```cpp
// Before:
{key code snippet before optimization}

// After:
{key code snippet after optimization}
```

**Applicability conditions**:
- {condition 1}
- {condition 2}

**Strategy IDs**: {P1, P2, ...} or {X-series if novel}
```

## 按算子类型索引

| 算子类型 | 已有方案 | 来源 |
|---------|---------|------|
| attention | X1: Causal S2 Block Skipping | FlashAttentionSimple evolution |
| elementwise, reduction, normalization | X2: Scalar-to-Vector Conversion | NMS + GroupNorm + TopK evolution |
| sort, topk, nms | X3: Hardware Sort Engine Replacement | NMS + TopK evolution |
| reduction | X4: Hybrid Scalar/Vector Dispatch | GroupNorm evolution |

_新方案沉淀时按此格式追加行_

## Recorded Solutions

### X1: Causal S2 Block Skipping (FlashAttentionSimple)

**Source**: FlashAttentionSimple evolution, arch_round8
**Speedup**: 1.003x → 1.83x (45.5% latency reduction at S=32K)
**Operator type**: attention
**Hardware**: Ascend 910B3

**Technique description**:
In causal attention, entire S2 blocks where all entries are masked (below the diagonal)
are skipped entirely — no cube matmul, no vector post-processing, no DMA.
The valid S2 range is computed per S1 block on the host side using the causal constraint:
`valid_s2_end = s1_end` (the last row of the current Q block determines the furthest valid K column).

**Key code change**:
```cpp
// Before: iterate over all S2 blocks
for (int s2 = 0; s2 < totalS2Blocks; s2++) { ProcessBlock(s1, s2); }

// After: skip invalid blocks
int validS2End = (s1BlockEnd) / blockSize + 1;
for (int s2 = 0; s2 < min(validS2End, totalS2Blocks); s2++) { ProcessBlock(s1, s2); }
```

**Applicability conditions**:
- Attention operator with causal (lower-triangular) mask
- Block-tiled computation (not element-wise attention)
- Sequence length > 512 (smaller sequences have few blocks to skip)

**Strategy IDs**: X-series (novel, discovered during evolution)

---

---

### X2: Scalar-to-Vector Conversion (NMS + GroupNorm + TopK)

**Source**: NMS (30_NMS) evolution, GroupNorm (11_GroupNorm_evo) P0, TopK (TopK_evo_0629) Round 1-3
**Speedup**: 11.25x (NMS), 1.84x geomean (GroupNorm fp32: 0.50x→, fp16: 1.93x, bf16: 4.31x), 4.08x (TopK, from 0.66x)
**Operator type**: elementwise, reduction, normalization, sort, topk, nms
**Hardware**: Ascend 910B2C

**Technique description**:
TileLang-generated AscendC code frequently contains scalar for-loops that bypass the NPU Vector unit entirely. The systematic replacement of these patterns with AscendC Vector API calls — ReduceSum for accumulation loops, Adds/Muls chains for element-wise math, Sort32/MrgSort for sorting/selection, and zero-copy sub-tensor slicing for batch operations — is the single highest-impact optimization for generated kernels. A hybrid dispatch (vector path for large workloads, scalar path for small ones) prevents regression on edge cases.

**Key code change**:
```cpp
// Pattern A: Scalar reduction → ReduceSum (+2-5x)
// Before: for(i) sum += x[i]; sum_sq += x[i]*x[i];
// After:
AscendC::ReduceSum<float, true>(scalarOut, xTensor, reduceTmp, count);

// Pattern B: Scalar element-wise → Adds/Muls chain (+3-10x)
// Before: for(i) y[i] = (x[i] - mean) * inv_std * w + b;
// After:
AscendC::Adds<float>(y, x, -mean, n);
AscendC::Muls<float>(y, y, inv_std, n);
AscendC::Muls<float>(y, y, w_val, n);
AscendC::Adds<float>(y, y, b_val, n);

// Pattern C: Scalar sort → Sort32/MrgSort (+5-100x for large n)
// Before: O(n²) bubble sort or O(nk) selection sort
// After:
AscendC::Sort<float, true>(vals, vals, indices, tmpBuf, n / 64);
```

**Applicability conditions**:
- Kernel contains `for` loops with scalar arithmetic (detected by grep for `for.*++.*sum\s*+=` patterns)
- Reduction dimension ≥ 64 elements (smaller: scalar overhead negligible)
- Element-wise operations ≥ 3 in chain (fewer: vector launch overhead may dominate)
- Sort/selection n ≥ 32 (smaller: scalar selection sort is acceptable)

**Strategy IDs**: X2 (cross-operator validation: works on NMS, GroupNorm, TopK; pattern abstracted to P85-P89 strategy cards)

### X3: Hardware Sort Engine Replacement (NMS + TopK)

**Source**: NMS (30_NMS) sort optimization, TopK (TopK_evo_0629) Round 1
**Speedup**: NMS case3: 2181x, TopK: 0.66x → 4.08x (hardware sort is key contributor)
**Operator type**: sort, topk, nms (any operator with sorting/selection)
**Hardware**: Ascend 910B2C

**Technique description**:
AscendC provides hardware-accelerated Sort/Sort32/MrgSort APIs executed on the Vector unit. These replace any scalar sorting or top-k selection algorithm (bubble sort, selection sort, heap-based top-k) with O(n log² n) bitonic sort hardware execution. Key patterns: (1) Negate → Sort ascending → Negate for descending sort; (2) Sort32(repeat=BATCH_ROWS) for batch-parallel small-row sorting; (3) MrgSort for tiled large-row scenarios; (4) Sort + Extract for top-k from sorted results.

**Key code change**:
```cpp
// NMS: Negate → Sort → Negate for descending
for (int32_t k = 0; k < M; k++) svScore(k) = -svScore(k);
AscendC::Sort<float, true>(svScore, svScore, orderU32, tmpF, SORT_SZ / 64);
for (int32_t k = 0; k < M; k++) svScore(k) = -svScore(k);

// TopK: Sort32(repeat=B) batch parallel for small rows
AscendC::Sort<float, true>(packedBuf, packedBuf, orderBuf, tmpBuf, 
                           curBatch * rowLen / 64);
```

**Applicability conditions**:
- Any kernel with sorting, top-k selection, or NMS-style threshold selection
- Element count ≥ 32 (below: scalar methods may be faster)
- Requires 2× data buffer + 1× index buffer in UB for Sort32

**Strategy IDs**: X3 (captured in P87 strategy card)

### X4: Hybrid Scalar/Vector Dispatch (GroupNorm)

**Source**: GroupNorm (11_GroupNorm_evo) P0 optimization
**Speedup**: eliminated regression cases; fp32 geomean improved from baseline 0.22x to 0.50x
**Operator type**: reduction, normalization (any operator with variable workload size)
**Hardware**: Ascend 910B2C

**Technique description**:
Vector API overhead (EnQue/DeQue pipeline synchronization) can dominate on small workloads. A hybrid architecture with an automatic dispatch threshold (e.g., spatial_size >= 128 → vector path, else → scalar path) ensures that small workloads don't suffer from vectorization overhead while large workloads get the full vector benefit. The scalar path uses AllocTensor → DataCopyPad → raw pointer → FreeTensor (no EnQue/DeQue), while the vector path uses DeQue → ReduceSum/Adds/Muls → EnQue.

**Key code change**:
```cpp
constexpr int32_t VECTOR_MIN_SPATIAL = 128;
if (count_spatial >= VECTOR_MIN_SPATIAL) {
    ProcessBatchedVectorized(...);  // EnQue/DeQue + Vector API
} else {
    ProcessBatchedScalar(...);      // AllocTensor + pointer + FreeTensor
}
```

**Applicability conditions**:
- Operator has variable per-tile workload size (e.g., different spatial dimensions)
- Vector path uses EnQue/DeQue (synchronization overhead ~tens of cycles)
- Requires profiling to tune the threshold for specific hardware

**Strategy IDs**: X4 (captured in P89 strategy card)

---

_Add new entries below as they are discovered._
