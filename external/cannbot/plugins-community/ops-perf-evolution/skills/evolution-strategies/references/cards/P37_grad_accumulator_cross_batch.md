---
id: P37
bottlenecks: [ub_memory_pressure]
op_families: [optimizer, elementwise]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P37: 梯度累加器跨 batch 常驻

## 核心思想
gradWeight 需要在所有 batch 和 seq 上累加。使用 TBuf 分配常驻 UB，在整个 B×loopS 循环中持续累加，最后一次性 ReduceSum 并搬出。

## 代码骨架

// === 改造后（专家模式）===
```cpp
TBuf<QuePosition::VECCALC> gradWeightBuf;
void Process() {
    LocalTensor<float> gradWeight = gradWeightBuf.Get<float>();
    Duplicate(gradWeight, 0.0f, baseS * calNum * 3);
    PipeBarrier<PIPE_V>();
    for (int64_t idxB = 0; idxB < batchSize; idxB++) {
        for (int64_t idxS = 0; idxS < loopS; idxS++) {
            GradWeightConv(sTileLen);
        }
    }
    SumAndCopyOutGradWeight();
}
```

## 关键修改点

1. 用细粒度同步替代粗粒度 SyncAll
2. 预期收益: 避免每个 batch 单独搬出梯度再在 GM 累加，减少 MTE3 搬运量

## 常见陷阱

⚠️ gradWeightBuf 是 UB 中最大的常驻 buffer 之一，baseS 的 tiling 直接受此约束
⚠️ 最终需要 ReduceSum 搬出，batch 数很大时累加精度可能需要 FP32 中间结果

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|DataCopy" op_kernel/*.cpp op_host/*_tiling.cpp
```
