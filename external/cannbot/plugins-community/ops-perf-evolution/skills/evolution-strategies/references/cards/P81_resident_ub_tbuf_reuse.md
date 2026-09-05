---
id: P81
bottlenecks: [ub_memory_pressure]
op_families: [attention, normalization, optimizer]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P81: Resident UB/TBuf Buffer Reuse (UB/TBuf 常驻 Buffer 复用)

## 核心思想
将小尺寸参数、跨迭代状态、梯度累加器或迭代式中间结果一次性搬入或初始化到 UB/TBuf 常驻 buffer，并在整个 `Process` 生命周期或多层循环中反复复用。该策略把原本每个 loop/batch/iteration 都要发生的 GM↔UB 搬运、Broadcast、Cast 或临时分配，改写为循环外一次初始化、循环内片上读写，适用于数据生命周期长且尺寸可控的参数与状态数据。

## 代码骨架

// === 改造前（基线）===
```cpp
for (int64_t bIdx = 0; bIdx < baseB; ++bIdx) {
    for (int64_t sIdx = 0; sIdx < baseS; ++sIdx) {
        DataCopyPad(weightLocal, weightGm, copyParams, padParams);
        Cast(weightFp32, weightLocal, RoundMode::CAST_NONE, alignBaseH);
        Compute(xLocalFp32, weightFp32, y0Fp32, y1Fp32, y2Fp32);
    }
}
```

// === 改造后（专家模式）===
```cpp
__aicore__ inline void Process() {
    LocalTensor<float> weightFp32 = this->inQueueW.template AllocTensor<float>();
    DataCopyPad(weightLocal, weightGm, copyParams, padParams);
    Cast(weightFp32, weightLocal, RoundMode::CAST_NONE, alignBaseH);

    for (int64_t bIdx = 0; bIdx < baseB; ++bIdx) {
        for (int64_t sIdx = 0; sIdx < baseS; ++sIdx) {
            Compute(xLocalFp32, weightFp32, y0Fp32, y1Fp32, y2Fp32);
        }
    }
    this->inQueueW.FreeTensor(weightFp32);
}
```

## 关键修改点

1. 预期收益: 小参数只搬运和 Cast 一次，MTE2 带宽留给主数据 tile，循环越深收益越明显。

## 常见陷阱

⚠️ 常驻 buffer 会长期占用 UB/TBuf，直接压缩主数据 tile 和双缓冲可用空间
⚠️ TBuf/VECCALC buffer 不受队列同步保护，需要手动使用 PipeBarrier 或明确阶段边界保证一致性
⚠️ 初始化阶段可能增加一次 Cast、Broadcast、Duplicate 或 Transpose 开销，仅在复用次数足够多时收益明显

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
