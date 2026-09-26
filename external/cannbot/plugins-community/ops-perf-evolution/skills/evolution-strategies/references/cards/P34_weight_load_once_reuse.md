---
id: P34
bottlenecks: [bus_contention, mte2_stall]
op_families: [normalization, matmul]
complexity: L1
conflicts_with: []
synergizes_with: [P41, P8]
has_preconditions: true
has_playbook: true
---

# P34: Weight 常驻与预计算复用

## 核心思想
将 weight/scale 等小尺寸、跨 loop 不变的数据在 Process 开头搬入 UB 并 cast 到 FP32，之后在整个 B×S 循环中反复使用。对于需要广播的 weight，还可在搬入后做 Broadcast 扩展到目标形状常驻 UB（见 Variant B）。

## 代码骨架

// === 改造后（专家模式）===
```cpp
__aicore__ inline void Process() {
    LocalTensor<float> weightFp32 = this->inQueueW.template AllocTensor<float>();
    DataCopyPad(weightLocal, weightGm, copyParams, padParams);
    Cast(localW0FP32, localW0, RoundMode::CAST_NONE, this->alignBaseH);
    Cast(localW1FP32, localW1, RoundMode::CAST_NONE, this->alignBaseH);
    for (int64_t bIdx = 0; bIdx < this->baseB; ++bIdx) {
        for (int64_t sIdx = 0; sIdx < this->baseS; ++sIdx) {
            Compute<DTYPE>(xLocalFp32, weightFp32, y0Fp32, y1Fp32, y2Fp32, ...);
        }
    }
    this->inQueueW.FreeTensor(weightFp32);
}
```

## 关键修改点

1. 预期收益: 消除 weight 重复搬运开销，MTE2 带宽留给数据 tile

## 常见陷阱

⚠️ weight buffer 在整个 Process 期间常驻 UB，减少可用于数据 tile 的空间
⚠️ 需要在 Process 开头额外做一次 Cast（如 half→float），增加初始化延迟
⚠️ Broadcast 预计算需要额外 UB 空间（原始 + Cast + Broadcast 三份），仅适用于单维度广播

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|GetBlockNum\|coreNum\|blockIdx\|SplitCore\|DataCopy" op_kernel/*.cpp op_host/*_tiling.cpp
```
