---
id: P64
bottlenecks: [bus_contention]
op_families: [index_scatter]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P64: 避免 GM 同地址访问冲突 (GM Address Conflict Avoidance)

## 核心思想
MTE2/MTE3/Scalar 访问 GM 时按 512 字节粒度对齐处理，多核同时访问连续 512 字节范围内的地址会被串行处理。通过错位访问顺序或修改切分策略，使各核同一轮访问不同 512B 区域，消除多核 GM 地址冲突。

## 代码骨架

// === 改造后（专家模式）===
```cpp
// 反例：所有核同一轮访问相同地址范围
for (int i = 0; i < loopOneCore; i++) {
    DataCopy(dst, src[i * blockSize], blockSize);
}

// 正例：错位访问，各核偏移不同
for (int i = 0; i < loopOneCore; i++) {
    int newProgress = (i + GetBlockIdx()) % loopOneCore;
    DataCopy(dst, src[newProgress * blockSize], blockSize);
}
```

## 关键修改点

1. 预期收益: 消除多核 GM 地址冲突导致的串行等待，显著降低 MTE2/MTE3 时间

## 常见陷阱

⚠️ 可能需要 SyncAll 全核同步配合
⚠️ 切分策略变更可能影响其他优化
⚠️ 错位访问增加地址计算复杂度

## 代码搜索关键词

```bash
grep -n "SyncAll\|SetFlag\|WaitFlag\|PipeBarrier\|GetBlockNum\|coreNum\|blockIdx\|SplitCore" op_kernel/*.cpp op_host/*_tiling.cpp
```
