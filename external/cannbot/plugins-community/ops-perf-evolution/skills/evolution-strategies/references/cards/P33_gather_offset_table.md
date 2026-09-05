---
id: P33
bottlenecks: [mte2_stall, undersize_transfer]
op_families: [pooling_gather]
complexity: L1
conflicts_with: []
synergizes_with: [P32]
has_preconditions: true
has_playbook: true
---

# P33: Gather 偏移表驱动的结构化数据提取

## 核心思想
通过预计算字节偏移表，使用 Gather 指令一次性从混合矩阵中提取多个子张量，偏移表只计算一次常驻 UB。

## 代码骨架

// === 改造后（专家模式）===
```cpp
// 预计算偏移表（Init 阶段，只执行一次）
for (uint32_t i = 0; i < V1_BASE_T; i++) {
    for (uint32_t j = 0; j < N_; j++)
        preOffsetBuf_.SetValue(offset1++, curOffset * sizeof(P));
    curOffset += N_;
    for (uint32_t j = 0; j < N_; j++)
        postOffsetBuf_.SetValue(offset2++, curOffset * sizeof(P));
    curOffset += nSquare;
}
Gather(hPreBuff_, matmulRes_, preOffsetBuf_, 0, lenT * N_);
Gather(hPostBuff_, matmulRes_, postOffsetBuf_, 0, lenT * N_);
```

## 关键修改点

1. 预期收益: 偏移表只计算一次，后续 Gather 提取无需逐元素寻址

## 常见陷阱

⚠️ 偏移表占用 UB 空间（元素数 × 4 字节），子张量数量多时占用显著
⚠️ offset 必须是字节偏移且 Gather 要求源数据在 UB 中连续
