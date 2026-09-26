---
id: P18
bottlenecks: [mte2_stall, partial_overlap]
op_families: [attention, cv_fusion, flash_attention]
complexity: L2
conflicts_with: []
synergizes_with: [P14, P53, P76]
has_preconditions: true
has_playbook: true
---

# P18: L1 7-Buffer Resident Partitioning (L1 七缓冲常驻分区)

## 核心思想
针对 Cube 核上 GM→L1→L0→Cube 的多级数据搬运流水线，将 L1 总容量（512KB）静态划分为 7 个等大小块（每块 72KB）：QP 区 4 块（2×2 矩阵布局）用于存放 Query/Softmax-P 数据，KV 区 3 块（三路旋转）用于存放 Key/Value 数据。QP 区采用"常驻"策略——仅在 N 方向首次迭代时从 GM 加载，后续 N 迭代直接复用 L1 中已有数据，消除冗余搬运；KV 区采用三路旋转缓冲，比双缓冲多一个空闲槽位，使 GM→L1 加载、L1→L0 消费、下一轮预加载三阶段完全重叠。该方案源自 FIA（Fused Infer Attention）MLA 高性能模板，专为 headDim=512+rope=64 的 MLA 场景设计。该方案同时支持 MTE2→L1→L0→Mmad 三级全流水调度（见 Variant D）。

## 代码骨架

// === 改造前（基线）===
```cpp
// 基线方案：QP 双缓冲，每次 N 迭代都重新加载 Q
static constexpr uint32_t L1_QP_SIZE = 128 * 1024;
pipe->InitBuffer(qpBufL1, L1_QP_SIZE * 2); // 256KB，仅 2 块

// 每次 N 迭代都要重新搬运 Q 数据
for (uint32_t nStart = 0; nStart < nSize; nStart += N_SPLIT_SIZE) {
    if (nStart == 0) {
        WaitFlag<HardEvent::MTE1_MTE2>(QP_EVENT0 + qpBufId % 2);
        CopyQGmToL1(...); // 每次都搬运
        SetFlag<HardEvent::MTE2_MTE1>(QP_EVENT0 + qpBufId % 2);
    }
    // nStart > 0 时仍需切换 buffer，无法常驻
}
```

// === 改造后（专家模式）===
```cpp
// L1 buffer 定义：QP 4块，每块 72KB
static constexpr uint32_t L1_BLOCK_SIZE = M_L1_SPLIT_SIZE * MM1_K_L1_SPLIT_SIZE * sizeof(Q_T);
// = 128 * 288 * 2 = 73728 bytes (72KB)

// 初始化：分配 4 块连续 L1 空间
pipe->InitBuffer(bufQPL1, L1_BLOCK_SIZE * 4); // 288KB
l1QPTensor = bufQPL1.Get<Q_T>();

// 2×2 索引映射：确保同一 M 块的两个 K 半块地址连续
__aicore__ inline uint32_t GetQPL1RealIdx(uint32_t mIdx, uint32_t k1Idx) {
    uint32_t idxMap[] = {0, 2}; // block 0,1 连续; block 2,3 连续
    return idxMap[mIdx % 2] + k1Idx;
}

// N 方向循环中的常驻复用
for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) {
    for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) {
        uint32_t ka = GetQPL1RealIdx(mIdx, kL1);
        LocalTensor<Q_T> aL1Tensor = l1QPTensor[ka * L1_BLOCK_OFFSET];

        if (nL1 == 0) {
            // 首次 N 迭代：从 GM 加载 Q 数据到 L1
            WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]);
            copyQueryGmToL1(dstTensor, queryGmTensor, gmCoord);
            // Q_rope 紧接 Q_nope 之后存放
            copyQueryGmToL1(dstRopeTensor, queryRopeGmTensor, gmCoordQRope);
            SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
            WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
        }
        // nL1 > 0 时：直接复用 L1 中已有的 Q 数据，零搬运开销
// ... (truncated)
```

## 关键修改点

1. 用细粒度同步替代粗粒度 SyncAll
2. 预期收益: 消除 N 方向内循环中 Q/P 数据的冗余 GM→L1 搬运，搬运次数从 O(N_loops) 降为 O(1)；对于 s2BaseSize=512、N_L1_S...

## 常见陷阱

⚠️ L1 占用率极高（504KB/512KB），几乎无剩余空间，不适合需要额外 L1 临时缓冲的场景
⚠️ 7 个 buffer 需要 7 个独立的 MTE1_MTE2 硬件事件 ID 进行同步管理，事件资源紧张
⚠️ 切分尺寸（M=128, K=288, N=128）为 MLA headDim=576 特化，泛化到其他 headDim 需要重新计算分区

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
