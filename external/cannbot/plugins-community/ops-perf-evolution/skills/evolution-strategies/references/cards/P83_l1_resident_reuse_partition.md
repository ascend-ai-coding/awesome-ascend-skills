---
id: P83
bottlenecks: [partial_overlap]
op_families: [matmul]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P83: L1 Resident Reuse with Multi-Buffer Partitioning (L1 常驻复用与多 Buffer 分区)

## 核心思想
在 FlashAttention、IFA、MLA 等 Cube 密集型算子中，可将 Q 或左矩阵在首次迭代时搬入 L1 后常驻，并结合 L1 多 buffer 分区，把“跨 N/S2 迭代复用”与“多阶段预取/消费流水”统一起来。该策略的核心不是固定采用某个 buffer 数量，而是先判断左矩阵是否在后续迭代保持不变，再依据 L1 容量把空间划分为常驻区与轮转区：常驻区负责保存跨迭代复用的 Q/左矩阵，轮转区负责承载 KV 或后续分块的预取与消费。若 shape 与 L1 预算允许，可进一步落到 MLA 类场景常见的 4 个 QP buffer + 3 个 KV buffer 的 7-buffer 分区。

## 代码骨架

// === 改造前（基线）===
```cpp
for (uint32_t n = 0; n < nloops; n++) {
    CopyQToL1(info, mL1Size);
    // ... 计算 ...
}
```

// === 改造后（专家模式）===
```cpp
__aicore__ inline LocalTensor<TRANS_T> LoadData(int curRow, int curCol,
    int tileHeight, int tileWidth, int batchNum = -1) {
    LocalTensor<TRANS_T> l1;
    UserDefDataType flag = MATMUL_PARAM_VAR.dataPtr_;
    LocalTensor<TRANS_T> dst;
    if (flag.reuseLeft) {
        // 非首轮：直接复用 L1 中已有的 Q/左矩阵数据
        l1 = MATMUL_MODULE(CubeInBuffer)->GetBuffer(flag.leftBufIdx);
        dst = l1[static_cast<int64_t>(callTimes_) * baseWidth_ * 64];
        ++callTimes_;
        return dst;
    } else {
        // 首轮：从 GM 加载 Q/左矩阵数据到 L1
        l1 = MATMUL_MODULE(CubeInBuffer)->AllocTensor(flag.leftBufIdx);
        dst = l1[static_cast<int64_t>(callTimes_) * baseWidth_ * 64];
        // ... 执行 DataCopy nd2nz ...
    }
    ++callTimes_;
    return dst;
}
```

## 关键修改点

1. 预期收益: 将 Q/左矩阵搬运次数从 O(N_loops) 降到 O(1)，跨 S2/N 方向循环越深收益越明显。

## 常见陷阱

⚠️ 左矩阵常驻会长期占用 L1，大常驻期间可分配给其他中间结果或 P buffer 的空间显著减少；例如 headDim=576、M=128 时仅 Q 常驻就约需 144KB
⚠️ 多 buffer 分区通常把 L1 利用率推到极高水平；以 MLA 7-buffer 分区为例，7×72KB 约占 504KB/512KB，迁移到其他 headDim 或算子形状时需要重新计算切分
⚠️ 常驻区与轮转区需要独立索引和事件同步，代码复杂度明显高于普通双缓冲

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|GetBlockNum\|coreNum\|blockIdx\|SplitCore\|DataCopy" op_kernel/*.cpp op_host/*_tiling.cpp
```
