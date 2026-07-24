---
id: P53
bottlenecks: [partial_overlap]
op_families: [cv_fusion, flash_attention]
complexity: L1
conflicts_with: []
synergizes_with: [P14, P18, P38]
has_preconditions: true
has_playbook: true
---

# P53: L1 常驻复用策略 (L1 Resident Reuse)

## 核心思想
在 FlashAttention/IFA 等 Cube 密集型算子中，将 Q 矩阵（左矩阵）在首次 N 方向迭代时加载到 L1 后常驻，后续 N 迭代直接复用 L1 中已有数据，消除冗余搬运。分为"大常驻"（跨多个 S2 循环复用）和"小常驻"（单基本块内部复用）两种模式。该策略与 P17 (L1 7-buffer resident partition) 互补，P17 侧重多 buffer 分区管理，本策略侧重单 buffer 的跨迭代复用逻辑。

## 代码骨架

// === 改造前（基线）===
```cpp
// 基线：每次 S2 迭代都重新加载 Q
for (uint32_t n = 0; n < nloops; n++) {
    CopyQToL1(info, mL1Size);  // 每次 N 迭代都搬运
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
        // 非首轮：直接复用 L1 中已有的 Q 数据
        l1 = MATMUL_MODULE(CubeInBuffer)->GetBuffer(flag.leftBufIdx);
        dst = l1[static_cast<int64_t>(callTimes_) * baseWidth_ * 64];
        ++callTimes_;
        return dst;
    } else {
        // 首轮：从 GM 加载 Q 数据到 L1
        l1 = MATMUL_MODULE(CubeInBuffer)->AllocTensor(flag.leftBufIdx);
        dst = l1[static_cast<int64_t>(callTimes_) * baseWidth_ * 64];
        // ... 执行 DataCopy nd2nz ...
    }
    ++callTimes_;
    return dst;
}
```

## 关键修改点

1. 预期收益: 消除 N 方向内循环中 Q 数据的冗余 GM→L1 搬运，搬运次数从 O(N_loops) 降为 O(1)；对于 s2BaseSize=512、N_L1_SPL...

## 常见陷阱

⚠️ 大常驻期间 L1 空间被 Q 占用，BMM2 阶段无法释放供 P 使用
⚠️ M 方向 > 128 时不支持大常驻（128×576×2 = 144KB）
⚠️ C1 和 C2 的首次循环需要加载的数据量大，可能造成断流

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|GetBlockNum\|coreNum\|blockIdx\|SplitCore\|DataCopy" op_kernel/*.cpp op_host/*_tiling.cpp
```
