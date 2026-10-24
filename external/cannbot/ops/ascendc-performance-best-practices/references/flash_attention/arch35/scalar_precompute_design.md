# Scalar Precompute 优化设计

> 适用架构：arch35（A5 / DAV_3510）；具体型号、SDK 与验证范围以正文为准。

## 1. 优化目标

针对 FIA decode 路径中 SCALAR ALU 主导的地址计算、除法取模、循环不变量重复计算问题，将可提前确定的乘法、除法和公共前缀移到热循环外，减少每个 `kvSIdx × kvNIncreIdx` 迭代中的 Scalar 指令。

适用判定：`SCALAR% >= 80%` 且 `SCALARLDST% < SCALAR%`。如果 `SCALARLDST%` 更高，应优先使用 `scalarldst_subbound_design.md`。

## 2. 架构概览

FIA decode 热路径结构：

```text
batch/head/Q-block
  └─ kvSIdx loop
      ├─ QK + online softmax
      ├─ PV + rescale-O
      └─ kvNIncreIdx loop
          ├─ KV offset
          ├─ workspace offset
          └─ mask / group / tile 参数计算
```

主要问题：

| 热点 | 原始形态 | 优化方向 |
|------|----------|----------|
| Workspace 基址 | `coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U)` 多处重复 | 外提为 `coreWsBase_` / `curStackWsBase` |
| KV offset | `blockSize * strideKV`、`l1NDynamic * strideKV` 反复乘 | `Update()` 中预乘 |
| Tile 参数 | `blockSize / l1NDynamic` 反复除 | 预计算 `tileNNumPerBaseBlock_` |
| Head 分组 | `qHeads / kvHeads` 反复除 | 预计算 `groupSize_` |

## 3. 关键参数 / 修改点

| 参数 | 类型 | 归属 | 计算方式 | 替换内容 |
|------|------|------|----------|----------|
| `coreWsBase_` | `uint64_t` | FIA decode 类 | `coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U)` | workspace 基址公共前缀 |
| `groupSize_` | `uint32_t` | FIA decode 类 | `qHeads / kvHeads` | head 分组除法 |
| `blockSizeStride_` | `uint32_t` | QK/PV decode block | `blockSize * strideKV` | KV block offset |
| `l1NDynStride_` | `uint32_t` | QK decode block | `l1NDynamic * strideKV` | L1 N 方向 offset |
| `tileNNumPerBaseBlock_` | `uint32_t` | QK decode block | `blockSize / l1NDynamic` | tile N 方向除法 |

关键约束：

- `PRE_LAUNCH + 1U == 3`，不是 2 的幂；禁止把 `% (PRE_LAUNCH + 1U)` 改成位与。
- 本优化不修改 L1 pingpong、MTE 事件链或 TilingData 结构。
- paged / non-paged cache 两个模板分支都要同步替换。

## 4. 核心计算循环 / 关键代码

### Workspace 基址外提

```cpp
// 循环外
coreWsBase_ = static_cast<uint64_t>(coreIdx) *
              WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U);

// kvSIdx 循环内
uint64_t curStackWsBase =
    coreWsBase_ + curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;

gmOffsetS = curStackWsBase + sWorkspaceIncreOffset;
gmOffsetP = curStackWsBase + pWorkspaceIncreOffset;
```

### KV offset 预乘

```cpp
__aicore__ inline void Update(uint32_t strideKV,
                              uint32_t blockSize,
                              uint32_t l1NDynamic) {
    blockSizeStride_ = blockSize * strideKV;
    l1NDynStride_ = l1NDynamic * strideKV;
    tileNNumPerBaseBlock_ =
        (l1NDynamic != 0U) ? (blockSize / l1NDynamic) : 0U;
}

__aicore__ inline void getKVOffset(GlobalTensor<int32_t> &gBlockTable,
                                   uint32_t &kOffset,
                                   uint32_t nowNIdx,
                                   uint32_t nL1Idx) {
    if constexpr (PAGED_CACHE_FLAG_) {
        uint32_t blockTableId = gBlockTable.GetValue(nowNIdx);
        kOffset = blockTableId * blockSizeStride_ + nL1Idx * l1NDynStride_;
    } else {
        kOffset = nowNIdx * blockSizeStride_ + nL1Idx * l1NDynStride_;
    }
}
```

## 5. 效果与适用边界

| 措施 | 预期效果 | 适用边界 |
|------|----------|----------|
| `coreWsBase_` / `curStackWsBase` | 消除多处 workspace 基址重复乘法 | 仅替换 workspace 环相关 offset，不替换无关 GM offset |
| `blockSizeStride_` / `l1NDynStride_` | 减少 KV offset 乘法 | `Update()` 必须在主循环前调用 |
| `tileNNumPerBaseBlock_` / `groupSize_` | 减少热循环除法 | 分组参数变化时必须重新计算 |

保守收益预期：

| 项目 | SCALAR 下降 |
|------|------------:|
| KV offset 预乘 | 5% - 15% |
| tile / group 参数外提 | 2% - 8% |
| workspace 基址外提 | 5% - 10% |
| 合计 | 12% - 30% |

若落地后出现 `SCALAR%` 下降但 case cycle 不降，通常说明实际瓶颈是 SCALARLDST，应切换到 SCALARLDST 减载策略。
