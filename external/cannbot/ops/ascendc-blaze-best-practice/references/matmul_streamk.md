# Matmul StreamK 模板（AIC+AIV 混合 K 切分）

> **适用架构**：DAV_3510（CANN 9.0.0-beta.2）
>
> **实现状态**：本文为设计文档，StreamK 工程模板（kernel + scheduler + epilogue + tiling）尚未交付，需按 §5 关键改造表自行从 `matmul_custom/` 演进。
>
> 前置阅读：[`matmul_pattern.md`](matmul_pattern.md) §0 模式总览、§10 三模板体系。StreamK 详细算法设计见 skill `ascendc-performance-best-practices` 的 `reference/matmul/streamk_design.md`。

## 1. 适用判据

StreamK 解决 MN 欠并行场景下的核间负载不均。**把空闲核拉去参与 K 维累加**，通过 workspace 做跨核 K 累加归约。

| 条件 | 阈值 |
|------|------|
| MN 块数 < AIC 核数 | `mCnt·nCnt ≤ aicNum/2`（纯 SK） |
| 或 MN 不整除 + 尾块无法细切 | `mCnt·nCnt % aicNum ≠ 0` 且余数 ≤ `aicNum/2`（DP+SK） |
| K 足够长 | K ≥ 4096（FP16 列） |

**反向不适用**：MN 块数 ≥ aicNum 且整除 → 纯AIC pingpong/SWAT 足够；K 太短 → 切分无收益。

## 2. 与纯AIC 模板的核心差异

| 维度 | 纯AIC | StreamK |
|------|-------|---------|
| **Kernel 属性** | `__global__ __aicore__ __cube__` | `__aicore__` (MIX，无 `__cube__` 限定) |
| **AIV 行为** | `if ASCEND_IS_AIV return` | 归约循环：WaitFlag→Add→Cast→CopyPad→SetFlag |
| **分核维度** | M、N 二维 | M、N、**K 三维** |
| **并行度** | `min(mCnt·nCnt, aicNum)` | `min(mCnt·nCnt·kCnt, aicNum)` |
| **输出路径** | AIC Fixpipe→GM | AIC→workspace(FP32)→AIV 归约+cast→GM |
| **Workspace** | 不需要 | **必需**：`aicNum × 256² × sizeof(FP32) + RPC` |
| **cmatrixInitVal** | `(iter0==0 && iter1==0)` | 每段首次 = true（段间不累加） |
| **unitFlag** | 始终 `FINAL_ACCUMULATION` | 按 `iter0+iter1` 判断末次 |
| **baseK** | 64 或 128 | 固定 64（保 L0 双缓冲） |
| **kL1 上限** | 按 L1 容量推导 | 额外 `STEPKA_THRESHOLD=4` 上限 |

## 3. 两种子模式

```
SK (全 K 切分)                           DP+SK (末轮 K 切分)
mCnt·nCnt ≤ aicNum/2                    mCnt·nCnt ≥ aicNum 非整除
                                        
每个 MN tile 都被 kCnt 个核切 K          稳态轮: 完整 DP
                                       末轮: tail tile 切 K
┌──── tile0 (K 被切 kCnt 段) ────┐      ┌稳态┐┌稳态┐...┌SK末轮┐
│ core0 core1 ... core_{kCnt-1} │      │c0  ││c1  │   │切K   │
└───────────────────────────────┘      └────┘└────┘   └──────┘
```

两种模式共用一套 kernel 实现，由 tiling 参数和 `CheckIsSkScene(tileIdx)` 区分。

## 4. 数据流与同步

```
AIC 核 (K 段 i)                       AIV 核
                                      
GM(A,B) → L1 → L0 → MMAD              
  │              │                    
  │              ▼                    
  │         L0C (FP32 部分和)          
  │              │                    
  │    CopyL0C2GM → workspace[i]       
  │              │                    
  │    CrossCoreSetFlag ──────────────▶ WaitFlag
  │                                      │
  │                              for k in 0..kCnt:
  │                                Add(UB, workspace[k])
  │                              Cast FP32→FP16/BF16
  │                              DataCopyPad → GM[C]
  │                                      │
  │◀─────────────── CrossCoreSetFlag ────┘ (可选)
```

**关键同步约束**：
- AIC 末轮结束后必须 `CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4>(AIC_SYNC_AIV_FLAG)` 通知 AIV
- 空闲 AIC 核 **必须** 在 return 前补发 flag，否则 AIV 永久等待
- `cmatrixInitVal` 在每个 SK 段首次调用时必须为 true（段间 L0C 不累加）

## 5. 从纯AIC 升级到 StreamK 的关键改造

| 改造项 | 纯AIC | StreamK |
|--------|-------|---------|
| 调度器 | `MatmulSwatScheduler` | `BlockSchedulerStreamK`（新增 `skKTileNum`、`CheckIsSkScene`） |
| Tiling | `MatmulTilingSwat` | `MatmulTilingStreamK`（新增 `skSingleCoreK`、`kCnt`） |
| Kernel | AIC-only 循环 | AIC/AIV 统一循环 + `ASCEND_IS_AIC`/`ASCEND_IS_AIV` 分发 |
| Epilogue | 无 | `BlockEpilogueStreamK`：workspace Add + Cast + DataCopyPad |
| Workspace | - | `aicNum × 256² × sizeof(FP32) + RPC` 在 host 侧分配 |
| 同步 | 无跨核 | `CrossCoreSetFlag`/`CrossCoreWaitFlag` 配对 |

## 6. 常见陷阱

| # | 现象 | 根因 | 修复 |
|---|------|------|------|
| P1 | workspace offset 错位 | 直接用 tileIdx 而非末轮 MN 序号 | `offset = ((tileIdx % usedCoreNum) / skKTileNum * skKTileNum + kIdx) * baseM * baseN` |
| P2 | AIV hang | 空闲 AIC 未发 flag | 提前 return 前补发 `CrossCoreSetFlag` |
| P3 | L0C 跨段污染 | `cmatrixInitVal` 固定 false | 每段首次 `(iter0==0 && iter1==0)` 置 true |
| P4 | MN 整除仍开 StreamK | 未走 IsCapable 门禁 | tiling 入口严格校验 |
| P5 | stepKa=1 失去 pingpong | baseK 过大 | 固定 `baseK=64` |

## 7. 参考

- 详细设计：skill `ascendc-performance-best-practices` 的 `reference/matmul/streamk_design.md`
- 性能优化策略总览：skill `ascendc-performance-best-practices` 的 `reference/matmul/guide.md`
- 返回入口：[`matmul_pattern.md`](matmul_pattern.md) §10
