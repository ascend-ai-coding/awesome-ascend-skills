# 场景开发指导：put-all-gather-quant-matmul

> 前置阅读：[`design.md`](design.md)（设计合同）。本文给出实现模板锚点与验收清单。

## 1. 文件清单与落点（改自官方 AG 算子）

| 文件 | 动作 | 契约 |
|:---|:---|:---|
| `kernel/{op}_tiling_data.h` | 新建（改自 `all_gather_mx_matmul_udma_tiling_data.h`） | `AllGatherMxMatmulUdmaTilingData` = `QuantMatmulTilingData mmTile` + `CommTilingData commTile`（**单份**——AG 不像 A2A 有独立 scaleCommTilingData，scale 切分由 kernel 从 mmTile.k 重推导 `scaleKLen = CeilDiv(k,64)×2`）；`#pragma pack(push,8)`+`alignas(8)`；CommContext 包装在 `Apace::AivComm` 命名空间 + 全局 using 导出 |
| `kernel/{op}_udma_impl.h` | 新建（改自 `all_gather_mx_matmul_udma_impl.h`） | Impl 类（Init 见 design.md §6 布局）+ `AllGatherProcess()`（AIV：预触发 SetFlag(0) + 逐轮 Commit/Wait/SyncAll/SetFlag(round+1) + Finalize）+ `MatmulProcess()`（AIC：组 KernelParams 委托 matmul kernel）+ `__global__` 入口（impl 头末尾，`KERNEL_TYPE_MIX_AIC_1_1`） |
| `kernel/{op}_kernel.h` | 新建（改自 `qmm_mx_kernel_ag_udma.h`） | `QmmMxKernelAgUdma` 模板类：HEAD/MAIN/TAIL 三区 FragmentTensor + `ResolveTileCtx` + 延迟构建/`UpdateMainRoundAddrs` + `SetL2Cache`（design.md §5 双 gate） |
| `src/main.cpp` | 新建（改自 AG ST） | host 推导（design.md §3 公式）；AG 无 head_m_size 参数；RunKernel 封装 launch |
| `scripts/gen_data.py` / `verify_result.py` | 新建 | 纯 numpy 自实现 FP8 编解码（无 ml_dtypes 依赖）；每 rank 独立 seed；bit-exact/1-ULP 判定 |

## 2. 关键实现锚点（官方源码，改造时逐一对照）

| # | 事实 | 锚点 |
|:---|:---|:---|
| A1 | 入口 ABI：`__gm__ CommContext*` 首参 + aGM/aScaleGM/bGM/bScaleGM/cGM + tilingData 按值（7 参） | `all_gather_mx_matmul_udma_impl.h:267-271` |
| A2 | Init 顺序（**与 A2A 不同**）：`InitBaseParams(tilingData)` **先于** ctx 提取；dataRegionBytes/scaleRegionBytes 在 ctx 提取后于 Init 内计算 | 同上 `:128-142` |
| A3 | AIV 循环：守卫 `GetBlockIdx() < rankSize_` 内 `allGatherScale_.Commit(); allGatherData_.Commit(); allGatherData_.Wait<BARRIER_DEVICE>()`；守卫外 `SyncAll<true>()` + `SetFlag(round+1)`；循环外预触发 `SetFlag(0)` | 同上 `:246-263` |
| A4 | BarrierMode：data `Init<BARRIER_NONE>` + scale `Init<BARRIER_DEVICE>`（scale 的 Wait 承担跨卡 fence；两对象共 channel） | 同上 `:166-171` |
| A5 | 三区 FragmentTensor：`MakeFragParam(paddedTailM, tailM, ...)` padding 表达；MAIN 延迟构建 + `UpdateAddrList` 换轮 | `qmm_mx_kernel_ag_udma.h:389-398,523-528,532-555` |
| A6 | matmul 调用：`mmadFrag(blockA, gmBlockB, blockScaleA, gmBlockScaleB, gmBlockBias, cFragAddrs_, mPerRank, tileM, tileCnt, tailM, rankCnt, Ni, singleShape, regionMPos, nPos, 0, blockC)`（17 参契约详见 [`compute.md`](../../fundamentals/compute.md) §7） | 同上 `:403-406` |
| A7 | WINDOW_LEN 复定义技巧（调度器窗口=1 非侵入式修改）：`#define WINDOW_LEN 1L` → include `block_scheduler_qbmm.h` → `#undef` | 同上 `:24-26` |
| A8 | 尾块拆分核数守卫：`(sch.GetEndBlockIdx()+1) * mTailTile * nTailTile <= GetBlockNum()` 才 `UpdateTailTile` | 同上 `:346-348` |
| A9 | AIC 等待：waitedMask 位去重（含预触发 id 0）+ 收尾 drain `t <= commTurn` **含端点** | 同上 `:370-373,410-414` |

## 3. 验证矩阵

| 维度 | 覆盖 | 判据 |
|:---|:---|:---|
| 精度 | rank 2/4/8 × 对齐 M × 非对齐 M（2061/3085 类触发尾块 + paddedTailM padding 路径）× K=32 最小对齐 × m<512 收缩 | bit-exact 或 ≤1 ULP（raw uint16 比较） |
| golden | 每 rank 独立 seed（`BASE_SEED+rank`）；B DN 落盘 `[N,K]` 不转置、scaleB `[N,K/32]`；AllGather=concatenate | gen_data 双侧校验（K%32、scale 偶数） |
| T 维度 | tileCnt=1 串行基线 → 多 tile（三区布局/flag 配对仅在 T>1 暴露） | R9/R5 |
| R>8 变体 | 扩数组后专测 R=16（越界回归） | design.md §6 硬约束 |
| perf | msprof + L2 flush + parse_prof（官方口径 avg of card avgs + skill 投产另报跨 rank max） | R15/R20 |

## 4. 常见 FAIL 信号与定位

| 现象 | 根因 | 修复方向 |
|:---|:---|:---|
| R>8 时数据错乱/越界崩溃 | `cFragAddrs_[8]` 等定长数组静默越界 | 扩数组（design.md §6） |
| 尾块行数错误/padding 区数据泄漏 | paddedTailM/realFragmentSize 表达错 | FragmentParam 成对设置（A5） |
| HEAD 区等待挂死 | 预触发 SetFlag(0) 丢失或 AIC 特判跳过 id 0 | 保持统一位掩码路径（A9） |
| 换轮后读到旧数据 | MAIN 换轮未先 wait 或 UpdateAddrList 顺序错 | 先 wait 再 update（A5） |
| scale 精度错 | gen_data scaleB 布局/换算错（元素数 vs 字节数） | nonSplitAxisSize 换算 = ka/32；DN 落盘约定 |
| 照抄 A2A 的 SWAT 开关 | AG 需 `SetOptimizeEnable(false)+SetMTailAlignEnable(true)`（A2A 默认相反） | design.md §3 |

## 5. 合规映射（本场景重点红线）

| 红线 | 本场景落点 |
|:---|:---|
| R1/R2 | 入口无 `__schedmode__`；含 `KERNEL_TYPE_MIX_AIC_1_1` |
| R3 | 变体 dtype 加入口 + host dispatch（官方 AG 单入口仅 E4M3——照抄会被判 FAIL，须自行补） |
| R4 | `block/`/`tiling/` 零修改（blaze_ext 属 block 层：新变体组件按 R4 纪律评估，优先参数化复用而非改共享层） |
| R5 | SetFlag(round+1) ↔ WaitFlag 同 idx 配对（含预触发 0） |
| R12 | UB 静态布局 512×2+32（MakeMemPtr，无 TPipe） |
| R13 | 通信对象与 TeamBarrier 均 totalJobs=rankSize（官方前 R 核映射） |
| R14 | 窗口 rank-major data 在前 scale 在后（winOffset=dataRegionBytes） |

> 完整 R1-R21 见 [`review-checklist.md`](../../review-checklist.md)。
