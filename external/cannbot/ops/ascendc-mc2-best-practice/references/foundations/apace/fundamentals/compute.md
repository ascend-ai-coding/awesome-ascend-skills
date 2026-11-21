# apace 计算原理与接口

> 本文档覆盖 apace 算子的计算侧：计算原理（Cube 核 MMAD 流水）、计算接口（Blaze 组件的作用与调用方式）、kernel 组织模式（QuantMatmulMxKernel 骨架）。官网当前上库 2 个计算 kernel：`apace/kernel/all_to_all_quant_matmul/quant_matmul_mx_kernel.h`（`QuantMatmulMxKernel`，CommPolicy 策略注入）与 `apace/kernel/all_gather_quant_matmul/qmm_mx_kernel_ag_udma.h`（`QmmMxKernelAgUdma`，FragmentTensor 版）。

## 目录

1. [计算原理：Cube 核 MMAD 流水](#1-计算原理cube-核-mmad-流水)
2. [计算接口清单](#2-计算接口清单)
3. [Kernel 组织模式：QuantMatmulMxKernel 骨架](#3-kernel-组织模式quantmatmulmxkernel-骨架)
4. [Run 流程与 MatmulMode 分支机制](#4-run-流程与-matmulmode-分支机制)
5. [Layout / Scale / L2 Cache 约定](#5-layout--scale--l2-cache-约定)
6. [关键常量](#6-关键常量)
7. [FragmentTensor（AllGather 场景）](#7-fragmenttensorallgather-场景)
8. [排错速查](#8-排错速查)
9. [集成 Checklist](#9-集成-checklist)

---

## 1. 计算原理：Cube 核 MMAD 流水

> 架构硬约束：Matmul 走 Blaze 模板（`BlockMmad` + `BlockScheduler` + `MatmulWithScaleMx`），禁止 `AscendC::Matmul` 等 asc-devkit 黑盒 API——黑盒 API 无法接入通算流水，无法在逐 tile 粒度插入通信等待（详见 SKILL.md 与 [`architecture.md`](architecture.md)）。

Blaze 集成方式：

| 方式 | 说明 | 核心组件 | 官网样例 |
|:---|:---|:---|:---|
| **组合模式** | kernel 类直接持有 `BlockMmad` + `BlockScheduler`，通信等待经 `CommPolicy` 模板参数策略注入，精细控制逐 tile 同步 | `QuantMatmulMxKernel` | `apace/kernel/all_to_all_quant_matmul/quant_matmul_mx_kernel.h` |
| 委托模式（`GemmUniversal` + 自定义 Epilogue） | 当前官网未上库样例 | — | — |

组合模式特征：

- kernel 类（`QuantMatmulMxKernel`）直接持有 `BlockMmad mmadOp_` 和 `CommPolicy commPolicy_` 成员，不经 `GemmUniversal` 包装
- 计算与通信的同步点由 kernel 内部逐 tile 调用 `commPolicy_.WaitTile(tileIdx)` 完成，具体等待机制（HCCL handle / CrossCore flag）由模板实参在编译期绑定
- 适合需要通信-计算逐 tile 重叠的场景

单核 matmul 块操作（`BlockMmad` 职责）的数据通路：GM → L1 → L0A/L0B → L0C → GM；多次 mmad 在 L0C 上累加（第 8 参为累加序号），达到 `splitKNum` 份后触发 fixpipe 写 GM（机制详见 §4.2）。

> 官网 `apace/block/blaze_ext/epilogue/` 当前仅有 `.gitkeep`（无 Epilogue 扩展实现）；`blaze_ext/gemm/block/qmm_mx_block_mmad_fragment.h` 是 AllGather 算子用的 Fragment 版 BlockMmad（见 §7）。

---

## 2. 计算接口清单

组合模式 PUT kernel（`apace/kernel/all_to_all_quant_matmul/quant_matmul_mx_kernel.h`，`QuantMatmulMxKernel`）使用的组件：

### 组件层级

| 层级 | 组件 | 作用 |
|:---|:---|:---|
| **Policy** | `MatmulWithScaleMx<0, false>` | MX 量化 matmul 分发策略；`0` = `NONE_FULL_LOAD_MODE`，`false` = 非 atomic（UDMA impl 中写作 `MatmulWithScaleMx<NONE_FULL_LOAD_MODE, false>`） |
| **Block** | `BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>` | 单核 matmul 块操作：GM → L1 → L0A/L0B → L0C → GM |
| **Scheduler** | `BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB, AType>` | 产出 M/N tile 坐标，管理尾块拆分（`UpdateTailTile`） |
| **Kernel** | `QuantMatmulMxKernel<ProblemShape, BlockMmad, BlockScheduler, CommPolicy>` | apace 组合模式 kernel，编排 Scheduler + Mmad + 通信等待 |
| **CommPolicy** | 算子自定义策略类 | 注入逐 tile 通信等待，契约见 §3 |

以上类型实例化见 `apace/kernel/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_hcomm_impl.h`（`AllToAllMxQuantMatmulHcommImpl` 的 `DispatchPolicy`/`BlockMmad`/`BlockScheduler`/`QuantMatmulKernelImpl` using 声明）。

### 头文件来源

| 组件 | 头文件 | 来源 |
|:---|:---|:---|
| `BlockMmad` | `blaze/gemm/block/block_mmad_qbmm_mx.h` | ops-tensor（外部库，FetchContent 拉取） |
| `BlockScheduler` | `blaze/gemm/block/block_scheduler_qbmm.h` | ops-tensor |
| `DispatchPolicy` | `blaze/gemm/policy/dispatch_policy.h` | ops-tensor |
| Layout/Tensor | `include/tensor_api/tensor.h` | ops-tensor |
| `QuantMatmulMxKernel` | `apace/kernel/all_to_all_quant_matmul/quant_matmul_mx_kernel.h` | ops-transformer（本仓 apace） |

> Blaze/tensor_api 来自 [cann/ops-tensor](https://gitcode.com/cann/ops-tensor) 仓，由 `cmake/third_party/ops-tensor.cmake` 按 pin 的 tag 拉取/检出（具体 tag 以仓内 cmake 文件当前值为准，会随版本演进）；用其它 ops-tensor 检出核对时，`WEIGHT_NZ`/`TRANS_A`/`TRANS_B` 等命名可能与本文不一致。组合模式不使用 BlockEpilogue。

---

## 3. Kernel 组织模式：QuantMatmulMxKernel 骨架

`QuantMatmulMxKernel`（`apace/kernel/all_to_all_quant_matmul/quant_matmul_mx_kernel.h`）是组合模式的核心计算类。

### 模板参数

| 参数 | 含义 | 约束 |
|:---|:---|:---|
| `ProblemShape` | 问题形状类型 | `Te::Shape<int64_t, int64_t, int64_t, int64_t>` 即 `<M, N, K, B>` |
| `BlockMmad` | 块计算类型 | 需暴露 `Params`/`L1Params`/`AType`/`LayoutA` 等嵌套类型及 `WEIGHT_NZ`/`TRANS_A`/`TRANS_B` 常量（ops-tensor `BlockMmad` 满足） |
| `BlockScheduler` | 块调度类型 | 需提供 `Params`、`GetTileIdx`/`GetBlockShape`/`GetTileCoord`/`UpdateTailTile` |
| `CommPolicy` | 通信等待策略 | 必须提供 `__aicore__ void WaitTile(uint32_t tileIdx)`；缺该接口则编译失败 |

CommPolicy 官网两个实现：

| 策略类 | 位置 | `WaitTile` 语义 |
|:---|:---|:---|
| `HcommCommWaitPolicy` | `apace/kernel/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_hcomm_impl.h` | `tileIdx==0` 先 Wait scale handle；再按 `tileIdx < headTileCnt_` 分别 Wait dataHead/dataTail handle（纯 AIC 核内 HCCL Wait） |
| `UdmaCommWaitPolicy` | `apace/kernel/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_udma_impl.h` | `CrossCoreWaitFlag<0x2, PIPE_MTE2>(tileIdx)`（等 AIV UDMA 通信完成的 CrossCore flag） |

### 嵌套类型

`MatmulMode`（枚举，决定单次 `Run` 的计算模式）：

| 值 | 含义 |
|:---|:---|
| `REMOTE = 0` | 仅计算通信同步来的远程数据（self rank 依 `localMatmul` 跳过或取本地 A） |
| `LOCAL = 1` | 仅本地计算（本 rank local A × 本 rank B 段） |
| `DEFERRED_SYNC = 2` | per-tile 本地先算驻留 L0C → wait_flag → 远程累加 → 单次 fixpipe |

`QBMMTiling`（tiling 配置）：`baseM` / `baseN` / `baseK` / `dbL0C` / `isBias`（枚举 `BIAS_DISABLED=0` / `BIAS_ENABLED=1`）。

`LocalParams`（本地 rank 相关参数）：

| 字段 | 含义 |
|:---|:---|
| `rankId` / `rankSize` | 本 rank ID / 总 rank 数 |
| `originalM` | 单卡负责的总 M 行数 |
| `localAGmAddr` / `localScaleAGmAddr` | 本地 A / 本地 ScaleA 的 GM 地址（不经通信 buffer） |
| `localMatmul` | 本地计算模式选择：`0` 融合 / `1` LOCAL 前置 / `2` DEFERRED_SYNC（见 §4.4） |
| `splitKNum` | L0C 累加份数（决定 mmad 内部 fixpipe 时机，见 §4.2） |
| `matmulMode` | 本次 `Run` 的 `MatmulMode` |
| `headTileSize` | 通信 head tile 的 M 行数（`CalcDependTileIdx` 的映射粒度） |

`Params`（顶层参数）：`problemShape` / `mmadParams`（`BlockMmad::Params`，含 `aGmAddr`/`bGmAddr`/`scaleAGmAddr`/`scaleBGmAddr`/`cGmAddr`/`biasGmAddr`）/ `l1Params` / `schParams` / `qbmmParams` / `localParams`。

### 公有接口

| 方法 | 职责 |
|:---|:---|
| `Init(params)` | 解析 `isBias`/`isAtomicAdd_`，`ResetGmAddr` 缓存各 GM 基址；AIV 直接返回（`ASCEND_IS_AIV` 守卫，AIC-only） |
| `Run(params)` | 主流程（见 §4.1） |
| `operator()(params)` | 等价 `Run(params)`，Impl 经此调用 |
| `GetCommPolicy()` | 返回 `CommPolicy&`，供 Impl 绑定通信状态（如 hcomm impl 中 `GetCommPolicy().state_ = &commState_`） |

### 编译期常量与关键成员

| 名称 | 取值/含义 |
|:---|:---|
| `weightNz` / `transA` / `transB` | 取自 `BlockMmad::WEIGHT_NZ/TRANS_A/TRANS_B`（官网实例化为 `transA=false, transB=true`） |
| `C0_SIZE` | `IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8`（FP4→64，FP8→32） |
| `kCacheLineAlignMask` | FP4→`0xff`（256B），FP8→`0x7f`（128B） |
| `SCALE_C0` | `2` |
| `mmadOp_` / `commPolicy_` | BlockMmad / CommPolicy 成员对象 |
| `aGmAddr_` vs `localAGmAddr_` | 远程数据基址（通信 buffer）vs 本地数据基址；ScaleA 同理（`scaleAGmAddr_` / `localScaleAGmAddr_`） |
| `isAtomicAdd_` | `matmulMode==REMOTE && localMatmul==1` 时置位（REMOTE 需原子累加到 C） |
| `needUpdateTail_` | 尾块拆分已生效标记（防重复 `UpdateTailTile`） |

---

## 4. Run 流程与 MatmulMode 分支机制

### 4.1 kernel 内 `Run` 流程

`QuantMatmulMxKernel::Run`（`apace/kernel/all_to_all_quant_matmul/quant_matmul_mx_kernel.h`）：

```
Run(params)
    ├── Init(params)                                ← isBias_/isAtomicAdd_ + GM 基址缓存
    ├── if isAtomicAdd_: SetAtomicAdd<CType>()      ← REMOTE+localMatmul==1 时开启原子累加
    ├── BlockScheduler bs(problemShape, schParams)
    ├── l0TileShape = {baseM, baseN, baseK, 0}
    ├── mmadOp_.Init(problemShape, l0TileShape, l1Params, isBias_, dbL0C > 1, splitKNum)
    ├── ProcessSingleBatch(params, bs, 0, true)     ← 主计算循环（见 §4.2）
    └── if isAtomicAdd_: SetAtomicNone()
```

### 4.2 ProcessSingleBatch 与 rank 遍历

`ProcessSingleBatch`（`apace/kernel/all_to_all_quant_matmul/quant_matmul_mx_kernel.h`）是逐 tile 主循环，按 `localParams` 分三个分支。

#### 循环骨架

```
ProcessSingleBatch(params, bs, restBatch, isTailRound)
    ├── 构建全局 Layout/Tensor（gmA 通信 buffer / gmALocal 本地 / gmB / gmScaleB / gmC ...）
    ├── 尾块拆分：needUpdateTail_ 或 isTailRound 条件下 bs.UpdateTailTile(mTailTile, nTailTile)
    ├── SetL2Cache(...)                                  ← 见 §5.3
    └── while (bs.GetTileIdx(blockIdx)):
        ├── GetBlockShape / GetTileCoord → singleShape, mPos, nPos
        ├── Slice gmBlockC / gmBlockBias
        └── 分支：
            ├── localFirst（LOCAL && localMatmul==1）: 本 rank local A × 本 rank B 段
            ├── deferredSync（DEFERRED_SYNC）: 见下
            └── REMOTE: 见下
```

#### 三个分支语义

| 分支 | A 数据来源 | rank 遍历 | L0C 累加 |
|:---|:---|:---|:---|
| LOCAL | `gmALocal`（`actualMPos = rankId*oriM + mPos`） | 仅本 rank B 段 | 单次 mmad，第 8 参 `0` |
| DEFERRED_SYNC | Phase 1：本 rank `gmALocal`；Phase 3：其它 rank `gmA`（通信 buffer） | Phase 3 遍历 `rank != rankId` | Phase 1 第 8 参 `0`（reset）；Phase 3 `remoteRankCnt` 从 1 递增，末次触发 fixpipe |
| REMOTE | `gmA` 通信 buffer；`rank==rankId` 时：`localMatmul==1` 跳过（`continue`），否则改用 `gmALocal` | 遍历全部 rank | `remoteRankCnt` 从 0 递增 |

#### mmadOp_ 调用签名

```cpp
mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape, remoteRankCnt);
```

第 8 参为 L0C 累加序号：`0` 表示 reset（覆盖写 L0C），非 0 表示在 L0C 上累加；达到 `splitKNum` 份后触发 fixpipe 写 GM。约束：每次 `Run` 内各 rank 的 mmad 次数必须等于 `splitKNum`，错配导致 C 不输出或累加错误。`splitKNum` 在 Impl 的 `SetupParams` 中按模式设置，hcomm 与 udma 映射不同：

| Impl | LOCAL | REMOTE 阶段 |
|:---|:---|:---|
| udma | `1` | `localMatmul==1` → `rankSize - 1`；`localMatmul==0`/`2` → `rankSize` |
| hcomm | `1` | `localMatmul != 0` → `rankDim_ - 1`；`localMatmul == 0` → `rankDim_`（`rankDim_` 为 hcomm impl 成员变量，等价 rankSize） |

> ⚠️ hcomm 下 `localMatmul==2` 的 mmad 次数与 `splitKNum` 错配 → 结果静默错误；mode 2 仅 UDMA impl 可达（完整论证见 [`fusion.md`](fusion.md) §5.1）。

#### CalcDependTileIdx 与 wait 去重

- `CalcDependTileIdx(mPos + blockM - 1, headTileSize, totalTiles)`：把当前 tile 末行映射到通信 tile 序号，越界 clamp 到 `totalTiles - 1`
- `waitedMask` 位掩码：同一 dependTileIdx 只 wait 一次；DEFERRED_SYNC 的 wait 必须在 self rank mmad 之后发出（kernel 内注释明示的契约）
- `!localFirst` 时循环末尾 drain：对 `0..totalTiles-1` 中未 wait 的位补 `commPolicy_.WaitTile(t)`

### 4.3 CommPolicy WaitTile 调用点

| # | 调用点 | 语义 |
|:---|:---|:---|
| 1 | REMOTE 分支：tile 循环内、读通信 buffer 前 | 等当前 tile 依赖的通信 tile 完成（`CalcDependTileIdx` 映射，位掩码去重） |
| 2 | DEFERRED_SYNC 分支：self rank mmad 之后、其它 rank mmad 之前 | 阻塞后续 shmem 读（位掩码去重） |
| 3 | 循环结束后（`!localFirst`） | drain：补齐所有 `totalTiles` 中尚未 wait 的 tile，保证 wait 与通信 set 严格配对 |

违反后果：漏 wait → 读到未同步完成的脏数据（精度错误）；wait/set 不配对 → 死锁。

### 4.4 MatmulMode kernel 机制

> 模式选型、语义对比与精度/性能权衡由 [`fusion.md`](fusion.md) 承载，本节只讲 kernel 内部机制。

`MatmulMode`（kernel 枚举）决定单次 `Run` 内 `ProcessSingleBatch` 的分支行为与 `isAtomicAdd_` 置位：

| MatmulMode | 分支行为（`ProcessSingleBatch`） | `isAtomicAdd_` |
|:---|:---|:---|
| `REMOTE` | 遍历全部 rank：`rank==rankId` 时 `localMatmul==1` 跳过（`continue`），否则切 `gmALocal` 本地直读；`remoteRankCnt` 从 0 递增 | `localMatmul==1` 时置位（REMOTE 需原子累加到 C，`Run` 包 `SetAtomicAdd`/`SetAtomicNone`） |
| `LOCAL` | 仅算本 rank：`gmALocal`（`actualMPos = rankId*oriM + mPos`）× 本 rank B 段，单发 mmad（第 8 参 `0`），`splitKNum=1` 直接 fixpipe 写 C | 不置位 |
| `DEFERRED_SYNC` | per-tile 三段：Phase 1 self mmad 驻留 L0C（第 8 参 `0`，reset）→ Phase 2 wait（位掩码去重）→ Phase 3 遍历 `rank != rankId` 累加（`remoteRankCnt` 从 1 递增，末次触发 fixpipe）；**单次 DEFERRED_SYNC `Run` 完成全部计算，无独立 LOCAL 前置 Run**（UDMA impl 仅 `localMatmul==1` 才执行 `RunLocalMatmul`） | 不置位 |

关键约束：

| # | 约束 | 违反后果 |
|:---|:---|:---|
| 1 | `localMatmul==1` 时 REMOTE 阶段必须开 `SetAtomicAdd<CType>`（kernel `Init` 自动置 `isAtomicAdd_`） | 远端累加覆盖 LOCAL 结果，精度错误 |
| 2 | DEFERRED_SYNC 的 wait 夹在 self mmad 与其它 rank mmad 之间（kernel 内注释明示的契约） | 提前 wait 失去 local 掩盖通信的收益；漏 wait 读脏数据 |
| 3 | LOCAL 阶段 `splitKNum=1`（单份直接 fixpipe 写 C） | 与 REMOTE 混用 splitKNum 导致 fixpipe 时机错误 |

> 注：早期 GET 版设计中的 `LocalDelay` 模板参数、`LocalCompute()`、`cGmSelfAddr` 直写等概念在官网当前代码中不存在，已由 `localMatmul` + `MatmulMode` 机制取代。

### 4.5 Impl 编排：hcomm vs UDMA

`AllToAllMxQuantMatmulHcommImpl::Run`（`apace/kernel/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_hcomm_impl.h`）：

```
Run()
    ├── if tilingData->localMatmul != 0: MatmulProcess(MatmulMode::LOCAL)   ← 本地块前置，掩盖通信
    ├── MatmulProcess(MatmulMode::REMOTE)
    ├── SyncAll()
    └── commState_.hccl_.Finalize()
```

每次 `MatmulProcess(mode)` = `SetupParams(params, mode)` + `quantMatmulKernelImpl_(params)`。

UDMA impl（`all_to_all_mx_quant_matmul_udma_impl.h`）的 Run 编排与 hcomm impl 差异较大，不能按"结构类似"套用：

| 维度 | hcomm impl | UDMA impl |
|:---|:---|:---|
| Run 结构 | 无 AIV/AIC 分支：`localMatmul != 0` 先 `MatmulProcess(LOCAL)`，再 `MatmulProcess(REMOTE)`，末尾 `SyncAll()` + `commState_.hccl_.Finalize()` | 按 `ASCEND_IS_AIV`/`ASCEND_IS_AIC` 分支：AIV 执行 `RunAllToAll()`（`Finalize` 在其末尾）；AIC 仅 `localMatmul == 1` 时先 `RunLocalMatmul()`，再无条件 `RunMatmul()`；`Run()` 内无 `SyncAll`/`Finalize` |
| 本地前置阈值 | `localMatmul != 0`（mode 2 也会触发 LOCAL 前置，错配后果见 [`fusion.md`](fusion.md)） | `localMatmul == 1` |
| 远程阶段模式 | 恒 `MatmulMode::REMOTE` | `RunMatmul()` 按 `localMatmul == 2` 选 `DEFERRED_SYNC`，否则 `REMOTE`（见 §4.4） |

### 4.6 staging 输出模式（compute-first 场景）

> **适用范围**：本节仅适用于 compute-first 算子（AIC 先算、AIV 后通信聚合，编排原理见 [`fusion.md`](fusion.md) §6.2）；通信在前算子（官网两个 PUT 算子）AIC 直写 yGm，不适用。
>
> 生产实现中 mm 不直写 yGm，而是写 staging（workspaceGM），由 AIV AllToAll PUT + 增量归约处理。要点：

1. **mm 输出即通信源**：staging 为连续 `[M, N]`，第 targetRank 段即为 PUT 发往 targetRank 的 chunk，零重排零额外拷贝：
   ```cpp
   params.cGM = stagingGm;  // vendor kernel LOCAL 模式一次覆盖全 M；T>1 时按轮次带行偏移
   ```
2. **无 L0C 跨 rank 累加**：计算在前架构不做 splitK/splitM，rank 退化（`rankId=0/rankSize=1/splitKNum=1`）⇒ `isAtomicAdd_` 恒 false。跨 rank 聚合由 AIV 增量归约完成（FP32 中间累加）。
3. **T>1 按轮拆子区间**：tile 大小不变、只缩 problem M，逐轮 SetFlag 驱动 mm/comm 重叠；默认 FragmentTensor 一次调用（见 item 5），vendor 例外路径拆 R×T 子调用（复用同一份 tiling，受 [`fusion.md`](fusion.md) §6.2.2 SCALAR 约束）。
4. **归约 FP32 累加**：AIV 侧 BF16 → Cast float32 → Add → Cast 回 BF16（CAST_RINT），精度保障详见 [`fusion.md`](fusion.md) §6.2.6。
5. （FragmentTensor 路径，默认）AIC 逐 tile 构建 FragmentTensor 打包 R 个 rank 段地址，一次调用覆盖 `R × curTileM` 行，`FragmentSliceCopy<true>` scatter 各 rank 段独立写 staging——消除 R 循环、调用开销最小；vendor 例外路径的选型判据见 [`fusion.md`](fusion.md) §6.2.2。

### 4.7 compute-first 算子的输入切分语义判定

开发 compute-first 算子时，**先判定输入 A 的分布语义，再写 golden**——语义选错则 golden 与 kernel 全线错误。**用户未明确切分方式时必须回到需求拷问（grill-protocol 维度 9）向用户澄清，禁止默认假设**——"每卡有 X/n"类表述未区分输入切分与输出分布，是最高发误判点。典型语义对照：

| 语义 | 每卡输入 A | 每卡计算 | 跨卡聚合 | 适用 |
|:---|:---|:---|:---|:---|
| **完整输入**（各卡 A 数据不同） | 完整 `[m, k]`（K 不切分） | 独立完整 matmul `C_i = A_i × B` → `[m, n]` | AlltoAll 按 M 轴分发 + AIV reduceSum 逐元素求和 → `[m/R, n]` | MoE/分布式推理各卡不同输入 |
| **K 轴切分**（TP 语义） | `[m, k/R]`（K 轴按 rank 切分，全 m 行） | K 轴部分和 `C_i = A_i × B_i` | 跨卡求和补全 K 维 | TP 场景 |

**完整输入语义推论**：

1. **无 splitK/splitM 部分和** → AIC 侧禁用 L0C 原子累加（`SetAtomicAdd` 不使用），跨卡聚合完全下沉 AIV reduceSum
2. **bias 可选**：`C_i = A_i × B + bias`，bias 在各卡本地 matmul 内完成（当前未实现时需在文档显式声明）
3. **golden 生成必须与语义一致**：完整输入语义下 gen_data 每卡独立生成完整 A、各自算完整 matmul 再按 M 轴 AlltoAll+求和；⚠️ 参照 `all_to_all_quant_matmul`（其 A 按 K 切分）写 golden 会误选 TP 语义——**参考算子的输入分布不等于本算子的输入分布**
4. **整除约束差异**：完整输入语义仅要求 `m % rankSize == 0`；TP 语义额外要求 `k % rankSize == 0` 且 per-rank K 满足 `CeilDiv(k/R, 64)` 为偶数（即单次 mm 的 K 长度满足 MXFP8 scale 对齐约束，统一表述见 [`fusion.md`](fusion.md) §6.2.10）

### 4.8 精度保障规则（compute-first + 归约场景）

以下来自 ops-transformer 注册版 MC2 算子的生产规则，直调开发同样适用：

| # | 规则 | 适用场景 |
|---|------|---------|
| 1 | **FP32 中间累加全链路**：Cube 侧 L0C 用 FP32 累加器；AIV 归约先 `Cast(..., CAST_NONE)` 成 fp32 → `Add` 累加 → 输出 `Cast(..., CAST_RINT)` | compute-first + 归约 |
| 2 | **尾块搬出必须 DataCopyPad** 按真实字节数拷贝——32B 向上对齐的 DataCopy 会越界写脏相邻 buffer | 通用（含归约） |
| 3 | **fp8_e8m0 scale 的 `VF_CALL<CastVf>` 单次 ≤ 32 元素**（32B VF 限制），一次传全部 scale 实测精度失败，必须分批 | MXFP8 量化 |
| 4 | 多核归约用「分片独占 + FP32 顺序累加」，不用原子加；跨 rank 归约在 L0C（FP32）或 AIV（FP32）完成 | compute-first + 多核归约 |

---

## 5. Layout / Scale / L2 Cache 约定

### 5.1 Layout 与 Tensor

#### Layout 工厂（`QuantMatmulMxKernel` 内）

| 工厂 | 定义 | 说明 |
|:---|:---|:---|
| `MakeLayoutA` | `FrameLayoutFormat<LayoutA, Int<C0_SIZE>>` | A：ND 扩展布局（实例化 `NDExtLayoutPtn`） |
| `MakeLayoutB` | `FrameLayoutFormat<LayoutB, Int<C0_SIZE>>` | B：DN 扩展布局（实例化 `DNExtLayoutPtn`，NZ） |
| `MakeLayoutC` | `FrameLayoutFormat<LayoutC, Int<C0_SIZE>>` | C：ND 扩展布局 |
| `MakeLayoutScaleA` | `transA ? ScaleADNLayoutPtn : ScaleANDLayoutPtn`，`Int<SCALE_C0>` | 按 `transA` 条件选择 |
| `MakeLayoutScaleB` | `transB ? ScaleBDNLayoutPtn : ScaleBNDLayoutPtn`，`Int<SCALE_C0>` | 按 `transB` 条件选择 |

> AG kernel（`qmm_mx_kernel_ag_udma.h`）中 `MakeLayoutC` 改用 `Int<C0_ELEMENT<CType>>`；Bias 布局用 `MakeFrameLayout<NDExtLayoutPtn>(1, N)`。

#### 全局形状约定（`ProcessSingleBatch` 内）

| Tensor | 全局形状 | 语义 |
|:---|:---|:---|
| `layoutA` / `layoutALocal` | `(rankSize * originalM, K)` | 通信 buffer / 本地 A 均按 rank 沿 M 轴拼接 |
| `layoutScaleA` | `(rankSize * originalM, scaleKLen)` | ScaleA 同形 |
| `layoutB` | `(rankSize * K, N)` | B 按 rank 沿 K 轴拼接（rank r 的 B 段起点 `r * K`） |
| `layoutScaleB` | `(rankSize * scaleKLen, N)` | ScaleB 同形 |
| `layoutC` | `(M, N)` | 输出 |

#### 关键不变量

| # | 不变量 |
|:---|:---|
| 1 | 官网实例化 `transA=false, transB=true`（`BlockMmad::TRANS_A/TRANS_B`） |
| 2 | `ProblemShape = <M, N, K, B>`；PUT kernel 中 `B=1`（`problemShape{m, n, k, 1UL}`） |
| 3 | `C0_SIZE`：FP8=32，FP4=64；`SCALE_C0=2` |
| 4 | Tensor 创建模式：`MakeTensor(MakeMemPtr<Location::GM>(addr), layout)` → `Slice(MakeCoord(mPos, kPos), MakeShape(tileM, K))` |

### 5.2 Scale 处理

#### MX FP8 量化 Scale

- A/B 为 FP8（`fp8_e4m3fn_t` / `fp8_e5m2_t`），Scale 为 `fp8_e8m0_t`（E8M0）
- Scale 沿 K 轴分组：每 `MXFP_DIVISOR_SIZE`（64）个 K 元素共享一个 scale
- 每行 scale 元素数：`scaleKLen = CeilDiv(K, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE`（`ProcessSingleBatch` 内计算）

#### Scale 通信策略（PUT 模式）

| Scale | 位置 | 通信 |
|:---|:---|:---|
| ScaleA（远端部分） | 通信 buffer（`scaleAGmAddr_`） | **参与通信**：hcomm impl 中作为独立 `AlltoAll` 任务下发（`scaleHandle_`），输出到 workspace `commX1ScaleGM1_` |
| ScaleA（本 rank） | 本卡 GM（`localScaleAGmAddr_`） | 不通信 |
| ScaleB | 本卡 GM（`scaleBGmAddr_`） | 不通信（每 rank 持有完整 B 的 ScaleB） |

`HcommCommWaitPolicy::WaitTile` 在 `tileIdx==0` 时先 Wait `scaleHandle_`——首次 data wait 前确保 scale 就绪，用 matmul 头开销掩盖 scale 通信。

#### Scale 在 Blaze 中的使用

`mmadOp_` 的第 3/4 参为 scaleA/scaleB 的 GM tensor slice；`MatmulWithScaleMx` dispatch policy 保证 BlockMmad 内部正确处理 MX scale。调用签名见 §4.2。

### 5.3 L2 Cache 优化

`QuantMatmulMxKernel::SetL2Cache` / `SetScaleL2Cache`（`apace/kernel/all_to_all_quant_matmul/quant_matmul_mx_kernel.h`）实现基于 cache line 对齐的 L2 hint。

#### 策略

| 数据 | 策略 | 条件 |
|:---|:---|:---|
| C 输出 | `CACHE_MODE_DISABLE` | `isAtomicAdd_` 为 true 时无条件（函数入口） |
| ScaleB | `DISABLE` / `NORMAL` | 按行字节数 cache line 对齐判断（见下） |
| B 矩阵 | `DISABLE` / `NORMAL` | `weightNz` 分支无条件 `DISABLE`；非 weightNz 按对齐判断（见下） |

#### 分支结构

- `SetL2Cache` 被 `fullMTile`（`curBaseM >= M`）gate：非 full M tile 直接返回
- `SetScaleL2Cache` 在 `Get<MNK_B>(problemShape) != 1` 时提前返回
- `transB=true`：判断 `scaleKRowBytes` 和 `scaleKL1RowBytes` 两个行字节数都对齐
- `transB=false`：判断 `scaleNStrideBytes = N * MXFP_MULTI_BASE_SIZE` 与 `scaleBaseNStrideBytes = baseN * MXFP_MULTI_BASE_SIZE` 对齐
- B 矩阵非 weightNz：`transB=true` 判 `K` 对齐；`transB=false` 判 `N` 和 `baseN` 同时对齐

#### 对齐判断公式

```cpp
// B 矩阵（非 weightNz, transB=true）:
bool bAlignForL2Stream = (K & kCacheLineAlignMask) == 0;
// 非 weightNz, transB=false:
bool bAlignForL2Stream = (N & kCacheLineAlignMask) == 0 && (baseN & kCacheLineAlignMask) == 0;
// kCacheLineAlignMask: FP8→0x7f (128B), FP4→0xff (256B)

// ScaleB（transB=true），两个行字节数都须对齐:
const int64_t scaleKRowBytes  = CeilDiv(K, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
const int64_t scaleKL1RowBytes = CeilDiv(scaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
bool scaleAlignForL2Stream = (scaleKRowBytes & kCacheLineAlignMask) == 0 &&
                             (scaleKL1RowBytes & kCacheLineAlignMask) == 0;
```

#### L2 Cache 不变量

| # | 不变量 |
|:---|:---|
| 1 | 对齐时禁用 L2 Cache（流式访问不缓存），不对齐时正常缓存 |
| 2 | 原子累加场景 C 必须禁用 L2 Cache（`isAtomicAdd_` 入口处理），避免多核写冲突经 L2 放大 |

---

## 6. 关键常量

来自 `apace/utils/constant.h`（host/device 共享）与 Blaze 头文件：

| 常量 | 值 | 含义 |
|:---|:---|:---|
| `MXFP_DIVISOR_SIZE` | 64 | MX scale 分组大小 |
| `MXFP_MULTI_BASE_SIZE` | 2 | MX scale 倍数 |
| `FP8_C0_SIZE` / `FP4_C0_SIZE` | 32 / 64 | FP8/FP4 C0 维度大小 |
| `SCALE_C0` | 2 | Scale C0 维度大小（kernel 类内定义） |
| `CUBE_BLOCK` | 16 | Cube 块大小 |
| `BASIC_BLOCK_SIZE_16/64/128/256/512` | 16/64/128/256/512 | host 侧 tiling baseM/baseN 搜索候选 |
| `MTE1_MTE2_EVENT_ID_NUM` / `MTE1_MTE2_EVENT_ID_NUM_MX` | 4 / 6 | MTE1→MTE2 事件 ID 数 |
| `MTE2_CACHELINE_SIZE` / `L2_ALIGN_SIZE` | 128 / 128 | cache line 对齐粒度 |
| `CACHELINE` | 512 | 对齐粒度 |
| `FINAL_ACCUMULATION` / `NON_FINAL_ACCUMULATION` | 3 / 2 | mmad 累加模式选择子 |

> kernel 内另有 `C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8` 与 `kCacheLineAlignMask`，两个 constexpr 均定义于 kernel 类内：`C0_SIZE` 引用的 `C0_SIZE_B4`/`C0_SIZE_B8` 来自 Blaze `common_utils.h`（ops-tensor）；`kCacheLineAlignMask` 是 kernel 类自有常量（Blaze `common_utils.h` 中不存在），`QuantMatmulMxKernel`（`quant_matmul_mx_kernel.h`）按 FP8/FP4 取 `0x7f`/`0xff`，`QmmMxKernelAgUdma`（`qmm_mx_kernel_ag_udma.h`）固定 `0x7f`。表中数值以头文件实际值为准。

---

## 7. FragmentTensor（AllGather 场景）

AllGather 算子（`apace/kernel/all_gather_quant_matmul/qmm_mx_kernel_ag_udma.h`，`QmmMxKernelAgUdma`）用 FragmentTensor 处理 AllGather 后的离散多 rank 数据，是除 `QuantMatmulMxKernel` 外的第二种组合模式 kernel。

### 7.1 AllGather 场景（通信在前）

#### FragmentTensor 接口（`apace/basic/fragment_tensor/fragment_tensor.h`）

| 接口 | 语义 | 约束 |
|:---|:---|:---|
| `FragmentTensor<Dims, MaxCnt, LayoutFactory, T>` | 沿 split axis 拼接多个离散 GM 段的虚拟 Tensor；`Slice` 零搬运返回子视图 | `MaxCnt` 用 `Apace::Basic::MAX_FRAGMENT_COUNT` |
| `MakeFragmentTensor<...>(fragParam, addrList)` | 创建入口（参数结构 + 地址表） | ⚠️ `addrList` 只存指针不拷贝——须在整个 FragmentTensor 生命周期内有效 |
| `FragmentParam<Dims>` | `{assembleAxis, assembledShape[Dims], fragmentSize, realFragmentSize, fragmentCnt}` | `Validate()` 不变量：`assembledShape[assembleAxis] == fragmentSize × fragmentCnt`、`0 < realFragmentSize ≤ fragmentSize`、`fragmentCnt ≠ 0`——padding 用 `fragmentSize > realFragmentSize` 表达（AG 尾块 paddedTailM/tailM 即此用法） |
| `GetFragment(idx)` / `GetFragmentAddr(idx)` / `GetFragmentCnt()` | 查询 fragment 段 | — |
| `UpdateAddrList(addrList)` | 轮次推进时原地更新各段地址 | 更新前必须确保旧地址不再被读（先 wait） |
| `Slice(coord, shape)` | 虚拟切片 | 跨 fragment 的 slice 由内部 `FragmentComposition` 解析 |
| `FragmentSliceCopy<isScatter>(copyHandle, tensor, fragmentTensor)` | 遍历离散 fragment 逐段 copy：`isScatter=false` 为 frag→tensor gather（GM→L1）；`isScatter=true` 为 tensor→frag scatter（L0C→GM fixpipe，按 `realFragmentSize` 截尾） | 是 shipped 代码中唯一的 fragment 遍历 API；设计文档中的 `ForEachFragment` 仅为伪码，仓内无此 API |

### QmmMxKernelAgUdma 区域模型

| 概念 | 语义 |
|:---|:---|
| `RegionTag { HEAD, MAIN, TAIL }` | HEAD=本 rank 数据；MAIN=远端轮次数据；TAIL=尾块 |
| `TileCtx` | `ResolveTileCtx(mPos, ...)` 返回当前 tile 的 `region` / `dependTileIdx` / `roundIdx` / 对应 FragmentTensor 指针 |
| dependTileIdx 映射 | HEAD → `0`（AIV 循环前预触发，AIC 经统一去重路径 wait 时立即返回）；MAIN round r → `r + 1`；TAIL → `commTurn` |
| 位掩码去重 wait | `waitedMask` 记录已 wait 的 dependTileIdx，`CrossCoreWaitFlag<0x2, PIPE_MTE2>(dependTileIdx)` 每个 id 只执行一次；循环末尾 drain 补齐 |

### QmmMxBlockMmadFragment（`apace/block/blaze_ext/gemm/block/qmm_mx_block_mmad_fragment.h`）

- 独立类（非 `BlockMmad` 偏特化），AG kernel 中以 `QmmMxBlockMmadFragment<0, false, AType, LayoutA, ...>` 实例化，内部 `DispatchPolicy = MatmulWithScaleMx<A_FULL_LOAD_MODE, ATOMIC_ADD>`
- `CopyAInL1` 内用 `FragmentSliceCopy<false>`（gather）把离散 fragment 的 A/ScaleA 搬入 L1；C 写出用 `FragmentSliceCopy<true>`（scatter，L0C→GM fixpipe）
- 架构门控：类定义整体位于 `#if __NPU_ARCH__ == 3510` 内，门外仅有前向声明——非 3510 目标实例化该类触发不完整类型编译错误（比空定义更安全的失败模式）

#### operator() 完整调用契约（17 参，自研 kernel 照抄锚点）

`Init(problemShape, l0TileShape{baseM,baseN,baseK,0}, l1Params{stepK*baseK, scaleKL1, nBufferNum}, isBias, dbL0c>1)` 后按以下顺序传参（`qmm_mx_kernel_ag_udma.h:403-406`）：

```
mmadFrag(blockA,              // FragmentTensor Slice（A，HEAD/MAIN/TAIL 区对应 frag）
         gmBlockB,            // 普通 GM tensor（B 全卡共享）
         blockScaleA,         // FragmentTensor Slice（scaleA）
         gmBlockScaleB,       // 普通 GM tensor
         gmBlockBias,         // 普通 GM tensor（可为 null tensor 空占位）
         cFragAddrs,          // GM_ADDR* 各 rank 的 C 输出基址数组（AG 为 [8] 定长成员）
         mPerRank, tileM, tileCnt, tailM,   // 区域划分参数
         remoteRankCnt,       // 本 tile 参与的 rank 数（控制 L0C 累加/fixpipe 时机）
         Ni,                  // C 行宽 N
         singleShape, regionMPos, nPos,     // 调度形状与坐标
         /*splitKIdx=*/0, blockC)           // blockC：FragmentTensor Slice（C 侧）
```

### 约束与违反后果

| # | 约束 | 违反后果 |
|:---|:---|:---|
| 1 | wait 与 AIV set 的 dependTileIdx 序列必须一一对应（含末尾 drain） | 死锁或读脏数据 |
| 2 | MAIN 轮次切换先 wait 再 `UpdateAddrList` | 地址更新后旧数据未消费完，精度错误 |
| 3 | HEAD 区 dependId=0 由 AIV 循环前预触发；AIC 仍经统一位掩码去重路径 wait 一次 id 0（因已预触发而立即返回，不阻塞） | 特判跳过 id 0 会破坏位掩码/drain 的统一配对假设 |
| 4 | AG 的 `SetL2Cache` 与 A2A 差异：无 isAtomicAdd 分支；gmB 按 DN 布局（transB）以 K 轴 128B 对齐判 streaming；scaleB 受 `fullMBlock && (N*scaleC0 与 baseN*scaleC0 均 128B 对齐)` **双 gate** | L2 命中率退化（大 B/scaleB 冲刷 cache） |

> MatmulMode 三模式在共享 kernel `ProcessSingleBatch` 的分支行为（LOCAL 读 localAGmAddr_ 且 B 只取本 rank K 段 / REMOTE 遍历全部 rank、`localMatmul==0` 时 self 段改读本地 GM / DEFERRED_SYNC 三相 wait 夹逼）详见 [`fusion.md`](fusion.md) §5.1；本节只列 AG 路径差异（无 localMatmul 字段、单 REMOTE 编排 + 三区 FragmentTensor）。

### 7.2 计算在前场景（ReduceScatter 语义）

> **选型提醒**：计算在前算子的 mm 内核**默认 FragmentTensor 自研 kernel**（消 R 循环），与 [`fusion.md`](fusion.md) §6.2.2 及 R16 红线一致；vendor 复用官方 `QuantMatmulMxKernel` 为例外路径（须 SCALAR 占比论证）。
>
> **vendor 复用不可行说明（类型不兼容）**：`QuantMatmulMxKernel` 的 `mmadParams.cGmAddr` 为 `GM_ADDR` 类型（单地址），**无法接受 FragmentTensor 对象**（多段地址数组）——C 输出无法 scatter 到 R 个 rank 段。只有 `QmmMxBlockMmadFragment` 配合自研 kernel 才支持 FragmentTensor C 输出（`operator()` 接受 `GM_ADDR* cFragAddrs` 数组，内部经 `FragmentSliceCopy<true>` 多段 scatter 写出并按 `realFragmentSize` 截尾）。此类型不兼容属设计阶段即可拦截的阻塞级选型错误。

FragmentTensor 的另一种应用：**计算在前**场景——A 数据全在本卡 GM 连续 `[m, k]`，各 rank 段天然连续排列，FragmentTensor 打包本卡 GM 地址（vs AllGather 打包 win 区离散地址）。

**与 AllGather 的关键差异**：

| 维度 | AllGather（通信在前，§7.1） | compute-first（计算在前，本节） |
|:---|:---|:---|
| 数据来源 | win 区（远端 rank 推来的离散地址） | 本卡 GM（连续 rank 段） |
| FragmentTensor 打包 | win 区各 rank 地址 | GM 中各 rank 段地址 |
| 跨核同步 | 需 WaitFlag 等通信完成 | **无需**（计算不依赖通信） |
| C 输出 | 直写 yGm | 写 workspace（后续 PUT 通信） |
| dependId 预触发 | HEAD 区 dependId=0 由 AIV 循环前预触发 | 无 dependId 预触发（计算不依赖通信） |

**BuildFragmentTensors 模式**：

```cpp
// 打包 R 个 rank 段地址（本卡 GM 连续）
for (uint32_t r = 0; r < rankSize; ++r) {
    addrListA[r] = params.aGM + r * mPerRank * dataBytesPerMRow;
    addrListC[r] = cGM + r * mPerRank * cBytesPerM;  // workspace 各 rank 段
}
headFragA_ = MakeFragmentTensor<2, MAX_FRAG, MakeLayoutA, AType>(
    MakeFragParam(fragM, realFragM, rankSize, k), addrListA);
```

- `fragM = headRows / rankSize = curTileM`（每个 rank 段在当前 tile 的行数）
- `realFragmentSize`：尾块非对齐时限制实际读取范围（见 [`fusion.md`](fusion.md) §6.2.7）

**ResolveTileCtx 的 HEAD/MAIN/TAIL 划分**：

| Region | 触发条件 | FragmentTensor | 行偏移 |
|:---|:---|:---|:---|
| HEAD | `mPos < headRows` | `headFragA_` / `headFragC_` | `mPos` |
| MAIN | `mPos < headRows + mainSectionRows` | `curMainA_` / `curMainC_`（按 `roundIdx` 更新地址） | `relPos % mainRoundRows` |
| TAIL | `mPos ≥ headRows + mainSectionRows` | `tailFragA_` / `tailFragC_` | `mPos - headRows - mainSectionRows` |

> per-tile 调用场景下通常只有 HEAD region（`tileCnt=0, tailCnt=0`），MAIN/TAIL 在多轮 per-tile 流水中通过外部 for-t 循环处理。

**Te::MakeTensor 构建 GM Tensor**：

计算在前场景的 B/ScaleB 从本卡 GM 读取（所有 rank 共享），用 `Te::MakeTensor` 构建 GM Tensor 后 Slice 传入 BlockMmadFragC：

```cpp
auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(bGmAddr_),
            MakeLayoutB{}(Ki, Ni));
auto gmBlockB = gmB.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(Ki, curNtile));
```

> 归约侧 UB/精度/双缓冲纪律（两种 UB 管理策略）见 [`fusion.md`](fusion.md) §6.2.6。

---

## 8. 排错速查

| 症状 | 可能原因 | 检查方法 |
|:---|:---|:---|
| 编译错误：`BlockMmad` 未找到 | ops-tensor 未拉取 | 检查 `cmake/third_party/ops-tensor.cmake` FetchContent 是否生效 |
| 编译错误：`WaitTile` 未找到 | CommPolicy 未实现契约接口 | CommPolicy 必须提供 `__aicore__ void WaitTile(uint32_t)` |
| 精度错误：C 全 0 | Scale 未正确传入 | 检查 `mmadParams.scaleAGmAddr/scaleBGmAddr` 与 `localScaleAGmAddr` 是否按模式指向通信 buffer / 本地地址 |
| 精度错误：部分 rank C 错误 | rank 遍历地址计算 | 检查 `actualMPos = rank * oriM + mPos` 与 B 段起点 `rank * K` |
| 精度错误：localMatmul=1 结果错 | 原子累加未生效 | 确认 REMOTE 阶段 `isAtomicAdd_` 置位（`SetAtomicAdd`/`SetAtomicNone` 配对）且 `splitKNum=rankSize-1` |
| 精度错误：DEFERRED_SYNC 结果错 | L0C reset/累次序错 | Phase 1 第 8 参必须为 `0`，Phase 3 从 1 递增 |
| 死锁 | WaitTile 与通信 set 不配对 | REMOTE/DEFERRED_SYNC 路径必须执行末尾 drain；UDMA 路径 flag idx 与 AIV set 一致 |
| 读脏数据（偶发精度错） | wait 时机过早/缺失 | DEFERRED_SYNC 的 wait 必须在 self mmad 之后；dependTileIdx 映射检查 `CalcDependTileIdx` |
| AIV 误参与 HCCL/地址初始化 | 缺 AIC 守卫 | `Init`/`ResetGmAddr` 依赖 `ASCEND_IS_AIV` 提前返回；新增 AIV 路径须自行加守卫（hcomm impl 文件头注释明示） |
| 性能差：L2 Hit Rate 低 | L2 hint 未生效 | 检查 `SetL2Cache` 的 `fullMTile` gate 与对齐条件 |
| 尾块精度错误 | 尾块拆分未生效 | 检查 `needUpdateTail_` / `bs.UpdateTailTile(mTailTile, nTailTile)` 是否触发 |

---

## 9. 集成 Checklist

新算子接入 Blaze/apace 计算层时逐项核对：

| # | 检查项 | 说明 |
|:---|:---|:---|
| 1 | ops-tensor 拉取 | `cmake/third_party/ops-tensor.cmake`（FetchContent，来源 gitcode.com/cann/ops-tensor） |
| 2 | Mix 算子任务类型 | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)`（见 `apace/tests/st/all_to_all_quant_matmul/src/kernel_launcher.h`） |
| 3 | 架构门控 | `blaze_ext/gemm/block/qmm_mx_block_mmad_fragment.h` 类定义在 `#if __NPU_ARCH__ == 3510` 内（门外仅前向声明），非 3510 实例化即编译错误 |
| 4 | ST 验证入口 | `apace/tests/st/{op}/`（官网现有 `all_to_all_quant_matmul`、`all_gather_quant_matmul` 两个算子） |
| 5 | CommPolicy 接线 | 策略类实现 `WaitTile(uint32_t)`；Impl 经 `GetCommPolicy()` 绑定通信状态（如 `state_` 指针） |
| 6 | 模式选择 | tiling `localMatmul` ∈ {0,1,2} 与 Impl 编排一致（见 §4.4）；REMOTE 阶段 `splitKNum` 按模式设置 |
| 7 | l1Params 字段对应 | `l1Params = {stepK*baseK, scaleKL1, nBufferNum}`，与 tiling 字段一一对应 |

---

## 后续阅读

- [`fusion.md`](fusion.md) — 通算融合组合模式（localMatmul 选型/flag 编排）
- [`operator-anatomy.md`](../operator-design/operator-anatomy.md) — 算子完整骨架（Impl/入口/host）
- `ascendc-api-best-practices` skill `references/` — 基础 API（CrossCore flag 编排见 `ascendc-api-best-practices` skill `references/api-crosscore-sync.md`）
- `ascendc-blaze-best-practice` skill — Blaze 通用最佳实践（MX 量化 matmul 与 Tiling 算法选型见其 scenarios 与 kernel-design 参考）
