# apace 算子解剖（kernel 侧）

> 本文档描述 apace 算子通用骨架（文件构成、tiling_data、Impl 类、入口函数），以官方已实现算子（AllToAll PUT / AllGather PUT）为示例。
>
> ⚠️ **子路径与符号名为逻辑引用，物理形态以实测为准**：文中 `block/aiv_comm/...`、`kernel/<op>/`、`..._udma_impl.h`、`...UdmaImpl` 等路径/文件/类名是官网快照的逻辑引用，CANN 内置树中物理位置与命名随版本漂移（如某版本通信层在 `core/aiv_comm/`、文件为 `..._urma_impl.h`、目录为 `kernel/fusions/<op>/`）。引用作证据时，以 Step 1 实测登记的实际路径/符号为准核对，禁止按本文写死。

## 目录

1. [已实现算子共性模式](#1-已实现算子共性模式)
2. [算子文件骨架](#2-算子文件骨架)
3. [Tiling 结构与 Host 填充](#3-tiling-结构与-host-填充)
4. [Impl 类契约](#4-impl-类契约)
5. [入口函数规则](#5-入口函数规则)
6. [AllGather 变体与 CCU 变体](#6-allgather-变体与-ccu-变体)
7. [计算在前算子解剖（compute-first 直调模式）](#7-计算在前算子解剖compute-first-直调模式)

---

## 1. 已实现算子共性模式

以下共性以官方 AllToAll / AllGather 两个 PUT 算子为示例。

### 成员持有

| 成员 | A2A PUT | AG PUT |
|:---|:---|:---|
| 计算 kernel | `quantMatmulKernelImpl_`（`QuantMatmulMxKernel<..., UdmaCommWaitPolicy>`） | `quantMatmulKernelImpl_`（`QmmMxKernelAgUdma<...>`） |
| data 通信对象 | `allToAllA_`（`CollectiveComm<AllToAll, PUT, AType, TeamBarrier>`） | `allGatherData_`（`CollectiveComm<AllGather, PUT, AType, TeamBarrier>`） |
| scale 通信对象 | `allToAllScaleA_`（`CollectiveComm<AllToAll, PUT, fp8_e8m0_t, TeamBarrier>`） | `allGatherScale_`（`CollectiveComm<AllGather, PUT, fp8_e8m0_t, TeamBarrier>`） |
| 跨卡同步 | `teamBarrier_`（`TeamBarrier`） | `teamBarrier_`（`TeamBarrier`） |
| 通信上下文 | `udmaCtx_` + `syncBuffer_`（= `&hcommCtx->udmaCtx / ubmemCtx`） | `udmaCtx_` + `ubmemCtx_`（另持有 `hcommCtx_` 原指针） |
| GM 基址 | `baseParams_.aGm / scaleAGm / bGm / scaleBGm / cGm` | `aGM_ / aScaleGM_ / bGM_ / bScaleGM_ / cGM_` |
| tiling | `const allToAllMatmulTilingData* tilingData_` | `const AllGatherMxMatmulUdmaTilingData* tilingData_` |

### Init 序列

```
保存 GM 地址与 tilingData 指针
  → 提取 udmaCtx_ / ubmemCtx_（rankId / rankSize 取自 udmaCtx_）
  → InitBaseParams() 推导（commTurn、headMSize/tileM、rankDataBytes/dataRegionBytes 等）
  → UB 静态分配（局部 ubOffset 线性累加：commBuf 512B + commScaleBuf 512B + barrierBuf 32B）
  → teamBarrier_.Init(barrierBuf, ubmemCtx, rankSize, GetBlockIdx())
  → data 通信 Init<BARRIER_NONE>（winOffset 缺省 = 0，Win 区起始段）
  → scale 通信 Init（winOffset = data 段字节数：A2A 为 rankSize × rankDataBytes，AG 为 dataRegionBytes_）
```

> AG 在 Init 末尾额外有一次 `SyncAll<true>()`；AG 的 `commTilingData_ / commTilingScale_` 是成员变量、在 Init 内由 `tilingData->commTile` 重推导后传给通信对象，A2A 则直接把 tiling 字段传给通信对象 Init。另注意 **Init 步骤顺序两算子不同**：A2A 先提取 ctx（rankId/rankSize）再 `InitBaseParams()`；AG 先 `InitBaseParams(tilingData)` 再提取 ctx，且 `dataRegionBytes_/scaleRegionBytes_` 在 ctx 提取之后于 Init 内计算（不在 InitBaseParams 中）——上面序列以 A2A 为基准展开，移植 AG 时以 `all_gather_mx_matmul_udma_impl.h::Init` 源码顺序为准。

### Run 编排

编译期 `ASCEND_IS_AIV` / `ASCEND_IS_AIC` 分支：**AIV 通信先行**（scale 先 Commit、data 后 Commit、同 channel 只 Wait 一次、`SyncAll<true>()`、`CrossCoreSetFlag<0x2, PIPE_MTE3>` 通知 AIC），**AIC 计算等待**（经 `WaitTile` / `CrossCoreWaitFlag<0x2, PIPE_MTE2>` 消费通信 tile，waitedMask 按位去重 + 尾部兜底）。

### 差异点表

| 维度 | A2A PUT | AG PUT |
|:---|:---|:---|
| 通信原语 | AllToAll | AllGather |
| 切分轴 | K（每 rank 持 Ka = K/rankNum，通信沿 M 推送） | M（`tileM = min(m, 512)`，不切 K） |
| AIC 编排 | `Run()`：`localMatmul` 0/1/2 分支（`RunLocalMatmul` + `RunMatmul`） | `Process()`：`FragmentTensor` HEAD/MAIN/TAIL 虚拟重排 + dependId 预触发 |
| `__global__` 入口 | 4 个 dtype 变体（`tests/st/all_to_all_quant_matmul/src/kernel_launcher.h`） | 1 个（impl 头文件末尾 `AllGatherQuantMatmulKernel`） |
| Impl 模板参数 | 5 个：`<AType, BType, CType, TransA, TransB>` | 3 个：`<AType, BType, CType>`（无转置参数） |
| scale Init barrier | 缺省 BarrierMode | 显式 `Init<BARRIER_DEVICE>` |
| 命名空间 | `Apace`（CommContext 在全局命名空间） | `AllGatherQuantMatmulImpl`（CommContext 在 `Apace::AivComm` 命名空间） |

---

## 2. 算子文件骨架

一个 apace 算子的 kernel 侧由四类文件构成（以官网两个算子为实例）：

| 文件 | 职责 | 官网实例 |
|:---|:---|:---|
| `<op>_tiling_data.h` | 算子级 tiling 结构 + `CommContext` 聚合体定义 | `kernel/all_to_all_quant_matmul/all_to_all_matmul_tiling_data.h`、`kernel/all_gather_quant_matmul/all_gather_mx_matmul_udma_tiling_data.h` |
| qbmm kernel 头文件 | Blaze matmul 调度（CommPolicy 注入）、tile 依赖解析 | `kernel/all_to_all_quant_matmul/quant_matmul_mx_kernel.h`、`kernel/all_gather_quant_matmul/qmm_mx_kernel_ag_udma.h` |
| `<op>_impl.h` | Impl 主类（Init + Run/Process） | `all_to_all_mx_quant_matmul_udma_impl.h`、`all_gather_mx_matmul_udma_impl.h` |
| `__global__` 入口 | 实例化 + Init + Run/Process | A2A：ST 的 `kernel_launcher.h`；AG：impl 头文件末尾（见 §5） |

### 依赖层次

```
外部库 (adv_api/hcomm, blaze, kernel_tiling)
    ↑
共享层 (block/aiv_comm/*, tiling/*)
    ↑
算子层 (quant_matmul_mx_kernel.h, all_to_all_matmul_tiling_data.h)
    ↑
Impl (all_to_all_mx_quant_matmul_udma_impl.h)
```

| 层次 | 关键头文件 | 提供能力 |
|:---|:---|:---|
| 外部库 | `adv_api/hcomm/hcomm.h`, `blaze/gemm/block/block_mmad_qbmm_mx.h`, `kernel_tiling/kernel_tiling.h` | Hcomm 底层通信、Blaze matmul、kernel 基础设施 |
| 共享层 | `block/aiv_comm/collective_comm_api.h`, `block/aiv_comm/collective_comm_context.h`, `tiling/comm_tiling_data.h`, `block/aiv_comm/barrier/barrier_ubmem.h` | CollectiveComm、CommContext、CommTilingData、TeamBarrier |
| 算子层 | `kernel/all_to_all_quant_matmul/quant_matmul_mx_kernel.h`, `kernel/all_to_all_quant_matmul/all_to_all_matmul_tiling_data.h` | Blaze matmul 调度（CommPolicy 注入）、本算子 tiling 结构 |

---

## 3. Tiling 结构与 Host 填充

### 3.1 三层 tiling 组合

apace tiling 数据结构采用三层组合：

```
Layer 1: CommTilingData (apace/tiling/comm_tiling_data.h)               ← 通信切分参数（5 字段）
Layer 2: QuantMatmulTilingData (apace/tiling/quant_matmul_tiling_data.h) ← 计算切分参数（17 字段）
Layer 3: 算子级 tiling_data (apace/kernel/<op>/<op>_tiling_data.h)        ← kernel 专属组合
```

组合关系图（官网两个算子）：

```
allToAllMatmulTilingData                 AllGatherMxMatmulUdmaTilingData
(apace/kernel/all_to_all_quant_matmul/   (apace/kernel/all_gather_quant_matmul/
  all_to_all_matmul_tiling_data.h)         all_gather_mx_matmul_udma_tiling_data.h)
├── CommTilingData commTilingData        ├── QuantMatmulTilingData mmTile   ← Layer 2
├── CommTilingData scaleCommTilingData   └── CommTilingData commTile        ← Layer 1
├── QuantMatmulTilingData tileQbmmTilingData
└── uint32_t localMatmul
```

> 官网 master 快照的 tiling/ 目录为 `comm_tiling_data.h`、`quant_matmul_tiling_{base,common,data,swat}.h`，无 `comm_tiling_base.h`（headMSize 推导内联在 ST main.cpp）；⚠️ CANN 9.1.0 内置版 tiling/ **另有** `comm_tiling_base.h`（`apace::CommTilingBase::GetCommTilingData`，将 headMSize 推导封装为基类）——两版本该文件存在差异，以 Step 1 实测登记的事实源为准（见 §3.7）。

### 3.2 CommTilingData（通信切分）

`CommTilingData` 是 apace 统一的通信切分参数，同时被 host 端切分算法和 kernel 端通信实现使用。定义在 `apace/tiling/comm_tiling_data.h`。

字段语义与不变量：

| 字段 | 含义 | 单位 | 不变量 |
|:---|:---|:---|:---|
| `splitAxisTileSize` | 头块大小 | 元素个数 | 头块切分轴长度 |
| `splitAxisTileCnt` | 头块数量 | 个 | ≥ 1 |
| `splitAxisTailSize` | 尾块大小 | 元素个数 | 无尾块时 = 0 |
| `splitAxisTailCnt` | 尾块数量 | 个 | 无尾块时 = 0 |
| `nonSplitAxisSize` | 非切分轴大小 | 元素个数 | 所有内轴乘积 |

**切分不变量**：`切分轴总元素数 = splitAxisTileSize × splitAxisTileCnt + splitAxisTailSize × splitAxisTailCnt`；每个 tile 字节大小 = `splitAxisTileSize × nonSplitAxisSize × sizeof(dtype)`。

PUT 模式映射：A 按 K 轴切分（每 rank 持有**全部 M 行 × 本卡 Ka = K/rankNum 列段**，即 `[M_total, Ka]`），通信沿 M 轴逐 tile 推送 A 与 ScaleA（M 行块 r → rank r）。host 侧填充见官方 AllToAll ST main.cpp `runAllToAllMatmul`：

| 字段 | PUT data 通道 | PUT scale 通道 |
|:---|:---|:---|
| `splitAxisTileSize` | headMSize（= `CeilDiv(usedCoreNum, nTile) * baseM`） | 同 data |
| `splitAxisTileCnt` | headTileCnt = m / headMSize | 同 data |
| `splitAxisTailSize` | tailMSize = m % headMSize | 同 data |
| `splitAxisTailCnt` | tailMSize > 0 ? 1 : 0 | 同 data |
| `nonSplitAxisSize` | `ka`（per-rank K） | `ka / 32`（MXFP 压缩比） |

> scale 通道直接复制 data 通道后改 `nonSplitAxisSize = ka / 32`。

AllGather 模式映射：host 侧填充见官方 AllGather ST main.cpp `RunAllGatherQuantMatmul`：切分轴 = M 轴（`tileM = min(m, 512)`，基准 `TILE_M=512`，`m < tileM` 时收缩为 `m`），`nonSplitAxisSize = k`（全量 K，AllGather 不切 K）。

> ⚠️ AllGather host 侧**不填充** scale 通道的通信 tiling——AG tiling 结构只有 `mmTile` + `commTile`，scale 通道的 `nonSplitAxisSize = scaleKLen` 由 kernel 侧 `Init` 重新推导，并非遗漏。

kernel 端消费：kernel 从 `CommTilingData` 推导通信轮次与 tile 依赖（`apace/kernel/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_udma_impl.h` `InitBaseParams`）：

| 派生量 | 推导公式 |
|:---|:---|
| `commTurn`（总流水步数） | = `splitAxisTileCnt + splitAxisTailCnt` |
| `headMSize` | = `splitAxisTileSize` |
| `axisM` / `axisKa` | = `tileQbmmTilingData.m` / `.k` |

### 3.3 QuantMatmulTilingData（计算切分）

`QuantMatmulTilingData` 是量化 matmul 的 tiling POD 结构（`apace/tiling/quant_matmul_tiling_data.h`）。

字段分类：

| 类别 | 字段 | 用途 |
|:---|:---|:---|
| **问题形状** | m, n, k | 传给 Blaze ProblemShape |
| **Base tile** | baseM, baseN, baseK | Blaze tile 形状，由 SWAT tiling 推导 |
| **Scale** | scaleKL1 | L1 中 scale 的 K 长度 |
| **尾块** | mTailTile, nTailTile, mBaseTailSplitCnt, nBaseTailSplitCnt, mTailMain, nTailMain | 尾块调度参数 |
| **资源** | usedCoreNum, stepK, nBufferNum, dbL0c | 核数、K 步进、L1 buffer、L0C 双缓冲 |

`#pragma pack(push, 8)` 与 `alignas(8)`：

```cpp
#pragma pack(push, 8)
struct alignas(8) QuantMatmulTilingData { /* 字段见上表 */ };
#pragma pack(pop)
```

**不变量**：字段顺序是 host-device 契约的一部分（源文件注释明示 "field order is part of the host-device contract"），layout 稳定性比重排便利性更重要。8 字节对齐确保 host/device 结构体布局一致。修改字段顺序或增删字段视为 ABI break。

### 3.4 算子级 tiling 结构

PUT：allToAllMatmulTilingData，定义在 `apace/kernel/all_to_all_quant_matmul/all_to_all_matmul_tiling_data.h`：

| 字段 | 类型 | 用途 |
|:---|:---|:---|
| `commTilingData` | `CommTilingData` | data 通道通信切分 |
| `scaleCommTilingData` | `CommTilingData` | scale 通道通信切分（`nonSplitAxisSize = ka / 32`） |
| `tileQbmmTilingData` | `QuantMatmulTilingData` | 计算 tiling（头尾块共用一份，尾块由 Blaze 尾块调度字段处理） |
| `localMatmul` | `uint32_t` | 0：不使能 local 先行；1：使能 AtomicAdd（2 = DEFERRED_SYNC，见 [`fusion.md`](../fundamentals/fusion.md)） |

同文件另有 CCU 变体 `ccuAllToAllMatmulTilingData`（`mc2InitTiling` + `mc2CcTiling` + `commTilingData` + `tileQbmmTilingData` + `localMatmul`），走 CCU 通信，非 UDMA 直调路径。

AG：AllGatherMxMatmulUdmaTilingData，定义在 `apace/kernel/all_gather_quant_matmul/all_gather_mx_matmul_udma_tiling_data.h`：

| 字段 | 类型 | 用途 |
|:---|:---|:---|
| `mmTile` | `QuantMatmulTilingData` | 计算 tiling（对 `rankNum × 单卡逻辑 M` 的总形状推导） |
| `commTile` | `CommTilingData` | 通信切分（`tileM = min(m, 512)`，见 §3.2） |

整体带 `#pragma pack(push, 8)` + `alignas(8)`。

> ⚠️ 官网无 GET 样例与 ReduceScatter 样例；GET 的 tiling 与环形缓冲结构未上库，本文不展开（任何相关结构描述均为原理推导，须经源码验证后方可采用）。

### 3.5 Win 区地址布局

PUT 模式下 Win 区按 rank-major 布局存放通信到的 A/scaleA 数据：

```
本 rank Win 区 (selfWinAddr = udmaCtx_->commBufferAddrs[rankId])
┌──────────────────────────────────────────────┐
│  data 段：rankSize × rankDataBytes            │  ← winOffset = 0
│  （rank0 A 块 │ rank1 A 块 │ ... ）           │
├──────────────────────────────────────────────┤
│  scale 段：rankSize × scaleKaSize × axisM     │  ← winOffset = rankSize × rankDataBytes
└──────────────────────────────────────────────┘
```

| 角色 | 地址 | 锚点 |
|:---|:---|:---|
| AIC 读远端 A | `selfWinAddr`（`mmadParams.aGmAddr`） | `RunMatmul` |
| AIC 读远端 scaleA | `selfWinAddr + rankSize × rankDataBytes`（`mmadParams.scaleAGmAddr`） | `RunMatmul` |
| AIC 读本地 A | `baseParams_.aGm`（`localParams.localAGmAddr`，本地 GM 直读） | `RunMatmul` / kernel `gmALocal` |

> **Win 区容量校验（host 侧必须执行）**：Win 区需求 = rankSize ×（data 段 + scale 段）≤ HCCL 内置 buffer 容量，超出则建链后写越界。host 侧 tiling/launch 前必须完成该校验（与 `HcclGetHcclBuffer` 实测值比对）。

### 3.6 CommContext 通信上下文

`CommContext` 由 `CommUdmaContext`（UDMA 通信通道）和 `CommUbmemContext`（Barrier 通道）组成。`CommUdmaContext`/`CommUbmemContext` 与常量 `COMM_MAX_RANK_NUM=64`、`COMM_WORKSPACE_SIZE=512` 定义在 `apace/block/aiv_comm/collective_comm_context.h`；`CommContext` 聚合体由各算子 tiling_data.h 定义（PUT 在全局命名空间，AG 在 `Apace::AivComm` 命名空间）。

**结构概要**：`CommUdmaContext` 含 `rankId`/`rankSize`/每 rank 的 `channelHandles[]` 和 `commBufferAddrs[]`；`CommUbmemContext` 含 `rankId`/`rankSize`/每 rank 的 `commBufferAddrs[]`（barrier 缓冲区 GM 地址）。

> 完整字段表与不变量详见 [`communication.md`](../fundamentals/communication.md) §5 通信上下文 CommContext。

Host 侧构造：Host 侧通过 HCCL channel 创建工具（`apace/utils/comm_channel_builder.h` `CommChannelBuilder::CreateDeviceContext`）构造 `CommContext`，写入 GM 后传给 kernel：

1. 创建 HCCL channel（URMA 数据通道 / UBMEM barrier 通道）
2. 获取每 rank 的 channelHandle 和 commBufferAddr
3. 填充 `CommUdmaContext` 和 `CommUbmemContext`
4. `HcclEngineCtxCopy` 写入 GM，将 GM 指针作为 kernel 第一参数

> 详细步骤见 [`communication.md`](../fundamentals/communication.md) §6 Host 侧建链机制。

按值传递规则验收条件：

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | tiling 传递方式 | 算子级 tiling_data 在 `__global__` 入口**按值传递**（见 `apace/tests/st/all_to_all_quant_matmul/src/kernel_launcher.h` 4 个入口） |
| 2 | CommContext 传递方式 | `CommContext` 通过 `__gm__` 指针传递（kernel 第一参数） |
| 3 | CommContext 不按值传递的原因 | 含 rankSize 级数组，体积大且由 Host 动态构造 |
| 4 | Impl 解析 | `Init()` 从 `hcommCtx` 解析 `udmaCtx_`/`ubmemCtx_`/`rankId_`/`rankSize_`（PUT 见 `all_to_all_mx_quant_matmul_udma_impl.h` `Init`，AG 见 `all_gather_mx_matmul_udma_impl.h` `Init`）；tilingData 在 Impl 内以 `const *` 持有 |

### 3.7 Host 侧 tiling 推导

SWAT tiling 算法：

| 文件 | 作用 |
|:---|:---|
| `apace/tiling/quant_matmul_tiling_common.h` | `QuantMatmulPlatformInfo`（硬件信息）、`QuantMatmulArgs`（问题形状）、`QuantMatmulRunInfo`（中间状态） |
| `apace/tiling/quant_matmul_tiling_base.h` | `QuantMatmulTilingBase` 基类：驱动 `InitCompileInfo → InitShapeArgs → DoOpTiling → PrintTilingData` |
| `apace/tiling/quant_matmul_tiling_swat.h` | `QuantMatmulTilingSwat`：SWAT 策略，实现 `CalcBasicBlock`/`CalcTailBasicBlock`/`CalcPathSpecificL1`/`CalStepKs`/`CalScaleFactors` |

GetTilingData 签名：

```cpp
// apace/tiling/quant_matmul_tiling_base.h QuantMatmulTilingBase：
void GetTilingData(uint64_t m, uint64_t n, uint64_t k, bool transA, bool transB, QuantMatmulTilingData& tilingData);
// 4 参重载等价于 transA=false, transB=true：
void GetTilingData(uint64_t m, uint64_t n, uint64_t k, QuantMatmulTilingData& tilingData);
```

| 参数 | PUT（all_to_all） | AG（all_gather） |
|:---|:---|:---|
| `m` | 单卡 M | `totalLogicalM = rankNum × (tileCnt×tileM + tailCnt×paddedTailM)` |
| `n` | 全量 N | 全量 N |
| `k` | per-rank K（ka = K/rankNum） | 全量 K |
| 转置 | `false, true`（ST 显式传 6 参） | 4 参重载（默认 false, true） |
| 额外配置 | — | `SetOptimizeEnable(false)` + `SetMTailAlignEnable(true)` |

headMSize 推导（PUT，内联于 ST main.cpp）：官网无 `comm_tiling_base.h`；通信切分与 matmul tile 粒度的对齐由 ST main.cpp 内联完成（`runAllToAllMatmul`）：

```
nTile      = CeilDiv(n, baseN)
headMSize  = CeilDiv(usedCoreNum, nTile) * baseM   // 与 matmul tile 粒度对齐
headTileCnt = m / headMSize
tailMSize   = m % headMSize
```

**不变量**：headMSize 确保每个通信 tile 的数据量匹配一份核算力覆盖的 M-range，实现通信与计算流水并行；scale 通道的 `nonSplitAxisSize` 按 MXFP 压缩比（`ka / 32`）缩减，与 data 通道共享切分结构。

host 输入校验：

| ST | 校验 |
|:---|:---|
| PUT（all_to_all main.cpp `parseArguments`） | `k % rankNum == 0`；`CeilDiv(ka, 64) % 2 == 0` |
| AG（all_gather main.cpp `ParseArgs`） | `k % 32 == 0`（MXFP8 量化）；`CeilDiv(k, 64) % 2 == 0` |

推导验收条件：

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | 输入解析 | 问题形状 (M, K, N, rankNum) + ST 入参校验通过 |
| 2 | GetTilingData | 先调 `GetTilingData` 得 `usedCoreNum/baseM/baseN`（PUT 用 ka，AG 用 totalLogicalM） |
| 3 | CommTilingData 填充 | 5 字段已填充，满足切分不变量；headMSize 与 matmul tile 粒度对齐 |
| 4 | scale 通道 | 复用 data 切分，`nonSplitAxisSize = ka / 32`（PUT）或 `scaleKLen = CeilDiv(k,64)×2`（AG，见 `all_gather_mx_matmul_udma_impl.h` `Init`） |
| 5 | CommContext 构造 | 通过 `CommChannelBuilder::CreateDeviceContext` 填充（详见 communication.md §6） |
| 6 | launch | tilingData 按值传递，CommContext 按指针传递 |

Scale 内存分配（PUT 模式）：Scale 的内存分配跟随其对应矩阵的数据分布。PUT 模式（A 按 K 轴切分，B 全量复制）：

| Scale | 分布模式 | 公式 | 说明 |
|:---|:---|:---|:---|
| ScaleA | K-split（随 A 切分） | `CeilDiv(ka, 64) × 2 × M` | ka = K/rankSize |
| ScaleB | 全量复制（随 B 复制） | `CeilDiv(K, 64) × 2 × N` | K = 完整 K |

> ⚠️ ScaleA 的**逻辑分布**为 per-rank `ka`；但 PUT ST main.cpp 实际按**全量 K** 分配 host 缓冲（`m * CeilDiv(k, 64) * 2`）——逻辑分布与 ST 分配有差异，以 ST 实测为准。

**常见错误**：PUT 模式下 ScaleB 全量复制，如果误用 `CeilDiv(ka, 64)` 会导致尺寸缩小 rankSize 倍，ReadFile 失败或读取越界。

> B 数据分布模式详见 [`architecture.md`](../fundamentals/architecture.md) §4 B 数据分布模式。

### 3.8 HCCL windows 模式例外

HCCL windows 模式（`GetHcclContext`）不支持直调：

- **不需要** `CommContext` 结构
- **不需要** `CommUdmaContext` / `CommUbmemContext`
- tiling_data.h 中**不定义** `CommContext`

> **注意**：官网 apace kernel/ 当前两个算子（all_to_all_quant_matmul、all_gather_quant_matmul）均为 UDMA 模式，使用 `CommContext`，不走 HCCL windows。ReduceScatter 语义的生产实现（3 级流水 + workspace）见 [`fusion.md`](../fundamentals/fusion.md) §6.2。

验收条件：

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | UDMA 模式判定 | tiling_data.h 含 `CommContext` 定义，kernel 入口含 `__gm__ CommContext*` 首参 |
| 2 | HCCL windows 模式判定 | 无 `__global__` 直调入口，不支持 CANNBot Kernel 直调工作流 |

---

## 4. Impl 类契约

主教学样例为官网 PUT 算子 `all_to_all_quant_matmul`（`kernel/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_udma_impl.h`）。

### 4.1 模板参数

| 参数 | 含义 | 取值 |
|:---|:---|:---|
| `AType` / `BType` | A/B 矩阵类型 | `fp8_e4m3fn_t` / `fp8_e5m2_t` |
| `CType` | C 输出类型 | `bfloat16_t` |
| `TransA` / `TransB` | A/B 转置标记 | `true` / `false` |

> **TransA/TransB 模板参数化（通用要求）**：所有含 MatMul 的 apace 算子 Impl 模板**必须**含 TransA/TransB 参数，Layout 根据 Trans 条件选择 `NDExtLayoutPtn`（非转置）或 `DNExtLayoutPtn`（转置）。禁止固定 Layout——固定 Layout 的算子无法支持转置场景，且与官方 AllGather/AllToAll 算子的模板签名不一致。

### 4.2 组件持有关系

```
AllToAllMxQuantMatmulUdmaImpl（成员定义见 all_to_all_mx_quant_matmul_udma_impl.h `class AllToAllMxQuantMatmulUdmaImpl`）
├── quantMatmulKernelImpl_ (QuantMatmulMxKernel<..., UdmaCommWaitPolicy>)  ← AIC 侧计算（CommPolicy 注入）
├── allToAllA_        (CollectiveComm<AllToAll, PUT, AType>)      ← AIV 侧通信：data
├── allToAllScaleA_   (CollectiveComm<AllToAll, PUT, fp8_e8m0_t>) ← AIV 侧通信：scale
├── teamBarrier_      (TeamBarrier)                               ← 跨卡同步
├── syncBuffer_       (CommUbmemContext*)  ← = &hcommCtx->ubmemCtx
├── udmaCtx_          (CommUdmaContext*)   ← = &hcommCtx->udmaCtx
├── baseParams_       (BaseParams)         ← GM 地址 / rankId / rankSize / commTurn / rankDataBytes
└── tilingData_       (allToAllMatmulTilingData*)
```

验收条件：

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | 计算与通信分离 | `quantMatmulKernelImpl_` 在 AIC，`allToAllA_` / `allToAllScaleA_` 在 AIV，无交叉调用 |
| 2 | 通信组件类型 | data 与 scale 各持一个 `CollectiveComm<AllToAll, PUT, ..., TeamBarrier>`，共享同一 channel |
| 3 | 同步组件独立 | `teamBarrier_` 持有独立 UB flag，与通信 workspace 不重叠 |
| 4 | 上下文注入 | `hcommCtx` 由外部传入，Impl 不自行创建 |
| 5 | 等待策略注入 | AIC 等待经 `CommPolicy` 模板参数（`UdmaCommWaitPolicy`）注入 kernel，编译期绑定，无运行期开销 |

### 4.3 Init 步骤不变量

`Init()` 初始化顺序（`all_to_all_mx_quant_matmul_udma_impl.h` `Init`）：

| # | 不变量 | 验收条件（符号锚点） |
|:---|:---|:---|
| 1 | GM 地址已保存 | `baseParams_.aGm / scaleAGm / bGm / scaleBGm / cGm` 赋值 |
| 2 | 上下文已提取 | `syncBuffer_ = &hcommCtx->ubmemCtx`、`udmaCtx_ = &hcommCtx->udmaCtx`，`rankId / rankSize` 取自 `udmaCtx_` |
| 3 | 基础参数已推导 | `InitBaseParams()`：`headMSize = commTilingData.splitAxisTileSize`，`commTurn = splitAxisTileCnt + splitAxisTailCnt`，`rankDataBytes = axisM × axisKa × sizeof(AType)` |
| 4 | Win 基址已取 | `baseParams_.selfWinAddr = udmaCtx_->commBufferAddrs[rankId]` |
| 5 | 双 commBuf 已分配 | data 和 scale 各分配 `COMM_WORKSPACE_SIZE`（512B，`block/aiv_comm/collective_comm_context.h`），局部 `ubOffset` 累加两次 |
| 6 | barrier 已分配 | `teamBarrier_.Init(barrierBuf, syncBuffer_, rankSize, GetBlockIdx())` |
| 7 | data 通信已初始化 | `allToAllA_.Init<BARRIER_NONE>(...)`，winOffset 缺省 = 0（Win 区起始段） |
| 8 | scale 通信已初始化 | `allToAllScaleA_.Init(..., winOffset = rankSize × rankDataBytes)`（data 段之后） |

双通信对象 winOffset 契约：

| 通信对象 | winOffset | 说明 |
|:---|:---|:---|
| data (`allToAllA_`) | 0 | Win 区起始段 |
| scale (`allToAllScaleA_`) | `rankSize × rankDataBytes` | data 段之后 |

> `rankDataBytes = axisM × axisKa × sizeof(AType)`，详见 [`communication.md`](../fundamentals/communication.md) §1 winOffset 多对象复用。

> 注意：PUT impl 的 UB 分配使用**局部变量** `uint32_t ubOffset = 0`（`Init` 内），不是成员变量。

### 4.4 UB 分配

PUT impl 的 UB 分配（局部 `ubOffset` 线性累加）：

| 分配 | 大小 | 用途 |
|:---|:---|:---|
| `commBuf` | 512B（`COMM_WORKSPACE_SIZE`，`block/aiv_comm/collective_comm_context.h`） | data 通信 workspace |
| `commScaleBuf` | 512B（`COMM_WORKSPACE_SIZE`） | scale 通信 workspace |
| `barrierBuf` | 32B（`UB_SIZE` = `BARRIER_FLAG_SIZE`，`block/aiv_comm/barrier/barrier_ubmem.h`） | 跨卡同步 flag |
| **合计（PUT）** | **1056B**（512B×2 + 32B） | 双通信对象 + barrier；GET 式单通信对象为 512B + 32B = **544B** |

验收条件：

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | 线性分配 | `ubOffset` 单调递增，不可回退 |
| 2 | 双对象翻倍 | 双通信对象共 2 × `COMM_WORKSPACE_SIZE` |
| 3 | 总量约束 | UB 分配总量不超过 UB 容量 |

---

## 5. 入口函数规则

参考算子为官网现有的两个 PUT 算子：`all_to_all_quant_matmul`（4 变体入口）与 `all_gather_quant_matmul`（单入口）。

### 5.1 第一红线：禁用 schedmode

**绝对不能**使用 `__schedmode__(1)` 和 `[[bisheng::core_ratio(1,1)]]`。

`__schedmode__(1)` 强制 AIC/AIV 串行调度，导致通算流水无法重叠 → **死锁**（`aclError:507015`）。

> 注：`aclError:507015` 为通用 timeout/trap 错误码，另有 MTE 未排空根因（见 [`fusion.md`](../fundamentals/fusion.md) §5）；遇此码需按两路径鉴别。

验收条件：

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | 无 schedmode | 代码中不含 `__schedmode__` 或 `core_ratio` |
| 2 | 核配比来源 | 核配比由 `KERNEL_TYPE_MIX_AIC_1_1` 保证，不依赖 schedmode |

### 5.2 核配比 KERNEL_TYPE_MIX_AIC_1_1

核配比（AIC:AIV 比例）唯一由 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)` 保证为 1:1。

- `KERNEL_TYPE_MIX_AIC_1_1` 表示 AIC 和 AIV 1:1 配对
- 每个 AIC 核对应一个 AIV 核，同一 block 内的 AIC/AIV 共享同一 `GetBlockIdx()`
- 不需要也不应该用 `__schedmode__` 或 `core_ratio` 额外指定
- 头文件来源：`kernel_basic_intf.h`（官网 ST 以 `apace/tests/st/all_to_all_quant_matmul/src/main.cpp` include `"kernel_basic_intf.h"`；impl 侧以 `apace/kernel/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_udma_impl.h` include `"basic_api/kernel_basic_intf.h"`）

验收条件：

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | 核配比宏 | 每个入口函数含 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)` |
| 2 | 无冗余属性 | 不含 `__schedmode__` / `core_ratio` |

### 5.3 入口变体：PUT 4 入口与 AG 单入口

PUT：4 变体入口（all_to_all_quant_matmul），定义在 `apace/tests/st/all_to_all_quant_matmul/src/kernel_launcher.h`，对应 4 种 FP8 类型组合：

| 入口函数名 | AType | BType | CType |
|:---|:---|:---|:---|
| `AllToAllQuantMatmulKernelE4M3E4M3_Udma` | `fp8_e4m3fn_t` | `fp8_e4m3fn_t` | `bfloat16_t` |
| `AllToAllQuantMatmulKernelE5M2E5M2_Udma` | `fp8_e5m2_t` | `fp8_e5m2_t` | `bfloat16_t` |
| `AllToAllQuantMatmulKernelE4M3E5M2_Udma` | `fp8_e4m3fn_t` | `fp8_e5m2_t` | `bfloat16_t` |
| `AllToAllQuantMatmulKernelE5M2E4M3_Udma` | `fp8_e5m2_t` | `fp8_e4m3fn_t` | `bfloat16_t` |

统一约定：

- 输出类型恒为 `bfloat16_t`
- 4 个入口的函数体**完全相同**，仅 Impl 模板类型参数不同
- Impl 均为 `Apace::AllToAllMxQuantMatmulUdmaImpl<AType, BType, CType, false, true>`（TransA=false, TransB=true）
- ⚠️ 官方 A2A ST `main.cpp` precision/perf 分支**硬编码调用 E4M3E4M3 入口**（无运行期 dtype dispatch）——4 个 dtype 入口虽已定义齐，但 host 按 dtype 运行期分派是本 skill 的生产纪律（R3），非官方 ST 现状，勿照抄官方硬编码写法

AG：单入口（all_gather_quant_matmul），仅 1 个 `__global__` 入口 `AllGatherQuantMatmulKernel`，定义在 impl 头文件 `apace/kernel/all_gather_quant_matmul/all_gather_mx_matmul_udma_impl.h` 末尾：

- Impl 为 `AllGatherQuantMatmulImpl::AllGatherMxMatmulUdmaImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t>`（仅 3 个类型参数，无转置参数）
- 函数体调 `Init(...) + Process()`（而非 `Run()`）

扩展非 FP8 dtype（扩展方向，官网未上库）：

| 目标 dtype | 扩展方式 | 注意事项 |
|:---|:---|:---|
| INT8 量化 | 新增 INT8 入口，AType/BType 改为 `int8_t` | Scale 处理逻辑需适配（非 MX 格式） |
| 纯 BF16 | 新增 BF16 入口，去掉 Scale 参数 | DispatchPolicy 从 `MatmulWithScaleMx` 改为普通策略 |
| FP32 | 新增 FP32 入口 | 注意 L0C 容量约束（FP32 占用翻倍） |

> 扩展 dtype 时需同步修改 Blaze DispatchPolicy 和 Scale 处理逻辑，详见 [`compute.md`](../fundamentals/compute.md)。

### 5.4 模板参数：dtype 与转置

PUT Impl 模板签名为 `template<typename AType, typename BType, typename CType, bool TransA, bool TransB>`（`apace/kernel/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_udma_impl.h`），入口固定实例化 `<..., false, true>`（TransA=false, TransB=true，由 Blaze BlockMmad layout 定义承接）。

> ⚠️ 官网 PUT/AG 入口均**无 `LocalDelay` 模板参数、无 `TPipe`**；local/remote 编排由 tiling 字段 `localMatmul` 在 Impl::Run 内分支（详见 [`fusion.md`](../fundamentals/fusion.md)）。

验收条件：

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | 模板参数 | PUT Impl 含 `<AType, BType, CType, TransA, TransB>`；AG Impl 含 `<AType, BType, CType>` |
| 2 | 转置实例化 | PUT 入口实例化 `TransA=false, TransB=true` |

### 5.5 入口签名要素

标准签名（UDMA 模式，PUT）：

```cpp
// apace/tests/st/all_to_all_quant_matmul/src/kernel_launcher.h
__global__ __aicore__ void AllToAllQuantMatmulKernelE4M3E4M3_Udma(
    __gm__ CommContext *hcommCtx,        // 1. 通信上下文（GM 指针）
    GM_ADDR aGM, GM_ADDR scaleAGM,       // 2-3. A 矩阵与 Scale
    GM_ADDR bGM, GM_ADDR scaleBGM,       // 4-5. B 矩阵与 Scale
    GM_ADDR cGM,                         // 6. C 输出
    allToAllMatmulTilingData tilingData) // 7. tiling（按值）
```

AG 单入口 `AllGatherQuantMatmulKernel` 签名同构（7 参数，`AllGatherMxMatmulUdmaTilingData tilingData` 按值）。

参数顺序约定：

| 位置 | 参数 | 传递方式 | 类型 |
|:---|:---|:---|:---|
| 1 | `hcommCtx` | `__gm__` 指针 | `CommContext*` |
| 2-6 | GM_ADDR | GM 地址 | `GM_ADDR` |
| 7 | tilingData | 按值 | 算子级 tiling_data |

参数数量：官网两个 UDMA 算子入口均为 7 个参数。委托模式算子（无 `__global__` 入口）不支持直调。

### 5.6 函数体固定模式

入口函数只做"实例化 + Init + Run/Process"三件事，其余逻辑全部封装在 Impl 类中：

```cpp
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
Apace::AllToAllMxQuantMatmulUdmaImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, false, true> impl;
impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, &tilingData);  // tiling 取址传 const*
impl.Run();   // AG 入口为 impl.Process()
```

不应出现的代码：

- 通信原语调用（`Commit`/`Wait`/`Finalize`）— 在 Impl 内部
- Blaze matmul 调用（`mmadOp_`）— 在 qbmm kernel 内部
- `CrossCoreSetFlag`/`CrossCoreWaitFlag` — 在 Impl/qbmm 内部
- `if ASCEND_IS_AIV` / `if ASCEND_IS_AIC` 分支 — 在 Impl::Run/Process 内部
- `__schedmode__` / `core_ratio` — **绝对禁止**
- `TPipe` — 官网两个算子入口均不建 TPipe（UB 缓冲用 `Te::MakeMemPtr` 静态分配，见 §4.3）；**`TPipe::InitBuffer` 与 `Te::MakeMemPtr<Te::Location::UB>` 必须二选一，禁止混用**——两套机制偏移空间不共享，混用导致地址重叠 → MTE2 UB out of bounds（507015）

验收条件：

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | 固定 3 步 | 函数体仅含 `KERNEL_TASK_TYPE_DEFAULT` → `Impl` 实例化 → `Init` + `Run`（PUT）/ `Process`（AG） |
| 2 | 无业务逻辑 | 不含通信原语、matmul 调用、flag 操作、AIC/AIV 分支 |
| 3 | 无禁用属性 | 不含 `__schedmode__` / `core_ratio` |

kernel_launcher.h 模式：

| 模式 | `__global__` 入口位置 | 官网实例 |
|:---|:---|:---|
| **kernel_launcher.h 模式** | ST 的 src/kernel_launcher.h | all_to_all_quant_matmul |
| **impl 内嵌模式** | impl.h 末尾 | all_gather_quant_matmul |

### 5.7 CommContext 传递模式

UDMA 模式：`CommContext*` 为入口第一参数（`__gm__` 指针传递，Impl::Init 提取 `udmaCtx`/`ubmemCtx`），支持直调；HCCL windows 模式不传（kernel 内部 `GetHcclContext`），**无 `__global__` 入口，不支持直调**。

**验收条件**：UDMA 入口签名含 `__gm__ CommContext*`；`CommContext` 聚合体定义在**本算子 tiling_data.h** 中（PUT 在全局命名空间；AG 在 `Apace::AivComm` 命名空间）。完整模式对比与 host 侧构造序列见 [`host-and-testing.md`](host-and-testing.md) §1/§2。

### 5.8 入口函数验收条件汇总

创建或修改入口函数后，必须满足以下验收条件（对应 [`review-checklist.md`](../review-checklist.md) R1/R2/R3/R5/R6）：

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | 无 schedmode | 代码中不含 `__schedmode__` 或 `core_ratio` |
| 2 | 有核配比 | 每个入口函数都含 `KERNEL_TYPE_MIX_AIC_1_1` |
| 3 | 入口数量 | PUT 型算子 4 个 dtype 变体入口；AG 型算子单入口（`Init`+`Process`） |
| 4 | flag idx 配对 | AIV `SetFlag` idx == AIC `WaitFlag` idx（PUT）；Set/Wait flagId 配对（通用） |
| 5 | CommContext 匹配 | UDMA 模式有 `__gm__ CommContext*` 参数 |

---

## 6. AllGather 变体与 CCU 变体

### 6.1 AllGather PUT 模式

> `all_gather_quant_matmul` UDMA 实现（`kernel/all_gather_quant_matmul/all_gather_mx_matmul_udma_impl.h`）采用 AllGather PUT + 组合模式。通信在前（AIV 先 AllGather A → AIC 从 Win 读 A 计算），与 AllToAll PUT 的关键差异：

| 维度 | AllToAll PUT | AllGather PUT |
|:---|:---|:---|
| 通信原语 | AllToAll | AllGather |
| 离散内存 | 无（Win 区连续 rank-major） | **FragmentTensor（HEAD/MAIN/TAIL 虚拟重排 + dependId 预触发）** |
| 切分轴 | K | **M** |
| `__global__` 入口 | 4 个 dtype 变体 | 1 个（`AllGatherQuantMatmulKernel`） |
| 双通信对象 | data + scale 同步 | data `Init<BARRIER_NONE>` + scale `Init<BARRIER_DEVICE>` + winOffset |

> FragmentTensor 详见 [`compute.md`](../fundamentals/compute.md) §7。

### 6.2 CCU 变体

> `all_to_all_quant_matmul` 另有一个 HCCL CCU 变体（`all_to_all_mx_quant_matmul_hcomm_impl.h`），面向框架注册场景，**不支持直调**（依赖框架创建的 HCCL 上下文，直调模式下指针无效）。与 UDMA 变体共用同一个 `QuantMatmulMxKernel`，仅 `CommPolicy` 模板参数不同。详见 [`architecture.md`](../fundamentals/architecture.md) §10 ④。

#### CCU/hcomm 注册形态模板（通往生产接入的桥梁）

批量生成的算子最终要落注册形态时，按以下官方模板改造（全部锚点出自 `all_to_all_mx_quant_matmul_hcomm_impl.h`）：

```cpp
// ① tiling 聚合结构（含框架通信参数）——ccuAllToAllMatmulTilingData
//    = Mc2InitTiling + Mc2CcTiling + CommTilingData + QuantMatmulTilingData + localMatmul
//    （all_to_all_matmul_tiling_data.h:37-43；Mc2InitTiling/Mc2CcTiling 来自 highlevel_api/kernel_tiling.h）

// ② Init（纯 AIC 内）：
commState_.hccl_.InitV2(AscendC::GetHcclContext<0>(), &tilingData_->mc2InitTiling);       // :177
commState_.hccl_.SetCcTilingV2(offsetof(TilingDataType, mc2CcTiling));                    // :178（用 offsetof 传偏移）
rankId_ = commState_.hccl_.GetRankId();  rankDim_ = commState_.hccl_.GetRankDim();       // :179-180

// ③ 批量下发（<true> = 立即 commit），三 handle：scale 一份 + data head/tail 各一份
scaleHandle_   = hccl_.AlltoAll<true>(x1Scale_, dst, sendCount, FP8E8M0, strideCount, /*opCnt=*/1);        // :188-191
dataHeadHandle_ = hccl_.AlltoAll<true>(x1_, dst, headSendCount, dtype, dataStrideCount, headTileCnt);     // :193-199（opCnt=tileCnt：轮次由 HCCL 侧批量下发）
dataTailHandle_ = ...;                                                                                     // :201-209（尾块单独一份）

// ④ WaitPolicy（逐 tile 等待映射）：tileIdx==0 等 scaleHandle；< headTileCnt 等 head，否则等 tail（:68-82）
quantMatmulKernelImpl_.GetCommPolicy().state_ = &commState_;   // 绑定（:176）

// ⑤ Run：local 前置（可选）→ REMOTE 计算（WaitTile → hccl_.Wait(handle)）→ SyncAll → hccl_.Finalize()（:217-225）
```

约束与差异（相对 UDMA 直调形态）：纯 AIC（cube-only）kernel，无 AIV/CrossCoreFlag；若混启 AIV 必须给全部 HCCL 调用与 SyncAll 加 `if ASCEND_IS_AIC` 守卫，否则 Prepare/Wait 失配死锁（文件头注释明示）；mode1 经共享 kernel 开启 AtomicAdd、mode2 存在 mmad 计数错配（详见 [`fusion.md`](../fundamentals/fusion.md) §5.1）；scale 的 workspace 须 512B 对齐（`x1ScaleLen` CeilDiv(512)×512，:169-171）。

---

## 7. 计算在前算子解剖（compute-first 直调模式）

> 本节给出「3 级流水 + staging 即通信源」架构的**计算在前（compute-first）算子**的文件级实现契约，以 ReduceScatter 语义算子（M 轴输出切分 + 跨 rank 求和：每卡完整本地 mm → staging → AllToAll PUT → 增量归约）的直调生产实现为示例。架构原理见 [`fusion.md`](../fundamentals/fusion.md) §6.2。字段命名与参数个数为示例实现的具体形态，新算子按语义自定义；**角色划分（通信切分 / 计算切分 / 通信派生量 / 归约粒度）与契约关系是通用部分**。

### 7.1 文件骨架

```
kernel/{op}/
├── {op}_tiling_data.h    # tiling 结构体（CommTilingData + 单份 mm tiling + 通信派生字段）+ CommContext
├── {op}_udma_impl.h      # Impl 类（RunMatmul(AIC) / RunPutCommReduce(AIV)）
├── {op}_frag_kernel.h    # 自研 FragmentTensor mm kernel（默认，消 R 循环；命名遵循 apace 惯例：qmm_mx_kernel_{rs/ag/a2a}_frag.h，骨架见 §7.7）
└── reduce_sum_ref.h     # AIV 增量归约（手动 UB 批量形态 + guard TBuf 隔离通信区，fusion.md §6.2.6）
src/
├── kernel_launcher.h     # 4 个 dtype 变体 __global__ 入口（9 参数，见 §7.2）
└── main.cpp              # host 前置校验清单（9 项）+ T 派生 + staging 分配 + dtype dispatch + 建链
```

> mm 内核形态：默认自研 FragmentTensor kernel（命名遵循 apace 惯例 `qmm_mx_kernel_{suffix}_frag.h`，如 ReduceScatter 为 `qmm_mx_kernel_rs_frag.h`；FragmentTensor 消 R 循环，R16）；vendor 复制官方 `quant_matmul_mx_kernel.h` 为例外（须 SCALAR 论证，且其 `cGmAddr` 为 `GM_ADDR` 类型与 FragmentTensor C 输出类型不兼容，见 [`compute.md`](../fundamentals/compute.md) §7.2）。归约文件命名 `reduce_sum_ref.h`（或 `{op}_reduce_sum.h`），内部为手动 UB 批量形态——"TPipe + guard TBuf" 仅为 UB 分配的可选实现方式之一，核心契约是**手动 UB 批量 + 通信区物理隔离**。

### 7.2 入口签名（9 参数）与 dtype 变体规则

**dtype 变体入口数由算子 dtype 合同决定**：合同含 E4M3/E5M2 双组合的 FP8 量化算子需要 4 个变体入口（E4M3E4M3/E5M2E5M2/E4M3E5M2/E5M2E4M3）；其他合同按组合数覆盖。host 侧按 dtype 参数运行期分派（dispatch 宏模板见 [`development-guide.md`](development-guide.md) §3.5）。硬编码单一入口 = 异 dtype 字节流被错误模板解释 → 精度系统性错误（matched_ratio 为 0 的典型特征）。

```cpp
__global__ __aicore__ void {Op}KernelE4M3E4M3_Udma(
    __gm__ CommContext *hcommCtx,          // 第一参数（不变）
    GM_ADDR aGM, GM_ADDR scaleAGM,
    GM_ADDR bGM, GM_ADDR scaleBGM,
    GM_ADDR biasGM,                        // 可选：按算子语义省略（位置在 scaleBGM 与 cGM 之间）
    GM_ADDR cGM,
    GM_ADDR workspaceGM,                   // staging：mm 输出 = PUT 通信源（host aclrtMalloc，M×N×sizeof(CType)）
    {Op}TilingData tilingData)             // 按值传递（不变）
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    Impl<...> impl;
    impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, biasGM, cGM, workspaceGM, &tilingData);
    impl.Run();
}
```

> 其余 3 个变体（E5M2E5M2/E4M3E5M2/E5M2E4M3）仅模板参数 `AType/BType` 不同（`fp8_e5m2_t` ↔ `fp8_e4m3fn_t` 组合），函数体完全一致。SWAT tiling 引擎模板参数可统一 `DT_FLOAT8_E4M3FN`（CANN 枚举无 E5M2，两者均 1 字节 tiling 参数相同）。

### 7.3 Impl Run 编排（AIV 严格分离）

AIC/AIV 编排骨架（统一 for-t 循环、严格分离错位流水、`Wait<BARRIER_NONE>` + 手动 CrossDevice 序列）以 [`fusion.md`](../fundamentals/fusion.md) §6.2.1 为唯一事实源，本节不重复。要点索引：

- AIC `RunMatmul()`：统一 `for t { 本轮 mm 子区间（R × GetTileM(t)，地址带 tileMOffset 偏移，默认 FragmentTensor 一次调用）; SetFlag }`，T=1 自然退化
- AIV：后 R 核通信（`jobIndex = GetBlockNum()-1-GetBlockIdx()`）、前 (核数-R) 核归约，错位流水 `AllToAll(t) ∥ Reduce(t-1)`，尾部补尾轮归约
- **禁止 `Wait<BARRIER_DEVICE>`**：本场景后 R 核映射下 TeamBarrier jobIndex 错位/守卫早退致内建 CrossDevice 失效 → 跨设备同步失败（机制见 [`communication.md`](../fundamentals/communication.md) CrossDevice 机制注）

### 7.4 tiling 结构体字段（host 填充）

compute-first 算子的 tiling 结构 = `CommTilingData`（单通信对象）+ 单份全量 mm tiling（策略 B）或多套 tiling（策略 A）+ 通信派生字段（mSeg/chunkBytes/tileMaxBytes/stagingSize）+ 归约粒度字段（redUbM/redUbN）；**无 `localMatmul` 字段**（计算在前架构无 LOCAL/REMOTE 双阶段，该字段属于通信在前算子）。完整字段表与 host 填充规则见 [`scenarios/compute-first-reduce-scatter/development.md`](../scenarios/compute-first-reduce-scatter/development.md) §5.7。

### 7.5 关键实现要点（速查，展开见各事实源）

| 要点 | 结论 | 事实源 |
|:---|:---|:---|
| mm 内核 | 默认 FragmentTensor 自研消 R 循环（一次调用覆盖 R×curTileM 行，约束 R≤32 = MAX_FRAGMENT_COUNT 单次构建上限，与 T 无关）；vendor 复制官方 kernel 为例外，须 DESIGN.md 论证 SCALAR 占比 | [`fusion.md`](../fundamentals/fusion.md) §6.2.2 |
| per-tile 子区间（AIC 红线） | 每轮 problem M = `R × GetTileM(t)`，A/C/ScaleA 地址带 `tileMOffset` 偏移；禁止全量 mm×T 与归约 `(void)turn`（T>1 精度失败的实证根因） | [`development-guide.md`](development-guide.md) §3.5 契约 |
| AIV 组织 | 默认严格分离：后 R 核通信（`jobIndex = GetBlockNum()-1-GetBlockIdx()`）/ 前 (核数-R) 核归约；`Wait<BARRIER_NONE>` + 手动 `teamBarrier_.CrossDevice()` | [`fusion.md`](../fundamentals/fusion.md) §6.2.1 |
| winOffset | Win 数据区与元数据区偏移按 host 建链布局确定、三处同源（共享 Win 区布局的一种已验证实现为 96B→128）；0 偏移覆盖元数据 → "假通过" | [`fusion.md`](../fundamentals/fusion.md) §6.2.4 / [`communication.md`](../fundamentals/communication.md) 陷阱 #12 |
| staging 即 PUT 源 | mm 输出连续 [M,N]，rank 段即 chunk，零重排；self chunk 从 staging 直读（PUT self 槽闲置为已知取舍） | [`fusion.md`](../fundamentals/fusion.md) §6.2.4 |
| 单通信对象 | 通信的是输出 C（CType），无 scale 对象；`CollectiveComm<AllToAll, PUT, CType, TeamBarrier>` 一个对象即可 | §7.2 入口签名 |
| flag 编排 | flagId ∈ [0, FLAG_ID_MAX)；T=1 单次 / T>1 逐轮计数配对（Set/Wait 严格配对；计数深度按 R9 口径不设数值拒绝）；零 tile 核无条件 Set；SyncAll 在分核守卫外 | [`fusion.md`](../fundamentals/fusion.md) §6.2.3 |
| 归约 | 手动 UB 批量形态（FP32 中间累加 + src 双缓冲 + 2D DataCopyPad blockCount=多行）；禁止 TQue 逐行模型（= 性能 FAIL） | [`fusion.md`](../fundamentals/fusion.md) §6.2.6 |
| 无回压通道 | 通道仅"mm 完成"一条（AIC→AIV）；死锁论证简化为"AIC 必然完成 → AIV Wait 必然解除" | [`fusion.md`](../fundamentals/fusion.md) §6.2.1 |

### 7.6 硬限制速查表

| 限制项 | 上限 | 出处 |
|--------|------|------|
| commTurn（PUT 轮次） | 轮次索引式 flagId：轮次索引 < 16 且避开 SyncAll 保留区（`SyncAll<true>` 占 14，实际安全上界 13）；计数式固定 flagId 不受轮次数限制 | `apace/utils/constant.h` FLAG_ID_MAX；`api-crosscore-sync.md` §3 |
| waitedMask | uint32（tile 总数 ≤ 32） | `quant_matmul_mx_kernel.h` |
| rankSize | ≤ 64（COMM_MAX_RANK_NUM） | `block/aiv_comm/collective_comm_context.h` |
| AG 变体 rankSize | ≤ 8（cFragAddrs_ 固定数组，仅官网 AG kernel 实现形态） | `qmm_mx_kernel_ag_udma.h` |
| FragmentTensor 片段数 | ≤ 32（MAX_FRAGMENT_COUNT，自研 FragmentTensor kernel 路径） | `basic/fragment_tensor/fragment_tensor.h` |
| 通信参与核 | rankSize ≤ AIV BlockNum | Commit/Wait 守卫 `GetBlockIdx() < rankSize` |
| 每通信对象 UB | 512B（COMM_WORKSPACE_SIZE） | `collective_comm_context.h` |
| Win 区数据/元数据分离 | PUT/GET 数据不得覆盖 Win 区内元数据/barrier 区（官方布局 barrier 在独立 BARRIER_BUF，数据区从 0 可用；共享布局按约定偏移跳过头部，实现形态见场景 design.md §3.6） | communication.md 陷阱 #12；fusion.md §6.2.4 |
| 单轮 PUT 数据量 | ⚠️ 风险提示（非硬上限）：无官方上限（bring-up 期过大单轮曾见间歇失败，边界未定）；大单轮按 case 复核，不作 host 强制拒绝（R14 口径） | communication.md 陷阱 #13；review-checklist R14 |

> 计算在前模式的硬限制（flagId ∈ [0,15] 硬件数量 + Set/Wait 配对纪律、通信轮次尾块策略、对齐约束、Win 容量、mm 段暴露边界）见 [`fusion.md`](../fundamentals/fusion.md) §6.2.10，本节不重复。AG `rankSize ≤ 8` 是官网 AG kernel 固定数组的实现上限；自研 FragmentTensor kernel 不受此限，受 `R ≤ 32`（MAX_FRAGMENT_COUNT 单次构建上限，与 T 无关）约束。

### 7.7 自研 FragmentTensor mm kernel 与 AIV 编排形态

`{op}_frag_kernel.h` 的 Params 结构契约、核心方法职责、与 AllGather kernel 的差异核对表，以及 AIV 编排两形态（统一 for-t 循环 / 单 tile 分支）——见 [`scenarios/compute-first-reduce-scatter/development.md`](../scenarios/compute-first-reduce-scatter/development.md) §5.8/§5.9（场景级落地形态，唯一事实源）。

---

## 后续阅读

- [`fusion.md`](../fundamentals/fusion.md) — PUT 编排验收 / AIC 等待机制 / localMatmul 三种模式
- [`host-and-testing.md`](host-and-testing.md) — host 初始化序列、kernel launch 与 ST 工程
- [`communication.md`](../fundamentals/communication.md) — Commit/Wait 底层机制、CommContext 字段、winOffset 实战
- [`compute.md`](../fundamentals/compute.md) / [`communication.md`](../fundamentals/communication.md) — 接口层
