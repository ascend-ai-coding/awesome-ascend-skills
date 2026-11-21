# 910B（A2）MC2 架构心智模型

本文档承载 910B MC2 skill 的"架构心智子能力"。面向第一次接触 910B MC2 的 Architect/Developer，建立通信方案选择、AIV/AIC 分工、HCCL AIC-only 下发、通算两层流水、M 轴切分并行的整体心智模型。读完应能回答：910B 有哪些通信方案、为何选 HCCL 高阶 API？Matmul 跑在哪个核？HCCL 该怎么下发？tileCnt 调什么？

> 官方依据（CANN 官方仓库 <https://gitcode.com/cann/asc-devkit>）：HCCL 高阶 API 见 `docs/api/SIMD-API/高阶API/HCCL通信类`；Matmul 高阶 API 见 `docs/api/SIMD-API/高阶API/矩阵计算`；通算融合指南见 `docs/guide/算子实践参考/SIMD算子实现/融合算子编程/通算融合`（均取 A2/910B 标记内容）。蓝本为已验证功能正常的 [`all_gather_matmul/`](all_gather_matmul)。

## 1. 910B（A2）MC2 是什么

910B（A2，dav-2201/arch22）上的 MC2 = 多卡间集合通信 + 单卡内 `AscendC::Matmul` 计算 + 通算两层流水掩盖通信。典型场景：
- AllGather + Matmul（多卡 Gather A 后各卡本地 Matmul，蓝本 `all_gather_matmul`）
- AllReduce + Matmul（各卡 Matmul 后对结果 AllReduce，蓝本 `matmul_all_reduce`）
- AlltoAll + Matmul（多卡数据重分布后 Matmul）

> 通算融合算子**不支持 Kernel 直调（`<<<>>>`）与入图（GE）开发，仅支持单算子 API 调用**（指南 `算子实现.md:3`）。故 910B MC2 工程是 **aclnn 单算子工程**（`build.sh`→`.run`+`libcust_opapi.so`），不是 `<<<>>>` 直调工程；工程架构详见 [`op_architecture.md`](op_architecture.md)。本期不支持 quant（int8 通信、per-token 量化等）。本 skill 仅覆盖 A2（910B），不涉及 A3（910_93）。

## 2. 910B 的通信方案与本 skill 的选择

910B 上实现多卡通信有**多种方案**，并非只有一条路径：

| 方案 | 性质 | 在 910B 的可用性 |
|------|------|------------------|
| **HCCL in-kernel 高阶 API（AICPU 引擎）** | 集合通信官方高阶 API，`prepare→(Commit)→Wait` 模型 | ✅ 官方支持（A2 服务端仅 `HCCL_SERVER_TYPE_AICPU`） |
| SHMEM / UDMA（`aclshmemx_*`） | 设备侧共享内存原语 | ❌ 910B 硬件不支持（950/DAV_3510 专属） |
| HCOMM 点对点（`hcomm_.WriteNbi` 等） | URMA/RDMA 原语，非集合 | ⚠️ 见现网部分算子使用，但非官方高阶集合 API |
| 窗口手动 MTE | 手动 `DataCopy` 到通信窗口 + 状态标志软同步 | ⚠️ 见现网部分算子使用，需自行管理同步 |
| RAC-server | 自研 all-reduce 路径 | ⚠️ legacy（如 `matmul_all_reduce` arch31） |
| host 侧 HCCL C-API | `HcclAllReduce` 等 | 仅 host 侧 context 创建，不可作 kernel 通信 |

**本 skill 选择 HCCL in-kernel 高阶 API（AICPU 引擎）**，原因：
- 是官方高阶 API，集合原语齐全（AllReduce/AllGather/ReduceScatter/AlltoAll/BatchWrite）。
- `prepare→（Commit）→Wait` 模型天然支持与 `AscendC::Matmul` 流水耦合（prepare 全部 → 逐 tile Wait 驱动计算）。
- 由 AICPU 服务端执行通信，AIC 同时做本地 Matmul，二者经"消息区"+`hccl.Wait` 流水重叠（指南 `算子实现.md:118-137`）。

```
910B（A2）：Host → HCCL(AICPU server) ┐   AscendC::Matmul(AIC 计算) ┐   通算两层流水
950       ：Host → SHMEM/UDMA(device) ┘  Blaze Cube                  ┘   通算两层流水
```

> 即：950 用 SHMEM/UDMA + Blaze；910B 本 skill 用 HCCL 高阶 API + `AscendC::Matmul`。这不是"910B 唯一可行路径"，而是本 skill 的选择。

## 3. AIV/AIC 分工（Matmul 跑在 AIC）

910B 为分离架构，一个 AI Core 由 **Cube Core（AIC）+ Vector Core（AIV）** 按 1:N 组合（910B 为 1:2）。910B 全系列子型号（910B1/B2/B3/B4/B2C）的 **AIC:AIV 配比一致**，故 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` 在各子型号通用。

**Matmul 的矩阵乘（MMAD）跑在 AIC/Cube 核**；AIV 只起"发起通知"作用并取回结果（A2 下经 AIC 内 Fixpipe 从 L0C 直出 GM，不经 UB；A2 无 L0C→UB 直连通路）。官方原文（`Matmul-Kernel侧接口/SetDim.md`）：
> "分离模式：Matmul API 都是从 AIV 侧发起的，调用 Iterate 计算时在 AIV 侧只会起到通知的作用，**通知 AIC 去做矩阵计算**，计算完成后 AIC 告知 AIV 计算完成……"

```
AIV（Vector）：发起 mm.Iterate() → 通知 AIC → ... → GetTensorC(gmC) 取 GM 结果
AIC（Cube） ：执行 MMAD（L1→L0→MMAD→L0C→GM，经 Fixpipe），不主动执行，由 AIV 触发
```

**HCCL 通信与 Matmul 都在 AIC 执行**（AIC-only，`if ASCEND_IS_AIC { … }` 正向门控，AIV 不执行算子主体/早返回），见 §4。这与通算融合指南的 `ASCEND_IS_AIV` 早返回 + `#define ASCENDC_CUBE_ONLY` 蓝本等价（`算子实现.md:451-457,549`）。

> 注：`g_coreType` 在本算子中仅用于 `mc2_nd_to_nz.h` 的 ND→NZ AIV/AIC 分工（`SET_G_CORE_TYPE_IS_AIV/AIC` 宏在 `:22-28`，判定在 `:98,479,531`），**不用于** HCCL 门控（HCCL 用 `ASCEND_IS_AIC`）。

## 4. HCCL 调用模式：AIC-only + ASCEND_IS_AIC 门控

**实际蓝本（grep 实证，[`x_all_gather_matmul_full_mesh.h`](all_gather_matmul/x_all_gather_matmul/op_kernel/x_all_gather_matmul_full_mesh.h)）= AIC-only**：HCCL prepare/Wait/Finalize 与 Matmul 都在 `if ASCEND_IS_AIC { … }` 正向门控内执行（`:75,101,220`）；AIV 不执行算子主体。**Finalize 前需跨核同步**：蓝本 `HcclFinalize()`（`:218`）用 `CrossCoreSetFlag<0,PIPE_FIX>(EVENT_ID_6)`+`CrossCoreWaitFlag(EVENT_ID_6)`（`:224-225`）后 `hccl_.Finalize()`（`:226`）——非 `SyncAll<true>()`，但效果等同（保证所有核计算结束再 Finalize）。

```cpp
Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;                  // :46（运行时 ASCEND_IS_AIC 门控）
GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();  // x_all_gather_matmul.cpp:71
// Init()（:52-61）不带门控，AIC/AIV 都执行
hccl_.InitV2(contextGM, tilingData);                  // V2（蓝本已用；:57）
hccl_.SetCcTilingV2(offsetof(Mc2Tiling::XAllGatherMatmulTilingData, mc2CcTiling)); // V2 offset；:58
if ASCEND_IS_AIC {                                    // HcclPrepare()，AIC-only 门控（非 g_coreType==AIV）；:75
    handleId_ = hccl_.AllGather<true>(...);           // :89 主块 prepare（<true>=同步通知服务端）
    if (cfg.tailCnt > 0) { tailHandleId_ = hccl_.AllGather<true>(...); }  // :91-92 尾块 prepare（有尾才发）
}
if ASCEND_IS_AIC {                                    // InnerProcess()；:101
    // MatmulKernelLocal()（:104）本地 rank 先算（与首轮通信重叠）
    // 逐 tile hccl_.Wait(:143) → 跳过本 rank(:149-150) → mm.Compute(:161)
}
// HcclFinalize()（:218 封装，:220 AIC 门控）：
//   CrossCoreSetFlag<0,PIPE_FIX>(EVENT_ID_6); // :224 跨核同步（非 SyncAll，效果等同）
//   CrossCoreWaitFlag(EVENT_ID_6);            // :225
//   hccl_.Finalize();                         // :226 默认 = Finalize<true>
```
> 引自蓝本工程实测代码 + 通算融合指南 `算子实现.md:451-457,549,487-544`。

> **关于 `g_coreType` / `SyncAll<true>()`**：HCCL 使用说明（`HCCL使用说明.md:149-157,198`）确有一段示例用 `if (g_coreType == AIV)` 门控 + `AscendC::SyncAll<true>();` 注释（"防 0 核提前 Finalize 致其他核 Wait 卡死"），那是 **AIV-下发场景** 的通用写法。MC2 通算融合蓝本走 **AIC-only**（`ASCEND_IS_AIC`）路径，不显式用 `g_coreType`/`SyncAll`，**但 Finalize 前仍需跨核同步**——蓝本用 `CrossCoreSetFlag<0,PIPE_FIX>(EVENT_ID_6)`+`CrossCoreWaitFlag(EVENT_ID_6)`（`full_mesh.h:224-225`）替代 `SyncAll`，效果等同（保证所有核计算结束再 Finalize）。本 skill 以蓝本（AIC-only）为准。

> **关于"单 block 驱动 HCCL"（`notifyFlag_`）**：现网 legacy 算子（如 `matmul_all_reduce` arch22）曾用 `notifyFlag_` 让仅 block 0 下发 HCCL——**参考蓝本 `all_gather_matmul` 全树 0 hit，不采用**；本 skill 统一用 AIC-only 全核模式。

### Commit/Wait 配对规则（官方约束，HCCL 文档实证）
- `Wait` 顺序须匹配 `Prepare` 顺序；`Commit`/`Wait` 调用次数 = Prepare 的 `repeat` 数。
- `Commit`/`Wait` 须与 `Prepare` 在**同一核类型**（AIC 或 AIV）调用。
- 一通信域内所有 `Prepare` 总调用次数 ≤ 63（A2；A3 另含 `InterHcclGroupSync`）。

## 5. 通算两层流水（prepare 全部 → 逐 tile Wait 驱动计算）

两条典型生命周期（详见 [`comm_hccl.md`](comm_hccl.md)）：

**蓝本 A（all_gather_matmul，通信在前）**——蓝本实测（`Process()` 四段：`HcclPrepare → Nd2NzBiasCast → InnerProcess → HcclFinalize`）：
```
InitV2 → SetCcTilingV2（不带门控）
[ASCEND_IS_AIC] AllGather<true>(主块) [+ AllGather<true>(尾块，仅 tailCnt>0)]  [prepare 全部]
[ASCEND_IS_AIC] 本地 rank 先算 → [逐 tile: Wait(i) → Matmul(i)，跳过本 rank]
[ASCEND_IS_AIC] CrossCoreSetFlag/WaitFlag → Finalize<true>()
```
流水：tile i+1 的 AllGather 与 tile i 的 Matmul 重叠（AI CPU 执行卡间通信，AIC 同时做本地 Matmul）。

**蓝本 B（matmul_all_reduce，计算在前，"一算一通信"）**——指南仅给调度图（`算子实现.md:111-114`）：
```
Init → SetCcTiling → AllReduce(prepare, repeat=tileCnt) → [逐 tile: Matmul(i) → Commit(i)] → Wait(全部) → Finalize<true>()
```
流水：tile i+1 的 Matmul 与 tile i 的 AllReduce 通信重叠（Commit 下发一次 repeat）。计算在通信前的算子建议本卡数据计算放最后，与末次通信互相掩盖（`算子实现.md:111`）。

```
tileCnt=1（串行基线，阶段一）:  MM ── AR ── MM ── AR ──
tileCnt=N （阶段三·3.2 扫描）  :   MM0 ─┐ MM1 ─┐ MM2 ─┐
                                  AR0 └─ AR1 └─ AR2 └─   (overlap)
```

## 6. 切分策略：M 轴通算并行

| 候选轴 | 通算并行性 | 实现难度 | 适用场景 |
|--------|-----------|----------|----------|
| M 轴 | 高（逐 tile 流水） | 中 | AllGather+MM / AllReduce+MM 主选 |
| K 轴 | 中（splitK） | 低 | K 大、M 小 |
| N 轴 | 低 | 高 | 较少单独用 |

- `tileCnt`（指南记 `tileNum`）= M 轴主块切分数；`tailNum`∈{0,1} 尾块（`算子实现.md:106,318-320`）。主块/尾块/本地 rank 各一套 `TCubeTiling`（`localTiling`/`tileTiling`/`tailTiling`）。
- L2CACHE 分裂（`enableL2Tile`）：大 N/K 时单次扫 gathered buffer。
> 详见 [`pipeline_tuning.md`](pipeline_tuning.md)。

## 7. 性能采集要点
910B 走 HCCL（AICPU），无 SHMEM B-matrix residency 问题，**无需 950 的 L2 cache flush**；但仍建议 warm-up。采集流程复用 [`../../shared/profiling_mc2.md`](../../shared/profiling_mc2.md)，跳过 `heavy_add_kernel`。
> 注：通算融合指南未述 L2 flush（指南仅覆盖通信/计算重叠机制）；"无需 flush" 为本 skill 据 HCCL/AICPU 模型的推理。

## 8. 后续阅读

| 想了解 | 读 |
|--------|-----|
| 算子架构与可复用框架 | [`op_architecture.md`](op_architecture.md) |
| HCCL API 与生命周期蓝本 | [`comm_hccl.md`](comm_hccl.md) |
| Matmul 接入 | [`matmul_fusion.md`](matmul_fusion.md) |
| PTA 接口生成 | `ops/torch-ascendc-op-extension` 路线 B（aclnn 注册） |
| CANNBot 工作流 | [`workflow_integration.md`](workflow_integration.md) |
| tileCnt 调优 | [`pipeline_tuning.md`](pipeline_tuning.md) |
| 性能采集 | [`../../shared/profiling_mc2.md`](../../shared/profiling_mc2.md)（无需 L2 flush） |
