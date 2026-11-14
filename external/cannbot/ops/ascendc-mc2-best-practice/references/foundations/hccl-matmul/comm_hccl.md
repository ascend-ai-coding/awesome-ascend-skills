# 通信层：HCCL in-kernel 高阶 API（A2/910B）

本文档承载 910B（A2）MC2 skill 的"通信子能力"。涵盖：HCCL 高阶 API 目录（A2）、V2 生命周期、AllGather/AllReduce 两条蓝本、algConfig、窗口优化、禁用非高阶方式清单、host 侧配置、排错速查。

> 官方依据（CANN 官方仓库 <https://gitcode.com/cann/asc-devkit>）：`docs/api/SIMD-API/高阶API/HCCL通信类`（取 A2=910B 标记内容）。本 skill 仅覆盖 A2，不涉及 A3（910_93）/950。

## 1. HCCL 高阶 API 对象与 A2 可用接口

HCCL 是运行在 AI Core 上的**集合通信任务客户端**：不自行执行通信，而是由 `Prepare` 接口把任务信息发给**服务端**（A2 上为 AICPU），`Commit` 通知服务端执行，`Wait` 阻塞等待完成。

- **服务端类型**（`HcclServerType`）：A2 **仅支持 `HCCL_SERVER_TYPE_AICPU`**；`HCCL_SERVER_TYPE_CCU`（950-only）、`HCCL_SERVER_TYPE_END` 在 A2 不可用。
- **对象**：`Hccl<serverType, config>`，默认 `Hccl<HCCL_SERVER_TYPE_AICPU>`；第二模板参 `HcclServerConfig{CoreType type; int64_t blockId}`（`CoreType::DEFAULT|ON_AIV|ON_AIC`）可固定下发核。
- **头文件**：`adv_api/hccl/hccl.h`（或官方 `lib/hccl/hccl.h`，以工程实际为准）。
- `HcclHandle = int8_t`；Prepare 返回 handleId（≥0 成功，−1 失败）。

### 1.1 A2 可用接口目录（官方签名）

| 接口 | 签名 | 说明 |
|------|------|------|
| `InitV2`（推荐） | `void InitV2(GM_ADDR context, const void* initTiling)` | context=`GetHcclContext<HCCL_GROUP_ID_0>()`；`initTiling` 须为 `Mc2InitTiling` **栈地址**（`GET_TILING_DATA_WITH_STRUCT`，非 GM）；须配 `SetCcTilingV2` |
| `SetCcTilingV2`（推荐） | `int32_t SetCcTilingV2(uint64_t offset)` | `offset`=`offsetof(T, mc2CcTiling)`；须在 `InitV2` 后、Prepare 前调用 |
| `AllReduce` | `template<bool commit=false> HcclHandle AllReduce(GM_ADDR send, GM_ADDR recv, uint64_t count, HcclDataType, HcclReduceOp op, uint8_t repeat=1)` | A2 支持 |
| `AllGather` | `template<bool commit=false> HcclHandle AllGather(GM_ADDR send, GM_ADDR recv, uint64_t sendCount, HcclDataType, uint64_t strideCount, uint8_t repeat=1)` | A2 支持 |
| `ReduceScatter` | `template<bool commit=false> HcclHandle ReduceScatter(GM_ADDR send, GM_ADDR recv, uint64_t recvCount, HcclDataType, HcclReduceOp op, uint64_t strideCount, uint8_t repeat=1)` | A2 支持 |
| `AlltoAll` | `template<bool commit=false> HcclHandle AlltoAll(GM_ADDR send, GM_ADDR recv, uint64_t dataCount, HcclDataType, uint64_t strideCount=0, uint8_t repeat=1)` | A2 支持（**AlltoAllV 不支持 A2**） |
| `BatchWrite` | `template<bool commit=false> HcclHandle BatchWrite(GM_ADDR batchWriteInfo, uint32_t itemNum, uint16_t queueID=0)` | A2 支持；`queueID` 仅 0；跨 AI Server、`remoteRankId`≠self |
| `Commit` | `void Commit(HcclHandle handleId)` | A2 支持 |
| `Wait` | `int32_t Wait(HcclHandle handleId)` | 返回 0/−1；顺序须匹配 Prepare |
| `Finalize` | `template<bool sync=true> void Finalize()` | **A2 仅支持 `sync=true`（默认）；`Finalize<false>()` 为 A3-only** |
| `Query` | `int32_t Query(HcclHandle handleId)` | 已完成轮数（max repeat） |
| `Iterate` | `template<bool sync=true> int32_t Iterate(HcclHandle, uint16_t* seqSlices, uint16_t seqSliceLen)` | 仅 `"AlltoAll=level0:fullmesh;level1:pairwise"`，A2 实用有限 |
| `GetWindowsInAddr` | `GM_ADDR GetWindowsInAddr(uint32_t rankId)` | 通信输入窗口地址 |
| `GetWindowsOutAddr` | `GM_ADDR GetWindowsOutAddr(uint32_t rankId)` | 通信输出窗口地址 |
| `GetRankId` / `GetRankDim` | `uint32_t GetRankId()` / `uint32_t GetRankDim()` | 通信域内本 rank id / rank 数 |

**A2 不支持**（勿用）：`AlltoAllV`、`AlltoAllvWrite`、`QueueBarrier`、`GetQueueNum`、`InterHcclGroupSync`。
**已废弃**：`Init`/`SetCcTiling`（V1）、`Mc2Msg`/v1/v2 TilingData（legacy，现网 arch22 算子仍用，本 skill 推荐 V2）。

> `<true>`/`<false>` 模板参 = `commit`：`true`=Prepare 时同步通知服务端执行；`false`=Prepare 时不通知、稍后自行 `Commit`（默认 false）。

## 2. 生命周期与 AIC-only 下发（蓝本实测，V2）

蓝本（[`x_all_gather_matmul_full_mesh.h`](all_gather_matmul/x_all_gather_matmul/op_kernel/x_all_gather_matmul_full_mesh.h)，230 行）= **AIC-only**：HCCL prepare/Wait/Finalize 与 Matmul 都在 `if ASCEND_IS_AIC { … }` 正向门控内（`:75,101,220`），AIV 不执行算子主体。主流程 `Process()`（`:64-70`）= `HcclPrepare() → Nd2NzBiasCast() → InnerProcess() → HcclFinalize()`，四段中三段各自带 AIC 门控。**注意 `InitV2`/`SetCcTilingV2` 不在门控内**——它们在 `Init()`（`:52-61`）里无条件执行，AIC/AIV 都会跑；只有 Prepare/Wait/Finalize 才 AIC-only。**Finalize 前需跨核同步**：蓝本 `HcclFinalize()`（`:218` 封装）用 `CrossCoreSetFlag<0, PIPE_FIX>(EVENT_ID_6)` + `CrossCoreWaitFlag(EVENT_ID_6)`（`:224-225`，注释保证所有核计算结束再 Finalize）后再 `hccl_.Finalize()`（`:226`）——**非 `SyncAll`，但效果等同**（防某核提前 Finalize 致其他核 Wait 卡死）。类似模式见上游 `ops-transformer/mc2/matmul_reduce_scatter/.../matmul_reduce_scatter_full_mesh.h:242-252`（<https://gitcode.com/cann/ops-transformer>，未在 cannbot 挂载）。

```cpp
GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();   // x_all_gather_matmul.cpp:71（index∈{0,1}，最多 2 通信域）
Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;                     // :46 运行时 ASCEND_IS_AIC 门控
hccl_.InitV2(contextGM, tilingData);                     // V2（蓝本已用；:57）；initTiling 须栈地址
hccl_.SetCcTilingV2(offsetof(Mc2Tiling::XAllGatherMatmulTilingData, mc2CcTiling)); // V2 offset；:58
if ASCEND_IS_AIC {                                       // HcclPrepare()，AIC-only（非 g_coreType==AIV）；:75
    handleId_ = hccl_.AllGather<true>(...);              // 主块 prepare（<true>=同步通知）；:89
    if (cfg.tailCnt > 0) {                               // :91 尾块仅在有尾时下发
        tailHandleId_ = hccl_.AllGather<true>(...);      // :92
    }
}
if ASCEND_IS_AIC {                                       // InnerProcess()；:101
    // MatmulKernelLocal()（:104，本地 rank 先算）
    // 逐 tile hccl_.Wait(handleId_)（:143）→ 跳过本 rank（:149-150）→ mm.Compute(index)（:161）
}
// HcclFinalize()（:218 封装，:220 AIC 门控）：
//   CrossCoreSetFlag<0, PIPE_FIX>(EVENT_ID_6);   // :224 跨核同步（非 SyncAll，效果等同）
//   CrossCoreWaitFlag(EVENT_ID_6);               // :225
//   hccl_.Finalize();                            // :226 默认模板参 = true
```
要点：
- **AIC-only**（`ASCEND_IS_AIC` 门控），AIV 不执行算子主体；**非 `g_coreType==AIV`、非单 block 驱动**。
- **Finalize 前需跨核同步**：蓝本 `HcclFinalize()`（`:218`）用 `CrossCoreSetFlag<0,PIPE_FIX>(EVENT_ID_6)`+`CrossCoreWaitFlag(EVENT_ID_6)`（`:224-225`）后 `hccl_.Finalize()`（`:226`）——**非 `SyncAll<true>()` 但效果等同**（保证所有核计算结束再 Finalize）。HCCL使用说明的 `SyncAll<true>()` 注释（`HCCL使用说明.md:198`）针对 AIV-下发场景；AIC-only 蓝本用 CrossCore flag 达成同等同步。
- **V2（蓝本已用）**：`InitV2(context, tilingData)` + `SetCcTilingV2(offset)`（`:57-58`）；V1 `Init`/`SetCcTiling` 已废弃。
- `InitV2` 的 `initTiling` 须为栈地址（`GET_TILING_DATA_WITH_STRUCT`，非 GM）；一 context 不可初始化多个 Hccl 对象。
- 一通信域 `Prepare` 总调用 ≤ 63（A2 仅计 Prepare；A3 另含 `InterHcclGroupSync`）；`Commit`/`Wait` 次数 = `repeat`，且与 `Prepare` 同核类型。

## 3. 蓝本 A：all_gather_matmul（AllGather + Matmul，通信在前）

蓝本为已验证的 [`all_gather_matmul/`](all_gather_matmul)（上游源自 `ops-transformer/mc2/all_gather_matmul` <https://gitcode.com/cann/ops-transformer>，未在 cannbot 挂载；蓝本已剥离为自包含示例工程，AIC-only，V2）：

```cpp
// Init()（:52-61）——无 AIC 门控，AIC/AIV 都执行
hccl_.InitV2(contextGM, tilingData);                     // V2（:57）
hccl_.SetCcTilingV2(offsetof(Mc2Tiling::XAllGatherMatmulTilingData, mc2CcTiling)); // :58
rankId_ = hccl_.GetRankId(); rankDim_ = hccl_.GetRankDim();   // :59-60

// HcclPrepare()（:73）
if ASCEND_IS_AIC {                                       // :75
    handleId_ = hccl_.AllGather<true>(aGM_, gatherGM_, aTileCnt, HcclDataType(cfg.dataType),
                                      aRankCnt, cfg.tileCnt);                       // :89
    if (cfg.tailCnt > 0) {                                                          // :91
        tailHandleId_ = hccl_.AllGather<true>(aGM_+off, gatherGM_+off, aTailCnt,
                                              HcclDataType(cfg.dataType), aRankCnt, cfg.tailCnt);  // :92
    }
}
// InnerProcess()（:99）
if ASCEND_IS_AIC {                                       // :101
    this->MatmulKernelLocal();                           // :104 本地 rank 先算（与首轮通信重叠）
    MatmulKernelGather(..., handleId_, cfg.tileCnt);     // :107 内部逐 tile Wait(:143) → mm.Compute(:161)
    if (cfg.tailCnt > 0) { /* 尾块同理 */ }              // :111
}
// HcclFinalize()（:218 封装，:220 AIC 门控）：CrossCoreSetFlag<0,PIPE_FIX>(EVENT_ID_6)（:224）+ CrossCoreWaitFlag（:225）→ hccl_.Finalize()（:226，默认=true）
```
要点：主块 prepare 恒发、尾块仅 `tailCnt>0` 时发（`<true>`=同步通知服务端）；`Wait` 逐 tile 驱动计算；本地 rank 数据先算（无需 Wait，与首轮通信重叠），远端循环里用 `if (rank == this->rankId_) continue;`（`:149-150`）跳过本 rank；**Finalize 前必跨核同步**（CrossCore flag，非 SyncAll）。

## 4. 蓝本 B：matmul_all_reduce（AllReduce + Matmul，"一算一通信"）

指南仅给调度图（`算子实现.md:111-114`），无完整代码；结构源自上游 `ops-transformer/mc2/matmul_all_reduce`（<https://gitcode.com/cann/ops-transformer>，未在 cannbot 挂载），按 AIC-only + V2 改写：

```cpp
if ASCEND_IS_AIC {                                  // AIC-only 门控
    hccl_.InitV2(contextGM, tilingData);           // V2
    hccl_.SetCcTilingV2(offsetof(T, mc2CcTiling));

    // Prepare（repeat=tileCnt）
    auto handleId = hccl_.AllReduce<false>(cGM, outputGM, cOffset, dataType, HCCL_REDUCE_SUM, tileCnt);

    // 融合循环：逐 tile Matmul → Commit（下发一次 repeat）
    for (i in tileCnt) {
        matmulCompute_.Compute(...);               // Matmul（AIC 计算）
        hccl_.Commit(handleId);
    }
    hccl_.Wait(handleId);                          // Wait 全部
    CrossCoreSetFlag<0, PIPE_FIX>(EVENT_ID_6);     // 跨核同步（非 SyncAll，效果等同；参考 reduce_scatter :242-252）
    CrossCoreWaitFlag(EVENT_ID_6);
    hccl_.Finalize();                              // 默认 = Finalize<true>
}
```
要点：`AllReduce` 一次 prepare `repeat=tileCnt`；逐 tile `Matmul → Commit`；最后 `Wait` 全部；**Finalize 前必跨核同步**（CrossCore flag）。计算在通信前的算子建议本卡数据计算放最后，与末次通信互相掩盖（`算子实现.md:111`）。
> 现网 arch22 legacy 用 `notifyFlag_` 让仅 block 0 下发 HCCL——**非官方推荐、非通用**（蓝本 0 hit），本 skill 统一用 AIC-only。

## 5. algConfig（A2：预留字段，仅 FullMesh）

A2 上 `algConfig` 为**预留字段，配置后不生效**，**仅支持 FullMesh 算法**（NPU 间全连接，任意两 NPU 可直收直发）。仍需传字符串（如 `"AllReduce=level0:fullmesh"`、`"AllGather=level0:fullmesh"`、`"BatchWrite=level0:fullmesh"`），但实质等价 FullMesh。

> 蓝本实际传的就是 `"AllGather=level0:fullmesh"`（[`arch22/x_all_gather_matmul_tiling_a2a3.cpp:46`](all_gather_matmul/x_all_gather_matmul/op_host/op_tiling/arch22/x_all_gather_matmul_tiling_a2a3.cpp#L46)）。`doublering`/`pairwise`/multi-level 均为 **A3-only**，A2 不支持（本 skill 不涉及 A3）；即便误传 `doublering` 在 A2 也不生效（预留字段，等价 FullMesh）。

## 6. 窗口/buffer 优化（官方接口）

- **`SetSkipBufferWindowCopy(value)`**（host 侧 `Mc2CcTilingConfig` setter）：通信输入是否放 windows（其他卡可访问的共享缓冲区）。
  - `0`（默认）：不放入 windows；`1`：同 0；`2`：放入 windows，**仅 AllReduce/AlltoAll**。
  - 蓝本由内部 `MC2_BUFFER_TYPE` 驱动该值：`param.gatherLen == 0` 取 `MC2_BUFFER_TYPE_DEFAULT`，否则取 `MC2_BUFFER_TYPE_OUTPUT`（[`x_all_gather_matmul_tiling_base.cpp:445-448`](all_gather_matmul/x_all_gather_matmul/op_host/op_tiling/x_all_gather_matmul_tiling_base.cpp#L445)）。注意枚举成员带 `MC2_BUFFER_TYPE_` 前缀。
- **`GetWindowsInAddr(rankId)` / `GetWindowsOutAddr(rankId)`**：取卡间通信 WindowsIn/Out 起始地址，可直接作计算 I/O 减少拷贝（无效 rankId 返回 nullptr）。**蓝本未使用**（全树 0 hit），A2 可用。
- **`SetSkipLocalRankCopy(value)`**：仅 AllGather/AlltoAll；`0`=输出本 rank，`1`=跳过本 rank 拷贝。**蓝本没调用该 setter**（全树 0 hit），即沿用默认 `0`——因为 `gather_out` 需要含本卡数据。

> `MC2_BUFFER_TYPE` 是蓝本 `mc2_common/.../matmul_formulaic_tiling.h:117-126` 的**内部枚举**，用于驱动官方 `SetSkipBufferWindowCopy` setter——不是官方 HCCL 高阶 API 本身。`HCCL_BUFFSIZE` 环境变量由 `mc2_tiling_utils.cpp` 的 `GetMaxWindowSize` 读取（默认 200MB），亦非 HCCL 高阶 API。注意区分两类 `PipeBarrier`：**基础 API 核内同步 `PipeBarrier`（`basic_api/kernel_operator_block_sync_intf.h`，`AscendC::PipeBarrier<pipe>()`）A2 支持**（同流水线内同步，非 A3-only）；**HCCL `QueueBarrier`（HCCL 通信类）才为 A3-only**。蓝本 Finalize 前同步用 `CrossCoreSetFlag`/`CrossCoreWaitFlag`（跨核），非上述任一 PipeBarrier。窗口优化统一经上述官方 setter。

## 7. host 侧配置（Mc2CcTilingConfig）

官方 host 侧 builder（`HCCL-Tiling侧接口/HCCL-Tiling构造函数.md`）：
```cpp
Mc2CcTilingConfig(const std::string& groupName, uint32_t opType,
                  const std::string& algConfig, uint32_t reduceType = 0,
                  uint8_t dstDataType = 0, uint8_t srcDataType = 0, uint8_t commEngine = 0);
// A2 opType（HcclCMDType）：HCCL_CMD_ALLREDUCE / ALLGATHER / REDUCE_SCATTER / ALLTOALL / BATCH_WRITE
//   （无 ALLTOALLV——A2 不支持）
// algConfig：A2 预留不生效（FullMesh）
// reduceType：SUM/MAX/MIN（A2 AllReduce/ReduceScatter 支持 SUM/MAX/MIN）
// dstDataType/srcDataType：A2 不生效（dtype 由 Prepare 的 HcclDataType 决定）
```
- setter：A2 可用 `SetOpType/SetGroupName/SetAlgConfig/SetCommEngine/SetDebugMode/SetSkipLocalRankCopy/SetSkipBufferWindowCopy`；`SetReduceType/SetStepSize/SetCommBlockNum/SetQueueNum` 在 A2 不可用（reduceType 只能经构造参设；`SetStepSize` 为 A3-only）。
- `GetTiling(::Mc2InitTiling&)` / `GetTiling(::Mc2CcTiling&)`：填充两个不透明 tiling blob。
- `Mc2InitTiling`（≤64B，**须为 TilingData 首字段**）；`Mc2CcTiling`（≤280B，**单算子最多 8 个通信任务**）。
- **HCCL context / 通信域创建（真实路径，非虚构 host launcher）**：
  - **context 由 torch_npu 分布式创建**：`dist.new_group(backend="hccl", ...)` → `pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)` 取通信域名字符串 → 作 `group` 属性透传给 aclnn；host 须先 `this->MC2().HcclGroup("group")` 配通信域名（`算子实现.md:207-215`），kernel 侧 `GetHcclContext<HCCL_GROUP_ID_0>()` 取回（`index`∈{0,1}，最多 2 通信域）。
  - **rankSize 解析**：蓝本 `mc2_common/utils/mc2_hcom_topo_info.cpp` eager 模式 dlsym `HcomGetRankSizeEx`（从 `libhccl.so`，经 `ASCEND_HOME_PATH` 定位 `<arch>-linux/lib64`）；infershape/tiling 经 `Mc2Hcom::MC2HcomTopology::CommGetInstSizeByGroup` 调用。
  - **aclnn host 调用**（指南 `算子实现.md:742-803`；HCCL host C-API `HcclCommInitAll`/`HcclGetCommName` 见 `docs/api/SIMD-API/高阶API/HCCL通信类`，CANN 官方仓库 <https://gitcode.com/cann/asc-devkit>）：`aclInit` → `HcclCommInitAll(rankDim, devices, comms)` → 每 rank 线程 `HcclGetCommName(comm, group)` → `aclnnXxxGetWorkspaceSize(...group...)`+`aclrtMalloc`+`aclnnXxx(...)`+`aclrtSynchronizeStream`。PyTorch 绑定用 `EXEC_NPU_CMD_V1` 宏 dlsym 同符号，生成步骤走 `ops/torch-ascendc-op-extension` 路线 B，本路线不重做 PTA。

## 8. dtype（A2）

- AllReduce/ReduceScatter：FP32/FP16/INT8/INT16/INT32/BFP16；reduce op = SUM/MAX/MIN。
- AllGather/AlltoAll：支持 `HcclDataType` 全枚举。
- dtype 由 Prepare 调用的 `HcclDataType` 参数决定（host 侧 dstDataType/srcDataType 在 A2 不生效）。

## 9. 禁用非高阶方式清单（Reviewer 必查）

| 方式 | 禁用特征 | 处置 |
|------|----------|------|
| SHMEM/UDMA | `aclshmemx_*`/`aclshmem_*`、`shmem.cmake`、`third_party/shmem` | 删除，改 HCCL |
| HCOMM 点对点 | `hcomm_.Init/WriteNbi/Drain/ReadNbi/Commit`、`Hcomm<>` | 改 HCCL 集合 |
| 窗口手动 MTE | 手动 `DataCopy` 到窗口 + 标志软同步 | 用 `AllGather/AllReduce` + `Commit/Wait` + `GetWindowsInAddr` |
| RAC-server | `rac_server.h` 自研 all-reduce | 用 `AllReduce` |
| host C-API 当 kernel 通信 | `HcclAllReduce`（仅 host context 创建可用） | kernel 内禁用 |
| A2 不支持的 HCCL 接口 | `AlltoAllV`/`AlltoAllvWrite`/`QueueBarrier`/`InterHcclGroupSync`/`Finalize<false>` | 不用 |

## 10. 排错速查

| 现象 | 可能原因 | 排查方向 |
|------|----------|----------|
| `Wait` 卡死 | AIV-下发缺 `SyncAll<true>()`/0 核提前 `Finalize`；AIC-only 缺 Finalize 前跨核同步 | AIV-下发：`Finalize` 前加 `SyncAll<true>()`；AIC-only 蓝本：`Finalize` 前加 `CrossCoreSetFlag`/`CrossCoreWaitFlag`（参考 `full_mesh.h:224-225`） |
| Prepare 失败/超 63 | 单通信域 Prepare 总调用 >63 | 拆通信域或减 Prepare |
| `Wait` 顺序错 | Commit/Wait 顺序与 Prepare 不匹配 | 严格按 Prepare 顺序 Wait |
| 误用 AlltoAllV | A2 不支持 | 改用 AlltoAll（等长） |
| 精度异常 | dtype 由 host dst/src 误设（A2 不生效） | 改 Prepare 的 `HcclDataType` |
| 编译找不到 `hccl.h` | HCCL lib 路径未配置 | `ASCEND_HOME_PATH`/`ASCEND_CUSTOM_OPP_PATH` 未设；`libhccl.so` 在 `<arch>-linux/lib64` |

## 11. 后续阅读

| 想了解 | 读 |
|--------|-----|
| 算子架构与可复用框架 | [`op_architecture.md`](op_architecture.md) |
| Matmul 计算层（AIC） | [`matmul_fusion.md`](matmul_fusion.md) |
| 架构心智/AIC-only | [`mc2_architecture.md`](mc2_architecture.md) |
| PTA 接口生成 | `ops/torch-ascendc-op-extension` 路线 B（aclnn 注册） |
| 基座工程改造 | [`codebase_map.md`](codebase_map.md) |
| CANNBot 工作流 | [`workflow_integration.md`](workflow_integration.md) |
