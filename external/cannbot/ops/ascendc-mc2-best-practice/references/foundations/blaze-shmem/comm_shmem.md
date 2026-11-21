# 通信层（SHMEM / UDMA）参考

本文档承载 MC2 skill 的"通信子能力"。涵盖：SHMEM API 目录、host/device 侧用法、禁止 HCCL 清单的执行细则、扩展其他通信原语（AllReduce / ReduceScatter / AllGather）的思路。

> 官方文档：<https://shmem-doc.pages.dev/>。本文件不重抄官方 API；只记录参考工程用到的子集 + MC2 场景下的使用要点。
>
> **接口来源核验**（基于 `gitcode.com/cann/shmem` 仓库源码核对，仓库 version v1.5.0 系列）：
>
> | 接口 | 仓库证据（`include/` 下头文件） |
> |------|--------------------------------|
> | `aclshmemx_init_attr` | `host/init/shmem_host_init.h:66` |
> | `aclshmemx_uniqueid_t` / `aclshmemx_init_attr_t` / `ACLSHMEMX_INIT_WITH_DEFAULT` | `host/shmem_host_def.h:106-111`、`host/shmem_host_def.h:164-181` |
> | `aclshmem_align` / `aclshmem_free` | `host/mem/shmem_host_heap.h:53,62` |
> | `aclshmem_my_pe` / `aclshmem_n_pes`（device 侧） | `device/team/shmem_device_team.h:27,35` |
> | `aclshmem_ptr` | device: `device/gm2gm/engine/shmem_device_mte.h:26`；host: `host/data_plane/shmem_host_rma.h:74` |
> | `aclshmemx_udma_put_nbi` / `aclshmemx_udma_quiet` | `device/gm2gm/engine/shmem_device_udma.h:63,106`（实现 `src/device/gm2gm/engine/shmem_device_udma.hpp`） |
> | `aclshmem_barrier_all` | device: `device/gm2gm/shmem_device_cc.h:76`；host: `host/data_plane/shmem_host_cc.h:49` |
> | `ACLSHMEM_DATA_OP_UDMA` / `ACLSHMEM_DATA_OP_MTE` | `host_device/shmem_common_types.h:81-86` |
> | `L2_CACHELINE_SIZE = 512` | `host_device/shmem_common_types.h:166-168` |
>
> 平台限制：UDMA 引擎仅在 `__NPU_ARCH__ == 3510`（Ascend 950/A5）上启用（`src/device/gm2gm/engine/shmem_device_udma.hpp:21-24`），与 [`../capability-declaration.md`](../../capability-declaration.md) 中 dav-3510 × UDMA 的已验证路径一致。

---

## 1. SHMEM 初始化（host 侧）

参考工程的 host 侧初始化在 `src/all_to_all_matmul.cpp::runAllToAllMatmul`，按以下顺序：

```cpp
// 1. ACL 标准 init
ACL_CHECK(aclInit(nullptr));
ACL_CHECK(aclrtSetDevice(deviceId));
ACL_CHECK(aclrtCreateStream(&stream));

// 2. SHMEM init（关键）
aclshmemx_init_attr_t attributes;
aclshmemx_uniqueid_t defaultFlagUid;
test_set_attr(rankId, rankNum, localMemSize, ipport, defaultFlagUid, &attributes);
attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;  // 选 UDMA 引擎
ACL_CHECK(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes));

// 3. 分配 SHMEM 空间（所有 rank 对齐）
void *shmemSpace = aclshmem_align(PACKAGE_SIZE, SHMEM_SPACE_SIZE);  // 512B 对齐，1 GB
```

### 多 rank 启动

参考工程用 `fork` 启动多进程（`src/all_to_all_matmul.cpp::main`），每个进程对应一个 rank：

```cpp
for (int rankId = 0; rankId < rankNum; ++rankId) {
    pid_t pid = fork();
    if (pid == 0) {
        // 子进程：setDevice(rankId) + SHMEM init + 跑 kernel
        runAllToAllMatmul(rankNum, rankId, ...);
        exit(ret);
    }
}
```

`tcp://127.0.0.1:8998` 是 rank 间发现的默认地址，可在 `test_set_attr` 里改。

### 关键参数

| 参数 | 含义 | 参考工程取值 |
|------|------|------------|
| `SHMEM_SPACE_SIZE` | 单卡 SHMEM 空间大小 | `1024 * 1024 * 1024` (1 GB) |
| `PACKAGE_SIZE` | SHMEM 对齐粒度（`aclshmem_align` 第一参） | `512` bytes。shmem 仓库已定义 `L2_CACHELINE_SIZE = 512`（`include/host_device/shmem_common_types.h:166-168`），参考工程取该值对齐 |
| `ACLSHMEM_DATA_OP_UDMA` | 数据搬运引擎选 UDMA（不是 RDMA） | 必须选 UDMA |

### 资源释放

```cpp
aclshmem_free(shmemSpace);
ACL_CHECK(aclshmem_finalize());
ACL_CHECK(aclrtDestroyStream(stream));
ACL_CHECK(aclrtResetDevice(deviceId));
ACL_CHECK(aclFinalize());
```

顺序不能错：先 free SHMEM 空间，再 finalize SHMEM，再销毁 stream/device。

---

## 2. SHMEM 设备侧 API 目录

参考工程在 `include/kernel/all_to_all_comm_udma.h` 用到的设备侧 API：

| API | 作用 | 调用位置 |
|-----|------|---------|
| `aclshmem_n_pes()` | 返回总 rank 数 | `InitParams` |
| `aclshmem_my_pe()` | 返回当前 rank ID | `InitParams` |
| `aclshmem_ptr(shmemSpace, pe)` | 把对称地址 `shmemSpace` 翻译到 `pe` 视角的 GM 地址（传本卡 rankId 即本卡基地址） | `GetDataAddrGm` / `GetScaleAddrGm` |
| `aclshmemx_udma_put_nbi(dst, src, ubuf, size, remoteRank)` | 非阻塞 Put：本地 src → 远程 dst | `PutSegmentToRank` |
| `aclshmemx_udma_quiet(remoteRank)` | 等本次 Put 数据已到达对端 PE（completed） | `PutSegmentToRank` |
| `aclshmem_barrier_all()` | 全卡 Barrier（跨核同步点） | `BarrierAll` |

### Put 的 dst 地址语义

UDMA Put 的 `dst` 是**本 rank 视角下本 rank 的 SHMEM 对称地址**。SHMEM 采用对称分配模型：所有 rank 通过 `aclshmem_align` 分配等大空间，本 rank 通过 `aclshmem_ptr(shmemSpace, rankId_)`（注意第二个参数是**本卡 rankId**）拿到本卡的 SHMEM 基地址。UDMA 引擎根据 `remoteRank` 参数将写入数据翻译到对端 rank 的对应地址。本 rank 不需要知道对端物理地址。

```cpp
// all_to_all_comm_udma.h::GetDataAddrGm
GM_ADDR baseAddr = (GM_ADDR)(aclshmem_ptr(shmemContextGM_, rankId_));
return baseAddr + bufferId * bufferBlockSize_;
```

注意 `aclshmem_ptr` 第二个参数是 `rankId_`（**本卡自己的 rank**），不是 remoteRank。本卡视角下本卡的 SHMEM 基地址就是其他卡 Put 过来的数据落点。UDMA 引擎负责地址翻译，开发者只需对本卡对称地址写入数据并指定 remoteRank。

### ubuf 参数

`aclshmemx_udma_put_nbi` 的第三个参数 `(__ubuf__ uint8_t*)nullptr` 在参考工程中始终为 `nullptr`。该参数为兼容 API 而保留的 UB 缓冲参数，UDMA 实现中直接 `(void)buf` 忽略（`src/device/gm2gm/engine/shmem_device_udma.hpp` 中 `aclshmemx_udma_put_nbi` 实现），因此传 `nullptr` 合法；MC2 的大块 Put 直接走 GM→GM。

---

## 3. AllToAll 通信实现剖析

参考工程的 `AllToAllComm` 类（`include/kernel/all_to_all_comm_udma.h`）实现了"All rank 互发 M 段"的 AllToAll 语义。核心逻辑：

### 3.1 Put 数据

```cpp
// PutToAllRanks: 当前 rank 把自己的 M 段发给所有其他 rank
template<typename XType>
__aicore__ inline void AllToAllComm<XType>::PutToAllRanks(
    uint64_t srcMOffset, uint64_t mSize, uint32_t bufferId)
{
  // 每个 Block 负责发送到对应的 remoteRank，避免卡间竞争
  if ((AscendC::GetBlockIdx() < rankSize_) && (AscendC::GetBlockIdx() != rankId_)) {
    PutSegmentToRank(AscendC::GetBlockIdx(), srcMOffset, mSize, bufferId);
  }
  BarrierAll();  // 等所有卡完成 Put
}
```

**关键设计**：不是单个 Block 串行 Put 给所有 remoteRank，而是多 Block 并发——Block i 负责发给 remoteRank i。这样 UDMA 引擎能并行下发多条指令。

### 3.2 Put 单段

```cpp
template<typename XType>
__aicore__ inline void AllToAllComm<XType>::PutSegmentToRank(
    uint32_t remoteRank, uint64_t srcMOffset, uint64_t mSize, uint32_t bufferId)
{
  GM_ADDR remoteWinAddr = GetDataAddrGm(bufferId);  // 本 rank 视角下自己的 SHMEM
  // 注意：Put 的 dst 是"本 rank 视角下本 rank 的 SHMEM 地址"，
  //       UDMA 引擎会把这个地址映射到 remoteRank 的 SHMEM；
  //       但 SHMEM 的地址翻译是按 rank 偏移做的，所以要加上 rankId 的偏移
  uint64_t dstDataOffset = rankId_ * commMSize_ * bytesPerMRow_;  // 对端按 srcRank 区分
  uint64_t srcDataOffset = remoteRank * M_ * bytesPerMRow_ + srcMOffset * bytesPerMRow_;
  uint64_t dataSize = mSize * bytesPerMRow_;

  aclshmemx_udma_put_nbi(
    remoteWinAddr + dstDataOffset,
    inputDataAddr_ + srcDataOffset,
    (__ubuf__ uint8_t*)nullptr,
    dataSize,
    remoteRank);
  aclshmemx_udma_quiet(remoteRank);  // 确保数据已到达对端 PE
}
```

**地址语义陷阱**：
- `dstDataOffset = rankId_ * commMSize_ * bytesPerMRow_`：对端收数据时，按"我从哪个 rank 来"放到对应位置，避免多 rank 数据互相覆盖；
- `srcDataOffset = remoteRank * M_ * bytesPerMRow_ + srcMOffset * bytesPerMRow_`：本 rank 的 A 矩阵逻辑上按 `remoteRank` 分段，第 i 段是要发给 remoteRank=i 的数据。

### 3.3 Scale 一次性 Put

Scale（量化系数）不参与 M 轴流水，参考工程在 AIV 启动时一次性 Put 全部 Scale：

```cpp
// all_to_all_matmul_impl.h::AllToAllProcess 开头
allToAllComm_.PutScaleToAllRanks(0, axisM_);  // 一次 Put 全 M 行的 Scale
```

因为 Scale 数据量小（`M * CeilDiv(K, 64) * 2` bytes，相比 A 的 `M * K` 小 32 倍），不值得做流水。

---

## 4. 扩展其他通信原语

MC2 算子不限于 AllToAll。下面给出扩展思路（参考工程未直接实现，但架构允许）：

### 4.1 AllReduce

AllReduce = AllToAll + Reduce。两步：
1. AllToAll 把每卡的段发到所有其他卡（复用参考工程的 `PutToAllRanks`）；
2. 接收端做本地 reduce（AIV 用 Vector 指令做加法）。

```cpp
// 伪代码：AllReduce 通信层
template<typename XType>
__aicore__ inline void AllReduceComm::ReduceAllRanks(uint32_t bufferId) {
  // bufferId 处的数据 layout: [rankSize][reduceLen]
  // 用 AIV Vector 把 rankSize 个段加到一起
  // ...（用 AscendC::Add 做 UB 内 reduce）
}
```

### 4.2 ReduceScatter

ReduceScatter = AllReduce + 每卡只保留一段。在 AllReduce 基础上，每张卡只 Reduce 自己负责的那段，输出落 GM。

### 4.3 AllGather

AllGather = 每卡 Put 自己段给所有其他卡，但不 reduce。本质就是 AllToAll 不带 reduce 步骤——参考工程的 `PutToAllRanks` 已经覆盖。

### 4.4 自定义原语注意点

- 必须保留 `aclshmemx_udma_quiet(remoteRank)` 调用，确保数据已到达对端 PE；
- Barrier 类操作用 `aclshmem_barrier_all()`；`aclshmemx_barrier_all_vec` 已废弃（shmem 仓库 `device/gm2gm/shmem_device_cc.h:85,93` 标记 `[[deprecated]]`，提示改用 `aclshmem_barrier_all`），不能用 HCCL 的同步原语；
- 跨核同步（AIV↔AIC）必须用 `CrossCoreSetFlag` / `CrossCoreWaitFlag`，不能用 SHMEM 的 barrier 替代。

---

## 5. 禁止 HCCL API 速查

以下 API 出现在 `operators/{op}/` 任意文件即视为违反约束 1，代码审查直接 FAIL：

完整禁止清单（7 类 18 个 API）：

| 类别 | API |
|------|-----|
| 初始化/终结 | `Hccl::Init()` / `Hccl::InitV2()` / `Hccl::Finalize()` |
| 任务调度 | `Hccl::Commit()` / `Hccl::Wait()` / `Hccl::Query()` / `Hccl::Iterate()` |
| 集合通信原语 | `Hccl::AllReduce()` / `Hccl::AllGather()` / `Hccl::ReduceScatter()` / `Hccl::AlltoAll()` / `Hccl::AlltoAllV()` |
| 写操作 | `Hccl::BatchWrite()` / `Hccl::AlltoAllvWrite()` |
| Tiling | `Hccl::SetCcTiling()` / `Hccl::SetCcTilingV2()` |
| 跨组同步 | `Hccl::InterHcclGroupSync()` |
| Context | `GetHcclContext<>()` |

### 历史背景与正确理解

这些 API 是 asc-devkit 官方通算融合路径（`docs/zh/api/SIMD-API/adv_api/HCCL_communication/`）的通信入口，也是官方 AllGatherMatmul 类算子样例使用的接口。**注意**：官方文档明确 Hccl 高阶 API 本身支持在 Kernel 内编排"先通信后计算 / 先计算后通信"（见 `HCCL_usage.md`），并非只能串行。本 skill 不用它们的真正原因只有一个：**官方通算融合仅支持单算子 API 调用（aclnn），不支持 Kernel 直调**；而本 skill 的 MC2 工程是 Kernel 直调形态，拿不到框架注入的通信上下文。直调场景下通信必须由 Kernel 自建（SHMEM/UDMA），才能与计算在同一 Kernel 内深度耦合。

---

## 6. SHMEM 编译与链接

参考工程的 `cmake/shmem.cmake` 自动处理 SHMEM 的拉取和构建。关键步骤：

1. `third_party/shmem` 由 cmake 首次配置时自动 `git clone --branch v1.5.0 --depth 1`（gitcode.com/cann/shmem），无需 git submodule；
2. CMake 首次配置时触发 `add_custom_target(cann_samples_shmem_dependencies)`：
   - 若 `third_party/shmem/CMakeLists.txt` 不存在，执行 `git clone`；
   - 用 `cmake -S ... -B .../build -DSOC_TYPE=Ascend950` 构建 shmem 共享库（`libshmem.so` 等）；
   - `cmake --build ... --target install` 安装到 `third_party/shmem/install/shmem/`；
3. 生成两个 IMPORTED 共享库：
   - `cann_samples_shmem_bootstrap_uid`：基于 UID 的 bootstrap；
   - `cann_samples_shmem_bootstrap_config_store`：配置存储；
4. `cann_samples::shmem` alias 提供头文件路径 + 链接库。

### Agent 开发注意

- **不要**手动修改 `third_party/shmem/` 内部；它是 clone 来的外部依赖，改动不会被保留；
- 新算子的 CMakeLists 应直接 `include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/shmem.cmake REQUIRED)`，与参考工程一致；
- 若 shmem 构建失败，常见原因：`ASCEND_HOME_PATH` 未设、`SOC_TYPE` 不是 `Ascend950`、网络无法访问 gitcode.com。

---

## 7. 排错速查

| 现象 | 可能原因 | 排查方向 |
|------|---------|---------|
| `aclshmem_align` 返回 nullptr | SHMEM 空间已被前次分配占满 | 检查 `aclshmem_finalize` 是否被调用；减小 `SHMEM_SPACE_SIZE` |
| `aclshmemx_udma_put_nbi` 后立即读 buffer 拿不到数据 | Put 是非阻塞的，没等 quiet/barrier | 在读前调 `aclshmemx_udma_quiet` 确保数据到达对端 |
| 多卡运行 deadlock | AIV 的 `CrossCoreSetFlag` idx 与 AIC 的 `CrossCoreWaitFlag` idx 不匹配 | 检查 `mLoopIdx` 在两侧是否一致 |
| Put 数据错位 | `dstDataOffset` 算错（rankId 没乘对 commMSize） | 打印 bufferId、srcMOffset、dstDataOffset 在 host 端验证 |
| 编译报 `aclshmemx_udma_put_nbi` 未定义 | 没链接 `cann_samples::shmem` | 检查 `target_link_libraries` 是否含 `cann_samples::shmem` |
| 运行报 `aclshmemx_init_attr failed` | `tcp://` 地址被占用或 rankNum 不匹配 | 改 `ipport`，确认所有 rank 进程同时启动 |

---

## 8. 后续阅读

| 想了解 | 读 |
|--------|---|
| Blaze Matmul 接入 | `matmul_blaze.md` |
| 通算融合整体架构 | `mc2_architecture.md` |
| 性能采集（L2 flush + msprof） | `profiling_mc2.md` |
| 参考工程改造食谱 | `codebase_map.md` |
