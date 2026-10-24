# SHMEM 能力梳理

> **仓内路径**：下文 `include/`、`src/`、`examples/` 等均指 `${SHMEM_REPO}/` 下路径。Read 前先 [定位 SHMEM_REPO](../../shmem-ops-dev/references/shmem-repo-resolution.md)。

本文按功能域梳理 SHMEM 当前提供的能力。梳理依据是对外头文件、Python 绑定以及内部实现模块的接口边界，不逐个展开 API 参数说明。

## 1. 总体定位

SHMEM 是面向昇腾平台的多机多卡内存访问与通信库，核心目标是让 Host 侧完成初始化、建链、对称内存和通信域管理，让 Device 侧在算子中直接访问远端对称内存并完成同步，从而支撑基础通信、跨卡数据搬运以及通算融合类算子。

从接口层次看，SHMEM 分为四层：

| 层次 | 主要入口 | 能力定位 |
| --- | --- | --- |
| C/C++ Host 接口 | `include/host/*`、`include/shmem.h` | 初始化、bootstrap、内存分配、Team 管理、Host 驱动 RMA、同步、日志和 profiling |
| Ascend C Device 接口 | `include/device/*` | 算子内 GM/UB 访问、RMA、signal、wait/test、barrier、atomic、底层 MTE/RDMA/SDMA 直驱 |
| Host/Device 共用状态 | `include/host_device/shmem_common_types.h` | PE/Team/同步/引擎配置/堆地址等共享元数据 |
| Python 接口 | `src/python/shmem`、`src/host/python_wrapper` | UID 初始化、NPU buffer/tensor 封装、RMA、signal、Team、日志和 TLS 配置 |

## 2. 功能域总览

| 功能域 | 对外或内部接口边界 | 提供的高层能力 |
| --- | --- | --- |
| 初始化与 bootstrap | `host/init`、`src/host/init`、`src/host/bootstrap` | 建立 PE 世界、交换初始化信息、创建控制面 allgather/barrier、初始化全局状态 |
| 对称内存与地址空间 | `host/mem`、`src/host/mem`、`src/host/entity` | 预留/映射跨 PE 对称虚拟地址空间，分配 HBM/Host 对称堆，支持远端地址翻译 |
| 传输与通信引擎 | `src/host/transport`、`device/*/engine` | 封装 MTE、RDMA/RoCE、SDMA 传输路径和底层资源 |
| RMA 数据面 | `host/data_plane/shmem_host_rma.h`、`device/gm2gm`、`device/ub2gm` | Put/Get、单元素 P/G、同步/异步、步长/非连续搬运、stream 化搬运 |
| Signal 与点对点同步 | `shmem_host_so.h`、`shmem_host_p2p_sync.h`、`shmem_device_so.h`、`shmem_device_p2p_sync.h` | put-with-signal、远端 signal 更新、wait/test 条件等待、异步句柄等待 |
| 集合同步与内存保序 | `shmem_host_cc.h`、`shmem_device_cc.h`、`shmem_device_mo.h` | Team barrier/sync、Device quiet/fence、FFTS 同步基址配置 |
| Team 通信域 | `host/team`、`device/team`、`src/host/team` | World team、按步长/二维拆分子 Team、PE 映射和 Team 元数据同步到 Device |
| 原子能力 | `device/gm2gm/shmem_device_amo.h` | Device 侧远端原子加 |
| Python/PyTorch 集成 | `src/python/shmem`、`pyshmem.cpp` | torchrun/UID 初始化、Buffer/Tensor 构造、Python 侧 RMA/signal/Team 封装 |
| 安全与 DFX | `host/init`、`host/utils`、`docs/debug` | TLS 配置、外部日志回调、日志级别、cycle profiling、调测辅助 |
| 场景样例 | `examples/*` | Allgather、Allreduce、ReduceScatter、Matmul 融合、KV shuffle、RDMA/SDMA、跨机 barrier 等参考实现 |

## 3. 初始化与 Bootstrap

初始化能力以 `aclshmemx_init_attr` 为核心，由 Host 侧完成。内部流程大致是：

1. 校验 PE 编号、PE 数量、本地对称内存大小、超时、数据引擎配置。
2. 根据 bootstrap flag 加载插件并建立控制面。
3. 初始化 HYBM 和对称内存堆。
4. 初始化 signal、Team、同步资源、profiling 状态。
5. 将 Host 侧 `aclshmem_device_host_state_t` 同步到 Device 元数据区。
6. 通过控制面 barrier 确认所有 PE 初始化完成。

Bootstrap 提供三类模式：

| 模式 | 内部实现 | 能力说明 |
| --- | --- | --- |
| Default | `aclshmem_bootstrap_config_store.so` | 使用 `attr.ip_port` 指定控制面地址，基于 config store 建立 rank 组 |
| UniqueID | `aclshmemx_get_uniqueid` + config store | 由一个进程生成 UID，包含 PE 0 的地址、端口和通信域标识，外部分发后初始化 |
| MPI | `aclshmem_bootstrap_mpi.so` | 复用 MPI communicator，借助 MPI 实现控制面 barrier/allgather/alltoall |

Config store bootstrap 内部提供控制面 `allgather`、`barrier`、`global_exit`，并支持 TLS 配置。UID 模式支持通过 `SHMEM_UID_SESSION_ID` 直接指定地址，或通过 `SHMEM_UID_SOCK_IFNAME` 选择网口。

## 4. 全局状态与元数据

SHMEM 的 Host/Device 共享状态集中在 `aclshmem_device_host_state_t`。该状态不是用户直接操作的数据结构，但它决定了 Device 侧接口能否在 kernel 中直接找到远端资源。

主要元数据包括：

- 全局 PE 信息：当前 `mype`、总 `npes`。
- 对称堆信息：本 PE 的 HBM/Host heap base、heap size。
- 远端 heap base 表：按 P2P、RDMA、SDMA 区分的 Device/Host heap base 数组。
- 拓扑与传输可达性：`topo_list` 标识 MTE/RoCE/SDMA 可用路径。
- Team 信息：`team_pools`、Team 映射表。
- 同步资源：Team 级 sync pool/counter、核间 sync pool/counter。
- 引擎配置：MTE/SDMA/RDMA 的 UB 起始地址、UB 大小、sync id。
- 传输资源：RDMA QP 信息、SDMA workspace、signal 地址、profiling 数据。

Host 侧还维护 `aclshmem_host_state_t`，保存默认 stream、event、block 数和 SDMA notify 资源。

## 5. 对称内存与地址空间

SHMEM 的内存能力分两层。

第一层是内部 HYBM 能力。`src/host/entity` 和 `src/host/mem/heap` 负责在 NPU 虚拟地址空间中预留地址、分配本地物理内存、交换/导入其他 PE 的 slice/entity 信息并完成 mmap。这样每个 PE 在一个规则的虚拟地址布局中拥有自己的对称堆，Device 侧可通过偏移关系定位远端地址。

第二层是对外对称堆管理。Host 侧 `aclshmem_malloc/calloc/align/free` 和 `aclshmemx_malloc/calloc/align/free` 基于内部 `memory_manager` 做 first-fit 分配、16 字节对齐、空闲块合并和线程内加锁保护。`aclshmemx_*` 额外区分 `DEVICE_SIDE` 和 `HOST_SIDE`，Host 侧对称内存依赖较新的 CANN fabric handle 能力。

需要注意的能力边界：

- 对称分配要求各 PE 按相同顺序申请相同大小，否则后续远端地址翻译会失去对称性。
- 分配接口内部带控制面 barrier，保证各 PE 内存视图一致。
- `aclshmem_ptr` / Device `aclshmem_ptr` 将本地对称地址翻译为指定 PE 的远端可访问地址。
- 远端可访问地址只是地址映射能力，不是数据传输原语；机内或跨机的跨 PE 搬运必须走 `aclshmem_*` RMA 或公开 `aclshmemx_*` MTE/SDMA/RDMA 数据面接口，禁止用 `DataCopy` 直接读写远端 symmetric GM。

## 6. 传输与通信引擎

SHMEM 将数据传输抽象为 MTE、RDMA/RoCE、SDMA 三类路径。

| 引擎 | 主要接口边界 | 能力说明 |
| --- | --- | --- |
| MTE | `device/gm2gm/engine/shmem_device_mte.h`、`device/ub2gm/engine/shmem_device_mte.h` | 默认数据搬运路径，支持 GM2GM、UB2GM、连续/非连续搬运和 UB 临时 buffer |
| RDMA/RoCE | `device/gm2gm/engine/shmem_device_rdma.h`、`src/host/transport/device_rdma` | 跨节点 Device RDMA，内部负责打开 RDMA 设备、注册 MR、查询/解析 memory key、建立 QP |
| SDMA | `device/gm2gm/engine/shmem_device_sdma.h`、`src/host/transport/device_sdma` | 通过 STARS stream、notify、workspace 和 AICPU 查询算子支持 SDMA 搬运；最多 40 个通道 |

Host 初始化时通过 `data_op_engine_type` 决定启用哪些数据操作引擎，并在 HYBM entity 中配置可达类型。Device 侧高阶 RMA 可以使用共享状态中的拓扑与引擎配置选择底层路径；低阶 engine 接口则允许算子显式调用 MTE/RDMA/SDMA 搬运并管理 UB/sync_id。

## 7. RMA 数据面

RMA 是 SHMEM 的核心数据面能力，覆盖 Host 驱动和 Device 直驱两类场景。

Host 侧 RMA 能力包括：

- typed put/get 和 raw putmem/getmem。
- 同步和非阻塞 NBI 版本。
- strided put/get。
- 单元素低延迟 p/g。
- stream 版本 putmem/getmem，用于和 ACL stream 顺序编排。
- MTE/SDMA/RDMA 的 UB 配置设置。

Device 侧 RMA 能力包括：

- GM2GM：Device global memory 到远端 symmetric global memory 的 put/get。
- UB2GM：LocalTensor/UB 到远端 GM，或远端 GM 到本地 UB。
- typed、raw size、GlobalTensor/LocalTensor 多种调用形态。
- 连续和非连续搬运，非连续场景由 `non_contiguous_copy_param` 描述 repeat、length、src/dst leading dimension。
- 同步接口和 NBI 接口；RDMA NBI 路径对同一 PE 的并发 RMA/AMO 有限制。

从使用层面看，Host RMA 更适合运行时调度和普通数据搬运，Device RMA 更适合通算融合算子，在 kernel 内直接发起跨 PE 数据读写。

## 8. Signal 与点对点同步

Signal 能力用于把数据搬运和完成通知合并起来，或单独进行远端信号更新。

主要能力包括：

- put-with-signal：完成 put 后对远端 signal 地址执行 SET 或 ADD。
- signal op：在指定 PE 上更新 signal 变量，Host 侧支持 stream 版本，Device 侧提供 kernel 内接口。
- wait/test：支持 `EQ/NE/GT/GE/LT/LE` 比较条件。
- wait/test 集合形态：all、any、some 以及 vector 比较值。
- handle wait：等待异步 RMA 句柄对应的操作完成。

这组接口用于构建细粒度点对点同步，典型搭配是 put/put_signal 发送数据后，接收侧通过 wait/test 判断数据是否可用。

## 9. 集合同步与内存保序

集合同步分 Host 和 Device 两侧。

Host 侧提供 Team 级 `barrier/sync` 以及 stream 版本 barrier。其实现会同步指定 stream，并下发 Device barrier kernel。

Device 侧提供：

- Team barrier / world barrier。
- Team sync / world sync。
- vector-core barrier 兼容接口。
- FFTS 配置设置，用于 MIX kernel 中 barrier 和跨核同步所需的 runtime 控制地址。
- `quiet/fence` 内存保序能力：`aclshmem_quiet` 将覆盖全引擎（MTE + SDMA + UDMA + RDMA）；单引擎场景应使用 `aclshmemx_mte_quiet` 等 engine-specific quiet。当前 fence 实现等价于 quiet，既保证顺序也保证完成。

Device barrier 有明确使用边界：主要用于 MIX kernel，不能和 `SyncAll` 混用；不同网络路径下可见性语义也不同，HCCS 场景可提供更强全局可见性，RDMA 参与时主要保证对目标 PE 内存的可见性。

## 10. Team 通信域

Team 是 SHMEM 的通信域抽象。初始化后默认存在 `ACLSHMEM_TEAM_WORLD`，内部保存当前 PE 在 Team 内的 rank、Team 起点、步长、大小、全局 PE 到 Team PE 的映射。

对外能力包括：

- 查询全局 PE 和 PE 总数。
- 查询当前 PE 在某个 Team 内的编号和 Team 大小。
- 从父 Team 按起点、步长、数量切分子 Team。
- 按二维空间切分 x/y Team。
- 在不同 Team 间翻译 PE 编号。
- 销毁 Team 和获取 Team 配置。

内部实现会在 Host 侧维护 Team pool，并将 Team 元数据复制到 Device。Device 侧接口可以在 kernel 中查询 Team 信息和 PE 映射，用于只让部分 rank 参与计算或同步。

## 11. 原子操作

当前显式对外的原子能力主要在 Device 侧：对远端对称地址执行 atomic add。支持的类型包括 `int8/int16/int32/half/bfloat16/float`。

Signal 的 SET/ADD 也可用于同步变量更新，但它属于 signal 同步语义，不等价于完整的通用 AMO 能力。

## 12. Python 与 PyTorch 集成

Python 层由 `_pyshmem` pybind 模块和 `shmem.core` 封装组成，能力分两级。

底层 `_pyshmem` 暴露较多 C API 对应能力，包括初始化、内存、Team、RMA、signal、TLS、日志等。上层 `shmem.core` 提供更 Python 化的接口：

- `init_final`：UID 生成、UID 初始化、finalize、版本查询。
- `memory`：Buffer 分配、释放、远端 Buffer 获取。
- `rma`：put/get、put_signal、signal_op、signal_wait。
- `direct`：PE/Team 查询、枚举类型、初始化状态。
- `construct_tensor`：基于 SHMEM 分配地址构造 NPU Tensor。

`shmem.__init__` 还提供 `aclshmem_create_tensor` / `aclshmem_free_tensor`，方便 PyTorch 侧把对称内存作为 NPU Tensor 使用。

## 13. 安全、日志与 Profiling

安全能力主要集中在 bootstrap config store：

- 支持 TLS 配置，安全文档建议开启加密连接。
- 可通过 Host/Python 接口在初始化前设置 TLS 开关、TLS 信息、私钥和私钥密码解密回调。

DFX 能力包括：

- 日志级别设置和外部 logger 回调。
- 环境变量控制日志级别、输出位置、是否打印到 stdout。
- Device cycle profiling：通过 `SHMEMI_PROF_START/END` 打点，再由 Host 侧 `aclshmemx_get_prof` 输出统计。
- 调测文档提供 Ascend C `printf` / `DumpTensor` 等组合使用方式。
- 内部 `under_api` 层对 ACL/RT/HAL/HCCP/OPAPI 动态加载做了封装，降低不同 CANN/驱动版本下的直接依赖耦合。

## 14. 场景能力与样例覆盖

仓库中的 examples 展示了 SHMEM 能支撑的场景能力：

| 场景 | 示例目录 | 体现的能力 |
| --- | --- | --- |
| 基础初始化 | `examples/init` | Host 初始化、finalize、bootstrap |
| 基础 RMA | `examples/allgather`、`examples/rdma_demo`、`examples/sdma` | MTE/RDMA/SDMA 数据搬运 |
| Host/Device 同步 | `examples/notifywait` | signal、wait 等同步能力 |
| 通算融合 | `examples/matmul_allreduce`、`examples/matmul_reduce_scatter`、`examples/allgather_matmul*` | 在 kernel 内融合计算和通信 |
| 动态调度/tiling | `examples/dynamic_tiling` | Host 生成调度，Device 执行通信/计算计划 |
| MoE/KV 类场景 | `examples/kv_shuffle`、`examples/dispatch_gmm_combine` | 分布式数据重排、GMM 分派和合并 |
| Python 集成 | `examples/python_extension`、`src/python`、`src/host/python_wrapper` | PyTorch/NPU tensor 与 SHMEM 能力组合 |

这些样例不是 SHMEM 核心 API 的全部能力边界，但能帮助理解库当前重点服务的场景：跨卡内存访问、细粒度同步、通信算子和通算融合。

## 15. 能力边界与使用注意

- SHMEM 对称堆依赖各 PE 对内存分配顺序和大小保持一致；DEBUG 模式会检查分配大小对称性。
- Host 发起的通信操作和 Device 发起的通信操作有各自的完成语义；从 Host 等待 Device 操作时需要使用 ACL stream/device synchronize 或 stream 化接口。
- Device barrier 主要面向 MIX kernel，使用前需要正确设置 FFTS 配置，并避免和 `SyncAll` 混用。
- RDMA 作为底层传输时，同一 PE 上并发 RMA/AMO 有限制，需要通过 sync_id、quiet/fence 或上层调度规避。
- SDMA 当前最多支持 `ACLSHMEM_SDMA_MAX_CHAN` 个通道，即 40 个 AIV 相关通道。
- Python 高层 `shmem.core.init` 当前聚焦 UID 初始化；更多初始化形态可通过底层 `_pyshmem` 或 C/C++ 接口使用。
