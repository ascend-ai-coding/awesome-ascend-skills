# SHMEM API 功能场景梳理

> custom-ops **禁止** `aclshmemi_*`；deprecated 的 `aclshmemx_barrier_*_vec` 改用 `aclshmem_barrier_*`；MTE-only 勿误用全引擎 `aclshmem_quiet` — 见 [internal-api-boundary.md](internal-api-boundary.md)（`aclshmemx_*` 扩展接口本身 **允许**）。

> **仓内路径**：下文 `examples/`、`install/shmem/include/` 等均指 `${SHMEM_REPO}/` 下路径。Read 前先 [定位 SHMEM_REPO](../../shmem-ops-dev/references/shmem-repo-resolution.md)。

本文按使用场景梳理 SHMEM 对外和内部接口提供的能力，重点说明接口族在系统中的职责、组合方式和边界，不展开逐个 API 的参数说明。

## 1. 接口层次和基本边界

| 层次 | 标识 / 前缀 | 职责 |
| --- | --- | --- |
| Host 侧 | `ACLSHMEM_HOST_API` | bootstrap、资源初始化、team 管理、symmetric heap 分配、Host RMA/同步、日志 |
| Device 侧 | `ACLSHMEM_DEVICE` | PE 查询、RMA 搬运、signal/wait、barrier、quiet/fence、atomic |
| 标准能力 | `aclshmem_*` | 标准 SHMEM 接口 |
| 扩展能力 | `aclshmemx_*` | 指定 bootstrap、stream 版本、MTE/SDMA/RDMA engine、profiling/debug |

核心前提：各 PE 拥有一致的 symmetric memory 视图。跨 PE 访问、`aclshmem_ptr`、RMA 目标地址、signal 地址、team 同步数据都依赖该前提。

跨 PE 数据传输必须通过 SHMEM 数据面接口：标准 `aclshmem_*` typed/putmem/getmem/p/g/iput/iget/put_signal，或公开 `aclshmemx_*` MTE/SDMA/RDMA engine 接口。`DataCopy` 只允许本 PE 内 GM/UB/local buffer 搬运。

## 2. Lifecycle：初始化、退出和 PE/Team 查询

### 2.1 Host bootstrap

Host 侧生命周期从 ACL 初始化和设备绑定开始，随后构造 `aclshmemx_init_attr_t` 并调用 `aclshmemx_init_attr`。当前支持三类 bootstrap：

- `ACLSHMEMX_INIT_WITH_DEFAULT`：由 SHMEM 默认控制面完成 PE 间 bootstrap，examples 中大量使用 `test_set_attr` 构造 `my_pe`、`n_pes`、`ip_port`、`local_mem_size` 和 `option_attr`。
- `ACLSHMEMX_INIT_WITH_MPI`：借助 MPI 进程组完成 rank 信息传递。
- `ACLSHMEMX_INIT_WITH_UNIQUEID`：由 rank 0 生成 `aclshmemx_uniqueid_t`，通过外部机制广播给其他 PE，再由 `aclshmemx_set_attr_uniqueid_args` 填充初始化参数。

生命周期辅助能力包括：

- `aclshmemx_init_status` 查询初始化状态。
- `aclshmem_info_get_version`、`aclshmem_info_get_name` 查询库信息。
- `aclshmem_finalize` 释放 SHMEM 资源。
- `aclshmem_global_exit` 用于异常路径下通知全局退出。
- TLS 相关配置接口用于 control store 的安全连接场景。

### 2.2 PE 和 Team

PE 查询能力同时存在于 Host 与 Device 侧：

- `aclshmem_my_pe` 返回当前 PE。
- `aclshmem_n_pes` 返回总 PE 数。
- `aclshmem_team_my_pe`、`aclshmem_team_n_pes` 查询 team 内 rank 和规模。
- `aclshmem_team_translate_pe` 在不同 team 间转换 PE 编号。

Host 侧还提供 team 构造与销毁：

- `aclshmem_team_split_strided` 适合按步长切分 team。
- `aclshmem_team_split_2d` 适合二维拓扑切分。
- `aclshmem_team_destroy` 释放 team。
- `aclshmem_team_get_config` 获取 team 配置。

Device 侧额外有 `aclshmem_team_pe_mapping`，用于 kernel 内部按 team 映射 PE。

## 3. Memory：symmetric heap、远端地址和内存类型

### 3.1 symmetric allocation

Host 侧通过 symmetric heap 分配所有 PE 可寻址的对称内存：

- `aclshmem_malloc`、`aclshmem_calloc`、`aclshmem_align` 分配 device-side symmetric memory。
- `aclshmem_free` 释放由上述接口分配的 symmetric memory。
- `aclshmemx_malloc`、`aclshmemx_calloc`、`aclshmemx_align`、`aclshmemx_free` 可指定 `aclshmem_mem_type_t`，区分 `DEVICE_SIDE` 和 `HOST_SIDE`。

symmetric allocation 关键约束：

| 约束 | 说明 |
| --- | --- |
| 对称性 | 各 PE 必须按相同顺序、相同大小分配和释放 |
| `local_mem_size` | 所有 PE 间保持一致，满足内部对齐要求 |
| 非对称分配风险 | 不会在访问点立刻报错，但导致远端地址映射错位；debug 构建更容易定位 |

### 3.2 symmetric address 和 `aclshmem_ptr`

`aclshmem_ptr(ptr, pe)` 把本 PE 的 symmetric 地址转换为远端 PE 的对应 symmetric 地址。

- Host 侧 `aclshmem_ptr(void *ptr, int pe)` 用于 Host 发起 RMA 或构造远端地址。
- Device 侧 `aclshmem_ptr(__gm__ void *ptr, int pe)` 用于 kernel 内跨 PE 访问 symmetric GM。
- RDMA/RoCE engine 提供 `aclshmem_roce_ptr`，用于 RoCE transport 下取得远端可写地址。

使用边界：

| 约束 | 说明 |
| --- | --- |
| `ptr` 来源 | 必须来自 symmetric allocation 或 SHMEM 管理的对称区域 |
| `pe` 范围 | 必须处于当前 world/team 可达范围 |
| 远端地址用途 | RMA 目标、signal、barrier/sync 状态地址都应来自 symmetric memory |
| 不是传输原语 | 拿到远端地址后仍须用 SHMEM put/get/engine API 搬运；禁止裸 `DataCopy` |

## 4. Transport：GM2GM、UB2GM、put/get、NBI 和 Engine

### 4.1 Host RMA

Host 侧 RMA 面向 CPU 发起的数据移动：

- typed put/get：`aclshmem_<type>_put`、`aclshmem_<type>_get`。
- bit-size put/get：`aclshmem_put8/16/32/64/128`、`aclshmem_get8/16/32/64/128`。
- scalar p/g：`aclshmem_<type>_p`、`aclshmem_<type>_g`。
- strided RMA：`iput`、`iget` 系列。
- byte RMA：`aclshmem_putmem`、`aclshmem_getmem`。
- non-blocking RMA：`*_nbi` 和 `putmem/getmem_nbi`。
- stream 版本：`aclshmemx_putmem_on_stream`、`aclshmemx_getmem_on_stream`。

Host 侧还可以通过 `aclshmemx_set_mte_config`、`aclshmemx_set_sdma_config`、`aclshmemx_set_rdma_config` 设置默认 engine buffer 和 sync id。

### 4.1.1 跨 PE 搬运禁用裸 `DataCopy`

在 Device kernel 中，`DataCopy` 可以用于本 PE 内 GM 与 UB、GM 与 GM 的局部搬运，也可以用于把已经通过 SHMEM get 拉到本地的临时 buffer 继续搬运。但它不能替代 SHMEM RMA：

- 禁止 `DataCopy(remote_ptr, local_ptr, ...)` 或 `DataCopy(local_ptr, remote_ptr, ...)` 直接访问远端 PE symmetric 地址。
- 禁止把 `aclshmem_ptr` 返回值当作普通本地 GM 地址交给 `DataCopy` 实现 put/get。
- 机内 HCCS/HBM 可达也不例外；语义上仍必须使用 `aclshmem_*` 或公开 `aclshmemx_*` 数据面接口，这样 SHMEM runtime 才能管理拓扑、engine、同步和可见性。

### 4.2 Device GM2GM

Device GM2GM 是 kernel 内从 GM 到远端 GM、或从远端 GM 到本地 GM 的主要路径：

- 标准 typed `put/get`、`p/g`、`iput/iget`、bit-size、`putmem/getmem` 覆盖同步和 NBI 形式。
- NBI 接口返回后不代表数据已经完成，需要配套 engine-specific quiet（如 MTE 用 `aclshmemx_mte_quiet`）、全局 `aclshmem_quiet`（多引擎混用）、barrier、event/flag 或 Host stream 同步保证可见性。
- `non_contiguous_copy_param` 描述 repeat、length、src_ld、dst_ld，用于非连续搬运。
- `AscendC::GlobalTensor` 重载适合模板化或算子框架内写法；raw `__gm__` 指针适合低层精确控制地址和 offset。

### 4.3 Device UB2GM

UB2GM 主要服务通信算子中的数据回写场景：

- `aclshmem_<type>_get_nbi(__ubuf__ T *dst, __gm__ T *src, ...)` 从远端 symmetric GM 拉到本地 UB。
- `aclshmem_<type>_put_nbi(__gm__ T *dst, __ubuf__ T *src, ...)` 从本地 UB 推到远端 symmetric GM。
- 同样提供 `LocalTensor`/`GlobalTensor` 重载和非连续搬运参数。

典型场景是先把远端数据拉入 UB，立刻做本地 compute，再写回 GM；或者把本地计算产物保留在 UB 中直接发送到远端 symmetric buffer。

### 4.4 MTE / SDMA / RDMA engine

低层 engine 接口用于明确选择数据通路和同步资源。

- MTE：`aclshmemx_mte_put_nbi`、`aclshmemx_mte_get_nbi`。适合 GM2GM 或 UB2GM 的高带宽搬运，可传入临时 UB buffer、buffer size、元素数和 `sync_id`。
- SDMA：`aclshmemx_sdma_put_nbi`、`aclshmemx_sdma_get_nbi`。适合 SDMA 通路；需要预留至少 64B UB buffer，Device 侧完成后使用 `aclshmemx_sdma_quiet` 或 `aclshmemx_sdma_notify_record`。
- RDMA/RoCE：`aclshmemx_roce_put_nbi`、`aclshmemx_roce_get_nbi`。适合跨机或 scale-out 通路；需要 RDMA 可达配置和大于 64B 的 UB buffer。底层不支持同一 PE 目标上的 RMA/AMO 并发乱序写，应通过 `sync_id`、pipeline 或分 PE 串行化规避。

transport 初始化时可通过 `option_attr.data_op_engine_type` 选择默认数据通路，如 `ACLSHMEM_DATA_OP_MTE`、`ACLSHMEM_DATA_OP_SDMA`、`ACLSHMEM_DATA_OP_ROCE`。

## 5. Sync：signal、wait/test、barrier、quiet/fence

### 5.1 signal 与 wait/test

signal 用于点对点完成通知：

- `aclshmemx_signal_op` 在 Device 侧更新远端 `sig_addr`，支持 `ACLSHMEM_SIGNAL_SET` 和 `ACLSHMEM_SIGNAL_ADD`，但该接口自身不保证一般意义上的原子性。
- `aclshmem_put_signal` 和 `aclshmem_put_signal_nbi` 将数据搬运与 signal 更新合并，适合“数据到达后通知远端”的模式。
- `aclshmem_signal_wait_until` 按 `ACLSHMEM_CMP_EQ/NE/GT/GE/LT/LE` 等条件等待本地 signal 满足条件。
- typed wait/test 系列支持单变量、all/any/some wait set、vector compare 等复杂等待模式。
- Host 侧有 `aclshmemx_signal_op_on_stream`、`aclshmemx_signal_wait_until_on_stream`，用于把 signal 操作放入 stream 顺序中。

### 5.2 barrier 和 sync

barrier 能力同时存在于 Host 和 Device：

- `aclshmem_barrier(team)`、`aclshmem_barrier_all()` 对 team 或 world 做集合同步。
- `aclshmem_sync(team)`、`aclshmem_sync_all()` 只保证此前本地 memory stores 的完成和可见性，不保证远端 RMA 更新完成。
- Host 侧提供 stream 版本 `aclshmemx_barrier_on_stream`、`aclshmemx_barrier_all_on_stream`。
注意事项：

| 要点 | 说明 |
| --- | --- |
| Device barrier 前提 | 要求 MIX kernel；纯 AIV kernel 或缺少有效 cube/vector 指令时可能被优化成不满足要求的 kernel 形态 |
| deprecated 接口 | `aclshmemx_barrier_vec`/`aclshmemx_barrier_all_vec` 已标记 deprecated；新代码用 `aclshmem_barrier`/`aclshmem_barrier_all` |
| Host vs Device 完成域 | CPU 发起的 barrier 只完成 CPU 通信；NPU 发起的 barrier 只完成 NPU 通信。Host 等待 kernel 内通信完成应结合 `aclrtSynchronizeStream`/`aclshmemx_handle_wait`/stream 版本接口 |

### 5.3 quiet / fence

Device 侧 memory ordering 能力：

- `aclshmem_quiet` 确保调用 PE 之前发出的**全引擎** symmetric data object 操作完成（MTE + SDMA + UDMA + RDMA）。
- **单引擎场景 SHOULD 使用 engine-specific quiet**：MTE → `aclshmemx_mte_quiet`；SDMA → `aclshmemx_sdma_quiet`；UDMA → `aclshmemx_udma_quiet(pe)`；RDMA/RoCE → `aclshmemx_roce_quiet`。勿在仅需 MTE drain 时调用全局 `aclshmem_quiet`。
- `aclshmem_fence` 在当前实现中与 `aclshmem_quiet` 等价，既保证顺序也保证完成。
- HCCS-only 场景下完成后可认为全局可见；HCCS + RDMA 场景下，当前说明只保证对目标 PE 内存的可见性。
- 亦可配合 event/flag（如 `SetFlag/WaitFlag<MTE3_S>`）或 Host `aclrtSynchronizeStream` 等等价同步。

## 6. Atomic：远端累加和本地归约辅助

公开 Device atomic 能力目前集中在 atomic add：

- `aclshmem_int8_atomic_add`
- `aclshmem_int16_atomic_add`
- `aclshmem_int32_atomic_add`
- `aclshmem_half_atomic_add`
- `aclshmem_bfloat16_atomic_add`
- `aclshmem_float_atomic_add`

这些接口在 Device 侧对指定 PE 的 symmetric GM 目标执行加法更新，属于异步接口。

| 要点 | 说明 |
| --- | --- |
| 目标地址 | 必须是 symmetric memory 中对应 PE 的可访问地址 |
| 完成保证 | 读取结果前需 quiet/barrier 等同步 |
| 替代模式 | 大块 reduce 可先 get 到本地 tmp，再本地累加回 state，规避 RDMA/SDMA 无法在搬运路径中累加的问题 |

## 7. Profiling 和 Debug

### 7.1 Host 日志

日志能力主要用于 Host 侧定位：

- 环境变量 `SHMEM_LOG_LEVEL` 控制日志级别，调试时常用 DEBUG 或 INFO。
- `SHMEM_LOG_TO_STDOUT` 控制是否输出到控制台。
- `SHMEM_LOG_PATH` 控制日志落盘路径，默认在 `${HOME}/shmem/log`。
- `aclshmemx_set_log_level` 可在程序内设置日志等级。
- `aclshmemx_set_extern_logger` 可接入外部 logger。

日志通常能覆盖 bootstrap、socket/control store、初始化成功/失败、finalize 等 Host 阶段；Device kernel 内问题仍需要配合 CANN 日志、DumpTensor、printf 或 profiling。

### 7.2 Device profiling

Device 侧 profiling 以 `GetSystemCycle` 为基础：

- kernel 中包含 `utils/prof/shmemi_prof.h`。
- 在关键搬运、等待、计算片段前后插入 `SHMEMI_PROF_START(frame_id)` 和 `SHMEMI_PROF_END(frame_id)`。
- Host 侧调用 `aclshmemx_get_prof(nullptr, true)` 打印指定 PE、Block、Frame 的 cycles、count 和平均耗时。
- 环境变量 `SHMEM_CYCLE_PROF_PE` 指定采集哪个 PE。

该能力适合定位单个搬运片段、等待片段或通信流水中的长尾。

### 7.3 Ascend C 调试

examples 中使用 Ascend C 调测能力定位 Device 侧问题：

- 通过 `-enable_ascendc_dump` 编译开关打开 dump。
- kernel 入口调用 `AscendC::InitDump`。
- 使用 `AscendC::DumpTensor` 查看 GM/UB Tensor 内容。
- 使用 `AscendC::printf` 打印 rank、offset、flag、地址等状态。
- `examples/utils/debug.h` 中提供 `ALL_DUMPSIZE`、`aclCheck` 和 dump 工作区辅助定义。

## 8. 使用边界速查

| 场景 | Host 可用 | Device 可用 | symmetric allocation 要求 | 完成/同步要求 |
| --- | --- | --- | --- | --- |
| init/finalize/bootstrap | 是 | 否 | 初始化时配置 symmetric heap 大小 | `finalize` 前释放业务 symmetric buffer |
| PE/team 查询 | 是 | 是 | 不要求业务 buffer，但要求 SHMEM 已初始化 | kernel 中查询依赖 Host 初始化完成 |
| `aclshmem_malloc/free` | 是 | 否 | 所有 PE 相同顺序、相同大小 | 跨 PE 使用前应确保初始化和必要 barrier 完成 |
| `aclshmem_ptr` | 是 | 是 | 输入地址必须是 symmetric 地址 | 返回地址只表示远端映射，不代表数据已同步 |
| Host put/get | 是 | 否 | 远端目标/源通常为 symmetric 地址 | NBI/stream 版本需显式同步 |
| Device GM2GM put/get | 否 | 是 | 远端目标/源为 symmetric GM | NBI 后用 engine quiet（MTE→`aclshmemx_mte_quiet`）或全局 `aclshmem_quiet`/barrier/event |
| Device UB2GM put/get | 否 | 是 | GM 侧为 symmetric 地址 | 需管理 UB 生命周期和事件同步 |
| MTE engine | Host 可配默认参数 | 是 | 远端 GM 为 symmetric 地址 | `aclshmemx_mte_quiet` 或 `sync_id`/事件保证流水顺序 |
| SDMA engine | Host 可配默认参数 | 是 | 远端 GM 为 symmetric 地址 | `aclshmemx_sdma_quiet` 或 notify/wait |
| RDMA/RoCE engine | Host 初始化选择 RoCE | 是 | 远端 GM 为 RDMA 可达 symmetric 地址 | 避免同 PE 并发冲突，必要时 `handle_wait` |
| signal/wait/test | stream 版本可用 | 是 | signal word 应在 symmetric memory | signal 前的数据可见性需由接口语义和同步保证 |
| barrier/sync | 是 | 是 | team 内 PE 必须一致参与 | Host/Device 完成域不同，Device barrier 要求 MIX kernel |
| quiet/fence | 未暴露通用 Host API | 是 | 作用于 symmetric data object 操作 | `aclshmem_quiet` 全引擎；单引擎用 `aclshmemx_*_quiet` |
| atomic add | 否 | 是 | 目标为 symmetric GM | 读取前需要 quiet/barrier 等完成保证 |
| profiling/log/debug | 是 | Device 埋点/printf/dump | profiling buffer 由库管理，dump buffer 由用户准备 | debug 代码需通过编译开关保护 |
