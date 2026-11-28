# SHMEM Examples 代码模式与规范

本文从 examples 中抽取可复用的代码组织、通信骨架和规范性要求，用于开发新 SHMEM 通信算子 kernel。

## 1. `main.cpp` 固定结构

一个完整 SHMEM example 的 Host 侧通常按以下阶段组织：

1. 解析 rank 参数：包括 `n_pes`、`pe_id`、`ipport`、设备数、起始设备、数据类型、shape、重复次数等。
2. ACL 初始化：调用 `aclInit`、`aclrtSetDevice`，需要异步执行时创建 `aclrtStream`。
3. SHMEM 初始化参数构造：设置 `my_pe`、`n_pes`、`ip_port`、`local_mem_size`、`option_attr.data_op_engine_type` 和超时参数。
4. SHMEM 初始化：调用 `aclshmemx_init_attr`，根据场景选择 default、MPI 或 uniqueid bootstrap。
5. 准备 kernel 辅助参数：需要 Device barrier 的 kernel 获取 `util_get_ffts_config()`；需要 dump 时分配 dump workspace。
6. 分配输入、输出和 symmetric buffer：普通 GM 使用 `aclrtMalloc`，跨 PE 共享状态、通信 staging、signal flags 使用 `aclshmem_malloc`。
7. 输入读取：官方 demo 有时内置 `rank + 常量` 输入生成；生成算子优先读取 `gen_data.py` 产出的 input bin。
8. Kernel launch：通过模板分发或 dtype 分支选择对应 kernel wrapper，并传入 stream、FFTS、GM 地址、symmetric 地址和 shape。
9. 完成等待：使用 `aclrtSynchronizeStream`、`aclshmemx_handle_wait`、`aclshmem_barrier_all` 或 control barrier，取决于通信是在 Host、Device、RDMA 还是 SDMA 中完成。
10. 结果写出：拷回 Host 并写入按 rank 隔离的输出文件；生成算子的精度校验交给 Python checker。
11. 释放资源：释放 Host pinned memory、普通 device memory、symmetric buffer、stream；最后 `aclshmem_finalize`、`aclrtResetDevice`、`aclFinalize`。

核心原则：SHMEM 初始化早于 symmetric allocation → 所有 kernel 完成后再释放 symmetric buffer → `aclshmem_finalize` 最后。

| 约束 | 说明 |
| --- | --- |
| 单进程单 PE | `main.cpp` 一次启动只绑一个 device/`my_pe`；多 PE 由 `scripts/run.sh`/launcher 启动 |
| Host 逻辑边界 | 复杂 tiling/route/packing 拆到独立 `.cpp/.h`；`main.cpp` 只做 `BuildPlan → PrepareInputs → LaunchOp → WriteOutputs` 编排 |

## 2. Device kernel 文件拆分习惯

examples 中常见三类拆分方式：

| 类型 | 结构 | 示例 |
| --- | --- | --- |
| 通信 demo | `main.cpp` 承担 Host 逻辑，`*_kernel.cpp/.h` 承担 kernel + launch wrapper | allgather |
| 单文件 demo | kernel 模板、launch wrapper、Host 测试逻辑放在一个 `main.cpp` | SDMA demo |

推荐边界：

| 位置 | 职责 |
| --- | --- |
| `main.cpp` | 编排：初始化、数据准备、launch、同步、结果写出 |
| 独立 `.cpp/.h` | 复杂运行时计划（route/tiling/packing 等） |
| `*_kernel.cpp/.h` | dtype/shape dispatch、launch wrapper、Device kernel |
| `param.h` / `utils.h` | 公共参数、索引常量、错误检查宏 |
| Device kernel | PE 查询、地址计算、搬运、计算、kernel 内同步 |

禁止 `main.cpp` fork/spawn 多 PE 子进程。

## 3. GlobalTensor / LocalTensor / raw pointer 写法

### 3.1 raw pointer

raw pointer 写法适合直接控制地址单位、偏移和 UB 预留位置：

- GM 地址使用 `__gm__ T *`。
- UB 地址使用 `__ubuf__ T *`，常通过固定 offset 构造临时 buffer。
- byte-level engine 接口常用 `GM_ADDR` 或 `uint8_t *` 做统一地址计算。

注意统一偏移单位：指针加法通常按元素计，`uint8_t *` 或 `GM_ADDR` 加法按字节计。混用时必须显式乘以 `sizeof(T)`。

raw pointer 不能绕过 SHMEM 数据面。即使通过 `aclshmem_ptr` 或 engine-specific pointer 得到远端 symmetric 地址，跨 PE 搬运也必须调用 `aclshmem_*` typed/putmem/getmem/put_signal 或公开 `aclshmemx_*` MTE/SDMA/RDMA 接口。`DataCopy` 只用于本 PE 内本地 GM/UB 搬运，禁止直接读写远端 PE 地址。

各引擎对非对称 GM 地址的支持不同，影响 kernel 是否需要 kernel 外搬运：

| 引擎 | put_nbi 的 src 能是用户 GM | get_nbi 的 dst 能是用户 GM | 含义 |
| --- | --- | --- | --- |
| MTE | 能 | 能 | kernel 内直接用 `put_nbi(symm, input_gm, ...)`，无需 kernel 外搬运 |
| SDMA | 能 | 能 | 同上 |
| UDMA | 能 | 能 | 同上 |
| RDMA | 不能 | 不能 | 双方数据都 MUST 在对称内存上，需要 Host 侧 `aclrtMemcpy` 先搬运 |

### 3.2 `AscendC::GlobalTensor`

`GlobalTensor` 写法适合模板化接口和算子框架：

- 使用 `SetGlobalBuffer(reinterpret_cast<__gm__ T *>(addr), elem_count)` 绑定 GM。
- 与 `aclshmemx_mte_*`、`aclshmemx_sdma_*`、typed RMA 的 tensor 重载配合。


### 3.3 `AscendC::LocalTensor`

`LocalTensor` 适合表达 UB buffer：

- 通过 `TPipe/TBuf` 分配，或手动设置 `logicPos`、`bufferAddr`、`dataLen`。
- SDMA/RDMA 通常需要至少 64B 临时 UB buffer。
- ping-pong 搬运时建议显式管理两个 `LocalTensor` 或两个 UB offset，并配套 `EVENT_ID0/1`。

## 4. put + signal + wait 模式

该模式适合生产者把数据推到远端或本地 symmetric buffer，消费者按 signal 判断数据是否可读。

基本流程：

1. 生产者计算本 core 负责的数据范围。
2. 通过 `aclshmemx_mte_put_nbi`、`aclshmemx_sdma_put_nbi`、`aclshmemx_roce_put_nbi` 或 typed `put` 把数据写入目标 PE 的 symmetric buffer。
3. 生产者用 event、quiet 或 engine-specific quiet 保证该段数据写入完成到所需语义。
4. 生产者调用 `aclshmemx_signal_op` 或 `put_signal` 更新目标 PE 的 signal word。
5. 消费者调用 `aclshmem_signal_wait_until` 或 typed wait 等待 signal 到达预期值。
6. 消费者读取 symmetric buffer，进入后续搬运或计算。

signal 值建议带上 epoch/magic，避免不同轮次复用同一个 signal slot 时误读旧值。多 core 场景下每个 core 应使用独立 flag offset 或通过 lane id 做隔离。

## 5. get + local compute 模式

该模式适合 reduce、scatter、KV shuffle 等“本地掌握调度、从远端拉数据后计算”的场景。

基本流程：

1. 根据 PE、phase、chunk 或 expert id 计算远端地址。
2. 通过 `aclshmem_ptr` 或 engine-specific pointer 获取远端 symmetric 地址。
3. 调用 `get_nbi` 或对应 `aclshmemx_*_get_nbi` 把远端数据拉到本地 GM 或 UB；不得用 `DataCopy` 直接读取远端地址。
4. 使用 event/quiet 等待当前 tile 到达。
5. 在本地做 elementwise add、atomic add、cast 或数据重排。
6. 将结果写回本地 output 或本地 symmetric state。

该模式的优点是写冲突容易控制：所有累加都在本 PE 本地完成，不要求远端并发 atomic。对于 RDMA/SDMA 这类搬运本身不做 reduce 的通路，通常先 get 到 `tmp_recv`，再本地累加回 `state_symm`。

## 6. AllGather / ReduceScatter / AllReduce 通用骨架

### 6.1 AllGather

AllGather 的常见骨架：

- 每个 PE 把本地 input 写入本 PE 的 symmetric staging 区。
- 通过 quiet/barrier 或 signal 通知其他 PE。
- 每个 PE 按 rank 顺序从各远端 PE 的 staging 区读取对应分片，写入本地 output 的 `[rank * local_count, (rank + 1) * local_count)`。
- 小数据可用固定 core 数、barrier 和一次搬运完成；大数据常按 chunk 循环，用一半 core 做本地 GM 到 symmetric staging，另一半 core 轮询 signal 并从远端拉取。

收益点是读写分离：生产阶段只写本 PE staging，消费阶段按 PE 拉取，避免多个 PE 同时写同一 output 分片。

### 6.2 ReduceScatter

ReduceScatter 的通用骨架：

- 将全量输入按 PE 或 chunk 切成目标分片。
- 每个 phase 根据调度决定本 PE 需要向谁发送、从谁接收哪些 chunk。
- 远端数据先进入本地 `tmp_recv` 或 UB tile。
- 本地对目标 chunk 做累加，最终只保留属于当前 PE 的 scatter 分片。
- phase 间需要 barrier 或 schedule-defined signal，确保前一轮 partial result 已完成后再覆盖 staging buffer。

对于非 ring 或跨机非均匀调度，建议 Host 预先生成 per-phase/per-peer range list，Device 按 range list 执行，避免 kernel 内复杂解析。

### 6.3 AllReduce

AllReduce 通常由 ReduceScatter + AllGather 组成：

- ReduceScatter 阶段把每个 chunk 的归约结果收敛到负责该 chunk 的 PE。
- AllGather 阶段把各 PE 拥有的归约后 chunk 广播给所有 PE。

- 对 schedule-driven allreduce，Host 只 launch 一个 fused kernel，Device 内部执行 init、reduce-scatter、all-gather、finalize，phase 间在 Device 侧同步。

AllReduce 的关键规范是明确“谁拥有某个 chunk 的写权”。同一目标 slice 不应被多个 AIV 无序写入，除非使用受控 atomic 或先聚合再写回。

### 6.4 统一实现 vs 大小分支

- **默认 single path**：一套 kernel 逻辑覆盖 S/L 档；UB 内 `while` 分块是内部细节，不是 `small`/`large` 两条路径
- **禁止**未证明收益就维护 `*_small_data` / `*_big_data`、`*_small` / `*_large` 并行实现
- **允许**仅因 `GVA_BUFF_MAX_SIZE` / symm 容量触发的**外层 tile 循环**（内存约束，不是 perf 分支）
- size 分支须附 profiling：**≥5%** kernel_bus_bandwidth_GBps 或时延收益才保留；否则合并并删除死代码

## 7. symmetric buffer 生命周期

symmetric buffer 一般分为三类：

- 数据 staging：存放本 PE 待其他 PE 读取的数据。
- 状态/结果：保存 partial result、reduce state 或 allreduce 最终结果。
- 同步区：signal flags、barrier counters、per-core ready flags。

生命周期建议：

- 在 `aclshmemx_init_attr` 成功后分配。
- 所有 PE 使用相同分配顺序和大小，即使某 PE 当前 case 不使用某段 buffer，也应保持布局一致。
- 通过固定 layout 管理同一 buffer 内的数据区和 flag 区，例如先放 `aiv_num * flag_interval`，再放数据区。
- 每轮复用前重置 signal/state，或使用 magic/epoch 区分轮次。
- kernel 和 Host 校验全部完成后再 `aclshmem_free`。

## 8. 多 PE rank 分支写法

多 PE kernel 中常见分支：

- `my_pe = aclshmem_my_pe()`、`n_pes = aclshmem_n_pes()` 作为所有 rank 逻辑入口。
- `if (peer == my_pe) continue;` 避免对本 PE 走远端路径。
- `pair_rank`、`left/right peer`、`x = block / core_per_rank` 用于 ring、shuffle 或 allgather peer 分配。
- `core_per_rank = comm_core_num / n_pes` 将通信 core 分配给不同 peer。
- `rank_start/rank_count` 用于测试或跨机分组，只验证/处理局部连续 rank。

建议把 PE 映射和 core 映射写成显式变量，不把复杂表达式散落在搬运调用参数中。这样更容易检查越界、tail 和跨 PE 地址。

## 9. tail/chunk 循环写法

数据搬运通常按 chunk 切分：

- 用 `ceil_div(total, chunk)` 计算轮数。
- 每个 core 先算 `base_per_core = total / core_num` 和 `extra = total % core_num`，前 `extra` 个 core 多处理 1 个元素或字节。
- 最后一个 core 或最后一个 chunk 处理 remainder。
- 大块搬运用 `while (remaining >= ub_size)` 处理整块，再单独处理尾块。
- ping-pong buffer 用两个 event id 交替等待和下发，降低搬运气泡。

**MUST（multi-chunk MTE put/get）**：同一 AIV 上连续多次 `aclshmemx_mte_put_nbi` / `mte_get_nbi` 写同一目标缓冲的不同偏移时，chunk 之间 **MUST** 插入完整流水同步（对齐 `examples/allgather`）：

```text
SetFlag/WaitFlag: MTE3_S → S_MTE2 → MTE3_MTE2
```

仅依赖单次 `mte_quiet` 不够：第二块可能覆盖第一块，表现为输出整体偏移约 `ub_elems`（或 fp16 序列差 `ub_elems * step`）。

建议在变量名中体现单位：

- `*_bytes` 表示字节数。
- `*_elems` 或 `elem_count` 表示元素数。
- `offset_bytes` 和 `offset_elems` 不混用。

## 10. 代码规范性

详细规范以 [code-style.md](code-style.md) 为准，此处只列模式层面的关键要求。生成代码时，先对齐官方 CANN samples 和 SHMEM examples 的 license、utils、CMake、run script、输入/golden/output 目录布局。

| 领域 | 关键要求 | 详见 code-style.md |
| --- | --- | --- |
| Host 错误检查 | 所有 ACL/SHMEM 返回值检查；cleanup 阶段用 `cleanup_ok` 累计 | §1 |
| 资源管理 | 多资源函数统一 cleanup 或 RAII；逆序释放 | §3 |
| `main.cpp` 边界 | 单进程单 PE；复杂逻辑拆独立 `.cpp/.h`；golden/checker 放 Python | §5.2 |
| Device 传输 | 跨 PE 必须 `aclshmem_*`/`aclshmemx_*`；禁止 `DataCopy` 远端地址 | §6.3 |
| NBI 完成 | event/quiet/barrier 显式完成路径 | §6.2 |
| UB/event | 集中命名常量，ping-pong 用不同 event id | §6.4 |
| signal 隔离 | 按 rank/core/phase 隔离；magic/epoch 单调递增 | — |
| CMake/脚本 | target-scoped；`SCRIPT_DIR`/`PROJECT_ROOT` 定位；输出按 rank 隔离 | §10 |
| 可测试性 | 固定 seed；rank 特征输入；结果文件含 PE id；精度阈值与 dtype 相关 | §8 |
