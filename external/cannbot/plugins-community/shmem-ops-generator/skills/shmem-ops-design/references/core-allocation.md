# SHMEM 分核设计指南

> **仓内路径**：下文 `examples/` 等均指 `${SHMEM_REPO}/` 下路径。Read 前先 [定位 SHMEM_REPO](../../shmem-ops-dev/references/shmem-repo-resolution.md)。

## 1. 概述

本文档指导 skill 为 SHMEM 算子选择分核策略。核心目标是把串行搬运变并行、把全局等待变局部同步、让通信与计算重叠。

### 三层分核模型

1. **PE 级分布**：每个 PE / rank 持有一段本地数据，通过 symmetric memory、MTE、SDMA、RDMA 与其他 PE 交换
2. **核级分工**：单 PE 内按 AIC、AIV 或传输 lane 分工——AIC 负责矩阵计算，AIV 负责搬运、同步、重排
3. **块级切片**：每个核处理一段连续数据、一个 tile 或一个远端 rank 的分片；尾块通过余数分配避免长尾

### 关键变量约定

| 变量 | 含义 |
| --- | --- |
| `aiv_num` | 本 kernel 的 AIV 核数，等于 `block_dim`（纯通信）或由 subcore 机制决定（通算融合） |
| `n_pes` | 参与的 PE / rank 总数 |
| `total_elements` | 待处理的总元素数 |
| `data_bytes` | 待处理的总字节数（`total_elements * sizeof(T)`） |
| `group_size` | 分组后每组的 AIV 数 |
| `core_per_rank` | 每个 PE 分配到的 AIV 数 |

---

## 2. 纯通信算子的三种基本分核模式

纯通信算子（`core_ratio(0,1)`，仅 AIV 参与）的分核归结为三种基本模式。一个算子可以在不同阶段组合使用多种模式。

### 2.1 模式 A：全员协作（All-AIV-Cooperate）

**定义**：所有 AIV 并发处理**同一块数据**的不同分片。不区分 PE，不做分组——所有核都在搬同一块 buffer 的不同区间。

**适用场景**：
- 需要将一块连续数据（如本地 input）快速搬到另一个位置（如 GVA / output）
- 数据属于单一来源，没有按 PE 区分的必要

**AIV 分配公式**：

```
len_per_core = total_elements / aiv_num
offset       = aiv_idx * len_per_core

// 尾块：最后一个 AIV 承担余数
if aiv_idx == aiv_num - 1:
    len_per_core = total_elements - len_per_core * aiv_idx
```

**执行示意**（搬运本地 input → GVA）：

```
数据块: [========================================]
         AIV 0    AIV 1    AIV 2    ...  AIV n-1
        [=====]  [=====]  [=====]       [======]  ← 最后一个核可能多/少几个元素
```

所有 AIV 同时开始、各自搬运自己的区间、无需互相同步。结束后通过一次 barrier 保证全部完成。

**参考实现**：`examples/allgather` 小数据 Step 1（所有 AIV 并发把本地 input 写入 GVA）

---

### 2.2 模式 B：按 PE 分组（Per-PE-Group）

**定义**：将 AIV 按远端 PE 编号分组，每组 AIV 专责**一个特定 PE** 的数据。同一组内的多个 AIV 并发处理该 PE 数据的不同分片。不同组之间完全独立、并行执行。

**适用场景**：
- 需要从多个远端 PE 拉取/推送数据
- 每个 PE 的数据量足够大，值得分配多个 AIV 并发搬运

**AIV 分配公式**：

```
core_per_rank = aiv_num / n_pes                      // 每个 PE 分配多少 AIV
pe_idx        = aiv_idx / core_per_rank              // 当前 AIV 负责哪个 PE
chunk_idx     = aiv_idx % core_per_rank              // 当前 AIV 在该 PE 组内的编号
len_per_core  = elements_per_pe / core_per_rank      // 每个 AIV 负责的数据量
offset        = chunk_idx * len_per_core

// 尾块
if chunk_idx == core_per_rank - 1:
    len_per_core = elements_per_pe - len_per_core * chunk_idx
```

**约束**：`aiv_num >= n_pes` 且 `aiv_num % n_pes == 0`

**执行示意**（从 n_pes 个远端 PE 拉数据，`core_per_rank=2`）：

```
PE 0 的数据: [===========]     PE 1 的数据: [===========]
              AIV 0  AIV 1                  AIV 2  AIV 3
             [=====][=====]                [=====][=====]
                  ↑                             ↑
            组 0（并发搬 PE 0）          组 1（并发搬 PE 1）
                         ← 组间完全并行 →
```

**`core_per_rank=1` 退化情况**：每个 AIV 独占一个 PE 的全部数据，各 AIV 完全独立。

**参考实现**：`examples/allgather` 小数据 Step 2（每个 AIV 从自己负责的 PE 拉数据到 output）

---

### 2.3 模式 C：按职责分组（Role-Split）

**定义**：将 AIV 按功能角色分成互不重叠的组，不同组并行执行**不同类型的任务**。各组内部可进一步使用模式 A 或 B 分配工作。

**适用场景**：
- 数据量较大，任务可以拆分为多个独立并行的子任务
- 存在天然的职责分界（读/写、K/V、RDMA/本地）

#### 子模式 C1：生产者/消费者分组

将 AIV 对半分为 send 组和 recv 组，send 组负责 local→GVA（使用模式 A 协作搬运），recv 组负责 remote GVA→output（使用模式 B 按 PE 分组拉取），两组并发执行。

**分组公式**：

```
group_size    = aiv_num / 2
// Send 组: AIV [0, group_size)        → 模式 A 协作搬运本地数据到 GVA
// Recv 组: AIV [group_size, aiv_num)  → 模式 B 按 PE 分组从远端拉数据

// Send 组内（模式 A）
send_len_per_core = total_elements / group_size
send_offset       = aiv_idx * send_len_per_core

// Recv 组内（模式 B）
core_per_rank     = group_size / n_pes
recv_aiv_local    = aiv_idx - group_size        // 组内编号
pe_idx            = recv_aiv_local / core_per_rank
chunk_idx         = recv_aiv_local % core_per_rank
recv_len_per_core = total_elements / core_per_rank
recv_offset       = chunk_idx * recv_len_per_core
```

**约束**：`aiv_num % 2 == 0` 且 `(aiv_num / 2) >= n_pes` 且 `(aiv_num / 2) % n_pes == 0`

**执行示意**：

```
Send 组（AIV 0 ~ group_size-1）:         Recv 组（AIV group_size ~ aiv_num-1）:
  模式 A 协作搬运 input → GVA               模式 B 按 PE 从远端 GVA → output
  每写完一个 chunk 发 signal                  轮询远端 flag 确认 ready 后拉数据
                    ← 两组并行 →
```

**参考实现**：`examples/allgather` 大数据路径

#### 子模式 C2：数据维度分组

当数据有多个天然独立的维度时，按维度将 AIV 等分，每组负责一个维度。组内使用模式 A 协作搬运该维度的数据。

**分组公式**：

```
n_dims            = 2                       // 如 K/V 两路
cores_per_dim     = aiv_num / n_dims
dim_idx           = aiv_idx / cores_per_dim  // 0=K, 1=V
dim_local_idx     = aiv_idx % cores_per_dim
core_size         = block_size / cores_per_dim
dim_data_offset   = dim_local_idx * core_size
```

**执行示意**（KV shuffle）：

```
K 组（AIV 0 ~ cores_per_dim-1）:          V 组（AIV cores_per_dim ~ aiv_num-1）:
  对每个 block：                             对每个 block：
    搬运 K cache 的 dim_local_idx 段            搬运 V cache 的 dim_local_idx 段
    使用 K 的 ping-pong UB + EVENT_ID 对        使用 V 的 ping-pong UB + EVENT_ID 对
                    ← 两组完全并行 →
```

**参考实现**：`examples/kv_shuffle`

#### 子模式 C3：按传输引擎分组

跨机（RDMA）和机内（MTE/SDMA）链路的延迟和带宽差异大，将 AIV 分为 RDMA 组和 local 组分别调度。

**分组公式**：

```
rdma_cores  = 按 RDMA 字节量 / 总字节量 * aiv_num（向上取整）
local_cores = aiv_num - rdma_cores
// AIV [0, rdma_cores)  → 跨机 RDMA peer
// AIV [rdma_cores, aiv_num) → 机内 local peer
```

**参考实现**：`examples/dynamic_tiling`

---

### 2.4 模式组合与选择

一个算子的不同阶段可以使用不同模式。以 allgather 为例：

| 路径 | Step 1（local→GVA） | Step 2（remote GVA→output） |
| --- | --- | --- |
| 小数据 | 模式 A（全员协作搬运） | 模式 B（按 PE 分组拉取） |
| 大数据 | 模式 C1 Send 组 = 模式 A | 模式 C1 Recv 组 = 模式 B |

**模式选择决策树**：

```
              该阶段的任务是什么？
             /                    \
     搬运单一数据块              需要区分来源/目标
     （不涉及多 PE）             （涉及多个 PE 或多维度）
           |                     /          |          \
       模式 A                 多 PE       多维度      跨机+机内
      全员协作              数据拉取     K/V 等       混合拓扑
                               |            |              |
                           模式 B        模式 C2       模式 C3
                          按 PE 分组    维度分组       引擎分组

如果"搬运"和"拉取"可以并行 → 用模式 C1 包裹（Send=模式 A, Recv=模式 B）
```

数据量阈值参考：allgather 样例以 2MB 为界切换模式 A+B → C1。实际设计中应根据同步开销和带宽收益权衡——数据量越小，多组并发的同步开销占比越高，倾向 A+B 串行。

---

## 3. 纯通信算子分核：SDMA 通道模型

SDMA 与 MTE 的分核思路不同：每个 AIV 不是按 PE 分组，而是负责总数据的一段连续地址区间，对**所有远端 PE** 发起 put/get。

**分片公式**：

```
comm_block_dim = aiv_num * sub_block_num    // 总并发通道数
base_per_core  = data_length / comm_block_dim
extra_bytes    = data_length % comm_block_dim

// 前 extra_bytes 个核多分 1 字节
if cur_block_idx < extra_bytes:
    my_len  = base_per_core + 1
    my_offset = cur_block_idx * (base_per_core + 1)
else:
    my_len  = base_per_core
    my_offset = extra_bytes * (base_per_core + 1) + (cur_block_idx - extra_bytes) * base_per_core
```

**约束**：`aiv_num` 不应超过 SDMA 通道上限，超过后加核无收益反增调度开销。

**同步方式**：每个 AIV 完成自己区间的 SDMA 后调用 `aclshmemx_sdma_quiet()` 等待完成，或使用 `notifywait` 模式让 Host 等待 notify 事件。

**参考实现**：`examples/sdma`、`examples/notifywait`

---

## 4. 通算融合算子的 AIC/AIV 协同

通算融合算子同时使用 AIC（Cube 矩阵计算）和 AIV（通信搬运）。与纯通信不同，AIC 和 AIV 不是独立的两组核，而是**同一 block 内的主核 + 子核（subcore）关系**。

### 4.1 Subcore 机制

CoC（Communication over Computation）模式下，每个 block 包含 1 个 AIC 主核 + 1 个 AIV 子核。kernel 声明时不带 `core_ratio(0,1)`（纯通信模式），而是使用默认或特定 AIC/AIV 比例。

```
AIC 核编号：coreIdx = GetBlockIdx()，coreNum = GetBlockNum()
AIV 核编号：subcoreIdx = GetSubBlockIdx()，subcoreNum = GetSubBlockNum()
```

AIC 和 AIV 在同一 block 内通过**跨核 flag** 交替执行，不需要 Host 侧启动多个 kernel。

### 4.2 两阶段流水（2-Stage Pipeline）

通算融合的核心并发机制是**双 workspace 交替**：

```
时间线：
  Stage 0    Stage 1    Stage 0    Stage 1    ...
  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
  │AIV 通信│ │AIC 计算│ │AIV 通信│ │AIC 计算│  ← AIC 视角
  │ws[0]   │ │ws[0]   │ │ws[1]   │ │ws[1]   │
  └────────┘ └────────┘ └────────┘ └────────┘
  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
  │  idle  │ │AIV 通信│ │AIC 计算│ │AIV 通信│  ← AIV 视角
  │        │ │ws[1]   │ │ws[1]   │ │ws[0]   │
  └────────┘ └────────┘ └────────┘ └────────┘

实际重叠：
  AIV 通信 ws[N]  与  AIC 计算 ws[N-1]  同时进行
```

- `WORKSPACE_STAGES = 2`：两块 symmetric workspace 交替使用
- 每轮通信填充 ws[stageId]，AIC 从 ws[1-stageId] 读取上一轮数据计算
- `commInterval` 控制每 N 个 GEMM block 触发一轮通信，直接影响流水粒度

### 4.3 跨核 Flag 同步协议

AIC 和 AIV 通过两组 flag 数组交替通知对方：

```
flagAicFinishStore[stageId]    // AIC → AIV：AIC 完成当前 stage 的计算/存储
flagAivFinishCompute[stageId]  // AIV → AIC：AIV 完成当前 stage 的通信

// AIC 侧循环
for each commIdx:
    stageId = commIdx % WORKSPACE_STAGES
    CrossCoreWaitFlag(flagAivFinishCompute[stageId])  // 等 AIV 通信完成
    ... 执行 GEMM ...
    CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageId])  // 通知 AIV

// AIV 侧循环
for each commIdx:
    stageId = commIdx % WORKSPACE_STAGES
    CrossCoreWaitFlag(flagAicFinishStore[stageId])  // 等 AIC 计算完成
    ... 执行 AllGather / ReduceScatter ...
    aclshmem_barrier_all()                       // 跨 PE 同步
    CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageId])  // 通知 AIC
```

**关键约束**：
- flag 数组长度 = `WORKSPACE_STAGES`，每个 stage 独立 flag
- 双方必须在每个 stage 各 set 一次 flag，否则死锁
- AIV 侧通信后需先做跨 PE barrier，再 set flag 给 AIC

### 4.4 典型融合模式

| 融合模式 | AIC 职责 | AIV 职责 | 流水结构 | 参考样例 |
| --- | --- | --- | --- | --- |
| AllGather + MatMul | GEMM（消费 allgathered A） | AllGather 远端 A 到 ws | 2-stage | `allgather_matmul` |
| MatMul + ReduceScatter | GEMM（输出到 ws 或本地 D） | ReduceScatter（SetAtomicAdd 归约到 D） | 2-stage | `matmul_reduce_scatter` |
| MatMul + AllReduce | GEMM | ReduceScatter + AllGather 串在一起 | 2-stage | `matmul_allreduce` |
| Dispatch-GMM-Combine | GMM1 + GMM2 | dispatch（MOE 路由、token 搬运）+ combine（unpermute） | 多阶段状态机 | `dispatch_gmm_combine` |

### 4.5 通算融合参数配置

通算融合的核心可调参数：

| 参数 | 含义 | 调优方向 |
| --- | --- | --- |
| `commInterval` | 每 N 个 GEMM block 触发一轮通信 | 小值→通信频繁、流水深；大值→计算批量大、通信少 |
| `commTileM` | 每轮通信的 M 维度行数 | 与 GEMM tile M0 匹配 |
| `commBlockM` | 通信块 M 维度 | 影响 ws 容量需求 |
| `commNpuSplit` | NPU 维度拆分因子 | 影响跨 PE 通信并发度 |
| `commDataSplit` | 数据维度拆分因子 | 影响每 AIV 子核的搬运粒度 |

**workspace 容量约束**（必须满足，否则通信分片过大导致 buffer 不足）：

```
// ReduceScatter
(block_num * commInterval) % n_pes == 0

// AllGather / AllReduce
commInterval * M0 * K0 * block_num < symmetric_buff_bytes / dtype_size / n_pes / WORKSPACE_STAGES
```

**参考实现**：`examples/dynamic_tiling`（搜索空间遍历 + 约束过滤）

---

## 5. 跨 PE 同步原语选择

| 原语 | 语义 | 适用场景 |
| --- | --- | --- |
| `signal_op` + `signal_wait_until` | 点对点 PE 同步 | 已知对端 PE、需要精确握手（如 kv_shuffle sender↔receiver） |
| `aclshmem_barrier_all` | 全 PE 栅栏 | 所有 PE 数据一致后同步（**新算子首选**） |
| `AscendC::SyncAll` | 同一 PE 内多 AIV 核间同步 | 纯 Vector kernel 阶段切换（如 allgather Step 1→Step 2）；见 [SyncAll](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/SIMD-API/%E5%9F%BA%E7%A1%80API/%E5%90%8C%E6%AD%A5%E6%8E%A7%E5%88%B6/%E6%A0%B8%E9%97%B4%E5%90%8C%E6%AD%A5/SyncAll.md) |
| `AscendC::PipeBarrier` | 单 AIV 内流水线同步 | 同一 core 内 MTE/Vector 等 pipe 之间的顺序与可见性 |
| `aclshmem_quiet` | 等待本 PE 已发出的**全引擎** RMA 完成（MTE + SDMA + UDMA + RDMA） | 混合引擎或收尾阶段，需一次性 drain 所有在途传输 |
| `aclshmemx_mte_quiet` | 等待本地 MTE 完成 | **仅** MTE put/get 后确保数据落地；勿用全局 `aclshmem_quiet` 代替 |
| `aclshmemx_sdma_quiet` | 等待本地 SDMA 完成 | SDMA 传输后确保完成 |
| `aclshmemx_udma_quiet` | 等待指定 PE 的 UDMA 完成 | PCIE/SIO 等 UDMA 通路 |
| `aclshmemx_roce_quiet` | 等待指定 PE 的 RDMA/RoCE 完成 | 跨机 RDMA 通路 |
| `CrossCoreSetFlag` / `CrossCoreWaitFlag` | AIC↔AIV 跨核同步 | 通算融合的 stage 交替 |

> **选型**：单引擎场景 **SHOULD** 使用 engine-specific quiet（如 MTE 用 `aclshmemx_mte_quiet`），与 `aclshmemx_sdma_quiet` 对称；仅在多引擎混用或无法区分引擎时再调用 `aclshmem_quiet`。

> **不推荐**：`aclshmemi_barrier_core_soft` 为仓内 **internal** 接口（`shmemi_device_cc.h`），**无公开 Device API**；custom-ops 新算子 **禁止** 直接调用。同一 PE 内多 AIV 同步 **SHOULD** 使用 `AscendC::SyncAll`（纯 Vector 全核硬同步 `SyncAll<true>()`；需部分核参与时用软同步 `SyncAll(gmWorkspace, ubWorkspace, usedCores)`）。仅 legacy example（如 dispatch/combine）仍保留该 internal 调用，新代码勿仿照。

---

## 6. 数据分片公式

### 6.1 均分 + 尾块处理

**方式一：末核承担余数**（简单，但末核可能负载偏高）

```
len_per_core = total / n_cores
offset       = aiv_idx * len_per_core
if aiv_idx == n_cores - 1:
    len_per_core = total - len_per_core * aiv_idx   // 末核多/少分
```

参考：`allgather`、`kv_shuffle`

**方式二：余数散布到前 N 核**（均衡，最大偏差仅 1 个元素）

```
base_per_core = total / n_cores
extra         = total % n_cores
if aiv_idx < extra:
    my_len    = base_per_core + 1
    my_offset = aiv_idx * (base_per_core + 1)
else:
    my_len    = base_per_core
    my_offset = extra * (base_per_core + 1) + (aiv_idx - extra) * base_per_core
```

参考：`sdma`

### 6.2 对齐约束

| 传输引擎 | 对齐要求 | 说明 |
| --- | --- | --- |
| MTE | 建议 32B 对齐 | 非对齐可工作但效率降低 |
| SDMA | 512B 对齐 | 小于 512B 的偏移需按 512B 粒度对齐 |
| CATLASS tile | M0 / K0 / N0 整除 | GEMM 分块必须与 tile 尺寸对齐 |

---

## 7. 同步机制详解

### 7.1 per-AIV 独立 Flag

每个 AIV 拥有独立的同步 flag 槽，避免多核写同一地址造成竞争。

**地址计算**：

```
SYNC_FLAG_INTERVAL = 16   // int32 粒度间隔，防止 false sharing
flag_addr = gva_sync_base + (rank * aiv_num + aiv_idx) * SYNC_FLAG_INTERVAL
```

**总分配量**：`sizeof(int32_t) * n_pes * aiv_num * SYNC_FLAG_INTERVAL`

**使用模式**：
- 发送端写自己的 flag：`signal_op(my_flag_addr, value, SIGNAL_SET, target_pe)`
- 接收端读对端的 flag：`signal_wait_until(remote_flag_addr, CMP_EQ, expected_value)`

参考：`allgather`、`kv_shuffle`

### 7.2 Epoch / Magic Number 防脏读

多轮迭代中，上一轮的 flag 值可能残留在远端。通过 magic number 区分不同轮次：

```
// 发送端：每轮递增
signal_value = (magic + iteration) * 1024

// 接收端：通过高位比较判断 epoch
if (remote_flag >> 10) != (expected_magic >> 10):
    // 仍是旧 epoch 的值，继续等待
```

参考：`allgather` 大数据路径

### 7.3 Ping-Pong UB 双缓冲

两个 UB buffer 交替使用，配合两个 EVENT_ID，实现 DMA 与处理的重叠：

```
// 初始化两个 buffer 和对应事件
buf[0] = ub_base + ping_offset
buf[1] = ub_base + pong_offset
event[0] = EVENT_ID0
event[1] = EVENT_ID1
pp = 0  // ping-pong 切换标志

for each chunk:
    WaitFlag<MTE3_MTE2>(event[pp])          // 等待当前 buffer 上一次 DMA 完成
    mte_get_nbi(..., buf[pp], event[pp])    // 发起 DMA 到当前 buffer
    SetFlag<MTE3_MTE2>(event[pp])           // 标记 DMA 已发起
    pp = 1 - pp                              // 切换到另一个 buffer
```

**UB 布局示例**（多维度场景）：
- 每个独立维度分配独立的 ping/pong buffer 和独立的 EVENT_ID 对
- 如 KV shuffle：K 维度用 `(buf[0], buf[1], EVENT_ID0, EVENT_ID1)`，V 维度用 `(buf[2], buf[3], EVENT_ID2, EVENT_ID3)`

参考：`allgather`（大数据 recv 组）、`kv_shuffle`

### 7.4 同一 PE 内多 AIV 同步（核间）

| 方式 | 推荐 | 说明 |
| --- | --- | --- |
| `AscendC::SyncAll<true>()` | **首选**（纯 Vector 全核） | 硬件核间同步；kernel 需 `__mix__(0, 1)` 等正确修饰符，见 asc-devkit SyncAll 约束 |
| `AscendC::SyncAll(gm, ub, usedCores)` | 部分 AIV 参与时 | 软同步；`gmWorkspace` 需 Host 或 kernel 侧初始化为 0 |
| `AscendC::PipeBarrier<PIPE_*>` | 单 core 内 pipe 同步 | 不同 AIV 之间**不能**替代核间栅栏 |
| `aclshmemi_barrier_core_soft` | **禁止**（新算子） | internal 实现细节；仓库无公开等价 API，勿在 custom-ops 中使用 |

`aclshmemi_barrier_core` 在 MIX kernel 下内部会走 `SyncAll`；非 MIX 时 fallback 到 `barrier_core_soft`——custom-ops 应显式选用 `SyncAll`，不依赖该 fallback。

---

## 8. 设计原则

1. **先判断数据规模**：小数据优先少核少同步（模式 A）；大数据才扩大核数和拆分粒度（模式 B）
2. **优先按连续地址切片**：每个核处理连续区间，便于合并传输、减少地址计算
3. **通信核数不要无上限扩大**：SDMA 有通道上限，AIV 占用同步和 UB 资源；超过瓶颈后加核只增调度成本
4. **通信 tile 与计算 tile 要匹配**：通算融合时通信分片应能直接喂给 GEMM tile，减少中间 GM 读写
5. **按拓扑区分传输引擎**：跨机 RDMA 和机内 MTE/SDMA 延迟带宽差异大，按字节量比例分配核数（模式 B3）
6. **对不规则负载做均衡**：MoE token、KV block 可能分布不均，按数据量而非简单按 rank 平均
7. **缩小同步边界**：用 stage / phase / tile 级同步替代全局 barrier

## 9. 分核设计检查清单

- [ ] 算子类型已确定（纯通信 / 通算融合 / MoE）
- [ ] 数据规模已评估，分核模式已选择（A / B1 / B2 / B3 / CoC）
- [ ] `aiv_num` 满足与 `n_pes` 的整除约束
- [ ] 每个 AIV 的具体职责已写明（哪个 AIV 编号范围 → 哪个 PE / 哪段数据 / 哪个维度）
- [ ] 数据分片公式已写明（`len_per_core`、`offset`、尾块处理方式）
- [ ] 同步方式已选择（per-AIV flag / `AscendC::SyncAll` / cross-core flag / 跨 PE barrier）并写入 design；**禁止**新算子使用 `aclshmemi_barrier_core_soft`
- [ ] UB 布局已规划（ping-pong 分配、每个缓冲区大小、EVENT_ID 分配）
- [ ] 通算融合已确认 `commInterval` 和 workspace 容量约束
- [ ] 对齐约束已检查（MTE 32B / SDMA 512B / CATLASS tile 整除）
