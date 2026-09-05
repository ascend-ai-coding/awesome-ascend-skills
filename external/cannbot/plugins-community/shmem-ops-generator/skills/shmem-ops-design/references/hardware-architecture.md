# 昇腾硬件架构与 SHMEM 约束

本文描述与 SHMEM 算子设计相关的昇腾 AI 处理器硬件特性和约束。设计和性能优化时必须参考本文中的硬件参数，不得使用错误的核数或带宽值。

## 1. AI Core 架构

昇腾 AI 处理器的 AI Core 是算子执行的基本单元。910B/910C 采用 AIC（AI Compute Core）与 AIV（AI Vector Core）**分离架构**（区别于 910A 的耦合模式），AIC 与 AIV 之间不具备直连数据通道，所有交互经由 L2 完成。

每个 AI Core 包含：

- **Scalar 单元**：标量计算和控制流，指令分发
- **Vector 单元**：向量计算，支持 fp16/bf16/fp32/int32 等类型的向量运算
- **Cube 单元**：矩阵计算，执行矩阵乘（MatMul），支持 fp16/bf16 输入、fp32 累加。单个 Cube 在 max 配置（16x16x16）下，一个时钟周期完成 4096 个 FP16 MACs
- **MTE（Memory Transfer Engine）**：
  - MTE2：外部存储 -> AI Core 内部存储（GM -> UB/L0）
  - MTE3：AI Core 内部存储 -> 外部存储（UB/L0 -> GM）
  - MTE1：AI Core 内部存储间搬运

## 2. 硬件参数速查表

### 2.1 核数与算力

| 参数 | 910B1 | 910B2 | 910B3/B4 | 910C |
| --- | --- | --- | --- | --- |
| AIC (Cube Core) 数量 | 25 | **24** | **20** | ~40-48 (双 die) |
| AIV (Vector Core) 数量 | 50 | **48** | **40** | ~80-96 (双 die) |
| 每 AIC 含 Cube 单元 | 1 | 1 | 1 | 1 |
| 每 AIC 含 Vector 单元 | 2 | 2 | 2 | 2 |
| FP16 算力 (TFLOPS) | ~414 | ~376 | ~320 | ~800 |
| INT8 算力 (TOPS) | ~828 | ~752 | ~640 | ~1600 |

> **重要**：设计文档中 `block_dim` 不得超过目标 SoC 的 AIV 核数。910B2 最多 48 个 AIV，910B3 最多 40 个 AIV。将 AIC 数量错设为 64 是常见错误。

### 2.2 存储层次与容量

| 存储 | 缩写 | 容量 | 特性 | SHMEM 用途 |
| --- | --- | --- | --- | --- |
| Global Memory (HBM) | GM | 64 GB (910B) / 96 GB (910C) | 大容量、高带宽、所有 core 可见 | 对称堆、输入输出 buffer、signal/state |
| Unified Buffer | UB | **192 KB** (per core) | 每 core 私有、低延迟 | DMA 中转、本地计算中间结果 |
| L1 Buffer | L1 | 512 KB | Cube 单元输入缓存 | MatMul 数据加载 |
| L0A/L0B | L0 | 各 64 KB | Cube 单元寄存器级缓存 | MatMul 操作数 |
| L0C | L0C | 128 KB | Cube 单元累加器 | MatMul 累加结果 |

关键容量约束：
- UB 总容量 192 KB/core；SHMEM examples 常用 `190*1024` 字节作为单次 DMA 搬运上限
- GM（HBM）容量决定对称内存最大 `local_mem_size`；SHMEM 默认对称堆上限 `100*1024*1024` 字节
- L0C 容量限制单次 MatMul 的 tile size

### 2.3 带宽参数

| 参数 | 910B1/B2 | 910B3 | 910C（双 Die） |
| --- | --- | --- | --- |
| HBM 带宽 | ~1.2 TB/s | ~1.2 TB/s | ~1.8-3.2 TB/s |
| **HCCS P2P 单链路带宽（单向）** | **~28 GB/s** | **~28 GB/s** | **~28 GB/s** |
| **HCCS P2P 单链路带宽（双向）** | **~56 GB/s** | **~56 GB/s** | **~56 GB/s** |
| HCCS 链路数（per NPU / per Die） | 7（full-mesh 直连其余 7 卡） | 7 | 7（per Die，连接到交换机） |
| P2P带宽 | 28GB/s(单向) | 28GB/s(单向) | 196GB/s(单向/单die) |
| L2 Cache 片上访问带宽 | ~4 TB/s | ~4 TB/s | - |

> **注意**：910C中同一卡上两个DIE通过SIO链路，P2P带宽为270GB/s。 196GB/s是不同卡上的DIE之间的带宽

### 2.4 服务器拓扑

| 服务器型号 | 芯片 | 卡数 | 节点内互联 | 节点间互联 |
| --- | --- | --- | --- | --- |
| Atlas 800T A2 | 910B | 8 | HCCS **full-mesh**（每 NPU 7 根 HCCS 直连其余 7 卡，任意两卡 28 GB/s 单向） | RoCE v2 (200 Gbps) |
| Atlas 800T A3 | 910B3 | 8 | HCCS **full-mesh** | RoCE v2 (200 Gbps) |
| 910C 集群 | 910C | 多卡 | HCCS **交换机组网**（每 Die 7 根 HCCS 连交换机，Die 间 ~196 GB/s） | RoCE v2 |

### 2.5 拓扑与集合通信算法

| 拓扑 | HCCL 典型算法 | 特征 | 适用场景 |
| --- | --- | --- | --- |
| full-mesh（910B 单机 8 卡） | **mesh 算法** | 所有 peer 链路并发收发，充分利用 N-1 条 HCCS | 单机 AllReduce/AllGather/ReduceScatter |
| switch（910C 双 Die） | 视 switch 带宽选择 | Die 间经 switch 转发，非直连，跨 Die 带宽受限 | 910C 节点内集合通信 |
| ring | ring 算法 | 沿环逐跳传递，每跳只用一条链路 | 跨节点 / ring 拓扑场景 |

性能优化时，如果当前实现的通信模式（如 peer 串行 get）与硬件拓扑不匹配（如 full-mesh 下只用了 1 条链路），应优先做**算法级重构**而非细粒度调参。

### 2.6 peak_bandwidth 选取

性能报告中的 `peak_bandwidth` 应按实际通信模式选取，而非一律使用单链路带宽：

| 通信模式 | peak_bandwidth | 说明 |
| --- | --- | --- |
| P2P 点对点（put/get 到单个 peer） | 28 GB/s | 单条 HCCS 链路单向 |
| 集合通信（AllReduce/AllGather 等） | N_links × 28 GB/s（如 910B3 8 卡: 7 × 28 = 196 GB/s） | 聚合带宽，因为 mesh 算法同时使用多条链路 |

报告中必须注明 `peak_bandwidth` 使用的值、SoC 型号和链路来源。

## 3. DMA 与传输引擎

### 3.1 MTE（片上 DMA）

SHMEM 通过 MTE 实现 GM <-> UB 和 GM <-> GM 的数据搬运。

- `aclshmemx_mte_put_nbi`：通过 MTE 将本地 GM 数据写入远端 PE 的对称内存
- `aclshmemx_mte_get_nbi`：通过 MTE 从远端 PE 的对称内存读取数据到本地
- 需要 UB buffer 作为中转
- 适合节点内 P2P 可达场景，延迟低
- 单次最优搬运大小：**190 KB**（`190*1024` 字节），最小 16 KB

### 3.2 RDMA/RoCE

SHMEM 通过 RDMA 引擎实现跨节点或 P2P 不可达场景的远程访问。

- `aclshmemx_rdma_put_nbi` / `aclshmemx_rdma_get_nbi`
- 不需要 UB 中转，直接 GM 到 GM
- 适合大数据量跨节点传输
- 需要 `-enable_rdma` 编译选项

### 3.3 SDMA

系统级 DMA，由 Host 侧发起。

- 适合 Host <-> Device 数据拷贝
- 部分场景可用于 Device 间大块搬运

## 4. 同步机制

### 4.1 核间同步

- `AscendC::SyncAll`：同一 PE 内多 AIV 核间同步（纯 Vector 首选；见 asc-devkit [SyncAll](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/SIMD-API/%E5%9F%BA%E7%A1%80API/%E5%90%8C%E6%AD%A5%E6%8E%A7%E5%88%B6/%E6%A0%B8%E9%97%B4%E5%90%8C%E6%AD%A5/SyncAll.md)）
- `AscendC::PipeBarrier<PIPE_*>`：单 AIV 内各流水线阶段同步
- **禁止** custom-ops 新代码调用 internal `aclshmemi_barrier_core_soft`（无公开 API；legacy example 除外）

### 4.2 跨 PE 同步

- `aclshmemx_signal_op`：向远端 PE 的 signal 地址写入值
- `aclshmem_signal_wait_until`：等待本地 signal 地址满足条件
- `aclshmem_barrier_all` / `aclshmem_team_sync`：集合同步
- `aclshmem_quiet`：等待本 PE 已发出的**全引擎** RMA 完成（MTE + SDMA + UDMA + RDMA）
- `aclshmemx_mte_quiet` / `aclshmemx_sdma_quiet` / `aclshmemx_udma_quiet` / `aclshmemx_roce_quiet`：单引擎 quiet；**仅 MTE put/get 时用 `aclshmemx_mte_quiet`，勿误用全局 `aclshmem_quiet`**
- `aclshmem_fence`：保证本 PE 后续读取看到之前收到的写入（当前实现等价于 `aclshmem_quiet`）

### 4.3 FFTS（Fast Function Transfer and Synchronization）

SHMEM 使用 FFTS 机制配置 kernel launch 参数和同步基址。`util_get_ffts_config()` 获取 FFTS 配置地址，通过 launch 参数传入 kernel。

## 5. 分核模型

| 概念 | 说明 |
| --- | --- |
| Block Dim | kernel launch 时指定的 AI Core 数量 |
| Block Idx | 当前 core 的编号，`GetBlockIdx()` 获取 |
| Block Num | 当前 kernel 的总 core 数，`GetBlockNum()` 获取 |

SHMEM 算子的典型分核模式：
- 前 N 个 core 负责通信（comm core），后 M 个 core 负责计算（compute core）
- 按 `GetBlockIdx()` 分支实现不同职责
- 动态 block_dim：小数据用少量 core（如 8），大数据用更多 core（如 `2*n_pes`）

### 5.1 分核上限约束

| SoC | AIV (Vector) 核数上限 | AIC (Cube) 核数上限 |
| --- | --- | --- |
| 910B2 | 48 | 24 |
| 910B3 | 40 | 20 |

`block_dim` 总数不得超过 AIV 核数。

## 6. 数据来源与注意事项

本文硬件参数综合自以下来源：
- 华为 CANN 工具包 `platform_config` 配置文件
- HAMi 虚拟化配置（aiCore / aiCPU 数量）
- 公开硬件评测与架构分析文章

> 华为昇腾芯片的详细参数并未完全公开，部分数据基于公开信息整合。如需精确参数，可通过 `npu-smi info` 查询实际硬件配置，或解析 CANN 工具包中的 `platform_config` 目录获取芯片子产品参数。
