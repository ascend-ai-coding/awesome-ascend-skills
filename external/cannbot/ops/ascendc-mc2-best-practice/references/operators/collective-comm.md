# 集合通信类算子设计模式

> 跨框架共性的集合通信类算子设计知识。框架无关的通信语义、轴切分、通算流水模式。框架具体实现见 `references/foundations/blaze-shmem/`、`references/foundations/apace/` 和 `references/foundations/hccl-matmul/`。

## 通信原语与典型算子

| 通信原语 | 典型融合模式 | 数据流 |
|---------|------------|--------|
| AllToAll | AllToAll + Matmul | 每个 rank 将 A 分块交给对应目的 rank → 接收方用来自各源 rank 的块做本地 Matmul |
| AllGather | AllGather + Matmul | 每 rank 持 A 的 1/rank → 拼成完整 A → Matmul |
| AllReduce | Matmul + AllReduce | 每 rank 算 C = A×B 部分和 → AllReduce 聚合 |
| ReduceScatter | Matmul + ReduceScatter | 每 rank 算 C 部分和 → ReduceScatter 输出 M/rank 段 |

### 通信原语选择判据

按各 rank 需要收到的数据选择通信原语。计算在通信之前还是之后，由融合算子的输入输出依赖决定。

| 判据 | AllToAll | AllGather | AllReduce | ReduceScatter |
|------|----------|-----------|-----------|---------------|
| **每个 rank 收到的数据** | 每个源 rank 发给本 rank 的块 | 全部源 rank 的输入分片 | 所有 rank 的输入逐元素归约后的完整结果 | 归约结果中分配给本 rank 的一段 |
| **典型场景** | 等长分块交换（如按目的卡重排 token） | TP 前向（每卡持 A 的 1/rank，拼完整 A 后独立 Matmul） | DDP 梯度聚合（各卡梯度 AllReduce 同步） | TP 反向梯度（Matmul 部分和按 M 切分输出） |
| **上表示例的计算依赖** | 交换输入块后计算 | 收到输入分片后计算 | 计算部分和后归约 | 计算部分和后归约并分发 |

**选择规则**：

1. **先定计算目标**：各 rank 需要完整输出还是部分输出？输入数据需要重排还是拼接？
2. **倒推通信原语**：
   - 各 rank 需完整 A 做 Matmul → AllGather（拼接输入）
   - 按目的 rank 交换等长分块 → AllToAll；专家路由的 token 数随对端变化时，按实际计数用 AllToAllV
   - 各 rank 提供待归约的贡献值且都需要完整结果 → AllReduce（如梯度同步）
   - 各 rank 提供待归约的贡献值且各自只需要一段结果 → ReduceScatter（如 TP 反向）
3. **验证流水可行性**：核对每个 tile 的生产、传输完成和消费顺序，找出可以同另一块计算重叠的传输。

> **实现路线**：blaze-shmem 直调通过 SHMEM 搬运数据；apace 直调须按[接口支持的原语和方向](../foundations/apace/fundamentals/communication.md)选取 `CollectiveComm` 实现。AllReduce 需要扩展组件；[compute-first ReduceScatter 场景](../foundations/apace/scenarios/compute-first-reduce-scatter/design.md)可用 AllToAll PUT 交换各分片的部分和，再在接收端累加（如 AtomicAdd）。910B 的[注册 / aclnn 路线](../foundations/hccl-matmul/)使用框架提供的 HCCL 高阶通信上下文。

## 切分轴语义

从各 rank 的张量形状和索引确定切分轴。下面是几种矩阵布局的示例：

| 切分轴 | 含义 | 决定 |
|--------|------|------|
| **通信切分轴** | 数据如何分布到各 rank | 根据接口布局确定发送块、接收后的拼接或切片维度 |
| **计算切分轴** | Matmul 输入或输出如何分块 | 沿 M/N 切分得到输出的不同行/列；沿 K 切分则需累加对应的部分积 |

### AllToAll + Matmul

```
P 个 rank，M 和 K 均可被 P 整除。源 rank i 持有 A 的第 i 个 K 分片 A_i[M × K/P]。
通信: A_i 沿 M 等分，目的 rank r 从每个源 i 收到 A_i[r*M/P:(r+1)*M/P, :]，形状为 [M/P × K/P]。
计算: rank r 持有完整 B，取对应的权重行块 B_i[K/P × N]，累加各接收块与 B_i 的乘积，得到 C_r[M/P × N]。
```

本例各目的 rank 合起来覆盖全部 M 行，每个目的 rank 在本地遍历 K 分片完成累加。

### AllGather + Matmul

```
rank i 持有: A_i[M_local × K]（M 的 1/rank 段）
通信: AllGather 拼接所有 rank 的 A_i → 完整 A[M × K]
计算: 完整 A × B → C[M × N]
```

切分轴：通信沿 M（拼接），计算无跨 rank 累加（每 rank 持有完整 A 后独立算）。

### Matmul + AllReduce

```
rank i 持有: A_i[M × K_local], B_i[K_local × N]
计算: C_i = A_i × B_i（部分和）
通信: AllReduce(C_i) → C = Σ C_i
```

切分轴：计算沿 K（部分和），通信沿全量（聚合）。

### Matmul + ReduceScatter

```
rank i 持有: A_i[M × K_local], B_i[K_local × N]
计算: C_i = A_i × B_i（部分和）
通信: ReduceScatter → rank r 获得 C[r*M/rank : (r+1)*M/rank, :] = Σ C_i 的 M/rank 段
```

切分轴：计算沿 K（部分和），通信沿 M（输出按 M 切分）。

## 通算流水模式

MC2 核心价值：通信与计算在 Kernel 内并行执行，通过流水掩盖通信开销。

### AIV 搬运输入、AIC 计算的流水示例

```
AIV（通信侧）                    AIC（计算侧）
─────────────                   ─────────────
tile 0: Commit(data[0])         tile 0: WaitFlag(0) → Matmul(data[0])
tile 1: Commit(data[1])         tile 1: WaitFlag(1) → Matmul(data[1])
...                              ...
```

掩盖条件：每个 tile 的计算耗时 ≥ 通信耗时，且流水深度足够。

### GET vs PUT 方向

| 方向 | 含义 | 适用场景 |
|------|------|---------|
| **PUT** | 发送方发起写，把本地块送到远端地址 | 发送方把已就绪的输入块或计算结果写入接收窗口 |
| **GET** | 接收方发起读，从远端地址取块 | 接收方在源数据就绪后读取输入分片或结果分片 |

> 框架具体实现差异见 `references/foundations/blaze-shmem/mc2_architecture.md` 和 `references/foundations/apace/fundamentals/fusion.md`。

### 跨核同步机制

AIV↔AIC 跨核同步是通算流水的核心：

| 机制 | 作用 |
|------|------|
| `CrossCoreSetFlag` / `CrossCoreWaitFlag` | AIV 写完通知 AIC 可读；AIC 读完通知 AIV 可覆盖 |
| `SyncAll` | 同卡参与核在共享阶段边界会合；远端数据何时可读由通信接口的完成与可见性语义决定 |
| `TeamBarrier` / `aclshmemx_barrier_all_vec` | 跨卡同步（apace 用 TeamBarrier；SHMEM 用 barrier_all_vec） |

> 同一跨核事件的 Set/Wait 应匹配同步模式和 flagId；PIPE 分别对应发信与等待侧的实际流水线。

## 框架映射

| 概念 | blaze-shmem 路线 | apace 路线 | hccl-matmul 路线 |
|------|-----------|-----------|------------------|
| 通信 API | `aclshmemx_udma_*` | `CollectiveComm<Op,Mode,T,Barrier>` 四段式 | `Hccl<HCCL_SERVER_TYPE_AICPU>` 高阶 V2 |
| 跨核同步 | `CrossCoreSetFlag/WaitFlag` | 同左 + `TeamBarrier` | Finalize 前 `CrossCoreSetFlag/WaitFlag` |
| 工程组织 | 独立 CMake 直调工程 | `kernel/<op>/` 复用 `block/` `tiling/` | aclnn 注册工程（`op_host`+`op_kernel`+`.run`） |
| 流水示例 | 4-buffer 流水 + M 轴切分 | localMatmul 0/1/2 + flag 编排 | prepare 全部 → 逐 tile Wait + M 轴 `tileCnt` |
| 参考文档 | `references/foundations/blaze-shmem/mc2_architecture.md` | `references/foundations/apace/fundamentals/fusion.md` | `references/foundations/hccl-matmul/mc2_architecture.md` |

## 约束共性

AIV+URMA 直调两条框架（blaze-shmem / apace）的集合通信类算子共享以下约束：

1. **Matmul 走 Blaze 模板** — 禁止 asc-devkit `AscendC::Matmul` 黑盒 API
2. **禁止 HCCL 高阶 API** — 这两种直调路线没有注册框架传入的 HCCL 集合通信上下文。
3. **架构白名单** — 仅 dav-3510（Ascend 950）已验证
4. **性能采集必须刷 L2 cache** — 前一轮热度污染本轮指标

hccl-matmul 路线（注册 / 910B）约束相反：必须走 HCCL 高阶 + `AscendC::Matmul`，无需 L2 flush。详见 [`../foundations/hccl-matmul/review-checklist.md`](../foundations/hccl-matmul/review-checklist.md)。

> 路线特有约束与逐项审查条件：blaze-shmem 路线见 [`../foundations/blaze-shmem/review-checklist.md`](../foundations/blaze-shmem/review-checklist.md)（HCCL 禁止清单见 [`../foundations/blaze-shmem/comm_shmem.md`](../foundations/blaze-shmem/comm_shmem.md) §5）；apace 路线见 [`../foundations/apace/review-checklist.md`](../foundations/apace/review-checklist.md)（基础约束论述见 [`../foundations/apace/architecture.md`](../foundations/apace/fundamentals/architecture.md) §10）。
