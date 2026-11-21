# 集合通信类算子设计模式

> 跨框架共性的集合通信类算子设计知识。框架无关的通信语义、轴切分、通算流水模式。框架具体实现见 `references/foundations/blaze-shmem/`、`references/foundations/apace/` 和 `references/foundations/hccl-matmul/`。

## 通信原语与典型算子

| 通信原语 | 典型融合模式 | 数据流 |
|---------|------------|--------|
| AllToAll | AllToAll + Matmul | 每 rank 持 M/rank 段 A → PUT 到远端 → 远端收全 A 后做本地 Matmul |
| AllGather | AllGather + Matmul | 每 rank 持 A 的 1/rank → 拼成完整 A → Matmul |
| AllReduce | Matmul + AllReduce | 每 rank 算 C = A×B 部分和 → AllReduce 聚合 |
| ReduceScatter | Matmul + ReduceScatter | 每 rank 算 C 部分和 → ReduceScatter 输出 M/rank 段 |

### 通信原语选择判据

选择通信原语取决于**数据分布目标**与**计算-通信重叠方向**：

| 判据 | AllToAll | AllGather | AllReduce | ReduceScatter |
|------|----------|-----------|-----------|---------------|
| **数据分布变化** | M 轴重排（每 rank 获得不同 M 段） | M 轴拼接（每 rank 获得完整 M） | 无分布变化（每 rank 获得完整 C） | M 轴切分（每 rank 获得 M/rank 段） |
| **通信方向** | PUT（推送输入到远端计算） | GET（拉取远端结果拼接） | 聚合（部分和累加） | 散射（部分和切分输出） |
| **典型场景** | MoE 专家路由（token 按专家分布到各卡） | TP 前向（每卡持 A 的 1/rank，拼完整 A 后独立 Matmul） | DDP 梯度聚合（各卡梯度 AllReduce 同步） | TP 反向梯度（Matmul 部分和按 M 切分输出） |
| **计算-通信重叠** | 通信在前，逐 tile 推送 → 远端逐 tile 计算 | 计算在前，逐 tile 出结果 → 通信拉回拼接 | 计算在前，通信在后聚合 | 计算在前，通信散射输出段 |
| **每 rank 输出** | 不同 M 段（需后续合并或各自独立使用） | 完整 A（可独立做 Matmul） | 完整 C（各 rank 相同） | M/rank 段 C（各 rank 不同段） |

**选择规则**：

1. **先定计算目标**：各 rank 需要完整输出还是部分输出？输入数据需要重排还是拼接？
2. **倒推通信原语**：
   - 各 rank 需完整 A 做 Matmul → AllGather（拼接输入）
   - 需将 token 按专家分发到各卡 → AllToAll（重排输入）
   - 各 rank 需完整 C（如梯度同步）→ AllReduce（聚合输出）
   - 各 rank 只需自己负责的 M 段 C（如 TP 反向）→ ReduceScatter（散射输出）
3. **验证流水可行性**：确认通信方向（PUT/GET）与计算侧（AIC Matmul）可逐 tile 重叠，掩盖条件为每 tile 计算耗时 ≥ 通信耗时。

> **约束**：四种原语在 **Kernel 直调**场景下均不走 HCCL 高阶 API（`Hccl::*`），而是通过 SHMEM（blaze-shmem 路线）或 CollectiveComm 四段式 API（apace 路线）实现。ReduceScatter 原语在 apace block 层未直接实现，其语义可通过 AllToAll PUT + AtomicAdd 模式替代（详见 [`../foundations/apace/fusion.md`](../foundations/apace/fundamentals/fusion.md) §6）。**注册 / aclnn** 形态下四种原语走 HCCL 高阶（hccl-matmul 路线，910B 已验证），见 [`../foundations/hccl-matmul/`](../foundations/hccl-matmul/)。

## 切分轴语义

MC2 算子涉及两个切分维度：

| 切分轴 | 含义 | 决定 |
|--------|------|------|
| **通信切分轴** | 数据如何分布到各 rank | AllToAll 沿 M 推送；AllGather 沿 M 拼接；ReduceScatter 沿 M 输出段 |
| **计算切分轴** | Matmul 的 K 轴部分和累加 | 每 rank 持 K/rank 段 → 各 rank 算部分 C → 跨 rank 累加 |

### AllToAll + Matmul

```
rank i 持有: A_i[M_local × K], B[K × N]（全量复制或按 K 切分）
通信: rank i 将 A_i 按 M/rank 段 PUT 到各 target rank
计算: 每 rank 收到所有 rank 的 A 段后，遍历 rank 做 Matmul 累加 → C[M × N]
```

切分轴：通信沿 M（每 rank 获得完整 M），计算沿 K（每 rank 的 K 段做部分和）。

### AllGather + Matmul

```
rank i 持有: A_i[M_local × K]（M 的 1/rank 段）
通信: AllGather 拼接所有 rank 的 A_i → 完整 A[M × K]
计算: 完整 A × B → C[M × N]
```

切分轴：通信沿 M（拼接），计算无跨 rank 累加（每 rank 持有完整 A 后独立算）。

### AllReduce + Matmul

```
rank i 持有: A_i[M × K_local], B[K_local × N]
计算: C_i = A_i × B_i（部分和）
通信: AllReduce(C_i) → C = Σ C_i
```

切分轴：计算沿 K（部分和），通信沿全量（聚合）。

### ReduceScatter + Matmul

```
rank i 持有: A_i[M × K_local], B[K_local × N]
计算: C_i = A_i × B_i（部分和）
通信: ReduceScatter → rank r 获得 C[r*M/rank : (r+1)*M/rank, :] = Σ C_i 的 M/rank 段
```

切分轴：计算沿 K（部分和），通信沿 M（输出按 M 切分）。

## 通算流水模式

MC2 核心价值：通信与计算在 Kernel 内并行执行，通过流水掩盖通信开销。

### 通用流水结构

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
| **PUT**（通信→计算） | AIV 先推数据到远端，AIC 从 Win 区读取计算 | AllToAll+Matmul（推送输入数据） |
| **GET**（计算→通信） | AIC 先算结果写到 Win 区，AIV 从远端拉回 | AllGather+Matmul（拉取结果段） |

> 框架具体实现差异见 `references/foundations/blaze-shmem/mc2_architecture.md` 和 `references/foundations/apace/fundamentals/fusion.md`。

### 跨核同步机制

AIV↔AIC 跨核同步是通算流水的核心：

| 机制 | 作用 |
|------|------|
| `CrossCoreSetFlag` / `CrossCoreWaitFlag` | AIV 写完通知 AIC 可读；AIC 读完通知 AIV 可覆盖 |
| `SyncAll` | 保证所有 block 完成当前轮通信（PUT 模式需要） |
| `TeamBarrier` / `aclshmemx_barrier_all_vec` | 跨卡同步（apace 用 TeamBarrier；SHMEM 用 barrier_all_vec） |

> flag 配对不变量：AIV SetFlag 的 `<MODE, PIPE, flagId>` 必须 == AIC WaitFlag 的三元组。

## 框架映射

| 概念 | blaze-shmem 路线 | apace 路线 | hccl-matmul 路线 |
|------|-----------|-----------|------------------|
| 通信 API | `aclshmemx_udma_*` | `CollectiveComm<Op,Mode,T,Barrier>` 四段式 | `Hccl<HCCL_SERVER_TYPE_AICPU>` 高阶 V2 |
| 跨核同步 | `CrossCoreSetFlag/WaitFlag` | 同左 + `TeamBarrier` | Finalize 前 `CrossCoreSetFlag/WaitFlag` |
| 工程组织 | 独立 CMake 直调工程 | `kernel/<op>/` 复用 `block/` `tiling/` | aclnn 注册工程（`op_host`+`op_kernel`+`.run`） |
| 流水模式 | 4-buffer 流水 + M 轴切分 | localMatmul 0/1/2 + flag 编排 | prepare 全部 → 逐 tile Wait + M 轴 `tileCnt` |
| 参考文档 | `references/foundations/blaze-shmem/mc2_architecture.md` | `references/foundations/apace/fundamentals/fusion.md` | `references/foundations/hccl-matmul/mc2_architecture.md` |

## 约束共性

AIV+URMA 直调两条框架（blaze-shmem / apace）的集合通信类算子共享以下约束：

1. **Matmul 走 Blaze 模板** — 禁止 asc-devkit `AscendC::Matmul` 黑盒 API
2. **禁止 HCCL 高阶 API** — 禁止 `Hccl::*`（服务端调度，无法通算融合）
3. **架构白名单** — 仅 dav-3510（Ascend 950）已验证
4. **性能采集必须刷 L2 cache** — 前一轮热度污染本轮指标

hccl-matmul 路线（注册 / 910B）约束相反：必须走 HCCL 高阶 + `AscendC::Matmul`，无需 L2 flush。详见 [`../foundations/hccl-matmul/review-checklist.md`](../foundations/hccl-matmul/review-checklist.md)。

> 路线特有约束与逐项审查条件：blaze-shmem 路线见 [`../foundations/blaze-shmem/review-checklist.md`](../foundations/blaze-shmem/review-checklist.md)（HCCL 禁止清单见 [`../foundations/blaze-shmem/comm_shmem.md`](../foundations/blaze-shmem/comm_shmem.md) §5）；apace 路线见 [`../foundations/apace/review-checklist.md`](../foundations/apace/review-checklist.md)（基础约束论述见 [`../foundations/apace/architecture.md`](../foundations/apace/fundamentals/architecture.md) §10）。
