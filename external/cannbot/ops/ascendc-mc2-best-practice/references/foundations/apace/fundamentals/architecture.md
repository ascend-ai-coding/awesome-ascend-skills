# APACE 架构总览

> 本文档是 apace 路线的"心智模型入口"。首次接触 apace 的开发者读完应建立：三层架构 → 组合模式 → GET/PUT 方向 → AIC/AIV 流水 → 6 大技术索引 → 与 blaze-shmem 路线的边界。
>
> 主教学样例为官网 PUT 算子 `all_to_all_quant_matmul`（UDMA 变体）。

## 目录

1. [APACE 是什么](#1-apace-是什么)
   - 1.5 [方法论原则](#15-方法论原则apace-算子开发通用)
2. [三层架构](#2-三层架构)
3. [实现模式](#3-实现模式)
4. [通信方向：GET vs PUT](#4-通信方向get-vs-put)
5. [算子对比表](#5-算子对比表)
6. [6 大关键技术索引](#6-6-大关键技术索引)
7. [AIC/AIV 分工与流水](#7-aicaiv-分工与流水)
8. [四段式通信 API 概览](#8-四段式通信-api-概览)
9. [与 blaze-shmem 路线对照](#9-与-blaze-shmem-路线对照)
10. [基础约束（apace 路线红线）](#10-基础约束apace-路线红线)
11. [新建算子文件清单 + 导航](#11-新建算子文件清单--导航)

---

## 1. APACE 是什么

**APACE**（**A**scend **P**Arallel **C**ommunication-compute **E**ngine）是昇腾 NPU 平台通算融合算子的架构底座，为各类通算融合算子提供可复用的 block 层接口、kernel 层参考实现和 tiling 算法。

**核心价值**：
- **降低开发门槛**：提供分层接口与参考实现，开发者通过接口组合即可构建融合算子，无需从零实现通信与计算的协同编排
- **提升性能**：通信与计算深度耦合，支持流水线重叠，充分发挥硬件算力
- **模块化复用**：block 层接口稳定可复用，kernel 层参考实现方便快速移植

**与 mc2 算子的关系**：

apace 支持**Kernel 直调**与**框架注册**两种调用形态。CANNBot 工作流以直调为主（`<<<>>>` 入口），注册形态为可选扩展。

**直调形态**（CANNBot 默认，官网两个 UDMA 算子均为此形态）：

```
kernel/<op>/<op>_impl.h          ──▶ apace/block        (block 接口，组合构建)
  └─ __global__ 入口              ──▶ apace/kernel       (参考实现)
tests/st/<op>/src/kernel_launcher.h  ──▶ <<<>>> 直调入口 (4 个 dtype 变体)
kernel/<op>/<op>_tiling_data.h   ──▶ apace/tiling       (tiling 接口)
```

**注册形态**（框架注册，含 op_host/op_kernel 完整工程，HCCL windows/CCU 变体仅限此形态）：

```
mc2/<op>/op_host    ──┐
                      └──▶ apace/tiling       (tiling 接口)
mc2/<op>/op_kernel  ──┐
                      ├──▶ apace/block        (block 接口，组合构建)
                      └──▶ apace/kernel       (算子框架，参考实现)
```

> ⚠️ **直调限制**：HCCL windows/CCU 模式无 `__global__` 入口，不支持直调（红线，详见 §10 ④）。

直调形态下，`kernel/<op>/` 下创建 Impl 类（含 `__global__` 入口），复用 `apace/block/` 接口组合构建融合 kernel，tiling 结构体引用 `apace/tiling/`。注册形态下，op_host 层使用 `apace/tiling` 切分算法，op_kernel 层基于 `apace/block` 接口组合构建。

---

## 1.5 方法论原则（apace 算子开发通用）

| # | 原则 | 内涵 |
|---|------|------|
| 1 | **事实源唯一** | 工程事实源 = CANN 内置 apace 直引（禁止整包复制）；知识事实源 = 每条契约单文件出处，他处只引用 |
| 2 | **设计三拷问** | 动手前依次回答：① golden 语义（每卡输入/输出分布与切分轴）② 通信方向（compute-first 还是 comm-first）③ 编排形态（严格分离还是时分复用）——三问答错，后续全部返工 |
| 3 | **契约驱动实现** | 每个实现决策对应一条"不变量 + 违反后果"契约（per-tile 子区间、winOffset、分核映射、flag 配对、归约批量）——契约来自生产失败案例，不是风格建议 |
| 4 | **失败模式前置** | 优先用失败案例检查实现：每条契约都来自生产失败案例而非风格建议——完整失败案例库（现象 → 根因 → 修复方向）见 [`review-checklist.md`](../review-checklist.md) 常见 FAIL 原因表 |
| 5 | **验收分层递进** | 编译 → 冒烟（T=1 单 rank 非全 0）→ 精度（T=1/T>1 双路径 × 多 rank × 多 shape）→ 性能（真实 shape × 多 rank 档 × 三路径对标）——每层门禁不过不进下一层 |
| 6 | **示例与合同边界纪律** | 单实例常量（winOffset、flagId、分核编号等）不得升格为普适门禁；示例值必须标注示例属性；路线条件规则（PUT/compute-first/GET）必须分路线给出，禁止以单路线形态充当通用合同 |

---

## 2. 三层架构

APACE 采用 **basic → block → kernel** 三层架构（自底向上），稳定性递减，抽象层级递减。依赖方向严格单向：kernel → block + tiling + 外部库（blaze/tensor_api/hcomm）。

```
┌─────────────────────────────────────────────────────┐
│  kernel/  (不稳定层 — 算子参考实现)                  │
│  新建算子在此创建，可直接修改                        │
├─────────────────────────────────────────────────────┤
│  block/   (稳定层 — 可复用单核接口)                  │
│  ├── aiv_comm/    AIV 核通信接口 (CRTP 四段式)       │
│  ├── blaze_ext/   Blaze 引擎通算融合扩展              │
│  └── aiv_compute/ AIV 核计算接口 (空占位)            │
├─────────────────────────────────────────────────────┤
│  basic/   (最底层 — tile 级内存抽象)                 │
│  └── fragment_tensor/  离散内存虚拟重排              │
├─────────────────────────────────────────────────────┤
│  tiling/  (稳定层 — Host 侧切分算法)                 │
│  utils/   (稳定层 — 通用工具与常量)                  │
└─────────────────────────────────────────────────────┘
```

### 各层职责与代表文件（对照官网 master 实际目录）

| 层 | 目录 | 职责 | 代表文件 |
|:---|:---|:---|:---|
| **basic** | `basic/fragment_tensor/` | tile 级内存抽象，FragmentTensor 离散内存虚拟重排 | `basic/fragment_tensor/fragment_tensor.h` |
| **block** | `block/aiv_comm/` | AIV 核通信接口，CRTP 四段式 API + 编译期分发 | `block/aiv_comm/collective_comm_api.h`、`all_to_all/all_to_all_udma_put.h`、`all_gather/all_gather_udma_put.h`、`barrier/barrier_ubmem.h` |
| | `block/blaze_ext/gemm/block/` | Blaze 引擎在通算融合场景下的扩展实现 | `block/blaze_ext/gemm/block/qmm_mx_block_mmad_fragment.h` |
| | `block/blaze_ext/epilogue/` | epilogue 扩展 | *(当前为空目录，委托模式 epilogue 官网未上库)* |
| | `block/aiv_compute/` | AIV 核计算接口 | *(当前为空目录)* |
| **kernel** | `kernel/<op>/` | 通算融合算子参考实现（官网 2 个算子目录） | `kernel/all_to_all_quant_matmul/`、`kernel/all_gather_quant_matmul/` |
| **tiling** | `tiling/` | Host 侧切分算法与 TilingData 结构定义 | `tiling/comm_tiling_data.h`、`tiling/quant_matmul_tiling_data.h` |
| **utils** | `utils/` | 通用工具与常量 | `utils/constant.h`、`utils/comm_channel_builder.h` |
| **tests** | `tests/st/` | ST 测试（2 个算子各一套） | `tests/st/all_to_all_quant_matmul/`、`tests/st/all_gather_quant_matmul/` |
| **docs** | `docs/` | 设计文档 | *(当前仅有 .gitkeep 占位)* |

### 稳定性规则

- **block/ 和 tiling/ 是稳定共享层**：除非创建新通信原语（超出常规开发范围），否则只在 `kernel/<op>/` 下创建文件
- **kernel/ 是不稳定层**：可直接参考、复制、修改
- 依赖方向单向：kernel → block + tiling + Blaze，禁止反向依赖

---

## 3. 实现模式

apace kernel 层的 Blaze 集成方式：

| 模式 | 说明 | 适用场景 | 官网状态 |
|:---|:---|:---|:---|
| **组合模式** | 自定义 matmul kernel + 通信对象，精细控制逐 tile 同步 | 需要通信-计算重叠，AIC/AIV 逐 tile 协同 | ✅ 官网 2 个算子均为组合模式 |
| **委托模式** | matmul 委托 GemmUniversal，仅加通信 prologue/epilogue | matmul 可标准化，通信仅在前后 | ⚠️ 官网未上库（`block/blaze_ext/epilogue/` 为空目录） |

### 组合模式文件构成（all_to_all_quant_matmul）

```
kernel/all_to_all_quant_matmul/
├── all_to_all_mx_quant_matmul_udma_impl.h    # Impl 类（UDMA PUT 变体）
├── all_to_all_mx_quant_matmul_hcomm_impl.h   # Impl 类（HCCL CCU 变体，框架注册场景）
├── quant_matmul_mx_kernel.h                  # 自定义 Blaze matmul 调度（AIC 侧计算，CommPolicy 注入）
└── all_to_all_matmul_tiling_data.h           # tiling 结构体 + CommContext
```

组合模式的核心特征：
- Impl 类直接持有 `CollectiveComm`（通信）+ matmul kernel（计算），不通过 GemmUniversal 包装
- PUT 模式：AIV 侧 `RunAllToAll()` 逐轮推 scale/data 到远端 Win（Commit 顺序 scale→data），`CrossCoreSetFlag` 通知 AIC；AIC 侧 kernel 内经 `CommPolicy::WaitTile` 等待通信 tile 就绪后计算

> 详见 [`operator-anatomy.md`](../operator-design/operator-anatomy.md)（Impl 骨架）和 [`compute.md`](compute.md)（Blaze 集成）

---

## 4. 通信方向：GET vs PUT

apace 通信方向由数据流决定：**GET = 计算→通信**（AIC 先算 C 写 Win 区，AIV 从远端拉回本 rank 段，沿 N 轴切分）；**PUT = 通信→计算**（AIV 先推数据到远端 Win 区，AIC 读取计算，沿 K 轴切分）。方向语义、钩子差异与 flag 编排对比以 [`fusion.md`](fusion.md) §2 为唯一事实源，本节只保留切分轴决策与 B 分布模式。

> M 轴是通用首选切分轴（不同 M 段独立通信/计算），但 GET 模式按 N 轴切分（每 rank 持有 N/rankSize 列），PUT 模式按 K 轴切分（每 rank 持有 K/rankSize 行）。官网 `all_gather_quant_matmul` 即按 M 轴切分。GET 模式官网暂无算子样例（见 §5）。

> ⚠️ **术语消歧**：本节「切分轴 = K」指 **rank 级数据分布轴**（每 rank 持有 A 的 K/rankSize 段）；而 `CommTilingData.splitAxis*` 字段沿 **M** 轴推进（通信 tile 经 `headMSize` 沿 M 切分，`nonSplitAxisSize = ka`）。字段映射详见 [`operator-anatomy.md`](../operator-design/operator-anatomy.md) §3，填 tiling 字段时不要混用两个"切分轴"概念。

### PUT 模式数据流（all_to_all_quant_matmul 实测语义）

```
全局逻辑 A 为 [M_total, K]，M_total = m × rankSize（m = 单卡输出行数）
                                     ↓ 输入按 K 轴分布（每卡全 M 行 × 本卡 K 段）
                     每卡持有 A 分片 [M_total, Ka]，Ka = K/rankSize
                                     ↓ AllToAll PUT 沿 M 行块（通信 tile 沿 M 推进）
                     AIV 把本卡 A/scaleA 的第 r 个 m 行块推给 rank r 的 Win 区
                     AIC 从本卡 Win 区收齐 R 个 m 行块（拼成 [m, K]），遍历 rank 乘 B 对应 K 段
                     L0C 累加部分和（localMatmul=1 时远端部分 AtomicAdd 落 C）
                                     ↓ 输出按 M 轴分布
                     每卡输出本卡 M 段 C = [m, N]（全局 [M_total, N]）
```

> ⚠️ 三处易错点（均经官方 ST `gen_data.py`/`main.cpp` 实证）：① 每卡输入是**全部 M 行**的 K 段（不是本卡 M 段的 K 段）；② 每卡输出是 `[m, N]` M 段（不是完整 `[M, N]`）；③ B 全量复制且逻辑布局为 R 个 K 段拼接 `[K, N]`（DN 列主序存储）。

### 切分轴决策表

为新算子选择切分轴时，按以下维度评估：

| 切分轴 | 通算并行性 | 实现难度 | 适用场景 | 官网样例 |
|:---|:---|:---|:---|:---|
| **M 轴** | 高（不同 M 段完全独立） | 低（A/C 都按 M 切，偏移简单） | 通用首选，适合 AllGather/ReduceScatter | all_gather_quant_matmul |
| **N 轴** | 中（需协调 B 的列切分） | 中（B 偏移 = rankOffset × N/rankSize × K） | GET 模式（计算→通信），每 rank 持有 C 的 N/rankSize 列 | ⚠️ 官网暂无 |
| **K 轴** | 低（需跨 rank 规约部分和） | 高（需要额外的 Reduce 步骤或 PUT 模式） | PUT 模式（通信→计算），每 rank 持有 B 的 K/rankSize 行 | all_to_all_quant_matmul |

**决策原则**：
1. **GET 模式（计算→通信）优先 N 轴**：AIC 算完 C 后，AIV 按 N 轴从远端拉取本 rank 应得的列段
2. **PUT 模式（通信→计算）优先 K 轴**：AIV 先把 A 的 K 段推到远端，AIC 从 Win 区读取计算
3. **M 轴是通用首选**：如果通信模式允许（如 AllGather），M 轴切分最简单且并行性最高

### B 数据分布模式

通信方向不仅决定切分轴，还决定 B 矩阵的数据分布方式：

| 通信模式 | B 分布 | 每 rank 持有 | 典型场景 |
|:---|:---|:---|:---|
| **GET**（计算→通信） | N-split | B_i[K×N_local] | N 轴 AllToAll，每 rank 输出不同 N 段 |
| **PUT**（通信→计算） | 全量复制 | B_full（rankSize × K_local × N 逻辑拼接） | K 轴 AllToAll + AtomicAdd 累加 |

**PUT 模式 B 布局**（官网样例）：每 rank 持有全部 rank 的 B K 段拼接，AIC 按远端 A 的来源 rank 选择对应的 B K 段（`quant_matmul_mx_kernel.h` `ProcessSingleBatch` 中 `gmB.Slice(rank * K, nPos)`）。

**选择规则**：
- B 全量复制 → 内存开销大（rankSize 倍），但 AIC 可按 rank 索引直接访问对应 K 段
- B N-split → 内存节省，每 rank 持有 B 的 N/rankSize 列
- **泛化规则**：PUT 模式（含 AtomicAdd 串行与 3 级流水 workspace 两种实现）要求 B 全量复制；GET 模式 B 可 N-split

> ScaleB 的内存分配公式与 B 分布一致，详见 [`operator-anatomy.md`](../operator-design/operator-anatomy.md) §3。

### tileSize 约束

`tileSize`（通信 tile 沿切分轴的大小）选择需满足以下约束：

| 约束 | 说明 |
|:---|:---|
| **baseM 整数倍** | 切分轴 tile 大小应取 Blaze base 块的整数倍，否则尾块处理复杂化 |
| **单次传输字节数** | ⚠️ 风险提示（非硬上限）：无官方约束（bring-up 期过大单轮曾见间歇失败，边界未定）——大单轮 PUT 按 case 复核，不作 shape 拒绝依据；带宽高效区间与可靠性上限是两个独立概念——唯一定义见 [`communication.md`](communication.md) 陷阱 #13 |
| **Win 区空间预算** | Win 区总需求不能超过容量（通常几十 MB） |
| **通信 tile 总数 ≤ 32** | AIC 侧 `waitedMask` 为 `uint32_t`，超出静默出错（详见 [`fusion.md`](fusion.md) §4.2）；与轮次索引型 flagId 的轮次索引 < FLAG_ID_MAX(16) 且避开 SyncAll 保留区（[`fusion.md`](fusion.md) §3.3）是两套机制的约束，取更严者；计数式固定 flagId 不受轮次数限制 |

**经验法则**：
- `tileCnt` 扫描范围 {1, 2, 4, 8, 16}（受上表两套上限约束：以轮次 tid 作 flagId 的编排下轮次索引 < 16 且避开 SyncAll 保留区；计数式固定 flagId 编排下按 case 实测确定上限）
- `tileCnt` 增大 → 通信粒度变细 → 通算重叠度提高，但同步开销增加

---

## 5. 算子对比表

官网 `kernel/` 下有 2 个算子目录：

| 算子 | 通信方向 | 切分轴 | Blaze 集成 | 通信引擎 | `__global__` | CommContext | 关键特征 | 当前状态 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| all_to_all_quant_matmul (udma) | 通信→计算 (PUT) | K | 组合 | UDMA | 有 (×4，在 `tests/st/all_to_all_quant_matmul/src/kernel_launcher.h`) | apace CommContext | localMatmul 0/1/2，双通信对象（data + scale），CommPolicy 注入 | ✅ 可用（有 ST 覆盖） |
| all_to_all_quant_matmul (hcomm) | 通信→计算 (PUT) | K | 组合 | HCCL (CCU) | 无 | HCCL | CCU 变体（`HcclServerType::HCCL_SERVER_TYPE_CCU` + `GetHcclContext`，框架注册场景），复用同一 `QuantMatmulMxKernel`，仅 `CommPolicy` 不同 | ⚠️ 非直调 |
| all_gather_quant_matmul (udma) | 通信→计算 (PUT) | M | 组合 | UDMA | 有 (×1，`all_gather_mx_matmul_udma_impl.h` 文件末尾) | apace CommContext | 单 `__global__` 入口 `Process()`，FragmentTensor，dependId 预触发 | ✅ 可用（有 ST 覆盖） |

> 官网无 GET 算子（GET 钩子 `all_to_all_udma_get.h` 存在但无算子使用）、无 compute-first（计算在前）算子样例、无委托模式 epilogue。compute-first 算子的实现契约以 [`fusion.md`](fusion.md) §6.2 与 [`operator-anatomy.md`](../operator-design/operator-anatomy.md) §7 为准。
>
> 标记"有 `__global__`"的样例可直接通过 `<<<>>>` 直调；标记"无"的需要外部框架提供入口（直调限制见 §10 ④）。

---

## 6. 6 大关键技术索引

APACE 设计提出 6 大关键技术。以下为逐条索引：

| # | 技术 | 官网状态 | 深入参考 | PUT 样例是否使用 |
|:---|:---|:---|:---|:---|
| 1 | **多通信引擎+协议统一抽象**（四段式 API） | 已实现 | [`communication.md`](communication.md) | 是 |
| 2 | **FragmentTensor 离散内存重映射** | 已实现 | [`compute.md`](compute.md) | 否（all_to_all）；是（all_gather UDMA） |
| 3 | **通信计算独立调度** | 已实现 | [`operator-anatomy.md`](../operator-design/operator-anatomy.md) | 是 |
| 4 | **灵活切分策略** | 已实现 | [`operator-anatomy.md`](../operator-design/operator-anatomy.md) | 是 |
| 5 | **CV 协同/单核独立模式** | 已实现 | [`operator-anatomy.md`](../operator-design/operator-anatomy.md) | 是 |
| 6 | **内存编程基座**（Tensor API 地址排布分离+UB 静态规划） | 部分实现 | [`compute.md`](compute.md) | 是 |

> 官网 `docs/` 目录当前仅有 .gitkeep 占位，无设计文档。以 `kernel/` 下的实际文件为准。
>
> **UB 容量**（技术 #6 相关）：DAV_3510（Ascend 950PR/950DT）硬件 UB = 256KB 物理 / 248KB 框架可用（`GetCoreMemSize(UB)` 运行时获取，示例值 253952 = 256KB − 8KB 框架预留）。MC2 通算融合算子中 AIV 归约侧推荐预算 `TOTAL_UB = 192KB`，预留 headroom 给 guard 通信区与框架开销；实际可分配上限 `MAX_UB_BYTES = 180KB`（6-slot 归约布局下每元素 18B，`maxElements = MAX_UB_BYTES / perElemBytes`）。**禁止硬编码**——应以 `GetCoreMemSize(UB)` 运行时获取，上述数值仅为工程参考预算；UB 容量等芯片规格的权威口径以 npu-arch skill 为唯一知识源。详见 [`communication.md`](communication.md) §4.1 UB 预算、[`fusion.md`](fusion.md) §6.2.6 归约 UB 布局、[`scenarios/compute-first-reduce-scatter/development.md`](../scenarios/compute-first-reduce-scatter/development.md) §5.4。

---

## 7. AIC/AIV 分工与流水

### 核心原理

同一份 kernel 二进制同时跑在 AIV（Vector 核）和 AIC（Cube 核）上，靠 `ASCEND_IS_AIV` / `ASCEND_IS_AIC` 编译期分支隔离。跨核同步用 `CrossCoreSetFlag` / `CrossCoreWaitFlag`。

### PUT 模式流水（all_to_all_quant_matmul，对照官网）

PUT 逐 tile 流水编排（AIV Commit→Wait→SyncAll→SetFlag，AIC WaitTile 消费）以 [`fusion.md`](fusion.md) §3.2 编排图为唯一事实源，本节不重复。

> 若 `localMatmul == 1`，AIC 在首个 `WaitTile` 前先执行 `RunLocalMatmul()`（本地 A × 本 rank B → C），与 AIV PUT 并行。

### GET 模式流水（⚠️ 官网暂无算子样例，仅语义示意）

```
AIC (生产者)                    AIV (消费者)
─────────────                   ─────────────
tile 0: Matmul → C[0] 写 Win    │
          SetFlag<0x2,PIPE_FIX>(0) ──→  WaitFlag<0x2,PIPE_FIX>(0)（消费侧按生产管线取 FIX；管线选择属原理推导，官方样例仅 MTE3→MTE2 一对）
                                       Commit(GET C[0] from remote Win)
                                       Wait(waitLast)
                                       SetFlag<0x2,PIPE_MTE3>(0) ──→ 回压
tile 1: Matmul → C[1] 写 Win    │
          (WaitFlag 等回压)      │
          SetFlag<0x2,PIPE_FIX>(1) ──→  ...
...                             │
```

> GET 语义：AIC 先算 C 写 Win，AIV 从远端 Win 拉回本 rank 应得的 N 段；Win 区按 bufferCount 槽位环形复用，带回压。钩子契约见 [`communication.md`](communication.md) §3。

### 关键同步点（PUT）

| 同步点 | 方向 | Flag | 含义 |
|:---|:---|:---|:---|
| `CrossCoreSetFlag<0x2, PIPE_MTE3>(tid)` | AIV→AIC | tid | 通信 tile tid 就绪 |
| `CrossCoreWaitFlag<0x2, PIPE_MTE2>(tid)` | AIC 等 AIV | tid | AIC 消费前等待（经 `CommPolicy::WaitTile`，`waitedMask` 去重） |
| `SyncAll<true>()` | AIV 全体 | — | 每轮通信后全局同步，保证 flag 可见性 |

> 详见 [`fusion.md`](fusion.md)

---

## 8. 四段式通信 API 概览

apace 通信层采用四段式 API，由 CRTP 基类 `CollectiveCommBase`（`block/aiv_comm/collective_comm_base.h`）提供公共逻辑，派生类实现 4 个钩子：

```
Init()      → 公共逻辑：jobIndex→targetRank 映射、tile 偏移计算 → 调用 PostInit()
Commit()    → 公共逻辑：tile 迭代、遍历 targetRank → 调用 DoCommit(rank, tileBytes)
Wait()      → 公共逻辑：遍历 targetRank → 调用 DoWait(rank)；waitLast=true 时按早退条件 `currentTileIdx_ != totalTiles - 1` 跳过（官网事实，详见 communication.md §2）
Finalize()  → 调用 DoFinalize()
```

### 统一类型别名

```cpp
using Comm = Apace::AivComm::CollectiveComm<
    Apace::AivComm::CommCollectiveOp::AllToAll,
    Apace::AivComm::CommMode::PUT,
    AType,
    Apace::AivComm::TeamBarrier>;
```

编译期通过 `CollectiveCommHelper` 模板特化将 `(Op, Mode)` 组合映射到具体实现类，零运行期开销。

### GET 模式钩子（block 层已实现，⚠️ 官网暂无算子使用）

> GET/PUT 完整四钩子契约（`PostInit` / `DoCommit` / `DoWait` / `DoFinalize`，含 barrier 顺序、self 跳过规则、地址公式）详见 [`communication.md`](communication.md) §3。

---

## 9. 与 blaze-shmem 路线对照

支持两条路线开发 MC2 通算融合算子。以下为本质差异：

| 维度 | blaze-shmem 路线 | apace 路线 |
|:---|:---|:---|
| **工程组织** | 单算子独立 CMake 工程，自带 include/ 全套 | 框架化：只新建 `kernel/<op>/`，复用稳定共享层 `block/` `tiling/` `basic/` |
| **通信抽象** | 直接调 `aclshmemx_udma_*`，手写 AllToAllComm 类 | `CollectiveComm<Op,Mode,T,Barrier>` 编译期分发 + CRTP 四段式 API |
| **跨卡同步** | `aclshmemx_barrier_all_vec`（SHMEM barrier） | `TeamBarrier`（UBMEM 协议 GM flag 轮询，支持部分核参与） |
| **Blaze 集成** | 细粒度 tile 级手写编排 | 组合模式（自定义 matmul kernel + CommPolicy 注入） |
| **通信引擎** | 仅 UDMA（SHMEM） | UDMA(Hcomm)；HCCL windows/CCU 仅限注册场景（§10 ④） |
| **内存抽象** | 无（连续 GM + SHMEM Win 区） | FragmentTensor（离散内存虚拟重排，all_gather UDMA 已使用） |
| **tiling 结构** | 分层类继承（base/common/swat 三层） | host 侧同为分层类继承（`QuantMatmulTilingBase` → `QuantMatmulTilingSwat`）；差异在**下发契约**：kernel 侧消费扁平 struct 组合（`QuantMatmulTilingData` + `CommTilingData` + CommContext，POD 按值传参）而非注册形态的嵌套 tiling key 结构 |

### 何时用哪条路线

| 信号 | 推荐路线 |
|:---|:---|
| 用户提到"apace"、"APACE"、"通算融合框架" | apace |
| 代码在 `ops-transformer` 仓 `mc2/common/op_kernel/apace/` 下 | apace |
| 用户提到"SHMEM"、"shmem"、"aclshmem" | SHMEM |
| 代码自带独立 CMake 工程 + `aclshmem_*` API | SHMEM |
| 用户未明确，但需要快速原型验证 | SHMEM（基底工程自包含） |

### 关键约束差异

**blaze-shmem 路线**：禁止所有 HCCL API（`Hccl::*` 全部禁止）

**apace 路线**：禁止 HCCL 高阶 API（`Hccl::AllReduce` 等服务端调度 API），但**允许 HCCL windows/CCU**（`GetHcclContext`，属 kernel 级 API，通信下发权在 Kernel 内）⚠️ 仅限非直调场景

> ⚠️ HCCL windows/CCU 的"允许"仅限算子框架注册场景；直调不支持（详见 §11 ④）。

---

## 10. 基础约束（apace 路线红线）

本节 4 条为 apace 路线的**基础约束**（对应红线 R1/R2/R4/R7）；完整红线（全局 + 场景约束）以 [`review-checklist.md`](../review-checklist.md) 为准。设计文档中必须显式确认基础约束；代码审查时交叉检查；违反任意一条 = FAIL。

### ① 禁止使用 `__schedmode__(1)` 和 `[[bisheng::core_ratio(1,1)]]`

会导致 AIC/AIV 串行调度→死锁（`aclError:507015`）。核配比唯一由 `KERNEL_TYPE_MIX_AIC_1_1` 保证为 1:1。

```cpp
// 正确
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

// 错误 — 会导致死锁
__schedmode__(1)
[[bisheng::core_ratio(1,1)]]
```

> 详见 [`operator-anatomy.md`](../operator-design/operator-anatomy.md) §5
>
> 注：`aclError:507015` 是泛化 timeout/trap 错误码，多根因共用——localMatmul=1 缺 PipeBarrier 的 MTE 异常也表现为该码（见 [`fusion.md`](fusion.md) §5.4）。

### ② Matmul 走 Blaze 模板

计算侧统一用 Blaze（`BlockMmad` + `BlockScheduler` + `MatmulWithScaleMx`）。禁止使用 `AscendC::Matmul` 等 asc-devkit 黑盒 API——无法接入通算流水。

### ③ 禁止修改 `block/` 和 `tiling/`

这两个目录是稳定共享层。除非创建新通信原语（超出常规开发范围），否则只在 `kernel/<op>/` 下创建文件。

### ④ 直调开发仅支持 UDMA 模式（HCCL windows/CCU 不支持直调）

对于 CANNBot **Kernel 直调工作流**（`<<<>>>` 直调），apace 路线**仅支持 UDMA 模式**算子（官网即 `all_to_all_quant_matmul` UDMA 变体、`all_gather_quant_matmul`）。HCCL windows/CCU 模式算子**无 `__global__` 入口，不支持直调开发**。

**ReduceScatter 语义**：ReduceScatter 原语未在 block 层实现（`CommCollectiveOp::ReduceScatter` 枚举值已预留但无分发实现），其语义（多 rank 部分和累加 + 输出按轴切分）可通过 compute-first 场景模式实现——编排模式见 [`fusion.md`](fusion.md) §6.2，场景合同见 [`scenarios/`](../scenarios/) 场景注册表。

**原因**：HCCL 模式依赖算子框架在 kernel launch 前通过 `HcclCreateOpResCtx` 创建通信上下文（`HcclOpParam`），该上下文由框架注入 kernel。直调模式下缺少框架的上下文初始化，`GetHcclContext` 返回的指针无效，会导致通信失败或输出全为 0。

**判断方法**：查看算子对比表（§5），标注"无 `__global__`"的算子不支持直调。

**替代方案**：如需开发 HCCL windows/CCU 模式算子，走算子框架注册流程（op_host/op_kernel 完整工程），不使用 CANNBot 直调工作流。

### 与 blaze-shmem 路线约束的区别

| 约束 | blaze-shmem 路线 | apace 路线 |
|:---|:---|:---|
| 通信 API | 必须走 SHMEM，禁 HCCL | 禁 HCCL 高阶 API，**允许 HCCL windows/CCU**（⚠️ 直调仅 UDMA） |
| Matmul | 必须走 Blaze | 必须走 Blaze |
| 流程 | 按 CANNBot 7 步 | 按 CANNBot 7 步 |
| 调度属性 | — | **禁 `__schedmode__(1)`** |
| 共享层 | — | **禁改 `block/` `tiling/`** |
| 直调支持 | ✅ | **仅 UDMA 模式** |

---

## 11. 新建算子文件清单 + 导航

### 组合模式文件构成（3-4 个文件）

```
kernel/<op>/
├── <op>_impl.h            # 必需：Impl 类（UDMA 变体）
├── <op>_tiling_data.h     # 必需：tiling 结构体 + CommContext
├── <qbmm>_kernel.h        # 必需（组合模式）：自定义 matmul 调度（CommPolicy 注入）
└── <op>_hcomm_impl.h      # 可选：HCCL CCU 变体（仅框架注册场景）
```

### 官网算子实例

官网 `kernel/` 下两个算子目录：`kernel/all_to_all_quant_matmul/`（AllToAll PUT，含 UDMA / HCCL CCU 双变体）与 `kernel/all_gather_quant_matmul/`（AllGather PUT）。文件职责与符号名详见 [`development-guide.md`](../operator-design/development-guide.md) §1；共性模式解剖见 [`operator-anatomy.md`](../operator-design/operator-anatomy.md) §1。

### 后续阅读

| 文档 | 何时读 |
|:---|:---|
| **本文档**（`architecture.md`） | 第一次了解 apace 三层架构、组合模式、GET/PUT 方向 |
| [`compute.md`](compute.md) | 计算原理与接口：Blaze 组件、QuantMatmulMxKernel、FragmentTensor |
| [`communication.md`](communication.md) | 通信原理与接口：CollectiveComm 四段式、GET/PUT 钩子、TeamBarrier、CrossCore flag、host 建链 |
| [`fusion.md`](fusion.md) | 通算融合组合：GET/PUT 选型、flag 编排、环形回压、localMatmul 模式 |
| [`operator-anatomy.md`](../operator-design/operator-anatomy.md) | 算子解剖（kernel 侧）：共性模式、tiling_data、Impl 契约、入口规则 |
| [`host-and-testing.md`](../operator-design/host-and-testing.md) | 算子解剖（host 与测试）：初始化序列、launch、ST 工程 |
| [`development-guide.md`](../operator-design/development-guide.md) | 开发新算子：REUSE/MODIFY 标记、改造场景、验收清单 |
| [`workflow/`](../workflow/) | 四步流程：Step 1 项目搭建 / Step 2 框架调查 / Step 3 设计 / Step 4 实现 |
| [`profiling_mc2.md`](../../../shared/profiling_mc2.md) | 性能采集与调优时：msprof task-based 采集 + L2 flush + 多卡后处理 |
| [`pipeline_tuning.md`](../../../shared/pipeline_tuning.md) | 通算并行调优：tileCnt 两阶段策略 |
