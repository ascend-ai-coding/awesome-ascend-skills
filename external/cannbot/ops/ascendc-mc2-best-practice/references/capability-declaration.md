# MC2 能力声明（路线登记表）

> 本文件登记 MC2 通算融合算子的**已验证实现路线**。每行 = 决策栈上一条一致路线：L0（chip × op_type × 调用形态）→ L1 通信路径 → L2 编程抽象。
> Architect 在需求分析阶段查询本表：确认路线可用性 → 路由到知识目录 → 定位参考实现。本表也登记**明确不支持**的组合（否定行），让否决有据可查。

## 决策栈速览

| 层 | 回答的问题 | 本表对应列 |
|----|-----------|-----------|
| L0 需求层 | 做什么、在哪跑、怎么被调用 | `chip` / `op_type` / `调用形态` |
| L1 通信路径层 | 数据怎么跨卡搬（引擎 + 协议） | `通信路径` |
| L2 编程抽象层 | 用什么写（技术底座：通信库+模板组装 / 模板库 / 裸 API） | `编程抽象` |
| L3 工程组织层 | 代码怎么摆 | 由 L2 决定，见对应知识目录 |
| L4 流水编排层 | 通信与计算怎么重叠 | 见对应知识目录文档 |

> 关键关系：L1 与 L2 是**多对多**——UDMA 上可用 blaze-shmem 或 apace 底座；apace 底座可跑 UDMA（直调）或 HCCL windows（注册）。因此路线必须整行登记，不能由轴自由组合推导。

## 已验证 / 规划路线

| chip | op_type | 调用形态 | 通信路径 | 编程抽象 | status | reference_impl | 知识目录 |
|------|---------|---------|---------|---------|--------|----------------|---------|
| dav-3510 | collective-comm | 直调 | AIV+URMA | blaze-shmem | supported | `references/foundations/blaze-shmem/all_to_all_matmul/` | `references/foundations/blaze-shmem/` |
| dav-3510 | collective-comm | 直调 | AIV+URMA | apace | supported | ops-transformer `mc2/common/op_kernel/apace/kernel/all_to_all_quant_matmul/`、`.../all_gather_quant_matmul/`（通信在前 PUT）；compute-first ReduceScatter 语义参考 `mc2/matmul_reduce_scatter/`、`mc2/quant_reduce_scatter/`（⚠️ 均为注册形态实现、未用 apace，仅作语义/golden 参考） | `references/foundations/apace/` |
| dav-3510 | moe | 直调 | MTE通信（AIV+UBMEM） | ascendc-api | supported | `references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/` | `references/foundations/ascendc-api/moe-dispatch-combine/` |
| dav-2201 | moe | 直调 | MTE通信（AIV+UBMEM） | ascendc-api | supported | 同上（compat 层抹平 dav-2201/dav-3510 window 结构差异） | 同上 |
| dav-2201 | collective-comm | 注册 | HCCL 高阶 | hccl-matmul | supported | `references/foundations/hccl-matmul/all_gather_matmul/` | `references/foundations/hccl-matmul/` |
| dav-3510 | moe | 直调 | AIV+URMA | apace | planned | ops-transformer `mc2/mega_moe/` | — |

> 注：MTE通信路径下 HCCL 仅用于 host 侧 window 资源分配（`HcclAllocComResourceByTiling`），不涉及 `Hccl::*` 高阶集合通信 API。UDMA 即 URMA 协议的同义称呼。

## 否定行（明确不支持）

| 组合 | status | 原因 |
|------|--------|------|
| 直调 × HCCL 高阶集合通信（`Hccl::*`）× AscendC::Matmul 高阶 API | unsupported | HCCL 集合通信库（Ascend C 高阶 API 一部分，基于 HCOMM 构建）依赖框架注入的通信上下文，Kernel 直调拿不到；官方通算融合仅支持 acnn 单算子调用（详见 [`foundations/blaze-shmem/comm_shmem.md`](foundations/blaze-shmem/comm_shmem.md) §5）。910B 集合通信+Matmul 应改走本表 `dav-2201 × collective-comm × 注册 × HCCL 高阶 × hccl-matmul` 行（详见 [`foundations/hccl-matmul/`](foundations/hccl-matmul/)） |
| 直调 × HCCL windows（`GetHcclContext`） | unsupported | 无 `__global__` 入口，不支持 Kernel 直调；仅限注册场景（apace 非直调变体，见 [`foundations/apace/fundamentals/architecture.md`](foundations/apace/fundamentals/architecture.md) §10 ④） |
| 直调 × CCU | unsupported | 同上，无 `__global__` 入口 |
| dav-2201 × UDMA（blaze-shmem 或 apace） | unsupported | UDMA 路径仅 dav-3510 已验证，其他架构行为未验证，禁止使用 |

未出现在任何行中的组合一律视为 unsupported（如 moe × blaze-shmem：无参考实现）。

## 字段说明

| 字段 | 含义 | 取值 |
|------|------|------|
| `chip` | 芯片编译宏架构 | `dav-3510`（Ascend 950PR/950DT）、`dav-2201`（Atlas A2/A3 系列）等 |
| `op_type` | 算子类型 | `collective-comm`（集合通信类）、`moe`（MOE 类），分类速查见 [`requirement-analysis/classification.md`](requirement-analysis/classification.md) |
| `调用形态` | 工程调用方式 | `直调`（`<<<>>>`）、`注册`（op_host + op_kernel 完整工程） |
| `通信路径` | 跨卡数据搬运方式（通信引擎 + 协议的组合） | `AIV+URMA`、`MTE通信`（AIV+UBMEM）、`CCU`（CCU+URMA，尚无直调参考工程）、`HCCL 高阶`，选项详见 [`requirement-analysis/comm-path-decision.md`](requirement-analysis/comm-path-decision.md) |
| `编程抽象` | 代码编写所基于的技术底座，值即 `references/foundations/` 下的底座目录名 | `blaze-shmem`（SHMEM 通信库 + Blaze 计算模板，手工组装）、`apace`（APACE 模板库，通信+计算+工程组织全包）、`ascendc-api`（裸 Ascend C API 全自建，MTE通信场景含 compat 层）、`hccl-matmul`（HCCL 高阶 + `AscendC::Matmul` 高阶，仅注册） |
| `status` | 可用性状态 | `supported`（有参考实现可开发）、`planned`（已规划无实现）、`unsupported`（不可用） |
| `reference_impl` | 参考实现路径 | skill 内相对路径或 ops-transformer 仓路径 |
| `知识目录` | 该路线的领域知识所在目录 | `references/foundations/` 下的目录（底座目录或其下路线目录），Architect 据此路由 |

## 查询规则

1. 按 L0 三列（`chip` / `op_type` / `调用形态`）过滤候选行
2. `supported` 优先于 `planned`；`planned` → 告知用户该组合尚无参考实现、风险自担；仅命中否定行或无行命中 → 给出确定答复（不可用 + 原因 + 替代建议）
3. 多行命中时按用户信号选编程抽象（见下表）
4. 按 `知识目录` 列路由到对应 references 目录；按 `reference_impl` 定位起手工程

### 编程抽象选择信号（多行命中时）

| 信号 | 编程抽象 |
|------|---------|
| 用户提到"apace"、"APACE"、"通算融合框架"、"CollectiveComm"、"CRTP"、"四段式 API" | apace |
| 代码在 `ops-transformer` 仓 `mc2/common/op_kernel/apace/` 下 | apace |
| 用户提到"SHMEM"、"shmem"、"aclshmem" | blaze-shmem |
| 代码自带独立 CMake 工程 + `aclshmem_*` API | blaze-shmem |
| 用户未明确，但需要快速原型验证 | blaze-shmem（基底工程自包含） |
| 用户提到"910B"、"HCCL 高阶"、"aclnn 通算融合" | hccl-matmul |
| 代码含 `Hccl<HCCL_SERVER_TYPE_AICPU>` + `AscendC::Matmul` / `aclnn*Matmul` | hccl-matmul |
| Brownfield 模式从代码推断 | 见 [`codebase-analysis.md`](codebase-analysis.md) |

> 例：moe × dav-3510 × 直调 命中两行——`supported`（MTE通信）优先，仅当用户明确要求 apace/AIV+URMA 路线时选 `planned` 行并告知风险。

## 扩展操作

| 扩展场景 | 操作 | 影响范围 |
|---------|------|---------|
| 新芯片验证通过 | 加/改行（`chip` 列） | 仅本表 |
| 新通信路径（如 CCU 直调落地） | 路径列加取值 + 加行 | 本表 + comm-path-decision.md |
| 新编程抽象（新技术底座） | 抽象列加取值 + 加行 + 新建 `references/foundations/{新底座}/` | 本表 + 一个知识目录 |
| 新算子类型 | `op_type` 加取值 + 加行（跨通信路径设计模式放 `references/operators/{type}.md`） | 本表 + operators/ |
| 注册形态路线验证（如 apace × HCCL windows × 注册） | 加行（调用形态=注册） | 仅本表 |

## 需求分析阶段相关文档

| 文档 | 用途 |
|------|------|
| [`requirement-analysis/template.md`](requirement-analysis/template.md) | REQUIREMENTS.md 模板（MC2 特有章节） |
| [`requirement-analysis/comm-path-decision.md`](requirement-analysis/comm-path-decision.md) | 通信路径选项（决策栈 L1 层） |
| [`requirement-analysis/classification.md`](requirement-analysis/classification.md) | MC2 算子分类速查表和可信源清单 |
| [`requirement-analysis/grill-protocol.md`](requirement-analysis/grill-protocol.md) | 需求拷问 8 维度判据 |
| [`requirement-analysis/quality-checklist.md`](requirement-analysis/quality-checklist.md) | 需求文档自检 14 项清单 + 开发就绪闸门 |
| [`operators/collective-comm.md`](operators/collective-comm.md) | 集合通信类算子跨编程抽象共性设计模式 |
| [`codebase-analysis.md`](codebase-analysis.md) | Brownfield：从代码推断路线坐标（Stub） |
