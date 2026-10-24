---
name: external-cannbot-ops-ascendc-mc2-best-practice
description: Ascend C MC2 通算融合算子（多卡通信+计算融合）开发最佳实践。当用户需要设计、实现、调试、移植或优化通算融合算子，或提及"MC2"、"SHMEM"、"APACE"、"通算融合"、"多卡通信直调"、"UDMA"、"URMA"、"AllToAll+Matmul"、"CollectiveComm"、"MoE
  Dispatch"、"MoE Combine"、"专家并行"、"EP"、"mega_moe"、"MTE通信"、"MTE window"、"token 路由分发"、"HCCL"、"aclnn
  通算融合"时使用。
original-name: ascendc-mc2-best-practice
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-24'
synced-commit: ec8a7b0462e3acdd0d1d2f857f1dae652e19f4f2
license: UNKNOWN
---

# Ascend C MC2 通算融合算子开发最佳实践

MC2（Matrix Computation & Communication）= 多卡间集合通信 + 单卡内 Blaze Cube 计算 + 通算两层流水掩盖通信开销。提供 MC2 通算融合算子开发的技术参考：通信与计算的 API 选择、时序约束、验收标准。

**知识分层（依赖方向单向，禁止反向引用）**：

| 层 | 技能 | 事实源范围 | 本 skill 的使用方式 |
|----|------|-----------|-------------------|
| 底层（通用 API） | `ascendc-api-best-practices` | Ascend C API 签名、硬件限制、平台差异（DataCopy/Cast/HardEvent/CrossCore flag/Hcomm/HCCL host 等） | 一律引用其 references，本 skill **不复述** API 细节 |
| 底层（Blaze 模板） | `ascendc-blaze-best-practice` | Blaze/tensor_api 通用模板选型、Tiling 算法选择 | 同上 |
| **本层（领域知识）** | 本 skill | MC2 特有：路线决策、apace/blaze-shmem/MoE 框架契约、通算流水编排、质量门禁 | — |

> 下层技能不引用本 skill；本 skill 文档中凡 API 级事实（签名、限制、硬件行为）均指向下层锚点，正文只保留 MC2 场景应用与编排规则。

本 skill 覆盖四种编码底座，每个底座支持特定的通信路径、算子类型与芯片组合（确切组合以 [`capability-declaration.md`](references/capability-declaration.md) 路径登记表为准）：

| 底座 | 支持的通信路径 | 支持的算子类型 | 支持芯片 | 知识目录 |
|------|---------|---------|------|------|
| **blaze-shmem** | AIV+URMA | collective-comm（AllToAll+Matmul、AllReduce+Matmul、TP/SP 融合） | dav-3510（Ascend 950PR/950DT） | [`references/foundations/blaze-shmem/`](references/foundations/blaze-shmem/) |
| **apace** | AIV+URMA | collective-comm（AllToAll/AllGather + QuantMatmul 融合、compute-first 类；AllReduce 类需先扩展通信组件，见 [`paradigm-mapping.md`](references/foundations/apace/operator-design/paradigm-mapping.md)） | dav-3510（Ascend 950PR/950DT） | [`references/foundations/apace/`](references/foundations/apace/) |
| **ascendc-api** | AIV+UBMEM | moe（MoE Dispatch/Combine、专家并行 EP） | dav-3510（Ascend 950PR/950DT）+ dav-2201（Atlas A2/A3 系列）双平台 | [`references/foundations/ascendc-api/`](references/foundations/ascendc-api/) |
| **hccl-matmul** | HCCL 高阶 | collective-comm（AllGather+Matmul、AllReduce+Matmul 等，aclnn 注册） | dav-2201（Atlas A2/A3 系列） | [`references/foundations/hccl-matmul/`](references/foundations/hccl-matmul/) |

> 四种底座抽象层级不同（blaze-shmem 是库+模板手工组装，apace 是模板框架，ascendc-api 是直接使用ascendc API，hccl-matmul 是官方 HCCL 高阶 + `AscendC::Matmul` 高阶、仅注册），但在"选什么写代码"这个决策点上是并列选项。apace 模板库虽基于 Ascend C 基础 API 构建，但与 ascendc-api / hccl-matmul 路线约束集、工程边界、可修改范围完全不同——详见 §1/§2/§3/§4 各路线特有约束。

## 何时使用 / 不适用

**使用信号**（任一即可）：
- 场景：多卡协同的通算融合算子（AllToAll+Matmul、AllReduce+Matmul、MoE Dispatch/Combine、多卡 EP/TP/SP 融合 Kernel）
- 关键词："MC2"、"SHMEM"、"APACE"、"apace"、"通算融合"、"多卡通信直调"、"UDMA"、"URMA"、"AllToAll+Matmul"、"CollectiveComm"、"MoE Dispatch"、"MoE Combine"、"专家并行"、"EP"、"mega_moe"、"MTE通信"、"MTE window"、"token 路由分发"、"910B"、"HCCL"、"aclnn 通算融合"
- 代码：现有工程同时出现通信 API（`shmem.h`/`aclshmem*` 或 `collective_comm_api.h`/`CollectiveComm`）与 `blaze/gemm/` 模板（blaze-shmem/apace 路线），或出现 `winContext`/`mc2Context`/`HcclOpParam` 与 MTE 通信窗口结构（MTE通信 → ascendc-api 路线），或出现 `Hccl<HCCL_SERVER_TYPE_AICPU>` 与 `AscendC::Matmul`（HCCL 高阶 → hccl-matmul 路线）

**不适用**（走其他 skill）：
- 纯单卡 Matmul（无跨卡通信）→ `ascendc-blaze-best-practice`（950）或 `ascendc-api-best-practices` 的 `api-matmul.md`（910B）
- Vector 类逐元素/归约算子（无 Cube、无跨卡通信）
- 通用 Ascend C API 用法查询 → `ascendc-api-best-practices`
- 非 3510 架构的 AIV+URMA 通算融合（MTE通信除外，支持 dav-2201 + dav-3510 双平台）

> 确切的支持组合（chip × 算子类型 × 调用形态 × 通信路径 × 编程抽象）以 [`references/capability-declaration.md`](references/capability-declaration.md) 路线登记表为准；表内同时登记**明确不支持**的组合及原因，命中否定行时直接答复用户不可用 + 替代建议。

## 决策树：先判断算子类型

```
用户要做通算融合算子
│
├─ 目标芯片 910B / dav-2201 / A2，且是集合通信+Matmul（AllGather/AllReduce/ReduceScatter/AlltoAll）？
│   → HCCL 高阶路径（仅注册 / aclnn，禁止 <<<>>> 直调）→ hccl-matmul 路线（下方§4）
│
├─ AllToAll+Matmul / AllReduce+Matmul / TP/SP 通算融合？（950）
│   → AIV+URMA 路径（仅 dav-3510 / Ascend 950）
│      ├─ 提到"apace"/"CollectiveComm"或代码在 ops-transformer/apace/ 下？→ apace 路线（下方§2）
│      └─ 否则 → blaze-shmem 路线（下方§1）
│
├─ MoE Dispatch / MoE Combine / mega_moe / 专家并行 EP？
│   → MTE通信 → ascendc-api 路线（dav-2201 + dav-3510 双平台，下方§3）
│      ├─ 先阅读已有实现？→ references/foundations/ascendc-api/moe-dispatch-combine/reading/
│      └─ 生成新算子或改造？→ references/foundations/ascendc-api/moe-dispatch-combine/samples/
│
└─ 不确定？
    → 看通信路径：SHMEM API = AIV+URMA 路径（仅 dav-3510）；winContext/mc2Context = MTE通信（dav-2201 + dav-3510 双平台）；
       Hccl<HCCL_SERVER_TYPE_AICPU> / InitV2 / aclnn*Matmul = hccl-matmul（dav-2201 注册）
```

选定路线后：

1. 查 [`capability-declaration.md`](references/capability-declaration.md)，确认 chip × 算子类型 × 调用形态组合命中 `supported` 行（命中否定行或无行 → 答复用户不可用 + 原因）
2. 进入需求分析（下节），输出 REQUIREMENTS.md
3. 按 §1/§2/§3/§4 进入对应路线的 references

## 需求分析（全路线共享）

> **MC2 特有需求维度**：除算子名/数学定义/dtype/shape 外，需求收集时还需明确通信路径（AIV+URMA / MTE通信 / HCCL 高阶）与编程抽象底座（blaze-shmem / apace / ascendc-api / hccl-matmul）。

| 文档 | 何时读 |
|------|--------|
| [`capability-declaration.md`](references/capability-declaration.md) | 确认路线可行性、多行命中时选编程抽象、定位参考实现 |
| [`requirement-analysis/grill-protocol.md`](references/requirement-analysis/grill-protocol.md) | 需求拷问 8 维度判据（生成 REQUIREMENTS.md 前逐项拷问，先拷问再生成） |
| [`requirement-analysis/template.md`](references/requirement-analysis/template.md) | 生成需求文档时（含 MC2 特有章节：组网规模、通信设计、开发就绪判断） |
| [`requirement-analysis/comm-path-decision.md`](references/requirement-analysis/comm-path-decision.md) | 用户询问通信路径选型时（HCCL 高阶 / AIV+URMA / MTE通信 / CCU） |
| [`requirement-analysis/classification.md`](references/requirement-analysis/classification.md) | 确定算子类型、查已有算子与可信源清单 |
| [`requirement-analysis/quality-checklist.md`](references/requirement-analysis/quality-checklist.md) | 需求文档生成后自检（14 项 + 开发就绪闸门） |
| [`operators/collective-comm.md`](references/operators/collective-comm.md) | 设计集合通信类算子时（通信语义、轴切分、通算流水的跨路线共性） |
| [`codebase-analysis.md`](references/codebase-analysis.md) | Brownfield 从代码推断路线坐标（Stub；Greenfield 不读） |

## 红线（全路线共性，与底座无关）

| # | 红线 | 一句话理由 | 详见 |
|---|------|-----------|------|
| ① | 架构白名单以路线登记表为准 | 未验证组合（如 dav-2201 × AIV+URMA）禁止使用 | [`capability-declaration.md`](references/capability-declaration.md) |
| ② | AIV+URMA 路径性能采集必须刷 L2 cache | SHMEM B 矩阵驻留会污染下一轮带宽；**hccl-matmul（AICPU）无需 flush** | [`shared/profiling_mc2.md`](references/shared/profiling_mc2.md) |

> 以下两条是 **AIV+URMA 路径下 blaze-shmem / apace 底座的选择性约束**，不是全路线共性。集合通信类通算融合在 **注册 / aclnn** 形态下走 §4 hccl-matmul 路线，此时 HCCL 高阶 API 与 `AscendC::Matmul` 高阶 API 恰是必选路径：
> - 禁止 HCCL 高阶 API（`Hccl::*`）—— HCCL 集合通信库依赖框架注入上下文，AIV+URMA 直调场景拿不到（详见 [`blaze-shmem/comm_shmem.md`](references/foundations/blaze-shmem/comm_shmem.md) §5，7 类 18 个 API 清单）
> - Matmul 走 Blaze 模板，禁止 `AscendC::Matmul` 高阶 API —— 同理（详见 [`blaze-shmem/matmul_blaze.md`](references/foundations/blaze-shmem/matmul_blaze.md) / [`apace/fundamentals/compute.md`](references/foundations/apace/fundamentals/compute.md)）

各底座特有约束与逐项审查清单见 §1/§2/§3/§4。

---

## §1 blaze-shmem 路线

SHMEM 通信库（cann/shmem，与 HCOMM 无关）驱动 AIV+URMA 跨卡搬运，Blaze 计算模板负责 Cube 计算，二者手工组装为独立 CMake 工程（自带 `include/` 全套）。通信侧 host `aclshmem_*`、device `aclshmemx_udma_*` + `aclshmem_barrier_all`；计算侧 `Blaze::Gemm::Block::BlockMmad` + `Blaze::Gemm::Kernel::*`。

### 红线 / 约束

| # | 约束 | 说明 | 详见 |
|---|------|------|------|
| R1 | 架构白名单 | CMakeLists.txt 中 `npu-arch` = dav-3510 | 全路线共性红线 ① |
| R2 | 禁止 HCCL 高阶 API（`Hccl::*`） | HCCL 集合通信库依赖框架注入上下文，AIV+URMA 直调场景拿不到 | [`comm_shmem.md`](references/foundations/blaze-shmem/comm_shmem.md) §5（7 类 18 个 API 清单） |
| R3 | 禁止 `AscendC::Matmul` 高阶 API | 高阶 API 不支持 AIV+URMA 直调场景 | [`matmul_blaze.md`](references/foundations/blaze-shmem/matmul_blaze.md) |
| R4 | 通信走 SHMEM | 头文件含 `shmem.h`，device 侧用 `aclshmemx_udma_*`/`aclshmem_barrier_all` | [`comm_shmem.md`](references/foundations/blaze-shmem/comm_shmem.md) |
| R5 | Matmul 走 Blaze | 头文件含 `blaze/gemm/block/block_mmad*.h` | [`matmul_blaze.md`](references/foundations/blaze-shmem/matmul_blaze.md) |
| R6 | L2 flush 证据 | 代码中含 L2 flush kernel 调用或等效实现 | 全路线共性红线 ② |
| R7 | 流程门禁完整 | `docs/` 下 DESIGN/PLAN/WALKTHROUGH/REVIEW.md 齐全；环境检查通过 | — |

**逐项审查清单**：[`review-checklist.md`](references/foundations/blaze-shmem/review-checklist.md)（含常见 FAIL 原因与修复方向，违反红线项 = FAIL）。

### References

| 文档 | 何时读 |
|------|--------|
| [`references/foundations/blaze-shmem/workflow_integration.md`](references/foundations/blaze-shmem/workflow_integration.md) | 设计 MC2 算子前，看 MC2 场景的技术要点和门禁 |
| [`references/foundations/blaze-shmem/mc2_architecture.md`](references/foundations/blaze-shmem/mc2_architecture.md) | 第一次设计 MC2 算子，建立 AIV/AIC 分工 + 4-buffer 流水 + M 轴切分心智模型 |
| [`references/foundations/blaze-shmem/comm_shmem.md`](references/foundations/blaze-shmem/comm_shmem.md) | 写/改通信层，查 SHMEM API、UDMA 用法、禁止 HCCL 清单（§5）、扩展其他通信原语 |
| [`references/foundations/blaze-shmem/matmul_blaze.md`](references/foundations/blaze-shmem/matmul_blaze.md) | 写/改计算层，选 Blaze 模板、改 DispatchPolicy、处理 Scale |
| [`references/shared/profiling_mc2.md`](references/shared/profiling_mc2.md) | 性能采集与调优时：msprof task-based 采集 + L2 flush + 多卡数据后处理 |
| [`references/shared/pipeline_tuning.md`](references/shared/pipeline_tuning.md) | 精度调试阶段用 tileCnt=1 做串行基线；性能调优阶段扫描 tileCnt 找最优值 |
| [`references/foundations/blaze-shmem/codebase_map.md`](references/foundations/blaze-shmem/codebase_map.md) | 工程搭建时，定位"哪些文件改/不改"与改造食谱 |
| `references/foundations/blaze-shmem/all_to_all_matmul/` | 编译验证过的基底工程，blaze-shmem 路线所有 MC2 算子的起手模板 |

---

## §2 apace 路线

APACE（Ascend PArallel Communication-compute Engine）是通算融合算子的架构底座，提供可复用的 block 层接口、kernel 层参考实现和 tiling 算法。APACE 自建通信基础 API（基于 HCOMM 通信基础库构建，与 HCCL 集合通信库无关）。apace 路线在 `kernel/<op>/` 下新建算子，复用稳定共享层 `block/` `tiling/`。

> **apace 路线接口契约基准**：[cann/ops-transformer](https://gitcode.com/cann/ops-transformer) `mc2/common/op_kernel/apace/` 子树，以 pin 的已验证快照为结构基准。样例代码经 scripts/fetch_apace.sh 现取现读（支持 `--ref`/`APACE_PIN_REF` 锚定 commit）；⚠️ master 结构会演进（如目录迁移、新增算子），拉取 master 后必须与快照 diff 校验，文档引用失效时更新文档——**已知漂移登记（pin 之后 10 个提交：block/→core/ 迁移、ReduceScatter UBMEM 官方模板、hcomm dcci 等）见 [`references/foundations/apace/fundamentals/communication.md`](references/foundations/apace/fundamentals/communication.md) §8，核对 master 前必读**。
>
> **工程实现默认事实源**：CANN 内置 apace 框架（路径随 CANN 打包形态实测定位，Step 1 登记；两种已验证形态：`opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/apace/`（cann-9.2.0）、`vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/apace/`（cann-9.1.0）），**直调独立工程默认 CMake 直引该路径，禁止整包复制**。

### 按请求目的路由

| 请求目的 | 执行 |
|---|---|
| 要求完整开发一个 MC2 通算融合算子 | Step 1 → Step 2 → Step 3 → Step 4（plugin 场景以 [`workflow_integration.md`](references/foundations/apace/workflow_integration.md) 的 7 步映射为主要消费文档） |
| 只要求设计/方案分析并输出设计文档 | Step 2 → Step 3 |
| 咨询、解释、评审、排障、能力查询 | 只读相关 references |

### 计算执行原则

通信与计算的流水编排、AIV-AIC 协同必须在 device 侧单次 launch 的融合 Kernel（MIX）中完成。host 侧只负责：数据准备与 buffer 分配、Tiling 计算、多 rank 启动与通信建链（rootInfo 交换、HCCL/Win 资源分配、跨 rank barrier）、Kernel launch（含 dtype dispatch）和结果搬运。不得将算子语义中的任何计算或通信步骤（如归约、数据搬移、状态同步）放到 host 侧执行；CPU golden 等精度验证基建不属于算子语义，不在此限。

### 方法论速览（五阶段流程 + 六个决策点）

任何 apace 算子开发遵循：**语义分析 → 路线决策 → 设计合同 → 实现组装 → 验证** 五阶段；设计合同必须对六个决策点逐项定值：

| 决策点 | 问题 | 一句话判据 |
|---|---|---|
| D1 数据流方向 | 通信在前还是计算在前？ | 通信的输入是原始输入（前）还是计算产物（后） |
| D2 切分轴与轮次 T | 沿哪个轴切、切几轮？ | 轴由数据分布语义定（非通信原语）；T 权衡同步开销 vs 流水粒度 |
| D3 数据通路 | 连续 GM 还是 FragmentTensor？ | 通信后数据在 GM 上连续还是离散 |
| D4 同步合同 | 谁通知谁、几次、在哪等？ | flag idx/次数严格配对；barrier totalJobs 按分核映射；SyncAll 在守卫外 |
| D5 累加合同 | 部分和在哪累加？ | L0C 单次 fixpipe / AtomicAdd / DEFERRED_SYNC / staging+增量归约四选一 |
| D6 入口合同 | 几个入口、怎么分发？ | 入口数 = dtype 组合数；MIX_AIC_1_1；tiling 按值 ABI |

完整方法论（每阶段的产出/门禁/回退点、决策点取值示例、跨阶段纪律）：[`operator-design/dev-methodology.md`](references/foundations/apace/operator-design/dev-methodology.md)。新算子先在该文档 §5 找 D1-D6 取值最接近的实现作骨架参考。

### 四步流程

1. **Step 1: Project Setup** — 建立算子工程骨架、校验 CANN 内置 apace 事实源与环境（dav-3510、多卡、HCCL）、登记只读参考点。不核对接口事实、不判定路线、不创建实现文件。→ [`workflow/step1-project-setup.md`](references/foundations/apace/workflow/step1-project-setup.md)
2. **Step 2: 设计前核对（轻量）** — 只读核对 apace 框架的接口事实（matmul 链路/通信接口/入口 ABI/官方覆盖性）；事实**默认内联记录于 DESIGN.md §0.3**（无需独立调查报告）。不做方案推荐、不匹配场景、不选择路线。→ [`workflow/step2-investigation.md`](references/foundations/apace/workflow/step2-investigation.md)
3. **Step 3: Design** — 依据需求和调查事实完成路线决策，生成定稿的 DESIGN 和可执行路线的 PLAN。只产出设计文档，不复制文件、不编写实现、不执行构建。→ [`workflow/step3-design.md`](references/foundations/apace/workflow/step3-design.md)
4. **Step 4: Implementation** — 按定稿的 DESIGN/PLAN 完成实现、多卡验证与性能采集；实现层问题在项目内修复，不改设计、不改接口、不扩大支持域。→ [`workflow/step4-implementation.md`](references/foundations/apace/workflow/step4-implementation.md)

> **plugin 消费流映射**：ops-direct-invoke plugin 执行 7 步流程（环境检查→设计→串讲→开发→审查→修复→验收→汇报），与本四步模型的对应为：plugin Step 1 ≈ 本 Step 1；plugin Step 2/2.5 ≈ 本 Step 2+3；plugin Step 3-7 ≈ 本 Step 4。**plugin 场景下以 [`workflow_integration.md`](references/foundations/apace/workflow_integration.md)（7 步映射）为主要消费文档**，四步文档为模型参考。

### 路线模型

```text
implementation_route: apace_native | apace_custom | unsupported
selected_scenario: <仅 apace_custom 填写>
```

按官方 README 的两种使用方式决策：

- **`apace_native`**：官方 `apace/kernel` 已有算子可直接调用或参考复用——不读取场景注册表；
- **`apace_custom`**：官方 kernel 未覆盖，但可基于 `apace/block` 接口组合构建（含 compute-first 等自研编排）——需求语义命中场景注册表语义判据时**默认查阅**对应场景指导（准入条件只含需求侧语义判据，不含实现侧决策）；
- **`unsupported`**：`apace/block` 接口层也无法支撑的组合，或场景语义零命中/多命中。

> **场景文档查阅不受官方覆盖性判定反向阻断**：本仓/用户工程中已存在某场景的生产实现时（如 compute-first 类算子已生成过），阅读该场景文档不需要"官方未覆盖"前提——场景文档同时是该类算子的改造/复用手册。

官方接口未覆盖不是绕过 apace 接口层、退回裸 Ascend C 全自建的授权——那是 ascendc-api 底座的路线选择，不是 apace 路线内的 fallback。

> **compute-first（ReduceScatter 类语义）算子生成阅读顺序**：[`fusion.md`](references/foundations/apace/fundamentals/fusion.md) §6.2（方法论：编排模式/事件语义/tiling 组织）→ [`scenarios/compute-first-reduce-scatter/design.md`](references/foundations/apace/scenarios/compute-first-reduce-scatter/design.md)（设计合同）→ [`scenarios/compute-first-reduce-scatter/development.md`](references/foundations/apace/scenarios/compute-first-reduce-scatter/development.md)（实现模板与验收）。方法论原则（事实源唯一/设计三拷问/契约驱动/失败前置/验收分层/示例边界）见 [`architecture.md`](references/foundations/apace/fundamentals/architecture.md) §1.5。

### 红线 / 约束（全局 R 系列 + 场景约束）

> 设计期与开发期必须逐条满足；完整定义、操作化检查方法与常见 FAIL 原因以 [`review-checklist.md`](references/foundations/apace/review-checklist.md) 为准。违反任意适用项 = FAIL。

**全局红线（所有 apace 算子必须满足）**：

| # | 约束 | 一句话理由 | 详见 |
|---|------|-----------|------|
| R1 | 禁止 `__schedmode__(1)` / `core_ratio(1,1)` | AIC/AIV 串行调度 → 死锁（507015）；核配比由 `KERNEL_TYPE_MIX_AIC_1_1` 保证 | architecture §10 ① |
| R2 | 每个入口含 `KERNEL_TYPE_MIX_AIC_1_1` | 核配比 1:1 唯一保证 | architecture §10 ① |
| R3 | 入口变体覆盖 dtype 合同全部组合 + host 运行期 dispatch | 硬编码单入口 → 异 dtype 字节流被错误模板解释，精度系统性失败 | operator-anatomy §5.3/§7.2 |
| R4 | `block/` `tiling/` 零修改 | 共享层漂移 = 幽灵 bug 温床 | architecture §10 ③ |
| R5 | CrossCore flag idx 配对 | AIV WaitFlag idx == AIC SetFlag idx | fusion §3 |
| R6 | CommContext 与引擎匹配 | UDMA 有 `__gm__ CommContext*`；HCCL windows 无 | communication |
| R7 | 禁止 `AscendC::Matmul` 高阶 API | 高阶 API 不支持直调场景 | compute |
| R8 | 禁止 `Hccl::*` 高阶 API | 依赖框架注入上下文，直调拿不到 | comm_shmem §5 |
| R11 | host 前置校验在 fork/建链前拒绝非法输入 | 非法输入进 kernel = 难查的死锁/精度问题（整除/对齐/核数/Win 容量等；compute-first 完整 9 项见场景文档） | development-guide §3.5 |
| R12 | UB 静态通信区物理隔离 | 混用重叠 → 通信数据被踩踏 → 死锁；**`TPipe` 与 `MakeMemPtr` 必须二选一，禁止混用** → 507015 | communication 陷阱 #9；operator-anatomy §4.3 |
| R13 | 通信对象 `totalJobs=rankSize` 多核并行 | 退化 totalJobs=1 → 通信时间放大 R 倍（已证伪臆造约束）；TeamBarrier totalJobs 按分核映射取值（官方前 R 核 = rankSize；compute-first 后 R 核 = 1 + 显式 CrossDevice） | communication §2.2 |
| R14 | Win 数据/元数据分离（硬红线）+ 单轮 PUT 大小（风险提示） | 覆盖元数据 → "假通过"；单轮 PUT 无官方硬上限（512KB 系 bring-up 经验，同平台生产实现 12.6MB 稳定），大单轮按 case 复核 | communication 陷阱 #12/#13 |
| R15 | 投产级性能验证门槛 | 真实大 shape × R=2/4 双档 × 三路径对标；仅基线 = 未达标 | host-and-testing |
| R20 | perf 模式 L2 flush 实接线 | 只分配 buffer 不调 kernel = 死代码 = MTE2 带宽虚高 | host-and-testing §4 |

**场景约束（按算子语义特征自动适用——含 compute-first 编排即适用 R9/R10/R16/R21、含归约模块即适用 R17-R19，不依赖场景注册表命中；完整定义见场景文档）**：

| # | 适用场景 | 约束 | 详见 |
|---|---------|------|------|
| R9 | compute-first | CrossCore flag Set/Wait 严格配对 + flagId ∈ [0, FLAG_ID_MAX)；计数器 0-15 衡量未消费积压（不构成 T 上限；T 上限按 case 实测），不设数值拒绝 | fusion §6.2.3 |
| R10 | compute-first | T 派生优先 `T \| mSeg` 无尾块；单尾块 ≤ 头块（16 对齐）PUT 直传合法；多尾块/尾>头走策略 A padding | fusion §6.2.7 |
| R16 | compute-first | mm 内核默认 FragmentTensor 消 R 循环；vendor R×T 子调用须论证 SCALAR 占比 | fusion §6.2.2 |
| R21 | compute-first | **localLast 编排禁止移除**：以"每轮 Set 双 flag"替代 → flagId 用量翻倍 + 丧失通信提前启动；`cFragAddrs_` 顺序写错才是 A/C 错位根因，非 localLast 本身 | fusion §6.2.2 |
| R17 | compute-first（含归约） | 归约 2D DataCopyPad 批量（blockCount=本批行数）；逐行归约 = 性能 FAIL | fusion §6.2.6 |
| R18 | compute-first（含 BF16→FP32 归约） | 归约独立 srcFP32 双缓冲，禁止 in-place 加宽 Cast | fusion §6.2.6 |
| R19 | compute-first（含归约） | 归约四类 HardEvent 同迭代配对 + 残留消费；Set 无配对 Wait = 挂死（507014） | fusion §6.2.6 |

### 精度与验收纪律

以下纪律适用于所有 apace 通算融合算子（与方法论原则（architecture §1.5）、全局红线与场景约束互补，不重复展开）：

1. **以设备输入事实源计算 Golden。** 先冻结每卡输入/输出分布与切分轴，再生成测试输入；golden 切分轴写错则精度验证整体失效（golden 语义先行，方法论原则 2①）。
2. **golden 的 dtype 链必须完整可执行。** 从设备输入字节（FP8/MX 编码）解码 → 累加 dtype → 输出 dtype 的每步转换顺序、舍入与 clamp 都在 DESIGN 中写明；通信侧与计算侧的 dtype 约定不一致时标记冲突并停止，不用"golden 通过"掩盖。
3. **覆盖会改变失败边界的形状与规模组合。** 精度验证至少覆盖：rank 数端点、对齐与非对齐/尾块 shape、多 tile（T>1）与连续重复 launch；T=1 全部通过不能外推到 T>1——串行基线会掩盖多 tile 的 Win 区布局与 flag 配对问题。
4. **公开输出与诊断中间量分开判定。** 校验脚本分别报告用户可见输出与 Win 区/中间阶段数据的状态：输出通过不能掩盖中间量异常；DESIGN 已声明非公开的中间量异常也不能记为输出失败。

### 失败案例记录

生产失败案例按"现象 → 根因 → 修复方向"补入 [`review-checklist.md`](references/foundations/apace/review-checklist.md) 常见 FAIL 原因表（失败模式前置，方法论原则 4）；跨算子可复用的归入 Skill，算子特有的留在项目记录。设备结论必须标注执行环境、设备节点、实际命令和返回码；编译/静态通过不能写成设备已验证，单 case 重跑通过不能抹掉原始挂起。

### References

| 文档 | 何时读 |
|------|--------|
| [`workflow_integration.md`](references/foundations/apace/workflow_integration.md) | **plugin 7 步流程 apace 映射（主要消费文档）**：设计/开发/审查/修复/验收各阶段的 apace 技术要点与门禁 |
| [`review-checklist.md`](references/foundations/apace/review-checklist.md) | 逐项审查清单（全局红线 + 场景约束索引 + 操作化检查方法 + 常见 FAIL 原因） |
| [`troubleshooting/failure-navigation.md`](references/foundations/apace/troubleshooting/failure-navigation.md) | 按现象定位排查方向 |
| [`fundamentals/`](references/foundations/apace/fundamentals/) | 架构、通信（含通信组件扩展指南/hcomm 未封装原语）、计算、融合组合模式基础知识 |
| [`operator-design/`](references/foundations/apace/operator-design/) | **开发方法论**（[`dev-methodology.md`](references/foundations/apace/operator-design/dev-methodology.md)：五阶段流程 + D1-D6 决策点）+ DESIGN/PLAN 模板 + 算子解剖 + 开发指南 + **语义范式映射与架构差异**（需求存在注册形态既有实现时必读 [`paradigm-mapping.md`](references/foundations/apace/operator-design/paradigm-mapping.md)：算子族选型表、七维差异、可参考 vs 必须重写清单） |
| [`scenarios/`](references/foundations/apace/scenarios/) | 自定义扩展场景注册表（PUT AllToAll / PUT AllGather 变体、GET 推导、compute-first 组合模式） |
| [`references/shared/pipeline_tuning.md`](references/shared/pipeline_tuning.md) | 通算并行调优：tileCnt 两阶段策略（两路线共享） |
| [`references/shared/profiling_mc2.md`](references/shared/profiling_mc2.md) | 性能采集：msprof + L2 flush + 多卡后处理（两路线共享） |

---

## §3 ascendc-api 路线（MTE通信，MoE Dispatch/Combine）

裸 Ascend C API 全自建（不经任何通信库/模板库）。通信路径为 MTE通信 = AIV+UBMEM（AIV 触发 MTE 执行跨卡搬运），host 侧经 HCCL 分配 window 资源——此处 HCCL 仅做资源分配，非 HCCL 集合通信库高阶 API。支持 dav-2201（Atlas A2/A3 系列）+ dav-3510（Ascend 950PR/950DT）双平台，compat 层抹平两平台 window 地址结构差异（dav-2201/dav-3510 在 MoE SDK 工程代码中惯称 A3/A5，官方产品名以 Atlas A3 系列 / Ascend 950PR/950DT 为准）。注意区别于 apace 路线——apace 模板库虽基于 Ascend C API 构建（其通信基础 API 基于 HCOMM），但属独立编码底座。

> **跨路径设计原则**：阅读或改造 `dispatch`、`combine` 任一侧时，都要同步核对另一侧的接口和状态语义；`expandIdx`、`epRecvCounts`、`epSendCounts` 等中间量必须成对理解；host 侧只传总核数，kernel 侧负责决定每阶段实际使用多少核以及如何切分。

### 红线 / 约束

#### ① 跨卡搬运走 MTE通信，禁止 HCCL 高阶通信原语做数据搬运

MoE Dispatch/Combine 通过 host 侧 `HcclAllocComResourceByTiling` 创建通信 window 资源，device 侧用 MTE + `DataCopyPad` 访问 window 地址做跨卡数据搬运。禁止用 `Hccl::AlltoAll()`/`Hccl::AlltoAllV()`/`Hccl::AllReduce()` 等高阶通信原语替代——它们是对整块数据的黑盒集合通信，无法在 Kernel 内按 token 粒度插入路由计算和状态协议逻辑。

#### ② window 地址必须走 compat 层，禁止直接硬编码平台结构体偏移

`winContext`（`mc2Context`）按平台解释成 `HcclA3OpResParam` 或 `HcclA5OpResParam`，字段布局不同（dav-2201 侧远端地址通过链表跳转，dav-3510 侧状态区在前 1MB）。必须通过 compat 封装的 `GetBaseWindAddrByRankId()`、`GetBaseWindStateAddrByRankId()` 等统一接口访问，不要在主流程中手写平台分支。**注意**：`HcclA3OpResParam`/`HcclA5OpResParam` 是样例内的**精简定义**（A3/A5 为工程惯称命名），仅用于工程内自建 context，不适用于解析真实 HCCL 返回的 context（真实解析须用 SDK 完整 `HcclOpResParam`，见 `mte-address-access.md` 重要警告）。

#### ③ 共享 GM/状态区走 DataCopyPad，禁止 GetValue/SetValue 直接访问

共享 GM、`workspaceGM`、状态区、window 数据区的默认读写路径是 `DataCopyPad` 或等价 GM 路径。`SetValue()`/`GetValue()` 直接访问共享区域会导致跨核数据观察不稳定。`SyncAll` 只保证执行同步，**不保证** Data Cache 与 GM 一致性。极少数场景如必须使用 `SetValue`/`GetValue`，必须额外补齐 `DataCacheCleanAndInvalid` 并重新验证核间可见性。

#### ④ 状态协议：先数据后状态、每核只写自己槽位、消费后清理

- 发布顺序：先写 count/offset/附带信息，再发布 ready 状态
- 等待侧只轮询自己负责的状态段
- 每核只写自己的共享槽位，禁止多核直接累加同一地址
- 状态消费完成后才清理本核负责段，不要沿用单核整块 reset 习惯

**Reviewer 速查**：

| # | 检查项 | 方法 |
|---|--------|------|
| M1 | 无 HCCL 高阶通信原语 | `grep -rn "Hccl::AlltoAll\|Hccl::AlltoAllV\|Hccl::AllReduce\|Hccl::AllGather\|Hccl::ReduceScatter" operators/{op}/` 应为空 |
| M2 | window 地址走 compat 层 | kernel 中使用 compat helper（`GetBaseWindAddrByRankId` 等），不直接硬编码 `HcclA3OpResParam`/`HcclA5OpResParam` 字段偏移 |
| M3 | 共享 GM 走 DataCopyPad | `grep -rn "SetValue\|GetValue" operators/{op}/` 检查是否用于共享 GM/状态区，若有需确认有 `DataCacheCleanAndInvalid` |
| M4 | 状态协议正确 | 检查发布顺序（先数据后状态）、每核只写自己槽位、消费后清理 |

### References

**先判断任务类型**：当任务同时包含"先阅读现有实现"与"再进行改造"两部分时，路径顺序为：先完成入口 A 的阅读路径，再进入入口 B 的规格补齐。

#### 入口 A：阅读已有实现

适用场景：先建立现有 dispatch、combine 或 mega_moe / 融合实现的理解框架。

- 通用方法：[`references/foundations/ascendc-api/moe-dispatch-combine/reading/common-reading-method.md`](references/foundations/ascendc-api/moe-dispatch-combine/reading/common-reading-method.md)
- dispatch 专用：[`references/foundations/ascendc-api/moe-dispatch-combine/reading/dispatch-reading-guide.md`](references/foundations/ascendc-api/moe-dispatch-combine/reading/dispatch-reading-guide.md)
- combine 专用：[`references/foundations/ascendc-api/moe-dispatch-combine/reading/combine-reading-guide.md`](references/foundations/ascendc-api/moe-dispatch-combine/reading/combine-reading-guide.md)
- mega_moe / 融合实现：[`references/foundations/ascendc-api/moe-dispatch-combine/reading/mega-moe-reading-guide.md`](references/foundations/ascendc-api/moe-dispatch-combine/reading/mega-moe-reading-guide.md)

进入具体 guide 前，先判断是否属于融合实现（输入侧沿用 dispatch 风格、输出侧直接产出 combine 最终输出、主入口同时包含路由/发送/expert 计算/聚合/回写、通信协议不能拆成独立 dispatch+combine、关键内存布局同时承载多阶段共享状态）。命中任一特征时归入 mega-moe-reading-guide。

#### 入口 B：生成新算子或在样例工程基础上改造

适用场景：生成 moe dispatch、生成 moe combine，或在样例工程基础上修改 dispatch/combine。不再套用通用 `add.asc` 模板，默认拆成三部分推进：

1. **samples** / 规格补齐与样例工程：[`references/foundations/ascendc-api/moe-dispatch-combine/samples/index.md`](references/foundations/ascendc-api/moe-dispatch-combine/samples/index.md) → `spec-template.md` → `dispatch-dataflow.md` 或 `combine-dataflow.md`；sample 内 helper 分层见 `sample-helper-map.md`
2. **api-rules** / MoE 特化 API 规则：[`references/foundations/ascendc-api/moe-dispatch-combine/api-rules/index.md`](references/foundations/ascendc-api/moe-dispatch-combine/api-rules/index.md)（`DataCopyPad`、window 地址获取、同步与可见性、状态协议、dispatch/combine 接口契约）
3. **tiling-scheme** / MoE 特化分核方案：[`references/foundations/ascendc-api/moe-dispatch-combine/tiling-scheme/index.md`](references/foundations/ascendc-api/moe-dispatch-combine/tiling-scheme/index.md) → `window-memory-layout.md` → `multi-core-formulas.md` → `split-core-design.md` → `double-buffer-protocol.md`

整体架构、阶段协作关系和设计动机背景见 [`references/foundations/ascendc-api/moe-dispatch-combine/reading/design-overview.md`](references/foundations/ascendc-api/moe-dispatch-combine/reading/design-overview.md)，不替代三部分主路径中的规格、API 规则或分核设计文档。

#### 知识目录

| 目录 | 何时读 |
|------|--------|
| `references/foundations/ascendc-api/moe-dispatch-combine/reading/` | 阅读已有 dispatch/combine/mega_moe 实现时 |
| `references/foundations/ascendc-api/moe-dispatch-combine/samples/` | 规格补齐、工程组织参考、编译链路、接口语义和文件落点 |
| `references/foundations/ascendc-api/moe-dispatch-combine/api-rules/` | MoE 特有的 window 地址获取、DataCopyPad 规则、同步可见性、状态协议、接口契约 |
| `references/foundations/ascendc-api/moe-dispatch-combine/tiling-scheme/` | window 物理布局、工作量公式、各阶段分核方案、双缓冲轮转协议 |

---

## §4 hccl-matmul 路线（910B / HCCL 高阶 + AscendC::Matmul，仅注册）

910B（A2，`dav-2201`）上的集合通信类通算融合走官方路径：通信用 HCCL in-kernel 高阶 API（`Hccl<HCCL_SERVER_TYPE_AICPU>`，AICPU 引擎），计算用 `AscendC::Matmul`，工程形态为 **aclnn 单算子注册**（`build.sh` → `.run` + `libcust_opapi.so`）。通算融合**不支持 `<<<>>>` 直调与 GE 入图**（官方通算融合指南）。本期仅覆盖 A2，不涉及 A3（910_93）/950，不支持 quant。

> **与其他路线的边界**：910B 硬件不支持 SHMEM/UDMA，Blaze 为 950 专属；§1/§2 的"禁止 HCCL / 禁止 `AscendC::Matmul`"只约束 AIV+URMA 直调，本路线正好相反。MoE 的 MTE 窗口搬运仍走 §3，不要用 `Hccl::AlltoAll` 替代 token 级路由。

### 红线 / 约束

| # | 约束 | 说明 | 详见 |
|---|------|------|------|
| R1 | 架构=910B(A2) | `ascend910b` / `dav-2201`；无 A3/910_93/950 | 全路线共性红线 ① |
| R2 | 通信走 HCCL 高阶 V2 | `Hccl<HCCL_SERVER_TYPE_AICPU>` + `InitV2`/`SetCcTilingV2`；禁 SHMEM/HCOMM/窗口手动 MTE/RAC | [`comm_hccl.md`](references/foundations/hccl-matmul/comm_hccl.md) |
| R3 | Matmul 走 `AscendC::Matmul` | 头 `lib/matmul/matmul_intf.h`；禁 Blaze；不调 `SetLocalWorkspace` | [`matmul_fusion.md`](references/foundations/hccl-matmul/matmul_fusion.md) |
| R4 | AIC-only + Finalize 前跨核同步 | Prepare/Wait/Finalize 在 `ASCEND_IS_AIC` 内；Finalize 前 `CrossCoreSetFlag`/`CrossCoreWaitFlag` | [`mc2_architecture.md`](references/foundations/hccl-matmul/mc2_architecture.md) |
| R5 | 仅注册 / aclnn | 禁止 `<<<>>>` 直调；host 须 `MC2().HcclGroup("group")` | [`op_architecture.md`](references/foundations/hccl-matmul/op_architecture.md) |
| R6 | 无 quant | 不引入 `SetQuant*` / `qbmm_mx` / int8 通信路径 | — |
| R7 | PTA 不在本路线重做 | 算子可跑通后走 `torch-ascendc-op-extension` **路线 B**（aclnn 注册） | 该 skill `routes/aclnn-registry.md` |

**逐项审查清单**：[`review-checklist.md`](references/foundations/hccl-matmul/review-checklist.md)（违反红线项 = FAIL）。

精度判据须与融合方向匹配：先通后算（AllGather+MM）可用逐元素 `np.isclose` + 误差元素比例；先算后通（MM+ReduceScatter/AllReduce）须用整体相对误差 `‖y_npu−y_fp32‖₂/‖y_fp32‖₂ ≤ 4ε`，scatter 类须逐 rank 校验。详见 [`workflow_integration.md`](references/foundations/hccl-matmul/workflow_integration.md) §1.4.1。

### References

| 文档 | 何时读 |
|------|--------|
| [`workflow_integration.md`](references/foundations/hccl-matmul/workflow_integration.md) | 设计/开发/验收各阶段的 910B MC2 动作与门禁 |
| [`mc2_architecture.md`](references/foundations/hccl-matmul/mc2_architecture.md) | 第一次接触：AIC-only、prepare→Wait 两层流水、M 轴切分 |
| [`comm_hccl.md`](references/foundations/hccl-matmul/comm_hccl.md) | 写/审通信层：HCCL 高阶 API、V2 生命周期、窗口优化 |
| [`matmul_fusion.md`](references/foundations/hccl-matmul/matmul_fusion.md) | 写/审计算层与 HCCL 的耦合（API 签名一律引用 `ascendc-api-best-practices` 的 `api-matmul.md`） |
| [`op_architecture.md`](references/foundations/hccl-matmul/op_architecture.md) | aclnn 注册模型与可复用 `mc2_common` |
| [`codebase_map.md`](references/foundations/hccl-matmul/codebase_map.md) | 从蓝本 `cp -r` 起手按 `[REUSE]/[MODIFY]` 定点改造 |
| [`pipeline_tuning.md`](references/foundations/hccl-matmul/pipeline_tuning.md) | 本路线 tileCnt 约束与公式化 tiling 系数；两阶段策略复用 [`shared/pipeline_tuning.md`](references/shared/pipeline_tuning.md) |
| [`references/shared/profiling_mc2.md`](references/shared/profiling_mc2.md) | msprof task-based + 多卡 last-N 取 max（本路线 **无需** 950 的 L2 flush） |
| `references/foundations/hccl-matmul/all_gather_matmul/` | 已验证参考实现（AllGather+Matmul，910B/HCCL/AICPU，aclnn）——`cp -r` 起手；**勿改蓝本** |
