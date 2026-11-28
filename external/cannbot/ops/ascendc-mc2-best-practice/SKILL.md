---
name: external-cannbot-ops-ascendc-mc2-best-practice
description: Ascend C 多卡通算融合（MC2）开发与故障咨询。可依据已有现象分析多来源归约的 Cast 与缓冲复用、用 msprof 时间线核对
  L2 冷缓存采样（cacheFlush、MTE2 带宽），或解读 APACE 的 PUT/GET 接口调查。适用于集合通信与 Matmul、SHMEM/UDMA
  与 AIV/URMA 多卡直调、HCCL/ACLNN 注册，以及 MTE 通信的 MoE Dispatch/Combine、mega_moe 等专家并行算子的选型、设计、实现和评审。
original-name: ascendc-mc2-best-practice
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-11-28'
synced-commit: 7a09cdfac4b2869dca6b6fd4531a0a5e539d6096
license: UNKNOWN
---

# Ascend C MC2 通算融合算子开发最佳实践

MC2（Matrix Computation & Communication）融合多卡通信与本地计算，通过通算流水掩盖通信开销。提供 MC2 通算融合算子开发的技术参考：通信与计算的 API 选择、时序约束、验收标准。

## 开发规范

普通 Kernel 直调在一张卡上启动、分块、验收。开发 MC2 要把这三件事扩展到所有 Rank：Host 如何建链，哪一段数据由哪张卡生产和消费，以及如何逐卡验证结果与通算重叠。按[算子类型与调用形态](#决策树先判断算子类型)确定搬运和同步的具体方式。

### 单卡直调与多卡融合的开发差异

| 环节 | 普通单卡 Kernel 直调 | 多卡通算融合 |
|---|---|---|
| Host 启动 | 在一张卡上准备张量和 stream，通过 `<<<>>>` 启动。 | 每个 Rank 绑定设备并建立通信域。直调路线准备对称内存或通信窗口；注册路线经 aclnn 调用，从框架取得通信上下文。 |
| Tiling 与实现 | 按本地张量切块、分核，安排片上缓冲。 | 把每卡分片对应到数据的来源、目的 Rank 和发送、接收或归约范围；按计算与通信依赖安排等待，消费完成后复用缓冲。 |
| 精度与性能验证 | 本卡输出与 Golden 比较，测量本卡 Kernel 耗时。 | 所有 Rank 联跑，逐卡与 Golden 中各自应得的完整结果或分片比较，检查跨卡完成顺序；性能采样条件一致时，用多卡时间线检查通算重叠。 |

## 决策树：先判断算子类型

集合通信与矩阵融合按张量分片搬运，专家并行按 token 的目标专家搬运。选定数据流后，再按芯片和调用方式选择底座。

| 算子场景与路线 | 通信与计算的实现 |
|---|---|
| 950 集合通信 + Matmul（AllToAll、AllReduce、TP/SP）：[blaze-shmem](references/foundations/blaze-shmem/) | AIV+URMA 直调，SHMEM 建链并分配对称内存。用 SHMEM API（如 `aclshmemx_udma_*`）搬运通信块，使用 Blaze 模板做 Matmul。 |
| 950 使用 APACE 的集合通信 + Matmul（如 AllGather + QuantMatmul）：[apace](references/foundations/apace/) | 直调 MIX Kernel。`CollectiveComm` 经 HCOMM 编排 AIV+URMA 通信，与 Blaze 矩阵计算配合；AllReduce 需先[扩展通信组件](references/foundations/apace/operator-design/paradigm-mapping.md)。 |
| MoE Dispatch/Combine、mega_moe 与专家并行 EP（dav-2201 / dav-3510）：[ascendc-api](references/foundations/ascendc-api/) | HCCL 只分配窗口。按 token 的目标 Rank 经 compat 取得窗口地址，禁止硬编码平台结构体偏移。生产核用 MTE `DataCopyPad` 搬数据，数据就绪后只发布自己的状态槽；消费后清槽再复用。共享 GM 和状态区的读写均用 `DataCopyPad`，禁止 `GetValue/SetValue`。[MTE 通信约束](#3-ascendc-api-路线mte通信moe-dispatchcombine) |
| 910B（A2 / dav-2201）集合通信 + Matmul：[hccl-matmul](references/foundations/hccl-matmul/) | 仅通过 aclnn 注册调用，禁止 `<<<>>>` 直调。`op_host` 生成矩阵与通信 Tiling，`op_kernel` 使用 HCCL 高阶集合通信（`Hccl<HCCL_SERVER_TYPE_AICPU>` / `InitV2`）和 `AscendC::Matmul`。 |

MoE 已有实现从[源码阅读](references/foundations/ascendc-api/moe-dispatch-combine/reading/)进入；生成或改造算子见[入口 B](#入口-b生成新算子或在样例工程基础上改造)。进入开发后按[分路线的开发流程与性能验收](#分路线的开发流程与性能验收)推进。

**按问题选入口**：
- 比较 SHMEM/UDMA 与 MTE 时，先区分新算子选型和已有代码识别：新算子按数据分布选，已有 Kernel 从 SHMEM 搬运调用或 `winContext` / `mc2Context` 窗口取址反查。各自的适用场景与识别依据见[通信路径选项](references/requirement-analysis/comm-path-decision.md#通信路径选项)。
- 已指定芯片、框架或调用形态，询问具体组合是否支持、该路线的约束、参考实现或 PTA 接入时，查一次[能力声明](references/capability-declaration.md)。说明命中路线及约束；支持的组合给出 `reference_impl` 和知识目录的路径，并按本页说明调用步骤。实际开发时再展开样例或加载 PTA Skill。
- 介绍工作流：询问 Kernel 直调的完整流程时，先按算子类型选通信路线，再按[本页流程表](#分路线的开发流程与性能验收)分别说明集合通信与矩阵融合的 AIV+URMA 七步、MoE Dispatch/Combine 的 MTE 三部分，以及各自的分块与缓存条件。进入具体路线开发时再查该底座的工作流文档。询问 APACE 工作流时，用[入口与执行范围](#按请求目的路由)说明独立开发和插件开发各从哪里开始，以及完整开发、只做设计或技术咨询各停在哪里。实施时再展开当前阶段的资料。
- 排障或解释已有调查：归约误差按[多卡精度排障](#多卡精度排障)定位首次偏差；APACE 接口调查按[路线模型](#路线模型)说明已核对的范围、可复用的接口和选定场景。询问故障怎样分类、记在哪里时，用[失败案例记录](#失败案例记录)给出所属阶段、目标文档及该表一行的填法。只问定位顺序或调查结论时，以题目给出的现象为依据，把需要执行的核验留在后续步骤。

**知识分层（依赖方向单向，禁止反向引用）**：

| 层 | 技能 | 事实源范围 | 本 skill 的使用方式 |
|----|------|-----------|-------------------|
| 底层（通用 API） | `ascendc-api-best-practices` | Ascend C API 签名、硬件限制、平台差异（DataCopy/Cast/HardEvent/CrossCore flag/Hcomm/HCCL host 等） | 一律引用其 references，本 skill **不复述** API 细节 |
| 底层（Blaze 模板） | `ascendc-blaze-best-practice` | Blaze/tensor_api 通用模板选型、Tiling 算法选择 | 同上 |
| **本层（领域知识）** | 本 skill | MC2 特有：路线决策、apace/blaze-shmem/MoE 框架契约、通算流水编排、质量门禁 | — |

> 下层技能不引用本 skill；本 skill 文档中凡 API 级事实（签名、限制、硬件行为）均指向下层锚点，正文只保留 MC2 场景应用与编排规则。

## 何时使用 / 不适用

**使用信号**（任一即可）：
- 场景：多卡协同的通算融合算子（AllToAll+Matmul、AllReduce+Matmul、MoE Dispatch/Combine、多卡 EP/TP/SP 融合 Kernel）
- 接入已有专家并行（Expert Parallelism，EP）接口时，直接查阅[推理与训练的 Dispatch/Combine 用法](references/foundations/ascendc-api/moe-dispatch-combine/api-rules/dispatch-combine-contracts.md#公开接口)。自建 Kernel 时再按通信路线选择开发底座。
- 关键词："MC2"、"SHMEM"、"APACE"、"apace"、"通算融合"、"多卡通信直调"、"UDMA"、"URMA"、"AllToAll+Matmul"、"CollectiveComm"、"MoE Dispatch"、"MoE Combine"、"专家并行"、"EP"、"mega_moe"、"MTE通信"、"MTE window"、"token 路由分发"、"910B"、"HCCL"、"aclnn 通算融合"
- 代码：现有工程同时出现通信 API（`shmem.h`/`aclshmem*` 或 `collective_comm_api.h`/`CollectiveComm`）与 `blaze/gemm/` 模板（blaze-shmem/apace 路线），或出现 `winContext`/`mc2Context`/`HcclOpParam` 与 MTE 通信窗口结构（MTE通信 → ascendc-api 路线），或出现 `Hccl<HCCL_SERVER_TYPE_AICPU>` 与 `AscendC::Matmul`（HCCL 高阶 → hccl-matmul 路线）

**不适用**（走其他 skill）：
- 纯单卡 Matmul（无跨卡通信）→ `ascendc-blaze-best-practice`（950）或 `ascendc-api-best-practices` 的 `api-matmul.md`（910B）
- Vector 类逐元素/归约算子（无 Cube、无跨卡通信）
- 通用 Ascend C API 用法查询 → `ascendc-api-best-practices`
- 非 3510 架构的 AIV+URMA 通算融合（MTE通信除外，支持 dav-2201 + dav-3510 双平台）

> 确切的支持组合（chip × 算子类型 × 调用形态 × 通信路径 × 编程抽象）以 [`references/capability-declaration.md`](references/capability-declaration.md) 路线登记表为准；表内同时登记**明确不支持**的组合及原因，命中否定行时直接答复用户不可用 + 替代建议。

### 分路线的开发流程与性能验收

明确路线后沿表中对应一行推进。未指定路线而需了解完整流程时，分别说明下表各行的开发步骤与验收条件；即使同为 Kernel 直调，MTE 的工程组织和缓存要求也与 SHMEM/UDMA 不同。

**M 轴切分的设计说明（AIV+URMA 流水）**：等长、无尾块时，`M = tileCnt × headMSize`，即总行数 = 块数 × 每个主块的行数。若单独处理尾块，应把它的实际行数计入总行数，并说明 [Host 与 Kernel 如何统计块数](references/shared/pipeline_tuning.md#1-tilecnt-是什么)。

| 通信路线 | 开发步骤 | 性能验收的 L2 条件 |
|---|---|---|
| AIV+URMA 直调（blaze-shmem 使用 SHMEM/UDMA；apace 使用 `CollectiveComm`） | 环境检查 → 设计（按上式将块数换算为每块行数）→ 设计串讲 → 开发 → 检视与按需修复 → 验收与汇报。两份插件指南均在 Step 2.5 串讲、Step 7 汇报。[APACE](references/foundations/apace/workflow_integration.md) 的 Step 6 验收精度和性能；[blaze-shmem](references/foundations/blaze-shmem/workflow_integration.md) 在开发时先验证精度，Step 6 做性能验收。 | [冷缓存采样](references/shared/profiling_mc2.md)要求每轮主 kernel 前执行 L2 flush。 |
| MTE 通信直调（ascendc-api，MoE Dispatch/Combine） | 生成或改造 MoE 算子默认按[三部分开发流程](#入口-b生成新算子或在样例工程基础上改造)推进：`samples` 补齐规格、搭建样例工程 → `api-rules` 核对窗口取址、搬运与同步 → `tiling-scheme` 设计窗口布局及多核工作量分配。 | 按该算子的采样协议控制缓存，无统一的逐轮 flush 要求。 |
| HCCL 高阶注册（hccl-matmul，AICPU） | 集合通信按 §4 经 aclnn 注册调用。 | 无需上述 AIV+URMA 的逐轮 L2 flush。 |

## 多卡精度排障

多卡结果与 Golden 普遍不符时，先按算子定义还原每个 rank 的输入和预期输出：对各矩阵标明 M、N、K 中哪根轴切分、哪些维度完整保留，输出由哪些 rank 的哪段数据得到。切分轴以该算子的分布约定为准，即使 shape 一致也可能取错片。
对照 `gen_data.py` 的取片、归约和输出归属。确认生成器切错轴，先修正并重生成 Golden，用同一输入复验。每卡语义一致后再定位 kernel：compute-first 先比本地计算、再比通信后输出；comm-first 先比接收数据、再比计算。

compute-first 归约结果系统性出错、误差随来源序累积时，先查归约前的 BF16→FP32 Cast（R18）。原地宽化会用 4 字节的 FP32 结果覆盖尚未读取的 2 字节 BF16 源元素，调整累加顺序无法恢复源值。为转换结果分配两个独立于 `srcBF16` 的 FP32 槽，按来源交替使用 `srcFP32[0]` 和 `srcFP32[1]`，累加器只占一份。逐来源比较 Cast 前后及累加后的值，定位首次偏差。

## 具体算子的需求分析

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

## 路线约束

架构白名单以[路线登记表](references/capability-declaration.md)为准，例如 dav-2201 不走 AIV+URMA。

> **先确定谁建立通信域，再选融合 Kernel。** §4 注册融合实现中的 HCCL 通信和 `AscendC::Matmul` 计算都依赖框架注入、由 `GetHcclContext` 取得的上下文。HCCL 用它初始化通信域，矩阵调度用该域的 Rank ID 确定本卡的分片和输出位置。直接用 `<<<>>>` 启动同一 Kernel 时，这两条路径都缺少启动所需的数据。
> - **blaze-shmem 直调**：Host 用 `aclshmemx_init_attr` 建链、`aclshmem_align` 分配对称内存；Device 用 `aclshmemx_udma_*` 搬运，按协议用 `aclshmem_barrier_all` 同步。融合 Kernel 等待数据就绪后调用 `Blaze::Gemm::Kernel::*` 矩阵模板，由其中的 `Blaze::Gemm::Block::BlockMmad` 执行块乘。见[通信接口](references/foundations/blaze-shmem/comm_shmem.md)和[矩阵计算](references/foundations/blaze-shmem/matmul_blaze.md)。
> - **apace 直调**：由 `CollectiveComm` 编排通信，计算侧复用 Blaze 的块级接口，见 [apace 矩阵计算](references/foundations/apace/fundamentals/compute.md)。

各底座特有约束与逐项审查清单见 §1/§2/§3/§4。

---

## §1 blaze-shmem 路线

SHMEM 通信库（cann/shmem，与 HCOMM 无关）驱动 AIV+URMA 跨卡搬运，Blaze 计算模板负责 Cube 计算，二者手工组装为独立 CMake 工程（自带 `include/` 全套）。计算侧 `Blaze::Gemm::Block::BlockMmad` + `Blaze::Gemm::Kernel::*`。

### 红线 / 约束

| # | 约束 | 说明 | 详见 |
|---|------|------|------|
| R1 | 架构白名单 | CMakeLists.txt 中 `npu-arch` = dav-3510 | 路线登记表 |
| R2 | 禁止 HCCL 高阶 API（`Hccl::*`） | 注册式 MC2 通过 `GetHcclContext` 获取通信域；SHMEM 直调使用 Host 自建的通信环境 | [`comm_shmem.md`](references/foundations/blaze-shmem/comm_shmem.md) §5（7 类 18 个 API 清单） |
| R3 | 禁止 `AscendC::Matmul` 高阶 API | 本路线以 `Block::BlockMmad` 与 `Kernel::*` 模板编排逐块计算和通信依赖 | [`matmul_blaze.md`](references/foundations/blaze-shmem/matmul_blaze.md) |
| R4 | 通信走 SHMEM | Host 已为全部参与 Rank 建好通信环境与对称内存；Device 使用约定的对端地址并在消费或复用窗口前确认传输完成 | [`comm_shmem.md`](references/foundations/blaze-shmem/comm_shmem.md) |
| R5 | Matmul 走 Blaze | 头文件含 `blaze/gemm/block/block_mmad*.h` | [`matmul_blaze.md`](references/foundations/blaze-shmem/matmul_blaze.md) |
| R6 | L2 flush 证据 | 采集记录中每轮主 kernel 前都执行了 L2 flush 或等效操作 | 本页“分路线的开发流程与性能验收” |
| R7 | blaze-shmem 工程交付 | `docs/` 下 DESIGN/PLAN/WALKTHROUGH/REVIEW.md 齐全；环境检查通过 | — |

**逐项审查清单**：[`review-checklist.md`](references/foundations/blaze-shmem/review-checklist.md)（含常见 FAIL 原因与修复方向，违反红线项 = FAIL）。

### References

| 文档 | 何时读 |
|------|--------|
| [`references/foundations/blaze-shmem/workflow_integration.md`](references/foundations/blaze-shmem/workflow_integration.md) | 设计 MC2 算子前，看 MC2 场景的技术要点和门禁 |
| [`references/foundations/blaze-shmem/mc2_architecture.md`](references/foundations/blaze-shmem/mc2_architecture.md) | 第一次设计 MC2 算子，建立 AIV/AIC 分工 + 4-buffer 流水 + M 轴切分心智模型 |
| [`references/foundations/blaze-shmem/comm_shmem.md`](references/foundations/blaze-shmem/comm_shmem.md) | 写/改通信层，查 SHMEM API、UDMA 用法、禁止 HCCL 清单（§5）、扩展其他通信原语 |
| [`references/foundations/blaze-shmem/matmul_blaze.md`](references/foundations/blaze-shmem/matmul_blaze.md) | 写/改计算层，选 Blaze 模板、改 DispatchPolicy、处理 Scale |
| [`references/shared/profiling_mc2.md`](references/shared/profiling_mc2.md) | 性能采集与调优时：msprof task-based 采集 + L2 flush + 多卡数据后处理 |
| [`references/shared/pipeline_tuning.md`](references/shared/pipeline_tuning.md) | 精度调试阶段用合法的 tileCnt=1 做串行基线；性能调优阶段扫描 tileCnt；设计时可用公式化 Tiling 优先选择候选 |
| [`references/foundations/blaze-shmem/codebase_map.md`](references/foundations/blaze-shmem/codebase_map.md) | 工程搭建时，定位"哪些文件改/不改"与改造食谱 |
| `references/foundations/blaze-shmem/all_to_all_matmul/` | 编译验证过的基底工程，blaze-shmem 路线所有 MC2 算子的起手模板 |

---

## §2 apace 路线

APACE（Ascend PArallel Communication-compute Engine）是通算融合算子的架构底座，提供可复用的 block 层接口、kernel 层参考实现和 tiling 算法。APACE 自建通信基础 API（基于 HCOMM 通信基础库构建，与 HCCL 集合通信库无关）。apace 路线在 `kernel/<op>/` 下新建算子，复用稳定共享层 `block/` `tiling/`。

> **apace 路线接口契约基准**：[cann/ops-transformer](https://gitcode.com/cann/ops-transformer) `mc2/common/op_kernel/apace/` 子树，以 pin 的已验证快照为结构基准。样例代码经 scripts/fetch_apace.sh 现取现读（支持 `--ref`/`APACE_PIN_REF` 锚定 commit）；⚠️ master 结构会演进（如目录迁移、新增算子），拉取 master 后必须与快照 diff 校验，文档引用失效时更新文档——**已知漂移登记（pin 之后 10 个提交：block/→core/ 迁移、ReduceScatter UBMEM 官方模板、hcomm dcci 等）见 [`references/foundations/apace/fundamentals/communication.md`](references/foundations/apace/fundamentals/communication.md) §8，核对 master 前必读**。
>
> **工程实现默认事实源**：CANN 内置 apace 框架（路径随 CANN 打包形态实测定位，Step 1 登记；两种已验证形态：`opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/apace/`（cann-9.2.0）、`vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/apace/`（cann-9.1.0）），**直调独立工程默认 CMake 直引该路径，禁止整包复制**。

### 按请求目的路由

从头开发有两个入口。使用 `ops-direct-invoke` 插件时，以 [`workflow_integration.md`](references/foundations/apace/workflow_integration.md) 为主入口，按其中的七步映射推进。独立开发从 [Step 1 工程搭建](references/foundations/apace/workflow/step1-project-setup.md) 开始，依次核对接口（Step 2）、定稿设计（Step 3）、实现并验收（Step 4）。

工作范围由交付目标决定：

| 请求目的 | 从哪里开始 | 交付内容与停止位置 |
|---|---|---|
| 完整开发 | 按上述开发方式选择入口 | 完成实现、多卡精度验证和性能采集。 |
| 仅交付设计 | 独立流程从 [Step 2 核对接口](references/foundations/apace/workflow/step2-investigation.md) 开始；插件走其设计阶段 | 独立流程完成 [Step 3 设计](references/foundations/apace/workflow/step3-design.md) 后停止。插件在 Step 2 产出设计，涉及交接时在 Step 2.5 串讲，Step 3 才开始开发。可执行路线交付 DESIGN.md 与 PLAN.md。 |
| 咨询、解释、评审 | 开发规范查[单卡与多卡对照](#单卡直调与多卡融合的开发差异)，流程概览查[分路线流程表](#分路线的开发流程与性能验收) | 只读与问题相关的资料；涉及具体实现或审查时，结合相应 references 说明依据。 |
| 排障 | [精度与验收纪律](#精度与验收纪律)及对应红线；动手排查见[故障导航](references/foundations/apace/troubleshooting/failure-navigation.md) | 仅询问定位顺序时，说明先查什么、如何判定及下一步；需要动手时再按症状取证。 |
| 源码核对结果与能力查询 | [路线模型](#路线模型) | 区分查询状态、实现路线和下一步，说明哪些接口可复用，计划中哪些部分仍需实现、注册和验证。 |

### 计算执行原则

同一个打包步骤放在哪里，取决于谁消费它的结果。

- **候选算子消费**：打包、通信、解包和矩阵计算在设备上的一次 MIX Kernel launch 内完成，由 AIV 与 AIC 协作。Host 准备原始输入和 buffer、计算 Tiling、多 Rank 建链（交换 rootInfo、分配窗口、跨 Rank barrier）、按 dtype 启动 Kernel 并回收结果。在 Host 预先排好数据会跳过候选算子应实现的搬运。
- **比对程序消费**：CPU Golden 可以独立模拟各 Rank 的打包、通信、归约与解包，算出期望结果。验证时，两条路径从同一份原始输入开始，Golden 的中间缓冲只供参考计算使用。

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

### APACE 独立开发的四步流程

1. **Step 1: Project Setup** — 建立算子工程骨架、校验 CANN 内置 apace 事实源与环境（dav-3510、多卡、HCCL）、登记只读参考点。不核对接口事实、不判定路线、不创建实现文件。→ [`workflow/step1-project-setup.md`](references/foundations/apace/workflow/step1-project-setup.md)
2. **Step 2: Framework Investigation（设计前核对）** — 只读核对 apace 框架的接口事实（matmul 链路/通信接口/入口 ABI/官方覆盖性）；事实**默认内联记录于 DESIGN.md §0.3**（无需独立调查报告）。此步只读取资料，不编译或运行设备；推荐方案、匹配场景与选择路线均在 Step 3。→ [`workflow/step2-investigation.md`](references/foundations/apace/workflow/step2-investigation.md)
3. **Step 3: Design** — 按需求中的通信语义、计算顺序和调用形态，对照 Step 2 已核实的接口选择路线，生成定稿的 DESIGN 和可执行路线的 PLAN。只产出设计文档，不复制文件、不编写实现、不执行构建。→ [`workflow/step3-design.md`](references/foundations/apace/workflow/step3-design.md)
4. **Step 4: Implementation** — 按定稿的 DESIGN/PLAN 完成实现、多卡验证与性能采集；实现层问题在项目内修复，不改设计、不改接口、不扩大支持域。→ [`workflow/step4-implementation.md`](references/foundations/apace/workflow/step4-implementation.md)

### 路线模型

```text
implementation_route: apace_native | apace_custom | unsupported
selected_scenario: <仅 apace_custom 填写>
```

PUT 表示发起方将数据写到对端，GET 表示发起方从对端读取；compute-first 表示先计算再通信。例如可先算出本卡矩阵块，再用 PUT 发给目的 Rank。读写方向与计算先后共同描述需求，是否能复用现成 Kernel 要用这两个条件去核对。

源码查询状态与实现路线分开记录：在声明的读取边界内未找到 kernel 或钩子记为 `not_found`，不等于 apace 不支持；记录方式见 [`step2-investigation.md`](references/foundations/apace/workflow/step2-investigation.md)。实现路线取决于需求语义和场景登记。已有调查满足判断条件时，同时给出查询状态、选定路线和仍需验证的部分：

- **`apace_native`**：官方 `apace/kernel` 已有算子可直接调用或参考复用——不读取场景注册表；
- **`apace_custom`**：官方 kernel 未覆盖需求，但可复用已验证的 `apace/block` 接口，且需求语义唯一命中[场景注册表](references/foundations/apace/scenarios/index.md)。登记 `selected_scenario` 并说明命中条件。PLAN 指明自研 kernel、该场景的通信分发注册位置和多 rank 验证方法。
- **`unsupported`**：`apace/block` 无法组合实现，或场景零命中、多命中时，APACE 路线到此结束。若要绕过 block、用裸 Ascend C 全自建，应另选 `ascendc-api` 底座并重新核对能力声明。官方 kernel 缺失本身不授权切换底座。

**GET AllGather + QuantMatmul 的调查与选型**：已核对的 GET 钩子只有 AllToAll 的 `all_to_all_udma_get.h`；在同一读取边界内，AllGather 的 GET 钩子和参考 kernel 均记为 `not_found`。保留官方 PUT AllGather 作为接口组合参照。N 轴切分、UDMA 直调等条件唯一命中 [`get-all-gather-quant-matmul`](references/foundations/apace/scenarios/get-all-gather-quant-matmul/design.md) 时，选 `apace_custom`；待核实的条件单独列出。按上述定制路线复用已验证的 block 层接口，实现并注册 GET kernel，逐 rank 对照 AllGather 数据和矩阵结果。

> **场景文档查阅不受官方覆盖性判定反向阻断**：本仓/用户工程中已存在某场景的生产实现时（如 compute-first 类算子已生成过），阅读该场景文档不需要"官方未覆盖"前提——场景文档同时是该类算子的改造/复用手册。

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
| R7 | 禁止 `AscendC::Matmul` 高阶 API | APACE 直调使用当前底座的 Blaze 矩阵计算接口 | compute |
| R8 | 禁止 `Hccl::*` 高阶 API | 依赖框架注入上下文，直调拿不到 | comm_shmem §5 |
| R11 | host 前置校验在 fork/建链前拒绝非法输入 | 非法输入进 kernel = 难查的死锁/精度问题（整除/对齐/核数/Win 容量等；compute-first 完整 9 项见场景文档） | development-guide §3.5 |
| R12 | UB 静态通信区物理隔离 | 混用重叠 → 通信数据被踩踏 → 死锁；**`TPipe` 与 `MakeMemPtr` 必须二选一，禁止混用** → 507015 | communication 陷阱 #9；operator-anatomy §4.3 |
| R13 | 通信对象 `totalJobs=rankSize` 多核并行 | 退化 totalJobs=1 → 通信时间放大 R 倍（已证伪臆造约束）；TeamBarrier totalJobs 按分核映射取值（官方前 R 核 = rankSize；compute-first 后 R 核 = 1 + 显式 CrossDevice） | communication §2.2 |
| R14 | Win 数据/元数据分离（硬红线）+ 单轮 PUT 大小（风险提示） | 覆盖元数据 → "假通过"；单轮 PUT 无官方硬上限（512KB 系 bring-up 经验，同平台生产实现 12.6MB 稳定），大单轮按 case 复核 | communication 陷阱 #12/#13 |
| R15 | 投产级性能验证门槛 | 真实大 shape × R=2/4 双档 × 三路径对标；仅基线 = 未达标 | host-and-testing |
| R20 | perf 模式逐轮执行并验证 L2 flush | 只分配 cacheFlush 而未调用 flush kernel 是死代码，热缓存会使 MTE2 带宽虚高。逐卡检查同一 msprof 采集窗口：flush kernel（如 heavy_add_kernel）记录数等于采集轮数，每轮先完成 flush 再执行主 kernel，耗时只取主 kernel | [`host-and-testing.md`](references/foundations/apace/operator-design/host-and-testing.md) §4 |

**场景约束（按算子语义特征自动适用——含 compute-first 编排即适用 R9/R10/R16/R21、含归约模块即适用 R17-R19，不依赖场景注册表命中；完整定义见场景文档）**：

| # | 适用场景 | 约束 | 详见 |
|---|---------|------|------|
| R9 | compute-first | CrossCore flag Set/Wait 严格配对 + flagId ∈ [0, FLAG_ID_MAX)；计数器 0-15 衡量未消费积压（不构成 T 上限；T 上限按 case 实测），不设数值拒绝 | fusion §6.2.3 |
| R10 | compute-first | T 派生优先 `T \| mSeg` 无尾块；单尾块 ≤ 头块（16 对齐）PUT 直传合法；多尾块/尾>头走策略 A padding | fusion §6.2.7 |
| R16 | compute-first | mm 内核默认 FragmentTensor 消 R 循环；vendor R×T 子调用须论证 SCALAR 占比 | fusion §6.2.2 |
| R21 | compute-first | **localLast 编排禁止移除**：以"每轮 Set 双 flag"替代 → flagId 用量翻倍 + 丧失通信提前启动；`cFragAddrs_` 顺序写错才是 A/C 错位根因，非 localLast 本身 | fusion §6.2.2 |
| R17 | compute-first（含归约） | 归约 2D DataCopyPad 批量（blockCount=本批行数）；逐行归约 = 性能 FAIL | fusion §6.2.6 |
| R18 | compute-first（含 BF16→FP32 归约） | 禁止 in-place 加宽 Cast；缓冲划分与定位见[多卡精度排障](#多卡精度排障) | [`fusion.md`](references/foundations/apace/fundamentals/fusion.md) §6.2.6 |
| R19 | compute-first（含归约） | 归约四类 HardEvent 同迭代配对 + 残留消费；Set 无配对 Wait = 挂死（507014） | fusion §6.2.6 |

### 精度与验收纪律

以下纪律适用于所有 apace 通算融合算子（与方法论原则（architecture §1.5）、全局红线与场景约束互补，不重复展开）：

1. **Golden 的每卡语义**见[多卡精度排障](#多卡精度排障)，逐 rank 的输入、输出分布核实后再比较中间结果。
2. **golden 的 dtype 链必须完整可执行。** 从设备输入字节（FP8/MX 编码）解码 → 累加 dtype → 输出 dtype 的每步转换顺序、舍入与 clamp 都在 DESIGN 中写明；通信侧与计算侧的 dtype 约定不一致时标记冲突并停止，不用"golden 通过"掩盖。
3. **覆盖会改变失败边界的形状与规模组合。** 精度验证至少覆盖：rank 数端点、对齐与非对齐/尾块 shape、多 tile（T>1）与连续重复 launch；T=1 全部通过不能外推到 T>1——串行基线会掩盖多 tile 的 Win 区布局与 flag 配对问题。
4. **公开输出与诊断中间量分开判定。** 校验脚本分别报告用户可见输出与 Win 区/中间阶段数据的状态：输出通过不能掩盖中间量异常；DESIGN 已声明非公开的中间量异常也不能记为输出失败。

### 失败案例记录

可复用的生产故障在 [`review-checklist.md` 的“常见 FAIL 原因”表](references/foundations/apace/review-checklist.md#常见-fail-原因)追加一行「现象 → 根因 → 修复方向」。算子专有问题按同样格式留在项目记录，执行证据按发生阶段填写。

| 阶段 | 执行环境与位置 | 实际命令与返回码 | 可得出的结论 |
|---|---|---|---|
| 构建 | 编译器所在的构建环境 | 构建命令、返回码 | 审查发现架构缺项属构建风险；编译器已报错则记为构建失败。 |
| 设备运行 | 实际运行环境（如运行容器与 CANN 版本）、设备节点 | 设备测试命令、返回码 | 上机执行失败且有对应证据时记为 `actual_failure`。仅审查或编译时填「未执行，待验证」；重跑与原始挂起各自留存。 |

### References

| 文档 | 何时读 |
|------|--------|
| [`workflow_integration.md`](references/foundations/apace/workflow_integration.md) | 插件开发的 APACE 技术指南，按七步映射查阅各阶段的接口、产物和验收条件 |
| [`review-checklist.md`](references/foundations/apace/review-checklist.md) | 逐项审查清单（全局红线 + 场景约束索引 + 操作化检查方法 + 常见 FAIL 原因） |
| [`troubleshooting/failure-navigation.md`](references/foundations/apace/troubleshooting/failure-navigation.md) | 按现象定位排查方向 |
| [`fundamentals/`](references/foundations/apace/fundamentals/) | 架构、通信（含通信组件扩展指南/hcomm 未封装原语）、计算、融合组合模式基础知识 |
| [`operator-design/`](references/foundations/apace/operator-design/) | **开发方法论**（[`dev-methodology.md`](references/foundations/apace/operator-design/dev-methodology.md)：五阶段流程 + D1-D6 决策点）+ DESIGN/PLAN 模板 + 算子解剖 + 开发指南 + **语义范式映射与架构差异**（需求存在注册形态既有实现时必读 [`paradigm-mapping.md`](references/foundations/apace/operator-design/paradigm-mapping.md)：算子族选型表、七维差异、可参考 vs 必须重写清单） |
| [`scenarios/`](references/foundations/apace/scenarios/) | 自定义扩展场景注册表（PUT AllToAll / PUT AllGather 变体、GET 推导、compute-first 组合模式） |
| [`references/shared/pipeline_tuning.md`](references/shared/pipeline_tuning.md) | 通算并行调优：tileCnt 两阶段策略与公式化 Tiling（两路线共享） |
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

### 红线与接入方式

| # | 约束 | 说明 |
|---|------|------|
| R1 | 架构=910B(A2) | `ascend910b` / `dav-2201`；无 A3/910_93/950 |
| R2 | 通信走 HCCL 高阶 V2 | `Hccl<HCCL_SERVER_TYPE_AICPU>` + `InitV2`/`SetCcTilingV2`；禁 SHMEM/HCOMM/窗口手动 MTE/RAC |
| R3 | Matmul 走 `AscendC::Matmul` | 头 `lib/matmul/matmul_intf.h`；禁 Blaze；不调 `SetLocalWorkspace` |
| R4 | AIC-only + Finalize 前跨核同步 | Prepare/Wait/Finalize 在 `ASCEND_IS_AIC` 内；Finalize 前 `CrossCoreSetFlag`/`CrossCoreWaitFlag` |
| R5 | 仅注册 / aclnn | 禁止 `<<<>>>` 直调；host 须 `MC2().HcclGroup("group")` |
| R6 | 无 quant | 不引入 `SetQuant*` / `qbmm_mx` / int8 通信路径 |
| R7 | PTA 接入 | 用 `build.sh -n <算子目录名>` 构建，安装生成的 `.run`，设置 `ASCEND_CUSTOM_OPP_PATH` 指向已安装的 vendor 目录。由 `torch-ascendc-op-extension` **路线 B** 生成 aclnn 注册接口；运行时从 torch_npu 的 HCCL 进程组取 `get_hccl_comm_name(rank)`，将域名作为 `group` 传给 aclnn。 |

路线咨询的答复到 R1–R7 为止。实际开始设计、构建或审查算子代码时，进入 [910B MC2 开发流程](references/foundations/hccl-matmul/workflow_integration.md)，按阶段定位参考实现和技术细节。
