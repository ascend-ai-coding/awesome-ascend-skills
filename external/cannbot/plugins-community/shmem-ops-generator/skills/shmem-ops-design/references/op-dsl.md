# SHMEM Operator DSL 说明

`shmem-op-dsl.yaml` 用来把用户需求、伪代码或参考实现归一化成统一的 SHMEM 算子设计描述，供 `shmem-ops-code-gen` 直接消费。DSL 的目标不是复述用户原话，而是把算子的接口、语义、通信、内存、同步、调度、正确性和性能契约写成可实现、可测试、可交接的结构化描述。

## 1. 设计原则

| 原则 | 要求 |
| --- | --- |
| 语义和调度分离 | `semantics` 描述算子“做什么”，`schedule` 描述算子“怎么执行得更好”；调度不能改变功能语义 |
| SHMEM 关键信息显式化 | 拓扑关系、peer 可见 buffer、同步原语、关键顺序约束不能省略 |
| 缺失信息先处理 | 关键语义缺失时先向用户提问；非关键或可安全推断的信息写入 `assumptions` |
| 面向生成 | DSL 里的字段必须能指导后续代码生成、测试和性能采集，不能只有 prose |
| 设计前确认 | 开始设计前必须先和用户确认 `op_name` 和 dtype/dtypes；推断值也必须回显确认 |

## 2. 顶层字段

| 字段 | 含义 | 最小内容 |
| --- | --- | --- |
| `meta` | 算子基本信息 | `op_name`、`op_kind`、`target_soc`、`scope` |
| `source` | 需求来源和归一化输入 | `kind`、`summary`、`user_confirmations`、`references` |
| `interface` | 输入、输出、属性和 shape 关系 | `inputs`、`outputs`、`attrs`、`shape_relations` |
| `semantics` | 功能语义 | `local_compute`、`communication`、`finalize`、`result` |
| `topology` | 通信域和 peer 关系 | `group`、`peer_model`、`addressing` |
| `memory` | 关键 buffer 和可见性 | 本地中间 buffer、symmetric/peer-visible buffer、输出 buffer |
| `schedule` | 分核、tile/chunk 和 phase | `core_partition`（含分核模式/overlap/同步/cross_pe_dataplane）、`tiling`、`phases`（仅 name+action） |
| `correctness` | 正确性参考和误差要求 | `oracle`、`tolerance`、`invariants`（结构体列表，每项含 invariant/test_method/case）、`case_matrix`（含 stress/边界 case） |
| `performance` | 性能目标和采集契约 | `metric`、`baseline`、`baseline_search`、`baseline_target`、`target_cases`、`min_scale`、`max_opt_rounds` |

## 3. 字段填写要求

| 字段 | 填写规则 |
| --- | --- |
| `meta.op_name` | 使用能落地为目录名和符号前缀的名称；必须来自用户确认，不能只靠推断 |
| `meta.torch_required` | **须 Phase 0 用户确认**；纯 C++ 交付设为 `false`，Phase 5.5 跳过；Agent 不得擅自默认 `true` |
| `meta.performance_required` | **须 Phase 0 AskQuestion #4**；`false` 时 Phase 6 跳过；Agent 不得擅自默认 `true` |
| `meta.performance_auto_optim` | **须 Phase 0 AskQuestion #5**；`true` 且 Phase 6 未达标时 **自动** Phase 6.5；`false` 时仅报告差距、禁止改 kernel |
| `meta.op_kind` | 取 `transport`、`collective`、`compute`、`fused_compute_comm`；无法判断时先提问。判定：`transport`=点对点 put/get；`collective`=多 PE 集合语义；`compute`=Device 本地计算、无跨 PE 通信；`fused_compute_comm`=Cube matmul/GEMM + SHMEM 跨 PE 通信（CoC，见 core-allocation §4） |
| `source.user_confirmations` | 记录用户确认的 `op_name` 和 dtype/dtypes；未确认时不得进入下游 |
| `interface.inputs/outputs` | 每个输入输出写清 `name`、`dtype`、`shape`、`role`；dtype 必须来自用户确认；输出还要写 `visibility` |
| `interface.shape_relations` | 写出跨输入输出的 shape 约束，例如 `A[1] == B[0]` |
| `semantics` | 推荐固定为 `local_compute`、`communication`、`finalize`、`result` 四行；**使用中文描述** |
| `topology` | 默认不能随意写 `all_pes`；只有用户需求或参考实现支持时才写入，否则提问或写入假设 |
| `topology.intra_node_topology` | 取值统一为 `full-mesh` / `switch` / `ring`；用户输入 `fullmesh` 时归一化为 `full-mesh` |
| `memory.buffers` | 不要求列出所有临时变量，但必须写出本地中间结果、peer-visible symmetric buffer、signal/state buffer |
| `schedule.core_partition` | **最先设计**。写明分核模式（引用 core-allocation.md 的模式 A/B/C）、每组核的编号范围和职责、使用的数据接口（`cross_pe_dataplane` 必须是 `aclshmem_*`/`aclshmemx_*`）、组间 overlap 方式、组间同步方式（cross-core flag / barrier / signal-wait） |
| `schedule.tiling` | **分核之后设计**。写明每组核的 tile、chunk、tail 处理方式（可以是符号公式，但要能指导后续 tiling 实现） |
| `schedule.phases` | **最后设计**。写 device 侧的大体执行流程，每个 phase 仅需 `name` 和 `action`（一句话中文描述该阶段做什么）；不需要写 input/output/dependency/sync 等微观细节 |
| `schedule.compute` | **仅 `fused_compute_comm` 必填**。`compute_path`：`device_catlass_coc`（默认）或 `host_aclnn`（过渡，**MUST** 附 `upgrade_plan` 与 `rationale`）；**禁止** `device_scalar_triple_loop`；见 [fused-compute-phase-gates.md §4](../../shmem-ops-dev/references/fused-compute-phase-gates.md) |
| `correctness` | 写 oracle、dtype tolerance、关键 invariants；invariants 为结构体列表，每项含 `invariant`（**使用中文描述**）、`test_method`（rank pattern / CPU golden / file golden / intermediate check）、`case`（适用 case 范围）；`case_matrix` 统一存放功能 case 和 stress/边界 case（boundary shape、tail、非均衡分片、signal 复用等），每项标注 `category`（functional / stress）；无法构造 oracle 时先提问 |
| `performance` | 写主指标、baseline、target cases 和优化轮数；`baseline_search` 记录已检查的 HCCL/aclnn 方案（无直接 baseline 时必填，禁止留空或写 “none”）；`baseline_target` 写达标阈值（有 baseline 时默认 `current >= 80% baseline`，无 baseline 时写 metric_only 指标目标，通信算子能计算带宽利用率时不低于 20%）；`min_scale` 写最小测试规模（集合通信 >= 256MB，计算/通算融合 hidden size >= 1000，无法满足时说明原因）；默认 `max_opt_rounds` 为 `5`，且不得超过 5 |
| `语言使用` | **描述性字段（semantics、schedule.phases.action、correctness.invariants 等）使用中文**；API 名称、数据类型、字段名、代码标识符使用英语 |

## 4. 缺失信息处理

信息缺失时按严重程度处理。关键语义缺失时必须向用户提问；只有非关键、可安全推断或不会影响实现正确性的内容，才可以写入 `assumptions`。

| 缺失项 | 处理方式 |
| --- | --- |
| 算子名称未确认 | 向用户提问；即使可从上下文推断，也必须回显并确认，不能用 `replace_me` 或未确认名称进入下游 |
| dtype/dtypes 未确认 | 向用户提问；即使参考实现能唯一确定，也必须回显并确认，不能写入 assumptions 后进入下游 |
| 输入输出、shape 不明确 | 向用户提问；如果参考实现能唯一确定，可写入 design.md Section 1.3 Assumptions 并说明来源 |
| local compute / communication / result 任一语义缺失 | 向用户提问，这是阻塞项 |
| 输出可见性不明确 | 向用户提问，需要明确 `replicated`、`sharded` 或 `owner_only` |
| topology、PE 范围、team/subgroup 不明确 | 向用户提问；只有明确是全体 PE 时才写 `all_pes` |
| correctness oracle 缺失 | 向用户提问，或把 oracle 设计列为 gap 并阻塞 code-gen |
| 性能 baseline 缺失 | 写入 `baseline: "metric_only"` 并在 `baseline_search` 记录搜索过程；**禁止** `"none"`。若用户要求与 HCCL/aclnn 对比但 baseline 未确定，先提问。存在 HCCL 等价原语（AllToAll/AllReduce 等）时 除非用户要求，否则**禁止** metric_only，**MUST** 选 HCCL baseline |
| target SoC 缺失 | 可默认 `ascend910b`，但必须写入 design.md Section 1.3 Assumptions |

提问时只问阻塞问题，优先一次问 1 到 3 个最关键问题。用户回答后再补全 DSL；不要把关键问题长期留在 design.md 的 Open Questions 中后继续进入代码生成。

## 5. 归一化规则

| 输入形态 | 归一化方法 |
| --- | --- |
| 自然语言 | 抽取输入输出、本地计算、跨 PE 通信、最终谁拿结果 |
| 用户伪代码 | 提升成 SHMEM 语义，不机械照抄临时变量和 API 名 |
| CUDA/NCCL/HCCL/NVSHMEM 参考 | 抽取语义和调度意图，框架特有 API 留在 `source.references` 或 `schedule.notes` |
| 现有 SHMEM example | 抽取可复用模块、buffer layout、同步规则和分核策略，写入 `schedule`；复用判断写入 Capability Mapping 表 |

常见语义映射：

| 参考表达 | DSL 表达 |
| --- | --- |
| send/recv | put/get 或 exchange |
| reduce | reduce、allreduce 或 reduce-scatter |
| allgather | gather from all PE into replicated output |
| ready/done/flag | signal、wait、barrier、quiet 或 fence |
| rank/local_rank | PE、team PE 或 subgroup PE |

## 6. 反模式

| 反模式 | 问题 |
| --- | --- |
| 在 `semantics` 里直接写 tile 大小、核数 | 把调度写进语义，后续无法安全调整性能 |
| 不写输出可见性 | 下游不知道输出是 replicated、sharded 还是 owner-only |
| 用 `replace_me` 占关键字段 | 会把阻塞问题传给 code-gen，导致实现阶段猜测 |
| 把缺失关键语义写进 design.md Assumptions | assumptions 只能保存可接受假设，不能替代用户确认 |
| `schedule` 中先写 phases 后写 core_partition | 正确顺序是先分核、再 tiling、最后 phases |
| `phases` 中写微观细节（input/output/dependency/sync） | phases 只需 name + action 一句话，微观行为由 core_partition 和 tiling 决定 |

## 7. 最小可交接 DSL

一份可进入下游实现阶段的 DSL 至少包含：接口定义、本地计算语义、通信语义、输出语义、拓扑、关键 buffer、分核策略（core_partition）、tiling、phase 流程、正确性参考（含 stress case）和性能 baseline。

完整 YAML 模板见 [design-template.md](design-template.md) section 2。填写时按本文 section 2-6 的字段要求逐项检查。
