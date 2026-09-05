# 合同设计与验证看护

## 目录

1. G1 目标、身份与事实源
2. 设计结论摘要
3. 接口合同与支持域
4. 源执行模型与活动闭包
5. 迁移范围矩阵
6. 内部执行合同
7. API 能力与组件证据
8. 不变量与允许差异
9. G1 验证义务设计
10. G3 具体用例与 runner
11. 精度、结果与路由合同
12. 性能计划
13. 构建、审查与交付计划
14. 设计冻结与变更控制
15. G1/G3 门禁与关闭条件

## 1. G1 目标、身份与事实源

G1 将公开接口合同、源活动行为和迁移要求转化为 `migration-design.md` 的迁移设计章节。G2 必须能够只依赖该章节、固定代码仓和本节指定的证据完成迁移，不重新调查关键源语义。具体 case、输入、runner 和原始实测在 G3 依据 G1 义务实例化。

在 G1 设计末尾写入稳定边界标记 `<!-- G1_MIGRATION_DESIGN_END -->`。G1 关闭时计算从文件开头到该标记（含标记）的 SHA256；G3 只能在标记之后追加验证设计，并在开始和关闭时复核 G1 分段 SHA256 不变。这样既允许同一 design 文档后续补充逐 case 表，又能证明迁移输入未被反向改写。

`migration-design.md` 开头必须记录：

- task ID、目标算子、目标场景和目标平台；
- G0 冻结的 original 仓、ops-tensor、submodule 和 third-party 身份；
- 对外接口文档路径、版本和 SHA256；
- `environment-state.json` revision 和 SHA256；
- 设计阶段使用的 checkout、工具和时间；
- 设计版本、作者代理和变更原因。

事实源优先级：

1. 目标平台和场景的算子对外接口文档；
2. 活动 ACLNN 接口、Host Tiling、checker、binary config、TPL_SEL、Kernel 和现有测试；
3. 固定实际解析 SHA 的 ops-tensor checkout 源码与 `CODING_CONVENTIONS.md`；不得跟踪获取后的移动分支；
4. 其他官方资料。

代码用于验证文档、恢复路由和发现不一致，不能无证据地把当前实现限制升级为公开接口限制。冲突必须列出来源、证据、影响和处理结论。G1 只读 `environment-state.json`，环境事实不能改变接口合同或缩小义务。

## 2. 设计结论摘要

设计文档必须用表格明确写出：

| 项目 | 必须回答 |
|---|---|
| 迁移目标 | 哪个入口、哪个 Kernel、哪个平台和哪个场景 |
| 默认保持 | Tiling、TilingKey、ABI、workspace、blockDim、任务集合、支持域、数学语义和性能合同 |
| 允许保留 | ABI、核索引、AIC/AIV 分支、硬件同步和设备模式控制 |
| 必须迁移 | Copy、Buffer、Tensor、layout、Matmul、资源生命周期和可表达计算 |
| 禁止变化 | 不得扩大或缩小公开支持域，不得改变 transpose、layout、数学或性能语义 |
| 当前阻塞项 | 所有 `unknown`、`当前不可表达` 和未解决的来源冲突 |

结论必须能直接指导 G2，不得只写“后续调查”或“参考现有实现”。

## 3. 接口合同与支持域

分别记录三类事实：

| 类别 | 含义 | G1 设计影响 |
|---|---|---|
| 支持约束 | 合法/非法输入和行为合同 | 形成验证义务，不在 G1 直接生成 case |
| 性能建议 | 推荐 shape、对齐或配置 | 形成性能义务，不能删除合法功能义务 |
| 实现路由 | TilingKey、模板或 fallback 选择 | 形成路由观察义务 |

支持域至少覆盖：

- dtype、format、layout、transpose、shape、rank、广播、batch 和 group；
- 属性组合、量化模式、bias/scale/offset 和可选参数；
- 对齐、非对齐、边界、tail、空 Tensor 和特殊值；
- 逻辑 shape、物理 storage shape、stride、offset、packing 和地址解释；
- 目标路由、合法 fallback、非法输入的拒绝阶段与行为；
- 平台、版本、资源和错误处理限制。

每个接口事实使用稳定的 `contract_id`，绑定文档条款、代码证据、结论和至少一条 G1 验证义务。文档声明支持但 G4 原始实测失败时保留支持事实和证据，调查 runner、输入、环境、原始实现或文档实现不一致；不得删除、改为非法或伪造通过。

## 4. 源执行模型与活动闭包

从目标活动入口恢复真实执行模型，不按文件名、CMCT/CGMCT 外形或相似组件套模板。填写特征矩阵：

```text
hasFrameworkLayers
computeKind: cube | vector | mix | other
usesHighLevelMatmul
usesOtherHighLevelComputeApi
usesManualBufferOrCopy
usesLowLevelCopyApi
usesLowLevelTensorApi
usesLegacyBufferApi
usesLowLevelMatmulApi
usesManualLayoutOrPacking
usesCustomSchedule
isAicAivMix
hasPreMainComputeVf
hasPostMainComputeVf
hasCustomSync
```

设计文档必须说明活动路径、合法 fallback、不可达代码、包装层、别名和编译开关的边界，并记录：

- ACLNN、Host Tiling、TilingKey、blockDim、workspace 和 Kernel ABI；
- `GM_ADDR`、ListTensor、指针解引用、batch/group 偏移和地址单位；
- block/batch/group/M/N/K 遍历、边界、tail 和任务覆盖；
- GM/L1/L0/UB 数据流、layout、stride、gap、padding、packing 和 transpose；
- Buffer 分组、数量来源、选择、生产消费、复用和生命周期；
- VF 的 mask/repeat/stride、dtype、舍入、饱和和中间 Buffer；
- 核内/跨核同步的方向、位置、次数、初始化、稳态和排空。

可能由模板、宏或分发选择的路径必须完成可达性证明后才能排除。每个活动行为使用稳定的 `behavior_id`，绑定源文件、调用路径和证据。

## 5. 迁移范围矩阵

为每个活动计算、搬运、Tensor、Buffer、layout、生命周期和同步行为建立一行：

| 字段 | 内容 |
|---|---|
| behavior_id | 稳定行为 ID |
| source_evidence | 源文件、入口、调用路径和文档证据 |
| source_semantics | 原始行为及可观察效果 |
| owner | Kernel、Scheduler、Storage、Tile 等 |
| migration_status | 五种中文状态之一 |
| target_api/component | Tensor API 或 Blaze 组件 |
| support_range | 支持范围 |
| reject_range | 拒绝范围 |
| resource_contract | 数量、选择、布局、生命周期和同步 |
| difference_class | 差异分类 |
| verification_obligations | G1 验证义务 ID |
| proof | 需要的代码、编译、功能或性能证据 |

状态只允许：

| 状态 | 判定 |
|---|---|
| `已有 Tensor API` | 固定 checkout 存在满足完整语义和支持域的公开 Tensor API |
| `已有 Blaze 组件` | concrete witness 的职责、签名、资源和约束与目标合同匹配 |
| `需要扩展 Blaze` | 可在冻结合同内于正确层级新增或扩展组件 |
| `运行时边界` | 仅 ABI、核索引、AIC/AIV 分支、硬件同步或设备模式无法由公开抽象替代 |
| `当前不可表达` | 固定能力无法表达且无法合理扩展，G1 必须 `blocked` |

`GetBlockIdx`、`GetBlockNum`、`GetTaskRation`、`SetFlag`、`WaitFlag`、`CrossCoreSetFlag`、`CrossCoreWaitFlag`、`SyncAll`、HF32 设置、`ASCEND_IS_AIC/AIV` 和 ABI 入口可以属于运行时边界，但必须绑定 owner、用途和范围矩阵条目。

Copy、DataCopyPad、Tensor view、Buffer 分配、packed 地址换算和矩阵计算不属于运行时边界。现有能力不足时使用 `需要扩展 Blaze`，不能直接保留低阶 AscendC API。

## 6. 内部执行合同

每个资源组必须冻结：

| 维度 | 必答问题 |
|---|---|
| 职责 | 资源服务哪个执行阶段和数据流 |
| 数量 | 来源字段、合法域、编译期/运行时决定方式 |
| 选择 | 每个生产者和消费者如何选择同一实例 |
| 布局 | shape、stride、gap、offset、packing、对齐和单位 |
| 生命周期 | 分配、首次生产、稳定复用、最后消费和释放 |
| 参数映射 | 源字段如何到达最终地址、索引、循环、资源选择或同步 |
| 同步 | 事件方向、次数、位置、初始状态和排空 |

职责、数量、选择、生产消费、布局或生命周期任一不同的资源默认独立。Buffer 数、轮转、并发度和同步是实现语义，不能为获得通过而静默减少、合并、降并发或删除同步。保持源 Set/Wait 协议，不得使用 LOCK 机械替换。

## 7. API 能力与组件证据

每个 Tensor API 或 Blaze 组件必须记录名称、固定 checkout 路径、concrete witness、支持/拒绝范围、参数化轴、资源与同步约束、语义匹配、正向实例化、负向实例化或编译期拒绝，以及低阶实现是否封装且未泄漏。

不支持的组件组合必须通过编译期约束或 `static_assert` 拒绝。现有组件不足时记录为 `需要扩展 Blaze`，不得把低阶 API 直接留在 Kernel、Block 或 Scheduler 编排层。

## 8. 不变量与允许差异

设计文档必须冻结：

- 主机 Tiling、TilingKey、ABI、workspace 和 blockDim；
- 任务集合、支持域和数学语义；
- transpose、layout、地址单位和物理 storage 语义；
- Buffer 数量、轮转、并发和同步协议；
- VF、转换、舍入、饱和和累加顺序；
- 功能、性能和错误处理合同。

差异分类只用：`STRUCTURAL_MOVE`、`API_MECHANICAL`、`TENSOR_API_REFACTOR`、`BLAZE_COMPONENT_EXTENSION`、`RUNTIME_BOUNDARY`、`MATMUL_EXPANSION`、`FRAMEWORK_GLUE`、`INACTIVE_REMOVAL`、`IMPLEMENTATION_DRIFT`。

每个计划差异必须记录：

```text
difference_class
→ intentional_change
→ 必要性
→ 影响
→ G1 验证义务
→ 功能/性能证据计划
→ review 证据
```

没有完整差异证据时默认恢复源行为。

## 9. G1 验证义务设计

G1 不生成具体 case，而是定义 G3 必须实例化的义务。覆盖设计遵循[用例覆盖设计](case-coverage-design.md)，至少形成：

| obligation_id | contract_id/behavior_id | 属性轴 | Shape 行为 | 对齐条件 | 路由/拒绝 | 必须观察的源行为 | 功能/性能属性 | 压缩或排除依据 |
|---|---|---|---|---|---|---|---|---|
| `obligation_<稳定ID>` | 明确 ID | 明确轴 | 明确行为断点 | 明确条件 | 明确路由 | 明确行为 | 功能、性能或非法 | 证据或 `N/A` 原因 |

义务必须覆盖合法/非法属性、Shape 行为、对齐/合法非对齐、tile/group/tail、路由、资源生命周期、内部行为和性能建议。每条义务至少指定一个未来具体 case；只有对外合同明确不可构造且有拒绝证据时才允许审计排除。数量、平均分布或少量样例不能替代义务覆盖。

G1 同时冻结功能判定（G4 original 稳定输出、G5 逐字节比较）、性能判定（G4/G5 逐例 msprof 比较）以及非法拒绝判定，但不生成其实际结果。

## 10. G3 具体用例与 runner

G3 才将 G1 义务实例化为具体用例。正式功能性能用例数量为 10~30；每个正式用例同时定义功能和性能要求。非法拒绝等辅助用例单独标记，不计入 10~30，除非接口合同明确要求其性能属性。

G3 的 `migration-design.md` 追加以下表格，不能只链接外部文件：

### 10.1 冻结用例总表

| case_id | case_kind | obligation_id | contract_id | behavior_id | 文档条款/覆盖理由 | dtype/format | 逻辑/物理 shape | layout/transpose | 属性与可选参数 | 边界/tail/Buffer | 预期路由或拒绝阶段 | 输出与动态元数据 | 逐字节规则 | guard/repeat | 功能要求 | 性能要求/门限 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `case_<稳定ID>` | 正式功能性能/非法辅助 | 明确 ID | 明确 ID | 明确 ID | 具体条款和不可替代义务 | 明确值 | 明确值 | 明确值 | 完整元组 | 明确类别 | 明确值 | 输出名、长度、动态元数据、inplace | original/Blaze 有效字节范围 | 可控检查和重复次数 | 必须执行的功能检查 | 是否 msprof、门限来源 |

### 10.2 输入与执行明细表

| case_id | Tensor/参数顺序 | dtype/format | logical/storage shape | stride/offset/packing | 属性值 | 输入生成规则 | seed/边界值 | 输入资产与 SHA256 | 预期路由 | 输出有效字节与元数据 | after-state | 执行与重置规则 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `case_<稳定ID>` | 完整接口顺序 | 每项明确 | 每项明确 | 每项明确 | 全部明确 | 分布或构造方式 | 明确值 | G3 路径及 SHA256 | 明确值 | 明确范围 | 明确对象 | warm-up、repeat、每轮重置 |

G3 冻结前必须补齐最终 Shape、seed、生成器版本、重抽记录、输入路径和 SHA256。不同输入、属性、路由或预期行为不得合并在一行。

runner 必须读取 G3 冻结的具体用例和输入资产，每次执行一个 `case_id`，支持 `original` 与 `blaze` 两个 role，并自动核对 obligation、case、runner 注册、输入、执行清单和实际结果集合完全相等。具体 Tensor 名称和数量由目标接口决定，本文件不预设算子字段。详细编码规则读取[ACLNN Runner 开发](aclnn-runner-generation.md)。

## 11. 精度、结果与路由合同

每条 case 只生成一次输入，G4/G5 读取相同文件。输入记录 Tensor 标识、dtype、format、逻辑/物理 shape、stride、offset、字节数、SHA256、随机种子、分布和边界规则。ACLNN 内部 padding、workspace 和 storage 对 runner 不透明；只检查 runner 可控制的 host buffer、输出初始化、文件长度、元数据、inplace after-state 和重复稳定性。

| 类型 | 规则 |
|---|---|
| original 稳定性 | 真实执行、输出完整、重复有效输出逐字节稳定；不要求 CPU golden |
| Blaze 功能验收 | original 与 Blaze 有效输出、必要动态元数据和 inplace after-state 逐字节一致 |
| 特殊位模式 | 不归一化 `+0/-0`、NaN payload 或 Inf |
| 动态元数据 | shape、length、index、count 单独比较 |
| 非法行为 | 比较拒绝阶段和行为 |

`close`、`rtol`、`atol`、平均误差和外部 golden 只能作为失败诊断字段，不能生成 `PASS`。

## 12. 性能计划

迁移 skill 只维护性能义务、case、身份、可比性和门限，不维护 msprof 命令、字段解释或瓶颈分析。G4/G5 正式性能测试必须调用 `ops-profiling`，使用相同 device、CANN、SoC、release、输入、warm-up、repeat 和逐例统计协议。

G1 记录性能义务和门限来源；G3 将其绑定到具体 case；G4 固定 original 原始数据；G5 固定 Blaze 原始数据并逐例比较。性能回退取证链为：

```text
case → ops-profiling 原始数据 → Kernel/package/runner 身份
→ behavior_id/内部执行合同 → 差异分类 → 性能结论
```

## 13. 构建、审查与交付计划

G1 冻结构建和审查计划：G3 使用相同公共选项分别构建 original/Blaze 正式 OPP，隔离 package 和加载环境；G2 完成迁移范围、反模式、组件和内部合同检查；G5 完成 original→Blaze 和 Blaze→original 双向行为映射、Convention、CMCT/CGMCT 和最终 review；G6 只创建本地提交。

## 14. 设计冻结与变更控制

G1 归档后冻结代码、文档、环境、支持域、迁移范围、行为模型、内部合同、验证义务、判定规则、性能门限、构建和 review 计划。G2/G3/G4/G5 不得改写这些事实。

G3 具体用例、runner、输入、逐字节规则和执行清单冻结后，G4/G5 不得删除、跳过、替换、缩减、重命名或重分类。发现具体验证设计错误返回 G3；发现支持域或源行为错误返回 G1；实现问题返回 G2。

## 15. G1/G3 门禁与关闭条件

### G1.1 合同与源执行模型恢复

- 对外接口文档逐条调查；
- 支持约束、性能建议和路由分离；
- 活动闭包、源行为和不变量完整；
- 每个活动行为有 `behavior_id`、证据和迁移状态；
- 没有未解释的 `unknown` 或 `当前不可表达`。

### G1.2 迁移范围与验证义务设计

- 所有行为都有 owner、API/组件和资源合同；
- 特征约束、合法/非法属性组合、Shape 行为类别、关键断点和对齐/合法非对齐分析完整；
- 所有验证义务都有合同/行为依据，压缩和排除都有可审计证据；
- 功能、性能、非法拒绝、路由和逐字节判定完整；
- 构建、反模式、review 和交付计划完整；
- 不要求具体 case、输入、runner 或测试结果。

### G1.3 设计自审与合同冻结

- `migration-design.md` 的 G1 必需章节完整；
- 迁移范围矩阵、行为映射、内部合同和验证义务无未解释缺口；
- 设计自审无未关闭 finding；
- G1 边界标记和分段 SHA256 已写入 record。

### G3 门禁

- G1 每条验证义务已绑定具体 case 或审计排除项；
- 正式功能性能用例数量为 10~30，辅助非法用例单独标记；
- 每个正式用例有功能要求、性能要求、输入、路由、输出和逐字节规则；
- G3 具体用例表、输入表、覆盖映射、case 资产、runner 注册和执行清单集合完全相等且唯一；
- 两套正式 OPP、Kernel、runner、manifest、命令、日志和 SHA256 完整；
- G1 分段 SHA256 与关闭 G1 时一致，G3 验证章节和最终整文件 SHA256 已固定；
- 用例没有因 Blaze 当前实现或试运行结果被删改；
- G3 不得以功能或性能结果提前声明迁移通过。

满足后分别自动进入 G2、G4，不等待批准。
