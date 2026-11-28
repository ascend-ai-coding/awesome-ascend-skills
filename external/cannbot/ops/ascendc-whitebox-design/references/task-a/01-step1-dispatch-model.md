# Task A Step 1：建立 Evaluator 指导书

## 职责

Step 1 是源码分发信息分析阶段。

本阶段读取限定 P0 入口源码，分析 tiling 侧 route 分支、kernel 侧 dispatch 分支、tiling-key 到 kernel 分支的映射关系，以及输入构造可控、且参与或控制 tiling 侧路径判断的代表字段，生成 `S2P1_route_dispatch_model.md` 和 `S2P1_tiling_glossary.md`。

`S2P1_route_dispatch_model.md` 用于记录 `case_values` 固定字段全集、字段语义、等价/推导关系、分支范围、分支条件、tiling/kernel 映射关系、evaluator 翻译规则和覆盖归类规则。

`S2P1_tiling_glossary.md` 用于记录进入 `case_values` 的字段语义、被排除的等价/推导变量，以及这些变量与 Step 5 输入构造约束之间的映射关系。

## 输入与输出

必须读取：

1. `S2P0_source_scope.md`
2. tiling P0 入口源码
3. kernel P0 dispatch 入口源码

禁止读取 scout、P1/P2、kernel 非 dispatch、Mapper 产物、schema 文件。

输出文件：

```text
S2P1_route_dispatch_model.md
S2P1_tiling_glossary.md
```

## 指导书内容要求

### 1. Case Values 字段集合与语义

主任务：统计 tiling 侧分支判断使用的变量，并从中筛选出具备独立输入构造自由度的变量，作为所有 route case 共用的 `case_values` 固定字段全集。

必填：

- `case_values_scope`：说明字段纳入标准、固定字段集合原则和等价/推导变量处理方式。
- `case_values_fields`：列出进入 `case_values` 的代表字段。
- `field_semantics`：说明字段的来源、输入构造语义、route 作用和到 Step 5 输入构造的映射。
- `equivalent_or_derived_variables`：记录不进入 `case_values`、但参与 route 判断的等价变量和推导变量，以及它们与代表字段或平台上下文的关系。

`case_values` 字段集合对所有 route case 固定，不按分支裁剪。

字段必须先属于 tiling 侧分支判断使用的变量，再进行筛选。只有同时满足以下条件的 tiling 侧变量才能进入 `case_values`：

- 参与或控制 tiling 侧路径判断；
- 表达独立的输入构造自由度。

当源码中存在多个 tiling 侧变量表达同一个输入构造自由度，且对 route 判断等价时，`case_values` 只保留一个变量。`case_values` 中的变量命名必须与 tiling 侧变量命名保持一致，禁止修改。不得使用输入 shape、输出状态、属性或其他上游构造字段替代 tiling 分支判断变量，即使这些字段可以间接推导出 tiling 变量。

参与路径判断但由 `case_values` 或平台上下文机械推导得到的变量，不进入 `case_values`。Evaluator 应在动态区推导这些变量，并用其完成 route 判断。

`case_values` 多样化只针对代表字段，不得通过加入等价变量、内部推导变量或不具备独立构造自由度的 shape/rank、接口层构造字段实现多样化。

通常不进入 `case_values` 的变量包括：

- 可由 `case_values` 或平台上下文机械推导出的 key、mode、branch id、block、align、factor、offset、size、loop、tail 等内部中间变量。
- 与某个 `case_values` 代表字段表达同一输入构造自由度的等价变量。
- 不具备独立构造自由度、仅作为完整输入实例组成部分的 rank、完整 materialized tensor shape、TensorList item shape、output shape 或接口层构造字段。
- 仅用于 Step 5 shape/rank 多样性、完整输入构造或生成便利的字段。
- 仅影响 tiling data、performance flag、strategy_vars、审计输出或内部优化标志的变量。
- P1/P2 深层实现或 kernel 模板内部实现变量。

不得按字段名称机械排除 rank、shape 或接口字段；如果它具备独立构造自由度并直接参与或控制 route 判断，可以作为代表字段纳入。反之，即使字段出现在源码分支中，只要它是其他字段的推导结果，也不得纳入。

本节只定义字段集合和语义，不描述具体取值范围，不规定 generator 的组织方式或取值算法。

### 2. 分支范围与条件

主任务：描述所有需要覆盖或审计的 tiling/kernel route 分支范围。

必填：

- `coverage_goals`：期望覆盖或审计的 tiling branch、kernel branch 和 path edge 范围。
- `tiling_branch_families`：tiling 侧 route 分支家族。
- `branch_priority`：if/else-if 优先级、前置否定条件和覆盖关系。
- `branch_conditions`：每类分支的成立条件、排除条件和依赖变量。
- `case_value_domains`：每个 `case_values` 字段在 route 分析中的合法范围、枚举范围或约束关系。
- `branch_value_constraints`：每类分支对 `case_values` 字段的约束条件、边界条件和前置排除条件。
- `equivalent_or_derived_conditions`：当 route 判断依赖未进入 `case_values` 的等价变量或推导变量时，说明等价关系、推导关系、来源字段和对应分支条件。
- `kernel_dispatch_scope`：kernel dispatch 的 key、dtype、模板、架构宏或平台条件范围。

分支条件表达要求：

- 对每类 `coverage_goals`，必须用自然语言准确描述其成立条件和前置排除条件。
- 若分支存在 if/else-if 优先级，必须说明该分支需要避开的更高优先级条件。
- 若某个分支是默认分支、回退分支或 fallback 分支，必须说明触发原因。
- 对每类分支，必须说明哪些 `case_values` 字段是该分支的 witness 约束字段，哪些 `case_values` 字段未被该分支直接约束但仍需合法取值。
- 若分支条件依赖未进入 `case_values` 的等价变量或推导变量，必须说明该变量由哪些 `case_values` 字段或平台上下文得到。
- 等价/推导关系只服务 evaluator 判断，不作为 `case_values` 字段、coverage goal 或多样性目标。
- 若 kernel dispatch 可达性依赖 tiling key、dtype、架构宏或平台条件，必须说明对应关系。
- 分支条件描述字段合法范围、边界和前置排除条件，但不规定 generator 的具体取值组织方式。
- 不要求逐条枚举 concrete case，但必须让 Step 2 能机械翻译 evaluator 和 case generator。
- 平台内存量（tiling 侧有效值）按 `soc_version` 固定（`ini_ub_size` 一律以**字节**计，若来源为 KB 需先 ×1024）：`ASCEND910B` 的 UB/L1 相对 ini 预留 256B（即 `GetCoreMemSize(UB) = ini_ub_size − 256B`）；`ASCEND950` 无预留（即 `GetCoreMemSize(UB) = ini_ub_size`）；其余架构不在范围，标记 unresolved。分支阈值按此有效值计算，不得使用 ini 名义值。

`case_values` 字段全集对每个分支保持不变。分支只声明本分支直接约束的 witness 字段；未被直接约束的字段仍须满足本节定义的合法域和跨字段约束。

### 3. Evaluator 翻译规则

主任务：把分支事实整理成 Step 2 可机械翻译的 evaluator 规则。

必填：

- `tiling_key_formula`：tiling key、mode、branch 或等价 route 标识的公式或来源。
- `tiling_eval_rules`：`eval_tiling_route(case)` 的判定逻辑、返回事实和 evidence 绑定方式。
- `kernel_eval_rules`：`eval_kernel_route(case, tiling_result)` 的 dispatch 方式、实例分支规则和 evidence 绑定方式。
- `kernel_branch_catalog`：完整 kernel 分支目录，用于 orphan 检测。
- `argument_binding_rules`：kernel dispatch 参数绑定规则。
- `platform_or_arch_rules`：架构宏、平台条件或编译条件对 route/kernel dispatch 的影响。

要求：

- evaluator 只能从 `case["case_values"]` 读取输入构造可控的代表字段。
- evaluator 可以在动态区计算 key、mode、align、block、factor、等价变量和其他内部推导变量。
- 内部推导变量和等价变量应进入 evaluator 返回的 strategy/audit facts，不得要求 generator 写入 `case_values`。
- kernel evaluator 只能使用 `case["case_values"]`、平台上下文和 `eval_tiling_route` 的返回事实。
- Task A 不构造真实 tensor，不校验完整算子实例，不分析 kernel 模板内部实现。

### 4. 覆盖与归类规则

主任务：定义 Step 3 执行结果的审计口径。

必填：

- `unsupported_sources`：非法输入、平台排除、架构排除或不支持场景如何归类。
- `unresolved_sources`：无法判定的情况。
- `tiling_only_rules`：tiling 可生成但无可达 kernel dispatch 的情况如何归类。
- `kernel_orphan_rules`：kernel catalog 中未被 route case 命中的分支如何归类。
- `dedup_rules`：tiling/kernel/path 事实去重 key。
- `evidence_rules`：证据绑定方式。

要求：

- `coverage_goals` 描述覆盖范围，不枚举最终 path list。
- 每类 coverage goal 必须能对应到明确分支条件。
- tiling-only、kernel orphan、unsupported、unresolved 的归类规则必须明确。
- 覆盖审计基于 tiling/kernel/path 事实；`case_values` 用于表达 route witness 的输入构造可控自由度，不表达完整输入实例。
- 不把 coverage goal 写成 generator 的具体取值方案。

## Tiling 变量语义表内容要求（S2P1_tiling_glossary.md）

独立文件，解释进入 `case_values` 的字段、被排除的等价/推导变量，以及 route 场景对应的 Step 5 输入构造约束。

必填两个章节：

### `variables`

- 覆盖进入 `case_values` 的输入构造可控代表字段。
- 覆盖未进入 `case_values`、但与路径判断有关的等价变量或推导变量。
- 每个变量需说明：原 tiling 变量名或语义名、来源表达式、类别、是否进入 `case_values`、参与或控制的 tiling 侧路径条件。
- 标注变量类别：case value representative、equivalent variable、derived route variable、platform context、internal audit only。
- 不记录仅影响 tiling data、performance flag、strategy_vars、审计输出或内部优化标志的变量，除非它们是解释 route 判断不可缺少的推导事实。

### `route_shape_constraints`

- 记录 route 场景到 Step 5 输入构造约束的映射。
- 记录 rank、完整 shape、TensorList item shape、output shape 和 optional input/output 构造信息；其中具备独立构造自由度并满足字段纳入标准的外部字段可进入 `case_values`，其余作为 Step 5 约束记录。
- 每条约束必须绑定到相关 `case_values` 字段、等价/推导条件、tiling route 分支或 kernel route 分支。
- 每条约束需说明：场景判别条件、输入构造所需约束、output/optional output 规则、dtype 约束、Step 5 消费方式。

## 禁止事项

- 不枚举最终 path list。
- 不粘贴完整 schema。
- 不分析 P1/P2 深层实现。
- 不分析 kernel 模板内部实现。
- 不因字段名称属于 rank、shape、TensorList item shape、output shape 或接口层构造信息就机械纳入或排除 `case_values`；必须依据字段纳入标准判断。
- 不为了 Step 5 shape/rank 覆盖、完整输入构造或生成便利，加入不具备独立构造自由度的字段。
- 不把等价变量、内部推导变量作为 `case_values` 字段、coverage goal 或多样性目标。
- 不在 `case_values` 中重复表达同一个输入构造自由度。
- 不按目标分支裁剪 `case_values` 字段集合。
- 不记录仅影响 tiling data、performance flag、strategy_vars、审计输出或内部优化标志的变量。
- 不规定 generator 的固定函数结构、固定 seed pool、固定 variants 组织方式或具体取值策略。

## 完成标准

1. 已读取 source scope 和 P0 入口源码。
2. 已生成 `S2P1_route_dispatch_model.md`。
3. 已生成 `S2P1_tiling_glossary.md`。
4. 指导书包含 Case Values 字段集合与语义、分支范围与条件、Evaluator 翻译规则、覆盖与归类规则四个小节。
5. `case_values` 固定字段全集只包含输入构造可控、且参与或控制 tiling 侧路径判断的代表字段。
6. 每个 `case_values` 字段均说明来源、语义、参与或控制的路径条件和 Step 5 映射方式。
7. 每个 `case_values` 字段的合法范围、边界条件和分支约束已在分支范围与条件章节说明。
8. 等价变量和由 `case_values` 或平台上下文机械推导的变量未进入 `case_values`。
9. 每类 coverage goal 均能对应到明确分支成立条件、前置排除条件、`case_values` witness 字段或等价/推导条件。
10. Evaluator 规则足以让 Step 2 翻译为可执行代码，且只依赖 `case_values`、平台上下文和 evaluator 内部推导变量。
11. Glossary 包含 `variables` 和 `route_shape_constraints` 两个章节。
12. Step 1 只定义字段全集、字段语义、变量合法域、等价/推导关系和 route/shape 映射事实，不规定 generator 的具体取值组织方式。
