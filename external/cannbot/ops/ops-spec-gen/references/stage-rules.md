# 11-stage L0 校验规则详解

> 本文件是 `SKILL.md §4` 的详细参考。SKILL.md 只保留 stage 概述表，完整子规则、numpy 子集 API、代码示例在此处。

---

## stage 1 — schema_static

按 `schemas/op-spec.json`（Draft 2020-12）逐字段校验：必填、类型、enum、pattern、additionalProperties。

---

## stage 2 — category_paradigm_consistency

校验 category↔paradigm 必含映射 + 4 套白名单 + paradigm 内部约束：

| 子规则 | 检查内容 |
|---|---|
| `required_paradigm_missing` | category → 必含 paradigm 是否齐全 |
| `fused_composite_basics` | category=FusedComposite ⇒ ≥ 2 条基础 paradigm（Elementwise 被其他范式吸收，不计入基础范式计数） |
| `mutually_exclusive` | ScatterUpdate 与 AtomicUpdate 不得共存 |
| `paradigm_constraint.numerical_stable` | NumericalStable ⇒ `numerical_stability.required: true` |
| `paradigm_constraint.fused_composite_*` | FusedComposite ⇒ composition 必填、primitives ≥ 2、op 在白名单、中间不泄漏、dataflow 闭合 |
| `paradigm_constraint.reduction_axis_missing` | Reduction（无 Recurrence）⇒ 必有 axis/dim/axes 输入或属性，或 `reduction.axis_source: fixed`，或 `reduction.axis_source: implicit_all` |
| `paradigm_constraint.argreduce_dtype` | ArgReduce ⇒ 输出 `dtype_rule` 经 numpy_expr 求值得到 `int32` 或 `int64`（典型：`y.dtype = np.int64`） |
| `paradigm_constraint.stateful_state_or_inplace` | Stateful ⇒ 至少一项 input.role=state 或 output.aliasing=inplace_with(...) |
| `paradigm_constraint.quantization_attrs` | Quantization ⇒ scale + zero_point |
| `paradigm_constraint.variable_output_flag` | VariableOutput ⇒ outputs[].data_dependent_shape: true |
| `paradigm_constraint.random_sampling_seed` | RandomSampling ⇒ seed 属性 |
| `paradigm_constraint.collective_hccl` | CollectiveCommunication ⇒ op.platform_constraints.requires_hccl: true |
| `invariant_kind_resolved.*` | invariants[].kind 在白名单 + required_fields + tolerance_inherit 约束（structural 必为 false） |
| `machine_check_kind_unknown` | boundary/extreme 的 machine_check.kind 在白名单 |
| `error_type_unknown` | raises_error.error_type 在错误语义类别枚举内 |
| `synthesize_pattern_unknown` | extreme_inputs.synthesize.patterns[].pattern 在白名单 |
| `synthesize_legacy_format` | 旧顶层 `pattern:` 写法 ⇒ WARN |
| `composition_without_fused_composite` | composition 存在但 paradigms 不含 FusedComposite ⇒ WARN |
| `paradigm_groups.fusion_min_basic` | fusion 组必须含 ≥ 2 条基础范式（Elementwise 不计入） |
| `paradigm_groups.combination_missing_switch` | combination 组必须有 switch 字段 |
| `paradigm_groups.combination_missing_when` | combination 组必须有 when 字段 |
| `paradigm_groups.mismatch` | paradigm_groups 中的 paradigms 必须都在 op.paradigms 中 |
| `paradigm_groups.combination_should_exist` | op.paradigms 含 ≥ 2 个基础范式（排除 NumericalStable / FusedComposite 等修饰符范式）但 paradigm_groups 缺失时 ⇒ WARN；若同时存在 `string_in` / `enum_in` / `int_in_range(lower=0)` 属性，消息会额外指出"强烈暗示横向组合" |
| `paradigm_groups.mode_switch_may_need_combination` | 仅 1 个基础范式但存在模式切换属性且其值包含"直通"语义 ⇒ WARN；提示该值可能使算子退化为 Elementwise，建议补范式 + paradigm_groups。模式切换属性检测范围：`string_in` / `enum_in`（值含 `none` / `identity` / `passthrough`）或 `int_in_range` 且 `lower_inclusive=0`（值 0 通常对应 none/identity 模式） |
| `format_variants.reduction_axes_negative` | format_variants[].reduction_axes 含负值 ⇒ ERR |
| `format_variants.reduction_axes_out_of_rank` | format_variants[].reduction_axes 中某值 ≥ variant.rank ⇒ ERR |
| `format_variants.oracle_kwargs_dim_mismatch` | format_variants[].oracle_kwargs.dim 与 reduction_axes 不一致 ⇒ WARN |
| `interface_contract.tensor_list_length_missing` | role=tensor_list 必须声明 list_length |
| `interface_contract.list_length_non_list` | 非 tensor_list 接口不得声明 list_length |
| `interface_contract.list_length_negative` | fixed/range 的列表长度必须非负 |
| `interface_contract.list_length_invalid_range` | list_length.range 必须满足 min <= max |
| `interface_contract.list_length_unknown_ref` | list_length.same_as 引用必须指向已声明接口 |
| `interface_contract.list_length_ref_not_list` | list_length.same_as 目标必须是 tensor_list |
| `interface_contract.list_length_cycle` | list_length.same_as 不得形成循环引用，单个环只报告一次 |
| `interface_contract.list_length_expression` | expression 必须遵循受限 DSL 且可证明非负 |
| `interface_contract.semantic_case_duplicate_id` | semantic_cases[].id 必须唯一 |
| `interface_contract.semantic_case_invalid_guard` | semantic_cases[].when 必须遵循受限 guard DSL |
| `interface_contract.semantic_case_unknown_member` | semantic case 只能引用已声明的 input/output |
| `interface_contract.semantic_case_conflicting_members` | 同一 semantic case 不得同时把成员列入正反集合 |
| `interface_contract.semantic_case_input_not_optional` | semantic case 只能改变 optional input 的存在性 |
| `interface_contract.semantic_case_output_not_optional` | semantic case 只能改变 optional output 的存在性 |
| `interface_contract.semantic_case_conflicting_same_guard` | 相同 guard 不得声明冲突的 I/O 存在性 |
| `interface_contract.layout_contract_duplicate_variant_id` | layout_contract.variants[].id 重复 ⇒ ERR |
| `interface_contract.layout_contract_invalid_guard` | layout_contract.variants[].when 未遵循 semantic_cases 的受限 guard DSL ⇒ ERR |
| `interface_contract.layout_contract_unknown_ref` | layout_contract 引用了未声明的 input/output Tensor ⇒ ERR |
| `interface_contract.layout_contract_ref_not_tensor` | layout_contract 引用的接口不是 role=tensor ⇒ ERR |
| `interface_contract.layout_contract_duplicate_tensor_ref` | 同一 layout variant 中重复声明同一 Tensor ⇒ ERR |
| `interface_contract.layout_contract_rank_mismatch` | 固定 rank input 的 logical_axes 数量不等于 rank ⇒ ERR |
| `data_distribution.accumulation_requires_normal` | 累加类算子（Reduction/ReductionComposite/Recurrence/AtomicUpdate/Histogram）的 tensor 输入未声明 `data_distribution: normal` ⇒ ERR |
| `data_distribution.non_accumulation_must_omit` | 非累加类算子声明了 `data_distribution` ⇒ ERR |

### paradigm_groups 字段说明

`op.paradigm_groups` 声明 paradigms 之间的组合关系，区分纵向融合与横向组合：

| kind | 语义 | 示例 |
|------|------|------|
| `fusion` | 纵向融合：多范式按顺序串联执行 | softmax: `[Reduction, Broadcast]` |
| `combination` | 横向组合：多范式择一，由属性值决定 | mse_loss: `[Reduction]` 或 `[Elementwise]`，取决于 `reduction` 属性 |

**规则**：
- 所有组的 paradigms 并集必须是 `op.paradigms` 的子集（未分组的视为修饰符范式，如 NumericalStable、Quantization、FusedComposite）
- `fusion` 组内 ≥ 2 基础范式（Elementwise 被其他范式吸收，不计入）
- `combination` 组必须有 `switch`（属性名）+ `when`（属性值）

**示例**：
```yaml
# softmax: 纯纵向融合
paradigms: [Reduction, Broadcast, NumericalStable, FusedComposite]
paradigm_groups:
  - kind: fusion
    paradigms: [Reduction, Broadcast]

# mse_loss: 横向组合
paradigms: [Reduction, Elementwise]
paradigm_groups:
  - kind: combination
    paradigms: [Elementwise]
    switch: reduction
    when: none
  - kind: combination
    paradigms: [Reduction]
    switch: reduction
    when: "*"
```

---

## stage 3 — shape_closure（numpy_expr 求值）

按 `outputs[].shape_rule_kind` 分流：

- **`numpy_expr`**（默认）— 在受限 AST 沙箱中执行 `outputs[].shape_rule`（numpy 子集表达式），求出每个输出的 SymbolicShape；若 `math_semantics.formula_kind=numpy_expr`，还会在正常边界样例上用具体 shape/attr 校验 formula 输出 shape，防止 `shape_rule` 写成简化/占位版本。
- **`data_dependent`** — 用于输出 shape 由输入数据值决定的算子（nonzero / unique / masked_select 等 VariableOutput，或 splitv 等 DynamicShape 的 value-driven 子场景）；不求解，但强制校验 `data_dependent_shape: true` + `shape_bounds.max_elements` + 建议配 `shape_rule_description`。只依赖 input shape / attribute 的输出（如 reshape / stridedslice 等 DynamicShape 的 attr-driven 子场景）必须写 `numpy_expr`。
- **`textual_only`** — 用于输出 shape 因数据排布格式而异的算子（如 NCHW/NHWC 的 Channel 轴位置不同）；不求解，但要求 `math_semantics.format_variants` 存在且配 `shape_rule_description`。shape_rule 可含 `${format_variants[].channel_axis}` 等占位符。

确定性 shape 规则必须完整写在 `shape_rule`：当输出 shape 同时依赖 input rank 和 attribute 时，不能只写默认分支再把完整语义放进 notes 或测试约束。纯 `Reduction` 算子不能把输出 shape 写成 `input.shape`，应按 `dim` / `keep_dims` 描述归约轴变化。

### shape_rule 示例

numpy_expr（matmul）：

```yaml
outputs:
  - name: c
    shape_rule_kind: numpy_expr
    shape_rule: |
      c.shape = (
          np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
          + ((a.shape[-1] if transpose_a else a.shape[-2]),)
          + ((b.shape[-2] if transpose_b else b.shape[-1]),)
      )
```

data_dependent（nonzero）：

```yaml
outputs:
  - name: indices
    shape_rule_kind: data_dependent
    data_dependent_shape: true
    shape_rule_description: |
      indices.shape = (K, rank(x))，K = count(x != 0)，仅运行时已知
    shape_bounds:
      max_elements: "prod(x.shape) * rank(x)"
      output_rank: 2
      static_dims: [{axis: 1, value: "rank(x)"}]
      dynamic_dims: [{axis: 0, name: K, depends_on_value_of: x}]
```

### 子规则表

| 子规则 | 检查内容 |
|---|---|
| `shape_closure.shape_rule_kind_missing` | outputs[].shape_rule_kind 未声明 |
| `shape_closure.shape_rule_kind_unknown` | 非 numpy_expr / data_dependent |
| `shape_closure.shape_rule_placeholder` | shape_rule 含 TODO / 占位 / 简化说明 |
| `shape_closure.reduction_shape_identity_suspicious` | 纯 Reduction 的 shape_rule 直接写成 input.shape |
| `shape_closure.synthesize_parse_error` | WARN — boundary synthesize 中 shape / attr 字面量无法解析，不能用于 shape oracle |
| `shape_closure.numpy_expr_shape_mismatch` | 正常边界样例下 shape_rule 形状与 formula 实际输出形状不一致 |
| `shape_closure.numpy_expr_formula_eval_error` | 正常边界样例下 formula 不能作为 shape oracle 执行 |
| `shape_closure.dsl_parse_error` | numpy_expr 语法错 |
| `shape_closure.dsl_eval_error` | 求值时 AttributeError / TypeError / 沙箱拒绝 |
| `shape_closure.unresolved_symbol` | 引用了未声明的 input / attribute 名 |
| `shape_closure.incompatible_dims` | 显式维冲突（如 matmul K 维同名约束失败） |
| `shape_closure.folded_dim_misuse` | 折叠维 `...x` 重复或不在首位 |
| `shape_closure.unregistered_symbol` | WARN — 显式维未在 `shape_constraints.symbols` 登记 |
| `shape_closure.rank_overflow` | 输出 rank 超过 `inputs[].rank_range` 上限 |
| `shape_closure.data_dependent_flag_missing` | `data_dependent` 但未配 `data_dependent_shape: true` |
| `shape_closure.data_dependent_missing_bounds` | `data_dependent` 但未配 `shape_bounds.max_elements` |
| `shape_closure.data_dependent_missing_description` | WARN — 建议补 `shape_rule_description` |
| `shape_closure.textual_only_requires_format_variants` | `textual_only` 但 `math_semantics.format_variants` 不存在 ⇒ ERR |
| `shape_closure.textual_only_requires_description` | `textual_only` 但缺 `shape_rule_description` ⇒ ERR |

### numpy_expr 支持的子集

按需可扩展，见 `scripts/evaluators/shape_eval.py`：

- `x.shape` / `x.shape[-1]` / `x.shape[:-2]`（切片 + 负索引）
- `tuple` 拼接 `+`（`(a.shape[-1],) + (b.shape[-1],)`）
- `np.broadcast_shapes(*shapes)`（numpy 标准 API 签名）
- `np.reduce_shape(shape, axis=dim, keepdims=keep_dims)`（归约输出 shape；折叠前缀形状需用负 axis 指向显式后缀维）
- `IfExp`：`a if cond else b`（符号求解用 attribute 默认值；具体样例校验用边界样例 attr）

沙箱规则同 stage 8：禁 import / def / class / for / while / lambda / dunder attribute；5 秒超时。

---

## stage 4 — dtype_closure

对 `dtype_policy.supported_combinations` 的每一行，执行 `outputs[].dtype_rule`（numpy 子集表达式）推导输出 dtype，与显式表交叉比对。

### dtype_rule 常见形式

```yaml
# 同 dtype 直传
dtype_rule: "y.dtype = x.dtype"
# 类型提升
dtype_rule: "c.dtype = np.promote_types(a.dtype, b.dtype)"
# 固定 dtype（如 ArgReduce 输出索引）
dtype_rule: "y.dtype = np.int64"
# 条件分支（如 complex_of）
dtype_rule: |
  out.dtype = np.complex64 if real.dtype == np.float32 else np.complex128
```

### 子规则表

| 子规则 | 检查内容 |
|---|---|
| `dtype_closure.combination_mismatch` | dtype_rule 推得 X 但显式表写 Y（典型："锁定意图"机制：当 numpy promote 表升级时，spec 不会被静默改语义） |
| `dtype_closure.combination_missing_output` | 组合行未声明某 output 的 dtype |
| `dtype_closure.dsl_parse_error` / `dsl_eval_error` | 表达式语法错 / 求值错 |
| `dtype_closure.unresolved_symbol` | dtype_rule 引用的 input 不在该组合内 |
| `dtype_closure.dtype_rule_kind_unknown` | 非 numpy_expr 的 dtype_rule_kind |

### numpy 子集支持

- `x.dtype`（input 的 dtype 字符串）
- `np.promote_types(a, b)` / `np.result_type(*xs)` — 复用 promote 表
- `np.int8 / int16 / int32 / int64 / uint8 / uint16 / uint32 / uint64 / float16 / float32 / float64 / bfloat16 / complex64 / complex128 / bool_` dtype 常量
- 比较与 IfExp：`a.dtype == np.float32`、`x if cond else y`

Promotion 表覆盖 fp16 / fp32 / fp64 / bf16 / int8 / int16 / int32 / int64 / uint* / bool / complex64 / complex128 + 窄浮点 fp4_e2m1 / fp8_e4m3fn / fp8_e5m2 / hifloat8（见 `scripts/evaluators/promote.py`）。

---

## stage 5 — broadcast_legality

校验算子计算中的 broadcast 语义是否合法。`broadcast` 字段描述的是**算子计算是否需要对数据进行 broadcast（数据复制/扩展）**，而非仅描述多输入 shape 是否需要对齐。

| kind | 语义 | 适用场景 |
|---|---|---|
| `numpy` | 计算涉及数据广播 | 多输入 elementwise（add/mul） |
| `none` | 计算不涉及任何 broadcast | 输入 shape 严格一致、无需数据扩展的算子 |
| `explicit` | 部分维度 broadcast + 部分维度不 broadcast | matmul（batch dims broadcast，M/K/N 不 broadcast） |

### 子规则表

| 子规则 | 检查内容 |
|---|---|
| `broadcast_legality.numpy_violation` | `kind: none` 但算子计算实际需要 broadcast 数据 |
| `broadcast_legality.incompatible_dims` | `kind: numpy` 下显式维冲突（不可右对齐） |
| `broadcast_legality.explicit_rules_uncovered` | `kind: explicit` 的 rules 未覆盖输入全部维度 |

v1 已实现：`numpy`（含折叠维）/ `none` / `explicit`（`broadcast.rules` 必须**恰好**一条 `scope: trailing` + 一条 `scope: leading`，policy 各支持 `numpy` / `no_broadcast`，覆盖 matmul 形式）。explicit 的高阶组合（多于 2 个 input、`scope: axis`）留待扩展。

---

## stage 6 — boundary_min_set

按 `op.paradigms` 检查 `boundary_conditions[]` / `extreme_inputs[]` 是否覆盖该范式必含的最低 case 集。数据源：`registries/boundary_min_cases.yaml`。

| 子规则 | 检查内容 |
|---|---|
| `boundary_min_set.missing_required_case` | 例如 Reduction ⇒ 必含"reduce 轴长=1 / rank=0 / 空 Tensor"；NumericalStable ⇒ 必含 fp16 上溢 extreme case |

匹配方式：每条 requirement 用 `match_any` 关键词列表与 case 描述做子串包含；任一命中即算覆盖。结构性约束走 `special_check` 直接检查。

覆盖的 paradigms 见 `registries/boundary_min_cases.yaml`（当前覆盖 Reduction / NumericalStable / SlidingWindow / Padding / IndexGather / ScatterUpdate / AtomicUpdate / MaskPredicate / SortSelect / ArgReduce / Histogram / Spectral / DynamicShape / RandomSampling / Quantization / Stateful / VariableOutput / Sparse / Contraction / Elementwise / Broadcast）。

---

## stage 7 — tolerance_coverage

| 子规则 | 检查内容 |
|---|---|
| `tolerance_coverage.uncovered_output_dtype` | `dtype_policy.supported_combinations` 中所有出现的输出 dtype 必须在 `numerical_tolerance.per_dtype` 中声明容差（FAIL） |
| `tolerance_coverage.tolerance_too_tight` | rtol 显著低于 dtype 单步舍入量级（fp32 < 1e-7、fp16 < 1e-4、bf16 < 1e-3）⇒ WARN |

---

## stage 8 — formula_smoke_eval

把 `math_semantics.formula` 在小 shape（默认 `[2,3]`）上用 numpy 跑一遍，确认能跑通且产出预期 dtype。

### 子规则表

| 子规则 | 检查内容 |
|---|---|
| `formula_smoke_eval.syntax_error` | formula 字符串语法错 |
| `formula_smoke_eval.formula_ast_disallowed` | AST 含禁止节点（import / def / class / for / while / try / lambda） |
| `formula_smoke_eval.formula_banned_name` | 用了 `__import__` / `exec` / `getattr` 等绕沙箱标识符 |
| `formula_smoke_eval.numpy_eval_error` | 运行时 numpy 报错（API 拼错 / 参数错 / shape 不对） |
| `formula_smoke_eval.formula_timeout` | 5 秒超时（不应发生，formula 应是纯向量化操作） |
| `formula_smoke_eval.missing_output` | formula 未给某个声明的 output 变量赋值 |
| `formula_smoke_eval.dtype_mismatch_at_runtime` | WARN — 运行时 dtype 与 supported_combinations 声明不一致 |
| `formula_smoke_eval.produces_unexpected_nan` | WARN — 中性输入下产出全 NaN |
| `formula_smoke_eval.empty_formula` | formula_kind=numpy_expr 但 formula 字符串为空 |

### 设计要点

- **AST 白名单沙箱** —— 只允许表达式、赋值、numpy 调用；禁止 import / def / class / 循环 / lambda
- **受限 globals** —— 只暴露 `np` / `math` + 安全 builtins 子集
- **5 秒超时** —— SIGALRM-based，防止意外死循环
- **小 shape** —— 折叠维 `...d` 丢弃；显式符号维默认 `3`；常量维原样使用
- **dtype 选择** —— 取 `supported_combinations[0]` 的输入 dtype；`bfloat16` 没有 numpy 原生类型，用 `float32` 替代

`formula_kind` 取值的影响：
- `numpy_expr` → 跑 stage 8
- `python_block` / `textual_only` → SKIP（写在 formula 里的 LaTeX / 自然语言无法 eval）

---

## stage 9 — oracle_reachable

真 import `math_semantics.reference_oracle.framework` 并走 `getattr` 链找到 `api`，确认它存在且 callable；同时校验 `kwargs` 中的 `${...}` 占位符引用真实存在的 attribute / input / output。

### 子规则表

| 子规则 | 检查内容 |
|---|---|
| `oracle_reachable.absent` | INFO — absent=true ⇒ stage 9 SKIP（语义由 invariants + boundary 覆盖） |
| `oracle_reachable.incomplete` | framework 或 api 字段缺失（且 absent=false） |
| `oracle_reachable.api_framework_mismatch` | api 字符串首段与 framework 字段不一致 |
| `oracle_reachable.framework_not_installed` | INFO — framework 未装；stage 9 走 SKIP（不算 FAIL）但**仍校验占位符** |
| `oracle_reachable.api_not_found` | framework 已装但 api 字符串中某段 `getattr` 失败（拼写错） |
| `oracle_reachable.api_not_callable` | api 找到了但不是 callable |
| `oracle_reachable.placeholder_unresolved` | `${attr.X}` / `${input.X}` / `${output.X}` 引用 spec 中不存在的名字 |
| `oracle_reachable.unknown_framework` | WARN — framework 不在已知列表（见 `registries/framework_oracle_registry.yaml`） |
| `oracle_reachable.dtype_unsupported` | WARN — supported_combinations 输入 dtype 不在 oracle.available_for_dtype 中 |

### 降级策略

- **framework 未装** ⇒ SKIP，不算失败（这是环境问题，不是 spec 缺陷）
- **api 拼错** ⇒ FAIL（前提是 framework 装了；没装则跳过）
- **占位符校验**与是否装框架**无关**，永远跑

性能：torch 冷启动 1-3 秒；同进程内多次调用因 import 缓存而即时返回。pre-commit 用户若没装 torch，stage 9 走 SKIP 不影响 < 1s 体验。

---

## stage 10 — formula_oracle_equiv

当 formula（numpy_expr）和 oracle（reachable）同时可用时，在**含特殊值（NaN / inf / 大数 / 带符号零 +0/-0）的输入**上同时执行 formula 和 oracle，比对输出。捕获 formula 使用了与框架实际实现**数学等价但计算不等价**形式的错误（如教科书合并形式 `m_new = β1·m + (1-β1)·grad` vs TF 增量形式 `m = m + (1-β1)·(grad-m)`，两者在 NaN/inf 传播、末位舍入、零符号上不一致）。

### 子规则表

| 子规则 | 检查内容 |
|---|---|
| `formula_oracle_equiv.skipped_non_numpy` | INFO — formula_kind != numpy_expr ⇒ stage 10 SKIP |
| `formula_oracle_equiv.absent` | INFO — oracle.absent=true ⇒ stage 10 SKIP |
| `formula_oracle_equiv.composition_not_supported` | INFO — oracle 使用 composition (DAG) 模式，值等价校验暂不支持 |
| `formula_oracle_equiv.numpy_not_installed` | INFO — numpy 未安装 |
| `formula_oracle_equiv.incomplete_oracle` | INFO — oracle framework 或 api 缺失 |
| `formula_oracle_equiv.framework_not_installed` | INFO — oracle framework 未安装 |
| `formula_oracle_equiv.api_unreachable` | INFO — oracle API 不可达（stage 9 已报错） |
| `formula_oracle_equiv.no_combination` | INFO — 缺 supported_combinations |
| `formula_oracle_equiv.oracle_call_failed` | INFO — 无法在测试输入上成功调用 oracle API |
| `formula_oracle_equiv.shape_mismatch` | WARN — formula 输出 shape ≠ oracle 输出 shape，跳过值比较 |
| `formula_oracle_equiv.nan_pattern_divergence` | **ERR** — formula 与 oracle 的 NaN 出现位置不一致（典型信号：公式形式差异导致 NaN 传播路径不同） |
| `formula_oracle_equiv.inf_pattern_divergence` | **ERR** — formula 与 oracle 的 inf/-inf 出现位置不一致 |
| `formula_oracle_equiv.zero_sign_divergence` | **ERR** — formula 与 oracle 的 +0/-0 符号模式不一致（IEEE 754 带符号零；符号差异暴露抵消/下溢路径上的计算形式差异） |
| `formula_oracle_equiv.value_divergence` | **ERR** — formula 与 oracle 在有限值上发散超过容差 |

### 测试输入生成

对 spec 声明的输入生成 7 组测试张量：

| 组 | 注入的特殊值 | 目的 |
|---|---|---|
| normal | 无（随机基线） | 验证基本语义一致 |
| nan | 第一个元素 NaN | 检测 NaN 传播路径差异 |
| inf | 第一个元素 +inf | 检测 inf 传播路径差异 |
| ninf | 第一个元素 -inf | 检测 -inf 传播路径差异 |
| large | 第一个元素 1e30 | 检测大数舍入差异 |
| pos_zero | 第一个元素 +0.0（仅浮点 dtype） | 检测带符号零传播差异（IEEE 754 +0/-0） |
| neg_zero | 第一个元素 -0.0（仅浮点 dtype） | 检测带符号零传播差异（IEEE 754 +0/-0） |

### 比对策略

1. **NaN 模式**：formula 和 oracle 输出的 NaN 位置必须一致。不一致 → ERR（公式形式差异的最强信号）
2. **inf 模式**：在非 NaN 位置上，inf/-inf 出现位置必须一致。不一致 → ERR
3. **零符号模式**：在两者均为有限值且为零的位置上，+0/-0 符号（`np.signbit`）必须一致。不一致 → ERR（仅浮点 dtype；符号差异暴露抵消/下溢路径上的计算形式差异）
4. **有限值**：在两者均为有限值的位置上，atol=1e-5、rtol=1e-5 内视为等价。超出 → ERR

### 设计要点

- **降级策略**：formula_kind 非 numpy_expr / oracle absent / composition 模式 / framework 未装 / oracle 调用失败 → SKIP，不阻塞
- **容差选择**：有限值用中等容差（1e-5），因为即使公式形式正确，求和顺序等差异也会产生末位不同；关键信号在 NaN/inf/零符号模式
- **最小依赖**：复用 stage 8 的 formula 沙箱执行 + stage 9 的 oracle API 解析，不重复实现
- **性能**：小 shape（默认 [2,3]）× 7 组测试输入，formula + oracle 各调用 7 次，总耗时 < 1s

---

## stage 11 — invariant_exec

当 formula（numpy_expr）可用且 spec 声明了 `invariants[]` 时，在生成的正常输入上执行 formula，**校验输出是否满足其声明的数学性质**。与 stage 10（formula vs 框架 oracle）不同，stage 11 **无需外部 oracle**——它验证 formula 是否符合自身声明的数学不变量（softmax 行和=1、relu 非负、add 交换律、matmul/reduce 零吸收……）。

**这是无 oracle 场景（absent / composition / oracle_call_failed）的关键机器校验**——stage 10 SKIP 时，stage 11 用 invariants 作为替代校验。设计动机：形式错的公式往往违反声明的数学性质（如未归一化的"softmax"行和≠1），stage 11 在 stage 10 失效时仍能抓到，打破了「最需要形式校验的场景恰好没有 oracle」的悖论。

### 可执行的 kind

| 组 | kind | 校验方式 |
|---|---|---|
| value | elementwise_ge / elementwise_le / elementwise_eq | 输出逐元素与阈值/常量比较 |
| value | reduce_equals | 沿 axis 归约（sum/mean/max/min）后与常量比较 |
| value | range_in | 输出 ∈ [lo, hi] |
| value | produces_in_set | 输出取值 ∈ 离散集合 |
| algebraic | equals_under_swap | 交换输入对，两次 formula 输出应一致 |
| algebraic | equals_input_when_other_is_zero | 置零一侧，输出应 == 另一侧 |
| algebraic | equals_when_input_is_zero | 置零任一数据输入，输出应 == value |

暂不机器执行的 kind（发 info 跳过）：`idempotent` / `associative_along_batch` / `monotone_along`（需特殊输入）/ `equals_input_when_other_is_identity`（composition only）/ `equals_after_op`（需外部 op）/ structural 组（`shape_equals_macro` / `no_leak_intermediates` / `permutation_of_input`）。

### 子规则表

| 子规则 | 检查内容 |
|---|---|
| `invariant_exec.skipped_non_numpy` | INFO — formula_kind != numpy_expr ⇒ stage 11 SKIP |
| `invariant_exec.no_invariants` | INFO — spec 未声明 invariants ⇒ SKIP |
| `invariant_exec.numpy_not_installed` | INFO — numpy 未安装 |
| `invariant_exec.no_combination` | INFO — 缺 supported_combinations |
| `invariant_exec.no_inputs` | INFO — 无法生成测试输入 |
| `invariant_exec.kind_not_executable` | INFO — 该 kind 暂不支持机器执行，跳过值校验 |
| `invariant_exec.reduce_equals_unresolved` | INFO — reducer/axis/value 无法静态解析（axis 来自运行时输入） |
| `invariant_exec.formula_no_output` | INFO — formula 未产出目标输出 |
| `invariant_exec.elementwise_ge_violated` | **ERR** — 输出有元素 < value |
| `invariant_exec.elementwise_le_violated` | **ERR** — 输出有元素 > value |
| `invariant_exec.elementwise_eq_violated` | **ERR** — 输出未逐元素 == value |
| `invariant_exec.reduce_equals_violated` | **ERR** — 沿 axis 归约后结果 ≠ value |
| `invariant_exec.range_in_violated` | **ERR** — 输出有元素落在 [lo, hi] 外 |
| `invariant_exec.produces_in_set_violated` | **ERR** — 输出取值不在声明集合内 |
| `invariant_exec.equals_under_swap_violated` | **ERR** — 交换输入对后输出不一致（非交换 formula 被声明为交换律） |
| `invariant_exec.equals_input_when_other_is_zero_violated` | **ERR** — 置零一侧后输出 ≠ 另一侧 |
| `invariant_exec.equals_when_input_is_zero_violated` | **ERR** — 置零数据输入后输出 ≠ value |

### 执行策略

1. 生成正常随机输入（**每个输入用派生 seed**，避免多输入相同数据掩盖非交换性）
2. 编译 formula（复用 stage 8 的 AST 沙箱），跑出输出
3. 逐 invariant 按 kind 校验输出；algebraic 类按需多次跑 formula（交换/置零）
4. **索引输入保护**：`axis`/`axes`/`dim`/`indices` 等整数张量不被置零（与 stage 5 broadcast 逻辑一致）
5. **空输出保护**：IndexGather 类零输入→空输出时跳过值比较
6. **容差**：`tolerance_inherit: true` 用 `numerical_tolerance.per_dtype`，否则用紧容差 1e-6

### 设计要点

- **打破悖论**：stage 10 在无 oracle 时 SKIP，stage 11 用 invariants 补位——最需要形式校验的场景（公式来自论文/模型知识）获得了机器校验
- **invariants 是独立锚点**：不经过 formula 自反，能测出 formula 语义错（SPEC-ORACLE-1 已要求值级、formula 无关）
- **最小依赖**：复用 stage 8 的 formula 沙箱执行，不依赖框架安装
- **降级策略**：formula_kind 非 numpy_expr / 无 invariants / numpy 未装 ⇒ SKIP，不阻塞
