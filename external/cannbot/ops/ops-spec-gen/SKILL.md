---
name: external-cannbot-ops-ops-spec-gen
description: 生成或校验算子 spec.yaml（算子的 L0 数学约束唯一真值）。当用户提及：生成 spec.yaml、新算子 spec 骨架、scaffold
  spec、validate spec.yaml、spec schema 校验、算子规格校验 时触发。
original-name: ops-spec-gen
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-02'
synced-commit: 342679f37ec1b052a8b64d6c9fd59f67e0c89073
license: UNKNOWN
---

# 算子 spec.yaml 生成与校验

为 CANNBot 算子产出符合 schema 的 `spec.yaml`，并对其执行完整 9-stage L0 校验：
- **stage 1** schema_static — JSON Schema 字段静态校验
- **stage 2** category_paradigm_consistency — category↔paradigm 一致性 + 4 套白名单 + paradigm 内部约束
- **stage 3** shape_closure — numpy_expr 求值 `outputs[].shape_rule`（含 `data_dependent` 分流；折叠/显式维 + `np.broadcast_shapes`）
- **stage 4** dtype_closure — dtype DSL 推导 vs supported_combinations 显式表交叉验证
- **stage 5** broadcast_legality — broadcast.kind/rules 模拟（numpy / none / explicit）
- **stage 6** boundary_min_set — 按 paradigms 检查 boundary/extreme case 是否覆盖最低集
- **stage 7** tolerance_coverage — `numerical_tolerance.per_dtype` 覆盖输出 dtype + 紧度启发式
- **stage 8** formula_smoke_eval — 把 formula 在小 shape (`[2,3]`) 上跑通，沙箱 numpy 执行
- **stage 9** oracle_reachable — 真 import framework + getattr 链找 api + 占位符校验

## 1. 何时使用

| 场景 | 使用分支 |
|---|---|
| 新算子起手，需要 spec.yaml 骨架 | **生成器**（§3） |
| 已有 spec.yaml，CI/PR 校验或 Designer Agent 重生成回灌 | **校验器**（§4） |
| 不清楚 25 category × 27 paradigm 怎么填 | 看 [references/spec-cheatsheet.md](references/spec-cheatsheet.md) |
| 想看完整范例（add / softmax / matmul，全部 PASS 校验） | 看 [examples/](examples/README.md) |

## 2. 输入与输出

| 项 | 说明 |
|---|---|
| 入参（生成） | 算子名、category、paradigms[]、inputs（带 dtype_set）、outputs |
| 出参（生成） | `<output_dir>/spec.yaml`（含 TODO 占位符，需手填 formula / oracle.api / supported_combinations / boundary cases） |
| 入参（校验） | spec.yaml 路径 |
| 出参（校验） | 9-stage findings；返回码 0=PASS / 1=FAIL（含 internal_error）/ 2=YAML 解析错；`--strict` 时 warning 也退 1 |

## 3. 生成 spec.yaml

### 3.1 交互式向导（推荐）

```
python3 scripts/generate_spec.py --output-dir <ops_dir>/<op_name>
```

向导会问：op 名称、category（25 类单选）、paradigms（在 category 必含基础上自由追加）、每个 input/output 名+dtype、promotion / broadcast / accumulation_order。

### 3.2 非交互式（CI / 脚本）

```
python3 scripts/generate_spec.py \
    --op-name softmax \
    --category reduction_composite \
    --paradigms Reduction,NumericalStable,FusedComposite \
    --inputs "x:float16,float32,bfloat16" \
    --outputs y \
    --description "Softmax along reduce_axis with max-shift stabilization" \
    --output-dir ops/softmax
```

注：`--inputs` 多个时用 `;` 分隔（避免与 dtype_set 的 `,` 冲突），如 `"a:float16,float32;b:float16,float32"`。

### 3.3 自动注入

按 `op.paradigms` 自动注入：
- `Reduction`（且无 `Recurrence`）⇒ 添加 `dim` attribute（`int_in_range_relative_to_rank`）
- `Quantization` ⇒ 添加 `scale` / `zero_point` attribute
- `RandomSampling` ⇒ 添加 `seed` attribute
- `Stateful` ⇒ 第一个 input 标 `role: state`
- `CollectiveCommunication` ⇒ `op.platform_constraints.requires_hccl: true`
- `category=contraction` / `category=reduction_composite` / `paradigms` 含 `NumericalStable+Reduction` ⇒ 添加 `accumulator_dtype: float32`
- `FusedComposite` ⇒ 添加 `composition.primitives`（2 条占位）+ `dataflow.no_leak: true`
- `NumericalStable` ⇒ `numerical_stability.required: true` + 一条 technique 占位

更多注入提示见 `registries/category_paradigm_map.yaml` 的 `paradigm_inject_hints` 段。

### 3.4 必须手填的 TODO

生成器只产骨架 + 按 paradigm 自动注入的 boundary/extreme case。算子 owner 必须手填：
- `outputs[].shape_rule` — numpy 子集表达式（如 `c.shape = np.broadcast_shapes(a.shape, b.shape)`）。
  若是 VariableOutput 范式的算子（shape 由输入值决定），改声明 `shape_rule_kind: data_dependent` +
  `shape_rule_description` + `shape_bounds.max_elements`，不写表达式。
- `outputs[].dtype_rule` — numpy 子集表达式（如 `c.dtype = np.promote_types(a.dtype, b.dtype)`、
  `y.dtype = x.dtype`、`y.dtype = np.int32`）
- `math_semantics.formula` — numpy 可 eval 的表达式
- `math_semantics.reference_oracle.api` — 真实的 torch/numpy/scipy API 全限定名
- `dtype_policy.supported_combinations` — 显式枚举 (input dtypes) → output dtypes
- `numerical_tolerance.per_dtype` — 覆盖 supported_combinations 中所有 output dtype

按需补充（生成器已自动注入按 paradigm 必含的最低 case，stage 6 通过；下面这些是把生成器的占位描述改成项目语境下的真值）：
- `boundary_conditions` — 校核 / 补全自动注入的退化 case 描述（详见 cheatsheet §B6）
- `extreme_inputs` — 校核 / 补全 NaN / Inf / 全零三类

## 4. 校验 spec.yaml

```
python3 scripts/validate_spec.py path/to/spec.yaml          # 文本输出，跑全 9 stage
python3 scripts/validate_spec.py path/to/spec.yaml --json   # 机器可读
python3 scripts/validate_spec.py path/to/spec.yaml --strict # 警告也 fail
python3 scripts/validate_spec.py path/to/spec.yaml --quiet  # 仅打 FAIL 的 stage
python3 scripts/validate_spec.py path/to/spec.yaml --stage 1 --stage 2  # 仅跑选中的 stage
```

### 4.1 stage 1 — schema_static

按 `schemas/op-spec.json`（Draft 2020-12）逐字段校验：必填、类型、enum、pattern、additionalProperties。

### 4.2 stage 2 — category_paradigm_consistency

校验 category↔paradigm 必含映射 + 4 套白名单 + paradigm 内部约束：

| 子规则 | 检查内容 |
|---|---|
| `required_paradigm_missing` | category → 必含 paradigm 是否齐全 |
| `fused_composite_basics` | category=fused_composite ⇒ ≥ 2 条基础 paradigm |
| `mutually_exclusive` | ScatterUpdate 与 AtomicUpdate 不得共存 |
| `paradigm_constraint.numerical_stable` | NumericalStable ⇒ `numerical_stability.required: true` |
| `paradigm_constraint.fused_composite_*` | FusedComposite ⇒ composition 必填、primitives ≥ 2、op 在白名单、中间不泄漏、dataflow 闭合 |
| `paradigm_constraint.reduction_axis_missing` | Reduction（无 Recurrence）⇒ 必有 axis/dim/axes |
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

### 4.3 stage 3 — shape_closure（numpy_expr 求值）

按 `outputs[].shape_rule_kind` 分流：

- **`numpy_expr`**（默认）— 在受限 AST 沙箱中执行 `outputs[].shape_rule`（numpy 子集表达式），求出每个输出的 SymbolicShape。
- **`data_dependent`** — 输出 shape 由输入数据值决定（nonzero / unique / masked_select 等）；不求解，但强制校验 `data_dependent_shape: true` + `shape_bounds.max_elements` + 建议配 `shape_rule_description`。

shape_rule 的 numpy_expr 示例（matmul）：

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

data_dependent 示例（nonzero）：

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

| 子规则 | 检查内容 |
|---|---|
| `shape_closure.shape_rule_kind_missing` | outputs[].shape_rule_kind 未声明 |
| `shape_closure.shape_rule_kind_unknown` | 非 numpy_expr / data_dependent |
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

numpy_expr 支持的子集（按需可扩展，见 `scripts/evaluators/shape_eval.py`）：
- `x.shape` / `x.shape[-1]` / `x.shape[:-2]`（切片 + 负索引）
- `tuple` 拼接 `+`（`(a.shape[-1],) + (b.shape[-1],)`）
- `np.broadcast_shapes(*shapes)`（numpy 标准 API 签名）
- `IfExp`：`a if cond else b`（cond 从 attribute 默认值解析为 bool）

沙箱规则同 stage 8：禁 import / def / class / for / while / lambda / dunder attribute；5 秒超时。

### 4.4 stage 4 — dtype_closure

对 `dtype_policy.supported_combinations` 的每一行，执行 `outputs[].dtype_rule`（numpy 子集表达式）推导输出 dtype，与显式表交叉比对。常见形式：

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

| 子规则 | 检查内容 |
|---|---|
| `dtype_closure.combination_mismatch` | dtype_rule 推得 X 但显式表写 Y（典型："锁定意图"机制：当 numpy promote 表升级时，spec 不会被静默改语义） |
| `dtype_closure.combination_missing_output` | 组合行未声明某 output 的 dtype |
| `dtype_closure.dsl_parse_error` / `dsl_eval_error` | 表达式语法错 / 求值错 |
| `dtype_closure.unresolved_symbol` | dtype_rule 引用的 input 不在该组合内 |
| `dtype_closure.dtype_rule_kind_unknown` | 非 numpy_expr 的 dtype_rule_kind |

numpy 子集支持：
- `x.dtype` （input 的 dtype 字符串）
- `np.promote_types(a, b)` / `np.result_type(*xs)` — 复用 promote 表
- `np.int8 / int16 / int32 / int64 / uint8 / uint16 / uint32 / uint64 / float16 / float32 / float64 / bfloat16 / complex64 / complex128 / bool_` dtype 常量
- 比较与 IfExp：`a.dtype == np.float32`、`x if cond else y`

Promotion 表覆盖 fp16 / fp32 / fp64 / bf16 / int8 / int16 / int32 / int64 / uint* / bool / complex64 / complex128 + 窄浮点 fp4_e2m1 / fp8_e4m3fn / fp8_e5m2 / hifloat8（见 `scripts/evaluators/promote.py`）。

### 4.5 stage 5 — broadcast_legality

按 `broadcast.kind` 模拟输入 shape 之间的对齐。

| 子规则 | 检查内容 |
|---|---|
| `broadcast_legality.numpy_violation` | `kind: none` 但输入 shape 实际需要广播 |
| `broadcast_legality.incompatible_dims` | `kind: numpy` 下显式维冲突（不可右对齐） |
| `broadcast_legality.explicit_rules_uncovered` | `kind: explicit` 的 rules 未覆盖输入全部维度 |

v1 已实现：`numpy`（含折叠维）/ `none` / `explicit`（`broadcast.rules` 必须**恰好**一条 `scope: trailing` + 一条 `scope: leading`，policy 各支持 `numpy` / `no_broadcast`，覆盖 matmul 形式）。explicit 的高阶组合（多于 2 个 input、`scope: axis`）留待扩展。

### 4.6 stage 6 — boundary_min_set

按 `op.paradigms` 检查 `boundary_conditions[]` / `extreme_inputs[]` 是否覆盖该范式必含的最低 case 集。数据源：`registries/boundary_min_cases.yaml`。

| 子规则 | 检查内容 |
|---|---|
| `boundary_min_set.missing_required_case` | 例如 Reduction ⇒ 必含"reduce 轴长=1 / rank=0 / 空 Tensor"；NumericalStable ⇒ 必含 fp16 上溢 extreme case；DynamicShape ⇒ shape_set ≥ 3 个 |

匹配方式：每条 requirement 用 `match_any` 关键词列表与 case 描述做子串包含；任一命中即算覆盖。结构性约束（如 shape_set 数量）走 `special_check` 直接检查。

覆盖的 paradigms：Reduction / NumericalStable / SlidingWindow / Padding / IndexGather / ScatterUpdate / AtomicUpdate / MaskPredicate / SortSelect / ArgReduce / Histogram / Spectral / DynamicShape / RandomSampling / Quantization / Stateful。其他 paradigms 当前无强制最低 case。

### 4.7 stage 7 — tolerance_coverage

| 子规则 | 检查内容 |
|---|---|
| `tolerance_coverage.uncovered_output_dtype` | `dtype_policy.supported_combinations` 中所有出现的输出 dtype 必须在 `numerical_tolerance.per_dtype` 中声明容差（FAIL） |
| `tolerance_coverage.tolerance_too_tight` | rtol 显著低于 dtype 单步舍入量级（fp32 < 1e-7、fp16 < 1e-4、bf16 < 1e-3）⇒ WARN |

注：本 skill 用 stage 4 输出 dtype 集合（在 stage 4 实现前用 `supported_combinations` 显式枚举作为代理）；两者通常一致。

### 4.8 stage 8 — formula_smoke_eval

把 `math_semantics.formula` 在小 shape（默认 `[2,3]`）上用 numpy 跑一遍，确认能跑通且产出预期 dtype。

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

设计要点：
- **AST 白名单沙箱** —— 只允许表达式、赋值、numpy 调用；禁止 import / def / class / 循环 / lambda
- **受限 globals** —— 只暴露 `np` / `math` + 安全 builtins 子集
- **5 秒超时** —— SIGALRM-based，防止意外死循环
- **小 shape** —— 折叠维 `...d` 丢弃；显式符号维默认 `3`；常量维原样使用
- **dtype 选择** —— 取 `supported_combinations[0]` 的输入 dtype；`bfloat16` 没有 numpy 原生类型，用 `float32` 替代

`formula_kind` 取值的影响：
- `numpy_expr` → 跑 stage 8
- `python_block` / `textual_only` → SKIP（写在 formula 里的 LaTeX / 自然语言无法 eval）

### 4.9 stage 9 — oracle_reachable

真 import `math_semantics.reference_oracle.framework` 并走 `getattr` 链找到 `api`，确认它存在且 callable；同时校验 `kwargs` 中的 `${...}` 占位符引用真实存在的 attribute / input / output。

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

降级策略：
- **framework 未装** ⇒ SKIP，不算失败（这是环境问题，不是 spec 缺陷）
- **api 拼错** ⇒ FAIL（前提是 framework 装了；没装则跳过）
- **占位符校验**与是否装框架**无关**，永远跑

性能：torch 冷启动 1-3 秒；同进程内多次调用因 import 缓存而即时返回。pre-commit 用户若没装 torch，stage 9 走 SKIP 不影响 < 1s 体验。

## 5. 文件结构

```
ops/ops-spec-gen/
├── SKILL.md                              # 本文件
├── references/spec-cheatsheet.md         # 字段速查（按需阅读）
├── examples/                             # PASS 校验的范例（教学 + CI fixture）
│   ├── README.md
│   ├── add/spec.yaml                     #   elementwise + Broadcast
│   ├── softmax/spec.yaml                 #   reduction_composite + NumericalStable + FusedComposite
│   ├── matmul/spec.yaml                  #   contraction（numpy_expr + IfExp transpose）
│   ├── complex/spec.yaml                 #   fused_composite + 复数 dtype（IfExp dtype_rule）
│   ├── cumsum/spec.yaml                  #   recurrence
│   ├── dropout/spec.yaml                 #   random_sampling
│   ├── fused_quant_matmul/spec.yaml      #   fused_composite + quantization（composition DAG）
│   └── nonzero/spec.yaml                 #   variable_output + IndexGather（data_dependent）
├── registries/                           # 11 个 yaml 注册表（白名单+映射，可独立维护）
│   ├── category_enum.yaml                #   25 类 category
│   ├── paradigm_enum.yaml                #   27 项 paradigm
│   ├── category_paradigm_map.yaml        #   category↔paradigm 必含映射 + 注入提示
│   ├── primitive_whitelist.yaml          #   composition.primitives.op
│   ├── invariant_kind_registry.yaml      #   invariants.kind（value/algebraic/structural 三组）
│   ├── machine_check_kind_registry.yaml  #   boundary/extreme.machine_check.kind
│   ├── synthesize_pattern_registry.yaml  #   extreme_inputs.synthesize.patterns[].pattern
│   ├── error_code_enum.yaml              #   raises_error.error_type
│   ├── boundary_min_cases.yaml           #   各 paradigm 必含的 boundary/extreme case（stage 6 数据源）
│   ├── framework_oracle_registry.yaml    #   stage 9 已知 framework 列表（unknown 触发 WARN）
│   └── tolerance_defaults.yaml           #   per_dtype 默认容差 + stage 7 紧度阈值（generator/validator 共享）
├── schemas/op-spec.json                  # JSON Schema (Draft 2020-12)
├── templates/spec.yaml.tmpl              # spec.yaml 起手模板
├── scripts/
│   ├── generate_spec.py                  # 生成器（交互/非交互）
│   ├── validate_spec.py                  # 校验器主入口（完整 9 stage）
│   └── evaluators/                       # numpy 子集 AST 求值器（stage 3-5/8/9 实现）
│       ├── _ast_sandbox.py               #   AST 白名单 / dunder 拒绝 / timeout（三 evaluator 共享）
│       ├── types.py                      #   Dim / SymbolicShape / DslError
│       ├── parser.py                     #   parse_shape_literal（解析 inputs[].shape.symbolic）
│       ├── shape_eval.py                 #   stage 3 SymbolicShapeEvaluator
│       ├── dtype_eval.py                 #   stage 4 DtypeEvaluator
│       ├── promote.py                    #   numpy 类型提升表
│       ├── broadcast.py                  #   numpy / none / explicit 广播模拟
│       ├── stages.py                     #   stage_3 / stage_4 / stage_5 入口
│       ├── formula_eval.py               #   stage_8 numpy 沙箱
│       └── oracle_check.py               #   stage_9 真 import + 占位符校验
└── tests/
    ├── conftest.py                       # pytest sys.path 注入
    ├── test_examples.py                  # pytest：所有 examples 必须 PASS
    ├── test_shape_literal.py             # parse_shape_literal 单测
    ├── test_shape_eval.py                # SymbolicShapeEvaluator 单测
    ├── test_dtype_eval.py                # DtypeEvaluator 单测
    ├── test_broadcast.py                 # broadcast 单测
    ├── test_promote.py                   # promote 表单测
    ├── test_formula_eval.py              # stage 8 沙箱单测
    └── test_oracle.py                    # stage 9 单测
```

## 6. 维护规则

注册表是 schema 的真值。当上游 schema 字段变更时，按以下表同步：

| 变更内容 | 同步位置 |
|---|---|
| 增减 category | `registries/category_enum.yaml` + `schemas/op-spec.json` 的 `properties.op.category.enum` |
| 增减 paradigm | `registries/paradigm_enum.yaml` + JSON Schema 同处 |
| category↔paradigm 必含映射 | `registries/category_paradigm_map.yaml` |
| 增减 PRIMITIVE_WHITELIST | `registries/primitive_whitelist.yaml` + JSON Schema |
| 增减 invariants.kind | `registries/invariant_kind_registry.yaml` + JSON Schema |
| 增减 machine_check.kind | `registries/machine_check_kind_registry.yaml` + JSON Schema |
| 增减 synthesize patterns | `registries/synthesize_pattern_registry.yaml` |
| 增减 paradigm 必含 boundary/extreme case（stage 6 数据） | `registries/boundary_min_cases.yaml` |
| 调整容差紧度阈值（stage 7 启发式）/ 默认容差 | `registries/tolerance_defaults.yaml`（generator + validator 共享数据源） |
| 扩展 shape_eval / dtype_eval 支持的 numpy API | `scripts/evaluators/shape_eval.py` / `dtype_eval.py`（在 `_NpNamespace` 中注册新 API + 在 _ShapeProxy / _DtypeProxy 上加方法） |
| 调整 numpy 类型提升规则 | `scripts/evaluators/promote.py` 的 `_FLOAT_RANK` / `_INT_RANK` / `promote_pair` |
| 调整沙箱白名单（AST 节点 / builtins） | `scripts/evaluators/_ast_sandbox.py` 的 `_ALLOWED_AST_NODES` / `_ALLOWED_BUILTINS` / `_BANNED_NAMES` |
| 新增 stage 9 已知 framework | `registries/framework_oracle_registry.yaml`（一行，无需改代码） |

每次变更后，跑 examples 回归确认无破坏：
```
pytest tests/test_examples.py -v
# 或：
for ex in examples/*/spec.yaml; do
  python3 scripts/validate_spec.py "$ex"
done
```
预期：4 个 example（add / softmax / matmul / complex）全 PASS。任一 FAIL 说明本次 schema/registry 变更破坏了向后兼容，需要回归 example 或回滚变更。

## 7. 依赖

- Python ≥ 3.10
- `pyyaml`、`jsonschema` (≥ 4.18 支持 Draft 2020-12)
- `numpy`（可选；缺失则 stage 8 SKIP）
- 至少一个 reference framework（可选；缺失则对应 stage 9 SKIP）—— 已知列表见 `registries/framework_oracle_registry.yaml`
- `pytest` (可选，仅运行 `tests/` 时需要)

```
pip install pyyaml jsonschema numpy pytest
# 想跑 stage 9 真 import 校验：另装目标 framework
pip install torch  # 或 jax / scipy / tensorflow
```

性能提示：首次跑 stage 8 时 numpy 冷启动 ~150 ms，之后同进程内 ~2 ms（pre-commit 钩子每次新进程时是冷启动；总耗时仍 < 1 s 可接受）。

降级矩阵：

| 已装 | stage 8 | stage 9 真 import |
|---|---|---|
| pyyaml + jsonschema 仅 | SKIP | SKIP（占位符校验仍跑） |
| + numpy | ✓ | SKIP（占位符校验仍跑） |
| + numpy + torch | ✓ | ✓（torch 类 oracle） |
| + numpy + 全 framework | ✓ | ✓ |

## 8. 与下游 skill / agent 的关系

### 8.1 字段职责分层

spec.yaml 中的字段按职责分为三层：

|| 字段 | 职责层级 | 说明 |
|------|----------|----------|------|
| **L0 语义** | `inputs` / `outputs` / `math_semantics` / `shape_constraints` / `dtype_policy` / `broadcast` | 数学约束 | 算子"应该做什么"，下游不可覆盖 |
| **L0 语义** | `boundary_conditions` / `extreme_inputs` | 边界语义 | 算子在边界/异常情况的行为预期，下游不可覆盖 |
| **L0 语义** | `numerical_tolerance` / `numerical_stability` / `determinism` | 精度语义 | 算子精度要求，下游不可覆盖 |
| **测试配置** | `test_matrix` | **已移出** | 测试生成参数由 `ascendc-st-design` skill 独立管理，不再纳入 L0 规格 |

### 8.2 测试配置管理

- **设计原则**：L0 规格只包含算子数学约束；测试生成参数由下游 skill 独立管理
- **下游职责**：`ascendc-st-design` skill 在 `operators/{op}/tests/st/design/02_测试配置.yaml` 中定义测试参数（seed、n_cases、shape_sampler 等）
- **边界/极端数据**：`boundary_conditions` / `extreme_inputs` 仍保留在 spec.yaml（是算子语义，不是测试参数）

### 8.3 上游与下游

- **上游**（提供）：Designer Agent 接收的算子需求 / REQUIREMENTS.md
- **下游**（消费）：
  - 9-stage L0 校验器（stage 3-9）以本 skill 输出的 spec.yaml 为输入
  - `pypto-op-design` skill 用 spec.yaml 中 `inputs / outputs / math_semantics.formula` 生成 DESIGN.md
  - Developer Agent 按 spec.yaml `numerical_stability.techniques.anti_pattern_id` 触发反模式审计
  - `ascendc-st-design` skill 用 `boundary_conditions / extreme_inputs` 作为测试用例来源（L0 语义），独立管理测试配置参数

## 9. 已知限制

- stage 3 numpy_expr 求值器暴露的 `np` 命名空间仅含 `broadcast_shapes`；如需 `reshape` / `transpose` / `concat` 等形状变换语义，需在 `scripts/evaluators/shape_eval.py` 的 `_NpNamespace` 与 `_ShapeTuple` 上扩展
- stage 4 dtype_eval 的 `np` 命名空间含 `promote_types` / `result_type` + dtype 常量字符串（int8..complex128 + bfloat16）；复杂条件走 `IfExp`
- stage 5 v1 仅 2 输入广播；`explicit` rules 仅支持 `trailing + leading` 组合（matmul 形式）
- stage 6 用关键词子串匹配判 case 是否覆盖；spec 作者用罕见措辞时可能漏匹配，可在 `boundary_min_cases.yaml` 的 `match_any` 中扩展关键词
- stage 7 用 `dtype_policy.supported_combinations` 显式枚举作为输出 dtype 代理
- stage 8 用 `[2,3]` 作为默认小 shape；折叠维 `...d` 展开为 1 维（保证 `np.nonzero` 等不接受 0-D 的 API 也能跑）；`bfloat16` 用 `float32` 模拟（dtype 比对自动放过 stand-in 对）
- stage 9 不真跑 oracle —— 只确认 callable 存在；运行结果与 spec.formula 对拍是 stage 8 + 测试阶段的事
- `scripts/validate_spec.py` 仍是单文件（stage 2 已拆 8 个子函数）；未来若团队规模扩大可拆 `scripts/validators/` 子包，与 `scripts/evaluators/` 对称
