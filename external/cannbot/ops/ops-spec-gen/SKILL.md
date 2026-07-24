---
name: external-cannbot-ops-ops-spec-gen
description: 生成或校验算子 spec.yaml（算子的 L0 数学约束唯一真值）。当用户提及：生成 spec.yaml、新算子 spec 骨架、scaffold
  spec、validate spec.yaml、spec schema 校验、算子规格校验 时触发。
original-name: ops-spec-gen
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# 算子 spec.yaml 生成与校验

为 CANNBot 算子产出符合 schema 的 `spec.yaml`，并对其执行完整 9-stage L0 校验：
- **stage 1** schema_static — JSON Schema 字段静态校验
- **stage 2** category_paradigm_consistency — category↔paradigm 一致性 + paradigm_groups + paradigm 内部约束
- **stage 3** shape_closure — numpy_expr 求值 `outputs[].shape_rule`（含 `data_dependent` 分流）
- **stage 4** dtype_closure — dtype DSL 推导 vs supported_combinations 显式表交叉验证
- **stage 5** broadcast_legality — 算子计算的 broadcast 语义校验（numpy / none / explicit）
- **stage 6** boundary_min_set — 按 paradigms 检查 boundary/extreme case 是否覆盖最低集
- **stage 7** tolerance_coverage — `numerical_tolerance.per_dtype` 覆盖输出 dtype + 紧度启发式
- **stage 8** formula_smoke_eval — 把 formula 在小 shape (`[2,3]`) 上跑通，沙箱 numpy 执行
- **stage 9** oracle_reachable — 真 import framework + getattr 链找 api + 占位符校验

## 1. 何时使用

| 场景 | 使用分支 |
|---|---|
| 新算子起手，需要 spec.yaml 骨架 | **生成器**（§3） |
| 已有 spec.yaml，CI/PR 校验或 Designer Agent 重生成回灌 | **校验器**（§4） |
| 不清楚 26 category × 27 paradigm 怎么填 | 先读 §3.0 分类决策流程（强制读 `category_enum.yaml`），再看 [references/spec-cheatsheet.md](references/spec-cheatsheet.md) |
| 想看完整范例（11 个 example，全部 PASS 校验） | 看 [examples/](examples/README.md) |

## 2. 输入与输出

| 项 | 说明 |
|---|---|
| 入参（生成） | 算子名、category、paradigms[]、inputs（带 dtype_set）、outputs |
| 出参（生成） | `<output_dir>/spec.yaml`（含 TODO 占位符，需手填 formula / oracle.api / supported_combinations / boundary cases） |
| 入参（校验） | spec.yaml 路径 |
| 出参（校验） | 9-stage findings；返回码 0=PASS / 1=FAIL（含 internal_error）/ 2=YAML 解析错；`--strict` 时 warning 也退 1 |

## 3. 生成 spec.yaml

### 3.0 分类决策（前置必读）

> **⚠️ 强制前置步骤**：在调用 `generate_spec.py` 之前，**必须**先读取分类注册表确定 category 和 paradigms，禁止凭直觉猜测。

**步骤**：

1. **读 `registries/category_enum.yaml`**：26 类 category 的定义 + 示例算子列表。逐条对照算子功能，找到匹配的 category。
2. **读 `registries/paradigm_enum.yaml`**：27 项 paradigm 的分组和优先级。按 category 确定必含 paradigm，再按需追加辅助 paradigm。
3. **读 `registries/category_paradigm_map.yaml`**：确认 category ↔ paradigm 必含映射（stage 2 会校验此映射）。

**决策示例**：

| 算子 | 读 category_enum.yaml 匹配过程 | 结果 |
|------|------|------|
| tile | "不涉及计算，仅通过 shape 变换/广播等方式重组织数据；e.g. reshape, transpose, cat, **tile**" | `LayoutTransform` |
| slice | "按整数索引读取；e.g. gather, index_select" | `IndexGather` |
| add(a,b) | "算子计算需要对数据进行广播；e.g. add, mul, where" | `Broadcast` |
| softmax | "reduction + 数值稳定/归一化复合" + "多基础范式融合" | `ReductionComposite` |

**常见混淆**：

| 易混淆对 | 区分依据 |
|---------|---------|
| **Broadcast** vs **LayoutTransform** | Broadcast = 广播是计算的**手段**（add/mul 需要先对齐 shape 再计算）；LayoutTransform = 数据重组织本身就是**目的**（tile/reshape/transpose 不涉及计算） |
| **IndexGather** vs **LayoutTransform** | IndexGather = 按**整数索引数组**采集元素（gather/embedding）；LayoutTransform = 按**规则**重排/扩展/裁剪（reshape/tile/slice 的索引是确定性推导而非自由索引） |
| **Reduction** vs **ArgReduce** | Reduction = 输出**值**（sum/max）；ArgReduce = 输出**索引**（argmax/argmin） |

**禁止**：仅凭 cheatsheet 的启发式规则选 category（那些规则覆盖面有限，是辅助参考而非决策依据）。

### 3.1 交互式向导（推荐）

```
python3 scripts/generate_spec.py --output-dir <ops_dir>/<op_name>
```

向导会问：op 名称、category（26 类单选）、paradigms（在 category 必含基础上自由追加）、每个 input/output 名+dtype、promotion / broadcast / accumulation_order。

### 3.2 非交互式（CI / 脚本）

```
python3 scripts/generate_spec.py \
    --op-name softmax \
    --category ReductionComposite \
    --paradigms Reduction,NumericalStable,FusedComposite \
    --inputs "x:float16,float32,bfloat16" \
    --outputs y \
    --description "Softmax along reduce_axis with max-shift stabilization" \
    --output-dir ops/softmax
```

注：`--inputs` 多个时用 `;` 分隔（避免与 dtype_set 的 `,` 冲突），如 `"a:float16,float32;b:float16,float32"`。

### 3.3 自动注入

按 `op.paradigms` 自动注入对应字段（dim attribute / accumulator_dtype / seed / composition 占位等）。完整注入规则见 `registries/category_paradigm_map.yaml` 的 `paradigm_inject_hints` 段。

> **注意**：自动注入基于 paradigm 启发式推断，必须对照 REQUIREMENTS.md 中算子接口描述确认是否需要。当算子接口描述（REG_OP 宏、ACLNN API 参数表、GE IR 表格或自然语言描述）中无归约轴相关参数时，非交互模式应指定 `--axis-source fixed` 或 `--axis-source implicit_all`，避免错误注入 `dim`/`keep_dims` attribute。详见 [usage-scenarios.md](references/usage-scenarios.md) 中 attribute / axis_source 选择规则表。

**format_variants**：当算子支持多种数据排布（NCHW/NHWC/NCDHW 等）且归约轴或计算逻辑因格式而异时，使用 `--format-variants` 参数声明每种格式的具体参数（归约轴、oracle kwargs 等）。生成器会在 `math_semantics` 下注入 `format_variants` 段。

**paradigm_groups 自动注入**：使用 `--paradigm-groups combination` 或 `--paradigm-groups fusion` 声明范式组合模式。`combination` 模式下 Elementwise 不会被自动过滤（它代表独立的范式分支），生成器为每个 paradigm 生成一条 combination 组（switch/when 为 TODO 占位符，需手填）。交互模式下，当选了 ≥ 2 个范式时会自动询问。

### 3.4 必须手填的 TODO

生成器只产骨架 + 按 paradigm 自动注入的 boundary/extreme case。算子 owner 必须手填：
- `outputs[].shape_rule` — numpy 子集表达式（如 `c.shape = np.broadcast_shapes(a.shape, b.shape)`）。
  确定性 shape 的完整 input shape / attribute 依赖必须写在这里；`notes` / `global_constraints`
  / `REQUIREMENTS.md` 里的文字不算机器可执行规则。
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
- `math_semantics.format_variants` — 若算子支持多种数据排布（NCHW/NHWC/NCDHW 等）且计算因格式而异，校核每种格式的 reduction_axes 和 oracle_kwargs

## 4. 校验 spec.yaml

```
python3 scripts/validate_spec.py path/to/spec.yaml          # 文本输出，跑全 9 stage
python3 scripts/validate_spec.py path/to/spec.yaml --json   # 机器可读
python3 scripts/validate_spec.py path/to/spec.yaml --strict # 警告也 fail
python3 scripts/validate_spec.py path/to/spec.yaml --quiet  # 仅打 FAIL 的 stage
python3 scripts/validate_spec.py path/to/spec.yaml --stage 1 --stage 2  # 仅跑选中的 stage
```

| stage | 名称 | 核心检查 |
|-------|------|---------|
| 1 | schema_static | JSON Schema 字段校验（必填、类型、enum、pattern） |
| 2 | category_paradigm_consistency | category↔paradigm 必含映射 + paradigm_groups + paradigm 内部约束 + 白名单 |
| 3 | shape_closure | `outputs[].shape_rule` numpy_expr 求值 + data_dependent 分流 |
| 4 | dtype_closure | `outputs[].dtype_rule` 推导 vs supported_combinations 交叉验证 |
| 5 | broadcast_legality | broadcast 语义校验（numpy / none / explicit） |
| 6 | boundary_min_set | 按 paradigms 检查 boundary/extreme case 最低覆盖 |
| 7 | tolerance_coverage | per_dtype 容差覆盖输出 dtype + 紧度启发式 |
| 8 | formula_smoke_eval | formula 小 shape 沙箱 numpy 执行 |
| 9 | oracle_reachable | 真 import framework + getattr 链 + 占位符校验 |

完整子规则表、numpy 子集 API 列表、代码示例见 [references/stage-rules.md](references/stage-rules.md)。

## 5. 文件结构

```
ops/ops-spec-gen/
├── SKILL.md                              # 本文件
├── references/
│   ├── spec-cheatsheet.md                # 字段速查（按需阅读）
│   ├── stage-rules.md                    # 9-stage 完整子规则 + numpy 子集 API
│   ├── usage-scenarios.md                # 应用场景（场景二 + 场景五）
│   └── error-codes.md                    # rule_id 全表（自动生成）
├── examples/                             # 11 个 PASS 校验的范例（教学 + CI fixture）
│   ├── README.md
│   ├── add/spec.yaml                     #   Broadcast
│   ├── softmax/spec.yaml                 #   ReductionComposite + Broadcast + FusedComposite
│   ├── matmul/spec.yaml                  #   Contraction + Broadcast
│   ├── complex/spec.yaml                 #   LayoutTransform + Broadcast
│   ├── cumsum/spec.yaml                  #   Recurrence
│   ├── dropout/spec.yaml                 #   RandomSampling
│   ├── fused_quant_matmul/spec.yaml      #   FusedComposite + Broadcast + Quantization
│   ├── nonzero/spec.yaml                 #   VariableOutput + IndexGather
│   ├── reduce_sum/spec.yaml              #   Reduction（axis_source=input_tensor）
│   ├── reduce_sum_fixed/spec.yaml        #   Reduction（axis_source=fixed）
│   └── reduce_sum_all/spec.yaml          #   Reduction（axis_source=implicit_all）
├── registries/                           # 13 个 yaml 注册表（白名单+映射，可独立维护）
│   ├── category_enum.yaml                #   26 类 category
│   ├── paradigm_enum.yaml                #   27 项 paradigm
│   ├── category_paradigm_map.yaml        #   category↔paradigm 必含映射 + 注入提示
│   ├── primitive_whitelist.yaml          #   composition.primitives.op
│   ├── invariant_kind_registry.yaml      #   invariants.kind（value/algebraic/structural 三组）
│   ├── machine_check_kind_registry.yaml  #   boundary/extreme.machine_check.kind
│   ├── synthesize_pattern_registry.yaml  #   extreme_inputs.synthesize.patterns[].pattern
│   ├── error_code_enum.yaml              #   raises_error.error_type
│   ├── boundary_min_cases.yaml           #   各 paradigm 必含的 boundary/extreme case（stage 6 数据源）
│   ├── framework_oracle_registry.yaml    #   stage 9 已知 framework 列表
│   ├── tolerance_defaults.yaml           #   per_dtype 默认容差 + stage 7 紧度阈值
│   ├── chip_registry.yaml                #   芯片白名单
│   └── anti_pattern_registry.yaml        #   反模式 ID 注册表
├── schemas/op-spec.json                  # JSON Schema (Draft 2020-12)
├── templates/spec.yaml.tmpl              # spec.yaml 起手模板
├── scripts/
│   ├── generate_spec.py                  # 生成器（交互/非交互）
│   ├── validate_spec.py                  # 校验器主入口（完整 9 stage）
│   ├── check_registry_schema_sync.py     # registry↔schema 同步检查
│   ├── dump_rule_ids.py                  # rule_id 全表生成
│   └── evaluators/                       # numpy 子集 AST 求值器（stage 3-5/8/9 实现）
│       ├── _ast_sandbox.py               #   AST 白名单 / dunder 拒绝 / timeout
│       ├── types.py                      #   Dim / SymbolicShape / DslError
│       ├── parser.py                     #   parse_shape_literal
│       ├── shape_eval.py                 #   stage 3 SymbolicShapeEvaluator
│       ├── dtype_eval.py                 #   stage 4 DtypeEvaluator
│       ├── promote.py                    #   numpy 类型提升表
│       ├── broadcast.py                  #   numpy / none / explicit 广播模拟
│       ├── stages.py                     #   stage_3 / stage_4 / stage_5 入口
│       ├── formula_eval.py               #   stage_8 numpy 沙箱
│       └── oracle_check.py               #   stage_9 真 import + 占位符校验
└── tests/                                # 14 个测试文件
    ├── conftest.py
    ├── test_examples.py                  # pytest：所有 examples 必须 PASS
    ├── test_broadcast.py
    ├── test_doc_drift.py
    ├── test_dtype_eval.py
    ├── test_formula_eval.py
    ├── test_generate_spec.py
    ├── test_main_pipeline.py
    ├── test_oracle.py
    ├── test_promote.py
    ├── test_registry_schema_sync.py
    ├── test_sandbox_escape.py
    ├── test_shape_eval.py
    ├── test_shape_literal.py
    └── test_stage2_constraints.py
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
| 调整容差紧度阈值（stage 7 启发式）/ 默认容差 | `registries/tolerance_defaults.yaml` |
| 扩展 shape_eval / dtype_eval 支持的 numpy API | `scripts/evaluators/shape_eval.py` / `dtype_eval.py` |
| 调整 numpy 类型提升规则 | `scripts/evaluators/promote.py` |
| 调整沙箱白名单（AST 节点 / builtins） | `scripts/evaluators/_ast_sandbox.py` |
| 新增 stage 9 已知 framework | `registries/framework_oracle_registry.yaml` |

每次变更后，跑 examples 回归确认无破坏：
```
pytest tests/test_examples.py -v
```
预期：11 个 example 全 PASS。任一 FAIL 说明本次 schema/registry 变更破坏了向后兼容，需要回归 example 或回滚变更。

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

降级矩阵：

| 已装 | stage 8 | stage 9 真 import |
|---|---|---|
| pyyaml + jsonschema 仅 | SKIP | SKIP（占位符校验仍跑） |
| + numpy | ✓ | SKIP（占位符校验仍跑） |
| + numpy + torch | ✓ | ✓（torch 类 oracle） |
| + numpy + 全 framework | ✓ | ✓ |

## 8. 与下游 skill / agent 的关系

### 8.1 字段所有权声明

> 本节是 spec.yaml 字段所有权的**唯一权威声明**。Agent 不得自行维护字段所有权列表，必须引用本节。

#### 完整字段列表

以下字段以 `spec.yaml` 为唯一真值源，禁止从 `REQUIREMENTS.md` 正文重新推导、覆盖或自行扩展：

| # | 字段 | 说明 | spec 生成 | 测试 |
|---|------|------|:---------:|:----:|
| 1 | `op.category` | 算子主分类（26 类单选） | ✓ | |
| 2 | `op.paradigms` | 算子范式标签（27 项多选） | ✓ | |
| 3 | `op.paradigm_groups` | 范式分组（fusion=纵向融合，combination=横向组合） | ✓ | ✓ |
| 4 | `op.platform_constraints.supported_chips` | 适用芯片白名单 | ✓ | |
| 5 | `inputs` | 输入张量/标量/state | ✓ | ✓ |
| 6 | `attributes` | 非张量参数 | ✓ | ✓ |
| 7 | `outputs` | 输出张量 | ✓ | ✓ |
| 8 | `outputs[].shape_rule` / `outputs[].shape_rule_kind` | 输出 shape 推导规则 | ✓ | ✓ |
| 9 | `outputs[].dtype_rule` / `outputs[].dtype_rule_kind` | 输出 dtype 推导规则 | ✓ | ✓ |
| 10 | `shape_constraints` | 符号表 + 全局约束（`global_constraints` 为咨询性 notes） | ✓ | |
| 11 | `dtype_policy` | promotion + supported_combinations | ✓ | ✓ |
| 12 | `broadcast` | 算子计算的 broadcast 语义（数据复制/扩展） | ✓ | ✓ |
| 13 | `math_semantics` | formula + oracle + invariants + composition | ✓ | ✓ |
| 14 | `numerical_tolerance` | per_dtype 容差 | ✓ | ✓ |
| 15 | `boundary_conditions` | 合法但退化的输入 case | ✓ | ✓ |
| 16 | `extreme_inputs` | NaN / Inf / 全零等异常 case | ✓ | ✓ |
| 17 | `determinism` | 确定性保证 | ✓ | ✓ |
| 18 | `numerical_stability` | 数值稳定性技术 | ✓ | ✓ |

#### 字段职责分层

| 职责层级 | 字段 | 说明 |
|----------|------|------|
| **L0 数学约束** | `inputs` / `outputs` / `math_semantics` / `shape_constraints` / `dtype_policy` / `broadcast` / `paradigm_groups` | 算子"应该做什么"，下游不可覆盖 |
| **L0 边界语义** | `boundary_conditions` / `extreme_inputs` | 算子在边界/异常情况的行为预期，下游不可覆盖 |
| **L0 精度语义** | `numerical_tolerance` / `numerical_stability` / `determinism` | 算子精度要求，下游不可覆盖 |
| **已移出** | `test_matrix` | 测试生成参数由 `ascendc-st-design` skill 独立管理 |

### 8.2 测试配置管理

- **设计原则**：L0 规格只包含算子数学约束；测试生成参数由下游 skill 独立管理
- **下游职责**：`ascendc-st-design` skill 在 `operators/{op}/tests/st/design/02_测试配置.yaml` 中定义测试参数
- **边界/极端数据**：`boundary_conditions` / `extreme_inputs` 仍保留在 spec.yaml（是算子语义，不是测试参数）

### 8.3 上游与下游

- **上游**（提供）：Designer Agent 接收的算子需求 / REQUIREMENTS.md
- **下游**（消费）：
  - 9-stage L0 校验器以本 skill 输出的 spec.yaml 为输入
  - `ascendc-st-design` skill 用 `boundary_conditions / extreme_inputs` 作为测试用例来源
  - Developer Agent 按 `numerical_stability.techniques.anti_pattern_id` 触发反模式审计

## 9. 已知限制

- stage 3 numpy_expr 求值器暴露的 `np` 命名空间仅含 `broadcast_shapes` / `reduce_shape`；如需更多形状变换语义，需在 `scripts/evaluators/shape_eval.py` 扩展
- stage 5 v1 仅 2 输入广播；`explicit` rules 仅支持 `trailing + leading` 组合（matmul 形式）
- stage 6 用关键词子串匹配判 case 是否覆盖；spec 作者用罕见措辞时可能漏匹配
- stage 8 用 `[2,3]` 作为默认小 shape；`bfloat16` 用 `float32` 模拟
- stage 9 不真跑 oracle —— 只确认 callable 存在

## 10. 应用场景

| 场景 | 描述 |
|------|------|
| 从 REQUIREMENTS.md 生成 spec | 读取需求文档，调用生成器，手填 TODO，跑 9-stage 校验 |
| spec 独立评审 | 14 条 SPEC-\* 条款逐项对照 spec ↔ REQUIREMENTS，输出评审报告 |

详见 [references/usage-scenarios.md](references/usage-scenarios.md)。
