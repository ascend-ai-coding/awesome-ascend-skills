# Task A Step 2：生成 Evaluator 和 Case Generator

## 职责

Step 2 是代码翻译阶段。根据 Step 1 产物，以两个模板为基础，仅补全动态区，生成可执行的 route evaluator 和 case generator。

生成器必须按以下顺序工作：

```text
分支约束代码生成 -> 约束域随机采样 -> route witness 验证 -> case_values 字段去重/多样性扩展 -> route 稳定性复验
```

Step 2 不重新分析源码、不修改 Step 1 的字段或分支模型、不构造完整算子实例。最终 JSON 由 Step 3 运行脚本生成和验收。

## 输入与输出

| 文件 | 用途 |
|---|---|
| `S2P1_route_dispatch_model.md` | 字段全集、合法域、分支条件和 evaluator 规则 |
| `S2P1_tiling_glossary.md` | 字段语义、推导关系和 Step 5 映射参考 |
| `s2p1_route_extractor_template.py` | evaluator 模板 |
| `s2p1_case_generator_template.py` | case generator 模板 |

输出：

```text
S2P1_route_extract.py
S2P1_case_generator.py
```

## 模板落地

在读取 Step 1 产物前，先复制模板：

```bash
cp <skill_base>/references/task-a/s2p1_route_extractor_template.py <whitebox>/S2P1_route_extract.py
cp <skill_base>/references/task-a/s2p1_case_generator_template.py <whitebox>/S2P1_case_generator.py
```

随后只允许编辑两个脚本的动态区。静态区的参数解析、字段校验、route 执行、输出和审计逻辑以模板为准，不得重复实现或改写。

## Evaluator 动态区

必须实现模板提供的以下接口：

```python
get_case_value_fields()
build_source_info(ctx)
eval_tiling_route(case)
eval_kernel_route(case, tiling)
build_kernel_branch_catalog(ctx)
```

要求：

- `get_case_value_fields()` 返回 Step 1 定义的完整固定字段集合。
- `case_values` 中的变量命名必须与 tiling 侧变量命名保持一致，禁止修改。
- `build_source_info()` 返回 P0 源码范围和证据。
- `eval_tiling_route()` 只从 `case["case_values"]` 读取外部字段，模拟真实 tiling 路由并返回推导事实和证据。
- `eval_kernel_route()` 根据 `case_values`、平台上下文和 tiling 结果模拟 kernel dispatch。
- `build_kernel_branch_catalog()` 返回完整 kernel dispatch 目录，用于 orphan 审计。
- 内部推导变量、等价变量和 route key 不得写回 `case_values`。
- evaluator 不构造或校验完整输入、输出、shape、rank 或接口实例。
- 平台内存量有效值必须按 Step 1 的固化规则计算：`ASCEND910B` 用 `ini_ub_size − 256`、`ASCEND950` 用 `ini_ub_size`（其余架构 unresolved，`ini_ub_size` 以字节计），并在所有由该值推导的分支阈值中使用该有效值，不得硬编码 ini 名义值。

## Case Generator 动态区

动态区只实现：

```python
generate_candidate_cases()
```

每条 candidate 必须包含：

```text
branch_group
sub_branch（可选）
case_values
expected（可选）
purpose
evidence
```

要求：

- 按 tiling 主分支组织 candidate，必要时再按子条件细分。
- 每条 `case_values` 使用 `get_case_value_fields()` 返回的完整字段集合，不得按目标分支裁剪。
- 当前分支未直接约束的字段也必须填入合法值，并满足跨字段约束。
- 动态区必须强制全局输入规模上限：由候选 `case_values` 推导出的算子全部输入元素总字节数（`Σ 各输入元素数 × dtype 字节宽度`，仅统计算子输入、不含输出）必须严格小于 `1GiB`（`1024^3 B`）。超出上限的候选必须重采样，且不得为满足该上限而破坏目标分支的成立条件。
- `case_values` 只包含 Step 1 声明的外部代表字段，不包含等价变量、内部推导变量或完整实例字段。
- Step 2 必须原样使用 Step 1 确定的 `case_values` 字段名和字段集合，不得重新筛选、重命名或加入未在 Step 1 中确定的字段。
- `expected` 只描述 route 验证事实，不把内部推导变量写入 `case_values`。
- 必须根据 `S2P1_route_dispatch_model.md` 中的 `branch_conditions`、`branch_value_constraints` 和 `branch_priority` 生成分支约束判断代码。
- 不得将每个分支实现为固定 `branch_values`、固定 witness 表、固定 seed pool 或有限候选值列表。
- 数值字段必须在当前分支合法域内随机采样；枚举字段必须在合法枚举域内轮转或随机采样。
- 随机采样必须使用固定 seed 控制可复现性，但固定 seed 不得等价于固定候选表。
- 每条 candidate 的 `expected` 必须包含目标 route 事实（如 branch/mode、dtype、norm_mode 或 tiling_key），不能只校验 `reachability`。
- 随机采样、字段去重和目标 route 复验必须在动态区生成逻辑中完成；静态框架对动态区返回的 candidate 只负责统一校验和记录，不得假设静态框架会为同一 candidate 自动产生新样本。
- 不覆盖静态区的字段校验、case object 构造和 route 执行函数。

## 分支覆盖与多样性

### 分支覆盖

- 覆盖 Step 1 声明的所有可达 tiling route 目标。
- 覆盖所有可达 kernel dispatch 目标。
- 合法 tiling 但没有 kernel dispatch 的情况保留为 tiling-only candidate。
- 不可达 kernel 分支保留在 catalog 中，由 Step 3 标记 orphan，不得伪造内部 key。
- 每个 candidate 必须满足目标分支的成立条件、前置排除条件、合法域和跨字段约束。

### 字段多样性

字段多样性不是在固定 witness 表之后追加少量 variants，而是 candidate 生成过程本身的约束。Generator 应在每个目标 route 的合法分支域内随机采样 `case_values`，并全局跟踪每个字段已经使用过的值。对于新的 route case，优先选择当前字段未使用过的合法值；仅当合法域有限、分支条件强约束、跨字段约束导致无法保持 route 稳定，或重试达到上限时，才允许重复字段值。

优先级为：

```text
分支覆盖 > 字段合法性 > route 稳定性 > kernel 可达性分类正确 > 字段值不重复
```

多样性扩展要求：

- 有限枚举字段尽量覆盖多个合法值。
- 数值字段优先使用边界值、边界邻近值和内部合法值。
- 未被当前分支直接约束的字段也应尽量产生合法变体。
- 每次变体都必须重新通过 evaluator 验证 expected route。
- 允许使用模板静态区提供的辅助工具，但不得把辅助工具输出作为固定候选池；候选值合法性和随机采样边界由动态区依据 Step 1 规则保证。

动态区生成停止条件不得由固定候选池大小决定。应由以下条件共同决定：

- 声明的 tiling route 目标已覆盖。
- 可达 kernel dispatch 目标已覆盖，tiling-only / orphan 已正确归类。
- `case_values` 代表字段的取值多样性达到当前分支合法域可支持的水平，或多轮采样不再产生新的合法非重复值。
- 达到最大采样次数或安全 case 数上限，防止死循环。

以下情况允许重复字段值：

- 分支条件强制字段固定。
- 合法值域已耗尽。
- 替换值会触发更高优先级分支或改变 expected route。
- 替换值会导致 unsupported 或 unresolved。
- 字段存在联合约束，无法独立替换。

不得通过加入等价变量、内部推导变量、shape、rank 或接口构造字段制造多样性。

## 边界与完成标准

- Step 2 不构造完整 inputs、outputs、attrs 或最终 shape；这些由 Step 5 Mapper 承接。
- Step 2 不读取或修改 Step 3、Step 5 产物。
- 不规定固定 seed pool、固定 case 数量或固定候选数量。
- 不允许以固定 witness 表、固定分支取值表或有限候选池替代分支约束域随机采样。
- 两个目标脚本均无 `TODO` / `NotImplementedError`，并通过 `py_compile`。
- evaluator 已实现五个动态接口，generator 仅实现 `generate_candidate_cases()`。
- 所有 candidate 的字段集合与 evaluator 契约完全一致。
- candidate 覆盖 Step 1 声明的可达 route，并正确保留 tiling-only 和 orphan 语义。
- 分支覆盖通过分支约束代码和随机采样完成；每个字段已尽量生成非重复合法取值，且每条 case 均由 evaluator 复验 route 稳定性。
- `S2P1_route_cases.json`、`S2P1_path_list.json` 和 `S2P1_cases.json` 的生成与验收由 Step 3 完成。
