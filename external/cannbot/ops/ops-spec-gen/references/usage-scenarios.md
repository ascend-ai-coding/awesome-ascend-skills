# 应用场景

> 本文描述从 REQUIREMENTS.md 出发的完整操作步骤，
> 包括 spec 生成（场景二）和 spec 独立评审（场景五）。

---

## 场景二：从 REQUIREMENTS.md 生成 spec.yaml

### 强制规则

| ID | 规则 |
|----|------|
| S1 | 必须使用 `scripts/generate_spec.py` 生成骨架，**禁止手写 spec.yaml** |
| S2 | 生成完成后必须跑 `scripts/validate_spec.py spec.yaml` 11-stage 校验全 PASS（stage 9/10/11 SKIP 视为通过） |
| S3 | （建议）`scripts/compute_spec_hash.py` 工具链尚未交付，v1 不要求锁 spec_hash；待工具与 schema 字段就绪后启用 |
| S4 | 字段值必须**与 REQUIREMENTS.md 一致**——dtype / shape 约束 / 平台限制 / 容差由 REQUIREMENTS 推导，不允许凭空添加 |
| S5 | numerical_stability.techniques.anti_pattern_id 引用必须在 `registries/anti_pattern_registry.yaml` 中已注册（如未来 schema 加 enum） |
| S6 | **必须填 `op.platform_constraints.supported_chips`**（来自 REQUIREMENTS §2 运行环境；与 chip_registry.yaml 对齐） |
| S7 | （建议）`interface_binding.arg_order` / `aclnn` / `ge_ir` 字段尚未纳入 `schemas/op-spec.json`（顶层 `additionalProperties: false`），v1 不填；待 schema 扩展后启用 |
| S8 | （建议）`performance_budget` 同上，schema 未定义，v1 不填 |
| S9 | （建议）`performance_baseline` 同上，schema 未定义，v1 不填 |

### 执行步骤

**Step 1: 读取 REQUIREMENTS.md**，提取以下字段映射到 spec.yaml：

| REQUIREMENTS.md 字段 | spec.yaml 字段 |
|---|---|
| 算子类别 | `op.category` |
| 算子范式（多选） | `op.paradigms` |
| **范式间组合方式**（横向组合 / 纵向融合） | **`op.paradigm_groups`** + `--paradigm-groups` 参数（见下方说明） |
| **算子接口描述中的非张量参数声明** | **`attributes[]` + `--axis-source` 参数**（见下方说明） |
| **数据排布格式支持**（NCHW/NHWC/NCDHW 等） | **`math_semantics.format_variants[]`** + `--format-variants` 参数 |
| 输入张量列表 + dtype | `inputs[].name / dtype_set / shape.symbolic` |
| 输出张量 + dtype 推导规则 | `outputs[].dtype_rule / shape_rule` |
| 数学公式 | `math_semantics.formula` |
| 参考实现 / oracle | `math_semantics.reference_oracle` |
| 数值稳定性技术 | `numerical_stability.techniques` |
| 精度容差 | `numerical_tolerance.per_dtype` |
| 边界 case | `boundary_conditions[]` / `extreme_inputs[]` |
| **§2 运行环境（芯片号）** | **`op.platform_constraints.supported_chips`** |
| **§2 运行环境（DAV 宏 / CANN 版本）** | `REQUIREMENTS.md` 继续承载；schema 未定义时不要写入 spec |
| **§5 ACLNN API 接口（参数列表 / 顺序）** | _v1 暂缓_：`interface_binding.*` 尚未纳入 schema |
| **§6 GE IR 定义（IR 算子名 / 动态 shape）** | _v1 暂缓_：`interface_binding.ge_ir.*` 尚未纳入 schema |
| **§8 资源约束（workspace 上限 / 对齐）** | _v1 暂缓_：`performance_budget` 尚未纳入 schema |
| **§7 性能指标（利用率 / 带宽 / 延迟）** | _v1 暂缓_：`performance_baseline` 尚未纳入 schema |

**attribute / axis_source 选择规则**：

查看 REQUIREMENTS.md 中的算子接口描述（可能以 REG_OP 宏、ACLNN API 参数表、GE IR 表格或自然语言描述等形式出现），判断归约轴是如何指定的：

| 接口特征 | `--axis-source` 值 | 说明 |
|---|---|---|
| 归约轴作为非张量参数声明（如 REG_OP 的 `.ATTR(dim, ...)`、ACLNN 参数表中的 int64/list 类型参数、GE IR 表格的 ATTR 行） | `attribute`（默认） | 归约轴作为 attribute 传入 |
| 归约轴作为整型 tensor 输入声明（如 REG_OP 的 `.INPUT(axes, ...)`、ACLNN 参数表中的 aclTensor 类型参数） | `input_tensor` | 归约轴作为 tensor input |
| 无归约轴参数，归约轴固定（如 reduce 特定维度） | `fixed` | 归约轴硬编码在算子内部 |
| 无归约轴参数，reduce 所有轴（输出为标量） | `implicit_all` | 隐式归约所有轴，无需显式指定 |

> **注意**：当 REQUIREMENTS.md 的算子接口描述中无归约轴相关参数时，必须根据算子语义选择 `fixed` 或 `implicit_all`，不能使用默认的 `attribute`（否则会注入不存在的 `dim`/`keep_dims` attribute）。

**paradigm_groups 选择规则**：

当算子有多个范式且不同属性值触发不同计算模式时，需要声明范式间组合方式：

| 判断条件 | `--paradigm-groups` 值 | 说明 |
|---|---|---|
| 属性值切换导致计算模式根本性变化（如 reduction=none→逐元素 vs reduction=sum/mean→归约；mode=training vs mode=inference） | `combination` | 横向组合：多范式择一，由属性值决定激活哪个。每个 paradigm 独立一条 combination 组 |
| 多范式串联执行（如先 Elementwise 再 Reduction 再 Elementwise，所有范式始终参与） | `fusion` | 纵向融合：所有范式按序执行 |
| 仅 1 个范式，或所有属性值走同一范式 | 不传（默认） | 无需声明组合关系 |

> **关键判断**：检查 REQUIREMENTS.md 中是否存在"模式切换"属性——某个属性值使算子退化为完全不同的计算模式（如 `reduction=none` 使 Reduce 算子退化为逐元素、`training=False` 跳过 dropout）。若存在，必须选 `combination` 并在 spec 中补 Elementwise 范式。

**Step 2: 调用生成器**（非交互式，CI 友好）：

```bash
python3 ops/ops-spec-gen/scripts/generate_spec.py \
    --op-name {operator_name} \
    --category {category} \
    --paradigms {Paradigm1},{Paradigm2},... \
    --paradigm-groups {combination|fusion} \
    --inputs "{name1}:{dtype1},{dtype2};{name2}:{dtype1},..." \
    --outputs {name} \
    --axis-source {attribute|input_tensor|fixed|implicit_all} \
    --description "{REQUIREMENTS 中的一句描述}" \
    --output-dir operators/{operator_name}/docs
```

> `--paradigm-groups` 仅当有 ≥ 2 个范式且存在模式切换时使用。`combination` 模式下 Elementwise 不会被自动过滤。生成后需手填每组 `switch`（属性名）和 `when`（属性值）。
>
> `--axis-source` 仅对 Reduction 类算子有效。默认值为 `attribute`，需根据上方规则表对照 REQUIREMENTS.md 算子原型选择正确值。

注：交互式向导用法见 SKILL.md §3.1。

**Step 3: 手填 TODO 字段**（生成器只给骨架，详见 SKILL.md §3.4）：
- `math_semantics.formula` — numpy 可 eval 的表达式
- `math_semantics.reference_oracle` — 单 callable api，或填 absent=true + governance 签字
- `math_semantics.invariants` — **≥1 条值级、formula 无关**的语义不变量（保范/单位元/对称/代数恒等/极限退化，源自 REQUIREMENTS 数学；**非** shape 结构断言、**非** formula 自反重述）。**SPEC-ORACLE-1 硬性要求**，`reference_oracle.absent=true` 时强制
- `dtype_policy.supported_combinations` — 显式枚举 (input dtypes) → output dtypes
- `numerical_tolerance.per_dtype` — 覆盖输出 dtype（默认值见 `registries/tolerance_defaults.yaml`）
- **`op.platform_constraints.supported_chips`** — 来自 REQUIREMENTS §2，与 `registries/chip_registry.yaml` 对齐
- **`op.paradigm_groups[].switch` / `when`** — 当使用 `--paradigm-groups combination` 时，需手填每组的 `switch`（属性名）和 `when`（属性值），将属性值映射到对应的范式子集
- _v1 暂缓_：`interface_binding` / `performance_budget` / `performance_baseline` 尚未纳入 schema（顶层 `additionalProperties: false`），不要写入；待 schema 扩展后启用

**Step 4: 跑 11-stage 校验**：

```bash
python3 ops/ops-spec-gen/scripts/validate_spec.py operators/{operator_name}/docs/spec.yaml
```

预期 stage 1-8 全 PASS。stage 9 在测试机未装 torch 时走 SKIP（不算失败）。stage 10 在 oracle 可达时比对 formula 与 oracle 的特殊值输出，SKIP 亦不算失败。任一 FAIL 必须修复后重跑，**禁止跳过**。校验详情见 SKILL.md §4。

**Step 5: 锁 spec_hash**（建议）：`compute_spec_hash.py` 工具链 v1 未交付；不要求执行，待工具就绪后再纳入流程。

### 特殊算子类：属性决定多输出（split / chunk / unbind）

当算子**输出个数与 shape 由属性决定**（如 split 的 `split_size` 决定切几段、每段多大），区别于 nonzero/where 这类**输出个数由输入数据决定**的算子。这类算子用：

```yaml
op:
  category: VariableOutput
  paradigms: [VariableOutput]
  output_count_determined_by: attribute   # 关键：声明输出个数由属性决定
outputs:
  - name: y0
    shape_rule_kind: numpy_expr           # 输出 shape 由属性可静态推导
    shape_rule: |
      y0.shape = list(x.shape); y0.shape[dim] = split_size[0]
```

- `output_count_determined_by: attribute` 会**豁免** `data_dependent_shape: true` 强制要求，以及 VariableOutput 默认必含的 nonzero 类 boundary case（empty_output / full_output / normal_sparse）——这些是为"数据决定输出个数"设计的，不适用于属性决定。
- **list 属性**（如 `split_size: list[int64]`）在 `boundary_conditions` / `extreme_inputs` 的 synthesize 里书写时，加引号或不加均可：`attr.split_size: "[2,2]"` 或 `attr.split_size: [2,2]`（框架会按字面量解析注入 formula/shape_rule 沙箱，仅采纳 list/tuple，不会误转字符串属性）。
- 完整范例见 `examples/split/spec.yaml`。

### 完成标志

- spec.yaml 已生成并通过 11-stage 校验
- 字段与 REQUIREMENTS.md 内容一致（dtype / shape / 平台 / 容差均可追溯到需求）

---

## 场景五：spec 独立评审（17 条 SPEC-\* 条款评审）

> 在 CP1.5 用户人工 review 前，先做 17 条 SPEC-\* 条款级评审——逐项对照 spec ↔ REQUIREMENTS
> 中**机器可判**的项。把明显错误（dtype 漏一个、芯片不匹配、attribute 凭空注入、错误码缺漏、性能字段没填、
> 公式形式与 oracle 不一致）先拦下，避免拿一份"机器自洽但语义错"的 spec 去骚扰用户。

### 前置条件

- 已存在 `operators/{operator_name}/docs/spec.yaml`（11-stage 全 PASS）
- 已存在 `operators/{operator_name}/docs/REQUIREMENTS.md`

### 17 条 SPEC-\* 条款

| 条款 ID | 检查项 | 数据来源对照 |
|---|---|---|
| **SPEC-CHIP-1** | spec.op.platform_constraints.supported_chips ⊆ REQUIREMENTS §2 目标芯片 | 字符串集合包含关系 |
| **SPEC-DAV-1** | _v1 暂缓_ — DAV 宏由 REQUIREMENTS / DESIGN 承载，`dav_macros` 尚未纳入 schema | — |
| **SPEC-DTYPE-1** | spec.dtype_policy.supported_combinations 输入 dtype 集 = REQUIREMENTS §4 支持类型集 | 集合相等 |
| **SPEC-DTYPE-2** | spec.inputs[].dtype_set 覆盖 REQUIREMENTS §4 数据类型 | 集合包含 |
| **SPEC-IO-1** | spec.inputs/outputs 数量 + name 与 REQUIREMENTS §5 ACLNN 参数列表对齐 | 长度 + 名字集合 |
| **SPEC-ATTR-1** | spec.attributes[].name 集合 ⊆ REQUIREMENTS 算子接口描述中声明的非张量参数集合（接口描述中无归约轴等 attribute 类参数时 spec.attributes 应为空） | 集合包含 |
| **SPEC-ARG-1** | _v1 暂缓_ — `interface_binding.arg_order` 尚未纳入 schema，待扩展后启用 | — |
| **SPEC-ERROR-1** | spec.op.error_codes ⊇ REQUIREMENTS §8 错误码集合 | 集合包含 |
| **SPEC-PERF-1** | _v1 暂缓_ — `performance_baseline` 尚未纳入 schema，待扩展后启用 | — |
| **SPEC-RES-1** | _v1 暂缓_ — `performance_budget` 尚未纳入 schema，待扩展后启用 | — |
| **SPEC-FORMULA-1** | spec.math_semantics.formula 至少引用所有 input name | 字符串包含 |
| **SPEC-FORMULA-2**（硬拦） | spec.math_semantics.formula 与 reference_oracle 在含特殊值（NaN/inf/大数/带符号零）输入上产出**一致的 NaN/inf/零符号模式**且有限值在容差内。stage 10 校验 PASS 或 SKIP 视为通过；stage 10 FAIL → ❌（详见下方判定细则） | stage 10 机器校验 |
| **SPEC-LAYOUT-1** | canonical 通道轴/规约轴须：① 显式（`formula` 可判）② 各 boundary/extreme 用例 `(C,) len == x.shape[该轴]` 自洽 ③ 等于 REQUIREMENTS 调用框架默认 format 通道位（无收窄豁免）。任一不满足 → ❌（详见下方判定细则） | formula 轴位置 + 调用框架默认 format 通道位 + 用例 shape 自洽 |
| **SPEC-ORACLE-1** | `math_semantics.invariants` 须含 ≥1 条**值级、formula 无关**的独立不变量（源自 REQUIREMENTS 数学；非结构断言、非 formula 自反）；`reference_oracle.absent=true` 时**强制**。无独立锚点 → ❌（详见下方判定细则） | reference_oracle 独立性 + invariants 值级且非 formula 自反 |
| **SPEC-PARADIGM-1** | spec.op.paradigms 与 category 隐含范式 + REQUIREMENTS 暗示的修饰范式对齐 | 集合差 |
| **SPEC-LIFECYCLE-1** | spec.op.lifecycle 与 REQUIREMENTS 描述匹配（experimental vs stable）| 字符串匹配 |
| **SPEC-INTERFACE-1** | _v1 暂缓_ — `interface_binding.*` 尚未纳入 schema，待扩展后启用 | — |

> **SPEC-LAYOUT-1 判定细则**（三查均为 ❌ 硬拦；收窄合理性另列必看清单供人复核，但**不改 ❌ 判定**）：
> - **① 显式（❌ 硬拦）**：formula 无法判出通道轴/规约轴（既没用 `x.shape[k]` 也无 shape_constraints 说明）→ ❌（轴不显式，find-proto 无从写进 proto §4，黑盒只能自行猜轴）。
> - **② 用例自洽（❌ 硬拦）**：任一 boundary/extreme 用例的 `(C,)` 张量长度 ≠ `x.shape[该轴]` → ❌。
> - **③ 与「调用框架默认 format」一致（❌ 硬拦，无收窄豁免）**：spec canonical 轴**必须等于** REQUIREMENTS「调用框架/主 format」的通道位（PyTorch=NCHW→轴1、TF=NHWC→轴-1 等）。不等 → **❌**，**即使 spec 声明"由适配层映射"也不豁免**——工作流不产出 transpose/适配层，黑盒直调 aclnn 传入的就是框架 layout 张量，轴不匹配必与 golden/kernel/用例撞轴（bn_training_update：spec 末轴、黑盒 NCHW → 535 全挂）。**唯一豁免**：transpose/适配层是本工作流显式交付物（默认没有）。

> **SPEC-ORACLE-1 判定细则**（golden 与 kernel 同源 spec.formula，共享错误则 matches_oracle 假绿；须挂一个不经过 formula 的独立锚点才测得出公式语义错）：
> - **① 值级不变量存在（absent=true 时 ❌ 硬拦）**：`invariants` 全是 `shape_equals`/`no_leak` 等结构项、无任何值级语义性质 → ❌。值级性质示例：保范 `‖out‖=‖in‖`（旋转/正交类）、单位元 `f(x, 恒等参数)=x`、对称/反对称、代数恒等（softmax 行和=1、归一化后统计量）、极限退化。
> - **② 独立性（非 formula 自反）**：值级不变量不得是 formula 的直接重述（把公式抄成断言不算独立）；须能在**不引用 formula** 的前提下由 REQUIREMENTS 数学独立推出。疑似自反 → ⚠ 入必看清单人复核。
> - **③ absent 不免验证**：`reference_oracle.absent=true` 时①为强制；absent=false 且指向真实独立外部实现时①可放宽为建议（但仍推荐同时有）。

> **SPEC-FORMULA-2 判定细则**（公式形式差异是 spec→design→kernel→黑盒 全链路返工的最大单因——数学等价但计算不等价的公式在 NaN/inf 传播与末位舍入上发散，导致 kernel 与 golden bit 级背离）：
> - **① stage 10 机器校验**：跑 `validate_spec.py` stage 10（formula_oracle_equiv）。stage 10 在含 NaN/inf/大数/带符号零的输入上同时执行 formula 和 oracle，比对 NaN 模式、inf 模式、零符号模式、有限值容差。PASS 或 SKIP → ✓；FAIL → ❌。
> - **② SKIP 不等于豁免**：stage 10 在以下情况 SKIP（视为通过，不阻塞）：formula_kind 非 numpy_expr、oracle absent、composition DAG 模式、framework 未安装、oracle 调用失败。SKIP 时应在报告中注明跳过原因，并提示用户在安装对应 framework 后重跑。
> - **③ 典型陷阱**：教科书合并形式 `m_new = β1·m + (1-β1)·grad` 与 TF 增量形式 `m = m + (1-β1)·(grad-m)` 数学等价但浮点不等价——当 `m=inf` 时前者产出 `inf`，后者产出 `NaN`（`inf - inf = NaN`）。stage 10 的 nan/inf 输入组会捕获此类差异。

### 执行步骤

1. **逐条对照 spec ↔ REQUIREMENTS**：按 17 条条款表逐项评审，每条输出 ✓/⚠/❌ + 证据（spec 字段值与 REQUIREMENTS 来源的对照）
2. **生成必看清单**：列出 agent 独立评审无法判但必须由人 review 的项（见下文）
3. **状态判定**：任一 ❌ → 状态=❌失败；全 ✓ 或 ⚠ → 状态=✅通过（⚠ 提示用户但不阻塞）
4. **输出 SPEC_REVIEW.md**：按下文报告格式模板输出

> **注意**：本场景只读、只评审、只输出报告；不得修改 spec.yaml。修复由场景二（spec-generation）执行。

### 必看清单（用户对照摘要必含）

agent 独立评审无法判但必须由人 review 的项：

> **强制规则**：以下每一项**必须**出现在评审报告的"必看清单"中。即使判断为 ✓ 也必须输出并附简要证据，不允许省略。

| 必看项 | 为什么 agent 判不了 |
|---|---|
| 公式数学意图 | "y = (x - mean) / sqrt(var + eps)" 写得对但用户真想要的可能是 RMSNorm 不是 LayerNorm |
| tolerance 数值合理性 | 1e-3 还是 5e-3？需要算子领域知识 + 上下游精度标准 |
| boundary case 业务覆盖 | 业务上 K 维 > 4096 是否常见？需领域知识 |
| composition 拆分合理性 | FusedComposite 算子的 primitives 拆分是否符合预期融合方式 |
| reference_oracle 选择 | 选 torch.matmul 还是 torch.linalg.matmul？两者数值差异在边界 case 可能很大 |
| **范式选择正确性** | spec.op.paradigms 列出的范式是否完整覆盖了算子的所有计算路径？典型陷阱：reduction=none 时算子退化为 Elementwise，但 spec 只声明了 Reduction + FusedComposite，漏掉了 Elementwise。agent 只能校验"声明的范式是否与 category 对齐"，无法判断"是否漏了一条独立的计算路径" |
| **范式组合方式（paradigm_groups）** | 多个范式之间的关系是横向组合（combination，按属性值择一激活）还是纵向融合（fusion，串联执行）？agent 只能做结构校验（switch/when 格式），无法判断组合方式是否符合算子实际的分发逻辑。典型陷阱：存在模式切换属性但未声明 paradigm_groups，导致下游 design 无法生成分区 TilingKey |
| **REQ「调用框架默认 format」判定是否可靠（SPEC-LAYOUT-1 ③）** | ③ 自动硬拦"spec 轴 ≠ 调用框架默认 format 通道位"。此必看项仅辅助复核：REQ 并列多框架、主次不清时人需定夺取哪个为准，避免 ③ 误判 |
| **不变量是否真独立 & 覆盖是否够（SPEC-ORACLE-1）** | agent 判得了"有没有值级不变量"，判不了"它是否是 formula 的隐性重述、是否强到能捕获真实语义错"。人需确认：所列不变量确实由 REQUIREMENTS/论文独立推出、且强度足以卡住 formula 的常见错（如 RoPE 的保范可卡住 cos/sin 扩展 tile/repeat 用错） |

### 报告格式（精确模板，供主 Agent 机读判定）

```markdown
**状态**: ✅通过 / ❌失败

**spec.yaml 路径**: operators/{op}/docs/spec.yaml
**REQUIREMENTS.md 路径**: operators/{op}/docs/REQUIREMENTS.md

## 17 条 SPEC-* 条款评审

| 条款 ID | 状态 | spec 字段值 | REQUIREMENTS 来源 | 证据 / 备注 |
|---------|------|-------------|------------------|------------|
| SPEC-CHIP-1   | ✓ | [Ascend910B, Ascend910D] | §2 Atlas A2/A3 训练系列 | 字段值与需求对齐 |
| SPEC-DAV-1    | ⚠ | v1 暂缓 | §2 编译宏 | DAV 宏尚未纳入 spec schema，由 REQUIREMENTS / DESIGN 承载 |
| SPEC-DTYPE-1  | ⚠ | {fp16, bf16}        | §4 fp16/bf16/fp32     | spec 漏 fp32；用户确认是否真要去掉 fp32？ |
| ...           | ... | ...                 | ...                   | ... |

## 必看清单（CP1.5 人工 review 用）

⚠ **公式数学意图**：spec.formula="y = np.exp(x) / np.exp(x).sum(axis=dim)"
   ⚠ 注意：未做 max-shift 数值稳定。REQUIREMENTS §4 公式同未提；但 NumericalStable 范式已声明。
   → 人工确认：是否真不做 max-shift？

⚠ **tolerance 数值合理性**：fp16 用 1.0e-3，bf16 用 4.0e-3
   → 来自 ops-precision-standard 默认；是否需要按业务收紧？

✓ **boundary case 业务覆盖**：含 reduce 轴=1 / rank=0 / 空 Tensor / fp16 上溢边界 共 4 条
   → 自动覆盖 Reduction + NumericalStable 范式必含 case；业务关键路径请人工确认

⚠ **范式选择正确性**：spec.paradigms=[Reduction, FusedComposite]
   → 人工确认：是否存在某些属性值使算子退化为完全独立的计算模式（如 reduction=none→逐元素）？
   若是，需要添加 Elementwise 范式并声明 paradigm_groups combination。

⚠ **范式组合方式**：spec.paradigm_groups 未声明
   → 人工确认：多个范式之间是横向组合（按属性值择一）还是纵向融合（串联执行）？
   若存在模式切换属性，需声明 paradigm_groups。

## 问题清单（仅状态=❌时必填）

| 条款 | 严重度 | 问题描述 | 修复建议 |
|------|--------|---------|---------|
| ...  | HIGH/MED/LOW | ... | ... |
```

### 主 Agent 处理规则（供调用方参考）

- 状态=✅ → 进入 CP1.5 人工确认
- 状态=❌ → 主 Agent 自动调 (scene: spec-generation) 按 SPEC_REVIEW 修订 spec.yaml，修订后**重跑 11-stage + 重跑本场景**；最多重试 2 次
- 禁止把 ❌ 报告直接抛给用户
