# 算子 spec.yaml 速查表

> 字段速查；权威约束以 `schemas/op-spec.json` 与 `registries/*.yaml` 为准。
> 子规则（rule_id 列表与失败码）见 [stage-rules.md](stage-rules.md)，阶段概览见 [SKILL.md §4](../SKILL.md)。

---

## A. 顶层 18 个字段（test_matrix 已移出）

### A0. 字段职责分层

| 职责层级 | 字段 | 说明 |
|----------|------|------|
| **契约版本** | `schema_version` / `support_universe` | `support_universe` 为 schema 可选字段，新生成的完整全集契约必须携带；旧规格可省略 |
| **L0 语义** | `reduction` | 可选归约轴来源及固定轴配置 |
| **L0 语义** | `op` / `inputs` / `attributes` / `outputs` / `semantic_cases` / `layout_contract` / `shape_constraints` / `dtype_policy` / `broadcast`（计算是否需要数据广播） / `math_semantics` | 数学约束，下游不可覆盖 |
| **L0 语义** | `boundary_conditions`（合法但退化的输入 case）/ `extreme_inputs`（NaN / Inf / 全零等异常输入 case） | 下游不可覆盖 |
| **L0 语义** | `numerical_tolerance` / `numerical_stability` / `determinism` | 精度语义，下游不可覆盖 |
| **已移出** | `test_matrix` | 测试生成参数由 `ascendc-st-design` skill 独立管理 |

当 `REQUIREMENTS.md` 与 `spec.yaml` 共存时，`spec.yaml` 只承载已经锁定的结构化 L0 契约；需求背景、接口自然语言说明、资源/性能目标来源等仍由 `REQUIREMENTS.md` 承载。下游方案设计和测试设计必须按对应 Agent 的「输入优先级与字段所有权」执行：上表 L0 语义字段以 `spec.yaml` 为准，冲突时停止并报告，不得从需求正文重新解释后覆盖。

```yaml
schema_version: 1          # 固定值
support_universe: huawei-cann-types-2026-10-15  # schema 可选，新契约必带；旧契约省略
op:                        # 元信息：name / version / description / category（单选 26 类） / paradigms（多选 27 项） / paradigm_groups（范式分组）
inputs: []                 # 张量/标量/state，按位置排列
attributes: []             # 非张量参数，含 machine_constraint
outputs: []                # 用 numpy 子集表达式描述 shape/dtype 推导规则
semantic_cases: []         # 可选输入/输出的条件存在性；when 只允许 attr.<name>、input.<name>.dtype 或 input.<name>.is_present
layout_contract: {}        # 可选逻辑布局变体：selector + 已声明 Tensor 的 logical_axes；不替代各 I/O 的 layout
reduction: {}              # 可选归约轴来源及固定轴配置
shape_constraints: {}      # 全局符号表 + global_constraints（咨询性字段，当前不参与 12-stage 机器校验，见 §D.8）+ notes
broadcast: {kind: none}     # 数据广播契约
dtype_policy: {}           # promotion + supported_combinations 显式枚举 + accumulator_dtype
math_semantics: {}         # formula + reference_oracle + invariants + composition (FusedComposite 必填)
numerical_stability: {}    # required + techniques (含 anti_pattern_id)
numerical_tolerance: {}    # per_dtype: {rtol, atol, metric}
boundary_conditions: []    # 机器可断言的退化/越界 case（L0 语义）
extreme_inputs: []         # NaN / Inf / 全零等异常 case（L0 语义）
determinism: {}            # accumulation_order + bitwise_reproducible（语义详解见 [accumulation_order.md](accumulation_order.md)）
# test_matrix 已移出，由 ascendc-st-design skill 独立管理
```

---

## B. 关键枚举与白名单

### B1. 26 类 category（单选）

```
Elementwise · Reduction · ReductionComposite · Contraction · LayoutTransform
IndexGather · ScatterUpdate · MaskPredicate · SortSelect · FusedComposite
RandomSampling · SlidingWindow · Collective · ControlFlow · Recurrence
Interpolation · Quantization · Stateful · ArgReduce · Sparse · Padding
VariableOutput · Spectral · Histogram · AtomicUpdate · Broadcast
```

### B2. 27 项 paradigm（PascalCase，多选）

```
基础 6      Elementwise · Broadcast · Reduction · Contraction · ArgReduce · LayoutTransform
访存 5      Padding · IndexGather · ScatterUpdate · AtomicUpdate · Interpolation
控制 3      MaskPredicate · SortSelect · Histogram
结构 3      SlidingWindow · ControlFlow · Recurrence
算法 2      NumericalStable · Spectral
形状状态 4  DynamicShape · VariableOutput · RandomSampling · Stateful
其他 4      Sparse · Quantization · CollectiveCommunication · FusedComposite
```

典型组合：
- `softmax` = Reduction + NumericalStable + FusedComposite
- `conv2d` = SlidingWindow + Contraction + Padding
- `flash_attention` = Contraction + Reduction + NumericalStable + FusedComposite + DynamicShape
- `scatter_add` = AtomicUpdate + DynamicShape
- `lstm` = Recurrence + Contraction + Stateful

### B3. category ↔ paradigm 必含映射（节选；完整在 `registries/category_paradigm_map.yaml`）

| category | 必含 |
|---|---|
| ReductionComposite | Reduction + FusedComposite |
| FusedComposite | FusedComposite + ≥ 2 基础 paradigm（Elementwise 被其他范式吸收，不计入） |
| SlidingWindow | SlidingWindow（通常再带 Contraction 或 Reduction） |
| ArgReduce | ArgReduce（输出 dtype_rule 必须求值为 `np.int32` 或 `np.int64`） |
| AtomicUpdate | AtomicUpdate（与 ScatterUpdate 互斥） |

#### 分类决策（从 REQUIREMENTS.md 推导）

> **⚠️ 分类权威来源是 `registries/category_enum.yaml`**（26 类定义 + 示例算子）。以下启发式仅作快速参考，不可替代读注册表。

分析算子的输入数量、计算模式、shape 推导规则，对照 `category_enum.yaml` 中的定义和示例选择 category：

- 不涉及计算，仅重组织数据（shape 变换/轴重排/拼接拆分/广播扩展/切片）→ 读 category_enum.yaml 中 `LayoutTransform` 的定义和示例列表
- 2+ tensor 输入，**计算本身需要**广播对齐 shape（如 add/mul/where）→ 读 category_enum.yaml 中 `Broadcast` 的定义和示例列表
- 仅 1 tensor 输入且逐元素计算 → 读 category_enum.yaml 中 `Elementwise` 的定义和示例列表
- 涉及归约 → 读 category_enum.yaml 中 `Reduction` / `ReductionComposite` 的定义和示例列表
- 涉及矩阵运算 → 读 category_enum.yaml 中 `Contraction` 的定义和示例列表
- 按整数索引采集元素 → 读 category_enum.yaml 中 `IndexGather` 的定义和示例列表

**易混淆场景**：
- **tile/repeat/expand** → LayoutTransform（数据重组织是目的，不是计算手段）
- **add/mul 的 shape 对齐** → Broadcast（广播是计算的前置步骤）
- **slice/gather 的区别** → 确定性裁剪按 LayoutTransform 的数据重组织语义判断；自由整数索引采集按 IndexGather 判断，具体以注册表定义和真实接口为准

记录决策理由（输入数量 → category_enum.yaml 匹配项 → category → paradigms 推导链）。

### B4. shape_rule (numpy_expr)

`outputs[].shape_rule` 是受限 numpy 子集表达式（不再是 DSL 函数）。完整语义、子规则和失败码见
[stage-rules.md 的 stage 3](stage-rules.md)。

| kind | 适用 | shape_rule 内容 |
|---|---|---|
| `numpy_expr` | 输出形状可静态从输入 shape + attribute 推出（绝大多数算子） | numpy 表达式 |
| `data_dependent` | 输出形状由输入**值**决定（nonzero / unique / masked_select 等 VariableOutput；splitv 等 DynamicShape value-driven） | 仅允许 `op.paradigms` 含 `VariableOutput` 或 `DynamicShape`；不写表达式，写 `shape_rule_description` + `shape_bounds` |
| `textual_only` | 输出形状因数据排布**格式**而异（如 NCHW vs NHWC 的 Channel 轴位置不同） | 仅允许 `math_semantics.format_variants` 存在；可含 `${format_variants[].channel_axis}` 占位符；必须配 `shape_rule_description` |

`inputs[].shape.symbolic` 列表元素三类：显式维（`"M"` 大写） / 折叠维（`"...d"` 小写，仅可作首元素） / 常量维（整数）。

> **`symbolic` 名字仅为 owner 自留命名，不参与跨 input 绑定**：两个 input 都写 `"K"` 或都写 `"...batch"` 不会被 evaluator 视为"必相等"。跨 input 形状关系应通过 `outputs.shape_rule` 表达并由 stage 3 求值；适用且有 numpy 时，该阶段还比较 shape_rule 与公式实际输出 shape。stage 5 校验广播语义，stage 8 试执行公式，均不能替代完整的输出 shape 规则。

形状规则支持维度乘以非负整数、除以正整数并向下取整，例如 `weight.shape[-1] // 2`。符号结果保留表达式（`N * 2` 不会变成常量 `2`），常量维度直接计算；未知数量的折叠维、除零和非法除数明确失败。`//` 是取整，不自动推定整除性；输入必须为偶数等约束仍由原规格声明。该能力不代表完整符号代数证明。

### B5. dtype_rule (numpy_expr)

`outputs[].dtype_rule` 同样是 numpy 子集表达式。常见形态：

| 写法 | 用途 |
|---|---|
| `y.dtype = x.dtype` | 与某 input 相同 |
| `c.dtype = np.promote_types(a.dtype, b.dtype)` | numpy 类型提升 |
| `c.dtype = np.result_type(a.dtype, b.dtype, c.dtype)` | 多输入类型提升 |
| `y.dtype = np.int32` / `np.float16` / `np.complex64` | 固定 dtype |
| `out.dtype = np.complex64 if real.dtype == np.float32 else np.complex128` | IfExp 条件分支 |

以下为常用 dtype 速查，并非全量枚举。完整 spec 词汇见 `schemas/op-spec.json` 的
`$defs.dtype.enum`；官方全集以同一 schema 的 `$defs.official_dtype.enum` 为准，
名称映射和上下文限制见 `x-support-catalog`，边界与能力要求见 [官方全集契约](support-universe.md)。

| 组 | dtype | 来源 / 备注 |
|---|---|---|
| 标准浮点 | `float16` · `float32` · `float64` · `bfloat16` | numpy 原生 |
| 窄浮点（ascend 私有） | `float4_e2m1` · `float4_e1m2` · `float8_e4m3fn` · `float8_e5m2` · `float8_e8m0` · `hifloat8` | Atlas A3 / 950；stage 7 走 fp16 stand-in。`e4m3fn` 的 FN = Finite-Numbers-only（无 Inf）；`e8m0` 仅作 OCP MX 缩放因子 |
| 复数 | `complex32` · `complex64` · `complex128` | complex32 来自 PyTorch 实验类型；ascend 暂未广泛部署 |
| 有符号整 | `int4` · `int8` · `int16` · `int32` · `int64` | int4 为 ascend 量化扩展 |
| 无符号整 | `uint1` · `uint4` · `uint8` · `uint16` · `uint32` · `uint64` | uint1 / uint4 为 ascend 量化扩展。uint1 与 bool 语义不同：uint1 是 1-bit 数值（可算术），bool 是逻辑布尔（True/False） |
| 布尔 | `bool` | 仅参与逻辑/掩码语义；如需 1-bit 数值用 `uint1` |
| 量化整型 | `qint8` · `qint16` · `qint32` · `quint8` · `quint16` | 已登记到官方全集，可出现在排除补集中；声明支持需要真实 dtype_rule / oracle 能力 |
| 其他官方成员 | 完整列表见 `schemas/op-spec.json` | 包括 int2 / uint2、float6、特殊类型及固定排除的状态/边界标记；枚举登记不代表计算能力 |

对于 evaluator 已实现的 dtype，`np.<dtype>`（如 `np.hifloat8`）与字符串字面量 `"hifloat8"` 等价。官方枚举登记与 evaluator 能力分别维护：扩充全集时更新 schema 中的官方枚举、名称映射和相关限制；增加计算支持时还必须实现 `evaluators/promote.py` 等相关能力并补充 dtype_rule / oracle 测试。不能仅靠登记枚举就宣称可计算。

#### CANN ↔ spec dtype 命名对照

CANN 文档使用 `DT_*` 大写命名；spec.yaml 走 numpy/PyTorch 风格小写。对照表如下：

| CANN 名 | spec 名 | 说明 |
|---|---|---|
| `DT_FLOAT` / `DT_FLOAT32` | `float32` | |
| `DT_FLOAT64` / `DT_DOUBLE` | `float64` | |
| `DT_FLOAT16` / `DT_HALF` | `float16` | |
| `DT_BF16` / `DT_BFLOAT16` | `bfloat16` | |
| `DT_FLOAT8_E4M3FN` | `float8_e4m3fn` | FN = Finite-Numbers-only |
| `DT_FLOAT8_E5M2` | `float8_e5m2` | |
| `DT_FLOAT8_E8M0` | `float8_e8m0` | OCP MXFP8 scale factor |
| `DT_HIFLOAT8` | `hifloat8` | 华为自研 HF8 |
| `DT_FLOAT4_E2M1` | `float4_e2m1` | |
| `DT_FLOAT4_E1M2` | `float4_e1m2` | |
| `DT_COMPLEX32/64/128` | `complex32` / `complex64` / `complex128` | |
| `DT_INT4/8/16/32/64` | `int4` / `int8` / `int16` / `int32` / `int64` | |
| `DT_UINT1/4/8/16/32/64` | `uint1` / `uint4` / `uint8` / `uint16` / `uint32` / `uint64` | |
| `DT_BOOL` | `bool` | |
| `DT_QINT8/16/32`、`DT_QUINT8/16` | `qint8` / `qint16` / `qint32` / `quint8` / `quint16` | 已进入官方全集枚举，可作为 unsupported 补集成员；声明支持仍需真实 dtype_rule、组合表、oracle 和运行环境，见 [官方全集契约](support-universe.md) |

### B5b. layout 枚举速查

以下仅为常用 layout 速查。完整词汇见 `schemas/op-spec.json` 的 `$defs.layout.enum`；
官方 format 全集由同一 schema 的 `$defs.official_format.enum` 定义，来源见
[华为 Format 原文](format.md)。旧契约保留 NZ / CSR / COO / BSR，
新契约将 NZ 归一为 FRACTAL_NZ，且不接受 CSR / COO / BSR。

| layout | 含义 | 典型用途 |
|---|---|---|
| `ND` | n-dim 任意排布（不约束物理顺序） | 默认；elementwise / reduction 等大多数算子 |
| `NCHW` | (Batch, Channel, Height, Width) | 4D 卷积 / 池化（PyTorch 默认） |
| `NHWC` | (Batch, Height, Width, Channel) | 4D 卷积（TensorFlow 默认） |
| `NCDHW` | (Batch, Channel, Depth, Height, Width) | 5D 体积卷积（PyTorch 默认） |
| `NDHWC` | (Batch, Depth, Height, Width, Channel) | 5D 体积卷积（TensorFlow 默认） |
| `NZ` | 小 Z 排布（昇腾 16×16 tile 内列优先） | 历史名，等同 `FRACTAL_NZ` 的简写；新算子建议用全名 |
| `FRACTAL_NZ` | 大 Z 小 z 折叠（NZ 的全名） | Cube / matmul 高速路径 |
| `FRACTAL_Z` | 大 Z 小 Z 折叠 | Conv weights NPU 专用排布 |
| `CSR` | Compressed Sparse Row（行压缩稀疏） | 稀疏矩阵；外层 indptr / indices / values 三元组 |
| `COO` | Coordinate List（坐标列表稀疏） | 稀疏矩阵；indices 二维 + values |
| `BSR` | Block Sparse Row（分块行压缩稀疏） | 块稀疏；常用于结构化剪枝 |

跨字段占位符（用于 oracle.kwargs / formula 等）：`${attr.<name>}` / `${input.<name>.shape[i]}` / `${output.<name>.dtype}` / `${format_variants[].<field>}`。

### B5c. layout_contract（逻辑布局变体）

`layout_contract` 是可选顶层字段，用于表达一个 layout selector 下，多个已声明 Tensor 的**逻辑轴名称和顺序**。它适用于 attention 的 BSND/BNSD、NCHW↔NHWC 转换、image/grid/output 的空间轴对应等；普通逐元素算子可省略。

```yaml
layout_contract:
  variants:
    - id: bsnd
      when: "attr.input_layout == 'BSND'"
      tensors:
        - {ref: input.query, logical_axes: [B, S1, N1, Dk]}
        - {ref: input.key,   logical_axes: [B, S2, N2, Dk]}
        - {ref: output.y,    logical_axes: [B, S1, N1, Dv]}
    - id: bnsd
      when: "attr.input_layout == 'BNSD'"
      tensors:
        - {ref: input.query, logical_axes: [B, N1, S1, Dk]}
        - {ref: input.key,   logical_axes: [B, N2, S2, Dk]}
        - {ref: output.y,    logical_axes: [B, N1, S1, Dv]}
```

- **`variants[].id`**：必填且全局唯一的 layout variant 标识符。
- **`variants[].when`**：可选；复用 `semantic_cases.when` 的受限 guard DSL。缺失表示该变体无条件生效。
- **`variants[].tensors[].ref`**：必填，引用已声明的 `input.<name>` 或 `output.<name>`，且该接口必须是 `role: tensor`。
- **`logical_axes`**：必填、非空、同一 Tensor 内唯一的标识符数组，按张量的维度位置顺序书写。它们是轴语义标签，不自动声明跨 Tensor 的长度相等。

`layout_contract` **不**添加 `physical_format`：物理/表示 format 仍唯一地由既有 `inputs[].layout` / `outputs[].layout` 声明。它也不重复声明 transpose/permutation 或 shape 公式；这些仍由 `math_semantics.formula` 与 `outputs[].shape_rule` 定义。固定 rank input 的 `logical_axes` 个数由 Stage 2 校验；输出没有 `rank_range`，不从其 shape rule 猜测 rank。

### B5d. data_distribution（输入数据分布策略）

`inputs[].data_distribution` — 声明该输入的测试数据分布策略，供下游 `ascendc-st-design` 消费。利用 `inputs` 的 `additionalProperties: true`，无需改 schema。

**填写规则（stage 2 强校验）**：

| 算子类别 | data_distribution 值 | 说明 |
|----------|----------------------|------|
| **累加类**（Reduction / ReductionComposite / Recurrence / AtomicUpdate / Histogram） | **必须 `normal`** | 正态分布能暴露累加顺序敏感的精度缺陷（大数吃小数），均匀分布会掩盖 |
| **非累加类**（其他所有 category） | **不填**（省略字段） | 非累加类算子不涉及累加精度，无需指定分布 |

生成器对累加类 tensor 输入自动注入 `data_distribution: normal`，非累加类不注入。

| 值 | 含义 |
|---|---|
| `normal` | 正态分布（累加类算子唯一合法值） |

### B5e. format_variants（数据排布格式变体）

当算子支持多种数据排布（如 NCHW/NHWC/NCDHW）且**归约轴或计算逻辑因格式而异**时，
在 `math_semantics.format_variants` 中声明每种格式的具体参数：

```yaml
outputs:
  - name: sum
    shape_rule_kind: numpy_expr
    # 规范格式 NCHW；NHWC/NCDHW 见 format_variants[].shape_rule
    shape_rule: "sum.shape = (x.shape[1],)"
    shape_rule_description: |
      输出 shape = (C,)。Channel 轴位置因格式而异，见 format_variants。

math_semantics:
  formula_kind: textual_only
  formula: |
    # 归约轴由 format_variants 查表取值
    x_f32 = x.astype(np.float32)
    sum = np.sum(x_f32, axis=${format_variants[].reduction_axes})
  format_variants:
    - format: NCHW
      rank: [4]
      channel_axis: 1
      reduction_axes: [0, 2, 3]
      shape_rule: "sum.shape = (x.shape[1],)"
      oracle_kwargs: {dim: [0, 2, 3]}
    - format: NHWC
      rank: [4]
      channel_axis: 3
      reduction_axes: [0, 1, 2]
      shape_rule: "sum.shape = (x.shape[3],)"
      oracle_kwargs: {dim: [0, 1, 2]}
    - format: NCDHW
      rank: [5]
      channel_axis: 1
      reduction_axes: [0, 2, 3, 4]
      shape_rule: "sum.shape = (x.shape[1],)"
      oracle_kwargs: {dim: [0, 2, 3, 4]}
```

- **format**：必填，值来自 `$defs/layout` 枚举（同 `inputs[].layout`）
- **rank**：必填，该格式适用的 rank 值列表（如 4D、5D）
- **reduction_axes**：必填，该格式下的归约轴列表
- **channel_axis**：可选，Channel 轴在该 format 中的位置（如 NCHW=1, NHWC=3）；供 shape_rule 引用
- **shape_rule**：可选，该 format 下输出 shape 的 numpy 子集表达式；覆盖 canonical shape_rule
- **oracle_kwargs**：可选，该格式对应的 oracle kwargs（与 `reference_oracle.kwargs` 同结构）

**formula 与 format_variants 的引用约定**：

当计算因格式而异时，`formula_kind` 应为 `textual_only`（stage 7 SKIP）。formula 中用
`${format_variants[].<field>}` 占位符引用 format_variants 中的字段值（如
`${format_variants[].reduction_axes}`），并在 formula 注释中列出每种格式的查表结果。
这与 `${attr.dim}` / `${input.axis}` 等跨字段占位符语法一致。

与 `inputs[].layout` 的关系：`layout` 声明输入支持的格式（metadata），`format_variants` 细化每种格式的具体计算参数。

### B6. 范式候选 boundary / extreme case（与 `registries/boundary_min_cases.yaml` 同步）

| paradigms 含 | 候选 case（依支持域和来源核对） | 检查段 |
|---|---|---|
| Reduction | "reduce 轴长=1"、"rank=0 标量"、"空 Tensor" | boundary |
| NumericalStable | "fp16 上溢" 类边界 | extreme |
| SlidingWindow | "stride > kernel" | boundary |
| Padding | "零 padding 退化"、"超大 padding" | boundary |
| IndexGather | 索引越界（负 / 超长）+ "全相同索引" | boundary + extreme |
| ScatterUpdate | 索引越界 | boundary |
| AtomicUpdate | 索引越界 + 索引冲突 | boundary + extreme |
| MaskPredicate | "全 True" + "全 False" mask | extreme |
| SortSelect | k=0 + k=N | boundary |
| ArgReduce | tie / 等值 | extreme |
| Histogram | "所有值落同一 bin" | boundary |
| Spectral | "长度=2 幂次" + "非 2 幂次" | boundary |
| DynamicShape | shape_set 要求已移至 ascendc-st-design | — |
| RandomSampling | 固定 seed 下确定性 | extreme |
| Quantization | 全零输入（zero_point 行为） | extreme |
| Stateful | 跨调用状态保持 | boundary |

### B7. invariants.kind 三组（详细见 `registries/invariant_kind_registry.yaml`）

| 组 | 用途 | tolerance_inherit |
|---|---|---|
| value | 直接约束输出值（阈值 / reduce 后等于常量 / 值域 / 离散集合） | 允许 |
| algebraic | 代数性质（交换律 / 结合律 / 单调 / 幺元零元） | 允许 |
| structural | 结构性约束（输出 shape / 中间不外漏） | **必须 false** |

### B8. machine_check.kind / synthesize patterns / PRIMITIVE_WHITELIST

完整列表见对应 `registries/*.yaml` 文件头注释；本表只提示在哪查：

| 字段 | 注册表文件 |
|---|---|
| `boundary/extreme.machine_check.kind` | `registries/machine_check_kind_registry.yaml` |
| `extreme_inputs.synthesize.patterns[].pattern` | `registries/synthesize_pattern_registry.yaml` |
| `composition.primitives[].op` | `registries/primitive_whitelist.yaml` |
| `raises_error.error_type` | `registries/error_code_enum.yaml` |

### B9. 支持与排除范围

新生成文件使用 `support_universe: huawei-cann-types-2026-10-15`，每个输入/输出都具有
`dtype_set`、`unsupported_dtypes`、`supported_layouts`、`unsupported_layouts`。
每一对列表必须互斥，且并集恰好等于对应官方全集（42 个 dtype / 57 个 format，含固定排除的标记值）。
支持集合必须先按完整需求确认，然后由生成器计算补集。

```yaml
support_universe: huawei-cann-types-2026-10-15
inputs:
  - name: x
    dtype_set: [float16, float32]
    # unsupported_dtypes: 由工具生成其余 40 个官方成员的完整列表
    layout: ND
    supported_layouts: [ND]
    # unsupported_layouts: 由工具生成其余 56 个官方成员的完整列表
    # role / rank_range / shape 等字段照常填写
outputs:
  - name: y
    dtype_set: [float16, float32]  # 必须等于输出组合表投影
    # unsupported_dtypes: 同样由工具生成完整列表
    layout: ND
    supported_layouts: [ND]
    # unsupported_layouts: 同样由工具生成完整列表
    # shape_rule / dtype_rule 等字段照常填写
```

上面是字段说明片段；注释不能替代真实的完整列表，不能直接作为完整 spec 校验。
工具使用 `--supported-formats "input.x:ND;output.y:ND"` 指定各接口完整格式范围；原有排除参数
作为额外断言保留。`NZ` 归一化为 `FRACTAL_NZ`，不会在全集中重复计数。
只有不带 `support_universe` 的旧 spec 才沿用部分排除语义；不能宣称它已覆盖官方全集。
上下文和芯片限制见 [官方全集契约](support-universe.md)。这些检查不验证设备运行时拒绝行为。

---

## C. 11-stage L0 校验器（全景）

| stage | 名称 | 范围 |
|---|---|---|
| 1 | schema_static | JSON Schema 必填项、类型、enum、pattern 和额外字段 |
| 2 | category_paradigm_consistency | category↔paradigm、paradigm_groups、内部约束和白名单；支持/排除声明的重叠、跨字段矛盾；全量契约检查全集覆盖与全集外值 |
| 3 | shape_closure | 求值 outputs[].shape_rule、检查 data_dependent 声明及适用时的公式输出 shape |
| 4 | dtype_closure | dtype_rule 推导与 supported_combinations 交叉验证 |
| 5 | broadcast_legality | numpy / none / explicit 广播语义 |
| 6 | boundary_min_set | 范式候选边界提示，结合已确认支持域核对适用性 |
| 7 | tolerance_coverage | 输出 dtype 容差覆盖与紧度启发式 |
| 8 | formula_smoke_eval | 在 numpy 沙箱运行小 shape formula；缺依赖或不适用时 SKIP |
| 9 | oracle_reachable | 检查 framework/API 可达性和占位符；不调用 oracle |
| 10 | formula_oracle_equiv | 适用时实际比较 formula 与 oracle 的特殊值输出 |
| 11 | invariant_exec | 执行公式并验证所声明的不变量；无需 oracle |
| 12 | extreme_check | 合成极端输入并核对 machine_check；适用的 matches_oracle 条目会调用 oracle |

PASS / FAIL / SKIP 是阶段状态；SKIP 表示相关检查没有执行，不代表实际通过。当前进程允许纯 info SKIP 而不阻断，overall PASS 或退出码 0 也不证明全部 12 阶段都已执行。stage 9 只检查可达性；stage 10 和 stage 12 的 matches_oracle 路径在依赖及条目条件满足时才会调用 oracle，stage 11 无需 oracle。退出码、依赖、SKIP 条件和 findings 以 [SKILL.md §4](../SKILL.md) 与 [stage-rules.md](stage-rules.md) 为准。

---

## D. 常见陷阱

1. **paradigms 字段**：必须 `paradigms: [Reduction, ...]`（PascalCase 受控集合）。
2. **category 与 paradigms 职责不同**：两者均使用注册表中的 PascalCase 拼写；category=FusedComposite 时 paradigms 必须含 FusedComposite + ≥ 2 条基础 paradigm，不能把小写 snake_case 当作有效 category。
3. **折叠维登记**：`"...d"` 折叠维**不应**登记到 `shape_constraints.symbols`；只有显式大写维（`M` / `K` / `N`）才登记。
4. **supported_combinations 必须显式枚举**：numpy_expr 推导（`np.promote_types(...)`）已能算，但显式表是"锁定意图"的真值，避免 promote 表升级时静默改语义。
5. **accumulator_dtype 必填触发**：category=Contraction / ReductionComposite / paradigms 含 `NumericalStable+Reduction` 时必填，且必须 ≥ 输入侧最高精度。
6. **invariants 至少 1 条**：ReductionComposite / NumericalStable 范式强制要求；structural 组的 kind 不允许 `tolerance_inherit: true`。
7. **boundary 与 extreme 不要混淆**：boundary 是合法但退化的 case（rank=0 / 空 Tensor / 参数越界 raise）；extreme 是异常输入（NaN / Inf / 上溢）。
8. **`global_constraints` 是咨询性字段**：当前 evaluator 不解析；与 `notes` 同语义，仅供下游设计/测试人工阅读。机器可断言的 shape 关系（如 matmul `K of x == K of w`）请改写在 `outputs.shape_rule` 的 numpy_expr 中，由 stage 3 求值。
9. **占位符不嵌套**：`${attr.dim}` 合法，`${attr.${attr.dim}}` 非法。
