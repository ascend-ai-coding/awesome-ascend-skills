# 算子 spec.yaml 速查表

> 字段速查；权威约束以 `schemas/op-spec.json` 与 `registries/*.yaml` 为准。
> 子规则（rule_id 列表与失败码）见 SKILL.md §4 各 stage，不在此处重复。

---

## A. 顶层 15 个字段（test_matrix 已移出）

### A0. 字段职责分层

| 职责层级 | 字段 | 说明 |
|----------|------|------|
| **L0 语义** | `op` / `inputs` / `attributes` / `outputs` / `shape_constraints` / `dtype_policy` / `broadcast`（计算是否需要数据广播） / `math_semantics` | 数学约束，下游不可覆盖 |
| **L0 语义** | `boundary_conditions`（合法但退化的输入 case）/ `extreme_inputs`（NaN / Inf / 全零等异常输入 case） | 下游不可覆盖 |
| **L0 语义** | `numerical_tolerance` / `numerical_stability` / `determinism` | 精度语义，下游不可覆盖 |
| **已移出** | `test_matrix` | 测试生成参数由 `ascendc-st-design` skill 独立管理 |

当 `REQUIREMENTS.md` 与 `spec.yaml` 共存时，`spec.yaml` 只承载已经锁定的结构化 L0 契约；需求背景、接口自然语言说明、资源/性能目标来源等仍由 `REQUIREMENTS.md` 承载。下游方案设计和测试设计必须按对应 Agent 的「输入优先级与字段所有权」执行：上表 L0 语义字段以 `spec.yaml` 为准，冲突时停止并报告，不得从需求正文重新解释后覆盖。

```yaml
schema_version: 1          # 固定值
op:                        # 元信息：name / version / description / category（单选 26 类） / paradigms（多选 27 项） / paradigm_groups（范式分组）
inputs: []                 # 张量/标量/state，按位置排列
attributes: []             # 非张量参数，含 machine_constraint
outputs: []                # 用 numpy 子集表达式描述 shape/dtype 推导规则
shape_constraints: {}      # 全局符号表 + global_constraints（咨询性字段，当前不参与 9-stage 机器校验，见 §D.8）+ notes
dtype_policy: {}           # promotion + supported_combinations 显式枚举 + accumulator_dtype
broadcast: {}              # 算子计算的 broadcast 语义（数据复制/扩展）。kind: numpy | none | explicit (+rules)
math_semantics: {}         # formula + reference_oracle + invariants + composition (FusedComposite 必填)
numerical_stability: {}    # required + techniques (含 anti_pattern_id)
numerical_tolerance: {}    # per_dtype: {rtol, atol, metric}
boundary_conditions: []    # 机器可断言的退化/越界 case（L0 语义）
extreme_inputs: []         # NaN / Inf / 全零等异常 case（L0 语义）
determinism: {}            # accumulation_order + bitwise_reproducible
# test_matrix 已移出，由 ascendc-st-design skill 独立管理
```

---

## B. 关键枚举与白名单

### B1. 26 类 category（单选）

```
elementwise · reduction · reduction_composite · contraction · layout_transform
index_gather · scatter_update · mask_predicate · sort_select · fused_composite
random_sampling · sliding_window · collective · control_flow · recurrence
interpolation · quantization · stateful · arg_reduce · sparse · padding
variable_output · spectral · histogram · atomic_update · broadcast
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
| reduction_composite | Reduction + FusedComposite |
| fused_composite | FusedComposite + ≥ 2 基础 paradigm（Elementwise 被其他范式吸收，不计入） |
| sliding_window | SlidingWindow（通常再带 Contraction 或 Reduction） |
| arg_reduce | ArgReduce（输出 dtype_rule 必须求值为 `np.int32` 或 `np.int64`） |
| atomic_update | AtomicUpdate（与 ScatterUpdate 互斥） |

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
- **slice/gather 的区别** → slice 是 LayoutTransform/IndexGather（确定性裁剪），gather 是 IndexGather（自由索引采集）

记录决策理由（输入数量 → category_enum.yaml 匹配项 → category → paradigms 推导链）。

### B4. shape_rule (numpy_expr)

`outputs[].shape_rule` 是受限 numpy 子集表达式（不再是 DSL 函数）。完整语义、子规则和失败码见
[SKILL.md §4.3](../SKILL.md#43-stage-3--shape_closurenumpy_expr-求值)。

| kind | 适用 | shape_rule 内容 |
|---|---|---|
| `numpy_expr` | 输出形状可静态从输入 shape + attribute 推出（绝大多数算子） | numpy 表达式 |
| `data_dependent` | 输出形状由输入**值**决定（nonzero / unique / masked_select 等 VariableOutput） | 仅允许 `op.paradigms` 含 `VariableOutput`；不写表达式，写 `shape_rule_description` + `shape_bounds` |
| `textual_only` | 输出形状因数据排布**格式**而异（如 NCHW vs NHWC 的 Channel 轴位置不同） | 仅允许 `math_semantics.format_variants` 存在；可含 `${format_variants[].channel_axis}` 占位符；必须配 `shape_rule_description` |

`inputs[].shape.symbolic` 列表元素三类：显式维（`"M"` 大写） / 折叠维（`"...d"` 小写，仅可作首元素） / 常量维（整数）。

> **`symbolic` 名字仅为 owner 自留命名，不参与跨 input 绑定**：两个 input 都写 `"K"` 或都写 `"...batch"` 不会被 evaluator 视为"必相等"。跨 input 形状关系（如 matmul `K==K`、elementwise 同形 / 可广播）必须通过 `outputs.shape_rule` 表达，由 stage 5/8 跑出来。stage 5 在 `broadcast.kind=none` 下只静态校核 rank 结构 + const 维相等，其余放到 stage 8 跑公式时由 numpy 自身验证。

### B5. dtype_rule (numpy_expr)

`outputs[].dtype_rule` 同样是 numpy 子集表达式。常见形态：

| 写法 | 用途 |
|---|---|
| `y.dtype = x.dtype` | 与某 input 相同 |
| `c.dtype = np.promote_types(a.dtype, b.dtype)` | numpy 类型提升 |
| `c.dtype = np.result_type(a.dtype, b.dtype, c.dtype)` | 多输入类型提升 |
| `y.dtype = np.int32` / `np.float16` / `np.complex64` | 固定 dtype |
| `out.dtype = np.complex64 if real.dtype == np.float32 else np.complex128` | IfExp 条件分支 |

dtype 常量字符串覆盖（25 个；与 `schemas/op-spec.json` `$defs.dtype.enum` 单一真值）：

| 组 | dtype | 来源 / 备注 |
|---|---|---|
| 标准浮点 | `float16` · `float32` · `float64` · `bfloat16` | numpy 原生 |
| 窄浮点（ascend 私有） | `float4_e2m1` · `float4_e1m2` · `float8_e4m3fn` · `float8_e5m2` · `float8_e8m0` · `hifloat8` | Atlas A3 / 950；stage 8 走 fp16 stand-in。`e4m3fn` 的 FN = Finite-Numbers-only（无 Inf）；`e8m0` 仅作 OCP MX 缩放因子 |
| 复数 | `complex32` · `complex64` · `complex128` | complex32 来自 PyTorch 实验类型；ascend 暂未广泛部署 |
| 有符号整 | `int4` · `int8` · `int16` · `int32` · `int64` | int4 为 ascend 量化扩展 |
| 无符号整 | `uint1` · `uint4` · `uint8` · `uint16` · `uint32` · `uint64` | uint1 / uint4 为 ascend 量化扩展。uint1 与 bool 语义不同：uint1 是 1-bit 数值（可算术），bool 是逻辑布尔（True/False） |
| 布尔 | `bool` | 仅参与逻辑/掩码语义；如需 1-bit 数值用 `uint1` |

写法上 `np.<dtype>`（如 `np.hifloat8`）与字符串字面量 `"hifloat8"` 等价；都会被 `dtype_eval._NpDtypeNamespace` 解析为字符串后落入 `promote_pair`。新增 dtype 时同步改三处：`schemas/op-spec.json` `$defs.dtype.enum` + `evaluators/promote.py` `_ALL_DTYPES` 相关表 + 在 `tests/test_dtype_eval.py::TestNamespaceSync` 加一条用例。

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
| `DT_QINT8/16/32`、`DT_QUINT8/16` | **暂不支持** | 结构化量化整型（带 scale / zero_point 元数据）；语义上应走 `inputs[].quantization` 字段而非 dtype 枚举，避免每条 supported_combinations 展开量化变体。如需对接 CANN 量化族，请单独提 issue |

### B5b. layout 枚举速查

`inputs[].layout` / `outputs[].layout` 接受以下 9 种，与 `schemas/op-spec.json` `$defs.layout.enum` 同步：

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

### B5c. format_variants（数据排布格式变体）

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

当计算因格式而异时，`formula_kind` 应为 `textual_only`（stage 8 SKIP）。formula 中用
`${format_variants[].<field>}` 占位符引用 format_variants 中的字段值（如
`${format_variants[].reduction_axes}`），并在 formula 注释中列出每种格式的查表结果。
这与 `${attr.dim}` / `${input.axis}` 等跨字段占位符语法一致。

与 `inputs[].layout` 的关系：`layout` 声明输入支持的格式（metadata），`format_variants` 细化每种格式的具体计算参数。

### B6. 范式必含 boundary / extreme case（与 `registries/boundary_min_cases.yaml` 同步）

| paradigms 含 | 必含 case | 检查段 |
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

---

## C. 9-stage L0 校验器（全景）

| stage | 名称 | 范围 |
|---|---|---|
| 1 | schema_static | JSON Schema |
| 2 | category_paradigm_consistency | category↔paradigm + 白名单 + 内部约束 |
| 3 | shape_closure | numpy_expr 求值 outputs[].shape_rule（含 data_dependent 分流） |
| 4 | dtype_closure | numpy_expr 求值 outputs[].dtype_rule，与 supported_combinations 交叉验证 |
| 5 | broadcast_legality | 算子计算的 broadcast 语义校验（numpy / none / explicit；2 输入 + matmul 形式 explicit rules） |
| 6 | boundary_min_set | 范式 → 最低 case 集（关键词子串匹配版） |
| 7 | tolerance_coverage | 容差 dtype 覆盖 + 紧度启发式 |
| 8 | formula_smoke_eval | 小 shape `[2,3]` 沙箱 numpy eval（缺 numpy ⇒ SKIP） |
| 9 | oracle_reachable | 真 import framework + getattr 链 + 占位符校验（缺 framework ⇒ SKIP，不算 FAIL） |

全部 PASS / SKIP ⇒ spec 视为 L0 契约通过。SKIP 不算失败，环境降级时常见。

---

## D. 常见陷阱

1. **paradigms 字段**：必须 `paradigms: [Reduction, ...]`（PascalCase 受控集合）。
2. **fused_composite 与 FusedComposite 不等价**：前者是 category（结构主分类），后者是 paradigm（"复合形态"标签）。category=FusedComposite 必须含 paradigm FusedComposite + ≥ 2 条基础 paradigm。
3. **折叠维登记**：`"...d"` 折叠维**不应**登记到 `shape_constraints.symbols`；只有显式大写维（`M` / `K` / `N`）才登记。
4. **supported_combinations 必须显式枚举**：numpy_expr 推导（`np.promote_types(...)`）已能算，但显式表是"锁定意图"的真值，避免 promote 表升级时静默改语义。
5. **accumulator_dtype 必填触发**：category=Contraction / reduction_composite / paradigms 含 `NumericalStable+Reduction` 时必填，且必须 ≥ 输入侧最高精度。
6. **invariants 至少 1 条**：reduction_composite / NumericalStable 范式强制要求；structural 组的 kind 不允许 `tolerance_inherit: true`。
7. **boundary 与 extreme 不要混淆**：boundary 是合法但退化的 case（rank=0 / 空 Tensor / 参数越界 raise）；extreme 是异常输入（NaN / Inf / 上溢）。
8. **`global_constraints` 是咨询性字段**：当前 evaluator 不解析；与 `notes` 同语义，仅供下游设计/测试人工阅读。机器可断言的 shape 关系（如 matmul `K of x == K of w`）请改写在 `outputs.shape_rule` 的 numpy_expr 中——它会真正被 stage 5 跑出来。
9. **占位符不嵌套**：`${attr.dim}` 合法，`${attr.${attr.dim}}` 非法。
