# 算子 spec.yaml 速查表

> 字段速查；权威约束以 `schemas/op-spec.json` 与 `registries/*.yaml` 为准。
> 子规则（rule_id 列表与失败码）见 SKILL.md §4 各 stage，不在此处重复。

---

## A. 顶层 15 个字段（test_matrix 已移出）

### A0. 字段职责分层

| 职责层级 | 字段 | 说明 |
|----------|------|------|
| **L0 语义** | `op` / `inputs` / `attributes` / `outputs` / `shape_constraints` / `dtype_policy` / `broadcast` / `math_semantics` | 数学约束，下游不可覆盖 |
| **L0 语义** | `boundary_conditions`（合法但退化的输入 case）/ `extreme_inputs`（NaN / Inf / 全零等异常输入 case） | 下游不可覆盖 |
| **L0 语义** | `numerical_tolerance` / `numerical_stability` / `determinism` | 精度语义，下游不可覆盖 |
| **已移出** | `test_matrix` | 测试生成参数由 `ascendc-st-design` skill 独立管理 |

当 `REQUIREMENTS.md` 与 `spec.yaml` 共存时，`spec.yaml` 只承载已经锁定的结构化 L0 契约；需求背景、接口自然语言说明、资源/性能目标来源等仍由 `REQUIREMENTS.md` 承载。下游方案设计和测试设计必须按对应 Agent 的「输入优先级与字段所有权」执行：上表 L0 语义字段以 `spec.yaml` 为准，冲突时停止并报告，不得从需求正文重新解释后覆盖。

```yaml
schema_version: 1          # 固定值
op:                        # 元信息：name / version / description / category（单选 25 类） / paradigms（多选 27 项）
inputs: []                 # 张量/标量/state，按位置排列
attributes: []             # 非张量参数，含 machine_constraint
outputs: []                # 用 numpy 子集表达式描述 shape/dtype 推导规则
shape_constraints: {}      # 全局符号表 + global_constraints（咨询性字段，当前不参与 9-stage 机器校验，见 §D.8）+ notes
dtype_policy: {}           # promotion + supported_combinations 显式枚举 + accumulator_dtype
broadcast: {}              # kind: numpy | none | explicit (+rules)
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

### B1. 25 类 category（单选）

```
elementwise · reduction · reduction_composite · contraction · layout_transform
index_gather · scatter_update · mask_predicate · sort_select · fused_composite
random_sampling · sliding_window · collective · control_flow · recurrence
interpolation · quantization · stateful · arg_reduce · sparse · padding
variable_output · spectral · histogram · atomic_update
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
| fused_composite | FusedComposite + ≥ 2 基础 paradigm |
| sliding_window | SlidingWindow（通常再带 Contraction 或 Reduction） |
| arg_reduce | ArgReduce（输出 dtype_rule 必须求值为 `np.int32` 或 `np.int64`） |
| atomic_update | AtomicUpdate（与 ScatterUpdate 互斥） |

### B4. shape_rule (numpy_expr)

`outputs[].shape_rule` 现在是受限 numpy 子集表达式（不再是 DSL 函数）。`shape_rule_kind` 取值：

| kind | 适用 | shape_rule 内容 |
|---|---|---|
| `numpy_expr` | 输出形状可静态从输入 shape + attribute 推出（绝大多数算子） | numpy 表达式（见下） |
| `data_dependent` | 输出形状由输入**值**决定（nonzero / unique / masked_select 等 VariableOutput） | 不写表达式，写 `shape_rule_description` + `shape_bounds` |

numpy_expr 中可用：
- `x.shape` / `x.shape[-1]` / `x.shape[:-2]`（属性 + 切片 + 负索引）
- `tuple` 拼接 `+`：`(a.shape[-1],) + (b.shape[-1],)`
- `np.broadcast_shapes(*shapes)`：numpy 广播标准 API
- `IfExp`：`a if cond else b`（attribute 默认值作为 cond）

未实现（需扩展 `scripts/evaluators/shape_eval.py`）：`reshape` / `transpose` / `permute` / `concat` / `stack` / `slice` / `tile` / `repeat`，及 `np.fft.*` / `np.linalg.*` 等。

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
| `NCHW` | (Batch, Channel, Height, Width) | 卷积 / 池化（PyTorch 默认） |
| `NHWC` | (Batch, Height, Width, Channel) | 卷积（TensorFlow 默认） |
| `NZ` | 小 Z 排布（昇腾 16×16 tile 内列优先） | 历史名，等同 `FRACTAL_NZ` 的简写；新算子建议用全名 |
| `FRACTAL_NZ` | 大 Z 小 z 折叠（NZ 的全名） | Cube / matmul 高速路径 |
| `FRACTAL_Z` | 大 Z 小 Z 折叠 | Conv weights NPU 专用排布 |
| `CSR` | Compressed Sparse Row（行压缩稀疏） | 稀疏矩阵；外层 indptr / indices / values 三元组 |
| `COO` | Coordinate List（坐标列表稀疏） | 稀疏矩阵；indices 二维 + values |
| `BSR` | Block Sparse Row（分块行压缩稀疏） | 块稀疏；常用于结构化剪枝 |

跨字段占位符（用于 oracle.kwargs 等）：`${attr.<name>}` / `${input.<name>.shape[i]}` / `${output.<name>.dtype}`。

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
| 5 | broadcast_legality | numpy / none / explicit 广播模拟（2 输入 + matmul 形式 explicit rules） |
| 6 | boundary_min_set | 范式 → 最低 case 集（关键词子串匹配版） |
| 7 | tolerance_coverage | 容差 dtype 覆盖 + 紧度启发式 |
| 8 | formula_smoke_eval | 小 shape `[2,3]` 沙箱 numpy eval（缺 numpy ⇒ SKIP） |
| 9 | oracle_reachable | 真 import framework + getattr 链 + 占位符校验（缺 framework ⇒ SKIP，不算 FAIL） |

全部 PASS / SKIP ⇒ spec 视为 L0 契约通过。SKIP 不算失败，环境降级时常见。

---

## D. 常见陷阱

1. **paradigms 字段**：必须 `paradigms: [Reduction, ...]`（PascalCase 受控集合）。
2. **fused_composite 与 FusedComposite 不等价**：前者是 category（结构主分类），后者是 paradigm（"复合形态"标签）。category=fused_composite 必须含 paradigm FusedComposite + ≥ 2 条基础 paradigm。
3. **折叠维登记**：`"...d"` 折叠维**不应**登记到 `shape_constraints.symbols`；只有显式大写维（`M` / `K` / `N`）才登记。
4. **supported_combinations 必须显式枚举**：numpy_expr 推导（`np.promote_types(...)`）已能算，但显式表是"锁定意图"的真值，避免 promote 表升级时静默改语义。
5. **accumulator_dtype 必填触发**：category=contraction / reduction_composite / paradigms 含 `NumericalStable+Reduction` 时必填，且必须 ≥ 输入侧最高精度。
6. **invariants 至少 1 条**：reduction_composite / NumericalStable 范式强制要求；structural 组的 kind 不允许 `tolerance_inherit: true`。
7. **boundary 与 extreme 不要混淆**：boundary 是合法但退化的 case（rank=0 / 空 Tensor / 参数越界 raise）；extreme 是异常输入（NaN / Inf / 上溢）。
8. **`global_constraints` 是咨询性字段**：当前 evaluator 不解析；与 `notes` 同语义，仅供下游设计/测试人工阅读。机器可断言的 shape 关系（如 matmul `K of x == K of w`）请改写在 `outputs.shape_rule` 的 numpy_expr 中——它会真正被 stage 5 跑出来。
9. **占位符不嵌套**：`${attr.dim}` 合法，`${attr.${attr.dim}}` 非法。
