---
schema_version: "2.1"
op_name: "{op_name}"
status: draft                       # draft | reviewed | final
last_updated: "{YYYY-MM-DD}"

# 关键接口契约（详细形参在 §2 写明）
compute_kind: "{vector|cube|mixed}"
dtypes: ["{fp32|bf16|fp16}"]
dynamic_axes: {dynamic_axes}        # 例如 ["B","S"]；无则填 []
precision: { rtol: 1e-3, atol: 1e-3 }
---

# {op_name} 设计方案

## 0. Decomposition Decision

> 决定本算子要拆成几个 module。Stage 4（pypto-op-construct）必须先读 §0.3 的 `module_count` 决定走 L0（单 module）还是 L1（多 module）路径。详见 skill `pypto-op-design`'s `SKILL.md` 第 0 轮。

### 0.1 复杂度信号采集

| 信号 | 值 | 采集方式 |
|------|----|---------|
| effective_lines | {N} | `python skills/pypto-op-design/scripts/count_golden_lines.py custom/<op>/<op>_golden.py` |
| loop_carried_state_groups | {N} | 人工统计：golden 中"上一步结果参与下一步"的独立递归状态组数（FA 的 m/l/o 算 1 组）|
| matmul_count | {N} | 人工统计：`torch.matmul` 出现次数 |
| cross_tile_reduce_count | {N} | 人工统计：跨 tile reduce（softmax 整套算 1 个；`sum`/`max` 跨超过单 tile 大小的 reduce 轴算 1 个）|

### 0.2 总复杂度公式

```text
L = effective_lines / 30                                   = {value}
S = loop_carried_state_groups                              = {value}
O = (matmul_count + cross_tile_reduce_count) / 3           = {value}

total_complexity = max(L, S, O)                            = {value}
```

### 0.3 module_count 决策

```text
if total_complexity < 1.3:
    module_count = 1                                       # L0：不拆
else:
    raw         = round(total_complexity)                  = {value}
    line_cap    = ceil(effective_lines / 12)               = {value}
    module_count = min(raw, line_cap)                      = {value}
```

**Decision**: `module_count = {1 | N}` → `decomposition_level: {L0 | L1}`

### 0.4 Heavy / Light op 分类 (跨 tile 通信为准)

Stage 4 拆分时使用，这里记录本算子涉及的 op 分类：

| Op 名称 | Heavy / Light | 说明 |
|---------|--------------|------|
| {api}   | {heavy/light}| {为什么} |

参考分类：
- **Heavy**：`pypto.matmul`、跨 tile reduce、softmax 整套、scan / recurrence step、outer product
- **Light**：elementwise、cast、tile 内 reduce、简单 reshape、`pypto.view`（强制合并到首 module）、`pypto.assemble`（强制合并到尾 module）

### 0.5 数据流断点（仅当 `module_count ≥ 2`）

在 golden 的数据流上选 `module_count - 1` 个语义清晰、可命名、可独立验证的中间张量作为模块边界。每个 module 内含若干 heavy ops + 相关 light ops，约等于 1 复杂度单位。

| Module | 内含 heavy ops | 内含 light ops merged | 边界输出张量（name, shape）| CU 估值 |
|--------|----------------|------------------------|---------------------------|---------|
| M1     | {ops}          | {ops, view (entry)}    | {tensor_name, shape}      | ~1.0    |
| M2     | {ops}          | {ops}                  | {tensor_name, shape}      | ~1.0    |
| MN     | {ops}          | {ops, assemble (exit)} | (final output)            | ~1.0    |

> Module count = 1（L0）时本段填 `N/A — single module covering entire kernel`。

---

## 1. 计算图与精度路由

### 1.1 API 调用序列

| 步骤 | 操作 | PyPTO API | 输入 dtype | 输出 dtype | 输出 shape | 备注 |
|------|------|-----------|------------|------------|-----------|------|
| 1    | {op} | `{api}`   |            |            |           |      |

### 1.2 精度路由

```text
输入({dtype}) → [步骤 N: cast] → 计算({dtype}) → [步骤 M: cast] → 输出({dtype})
```

| 转换位置 | 转换方向 | 原因 |
|---------|---------|------|
| 步骤 N 前 | BF16 → FP32 | `pypto.sum` 要求 FP32 |

### 1.3 替代方案（已排除）

| 替代方案 | 排除原因 |
|---------|---------|
|         |         |

---

## 2. 数据规格

### 2.1 Kernel 函数签名

```python
@pypto.frontend.jit(runtime_options={"run_mode": "npu"})
def {op_name}_kernel(
    {input}: pypto.Tensor([{shape}], pypto.{dtype}),    # 输入说明
    {output}: pypto.Tensor([{shape}], pypto.{dtype}),   # 输出说明
):
    ...
```

### 2.2 动态轴分析

> 仅运行时才确定大小的轴标 `pypto.DYNAMIC`；编译期已知的轴写常量。

| 维度名 | 是否动态 | 取值范围 / 常量 | 标注方式 |
|--------|---------|-----------------|---------|
| {axis} | 是/否    | {range/const}   | `pypto.DYNAMIC` / 数值 |

### 2.3 值类型分析（避免 SymbolicScalar 误用）

| 变量 | 来源 | 类型 | 注意事项 |
|------|------|------|---------|
| {var} | `x.shape[i]`（动态轴） | SymbolicScalar | 不可用于 Python `if/range`，不可索引 list |
| {var} | 字面量 / 编译期常量 | int / float | 常规使用 |
| {var} | `pypto.Element(...)` | Element | 用于标量参与 op 计算 |

---

## 3. Tiling 策略

### 3.1 算子类型

{Vector / Cube / Hybrid}

### 3.2 Tiling 推导

#### 3.2.1 Shape analysis (思考起点)

- **同时驻留 UB 的 Tensor**：

| Tensor | 用途 | shape | dtype | 大小估算 |
|--------|------|-------|-------|------|
| {name} | {role} | {shape} | {dtype} | {bytes} |

- **算子分类**: {Pointwise / Reduction / Cube-heavy / Mixed}
- **若含 matmul，列举 shape**：

| Matmul | A shape | B shape | 输出 shape | M / K / N |
|--------|---------|---------|-----------|-----------|
| {name} | {shape} | {shape} | {shape}   | {M} / {K} / {N} |

- **列举所有 vec op 的 shape class** (per-stage 判定):

| Shape class | 该 class 的 op | 期望 vec tile |
|-------------|----------------|---------------|
| `[1,1,BT,K]` | {ops}          | `(1,1,BT,K)`  |
| `[1,1,BT,BT]` | {ops}          | `(1,1,BT,BT)` |

→ 若 shape class 数 ≥ 2，则需要 per-stage `set_vec_tile_shapes` (在 §3.2.5 中体现)

- **尾轴 + dtype**: 尾轴 = {axis}, dtype = {dtype} → alignment 单位 = {8 (FP32) or 16 (BF16/FP16)} 元素
- **并行化候选轴**: {batch / sequence / heads / 等}

#### 3.2.2 Tile design + rationale (mandatory)

- **Stage 7 前 Tile shape 基线**：
  - Cube / matmul：统一使用 128 基线，首跑前不做性能化切分。
  - Vector：沿用默认 vec tile 规则，每轴通常在 `[16, 64]`，按 alignment / UB / rank 约束调整。
  - 如因 API/shape 硬约束无法使用 Tile shape 基线，必须说明例外原因和替代值。
- **采用 vec tile**: `pypto.set_vec_tile_shapes({tile_dims})`
- **采用 cube tile** (若含 matmul): `pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])`
- **Rationale** (单句, mandatory):
  > {Cube tile：Stage 7 前统一 128；Vec tile：默认 [16, 64] 范围内稳定值；偏离时说明 API/shape 硬约束原因}
- **Source**:
  - [ ] User-specified (见 SPEC.md §{X} 或 prompt)
  - [ ] Stage 7 前 Tile shape 基线（详见 SKILL.md §R2 步骤 1.6 + quick_ref.md）
  - [ ] API/shape hard-constraint exception (rationale 见上)

> **Note**: Stage 7 前只使用稳定首跑基线。性能优化由 Stage 7 optimizer 负责，不要在 architect 阶段做训练/decode/核利用率等性能化 tile 分支。

#### 3.2.3 Alternatives considered (optional, 推荐填)

| 备选 tile | 推荐度 | 否决理由 |
|-----------|--------|---------|
| {tile A}  | {x/10} | {1 句否决理由} |
| {tile B}  | {x/10} | {1 句}         |

> **可选**: 若 tile 选择有明显约束权衡（例如 vec tile 在 `[16, 64]` 范围内多个合理选择，或 API/shape 迫使 cube 偏离 128），建议列出。性能权衡是 Stage 7 的工作，architect 阶段不强制穷举。

#### 3.2.4 PyPTO syntax compliance (machine-verifiable, 全部 ☑ required)

- [ ] 尾轴 alignment OK: 尾轴 tile = {value}, 是 {8 (FP32) or 16 (BF16/FP16)} 倍数
- [ ] tile 维数 ({n}) == 操作 tensor 维数 ({m})
- [ ] UB 预算 (per op): `tile_size × dtype × tensor_count` = {bytes} ≤ UB 容量；tensor_count 由 op 类型决定（unary=2, binary=3, reduce/expand 保守 4）
- [ ] 展开约束: `(shape/tile) × n` = {value} < 18000
- [ ] cube tile dim 关系 (若含 matmul): mL0={a} ≤ mL1={b}, kL0 ≤ kL1, nL0 ≤ nL1
- [ ] cube tile 16 元素 alignment (BF16/FP16 场合): 全 dim 是 16 元素倍数
- [ ] broadcast 1 轴 rule: 无隐式 multi-axis expand，broadcast 轴用 size-1 显式标
- [ ] vec tile per-stage 设置: 若 §3.2.1 列举的 vec-op shape class ≥ 2，则在 §3.2.5 中按 stage 重新 set
- [ ] tile 参数全部编译期静态: `set_vec_tile_shapes` / `set_cube_tile_shapes` 的每个值（含 list 元素）是 Python int 字面量或解析到字面量的常量 Assign；**无** kernel 入参 / `tensor.shape[i]` / SymbolicScalar / 运行时计算
- [ ] Stage 7 前 Tile shape 基线已应用，或已说明 API/shape 硬约束例外
- [ ] vec tile axis ∈ [16, 64]：每个轴值在该范围内；超 UB 已下调至 <16 并在 §3.2.2 rationale 中说明；不要超过 64（Stage 7 optimizer 上调）

> 任一 ☐ 未勾选 → 回到 §3.2.2 重新设计 tile

#### 3.2.5 最终 tile 实现形 (Stage 5 coder 引用)

```python
# 单一值 (shape range 窄时):
pypto.set_vec_tile_shapes({tile_dims})
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

# Per-stage (多 vec-op shape class 时):
pypto.set_vec_tile_shapes({stage_A_tile})
# ... stage A 计算 ...
pypto.set_vec_tile_shapes({stage_B_tile})
# ... stage B 计算 ...
```

---

## 4. Loop 与数据流

### 4.1 维度判定

| 轴 | 维度大小 | 编译期 / 运行期 | Loop 处理 |
|----|---------|----------------|----------|
| {axis} | {N} | 编译期已知 | 不需要 loop |
| {axis} | DYNAMIC | 运行期 | `pypto.loop(N, name=...)` |

### 4.2 完整伪代码

> 必须标注每个变量类型（SymbolicScalar / int / Tensor），以及 view/assemble 的 offset。

```python
@pypto.frontend.jit
def {op}_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, D], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, D], pypto.DT_BF16),
):
    pypto.set_vec_tile_shapes(...)

    B = x.shape[0]    # SymbolicScalar
    for b in pypto.loop(B, name="batch"):
        x_tile = pypto.view(x, [1, D], [b, 0])
        # 计算
        ...
        pypto.assemble(result, [b, 0], out)
```

### 4.3 跨迭代状态（如有）

| 状态名 | 初始化 | 更新方式 | submit_before_loop |
|--------|--------|---------|--------------------|
|        |        |         |                    |

### 4.4 尾块处理

- **方案**：`pypto.view` / `pypto.reshape` 传 `valid_shape=...` / padding / 不需要（说明原因）
  - 注意：`valid_shape` 是 `pypto.view` / `pypto.reshape` 的参数，**不是** `pypto.assemble` 的参数

---

## 5. 约束自检清单

| # | 约束 | 是否满足 | 备注 |
|---|------|---------|------|
| 1 | 所有 sum 输入已转 FP32 |  |  |
| 2 | matmul 两侧 dtype 一致 |  |  |
| 3 | TileShape 维度数 = 操作数维度数 |  |  |
| 4 | Stage 7 前 Tile shape 基线已应用，或已说明硬约束例外 |  |  |
| 5 | 尾轴满足对齐 |  |  |
| 6 | 同阶段 UB 占用 ≤ 容量 |  |  |
| 7 | 表达式展开 < 18000 |  |  |
| 8 | 输出经 `[:]` / `assemble` 显式写回 |  |  |
| 9 | 无 view/assemble 同张量回环 |  |  |
| 10 | 动态轴标 `pypto.DYNAMIC` |  |  |
| 11 | 动态 loop 提供 `unroll_list`（**嵌套时仅最内层 `pypto.loop`**） |  |  |
| 12 | `unroll_list` 在 Stage 6 之前**只含单一值**（默认 `[1]`；有依据时可用其它单值并在 §4 记录理由；禁止多值，调优留到 Stage 7） |  |  |
| 13 | 跨迭代状态用 `submit_before_loop=True` |  |  |
| 14 | 尾块在 `pypto.view` / `pypto.reshape` 处传 `valid_shape=...`（**不是** `pypto.assemble` 的参数） |  |  |
| 15 | 无 SymbolicScalar 用作 `**` / list index / Python `if` |  |  |

### 开放问题

| # | 问题 | 影响范围 | 待解决方式 |
|---|------|---------|-----------|

---

## 6. 验证方案

### 6.1 测试配置

| 用例 | 输入 shape | dtype | 重点验证 |
|------|----------|-------|---------|
|      |          |       |         |

### 6.2 精度容忍度

| dtype | rtol | atol |
|-------|------|------|
| FP32  | 1e-5 | 1e-5 |
| BF16  | 1e-3 | 1e-3 |
