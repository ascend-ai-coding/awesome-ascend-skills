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

- **同时驻留 UB 的 Tensor**：

| Tensor | 用途 | shape | dtype | 大小估算 |
|--------|------|-------|-------|------|
| {name} | {role} | {shape} | {dtype} | {bytes} |

- **推导步骤**：
  1. 尾轴对齐：{dtype} → {N} 元素对齐
  2. UB 预算：tensor_count × tile_size × dtype_bytes ≤ UB
  3. 展开约束：`tile_size × repeat ≤ 展开上限 18000`

- **最终 tile**：

```python
pypto.set_vec_tile_shapes({tile_dims})
# 若含 matmul：
pypto.set_cube_tile_shapes([{M}, {K}], [{K}, {N}], [{M}, {N}])
```

### 3.3 替代方案

| 备选 tile | 否决理由 |
|-----------|---------|
| {tile} | {reason} |

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

- **方案**：valid_shape / padding / 不需要（说明原因）

---

## 5. 约束自检清单

| # | 约束 | 是否满足 | 备注 |
|---|------|---------|------|
| 1 | 所有 sum 输入已转 FP32 |  |  |
| 2 | matmul 两侧 dtype 一致 |  |  |
| 3 | TileShape 维度数 = 操作数维度数 |  |  |
| 4 | 尾轴满足对齐 |  |  |
| 5 | 同阶段 UB 占用 ≤ 容量 |  |  |
| 6 | 表达式展开 < 18000 |  |  |
| 7 | 输出经 `[:]` / `assemble` 显式写回 |  |
| 8 | 无 view/assemble 同张量回环 |  |
| 9 | 动态轴标 `pypto.DYNAMIC` |  |
| 10 | 动态 loop 提供 `unroll_list` |  |
| 11 | 跨迭代状态用 `submit_before_loop=True` |  |
| 12 | 尾块用 `valid_shape` 处理 |  |
| 13 | 无 SymbolicScalar 用作 `**` / list index / Python `if` |  |

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

