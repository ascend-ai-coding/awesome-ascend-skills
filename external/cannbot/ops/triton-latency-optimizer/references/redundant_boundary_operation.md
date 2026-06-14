# 消除冗余的边界运算

## 概述

冗余的边界运算是指：当 `tl.load` 已经通过 `mask + other=d` 在边界区域确定了常量值 `d`，后续任何显式将该区域重置为 `d` 的运算都是冗余的。

这不仅是 `tl.where` 的问题。开发者常用 `* mask`、`+ 0`、`- 0` 等方式隐式做边界保护，这些在 Ascend NPU 上同样会引入额外的向量选择指令或算术单元占用，导致循环体膨胀、流水打断。

本 Skill 提供一个统一的**已知值区域(Known-Value Region)** 分析框架，用于识别并消除所有基于边界已知值的冗余运算。

---

## 核心抽象：Known-Value Region（KVR）

对于张量 `T`，若存在 mask `M`，使得在 `M=False`（或 `M=True`）的所有位置上 `T` 的值**确定为常量 `C`**，则称 `T` 具有已知值区域 `(M, C)`。

**传播规则**：纯函数运算的输出 KVR 由其输入的 KVR 按标量语义推导。

**冗余判定**：若某运算的目的就是将区域 `(M, C)` 设为 `C`，而输入在该区域上**已经确定等于 `C`**，则该运算冗余，可直接删除或替换为输入本身。

---

## 适用条件

### 1. 数据源已具备已知值区域

| 来源 | KVR 推导 | 说明 |
|------|---------|------|
| `tl.load(..., mask=M, other=C)` | `(M, C)` | 边界处值为 `C` |
| `tl.full(shape, C)` | `(⊤, C)` | 全张量值为 `C` |
| `tl.broadcast_to(C, shape)` | `(⊤, C)` | 标量广播 |
| `tl.where(M, X, C)` | `(M, C)` | `M=False` 处语义保证为 `C`，天然成立 |
| `tl.where(M, X, C)` | `(⊤, C)` | 若 `X` 在 `M=True` 处 KVR 亦为 `C`，则全张量为 `C` |

### 2. 运算链纯封闭（无副作用）

允许运算：
- 算术：`+ - * ** .to()`
- 逐元素：`exp abs max min clamp` 等
- 归约：`sum`（当输入 KVR 为 `0` 时，输出 KVR 亦为 `0`）

**禁止**运算（保守跳过）：
- `/ //`（除零风险）
- `store`、`atomic_add` 等副作用操作
- 自定义函数、控制流

### 3. 冗余操作与 KVR 匹配

| 运算形式 | 冗余条件 | 重写规则 |
|---------|---------|---------|
| `tl.where(M, expr, C)` | `expr` 在 `M=False` 处 KVR 为 `C` | → `expr` |
| `expr * 1.0` | 恒等 | → `expr` |
| `expr + 0.0` | 恒等 | → `expr` |
| `expr - 0.0` | 恒等 | → `expr` |
| `expr ** 1` | 恒等 | → `expr` |
| `tl.maximum(expr, C)` | `expr` 在相关区域 KVR ≥ `C` | → `expr` |
| `tl.minimum(expr, C)` | `expr` 在相关区域 KVR ≤ `C` | → `expr` |
| `tl.abs(expr)` | `expr` 在相关区域 KVR ≥ `0` | → `expr` |

---

## 常见冗余模式

| 数据源 | 运算链 | 冗余运算 |
|--------|--------|---------|
| `load(..., other=0.0)` | `a + b` | `where(m, a+b, 0.0)` |
| `load(..., other=0.0)` | `a * b` | `where(m, a*b, 0.0)`、`a*b * mask` |
| `load(..., other=0.0)` | `exp(a+b)` | `where(m, exp(a+b), 1.0)` |
| `load(..., other=0.0)` | `sum(x_sq, axis=0)` | `where(m, sum(x_sq), 0.0)` |
| `load(..., other=1.0)` | `a * b` | `where(m, a*b, 1.0)` |
| `load(..., other=-inf)` | `max(a, b)` | `where(m, max(a,b), -inf)` |
| `load(..., other=+inf)` | `min(a, b)` | `where(m, min(a,b), +inf)` |
---

## 非冗余场景（禁止删除）

| 场景 | 原因 |
|------|------|
| 运算链含 `/` 或 `//` | `0/0=NaN`，边界值不确定 |
| `where/min/max` 的 mask 与 `load` 的 mask **不同** | 保护范围不一致 |
| `where` 的 default 与 `load` 的 other **不同** | 边界目标值不匹配 |
| 运算链含未受保护的 `tl.load`（无 mask） | 引入了不确定性 |
---

## 优化建议

### 核心思想

不针对单一算子做模式匹配，而是建立 KVR 数据流分析：
1. 从 `tl.load(..., mask=M, other=C)` 和 `tl.full(C)` 建立初始 KVR
2. 按标量语义向前传播 KVR
3. 遇到 `where / *mask / +0 / *1 / max / min / abs` 等运算时，检查输入的 KVR 是否已满足运算目标
4. 若满足，删除冗余运算

### 示例一：where 冗余（RMSNorm）

```python
# 优化前
h_val = tl.load(ptr_h + idx, mask=m, other=0.0)
r_val = tl.load(ptr_r + idx, mask=m, other=0.0)
x_f32 = h_val.to(tl.float32) + r_val.to(tl.float32)
x_sq  = x_f32 * x_f32
x_sq  = tl.where(m, x_sq, 0.0)          # ❌ 冗余：0+0=0, 0*0=0
sum_sq = tl.sum(x_sq, axis=0)

# 优化后
h_val = tl.load(ptr_h + idx, mask=m, other=0.0)
r_val = tl.load(ptr_r + idx, mask=m, other=0.0)
x_f32 = h_val.to(tl.float32) + r_val.to(tl.float32)
x_sq  = x_f32 * x_f32                    # ✅ 边界处自然为 0.0
sum_sq = tl.sum(x_sq, axis=0)
```

### 示例二：乘法模拟 mask 冗余

```python
# 优化前
a = tl.load(ptr_a + idx, mask=m, other=0.0)
b = tl.load(ptr_b + idx, mask=m, other=0.0)
x = (a + b) * m.to(tl.float32)           # ❌ 冗余：边界处 a+b 已是 0

# 优化后
a = tl.load(ptr_a + idx, mask=m, other=0.0)
b = tl.load(ptr_b + idx, mask=m, other=0.0)
x = a + b                                # ✅ 删除 *mask
```

### 示例三：复合 KVR 传播（exp）

```python
# 优化前
a = tl.load(ptr_a + idx, mask=m, other=0.0)
b = tl.load(ptr_b + idx, mask=m, other=0.0)
x = tl.exp(a + b)
x = tl.where(m, x, 1.0)                  # ❌ 冗余：exp(0+0)=1.0

# 优化后
a = tl.load(ptr_a + idx, mask=m, other=0.0)
b = tl.load(ptr_b + idx, mask=m, other=0.0)
x = tl.exp(a + b)                        # ✅ 边界处自然为 1.0
```

---

## 关键点

1. **KVR 是统一的分析框架**
   - 不针对 `where`、`+0`、`*1` 分别写死规则，而是统一问："边界处是否已经等于目标值？"
   - 新增冗余模式只需补充标量常量折叠表，无需改动分析框架。

2. **除法是唯一红线**
   - 运算链中只要出现 `/` 或 `//`，整链的 KVR 传播立即截断，保守保留外层保护。
   - 因为 `0/0=NaN` 会污染后续 `sum` 等归约，即使边界值"看起来"是 0 也不安全。

3. **sum 的 KVR 可传播**
   - `tl.sum(x, axis=0)` 若输入 `x` 的 KVR 为 `(M, 0)`，则输出 KVR 亦为 `0`。
   - 这是 RMSNorm / LayerNorm 中最常见的"where(..., 0.0)后接sum"场景的消除依据。

4. **store 不参与 KVR**
   - `tl.store` 的 `mask` 是副作用保护，不是值语义。KVR 分析只针对纯算术/逐元素运算链，不跨越 store。
