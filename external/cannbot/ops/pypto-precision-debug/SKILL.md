---
name: external-cannbot-ops-pypto-precision-debug
description: PyPTO 算子精度问题排查技能。专注于用户代码层面的语法逻辑检查和规避方法尝试。当算子精度验证失败、输出结果异常、计算错误、数值偏差、或任何与精度相关的问题时使用此技能。
original-name: pypto-precision-debug
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-26'
synced-commit: ac5bbd2b4cf427d011874e11f8d1e8b1bef66eda
license: UNKNOWN
---

# PyPTO 算子精度问题排查技能

此技能专注于**用户代码层面**的精度问题排查，提供语法逻辑检查和规避方法。

## 核心定位

### ⭐ 职责范围

**本技能负责**：
- 用户代码语法逻辑检查
- 提供规避方法和工具
- 尝试各种可能的解决方案

**本技能不负责**：
- 底层框架代码分析
- 深层根因定位
- 框架 Bug 确认

**结论判断**：
- 如果规避方法有效 → 提供解决方案
- 如果规避方法无效 → 报告可能是底层框架层面问题

---

## 常见规避方法速查表

**⚠️ 遇到精度问题时，按优先级顺序逐一尝试以下规避方法。优先级 1 为默认推荐方案。**

| 优先级 | 问题现象 | 规避方法 | 代码示例 | 原因说明 |
|-------|---------|---------|---------|---------|
| 1 ★推荐 | 使用旧前端写法 | 切换到 `pypto.frontend.jit` | `@pypto.frontend.jit` | 新前端是 PyPTO 推荐写法，旧前端已不再维护，可避免多种已知问题 |
| 2 | view + reshape 精度异常 | 避免 `inplace=True` | `pypto.reshape(tensor, shape, inplace=False)` | inplace=True 在 view 后会错误修改内存地址，导致数据指向错误区域 |
| 3 | 循环展开后精度异常 | `unroll_list=[1]` | `pypto.loop(range(n), unroll_list=[1])` | 关闭循环展开，规避 RegisterCopy pass 的寄存器拷贝 bug |
| 4 | 嵌套循环精度异常 | `submit_before_loop=True` | `pypto.loop(range(m), submit_before_loop=True)` | 确保子循环正确提交，避免并行执行时的内存覆盖 |
| 5 | 特定 shape 精度异常 | 调整 shape | 避免尾轴为 1，避免非整除 | 特定 shape 可能触发 Pass 推导边界情况，导致 valid_shape 错误 |
| 6 | 编译器优化异常 | `+0.0` 技巧 | `result = compute(...) + 0.0` | 阻止编译器过度优化，保留计算操作完整性 |
---

## ⭐ 重要提示：使用新前端写法

**强烈建议使用 `pypto.frontend.jit` 而非 `pypto.jit`！**

新前端（`pypto.frontend.jit`）是 PyPTO 推荐的写法，旧前端（`pypto.jit`）已不再维护。使用新前端可避免许多已知问题。

**推荐写法**：

```python
# ✅ 推荐：新前端写法
@pypto.frontend.jit
def my_kernel(
    input_tensor: pypto.Tensor(shape, pypto.DT_FP32),
    output_tensor: pypto.Tensor(shape, pypto.DT_FP32),
):
    ...

# ❌ 不推荐：旧前端写法（已废弃）
@pypto.jit
def my_kernel(input_tensor, output_tensor):
    ...
```

---

## 排查决策树

```
精度问题
    │
    ├─ 步骤 0：前端写法检查
    │   └─ 使用 pypto.jit？ ──是──▶ 切换到 pypto.frontend.jit 重试
    │
    ├─ 步骤 1：用户代码语法检查
    │   ├─ 数据类型正确？
    │   ├─ shape 定义正确？
    │   └─ valid_shape 配置正确？
    │
    ├─ 步骤 2：快速规避方法尝试
    │   ├─ 避免 view + reshape inplace=True
    │   ├─ unroll_list=[1]
    │   ├─ submit_before_loop=True
    │   └─ +0.0 技巧
    │
    ├─ 步骤 3：二分定位（如需要）
    │
    └─ 步骤 4：结论判断
        ├─ 规避方法有效 ──▶ 提供解决方案
        └─ 规避方法无效 ──▶ 报告可能是框架问题
```

---

## 完整工作流程

### 步骤 0：检查前端写法

**⚠️ 最高优先级：首先检查用户是否使用了旧前端写法！**

**检查内容**：查看用户代码中的装饰器是 `pypto.jit` 还是 `pypto.frontend.jit`

**依据参考**：
- [docs/tutorials/development/compile.md](../../../docs/tutorials/development/compile.md)：官方文档使用 `@pypto.frontend.jit` 作为标准写法
- [docs/tutorials/introduction/quick_start.md](../../../docs/tutorials/introduction/quick_start.md)：快速入门示例使用 `@pypto.frontend.jit`

**如果用户使用 `pypto.jit`**：

1. **立即建议用户切换到新前端**
2. **向用户展示修改示例**：

```python
# 修改前
@pypto.jit
def my_kernel(input_tensor, output_tensor):
    ...

# 修改后
@pypto.frontend.jit
def my_kernel(
    input_tensor: pypto.Tensor(shape, pypto.DT_FP32),
    output_tensor: pypto.Tensor(shape, pypto.DT_FP32),
):
    ...
```

3. 重新运行测试验证精度

**如果问题解决**：结束排查，建议用户使用新前端写法

**如果问题未解决**：继续下一步

**阶段总结**：
```markdown
## 步骤 0 总结：前端写法检查

### 检查结果
- 使用的前端：[pypto.jit / pypto.frontend.jit]
- 是否切换：[是/否]

### 结论
- [问题已解决 / 需要继续排查]
```

---

### 步骤 1：用户代码语法检查

**⚠️ 重要：先检查用户代码逻辑是否正确，再尝试规避方法。**

#### 1.1 数据类型检查

**检查项**：

- [ ] 输入输出数据类型是否一致
- [ ] 中间计算是否需要更高精度
- [ ] 是否存在隐式类型转换

**依据参考**：通用编程最佳实践。**注意：此检查项属于数值计算通用检查方法。**

**常见问题**：

| 问题 | 现象 | 解决方法 |
|-----|------|---------|
| FP16 精度损失 | 大数值计算误差 | 使用 FP32 或调整计算顺序 |
| 类型不匹配 | 计算结果异常 | 确保类型一致 |

#### 1.2 Shape 定义检查

**检查项**：

- [ ] shape 是否正确定义
- [ ] 动态轴是否正确标记

**依据参考**：通用编程最佳实践。**注意：此检查项属于张量编程通用检查方法。**

**常见问题**：

| 问题 | 现象 | 解决方法 |
|-----|------|---------|
| shape 与实际数据不匹配 | 越界或数据截断 | 检查 shape 定义 |
| 动态轴标记错误 | 编译或运行时错误 | 检查动态轴定义 |

#### 1.3 valid_shape 配置检查

**检查项**：

- [ ] view/reshape 时 valid_shape 参数是否正确
- [ ] 动态 shape 场景下 valid_shape 表达式是否正确

**依据参考**：
- [pypto.view API 文档](../../../docs/api/operation/pypto-view.md)：`valid_shape` 是 `view` 方法的参数，用于指定视图的有效数据大小
- [pypto.reshape API 文档](../../../docs/api/operation/pypto-reshape.md)：`valid_shape` 是 `reshape` 方法的参数，用于指定输出 Tensor 的有效数据 Shape

**示例**：

```python
# ✅ 正确：view 时直接传入 valid_shape 参数
tensor_view = pypto.view(tensor, shape=[4, 4], offsets=[0, 0], valid_shape=[2, 4])

# ✅ 正确：reshape 时直接传入 valid_shape 参数
tensor_reshaped = pypto.reshape(tensor, new_shape, valid_shape=valid_shape_expr)
```

**常见问题**：

| 问题 | 现象 | 解决方法 |
|-----|------|---------|
| valid_shape 未正确设置 | 边界数据处理错误 | 在 view/reshape 时正确传入 valid_shape 参数 |
| valid_shape 与实际数据不匹配 | 数据截断或越界 | 确保 valid_shape 与实际有效数据大小一致 |

**阶段总结**：
```markdown
## 步骤 1 总结：用户代码语法检查

### 检查结果
- 数据类型检查：[正常/异常 - 描述问题]
- Shape 定义检查：[正常/异常 - 描述问题]
- valid_shape 配置检查：[正常/异常 - 描述问题]

### 发现的问题
1. [问题描述1]
2. [问题描述2]
...

### 修复建议
1. [修复建议1]
2. [修复建议2]
...

### 结论
- [问题已解决 / 需要继续排查]
```

---

### 步骤 2：快速规避方法尝试

**⚠️ 重要：以下规避方法按优先级逐一尝试，记录每种方法的效果。**

#### 2.1 避免 view + reshape inplace=True

**适用场景**：`pypto.view` 后再做 `pypto.reshape(inplace=True)` 导致精度严重偏差

**问题现象**：精度严重偏差（96%+ 元素超容差）

**操作**：

```python
# ❌ 避免写法
tensor_view = pypto.view(tensor, new_shape, ...)
result = pypto.reshape(tensor_view, final_shape, inplace=True)  # 精度严重偏差

# ✅ 推荐写法
tensor_view = pypto.view(tensor, new_shape, ...)
result = pypto.reshape(tensor_view, final_shape, inplace=False)  # 精度正确
```

**判断**：问题是否解决

**依据参考**：Issue #343 - `pypto.view` 后再做 `pypto.reshape(inplace=True)` 时，内存地址或元数据处理错误，inplace=True 的 reshape 试图在原 tensor 上就地修改，但 view 的元数据（offset、valid_shape）未被正确传播，导致 reshape 结果指向错误的内存区域。

#### 2.2 尝试 unroll_list=[1]

**适用场景**：循环展开后精度异常

**操作**：

```python
# 在循环中添加 unroll_list=[1]
for i in pypto.loop(range(n), unroll_list=[1]):
    ...
```

**判断**：问题是否解决

**依据参考**：Issue #223, #341 - RegisterCopy pass 在处理 unroll 场景时，对 tensor 的寄存器拷贝逻辑存在 bug，某些 unroll 配置下 tensor 的 Copy 路径选择错误，导致数据写入位置偏移。设置 `unroll_list=[1]` 可关闭循环展开，规避此问题。

#### 2.3 尝试 submit_before_loop=True

**适用场景**：嵌套循环精度异常

**操作**：

```python
# 在子循环中添加 submit_before_loop=True
for i in pypto.loop(range(n)):
    for j in pypto.loop(range(m), submit_before_loop=True):
        ...
```

**判断**：问题是否解决

**依据参考**：[docs/api/controlflow/pypto-loop.md](../../../docs/api/controlflow/pypto-loop.md) - `submit_before_loop` 参数用于控制嵌套循环的调度行为，确保子循环在父循环迭代前正确提交，避免并行执行时的内存覆盖问题。

#### 2.4 尝试 +0.0 技巧

**适用场景**：编译器优化导致精度异常

**操作**：

```python
# 在计算结果上添加 + 0.0
result = compute(...) + 0.0
```

**判断**：问题是否解决

**依据参考**：编译器优化 pass 在某些场景下会错误地消除或重排序操作，添加 `+0.0` 可以阻止过度优化，保留计算操作的完整性。

#### 2.5 尝试调整 shape

**适用场景**：特定 shape 精度异常

**操作**：

- 调整 tensor shape，避免尾轴为 1
- 避免非整除情况
- 尝试不同的 shape 组合

**判断**：问题是否解决

**依据参考**：Issue #498, #787 - 特定 shape（如尾轴为 1、非整除）可能触发 Pass 推导的边界情况，导致 valid_shape 传播错误或 buffer 越界。调整 shape 可规避这些边界场景。

**阶段总结**：
```markdown
## 步骤 2 总结：快速规避方法尝试

### 尝试的规避方法及结果
| 方法 | 是否尝试 | 是否有效 | 备注 |
|-----|---------|---------|------|
| 避免 view + reshape inplace=True | [是/否] | [有效/无效] | [描述] |
| unroll_list=[1] | [是/否] | [有效/无效] | [描述] |
| submit_before_loop=True | [是/否] | [有效/无效] | [描述] |
| +0.0 技巧 | [是/否] | [有效/无效] | [描述] |
| 调整 shape | [是/否] | [有效/无效] | [描述] |

### 有效的规避方法
[如果有，列出有效的方法]

### 结论
- [问题已解决 / 需要继续排查（二分定位）]
```

---

### 步骤 3：二分定位（如需要）

如果上述方法无法定位问题，使用 `pypto-precision-compare` skill 查找定位具体问题 op。

**阶段总结**：
```markdown
## 步骤 3 总结：二分定位

### 二分定位结果
- 是否执行二分定位：[是/否]
- 定位到的问题 op：[op 名称/未定位到]
- 问题代码位置：[文件名:行号]

### 问题分析
[描述定位到的问题及可能原因]

### 结论
- [已定位到问题 / 未定位到问题]
```

---

### 步骤 4：结论判断

#### 情况 A：规避方法有效

**输出**：

```markdown
## 排查结论

### 问题类型
[描述问题类型]

### 有效的规避方法
[描述有效的规避方法]

### 解决方案
[提供具体的代码修改建议]

### 注意事项
[使用规避方法时的注意事项]
```

#### 情况 B：规避方法无效

**输出**：

```markdown
## 排查结论

### 问题现象
[描述问题现象]

### 已尝试的规避方法
1. [方法1] - 无效
2. [方法2] - 无效
3. [方法3] - 无效
...

### 结论
经过多种规避方法尝试，问题仍未解决。**可能是底层框架层面的问题**。
```

---

## 检查清单

使用此 skill 时，确保：

- [ ] **步骤 0**：检查前端写法（最高优先级）
  - [ ] 检查是否使用 `pypto.jit`
  - [ ] 如使用旧写法，建议切换到 `pypto.frontend.jit`

- [ ] **步骤 1**：用户代码语法检查
  - [ ] 数据类型
  - [ ] Shape 定义
  - [ ] valid_shape 配置

- [ ] **步骤 2**：快速规避方法尝试
  - [ ] 避免 view + reshape inplace=True
  - [ ] unroll_list=[1]
  - [ ] submit_before_loop=True
  - [ ] +0.0 技巧
  - [ ] 调整 shape

- [ ] **步骤 3**：二分定位（如需要）

- [ ] **步骤 4**：结论判断
  - [ ] 规避方法有效 → 提供解决方案
  - [ ] 规避方法无效 → 报告可能是框架问题

---

## 参考资料

### 文档资料

| 文档 | 路径 | 说明 |
|------|------|------|
| 已知问题文档 | [docs/tutorials/appendix/issue.md](../../../docs/tutorials/appendix/issue.md) | 常见问题及解决方案 |
| 循环开发指南 | [docs/tutorials/development/loops.md](../../../docs/tutorials/development/loops.md) | loop 使用方法 |

### API 文档

| API | 文档路径 | 说明 |
|-----|---------|------|
| pypto.loop | [docs/api/controlflow/pypto-loop.md](../../../docs/api/controlflow/pypto-loop.md) | 循环接口，含 submit_before_loop 参数 |
| pypto.loop_unroll | [docs/api/controlflow/pypto-loop_unroll.md](../../../docs/api/controlflow/pypto-loop_unroll.md) | 循环展开接口，含 unroll_list 参数 |
| pypto.tensor | [docs/tutorials/development/tensor_creation.md](../../../docs/tutorials/development/tensor_creation.md) | Tensor 创建方法 |
| pypto.reshape | [docs/api/operation/pypto-reshape.md](../../../docs/api/operation/pypto-reshape.md) | reshape 操作，含 inplace 参数 |
| pypto.view | [docs/api/operation/pypto-view.md](../../../docs/api/operation/pypto-view.md) | view 操作 |
| valid_shape | [docs/tutorials/development/tensor_operation.md](../../../docs/tutorials/development/tensor_operation.md) | Tensor 操作方法，含 valid_shape 说明 |
