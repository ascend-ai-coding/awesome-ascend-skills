# 值依赖分析

> 算子输入是否在 Tiling 阶段需要读取值（如 axes tensor），影响框架三处契约声明。

## 1 值依赖判定法则

**infershape 或 tiling 阶段要解引用该张量的 `GetData<T>()` → 必须值依赖。** 反之（只读 shape/dtype）→ 不需要。

| 输入语义 | 是否值依赖 |
|---------|----------|
| 主数据张量 `x` | ❌ |
| 轴号张量 `axes` / `dims` | ✅ |
| 标量参数张量（mean/var/scale） | ❌（一般） |
| shape 描述张量（如 reshape 的 target_shape） | ✅ |

## 2 三处契约声明（缺一不可）

值依赖张量必须在以下三处同时声明，缺一会导致 infershape/tiling 阶段拿不到数据：

| # | 位置 | 声明 |
|---|------|------|
| 1 | `op_def.cpp` 输入定义 | `this->Input("axes").ParamType(REQUIRED).ValueDepend(OPTIONAL)...` |
| 2 | `IMPL_OP_INFERSHAPE` 注册块 | `.InputsDataDependency({idx})` |
| 3 | `IMPL_OP_OPTILING` 注册块 | `.TilingInputsDataDependency({idx})` |

- `idx` = 该输入在 Input 链中的索引（从 0 开始）
- `ValueDepend(OPTIONAL)`：该输入可以是图编译期常量或运行期张量
- `ValueDepend(REQUIRED)`：强制该输入是常量

## 3 典型范式示例

| 范式 | 值依赖 | 说明 |
|------|--------|------|
| reduction | axes tensor | `this->Input("axes").ParamType(REQUIRED).ValueDepend(OPTIONAL)` |
| layout_transform | target_shape tensor | reshape 类算子的目标形状张量 |

## 4 一致性约束

- **infershape 与 tiling 对同一输入的值依赖声明必须一致**：若 infershape 声明了值依赖而 tiling 漏配，tiling 阶段 `GetInputTensor` 拿不到 data
- **空 tensor 不在 infershape 端短路**：output shape 自然带 0 维，由 tiling/kernel 端处理
