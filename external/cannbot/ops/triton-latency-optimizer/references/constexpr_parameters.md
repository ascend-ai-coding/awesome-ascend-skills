# 入参静态化 优化模式

## 概述

在 Triton NPU kernel 中，将固定数值的入参声明为 `tl.constexpr`，可以让编译器在编译时进行更多的常量折叠和常量传播优化，从而提升 kernel 的执行效率。

## 触发条件

**当代码中存在以下固定数值参数时，应考虑将其声明为 `tl.constexpr`**：

1. **固定的 BLOCK_SIZE**：如 `BLOCK_M`、`BLOCK_N`、`BLOCK_K` 等
2. **固定的 STRIDE**：如 `stride_m`、`stride_n` 等
3. **模型配置超参数**：如 MoE 场景中的 `num_experts`、`topk_numel`、`seq_len` 等。这些值在模型训练/推理过程中通常是固定配置（如 `num_experts=128`），不应仅凭变量名判断为运行时变量。若该参数来自 Python 层的固定配置，应优先尝试声明为 `tl.constexpr`
4. **其他在 kernel 生命周期内不会变化的常量参数**

如果已有入参中的某个参数对性能影响很大，且在kernel生命周期内不会变化，如若不确定则应该**询问用户是否可以将该参数设置为 `tl.constexpr`**。

## 优化方法

### 原始代码

```python
@triton.jit
def kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_an,  # 这些是入参，但实际运行时是固定值
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # ...
```

### 优化后代码

```python
@triton.jit
def kernel(
    A, B, C,
    M, N, K,
    stride_am: tl.constexpr,  # 声明为 constexpr
    stride_an: tl.constexpr,  # 声明为 constexpr
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # ...
```

## 关键点

1. **常量性质**：只有那些在 kernel 运行时不会变化的参数才适合声明为 `tl.constexpr`
2. **性能影响**：对于性能敏感的参数（如 BLOCK_SIZE），应优先考虑声明为 `tl.constexpr`
3. **用户确认**：如果不确定某个参数是否可以设为 constexpr，应询问用户

## 性能收益

将固定参数声明为 `tl.constexpr` 可以：
- 启用编译时常量折叠
- 帮助编译器进行更 aggressive 的常量传播
- 减少运行时分支判断开销
