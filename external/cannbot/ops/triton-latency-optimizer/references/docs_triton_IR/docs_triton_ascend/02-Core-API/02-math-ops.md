# 数学运算 API 及 NPU 适配差异

## 概述

本文档详细描述 Triton-Ascend 中的数学运算 API，包括算术运算（add/sub/mul/div）、初等函数（exp/log/sin/cos/sqrt/rsqrt/erf）、取整函数（ceil/floor）、融合运算（fma/sigmoid/softmax）等。在 Ascend NPU 上，数学运算由 Vector 计算单元执行，与 GPU 的 CUDA Core 对应。NPU 对部分数据类型的支持存在差异，特别是 bfloat16 的部分操作需要先提升为 float32 再计算，以及 fp16/bf16 的除法操作需要提升精度。

关键词：数学运算, abs, add, sub, mul, div, exp, log, sin, cos, sqrt, rsqrt, erf, ceil, floor, fma, sigmoid, softmax, NPU, Vector, bfloat16, 精度提升

---

## API 参考

### 算术运算

#### tl.add / `+`

```python
triton.language.add(x, y, sanitize_overflow=True, _builder=None) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `tensor` 或 `scalar` | 第一个操作数 |
| `y` | `tensor` 或 `scalar` | 第二个操作数 |
| `sanitize_overflow` | `bool`，默认 `True` | 是否检测整数溢出 |

支持所有数值类型。当操作数为指针时，执行指针偏移计算（`addptr`）。浮点使用 `fadd`，整数使用 `add`。

#### tl.sub / `-`

```python
triton.language.sub(x, y, sanitize_overflow=True, _builder=None) -> tensor
```

参数同 `add`。支持指针减偏移（`addptr` with negation）。

#### tl.mul / `*`

```python
triton.language.mul(x, y, sanitize_overflow=True, _builder=None) -> tensor
```

参数同 `add`。浮点使用 `fmul`，整数使用 `mul`。

#### tl.div / `/`

```python
# 真除法（自动类型提升）
x / y  # 等价于 tl.truediv(x, y)

# 快速除法
triton.language.fdiv(x, y, ieee_rounding=False, _builder=None) -> tensor

# 精确除法
triton.language.div_rn(x, y, _builder=None) -> tensor
```

| 函数 | 支持类型 | 说明 |
|------|---------|------|
| `truediv` | 所有数值 | 整数除法自动提升为 float32；fp16/bf16 除法提升为 float32 |
| `fdiv` | 浮点类型 | 快速除法，不保证 IEEE 精度 |
| `div_rn` | bf16/fp16/fp32/fp8/fp64 | 精确除法，IEEE 四舍五入 |

**NPU 注意**：fp16 和 bf16 的除法运算在 NPU 上会自动提升为 float32 执行，这是由 `computation_type_impl` 中的规则决定的。

#### tl.mod / `%`

```python
x % y  # 取模运算
```

浮点使用 `frem`，有符号整数使用 `srem`，无符号整数使用 `urem`。不同符号类型的整数不能直接取模。

#### tl.cdiv

```python
triton.language.cdiv(x, div, _builder=None) -> tensor
```

向上取整除法。整数使用 `(x + div - 1) // div`，浮点使用 `ceil(x / div)`。

### 初等函数

以下函数均支持 `bf16, fp16, fp32, fp8e4nv, fp8e5, fp64` 类型（除非特别说明）：

| 函数 | 签名 | 说明 | NPU 支持类型 |
|------|------|------|-------------|
| `exp` | `exp(x) -> tensor` | 逐元素自然指数 e^x | bf16, fp16, fp32, fp8, fp64 |
| `exp2` | `exp2(x) -> tensor` | 逐元素 2^x | bf16, fp16, fp32, fp8, fp64 |
| `log` | `log(x) -> tensor` | 逐元素自然对数 | bf16, fp16, fp32, fp8, fp64 |
| `log2` | `log2(x) -> tensor` | 逐元素 log2 | bf16, fp16, fp32, fp8, fp64 |
| `sin` | `sin(x) -> tensor` | 逐元素正弦 | bf16, fp16, fp32, fp8, fp64 |
| `cos` | `cos(x) -> tensor` | 逐元素余弦 | bf16, fp16, fp32, fp8, fp64 |
| `sqrt` | `sqrt(x) -> tensor` | 快速平方根 | bf16, fp16, fp32, fp8, fp64 |
| `sqrt_rn` | `sqrt_rn(x) -> tensor` | 精确平方根（IEEE 四舍五入） | bf16, fp16, fp32, fp8, fp64 |
| `rsqrt` | `rsqrt(x) -> tensor` | 平方根倒数 1/sqrt(x) | bf16, fp16, fp32, fp8, fp64 |
| `erf` | `erf(x) -> tensor` | 误差函数 | bf16, fp16, fp32, fp8, fp64 |
| `abs` | `abs(x) -> tensor` | 绝对值（支持所有数值类型） | 所有数值类型 |
| `floor` | `floor(x) -> tensor` | 向下取整（仅浮点） | bf16, fp16, fp32, fp8, fp64 |
| `ceil` | `ceil(x) -> tensor` | 向上取整（整数类型为 no-op） | bf16, fp16, fp32, fp8, fp64 |

### 融合运算

#### tl.fma

```python
triton.language.fma(x, y, z, _builder=None) -> tensor
```

融合乘加：`x * y + z`。三个操作数会进行类型统一化。

#### tl.sigmoid

```python
triton.language.sigmoid(x) -> tensor
```

Sigmoid 函数：`1 / (1 + exp(-x))`。通过 JIT 编译实现。

#### tl.softmax

```python
triton.language.softmax(x, ieee_rounding=False) -> tensor
```

Softmax 函数：`exp(x - max(x)) / sum(exp(x - max(x)))`。通过 JIT 编译实现。

### 其他运算

| 函数 | 签名 | 说明 |
|------|------|------|
| `umulhi` | `umulhi(x, y) -> tensor` | 2N 位乘积的高 N 位，仅支持 int32/int64/uint32/uint64 |
| `maximum` | `maximum(x, y, propagate_nan=NONE) -> tensor` | 逐元素最大值 |
| `minimum` | `minimum(x, y, propagate_nan=NONE) -> tensor` | 逐元素最小值 |
| `clamp` | `clamp(x, min, max, propagate_nan=NONE) -> tensor` | 将值限制在 [min, max] 范围内 |

### 类型提升规则

Triton 遵循以下隐式类型提升规则（由 `computation_type_impl` 实现）：

1. 若任一操作数为 float64，结果为 float64
2. 若任一操作数为 float32，结果为 float32
3. 若任一操作数为 fp16，除除法外结果为 fp16；除法提升为 float32
4. 仅当两个操作数均为 bf16 时结果为 bf16；bf16 与其他类型运算提升为 float32；bf16 除法提升为 float32
5. 不同 fp8 类型运算提升为 fp16

---

## 代码示例

### 基础用法：算术运算与初等函数

```python
import triton
import triton.language as tl

@triton.jit
def math_basic_kernel(
    x_ptr, y_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    result = tl.exp(x) + tl.sin(y) * tl.sqrt(x)
    result = tl.fma(x, y, result)
    result = tl.abs(result)
    result = tl.clamp(result, 0.0, 100.0)

    tl.store(out_ptr + offsets, result, mask=mask)
```

### 进阶用法：GELU 激活函数实现

```python
@triton.jit
def gelu_kernel(
    x_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    inner = x * sqrt_2_over_pi * (1.0 + coeff * x * x)
    tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    result = 0.5 * x * (1.0 + tanh_inner)

    tl.store(out_ptr + offsets, result, mask=mask)
```

---

## NPU 适配要点

### 1. bfloat16 精度提升

NPU 的 Vector 计算单元**不支持** bfloat16 的 FMAX、FMIN、FCMP 操作。因此，当 bfloat16 张量参与 `maximum`、`minimum`、比较运算时，会自动提升为 float32 执行。这由 `_promote_bfloat16_to_float32` 函数处理：

```python
def _promote_bfloat16_to_float32(t, _builder=None):
    scalar_ty = t.type.scalar
    if scalar_ty is bfloat16:
        return t.to(float32, _builder=_builder)
    return t
```

### 2. fp16/bf16 除法精度提升

fp16 和 bf16 的除法运算（`/`、`fdiv`）在 NPU 上会自动提升为 float32 执行。这是因为 NPU 的 Vector 单元不原生支持低精度浮点除法。如果需要低精度结果，需要在计算后显式转换回 fp16/bf16。

### 3. NPU 上不支持的数学函数

以下函数在 NPU 上可能不支持或有特殊限制：

| 函数 | 限制 |
|------|------|
| `umulhi` | 仅支持 int32/int64/uint32/uint64 |
| `sqrt_rn` | 精确平方根可能比快速平方根慢 |
| `div_rn` | 精确除法可能比快速除法慢 |

### 4. 浮点精度差异

NPU Vector 单元的浮点计算精度与 GPU CUDA Core 可能存在微小差异：
- `sqrt` / `rsqrt` 的快速版本可能不满足 IEEE 精度要求
- `fdiv` 的快速除法可能不满足 IEEE 精度要求
- 需要精确结果时，应使用 `sqrt_rn` 和 `div_rn`

### 5. sigmoid / softmax 的实现

`sigmoid` 和 `softmax` 是通过 JIT 编译的高层函数，不是底层硬件指令：
- `sigmoid(x) = 1 / (1 + exp(-x))`
- `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`

这些函数在 NPU 上的行为与 GPU 一致，但中间计算的精度提升规则可能影响最终结果的数值精度。

---

## 常见问题

**Q1: 为什么 bf16 张量的 maximum/minimum 结果是 float32 类型？**

A: NPU 硬件不支持 bf16 的 FMAX/FMIN 操作，因此 `_promote_bfloat16_to_float32` 会自动将 bf16 提升为 float32。如果需要 bf16 结果，需显式转换：`tl.maximum(x, y).to(tl.bfloat16)`。

**Q2: fp16 除法在 NPU 上的精度如何？**

A: fp16 除法会自动提升为 float32 执行，精度与 float32 除法一致。结果类型取决于类型提升规则。

**Q3: sqrt 和 sqrt_rn 应该选哪个？**

A: 如果对精度要求不高（如神经网络推理），使用 `sqrt` 即可，性能更好。如果需要 IEEE 精确结果（如数值算法），使用 `sqrt_rn`。

**Q4: 为什么整数除法会变成浮点？**

A: Triton 的 `/` 运算符执行真除法（`truediv`），整数除法会自动提升为 float32。如果需要整数地板除，使用 `//` 运算符。

**Q5: fma 和 x * y + z 有什么区别？**

A: `fma` 是融合乘加，在硬件层面只进行一次舍入，精度更高。`x * y + z` 先对乘法结果舍入，再对加法结果舍入，可能引入额外的精度损失。

---

## 相关文档

- [01-memory-ops.md](./01-memory-ops.md) - 内存操作 API
- [03-reduction-ops.md](./03-reduction-ops.md) - 归约操作
- [04-linear-algebra-ops.md](./04-linear-algebra-ops.md) - 线性代数操作

## 源码参考

- [math.py - 数学函数定义](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/math.py)
- [semantic.py - 类型提升规则](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L46-L109)
- [semantic.py - 算术运算实现](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L219-L352)
- [core.py - _promote_bfloat16_to_float32](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L2100-L2106)
- [standard.py - sigmoid/softmax 实现](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/standard.py#L43-L57)
