# libdevice 数学函数库

## 概述

`libdevice` 是 Triton-Ascend 提供的数学函数库，封装了昇腾 NPU 上高性能数学运算的外部函数调用（extern call）。与标准 `tl.math` 模块不同，`libdevice` 的函数直接映射到华为 HMF（Huawei Math Function）底层库的硬件加速实现，在特定数据类型和场景下具有更高的精度和性能。

`libdevice` 模块中的函数存在**两种实现路径**：
- **libdevice 路径**（`TRITON_ENABLE_LIBDEVICE` 开启时）：直接调用 HMF 库的 `__hmf_*` 函数，通常仅支持 fp32，精度更高
- **标准路径**（`TRITON_ENABLE_LIBDEVICE` 关闭时，默认）：部分函数使用 Triton IR 原生操作组合实现，支持更多数据类型（fp16/bf16/fp32），但精度可能略低

## 导入方式

```python
from triton.language.extra.cann import libdevice
```

## 关键概念

### TRITON_ENABLE_LIBDEVICE 环境变量

`TRITON_ENABLE_LIBDEVICE` 控制 libdevice 函数的实现路径选择：

| 值 | 行为 | 影响 |
|----|------|------|
| `False`（默认） | 使用标准路径实现 | 支持 fp16/bf16/fp32 多种数据类型，部分函数通过 Triton IR 原生操作组合实现 |
| `True` | 使用 libdevice 路径实现 | 仅支持 fp32（部分支持 fp16），直接调用 HMF 库函数，精度更高 |

```python
import os
os.environ["TRITON_ENABLE_LIBDEVICE"] = "True"
```

### libdevice 路径 vs 标准路径的差异

| 维度 | libdevice 路径 | 标准路径 |
|------|---------------|---------|
| 数据类型支持 | 通常仅 fp32（部分 fp16） | fp16/bf16/fp32 |
| 底层实现 | HMF `__hmf_*` 函数 | Triton IR 原生操作组合 |
| 精度 | 更高（硬件优化的数学库） | 可能略低（多项式近似或操作组合） |
| bf16 支持 | 通常不支持（会 static_assert 失败） | 支持（先转 fp32 计算） |
| 编译依赖 | 需要链接 libdevice bitcode | 无额外依赖 |

### 何时使用 libdevice

1. **需要更高精度的数学运算**：如 `pow`、`tanh`、`erfinv` 等对精度敏感的函数
2. **`tl.math` 中缺少的函数**：如 `log1p`、`relu`、`atan2`、`fmod` 等在标准 `tl.math` 中未提供的函数
3. **NPU 特有优化**：如 `fast_dividef`、`fast_expf`、`div_rz` 等硬件加速版本

## API 参考

### 一元数学函数

#### libdevice.pow

```python
libdevice.pow(base, exponent, _builder=None) -> tensor
```

计算 `base^exponent`。

| 路径 | 支持类型 | 底层函数 |
|------|---------|---------|
| libdevice | fp32 | `__hmf_pow_fp32` |
| 标准 | fp32, fp16, bf16 | `__hmf_powf` / `__hmf_powDh` / `__hmf_powDb` |

**使用示例**（RoPE 算子中的频率计算）：

```python
from triton.language.extra.cann import libdevice

offset_n = tl.arange(0, DIM // 2)
inv_freq = libdevice.pow(base, -2.0 / DIM * offset_n)
```

#### libdevice.tanh

```python
libdevice.tanh(x, _builder=None) -> tensor
```

计算双曲正切。bf16 输入会自动提升为 fp32 计算，结果转回 bf16。

| 路径 | 支持类型 | 底层函数 |
|------|---------|---------|
| libdevice | fp32 | `__hmf_tanh_fp32` |
| 标准 | fp32, fp16 | `__hmf_tanhf` / `__hmf_tanhDh` |

**使用示例**（softcap 算子）：

```python
from triton.language.extra.cann import libdevice

y = softcap * (libdevice.tanh(x.to(tl.float32) / softcap)).to(x.dtype)
```

#### libdevice.tan

```python
libdevice.tan(x, _builder=None) -> tensor
```

计算正切。

| 支持类型 | 底层函数 |
|---------|---------|
| fp32 | `__hmf_tanf` |
| fp16 | `__hmf_tanDh` |

#### libdevice.atan

```python
libdevice.atan(x, _builder=None) -> tensor
```

计算反正切。

| 支持类型 | 底层函数 |
|---------|---------|
| fp32 | `__hmf_atanf` |
| fp16 | `__hmf_atanDh` |

#### libdevice.log1p

```python
libdevice.log1p(x, _builder=None) -> tensor
```

计算 `log(1 + x)`，对 `x` 接近 0 的情况数值稳定性优于 `log(1 + x)`。

| 支持类型 | 底层函数 |
|---------|---------|
| fp32 | `__hmf_log1pf` |
| fp16 | `__hmf_log1pDh` |

#### libdevice.expm1

```python
libdevice.expm1(x, _builder=None) -> tensor
```

计算 `e^x - 1`，对 `x` 接近 0 的情况数值稳定性优于 `exp(x) - 1`。

| 路径 | 支持类型 | 底层函数 |
|------|---------|---------|
| libdevice | fp32, fp16 | `__hmf_expm1_fp32` / `__hmf_expm1_fp16` |
| 标准 | fp32 | `exp(x) - 1` |

#### libdevice.acos

```python
libdevice.acos(x, _builder=None) -> tensor
```

计算反余弦。

| 路径 | 支持类型 | 底层函数 |
|------|---------|---------|
| libdevice | fp32, fp16 | `__hmf_acos_fp32` / `__hmf_acos_fp16` |
| 标准 | fp32 | 多项式近似 + 分段计算 |

**注意**：libdevice 路径不支持 bf16，会触发 `static_assert` 失败。

#### libdevice.asin

```python
libdevice.asin(x, _builder=None) -> tensor
```

计算反正弦。标准路径通过 `π/2 - acos(x)` 实现。

| 路径 | 支持类型 | 底层函数 |
|------|---------|---------|
| libdevice | fp32, fp16 | `__hmf_asin_fp32` / `__hmf_asin_fp16` |
| 标准 | fp32 | `π/2 - acos(x)` |

#### libdevice.sinh / cosh

```python
libdevice.sinh(x, _builder=None) -> tensor
libdevice.cosh(x, _builder=None) -> tensor
```

计算双曲正弦/双曲余弦。

| 路径 | 支持类型 | 底层函数 |
|------|---------|---------|
| libdevice | fp32, fp16 | `__hmf_sinh_fp32` / `__hmf_cosh_fp32` 等 |
| 标准 | fp32 | `(exp(x) - exp(-x)) / 2` / `(exp(x) + exp(-x)) / 2` |

**注意**：libdevice 路径不支持 bf16。

#### libdevice.acosh / asinh / atanh

```python
libdevice.acosh(x, _builder=None) -> tensor
libdevice.asinh(x, _builder=None) -> tensor
libdevice.atanh(x, _builder=None) -> tensor
```

计算反双曲余弦/反双曲正弦/反双曲正切。

| 路径 | 支持类型 | 底层函数 |
|------|---------|---------|
| libdevice | fp32, fp16 | `__hmf_acosh_fp32` / `__hmf_asinh_fp32` / `__hmf_atanh_fp32` 等 |
| 标准 | fp32 | 通过 `log`、`sqrt` 等组合实现 |

**注意**：libdevice 路径不支持 bf16。

#### libdevice.erfinv

```python
libdevice.erfinv(x, _builder=None) -> tensor
```

计算逆误差函数。仅支持 fp32。

| 路径 | 支持类型 | 底层函数 |
|------|---------|---------|
| libdevice | fp32 | `__hmf_erfinv_fp32` |
| 标准 | fp32 | 有理函数近似 + Newton 迭代 |

**精度注意**：当输入 `x` 非常接近 ±1 时（`1 - |x| < 1.1e-4`），结果趋向 ±∞，数值不稳定。

#### libdevice.gamma / lgamma

```python
libdevice.gamma(x, _builder=None) -> tensor
libdevice.lgamma(x, _builder=None) -> tensor
```

计算 Gamma 函数及其对数。仅支持 fp32。

| 路径 | 支持类型 | 底层函数 |
|------|---------|---------|
| libdevice | fp32 | `__hmf_lgamma_fp32`（lgamma） |
| 标准 | fp32 | Lanczos 近似（gamma）/ `log(abs(gamma(x)))`（lgamma） |

**精度注意**：当输入接近负整数时，Gamma 函数数值不稳定。

#### libdevice.cyl_bessel_i0

```python
libdevice.cyl_bessel_i0(x, _builder=None) -> tensor
```

计算第一类修正 Bessel 函数（0 阶）。仅支持 fp32。

| 路径 | 支持类型 | 底层函数 |
|------|---------|---------|
| libdevice | fp32 | `__hmf_cyl_bessel_i0_fp32` |
| 标准 | fp32 | Chebyshev 多项式近似 |

**注意**：libdevice 路径下 fp16 输入会触发 `static_assert` 失败（源码中的错误信息误写为 "bf16"）。

#### libdevice.log10

```python
libdevice.log10(x, _builder=None) -> tensor
```

计算以 10 为底的对数。

| 路径 | 支持类型 | 底层函数 |
|------|---------|---------|
| libdevice | fp32 | `__hmf_log10_fp32` |
| 标准 | fp32 | `log(x) / log(10)` |

#### libdevice.trunc

```python
libdevice.trunc(x, _builder=None) -> tensor
```

向零截断取整。

| 路径 | 支持类型 | 底层函数 |
|------|---------|---------|
| libdevice | fp32, fp16 | `__hmf_trunc_fp32` / `__hmf_trunc_fp16` |
| 标准 | fp32 | `floor(x)` if `x >= 0` else `ceil(x)` |

#### libdevice.round

```python
libdevice.round(x, _builder=None) -> tensor
```

四舍五入取整。仅支持 fp32，底层函数 `__hmf_roundf`。

#### libdevice.nearbyint

```python
libdevice.nearbyint(x, _builder=None) -> tensor
```

按当前舍入模式取整（银行家舍入）。

| 路径 | 支持类型 | 底层函数 |
|------|---------|---------|
| libdevice | fp32 | `__hmf_nearbyint_fp32` |
| 标准 | fp32 | 银行家舍入逻辑 |

#### libdevice.rint

```python
libdevice.rint(x, _builder=None) -> tensor
```

按当前舍入模式取整。在 Ascend910_9589 上使用 HMF 硬件实现，其他平台使用软件实现。

| 平台 | 支持类型 | 底层函数 |
|------|---------|---------|
| Ascend910_9589 | fp32, fp16, bf16 | `__hmf_rint` |
| 其他 | fp32, fp16, bf16 | 银行家舍入逻辑 |

#### libdevice.signbit

```python
libdevice.signbit(x, _builder=None) -> tensor
```

返回符号位。返回 int32 类型（1 表示负数，0 表示非负）。

| 路径 | 支持类型 | 底层函数 |
|------|---------|---------|
| libdevice | fp32, fp16 | `__hmf_signbit_fp32` / `__hmf_signbit_fp16` |
| 标准 | fp32, fp16, bf16 | bitcast + 移位 |

#### libdevice.reciprocal

```python
libdevice.reciprocal(x, _builder=None) -> tensor
```

计算倒数 `1/x`。

| 支持类型 | 底层函数 |
|---------|---------|
| fp32 | `__hmf_recipf` |
| fp16 | `__hmf_recipDh` |

#### libdevice.relu

```python
libdevice.relu(x, _builder=None) -> tensor
```

计算 ReLU `max(0, x)`。

| 支持类型 | 底层函数 |
|---------|---------|
| fp32 | `__hmf_reluf` |
| fp16 | `__hmf_reluDh` |

#### libdevice.ilogb / ldexp

```python
libdevice.ilogb(x, _builder=None) -> tensor
libdevice.ldexp(x, exp, _builder=None) -> tensor
```

- `ilogb(x)`：提取浮点数的指数部分（以 `FLT_RADIX` 为底）
- `ldexp(x, exp)`：计算 `x * 2^exp`

| 函数 | 支持类型 | 底层函数 |
|------|---------|---------|
| `ilogb` | fp32, fp16 | `__hmf_ilogbf` / `__hmf_ilogbDh` |
| `ldexp` | fp32+int32, fp16+int32 | `__hmf_ldexpf` / `__hmf_ldexpDh` |

### 二元数学函数

#### libdevice.atan2

```python
libdevice.atan2(y, x, _builder=None) -> tensor
```

计算双参数反正切 `atan2(y, x)`，返回值范围 `[-π, π]`。

| 路径 | 支持类型 | 底层函数 | 说明 |
|------|---------|---------|------|
| libdevice | fp32, fp16 | `__hmf_atan2_fp32` / `__hmf_atan2_fp16` | 不支持 bf16（static_assert 失败） |
| 标准 | fp16, fp32, bf16 | 通过 `atan(y/x)` + 象限修正实现 | 由 `extension.math_ops.atan2` 覆盖，支持 bf16 |

**注意**：当 `TRITON_ENABLE_LIBDEVICE=False`（默认）时，`libdevice.atan2` 会被 `extension.math_ops.atan2` 覆盖，因此实际上支持 bf16。

#### libdevice.hypot

```python
libdevice.hypot(x, y, _builder=None) -> tensor
```

计算欧几里得距离 `sqrt(x^2 + y^2)`。

| 路径 | 支持类型 | 底层函数 |
|------|---------|---------|
| libdevice | fp32, fp16 | `__hmf_hypot_fp32` / `__hmf_hypot_fp16` |
| 标准 | fp32 | `sqrt(x*x + y*y)` |

**注意**：libdevice 路径不支持 bf16。

#### libdevice.fmod

```python
libdevice.fmod(x, y, _builder=None) -> tensor
```

计算浮点取余 `x - trunc(x/y) * y`。仅支持 fp32，底层函数 `__hmf_fmod_fp32`。

#### libdevice.copysign

```python
libdevice.copysign(x, y, _builder=None) -> tensor
```

构造一个值，大小为 `|x|`，符号为 `y` 的符号。

| 路径 | 支持类型 | 底层函数 |
|------|---------|---------|
| libdevice | fp32 | `__hmf_copysign_fp32` |
| 标准 | fp32 | `abs(x)` + 符号判断 |

#### libdevice.nextafter

```python
libdevice.nextafter(x, y, _builder=None) -> tensor
```

返回从 `x` 向 `y` 方向的下一个可表示浮点数。

| 路径 | 支持类型 | 底层函数 |
|------|---------|---------|
| libdevice | fp32, fp16 | `__hmf_nextafter_fp32` / `__hmf_nextafter_fp16` |
| 标准 | fp16, fp32 | bitcast + 整数增减 |

### 特殊运算

#### libdevice.div_rz

```python
libdevice.div_rz(x, y, _builder=None) -> tensor
```

向零截断除法。仅支持 fp32，底层函数 `__hmf_div_rz_fp32`。

#### libdevice.fast_dividef

```python
libdevice.fast_dividef(x, y, _builder=None) -> tensor
```

快速浮点除法，不保证 IEEE 精度。内部使用 `semantic.fdiv(x, y, False)`。

#### libdevice.fast_expf

```python
libdevice.fast_expf(x, _builder=None) -> tensor
```

快速指数运算，内部使用 `_builder.create_exp`。

#### libdevice.float_as_int

```python
libdevice.float_as_int(x, _builder=None) -> tensor
```

将 fp32 的位模式重新解释为 int32。仅支持 fp32，底层函数 `__hmf_float_as_int_fp32`。

### 判断函数

#### libdevice.isnan / isinf

```python
libdevice.isnan(x, _builder=None) -> tensor  # 返回 int1
libdevice.isinf(x, _builder=None) -> tensor  # 返回 int1
```

判断是否为 NaN / Inf。支持 fp32、fp16、bf16，返回 int1 类型。

#### libdevice.isfinited / finitef

```python
libdevice.isfinited(x, _builder=None) -> tensor  # 返回 int1
libdevice.finitef(x, _builder=None) -> tensor    # 返回 int1
```

- `isfinited`：判断是否为有限值（非 NaN 且非 Inf），支持 fp32、fp16、bf16
- `finitef`：判断是否为有限值，仅支持 fp32

**注意**：`isfinited` 和 `finitef` 定义在 `extension.math_ops` 中，通过 `__init__.py` 附加到 libdevice 模块。

### 从 tl.math 重导出的函数

以下函数从标准 `tl.math` 模块重导出到 libdevice，行为与 `tl.math` 完全一致：

| 函数 | 说明 |
|------|------|
| `libdevice.exp` | 指数函数 |
| `libdevice.exp2` | 2 为底的指数函数 |
| `libdevice.log` | 自然对数 |
| `libdevice.log2` | 2 为底的对数 |
| `libdevice.cos` | 余弦 |
| `libdevice.sin` | 正弦 |
| `libdevice.sqrt` | 平方根 |
| `libdevice.sqrt_rn` | IEEE 四舍五入平方根 |
| `libdevice.rsqrt` | 平方根倒数 |
| `libdevice.div_rn` | IEEE 四舍五入除法 |
| `libdevice.erf` | 误差函数 |
| `libdevice.floor` | 向下取整 |
| `libdevice.ceil` | 向上取整 |
| `libdevice.fdiv` | 快速除法 |
| `libdevice.fma` | 融合乘加 |
| `libdevice.abs` | 绝对值 |
| `libdevice.umulhi` | 无符号高位乘法 |

**特殊行为**：`math.tanh` 被 libdevice 中的实现覆盖（`math.tanh = libdevice.tanh`），因此 `tl.math.tanh` 实际调用的是 libdevice 的 tanh 实现。

## 代码示例

### 示例 1：RoPE 算子中使用 libdevice.pow

```python
import triton
import triton.language as tl
from triton.language.extra.cann import libdevice

@triton.jit
def rope_kernel(in_ptr, pos_ptr, cu_seqlens, out_ptr,
                head: tl.constexpr, base,
                DIM: tl.constexpr, max_seq_len,
                REVERSE: tl.constexpr, BLOCK_M: tl.constexpr):
    start_b = tl.program_id(0)
    begin = tl.load(cu_seqlens + start_b)
    len = tl.load(cu_seqlens + start_b + 1) - begin

    offset_n = tl.arange(0, DIM // 2)
    inv_freq = libdevice.pow(base, -2.0 / DIM * offset_n)

    for start_m in range(0, tl.cdiv(max_seq_len, BLOCK_M)):
        if start_m * BLOCK_M < len:
            pos = tl.load(pos_block_ptr, boundary_check=(0,))
            freqs = pos[:, None] * inv_freq[None, :]
            sin = tl.sin(freqs)
            cos = tl.cos(freqs)
            if REVERSE:
                sin = -sin
            x1 = x0 * cos[:, None, :] - y0 * sin[:, None, :]
            y1 = x0 * sin[:, None, :] + y0 * cos[:, None, :]
```

### 示例 2：Softcap 算子中使用 libdevice.tanh

```python
import triton
import triton.language as tl
from triton.language.extra.cann import libdevice

@triton.jit
def softcap_fwd_kernel(x_ptr, y_ptr, n_elements, softcap,
                       BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = softcap * (libdevice.tanh(x.to(tl.float32) / softcap)).to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)
```

### 示例 3：启用 libdevice 路径获得更高精度

```python
import os
os.environ["TRITON_ENABLE_LIBDEVICE"] = "True"

import triton
import triton.language as tl
from triton.language.extra.cann import libdevice

@triton.jit
def high_precision_pow_kernel(x_ptr, y_ptr, n, base,
                              BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    result = libdevice.pow(base, x)
    tl.store(y_ptr + offsets, result, mask=mask)
```

## NPU 适配要点

1. **libdevice.pow vs tl.pow**：在 NPU 上，`libdevice.pow` 调用 HMF 库的 `__hmf_powf`，精度高于 `tl.pow` 的多项式近似实现。RoPE 等对精度敏感的算子应优先使用 `libdevice.pow`。

2. **libdevice.tanh vs tl.tanh**：`math.tanh` 已被 libdevice.tanh 覆盖，因此 `tl.math.tanh` 实际调用的就是 libdevice 的实现。bf16 输入会自动提升为 fp32 计算。

3. **TRITON_ENABLE_LIBDEVICE 的取舍**：
   - 开启时：精度更高，但数据类型支持受限（多数函数仅 fp32），且需要链接 libdevice bitcode
   - 关闭时：支持更多数据类型（fp16/bf16/fp32），但部分函数精度略低
   - **建议**：对精度敏感的算子（如 RoPE、Softcap）开启；对精度要求不高的算子保持默认

4. **bf16 的限制**：libdevice 路径下，`acos`、`asin`、`sinh`、`cosh`、`acosh`、`asinh`、`atanh`、`atan2`、`hypot` 等函数不支持 bf16 输入，会触发编译时 `static_assert` 失败。使用 bf16 时需先手动转换为 fp32。

5. **libdevice 函数在 SIMD 模式下的表现**：libdevice 的 extern 函数在 SIMD 编译路径下会被编译器识别为 Vector 操作，自动参与 CV 融合和流水线优化。

6. **与标准 tl.math 的关系**：libdevice 不是 tl.math 的替代品，而是补充。标准数学运算（add/sub/mul/div/exp/log/sin/cos/sqrt 等）应继续使用 `tl` 原生 API；libdevice 主要提供 tl.math 中缺少的函数（pow、tanh、log1p、relu 等）或需要更高精度的场景。

## 常见问题

**Q: libdevice.pow 和 tl.pow 有什么区别？**

A: `libdevice.pow` 直接调用 HMF 库的 `__hmf_powf`，是硬件优化的高精度实现。`tl.pow` 是 Triton IR 原生操作。在 NPU 上，`libdevice.pow` 的精度通常更高，特别是对于 RoPE 等需要计算 `base^(-2i/d)` 的场景，推荐使用 `libdevice.pow`。

**Q: 什么时候应该开启 TRITON_ENABLE_LIBDEVICE？**

A: 当算子对数学函数精度要求较高时（如 RoPE 的频率计算、Softcap 的 tanh 计算），建议开启。开启后多数函数仅支持 fp32，如果算子使用 fp16/bf16，需要手动进行类型转换。

**Q: 为什么 bf16 输入会 static_assert 失败？**

A: HMF 库的 `__hmf_*` 函数在 libdevice 路径下通常仅提供 fp32 实现。对于 bf16 输入，部分函数（如 tanh）会自动提升为 fp32 计算，但其他函数（如 acos、sinh 等）未实现此转换逻辑，因此直接报错。解决方案是手动将 bf16 转为 fp32 后调用。

**Q: libdevice.tanh 和 tl.tanh 是同一个函数吗？**

A: 是的。在 `__init__.py` 中有 `math.tanh = libdevice.tanh` 的赋值，因此 `tl.math.tanh` 实际调用的就是 `libdevice.tanh`。直接使用 `libdevice.tanh` 只是显式表明意图。

**Q: libdevice 函数可以在 SIMT 模式下使用吗？**

A: 可以。libdevice 的 extern 函数在 SIMT 模式下同样可用，但性能可能不如 SIMD 模式。建议在 SIMD 模式下使用以获得最佳性能。

## 相关文档

- [02-math-ops.md](../02-Core-API/02-math-ops.md) - 标准 Triton 数学运算 API
- [01-extension-overview.md](./01-extension-overview.md) - Ascend 扩展 API 总览
- [07-compile-options.md](../04-Compilation-Pipeline/07-compile-options.md) - 编译选项（含 TRITON_ENABLE_LIBDEVICE）

## 源码参考

- [libdevice.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/libdevice.py) - libdevice 函数定义（所有 `__hmf_*` 外部函数声明）
- [__init__.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/__init__.py) - 模块初始化（libdevice 与 extension 的整合、math.tanh 覆盖）
- [math_ops.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/math_ops.py) - atan2、isfinited、finitef 定义
- [utils.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/utils.py) - `triton_enable_libdevice()` 函数定义
