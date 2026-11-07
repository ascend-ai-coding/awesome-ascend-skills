# 内置宏参考

> 快速参考导航 - 以下是本分类的完整宏定义列表

# 内置宏

涵盖 bfloat16、half、float 三种浮点类型的特殊值、数学常数和边界值。

## bfloat16 特殊值 (asc_bf16.h)

| 宏名 | 描述 |
|------|------|
| ASCRT_INF_BF16 | 正无穷大 |
| ASCRT_MAX_NORMAL_BF16 | 最大表示值 |
| ASCRT_MIN_DENORM_BF16 | 最小表示值 |
| ASCRT_NAN_BF16 | NaN |
| ASCRT_NEG_ZERO_BF16 | 负零 |
| ASCRT_ONE_BF16 | 1.0 |
| ASCRT_ZERO_BF16 | 正零 |

## half 特殊值 (asc_fp16.h)

| 宏名 | 描述 |
|------|------|
| ASCRT_INF_FP16 | 正无穷大 |
| ASCRT_MAX_NORMAL_FP16 | 最大表示值 |
| ASCRT_MIN_DENORM_FP16 | 最小表示值 |
| ASCRT_NAN_FP16 | NaN |
| ASCRT_NEG_ZERO_FP16 | 负零 |
| ASCRT_ONE_FP16 | 1.0 |
| ASCRT_ZERO_FP16 | 正零 |

## float 特殊值 (math_constants.h)

| 宏名 | 描述 |
|------|------|
| ASCRT_INF_F | 正无穷大 |
| ASCRT_MAX_NORMAL_F | 最大表示值 |
| ASCRT_MIN_DENORM_F | 最小表示值 |
| ASCRT_NAN_F | NaN |
| ASCRT_NEG_ZERO_F | 负零 |
| ASCRT_ONE_F | 1.0 |
| ASCRT_ZERO_F | 正零 |

## 数学常数 (math_constants.h)

| 宏名 | 描述 |
|------|------|
| ASCRT_SQRT_HALF_F | √0.5 |
| ASCRT_SQRT_HALF_HI_F | √0.5 高位部分 |
| ASCRT_SQRT_HALF_LO_F | √0.5 低位部分 |
| ASCRT_SQRT_TWO_F | √2 |
| ASCRT_THIRD_F | 1/3 |
| ASCRT_PIO4_F | π/4 |
| ASCRT_PIO2_F | π/2 |
| ASCRT_PI_F | π |
| ASCRT_TWOPI_F | 2π |
| ASCRT_ONE_OVER_PI_F | 1/π |
| ASCRT_TWO_OVER_PI_F | 2/π |
| ASCRT_LN2_F | ln(2) |
| ASCRT_LN10_F | ln(10) |
| ASCRT_LOG2E_F | log2(e) |
| ASCRT_LOG10E_F | log10(e) |
| ASCRT_LN2_HI_F | ln(2) 高位部分 |
| ASCRT_LN2_LO_F | ln(2) 低位部分 |
