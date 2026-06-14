> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_duplicate_scalar.md

# asc_duplicate_scalar

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

根据mask将value填充到目的操作数dst。

计算公式如下：

$$
dst_i = value
$$

## 函数原型

```cpp
__simd_callee__ inline void asc_duplicate_scalar(vector_uint8_t& dst, uint8_t value, vector_bool mask)
__simd_callee__ inline void asc_duplicate_scalar(vector_int8_t& dst, int8_t value, vector_bool mask)
__simd_callee__ inline void asc_duplicate_scalar(vector_fp8_e4m3fn_t& dst, fp8_e4m3fn_t value, vector_bool mask)
__simd_callee__ inline void asc_duplicate_scalar(vector_fp8_e5m2_t& dst, fp8_e5m2_t value, vector_bool mask)
__simd_callee__ inline void asc_duplicate_scalar(vector_uint16_t& dst, uint16_t value, vector_bool mask)
__simd_callee__ inline void asc_duplicate_scalar(vector_int16_t& dst, int16_t value, vector_bool mask)
__simd_callee__ inline void asc_duplicate_scalar(vector_half& dst, half value, vector_bool mask)
__simd_callee__ inline void asc_duplicate_scalar(vector_bfloat16_t& dst, bfloat16_t value, vector_bool mask)
__simd_callee__ inline void asc_duplicate_scalar(vector_uint32_t& dst, uint32_t value, vector_bool mask)
__simd_callee__ inline void asc_duplicate_scalar(vector_int32_t& dst, int32_t value, vector_bool mask)
__simd_callee__ inline void asc_duplicate_scalar(vector_float& dst, float value, vector_bool mask)
__simd_callee__ inline void asc_duplicate_scalar(vector_uint8_t& dst, uint8_t value)
__simd_callee__ inline void asc_duplicate_scalar(vector_int8_t& dst, int8_t value)
__simd_callee__ inline void asc_duplicate_scalar(vector_uint16_t& dst, uint16_t value)
__simd_callee__ inline void asc_duplicate_scalar(vector_int16_t& dst, int16_t value)
__simd_callee__ inline void asc_duplicate_scalar(vector_uint32_t& dst, uint32_t value)
__simd_callee__ inline void asc_duplicate_scalar(vector_int32_t& dst, int32_t value)
__simd_callee__ inline void asc_duplicate_scalar(vector_half& dst, half value)
__simd_callee__ inline void asc_duplicate_scalar(vector_float& dst, float value)
__simd_callee__ inline void asc_duplicate_scalar(vector_bfloat16_t& dst, bfloat16_t value)
__simd_callee__ inline void asc_duplicate_scalar(vector_fp8_e4m3fn_t& dst, fp8_e4m3fn_t value)
__simd_callee__ inline void asc_duplicate_scalar(vector_fp8_e5m2_t& dst, fp8_e5m2_t value)
__simd_callee__ inline void asc_duplicate_scalar(vector_fp8_e8m0_t& dst, fp8_e8m0_t value)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量数据寄存器）。 |
| value | 输入 | 源操作数（标量）。 |
| mask | 输入 | 源操作数掩码（掩码寄存器）。用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义.md](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
vector_half dst;
half value = 1.0;
vector_bool mask = asc_create_mask_b16(PAT_ALL);
asc_duplicate_scalar(dst, value, mask);
```