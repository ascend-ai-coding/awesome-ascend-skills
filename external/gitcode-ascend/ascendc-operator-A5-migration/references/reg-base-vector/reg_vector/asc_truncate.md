> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_truncate.md

# asc_truncate

## 产品支持情况

| 产品     | 是否支持 |
| ----------- | :----: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |


## 功能说明

将源操作数的浮点数元素截断到整数位，同时源操作数的数据类型保持不变，并支持多种舍入模式。

- RINT舍入模式：四舍六入五成双舍入
- ROUND舍入模式：四舍五入舍入
- FLOOR舍入模式：向负无穷舍入
- CEIL舍入模式：向正无穷舍入
- TRUNC舍入模式：向零舍入

## 函数原型

```cpp
// RINT舍入模式
__simd_callee__ inline void asc_rint(vector_half& dst, vector_half src, vector_bool mask)
// RINT舍入模式
__simd_callee__ inline void asc_rint(vector_bfloat16_t& dst, vector_bfloat16_t src, vector_bool mask)
// RINT舍入模式
__simd_callee__ inline void asc_rint(vector_float& dst, vector_float src, vector_bool mask)
// ROUND舍入模式
__simd_callee__ inline void asc_round(vector_half& dst, vector_half src, vector_bool mask)
// ROUND舍入模式
__simd_callee__ inline void asc_round(vector_bfloat16_t& dst, vector_bfloat16_t src, vector_bool mask)
// ROUND舍入模式
__simd_callee__ inline void asc_round(vector_float& dst, vector_float src, vector_bool mask)
// FLOOR舍入模式
__simd_callee__ inline void asc_floor(vector_half& dst, vector_half src, vector_bool mask)
// FLOOR舍入模式
__simd_callee__ inline void asc_floor(vector_bfloat16_t& dst, vector_bfloat16_t src, vector_bool mask)
// FLOOR舍入模式
__simd_callee__ inline void asc_floor(vector_float& dst, vector_float src, vector_bool mask)
// CEIL舍入模式
__simd_callee__ inline void asc_ceil(vector_half& dst, vector_half src, vector_bool mask)
// CEIL舍入模式
__simd_callee__ inline void asc_ceil(vector_bfloat16_t& dst, vector_bfloat16_t src, vector_bool mask)
// CEIL舍入模式
__simd_callee__ inline void asc_ceil(vector_float& dst, vector_float src, vector_bool mask)
// TRUNC舍入模式
__simd_callee__ inline void asc_trunc(vector_half& dst, vector_half src, vector_bool mask)
// TRUNC舍入模式
__simd_callee__ inline void asc_trunc(vector_bfloat16_t& dst, vector_bfloat16_t src, vector_bool mask)
// TRUNC舍入模式
__simd_callee__ inline void asc_trunc(vector_float& dst, vector_float src, vector_bool mask)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量数据寄存器）。 |
| src | 输入 | 源操作数（矢量数据寄存器）。 |
| mask | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义.md](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
vector_half src;
vector_half dst;
vector_bool mask;
asc_rint(dst, src, mask);
```
