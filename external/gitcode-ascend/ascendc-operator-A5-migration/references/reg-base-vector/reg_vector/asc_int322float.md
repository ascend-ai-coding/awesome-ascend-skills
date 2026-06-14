> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_int322float.md

# asc_int322float

## 产品支持情况

| 产品     | 是否支持 |
| ----------- | :----: |
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

将int32_t类型转化为float类型，并支持多种舍入模式。

- RINT舍入模式：四舍六入五成双舍入
- ROUND舍入模式：四舍五入舍入
- FLOOR舍入模式：向负无穷舍入
- CEIL舍入模式：向正无穷舍入
- TRUNC舍入模式：向零舍入

## 函数原型

```cpp
// RINT舍入模式
__simd_callee__ inline void asc_int322float_rn(vector_float& dst, vector_int32_t src, vector_bool mask)
// ROUND舍入模式
__simd_callee__ inline void asc_int322float_rna(vector_float& dst, vector_int32_t src, vector_bool mask)
// FLOOR舍入模式
__simd_callee__ inline void asc_int322float_rd(vector_float& dst, vector_int32_t src, vector_bool mask)
// CEIL舍入模式
__simd_callee__ inline void asc_int322float_ru(vector_float& dst, vector_int32_t src, vector_bool mask)
// TRUNC舍入模式
__simd_callee__ inline void asc_int322float_rz(vector_float& dst, vector_int32_t src, vector_bool mask)
```

## 参数说明

| 参数名    | 输入/输出 | 描述                |
| :------ | :----- | :----------------- |
| dst    | 输出    | 目的操作数（矢量数据寄存器）。            |
| src    | 输入    | 源操作数（矢量数据寄存器）。             |
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
vector_int32_t src;
vector_float dst;
vector_bool mask = asc_create_mask_b32(PAT_ALL);
asc_loadalign(src, src_addr); // src_addr是外部输入的UB内存空间地址。
asc_int322float_rn(dst, src, mask);
```
