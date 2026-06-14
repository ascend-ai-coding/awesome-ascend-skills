> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_bfloat162e1m2x2.md

# asc_bfloat162e1m2x2

## 产品支持情况

| 产品     | 是否支持 |
| ----------- | :----: |
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

将bfloat16_t类型转化为fp4x2_e1m2_t类型，并支持多种舍入模式。

- RINT舍入模式：四舍五入成双舍入
- ROUND舍入模式：四舍五入舍入
- FLOOR舍入模式：向负无穷舍入
- CEIL舍入模式：向正无穷舍入
- TRUNC舍入模式：向零舍入

由于源操作数与目的操作数类型位宽比为4:1，写入数据时需要将一个VL大小的数据分为四部分，根据不同接口选取索引0、索引1、索引2或者索引3。

## 函数原型

```cpp
// RINT舍入模式, 数据写入索引0的位置
__simd_callee__ inline void asc_bfloat162e1m2x2_rn(vector_fp4x2_e1m2_t& dst, vector_bfloat16_t src, vector_bool mask)
// RINT舍入模式, 数据写入索引1的位置
__simd_callee__ inline void asc_bfloat162e1m2x2_rn_v2(vector_fp4x2_e1m2_t& dst, vector_bfloat16_t src, vector_bool mask)
// RINT舍入模式, 数据写入索引2的位置
__simd_callee__ inline void asc_bfloat162e1m2x2_rn_v3(vector_fp4x2_e1m2_t& dst, vector_bfloat16_t src, vector_bool mask)
// RINT舍入模式, 数据写入索引3的位置
__simd_callee__ inline void asc_bfloat162e1m2x2_rn_v4(vector_fp4x2_e1m2_t& dst, vector_bfloat16_t src, vector_bool mask)
// ROUND舍入模式, 数据写入索引0的位置
__simd_callee__ inline void asc_bfloat162e1m2x2_rna(vector_fp4x2_e1m2_t& dst, vector_bfloat16_t src, vector_bool mask)
// ROUND舍入模式, 数据写入索引1的位置
__simd_callee__ inline void asc_bfloat162e1m2x2_rna_v2(vector_fp4x2_e1m2_t& dst, vector_bfloat16_t src, vector_bool mask)
// ROUND舍入模式, 数据写入索引2的位置
__simd_callee__ inline void asc_bfloat162e1m2x2_rna_v3(vector_fp4x2_e1m2_t& dst, vector_bfloat16_t src, vector_bool mask)
// ROUND舍入模式, 数据写入索引3的位置
__simd_callee__ inline void asc_bfloat162e1m2x2_rna_v4(vector_fp4x2_e1m2_t& dst, vector_bfloat16_t src, vector_bool mask)
// FLOOR舍入模式, 数据写入索引0的位置
__simd_callee__ inline void asc_bfloat162e1m2x2_rd(vector_fp4x2_e1m2_t& dst, vector_bfloat16_t src, vector_bool mask)
// FLOOR舍入模式, 数据写入索引1的位置
__simd_callee__ inline void asc_bfloat162e1m2x2_rd_v2(vector_fp4x2_e1m2_t& dst, vector_bfloat16_t src, vector_bool mask)
// FLOOR舍入模式, 数据写入索引2的位置
__simd_callee__ inline void asc_bfloat162e1m2x2_rd_v3(vector_fp4x2_e1m2_t& dst, vector_bfloat16_t src, vector_bool mask)
// FLOOR舍入模式, 数据写入索引3的位置
__simd_callee__ inline void asc_bfloat162e1m2x2_rd_v4(vector_fp4x2_e1m2_t& dst, vector_bfloat16_t src, vector_bool mask)
// CEIL舍入模式, 数据写入索引0的位置
__simd_callee__ inline void asc_bfloat162e1m2x2_ru(vector_fp4x2_e1m2_t& dst, vector_bfloat16_t src, vector_bool mask)
// CEIL舍入模式, 数据写入索引1的位置
__simd_callee__ inline void asc_bfloat162e1m2x2_ru_v2(vector_fp4x2_e1m2_t& dst, vector_bfloat16_t src, vector_bool mask)
// CEIL舍入模式, 数据写入索引2的位置
__simd_callee__ inline void asc_bfloat162e1m2x2_ru_v3(vector_fp4x2_e1m2_t& dst, vector_bfloat16_t src, vector_bool mask)
// CEIL舍入模式, 数据写入索引3的位置
__simd_callee__ inline void asc_bfloat162e1m2x2_ru_v4(vector_fp4x2_e1m2_t& dst, vector_bfloat16_t src, vector_bool mask)
// TRUNC舍入模式, 数据写入索引0的位置
__simd_callee__ inline void asc_bfloat162e1m2x2_rz(vector_fp4x2_e1m2_t& dst, vector_bfloat16_t src, vector_bool mask)
// TRUNC舍入模式, 数据写入索引1的位置
__simd_callee__ inline void asc_bfloat162e1m2x2_rz_v2(vector_fp4x2_e1m2_t& dst, vector_bfloat16_t src, vector_bool mask)
// TRUNC舍入模式, 数据写入索引2的位置
__simd_callee__ inline void asc_bfloat162e1m2x2_rz_v3(vector_fp4x2_e1m2_t& dst, vector_bfloat16_t src, vector_bool mask)
// TRUNC舍入模式, 数据写入索引3的位置
__simd_callee__ inline void asc_bfloat162e1m2x2_rz_v4(vector_fp4x2_e1m2_t& dst, vector_bfloat16_t src, vector_bool mask)
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
vector_bfloat16_t src;
vector_fp4x2_e1m2_t dst;
vector_bool mask = asc_create_mask_b16(PAT_ALL);
asc_loadalign(src, src_addr); // src_addr是外部输入的UB内存空间地址。
asc_bfloat162e1m2x2_rn(dst, src, mask);
```
