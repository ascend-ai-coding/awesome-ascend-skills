> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_half2int8.md

# asc_half2int8

## 产品支持情况

| 产品     | 是否支持 |
| ----------- | :----: |
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

将half类型转化为int8_t类型，并支持多种舍入模式。

- RINT舍入模式：四舍六入五成双舍入
- ROUND舍入模式：四舍五入舍入
- FLOOR舍入模式：向负无穷舍入
- CEIL舍入模式：向正无穷舍入
- TRUNC舍入模式：向零舍入

- 非饱和模式：
  - 输入数据超过输出类型最值时，结果被截断为目标格式的数据宽度，例如输入half值为257，输出int8_t值为1。
  - 输入数据为+/-inf时, 返回输出类型的最值。
  - 输入数据为nan时，返回0。
- 饱和模式：
  - 输入数据超过输出类型最值时，返回输出类型的对应最值，例如输入half值为257，输出int8_t值为127。
  - 输入数据为+/-inf时, 返回输出类型的最值。
  - 输入数据为nan时，返回0。

## 函数原型

```cpp
// RINT舍入模式, 非饱和模式，数据写入索引为偶数的位置
__simd_callee__ inline void asc_half2int8_rn(vector_int8_t& dst, vector_half src, vector_bool mask)
// RINT舍入模式, 饱和模式，数据写入索引为偶数的位置
__simd_callee__ inline void asc_half2int8_rn_sat(vector_int8_t& dst, vector_half src, vector_bool mask)
// RINT舍入模式, 非饱和模式，数据写入索引为奇数的位置
__simd_callee__ inline void asc_half2int8_rn_v2(vector_int8_t& dst, vector_half src, vector_bool mask)
// RINT舍入模式, 饱和模式，数据写入索引为奇数的位置
__simd_callee__ inline void asc_half2int8_rn_sat_v2(vector_int8_t& dst, vector_half src, vector_bool mask)
// ROUND舍入模式, 非饱和模式，数据写入索引为偶数的位置
__simd_callee__ inline void asc_half2int8_rna(vector_int8_t& dst, vector_half src, vector_bool mask)
// ROUND舍入模式, 饱和模式，数据写入索引为偶数的位置
__simd_callee__ inline void asc_half2int8_rna_sat(vector_int8_t& dst, vector_half src, vector_bool mask)
// ROUND舍入模式, 非饱和模式，数据写入索引为奇数的位置
__simd_callee__ inline void asc_half2int8_rna_v2(vector_int8_t& dst, vector_half src, vector_bool mask)
// ROUND舍入模式, 饱和模式，数据写入索引为奇数的位置
__simd_callee__ inline void asc_half2int8_rna_sat_v2(vector_int8_t& dst, vector_half src, vector_bool mask)
// FLOOR舍入模式, 非饱和模式，数据写入索引为偶数的位置
__simd_callee__ inline void asc_half2int8_rd(vector_int8_t& dst, vector_half src, vector_bool mask)
// FLOOR舍入模式, 饱和模式，数据写入索引为偶数的位置
__simd_callee__ inline void asc_half2int8_rd_sat(vector_int8_t& dst, vector_half src, vector_bool mask)
// FLOOR舍入模式, 非饱和模式，数据写入索引为奇数的位置
__simd_callee__ inline void asc_half2int8_rd_v2(vector_int8_t& dst, vector_half src, vector_bool mask)
// FLOOR舍入模式, 饱和模式，数据写入索引为奇数的位置
__simd_callee__ inline void asc_half2int8_rd_sat_v2(vector_int8_t& dst, vector_half src, vector_bool mask)
// CEIL舍入模式, 非饱和模式，数据写入索引为偶数的位置
__simd_callee__ inline void asc_half2int8_ru(vector_int8_t& dst, vector_half src, vector_bool mask)
// CEIL舍入模式, 饱和模式，数据写入索引为偶数的位置
__simd_callee__ inline void asc_half2int8_ru_sat(vector_int8_t& dst, vector_half src, vector_bool mask)
// CEIL舍入模式, 非饱和模式，数据写入索引为奇数的位置
__simd_callee__ inline void asc_half2int8_ru_v2(vector_int8_t& dst, vector_half src, vector_bool mask)
// CEIL舍入模式, 饱和模式，数据写入索引为奇数的位置
__simd_callee__ inline void asc_half2int8_ru_sat_v2(vector_int8_t& dst, vector_half src, vector_bool mask)
// TRUNC舍入模式, 非饱和模式，数据写入索引为偶数的位置
__simd_callee__ inline void asc_half2int8_rz(vector_int8_t& dst, vector_half src, vector_bool mask)
// TRUNC舍入模式, 饱和模式，数据写入索引为偶数的位置
__simd_callee__ inline void asc_half2int8_rz_sat(vector_int8_t& dst, vector_half src, vector_bool mask)
// TRUNC舍入模式, 非饱和模式，数据写入索引为奇数的位置
__simd_callee__ inline void asc_half2int8_rz_v2(vector_int8_t& dst, vector_half src, vector_bool mask)
// TRUNC舍入模式, 饱和模式，数据写入索引为奇数的位置
__simd_callee__ inline void asc_half2int8_rz_sat_v2(vector_int8_t& dst, vector_half src, vector_bool mask)
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

- 使能饱和模式和非饱和模式生效时，需配置ctrl寄存器，ctrl寄存器的详细说明请参见[asc_set_ctrl.md](../../sys_var/asc_set_ctrl.md)。

## 调用示例

```cpp
vector_half src;
vector_int8_t dst;
vector_bool mask = asc_create_mask_b16(PAT_ALL);
asc_loadalign(src, src_addr); // src_addr是外部输入的UB内存空间地址。
asc_half2int8_rn(dst, src, mask);
```
