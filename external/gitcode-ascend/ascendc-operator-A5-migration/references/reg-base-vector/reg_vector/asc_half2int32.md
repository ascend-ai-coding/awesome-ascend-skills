> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_half2int32.md

# asc_half2int32

## 产品支持情况

| 产品     | 是否支持 |
| ----------- | :----: |
| Ascend 950PR/Ascend 950DT | √    |

## 功能说明

将vector_half类型的源操作数以256B为单位分为两部分，读取其中一部分元素，将其转换成vector_int32_t类型写入目的操作数，并支持多种舍入模式：

- RINT舍入模式：四舍六入五成双舍入。

- ROUND舍入模式：四舍五入舍入。

- FLOOR舍入模式：向负无穷舍入。

- CEIL舍入模式：向正无穷舍入。

- TRUNC舍入模式：向零舍入。

以asc_half2int32_rd接口为例：

- asc_half2int32_rd：将源操作数的上半部分写入目的操作数。

- asc_half2int32_rd_v2：将源操作数的下半部分写入目的操作数。

## 函数原型

```cpp
// FLOOR舍入模式
__simd_callee__ inline void asc_half2int32_rd(vector_int32_t& dst, vector_half src, vector_bool mask)
__simd_callee__ inline void asc_half2int32_rd_v2(vector_int32_t& dst, vector_half src, vector_bool mask)
// RINT舍入模式
__simd_callee__ inline void asc_half2int32_rn(vector_int32_t& dst, vector_half src, vector_bool mask)
__simd_callee__ inline void asc_half2int32_rn_v2(vector_int32_t& dst, vector_half src, vector_bool mask)
// ROUND舍入模式
__simd_callee__ inline void asc_half2int32_rna(vector_int32_t& dst, vector_half src, vector_bool mask)
__simd_callee__ inline void asc_half2int32_rna_v2(vector_int32_t& dst, vector_half src, vector_bool mask)
// CEIL舍入模式
__simd_callee__ inline void asc_half2int32_ru(vector_int32_t& dst, vector_half src, vector_bool mask)
__simd_callee__ inline void asc_half2int32_ru_v2(vector_int32_t& dst, vector_half src, vector_bool mask)
// TRUNC舍入模式
__simd_callee__ inline void asc_half2int32_rz(vector_int32_t& dst, vector_half src, vector_bool mask)
__simd_callee__ inline void asc_half2int32_rz_v2(vector_int32_t& dst, vector_half src, vector_bool mask)
```

## 参数说明

| 参数名       | 输入/输出 | 描述               |
| --------- | ----- | ---------------- |
| dst       | 输出    | 目的操作数（矢量数据寄存器）。            |
| src | 输入    | 源操作数（矢量数据寄存器）。            |
| mask     | 输入    | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。        |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义.md](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 使能饱和模式和非饱和模式生效时，需配置ctrl寄存器，ctrl寄存器的详细说明请参见[asc_set_ctrl.md](../../sys_var/asc_set_ctrl.md)。

## 调用示例

```cpp
vector_int32_t dst;
vector_half src;
vector_bool mask;
asc_half2int32_rd(dst, src, mask);    // FLOOR舍入模式，将src的上半部分转换成vector_int32_t类型并写入dst
asc_half2int32_rd_v2(dst, src, mask);    // FLOOR舍入模式，将src的下半部分转换成vector_int32_t类型并写入dst
```