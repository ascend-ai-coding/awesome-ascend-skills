> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_float2e4m3.md

# asc_float2e4m3

## 产品支持情况

| 产品     | 是否支持 |
| ----------- | :----: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## 功能说明

将float类型转化为fp8_e4m3fn_t类型，支持RINT舍入模式。

根据输入数据的大小分为非饱和模式和饱和模式。

- 非饱和模式：
  - 输入数据超过输出类型最值时，返回输出类型的对应符号inf值。
  - 输入数据为+/-inf时, 返回+/-inf。
  - 输入数据为nan时，返回nan。
- 饱和模式：
  - 输入数据超过输出类型最值时，返回输出类型的对应最值。
  - 输入数据为+/-inf时, 返回输出类型的对应最值。
  - 输入数据为nan时，返回0。

由于源操作数与目的操作数类型位宽比为4:1，写入数据时需要将一个VL大小的数据分为四部分，根据不同接口选取索引0、索引1、索引2或者索引3。

## 函数原型

```cpp
// RINT舍入模式，非饱和模式，数据写入索引为0的位置
__simd_callee__ inline void asc_float2e4m3_rn(vector_fp8_e4m3fn_t& dst, vector_float src, vector_bool mask)
// RINT舍入模式，饱和模式，数据写入索引为0的位置
__simd_callee__ inline void asc_float2e4m3_rn_sat(vector_fp8_e4m3fn_t& dst, vector_float src, vector_bool mask)
// RINT舍入模式，非饱和模式，数据写入索引为1的位置
__simd_callee__ inline void asc_float2e4m3_rn_v2(vector_fp8_e4m3fn_t& dst, vector_float src, vector_bool mask)
// RINT舍入模式，饱和模式，数据写入索引为1的位置
__simd_callee__ inline void asc_float2e4m3_rn_sat_v2(vector_fp8_e4m3fn_t& dst, vector_float src, vector_bool mask)
// RINT舍入模式，非饱和模式，数据写入索引为2的位置
__simd_callee__ inline void asc_float2e4m3_rn_v3(vector_fp8_e4m3fn_t& dst, vector_float src, vector_bool mask)
// RINT舍入模式，饱和模式，数据写入索引为2的位置
__simd_callee__ inline void asc_float2e4m3_rn_sat_v3(vector_fp8_e4m3fn_t& dst, vector_float src, vector_bool mask)
// RINT舍入模式，非饱和模式，数据写入索引为3的位置
__simd_callee__ inline void asc_float2e4m3_rn_v4(vector_fp8_e4m3fn_t& dst, vector_float src, vector_bool mask)
// RINT舍入模式，饱和模式，数据写入索引为3的位置
__simd_callee__ inline void asc_float2e4m3_rn_sat_v4(vector_fp8_e4m3fn_t& dst, vector_float src, vector_bool mask)

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

- 使能饱和模式和非饱和模式生效时，需配置ctrl寄存器，ctrl寄存器的详细说明请参见[asc_set_ctrl.md](../../sys_var/asc_set_ctrl.md)。

## 调用示例

```cpp
vector_float src;
vector_fp8_e4m3fn_t dst;
vector_bool mask;
asc_float2e4m3_rn(dst, src, mask);
```
