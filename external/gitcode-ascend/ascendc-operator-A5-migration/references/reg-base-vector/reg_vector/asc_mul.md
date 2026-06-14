> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_mul.md

# asc_mul

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| <cann-filter npu_type="950"><term>Ascend 950PR/Ascend 950DT</term> | √ </cann-filter> |

## 功能说明

对源操作数src0和src1进行乘法运算，将结果写入目的操作数dst。计算公式如下：

$$
dst_i = src0_i * src1_i
$$

## 函数原型

```cpp
__simd_callee__ inline void asc_mul(vector_int16_t& dst, vector_int16_t src0, vector_int16_t src1, vector_bool mask)
__simd_callee__ inline void asc_mul(vector_uint16_t& dst, vector_uint16_t src0, vector_uint16_t src1, vector_bool mask)
__simd_callee__ inline void asc_mul(vector_half& dst, vector_half src0, vector_half src1, vector_bool mask)
__simd_callee__ inline void asc_mul(vector_bfloat16_t& dst, vector_bfloat16_t src0, vector_bfloat16_t src1, vector_bool mask)
__simd_callee__ inline void asc_mul(vector_int32_t& dst, vector_int32_t src0, vector_int32_t src1, vector_bool mask)
__simd_callee__ inline void asc_mul(vector_uint32_t& dst, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask)
__simd_callee__ inline void asc_mul(vector_float& dst, vector_float src0, vector_float src1, vector_bool mask)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量数据寄存器）。 |
| src0 | 输入 | 源操作数（矢量数据寄存器）。 |
| src1 | 输入 | 源操作数（矢量数据寄存器）。 |
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
vector_half src0;
vector_half src1;
vector_half dst;
vector_bool mask;
asc_mul(dst, src0, src1, mask);
```