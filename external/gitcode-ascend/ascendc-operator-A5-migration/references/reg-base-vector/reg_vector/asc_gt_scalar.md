> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_gt_scalar.md

# asc_gt_scalar

## 产品支持情况

| 产品                  | 是否支持  |
|:-------------------------| :------: |
| Ascend 950PR/Ascend 950DT | √    |

## 功能说明

对源操作数执行逐元素比较。对于src0_i > value，若条件成立则目的操作数对应结果位为1，否则为0，每个元素的比较结果占一个bit。

## 函数原型

```c++
__simd_callee__ inline void asc_gt_scalar(vector_bool& dst, vector_uint8_t src, uint8_t value, vector_bool mask)
__simd_callee__ inline void asc_gt_scalar(vector_bool& dst, vector_int8_t src, int8_t value, vector_bool mask)
__simd_callee__ inline void asc_gt_scalar(vector_bool& dst, vector_bfloat16_t src, bfloat16_t value, vector_bool mask)
__simd_callee__ inline void asc_gt_scalar(vector_bool& dst, vector_uint16_t src, uint16_t value, vector_bool mask)
__simd_callee__ inline void asc_gt_scalar(vector_bool& dst, vector_int16_t src, int16_t value, vector_bool mask)
__simd_callee__ inline void asc_gt_scalar(vector_bool& dst, vector_uint32_t src, uint32_t value, vector_bool mask)
__simd_callee__ inline void asc_gt_scalar(vector_bool& dst, vector_int32_t src, int32_t value, vector_bool mask)
__simd_callee__ inline void asc_gt_scalar(vector_bool& dst, vector_half src, half value, vector_bool mask)
__simd_callee__ inline void asc_gt_scalar(vector_bool& dst, vector_float src, float value, vector_bool mask)
```

## 参数说明

| 参数名   | 输入/输出 | 描述                             |
|:------| :--- |:-------------------------------|
| dst   | 输出 | 目的操作数（掩码寄存器）。                 |
| src   | 输入 | 源操作数（矢量数据寄存器）。                  |
| value | 输入 | 源操作数（标量）的起始地址。                  |
| mask  | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
vector_bool dst;
vector_half src;
half value;
vector_bool mask = asc_create_mask_b16(PAT_ALL);
asc_loadalign(src, src_addr); // src_addr是外部输入的UB内存空间地址。
asc_gt_scalar(dst, src, value, mask);
```