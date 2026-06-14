> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_sub.md

# asc_sub

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

根据mask对源操作数src0、src1进行按元素相减的操作，将结果写入目的操作数dst。
源操作数为uint32_t或int32_t时，提供带借位数据的接口，src0、src1相减时如果产生借位，在carry中对应位置每4bit的最低位写0，否则写1。

计算公式如下：

$$
dst_i = src0_i - src1_i
$$

carry 输出借位值示例说明：

| 数据类型  | 是否借位 | 示例说明 |
| :----- | :------- | :------- |
| uint32_t 数据类型 | 不产生借位 | src0_i = 5, src1_i = 2 <br> dst_i = src0_i - src1_i = 3 <br> carry中对应位置4bit的最低位写1：carry_i = 1 |
| uint32_t 数据类型 | 产生借位 | src0_i = 5, src1_i = 7 <br> dst_i = src0_i - src1_i = -2 <br> carry中对应位置4bit的最低位写0：carry_i = 0 |
| int32_t 数据类型 | 不产生借位 | src0_i = 5, src1_i = 2 <br> dst_i = src0_i - src1_i = 3 <br> carry中对应位置4bit的最低位写1：carry_i = 1 |
| int32_t 数据类型 | 产生借位 | src0_i = 5, src1_i = -7 <br> dst_i = src0_i - src1_i = 12 <br> carry中对应位置4bit的最低位写0：carry_i = 0 |

## 函数原型

- 不带借位数据的接口

    ```cpp
    __simd_callee__ inline void asc_sub(vector_uint8_t& dst, vector_uint8_t src0, vector_uint8_t src1, vector_bool mask)
    __simd_callee__ inline void asc_sub(vector_int8_t& dst, vector_int8_t src0, vector_int8_t src1, vector_bool mask)
    __simd_callee__ inline void asc_sub(vector_uint16_t& dst, vector_uint16_t src0, vector_uint16_t src1, vector_bool mask)
    __simd_callee__ inline void asc_sub(vector_int16_t& dst, vector_int16_t src0, vector_int16_t src1, vector_bool mask)
    __simd_callee__ inline void asc_sub(vector_half& dst, vector_half src0, vector_half src1, vector_bool mask)
    __simd_callee__ inline void asc_sub(vector_bfloat16_t& dst, vector_bfloat16_t src0, vector_bfloat16_t src1, vector_bool mask)
    __simd_callee__ inline void asc_sub(vector_uint32_t& dst, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask)
    __simd_callee__ inline void asc_sub(vector_int32_t& dst, vector_int32_t src0, vector_int32_t src1, vector_bool mask)
    __simd_callee__ inline void asc_sub(vector_float& dst, vector_float src0, vector_float src1, vector_bool mask)
    ```

- 带借位数据的接口

    ```cpp
    __simd_callee__ inline void asc_sub(vector_bool& carry, vector_uint32_t& dst, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask)
    __simd_callee__ inline void asc_sub(vector_bool& carry, vector_int32_t& dst, vector_int32_t src0, vector_int32_t src1, vector_bool mask)
    ```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| carry | 输出 | 目的借位值（掩码寄存器）。 |
| dst | 输出 | 目的操作数（矢量数据寄存器）。 |
| src0 | 输入 | 源操作数0（矢量数据寄存器）。 |
| src1 | 输入 | 源操作数1（矢量数据寄存器）。 |
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
vector_half src0;
vector_half src1;
vector_bool mask = asc_create_mask_b16(PAT_ALL);
asc_loadalign(src0, src0_addr); // src0_addr是外部输入的UB内存空间地址。
asc_loadalign(src1, src1_addr); // src1_addr是外部输入的UB内存空间地址。
// mask为源操作数掩码
asc_sub(dst, src0, src1, mask);
```