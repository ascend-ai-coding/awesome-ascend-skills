> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_load/asc_gather_datablock.md

# asc_gather_datablock

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

给定源操作数在UB中的基地址和索引，根据索引位置将源操作数按DataBlock收集到目的操作数中。每个DataBlock长度为32B。
## 函数原型

- 支持掩码控制，允许选择性收集数据

    ```cpp
    __simd_callee__ inline void asc_gather_datablock(vector_int8_t& dst, __ubuf__ int8_t* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather_datablock(vector_uint8_t& dst, __ubuf__ uint8_t* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather_datablock(vector_int16_t& dst, __ubuf__ int16_t* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather_datablock(vector_uint16_t& dst, __ubuf__ uint16_t* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather_datablock(vector_int32_t& dst, __ubuf__ int32_t* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather_datablock(vector_uint32_t& dst, __ubuf__ uint32_t* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather_datablock(vector_int64_t& dst, __ubuf__ int64_t* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather_datablock(vector_uint64_t& dst, __ubuf__ uint64_t* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather_datablock(vector_half& dst, __ubuf__ half* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather_datablock(vector_float& dst, __ubuf__ float* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather_datablock(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather_datablock(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather_datablock(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather_datablock(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather_datablock(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather_datablock(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather_datablock(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather_datablock(vector_int4x2_t& dst, __ubuf__ int4b_t* src, vector_uint32_t index, vector_bool mask)
    ```
- 无掩码控制，默认收集所有索引对应的数据块
    ```cpp
    __simd_callee__ inline void asc_gather_datablock(vector_int8_t& dst, __ubuf__ int8_t* src, vector_uint32_t index)
    __simd_callee__ inline void asc_gather_datablock(vector_uint8_t& dst, __ubuf__ uint8_t* src, vector_uint32_t index)
    __simd_callee__ inline void asc_gather_datablock(vector_int16_t& dst, __ubuf__ int16_t* src, vector_uint32_t index)
    __simd_callee__ inline void asc_gather_datablock(vector_uint16_t& dst, __ubuf__ uint16_t* src, vector_uint32_t index)
    __simd_callee__ inline void asc_gather_datablock(vector_int32_t& dst, __ubuf__ int32_t* src, vector_uint32_t index)
    __simd_callee__ inline void asc_gather_datablock(vector_uint32_t& dst, __ubuf__ uint32_t* src, vector_uint32_t index)
    __simd_callee__ inline void asc_gather_datablock(vector_int64_t& dst, __ubuf__ int64_t* src, vector_uint32_t index)
    __simd_callee__ inline void asc_gather_datablock(vector_half& dst, __ubuf__ half* src, vector_uint32_t index)
    __simd_callee__ inline void asc_gather_datablock(vector_float& dst, __ubuf__ float* src, vector_uint32_t index)
    __simd_callee__ inline void asc_gather_datablock(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, vector_uint32_t index)
    __simd_callee__ inline void asc_gather_datablock(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, vector_uint32_t index)
    __simd_callee__ inline void asc_gather_datablock(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, vector_uint32_t index)
    __simd_callee__ inline void asc_gather_datablock(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, vector_uint32_t index)
    __simd_callee__ inline void asc_gather_datablock(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, vector_uint32_t index)
    __simd_callee__ inline void asc_gather_datablock(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, vector_uint32_t index)
    __simd_callee__ inline void asc_gather_datablock(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, vector_uint32_t index)
    __simd_callee__ inline void asc_gather_datablock(vector_int4x2_t& dst, __ubuf__ int4b_t* src, vector_uint32_t index)
    ```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量数据寄存器）。 |
| src | 输入 | 源操作数（矢量）的起始地址。 |
| index | 输入 | dst中每个DataBlock在UB中相对于src的索引位置。索引位置要大于等于0且32B对齐。 |
| mask | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 源操作数在UB中的起始地址需要32B对齐。
- 索引位置要大于等于0且32B对齐。
- 索引可以存在相同的值，即可以多次读取源操作数中同一个DataBlock的数据。

## 调用示例

- 支持掩码控制，允许选择性收集数据
    ```cpp
    vector_int8_t dst;
    __ubuf__ int8_t src = (__ubuf__ int8_t*)asc_get_phy_buf_addr(0);
    vector_uint32_t index;
    vector_bool mask = asc_create_mask_b8(PAT_ALL);
    asc_loadalign(index, index_addr); // index_addr是外部输入的UB内存空间地址。
    asc_gather_datablock(dst, src, index, mask);
    ```
- 无掩码控制，默认收集所有索引对应的数据块
    ```cpp
    vector_int8_t dst;
    __ubuf__ int8_t src = (__ubuf__ int8_t*)asc_get_phy_buf_addr(0);
    vector_uint32_t index;
    asc_loadalign(index, index_addr); // index_addr是外部输入的UB内存空间地址。
    asc_gather_datablock(dst, src, index);
    ```