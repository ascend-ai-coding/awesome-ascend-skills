> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_load/asc_gather.md

# asc_gather

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

根据索引位置index将源操作数src按元素收集到目的操作数dst中。

![asc_gather](../../figures/asc_gather.png)

## 函数原型

- 收集UB中的元素

    ```cpp
    __simd_callee__ inline void asc_gather(vector_int16_t& dst, __ubuf__ int8_t* src, vector_uint16_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather(vector_uint16_t& dst, __ubuf__ uint8_t* src, vector_uint16_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather(vector_int16_t& dst, __ubuf__ int16_t* src, vector_uint16_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather(vector_uint16_t& dst, __ubuf__ uint16_t* src, vector_uint16_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather(vector_int32_t& dst, __ubuf__ int32_t* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather(vector_uint32_t& dst, __ubuf__ uint32_t* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather(vector_half& dst, __ubuf__ half* src, vector_uint16_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather(vector_float& dst, __ubuf__ float* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, vector_uint16_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, vector_uint16_t index,vector_bool mask)
    __simd_callee__ inline void asc_gather(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, vector_uint16_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, vector_uint16_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, vector_uint16_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather(vector_int16_t& dst, __ubuf__ int16_t* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather(vector_uint16_t& dst, __ubuf__ uint16_t* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather(vector_half& dst, __ubuf__ half* src, vector_uint32_t index, vector_bool mask)
    __simd_callee__ inline void asc_gather(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, vector_uint32_t index, vector_bool mask)
    ```

- 收集矢量数据寄存器中的元素

    ```cpp
    __simd_callee__ inline void asc_gather(vector_int8_t& dst, vector_int8_t src, vector_uint8_t index)
    __simd_callee__ inline void asc_gather(vector_uint8_t& dst, vector_uint8_t src, vector_uint8_t index)
    __simd_callee__ inline void asc_gather(vector_int16_t& dst, vector_int16_t src, vector_uint16_t index)
    __simd_callee__ inline void asc_gather(vector_uint16_t& dst, vector_uint16_t src, vector_uint16_t index)
    __simd_callee__ inline void asc_gather(vector_int32_t& dst, vector_int32_t src, vector_uint32_t index)
    __simd_callee__ inline void asc_gather(vector_uint32_t& dst, vector_uint32_t src, vector_uint32_t index)
    __simd_callee__ inline void asc_gather(vector_half& dst, vector_half src, vector_uint16_t index)
    __simd_callee__ inline void asc_gather(vector_bfloat16_t& dst, vector_bfloat16_t src, vector_uint16_t index)
    __simd_callee__ inline void asc_gather(vector_fp8_e4m3fn_t& dst, vector_fp8_e4m3fn_t src, vector_uint8_t index)
    __simd_callee__ inline void asc_gather(vector_fp8_e5m2_t& dst, vector_fp8_e5m2_t src, vector_uint8_t index)
    __simd_callee__ inline void asc_gather(vector_fp8_e8m0_t& dst, vector_fp8_e8m0_t src, vector_uint8_t index)
    __simd_callee__ inline void asc_gather(vector_hifloat8_t& dst, vector_hifloat8_t src, vector_uint8_t index)
    ```

## 参数说明

- 收集UB中的元素

    | 参数名  | 输入/输出 | 描述 |
    | :----- | :------- | :------- |
    | dst | 输出 | 目的操作数（矢量数据寄存器）。 |
    | src | 输入 | 源操作数（矢量）的起始地址。 |
    | index | 输入 | 数据索引（矢量数据寄存器），dst中每个元素在UB中相对于src的索引位置。 |
    | mask | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

- 收集矢量数据寄存器中的元素

    | 参数名  | 输入/输出 | 描述 |
    | :----- | :------- | :------- |
    | dst | 输出 | 目的操作数（矢量数据寄存器）。 |
    | src | 输入 | 源操作数（矢量数据寄存器）。 |
    | index | 输入 | 数据索引（矢量数据寄存器）。 |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义.md](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 收集UB中的元素

    当src的数据类型为b8数据类型时，dst为b16数据类型，这种情况下目的操作数的低8位与源操作数相同，高8位自动补0。例如src为int8_t数据类型：

    40 = 0b00101000 -> 0b0000000000101000, 扩充至16位后等于40；

    -40 = 0b11011000 -> 0b0000000011011000，扩充至16位后等于216。

- 收集矢量数据寄存器中的元素

    src为矢量数据寄存器类型，位宽是固定的VL, 存储的元素个数固定。如果index中索引值超出当前矢量数据寄存器中能存储的最大元素个数时，按照如下方式处理：设定当前矢量数据寄存器所能存储的最大数据元素个数为vl_length, index中索引值为i，索引值更新为i % vl_length。

## 调用示例

- 收集UB中的元素

    ```cpp
    vector_half dst;
    __ubuf__ half* src = (__ubuf__ half*)asc_get_phy_buf_addr(0);
    vector_uint16_t index;
    vector_bool mask = asc_create_mask_b16(PAT_ALL);
    asc_loadalign(index, index_addr); // index_addr是外部输入的UB内存空间地址。
    asc_gather(dst, src, index, mask);
    ```

- 收集矢量数据寄存器中的元素

    ```cpp
    vector_half dst;
    vector_half src;
    asc_loadalign(src, src_addr); // src_addr是外部输入的UB内存空间地址。
    asc_loadalign(index, index_addr); // index_addr是外部输入的UB内存空间地址。
    asc_gather(dst, src, index);
    ```
