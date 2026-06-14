> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_store/asc_storeunalign_postupdate.md

# asc_storeunalign_postupdate

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

reg计算数据搬运接口，适用于从矢量数据寄存器连续非32B对齐的起始地址连续搬出到UB的场景。

该接口每调用一次接口会更新目的操作数在UB上的地址。

## 函数原型

- 不指定存储偏移量

    ```cpp
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ int8_t* dst, vector_store_unalign& src0, vector_int8_t src1)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ uint8_t* dst, vector_store_unalign& src0, vector_uint8_t src1)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ int16_t* dst, vector_store_unalign& src0, vector_int16_t src1)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ uint16_t* dst, vector_store_unalign& src0, vector_uint16_t src1)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ int32_t* dst, vector_store_unalign& src0, vector_int32_t src1)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ uint32_t* dst, vector_store_unalign& src0, vector_uint32_t src1)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ int64_t* dst, vector_store_unalign& src0, vector_int64_t src1)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ half* dst, vector_store_unalign& src0, vector_half src1)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ float* dst, vector_store_unalign& src0, vector_float src1)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ bfloat16_t* dst, vector_store_unalign& src0, vector_bfloat16_t src1)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ fp8_e4m3fn_t* dst, vector_store_unalign& src0, vector_fp8_e4m3fn_t src1)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ fp8_e5m2_t* dst, vector_store_unalign& src0, vector_fp8_e5m2_t src1)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ fp8_e8m0_t* dst, vector_store_unalign& src0, vector_fp8_e8m0_t src1)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ fp4x2_e2m1_t* dst, vector_store_unalign& src0, vector_fp4x2_e2m1_t src1)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ fp4x2_e1m2_t* dst, vector_store_unalign& src0, vector_fp4x2_e1m2_t src1)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ int4b_t* dst, vector_store_unalign& src0, vector_int4x2_t src1)
    ```

- 使用uint32_t作为存储偏移量

    ```cpp
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ int8_t*& dst, vector_store_unalign& src0, vector_int8_t src1, uint32_t count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ uint8_t*& dst, vector_store_unalign& src0, vector_uint8_t src1, uint32_t count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ int16_t*& dst, vector_store_unalign& src0, vector_int16_t src1, uint32_t count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ uint16_t*& dst, vector_store_unalign& src0, vector_uint16_t src1, uint32_t count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ int32_t*& dst, vector_store_unalign& src0, vector_int32_t src1, uint32_t count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ uint32_t*& dst, vector_store_unalign& src0, vector_uint32_t src1, uint32_t count) 
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ int64_t*& dst, vector_store_unalign& src0, vector_int64_t src1, uint32_t count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ half*& dst, vector_store_unalign& src0, vector_half src1, uint32_t count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ float*& dst, vector_store_unalign& src0, vector_float src1, uint32_t count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ bfloat16_t*& dst, vector_store_unalign& src0, vector_bfloat16_t src1, uint32_t count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ fp8_e4m3fn_t*& dst, vector_store_unalign& src0, vector_fp8_e4m3fn_t src1, uint32_t count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ hifloat8_t*& dst, vector_store_unalign& src0, vector_hifloat8_t src1, uint32_t count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ fp8_e5m2_t*& dst, vector_store_unalign& src0, vector_fp8_e5m2_t src1, uint32_t count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ fp8_e8m0_t*& dst, vector_store_unalign& src0, vector_fp8_e8m0_t src1, uint32_t count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ fp4x2_e2m1_t*& dst, vector_store_unalign& src0, vector_fp4x2_e2m1_t src1, uint32_t count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ fp4x2_e1m2_t*& dst, vector_store_unalign& src0, vector_fp4x2_e1m2_t src1, uint32_t count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ int4b_t*& dst, vector_store_unalign& src0, vector_int4x2_t src1, uint32_t count)
    ```

- 使用iter_reg作为存储偏移量

    ```cpp
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ int8_t* dst, vector_store_unalign& src0, vector_int8_t src1, iter_reg& count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ uint8_t* dst, vector_store_unalign& src0, vector_uint8_t src1, iter_reg& count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ int16_t* dst, vector_store_unalign& src0, vector_int16_t src1, iter_reg& count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ uint16_t* dst, vector_store_unalign& src0, vector_uint16_t src1, iter_reg& count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ int32_t* dst, vector_store_unalign& src0, vector_int32_t src1, iter_reg& count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ uint32_t* dst, vector_store_unalign& src0, vector_uint32_t src1, iter_reg& count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ int64_t* dst, vector_store_unalign& src0, vector_int64_t src1, iter_reg& count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ half* dst, vector_store_unalign& src0, vector_half src1, iter_reg& count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ float* dst, vector_store_unalign& src0, vector_float src1, iter_reg& count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ bfloat16_t* dst, vector_store_unalign& src0, vector_bfloat16_t src1, iter_reg& count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ fp8_e4m3fn_t* dst, vector_store_unalign& src0, vector_fp8_e4m3fn_t src1, iter_reg& count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ hifloat8_t* dst, vector_store_unalign& src0, vector_hifloat8_t src1, iter_reg& count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ fp8_e5m2_t* dst, vector_store_unalign& src0, vector_fp8_e5m2_t src1, iter_reg& count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ fp8_e8m0_t* dst, vector_store_unalign& src0, vector_fp8_e8m0_t src1, iter_reg& count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ fp4x2_e2m1_t* dst, vector_store_unalign& src0, vector_fp4x2_e2m1_t src1, iter_reg& count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ fp4x2_e1m2_t* dst, vector_store_unalign& src0, vector_fp4x2_e1m2_t src1, iter_reg& count)
    __simd_callee__ inline void asc_storeunalign_postupdate(__ubuf__ int4b_t* dst, vector_store_unalign& src0, vector_int4x2_t src1, iter_reg& count)
    ```

## 参数说明

- 不指定存储偏移量

    | 参数名  | 输入/输出 | 描述 |
    | :----- | :------- | :------- |
    | dst | 输出 | 目的操作数（矢量）的起始地址。 |
    | src0 | 输入/输出 | 非对齐寄存器，用于保存非对齐数据，长度32B。 |
    | src1 | 输入 | 源操作数（矢量数据寄存器）。 |

- 使用uint32_t作为存储偏移量

    | 参数名  | 输入/输出 | 描述 |
    | :----- | :------- | :------- |
    | dst | 输出 | 目的操作数（矢量）的起始地址。 |
    | src0 | 输入/输出 | 非对齐寄存器，用于保存非对齐数据，长度32B。 |
    | src1 | 输入 | 源操作数（矢量数据寄存器）。 |
    | count | 输入 | 存储搬运的个数。 |

- 使用iter_reg作为存储偏移量

    | 参数名  | 输入/输出 | 描述 |
    | :----- | :------- | :------- |
    | dst | 输出 | 目的操作数（矢量）的起始地址。 |
    | src0 | 输入/输出 | 非对齐寄存器，用于保存非对齐数据，长度32B。 |
    | src1 | 输入 | 源操作数（矢量数据寄存器）。 |
    | count | 输入 | 存储搬运的个数（地址寄存器）。 |

矢量数据寄存器、非对齐寄存器和地址寄存器的详细说明请参见[reg数据类型定义.md](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 该接口中的dst不需要32B对齐。
- 使用uint32_t作为存储偏移量时，本接口应与[asc_storeunalign_post_postupdate](./asc_storeunalign_post_postupdate.md)接口配合使用。asc_storeunalign_post_postupdate接口用于处理该场景下非对齐搬出的尾块。
- 使用iter_reg作为存储偏移量时，本接口应与[asc_storeunalign_post](./asc_storeunalign_post.md)接口配合使用。asc_storeunalign_post接口用于处理该场景下非对齐搬出的尾块。

## 调用示例

- 不指定存储偏移量

    ```cpp
    // dst地址为8，非32B对齐。范围为[8:520]。
    __ubuf__ uint32_t* dst = (__ubuf__ uint32_t*)asc_get_phy_buf_addr(8);
    vector_store_unalign ureg;
    vector_uint32_t src;
    asc_storeunalign_postupdate(dst, ureg, src);
    asc_storeunalign_post(dst, ureg);
    ```

- 使用uint32_t作为存储偏移量

    ```cpp
    // dst地址为8，非32B对齐。范围为[8:520]。
    __ubuf__ uint32_t* dst = (__ubuf__ uint32_t*)asc_get_phy_buf_addr(8);
    vector_store_unalign ureg;
    vector_uint32_t src;
    uint32_t count = 64;
    uint32_t repeat = 2;
    for (uint32_t i = 0; i < repeat; i++) {
        // 其他reg操作

        // 第一次：src[0:62] 写入 dst[0:62]（即UB[8:256]）；同时src[62:64] 写入 ureg[0:2]；同时dst自增为dst[64]（即UB[264]）
        // 第二次：ureg[0:2] + src[0:62] 写入 dst[-2:62]（即UB[256:512]）；同时src[62:64] 写入 ureg[0:2]；同时dst自增为dst[128]（即UB[520]）
        asc_storeunalign_postupdate(dst, ureg, src, count);
    }
    // ureg[0:2] 写入 dst[-2:0]（即UB[512:520]）；同时dst不自增
    asc_storeunalign_post_postupdate(dst, ureg, 0);
    ```

- 使用iter_reg作为存储偏移量

    ```cpp
    // dst地址为8，非32B对齐。占用的地址为[8:520]。
    __ubuf__ uint32_t* dst = (__ubuf__ uint32_t*)asc_get_phy_buf_addr(8);
    vector_store_unalign ureg;
    vector_uint32_t src;
    iter_reg count = asc_create_iter_reg_b32(64);
    uint32_t repeat = 2;
    for (uint32_t i = 0; i < repeat; i++) {
        // 其他reg操作

        // 第一次：src[0:62] 写入 dst[0:62]（即UB[8:256]）；同时src[62:64] 写入 ureg[0:2]；同时dst自增为dst[64]（即UB[264]）
        // 第二次：ureg[0:2] + src[0:62] 写入 dst[-2:62]（即UB[256:512]）；同时src[62:64] 写入 ureg[0:2]；同时dst自增为dst[128]（即UB[520]）
        asc_storeunalign_postupdate(dst, ureg, src, count);
    }
    // dst结束地址为520，但此时dst已经自增到了520，所以offset要配置为0
    iter_reg offset = asc_create_iter_reg_b32(0);
    // ureg[0:2] 写入 dst[-2:0]（即UB[512:520]）
    asc_storeunalign_post(dst, ureg, offset);
    ```
