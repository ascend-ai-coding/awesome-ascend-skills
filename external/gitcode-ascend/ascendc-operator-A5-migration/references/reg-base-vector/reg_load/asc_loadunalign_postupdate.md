> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_load/asc_loadunalign_postupdate.md

# asc_loadunalign_postupdate

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## 功能说明

数据搬运接口，适用于从UB非32B对齐地址起始连续搬入。

- asc_loadunalign_postupdate（不带iter_reg入参）：配合asc_loadunalign（不带iter_reg入参）接口使用。
- asc_loadunalign_postupdate（带iter_reg入参）：配合asc_loadunalign（带iter_reg入参）接口使用。

## 函数原型

```cpp
__simd_callee__ inline void asc_loadunalign_postupdate(vector_int8_t& dst, vector_load_unalign& src0, __ubuf__ int8_t *src1, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_uint8_t& dst, vector_load_unalign& src0, __ubuf__ uint8_t *src1, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_fp4x2_e2m1_t& dst, vector_load_unalign& src0, __ubuf__ fp4x2_e2m1_t *src1, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_fp4x2_e1m2_t& dst, vector_load_unalign& src0, __ubuf__ fp4x2_e1m2_t *src1, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_fp8_e8m0_t& dst, vector_load_unalign& src0, __ubuf__ fp8_e8m0_t *src1, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_fp8_e5m2_t& dst, vector_load_unalign& src0, __ubuf__ fp8_e5m2_t *src1, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_fp8_e4m3fn_t& dst, vector_load_unalign& src0, __ubuf__ fp8_e4m3fn_t *src1, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_hifloat8_t& dst, vector_load_unalign& src0, __ubuf__ hifloat8_t *src1, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_half& dst, vector_load_unalign& src0, __ubuf__ half *src1, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_int16_t& dst, vector_load_unalign& src0, __ubuf__ int16_t *src1, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_uint16_t& dst, vector_load_unalign& src0, __ubuf__ uint16_t *src1, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_bfloat16_t& dst, vector_load_unalign& src0, __ubuf__ bfloat16_t *src1, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_int32_t& dst, vector_load_unalign& src0, __ubuf__ int32_t *src1, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_uint32_t& dst, vector_load_unalign& src0, __ubuf__ uint32_t *src1, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_float& dst, vector_load_unalign& src0, __ubuf__ float *src1, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_int64_t& dst, vector_load_unalign& src0, __ubuf__ int64_t *src1, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_int4x2_t& dst, vector_load_unalign& src0, __ubuf__ int4b_t *src1, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_int8_t& dst, vector_load_unalign& src0, __ubuf__ int8_t *src1, iter_reg offset, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_uint8_t& dst, vector_load_unalign& src0, __ubuf__ uint8_t *src1, iter_reg offset, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_fp4x2_e2m1_t& dst, vector_load_unalign& src0, __ubuf__ fp4x2_e2m1_t *src1, iter_reg offset, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_fp4x2_e1m2_t& dst, vector_load_unalign& src0, __ubuf__ fp4x2_e1m2_t *src1, iter_reg offset, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_fp8_e8m0_t& dst, vector_load_unalign& src0, __ubuf__ fp8_e8m0_t *src1, iter_reg offset, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_fp8_e5m2_t& dst, vector_load_unalign& src0, __ubuf__ fp8_e5m2_t *src1, iter_reg offset, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_fp8_e4m3fn_t& dst, vector_load_unalign& src0, __ubuf__ fp8_e4m3fn_t *src1, iter_reg offset, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_hifloat8_t& dst, vector_load_unalign& src0, __ubuf__ hifloat8_t *src1, iter_reg offset, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_int16_t& dst, vector_load_unalign& src0, __ubuf__ int16_t *src1, iter_reg offset, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_uint16_t& dst, vector_load_unalign& src0, __ubuf__ uint16_t *src1, iter_reg offset, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_half& dst, vector_load_unalign& src0, __ubuf__ half *src1, iter_reg offset, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_bfloat16_t& dst, vector_load_unalign& src0, __ubuf__ bfloat16_t *src1, iter_reg offset, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_int32_t& dst, vector_load_unalign& src0, __ubuf__ int32_t *src1, iter_reg offset, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_uint32_t& dst, vector_load_unalign& src0, __ubuf__ uint32_t *src1, iter_reg offset, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_float& dst, vector_load_unalign& src0, __ubuf__ float *src1, iter_reg offset, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_int64_t& dst, vector_load_unalign& src0, __ubuf__ int64_t *src1, iter_reg offset, uint32_t count)
__simd_callee__ inline void asc_loadunalign_postupdate(vector_int4x2_t& dst, vector_load_unalign& src0, __ubuf__ int4b_t *src1, iter_reg offset, uint32_t count)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量）。 |
| src0 | 输入 | 源操作数（矢量数据寄存器）。 |
| src1 | 输入 | 源操作数（矢量）的起始地址。 |
| offset | 输入 | 存储地址的偏移量。实际搬运UB起始地址为src1+offset |
| count | 输入 | 搬运完成后offset在自增前被更新为offset+count|

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义.md](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 接口中的src1不需要32B对齐。

## 调用示例

```cpp
vector_int8_t dst;
vector_load_unalign src0;
__ubuf__ int8_t* src1;
uint32_t count;
asc_loadunalign_postupdate(dst, src0, src1, count);
```