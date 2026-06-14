> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_load/asc_loadalign/asc_loadalign_v5.md

# asc_loadalign

## 产品支持情况

| 产品     | 是否支持 |
| ----------- | :----: |
| Ascend 950PR/Ascend 950DT | √    |

## 功能说明

对齐数据搬运接口，从UB连续对齐搬入目的操作数，支持多种搬入模式，接口使用repeat stride模式。

## 函数原型

```cpp
// repeat stride模式搬出
__simd_callee__ inline void asc_loadalign(vector_int8_t& dst, __ubuf__ int8_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign(vector_uint8_t& dst, __ubuf__ uint8_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign(vector_int16_t& dst, __ubuf__ int16_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign(vector_uint16_t& dst, __ubuf__ uint16_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign(vector_int32_t& dst, __ubuf__ int32_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign(vector_uint32_t& dst, __ubuf__ uint32_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign(vector_int64_t& dst, __ubuf__ int64_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign(vector_half& dst, __ubuf__ half* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign(vector_float& dst, __ubuf__ float* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign(vector_int4x2_t& dst, __ubuf__ int4b_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
// repeat stride模式搬出，同时开启硬件自动Post Update模式
__simd_callee__ inline void asc_loadalign_postupdate(vector_int8_t& dst, __ubuf__ int8_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign_postupdate(vector_uint8_t& dst, __ubuf__ uint8_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign_postupdate(vector_int16_t& dst, __ubuf__ int16_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign_postupdate(vector_uint16_t& dst, __ubuf__ uint16_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign_postupdate(vector_int32_t& dst, __ubuf__ int32_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign_postupdate(vector_uint32_t& dst, __ubuf__ uint32_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign_postupdate(vector_int64_t& dst, __ubuf__ int64_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign_postupdate(vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign_postupdate(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign_postupdate(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign_postupdate(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign_postupdate(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign_postupdate(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign_postupdate(vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign_postupdate(vector_half& dst, __ubuf__ half*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign_postupdate(vector_float& dst, __ubuf__ float*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
__simd_callee__ inline void asc_loadalign_postupdate(vector_int4x2_t& dst, __ubuf__ int4b_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
```

## 参数说明

| 参数名       | 输入/输出 | 描述               |
| --------- | ----- | ---------------- |
| dst       | 输出    | 目的操作数（矢量数据寄存器）。            |
| src | 输入    | 源操作数（矢量）的起始地址。            |
| block_stride | 输入 | 源操作数单次迭代内不同DataBlock间地址步长。 |
| repeat_stride | 输入 | 源操作数相邻迭代间相同DataBlock的地址步长。 |
| mask | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义.md](../../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
// asc_loadalign
__ubuf__ int8_t src;
vector_int8_t dst;
uint16_t block_stride;
uint16_t repeat_stride;
vector_bool mask = asc_create_mask_b8(PAT_ALL);
asc_loadalign(dst, src, block_stride, repeat_stride, mask);

// asc_loadalign_postupdate
__ubuf__ int8_t src;
vector_int8_t dst;
uint16_t block_stride;
uint16_t repeat_stride;
vector_bool mask = asc_create_mask_b8(PAT_ALL);
asc_loadalign_postupdate(dst, src, block_stride, repeat_stride, mask);
```
