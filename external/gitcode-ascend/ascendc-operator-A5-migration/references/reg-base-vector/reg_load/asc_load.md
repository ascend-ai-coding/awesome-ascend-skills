> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_load/asc_load.md

# asc_load

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

reg计算数据搬运接口，支持从UB非32字节对齐的源地址src搬运至矢量数据寄存器，搬运量为VL。

连续搬运时，用户需手动更新src地址。

## 函数原型

```cpp
__simd_callee__ inline void asc_load(vector_int8_t& dst, __ubuf__ int8_t* src)
__simd_callee__ inline void asc_load(vector_uint8_t& dst, __ubuf__ uint8_t* src)
__simd_callee__ inline void asc_load(vector_int16_t& dst, __ubuf__ int16_t* src)
__simd_callee__ inline void asc_load(vector_uint16_t& dst, __ubuf__ uint16_t* src)
__simd_callee__ inline void asc_load(vector_int32_t& dst, __ubuf__ int32_t* src)
__simd_callee__ inline void asc_load(vector_uint32_t& dst, __ubuf__ uint32_t* src)
__simd_callee__ inline void asc_load(vector_half& dst, __ubuf__ half* src)
__simd_callee__ inline void asc_load(vector_float& dst, __ubuf__ float* src)
__simd_callee__ inline void asc_load(vector_int64_t& dst, __ubuf__ int64_t* src)
__simd_callee__ inline void asc_load(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src)
__simd_callee__ inline void asc_load(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src)
__simd_callee__ inline void asc_load(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src)
__simd_callee__ inline void asc_load(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src)
__simd_callee__ inline void asc_load(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src)
__simd_callee__ inline void asc_load(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src)
__simd_callee__ inline void asc_load(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src)
__simd_callee__ inline void asc_load(vector_int4x2_t& dst, __ubuf__ int4b_t* src)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量数据寄存器）。 |
| src | 输入 | 源操作数（矢量）的起始地址。 |

矢量数据寄存器的详细说明请参见[reg数据类型定义.md](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
vector_half dst;
__ubuf__ half* src = (__ubuf__ half*)asc_get_phy_buf_addr(0);
asc_load(dst, src);
```
