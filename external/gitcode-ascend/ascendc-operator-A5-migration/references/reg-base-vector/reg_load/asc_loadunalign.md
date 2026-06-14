> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_load/asc_loadunalign.md

# asc_loadunalign

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

reg计算数据搬运接口，适用于从UB非32B对齐的起始地址连续搬入矢量数据寄存器的场景。

该接口不改变源操作数在UB上的地址，每次循环的调用都需要用户手动更新源操作数在UB上的地址。

## 函数原型

```cpp
__simd_callee__ inline void asc_loadunalign(vector_int8_t& dst, vector_load_unalign& src0, __ubuf__ int8_t *src1)
__simd_callee__ inline void asc_loadunalign(vector_uint8_t& dst, vector_load_unalign& src0, __ubuf__ uint8_t *src1)
__simd_callee__ inline void asc_loadunalign(vector_int16_t& dst, vector_load_unalign& src0, __ubuf__ int16_t *src1)
__simd_callee__ inline void asc_loadunalign(vector_uint16_t& dst, vector_load_unalign& src0, __ubuf__ uint16_t *src1)
__simd_callee__ inline void asc_loadunalign(vector_int32_t& dst, vector_load_unalign& src0, __ubuf__ int32_t *src1)
__simd_callee__ inline void asc_loadunalign(vector_uint32_t& dst, vector_load_unalign& src0, __ubuf__ uint32_t *src1)
__simd_callee__ inline void asc_loadunalign(vector_half& dst, vector_load_unalign& src0, __ubuf__ half *src1)
__simd_callee__ inline void asc_loadunalign(vector_float& dst, vector_load_unalign& src0, __ubuf__ float *src1)
__simd_callee__ inline void asc_loadunalign(vector_int64_t& dst, vector_load_unalign& src0, __ubuf__ int64_t *src1)
__simd_callee__ inline void asc_loadunalign(vector_bfloat16_t& dst, vector_load_unalign& src0, __ubuf__ bfloat16_t *src1)
__simd_callee__ inline void asc_loadunalign(vector_fp8_e4m3fn_t& dst, vector_load_unalign& src0, __ubuf__ fp8_e4m3fn_t *src1)
__simd_callee__ inline void asc_loadunalign(vector_fp8_e5m2_t& dst, vector_load_unalign& src0, __ubuf__ fp8_e5m2_t *src1)
__simd_callee__ inline void asc_loadunalign(vector_fp8_e8m0_t& dst, vector_load_unalign& src0, __ubuf__ fp8_e8m0_t *src1)
__simd_callee__ inline void asc_loadunalign(vector_fp4x2_e2m1_t& dst, vector_load_unalign& src0, __ubuf__ fp4x2_e2m1_t *src1)
__simd_callee__ inline void asc_loadunalign(vector_fp4x2_e1m2_t& dst, vector_load_unalign& src0, __ubuf__ fp4x2_e1m2_t *src1)
__simd_callee__ inline void asc_loadunalign(vector_hifloat8_t& dst, vector_load_unalign& src0, __ubuf__ hifloat8_t *src1)
__simd_callee__ inline void asc_loadunalign(vector_int4x2_t& dst, vector_load_unalign& src0, __ubuf__ int4b_t *src1)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量数据寄存器）。 |
| src0 | 输入/输出 | 非对齐寄存器，用于保存非对齐数据，长度32B。 |
| src1 | 输入 | 源操作数（矢量）的起始地址。 |

矢量数据寄存器和非对齐寄存器的详细说明请参见[reg数据类型定义.md](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 该接口中的src1不需要32B对齐。
- 调用该接口之前，需要调用[asc_loadunalign_pre](./asc_loadunalign_pre.md)进行非对齐搬入前的初始化。

## 调用示例

```cpp
vector_load_unalign ureg;
__ubuf__ half* src = (__ubuf__ half*)asc_get_phy_buf_addr(0);
uint32_t offset = 5;
asc_loadunalign_pre(ureg, src + offset);
asc_loadunalign(dst, ureg, src + offset);
```
