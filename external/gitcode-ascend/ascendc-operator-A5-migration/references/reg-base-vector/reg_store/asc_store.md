> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_store/asc_store.md

# asc_store

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| <cann-filter npu_type="950"><term>Ascend 950PR/Ascend 950DT</term>  | √ </cann-filter>|

## 功能说明

reg计算数据搬运接口，适用于从矢量数据寄存器搬出到UB的场景，不区分是否对齐，在追求极致性能时，对齐场景推荐使用[asc_store_align](./asc_storealign)接口。

## 函数原型

```cpp
// 不传入count参数，默认搬运VL（Vector Length）长度的元素
__simd_callee__ inline void asc_store(__ubuf__  int8_t* dst, vector_int8_t src)
__simd_callee__ inline void asc_store(__ubuf__  uint8_t* dst, vector_uint8_t src)
__simd_callee__ inline void asc_store(__ubuf__  fp4x2_e2m1_t* dst, vector_fp4x2_e2m1_t src)
__simd_callee__ inline void asc_store(__ubuf__  fp4x2_e1m2_t* dst, vector_fp4x2_e1m2_t src)
__simd_callee__ inline void asc_store(__ubuf__  int4b_t* dst, vector_int4x2_t src)
__simd_callee__ inline void asc_store(__ubuf__  fp8_e8m0_t* dst, vector_fp8_e8m0_t src)
__simd_callee__ inline void asc_store(__ubuf__  fp8_e5m2_t* dst, vector_fp8_e5m2_t src)
__simd_callee__ inline void asc_store(__ubuf__  fp8_e4m3fn_t* dst, vector_fp8_e4m3fn_t src)
__simd_callee__ inline void asc_store(__ubuf__  hifloat8_t* dst, vector_hifloat8_t src)
__simd_callee__ inline void asc_store(__ubuf__  int16_t* dst, vector_int16_t src)
__simd_callee__ inline void asc_store(__ubuf__  uint16_t* dst, vector_uint16_t src)
__simd_callee__ inline void asc_store(__ubuf__  half* dst, vector_half src)
__simd_callee__ inline void asc_store(__ubuf__  bfloat16_t* dst, vector_bfloat16_t src)
__simd_callee__ inline void asc_store(__ubuf__  int32_t* dst, vector_int32_t src)
__simd_callee__ inline void asc_store(__ubuf__  uint32_t* dst, vector_uint32_t src)
__simd_callee__ inline void asc_store(__ubuf__  float* dst, vector_float src)
__simd_callee__ inline void asc_store(__ubuf__  int64_t* dst, vector_int64_t src)

// 传入count参数，搬运count个元素
__simd_callee__ inline void asc_store(__ubuf__  int8_t* dst, vector_int8_t src, uint32_t count)
__simd_callee__ inline void asc_store(__ubuf__  uint8_t* dst, vector_uint8_t src, uint32_t count)  
__simd_callee__ inline void asc_store(__ubuf__  int16_t* dst, vector_int16_t src, uint32_t count)
__simd_callee__ inline void asc_store(__ubuf__  uint16_t* dst, vector_uint16_t src, uint32_t count)
__simd_callee__ inline void asc_store(__ubuf__  int32_t* dst, vector_int32_t src, uint32_t count)
__simd_callee__ inline void asc_store(__ubuf__  uint32_t* dst, vector_uint32_t src, uint32_t count)
__simd_callee__ inline void asc_store(__ubuf__  float* dst, vector_float src, uint32_t count)
__simd_callee__ inline void asc_store(__ubuf__  int64_t* dst, vector_int64_t src, uint32_t count)
__simd_callee__ inline void asc_store(__ubuf__  bfloat16_t* dst, vector_bfloat16_t src, uint32_t count)
__simd_callee__ inline void asc_store(__ubuf__  fp8_e8m0_t* dst, vector_fp8_e8m0_t src, uint32_t count)
__simd_callee__ inline void asc_store(__ubuf__  fp8_e5m2_t* dst, vector_fp8_e5m2_t src, uint32_t count)
__simd_callee__ inline void asc_store(__ubuf__  fp8_e4m3fn_t* dst, vector_fp8_e4m3fn_t src, uint32_t count)
__simd_callee__ inline void asc_store(__ubuf__  hifloat8_t* dst, vector_hifloat8_t src, uint32_t count)
__simd_callee__ inline void asc_store(__ubuf__  fp4x2_e2m1_t* dst, vector_fp4x2_e2m1_t src, uint32_t count)
__simd_callee__ inline void asc_store(__ubuf__  fp4x2_e1m2_t* dst, vector_fp4x2_e1m2_t src, uint32_t count)
__simd_callee__ inline void asc_store(__ubuf__  int4b_t* dst, vector_int4x2_t src, uint32_t count)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src | 输入 | 源操作数（矢量数据寄存器）。 |
| count | 输入 | 搬运元素数量。 |

矢量数据寄存器的详细说明请参见[reg数据类型定义.md](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- count的取值范围为[1, VL/sizeof(dst)]，VL为矢量长度。超出该范围会导致未定义行为。

## 调用示例

```cpp
vector_half src;
__ubuf__ half* dst = (__ubuf__ half*)asc_get_phy_buf_addr(0);
asc_store(dst, src);
```


