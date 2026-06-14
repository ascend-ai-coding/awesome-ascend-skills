> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_intlv.md

# asc_intlv

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <cann-filter npu_type="950"><term>Ascend 950PR/Ascend 950DT</term>  | √ </cann-filter>|

## 功能说明

将源操作数src0和src1中的元素交织存入目的操作数dst0和dst1中。操作数为掩码寄存器时，不同数据类型的接口决定了交织的位宽大小，例如对于b32类型，交织时以4bit为一组。

![asc_intlf示例](../../figures/asc_intlf示例.png)

## 函数原型

  ```cpp
  // 操作数为掩码寄存器
  __simd_callee__ inline void asc_intlv_b8(vector_bool& dst0, vector_bool& dst1, vector_bool src0, vector_bool src1)
  __simd_callee__ inline void asc_intlv_b16(vector_bool& dst0, vector_bool& dst1, vector_bool src0, vector_bool src1)
  __simd_callee__ inline void asc_intlv_b32(vector_bool& dst0, vector_bool& dst1, vector_bool src0, vector_bool src1)
  // 操作数为矢量数据寄存器
  __simd_callee__ inline void asc_intlv(vector_int8_t& dst0, vector_int8_t& dst1, vector_int8_t src0, vector_int8_t src1)
  __simd_callee__ inline void asc_intlv(vector_uint8_t& dst0, vector_uint8_t& dst1, vector_uint8_t src0, vector_uint8_t src1)
  __simd_callee__ inline void asc_intlv(vector_fp8_e5m2_t& dst0, vector_fp8_e5m2_t& dst1, vector_fp8_e5m2_t src0, vector_fp8_e5m2_t src1)
  __simd_callee__ inline void asc_intlv(vector_fp8_e4m3fn_t& dst0, vector_fp8_e4m3fn_t& dst1, vector_fp8_e4m3fn_t src0, vector_fp8_e4m3fn_t src1)
  __simd_callee__ inline void asc_intlv(vector_hifloat8_t& dst0, vector_hifloat8_t& dst1, vector_hifloat8_t src0, vector_hifloat8_t src1)
  __simd_callee__ inline void asc_intlv(vector_fp8_e8m0_t& dst0, vector_fp8_e8m0_t& dst1, vector_fp8_e8m0_t src0, vector_fp8_e8m0_t src1)
  __simd_callee__ inline void asc_intlv(vector_int16_t& dst0, vector_int16_t& dst1, vector_int16_t src0, vector_int16_t src1)
  __simd_callee__ inline void asc_intlv(vector_uint16_t& dst0, vector_uint16_t& dst1, vector_uint16_t src0, vector_uint16_t src1)
  __simd_callee__ inline void asc_intlv(vector_half& dst0, vector_half& dst1, vector_half src0, vector_half src1)
  __simd_callee__ inline void asc_intlv(vector_bfloat16_t& dst0, vector_bfloat16_t& dst1, vector_bfloat16_t src0, vector_bfloat16_t src1)
  __simd_callee__ inline void asc_intlv(vector_int32_t& dst0, vector_int32_t& dst1, vector_int32_t src0, vector_int32_t src1)
  __simd_callee__ inline void asc_intlv(vector_uint32_t& dst0, vector_uint32_t& dst1, vector_uint32_t src0, vector_uint32_t src1)
  __simd_callee__ inline void asc_intlv(vector_float& dst0, vector_float& dst1, vector_float src0, vector_float src1)
  ```

## 参数说明

| 参数名       | 输入/输出 | 描述                |
| --------- | ----- | ----------------- |
| dst0       | 输出    | 目的操作数（矢量数据寄存器或掩码寄存器）。 |
| dst1       | 输出    | 目的操作数（矢量数据寄存器或掩码寄存器）。 |
| src0       | 输入    | 源操作数（矢量数据寄存器）。 |
| src1       | 输入    | 源操作数（矢量数据寄存器）。 |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义.md](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
vector_int8_t dst0;
vector_int8_t dst1;
vector_int8_t src0;
vector_int8_t src1;
asc_loadalign(src0, src0_addr); // src0_addr是外部输入的UB内存空间地址。
asc_loadalign(src1, src1_addr); // src1_addr是外部输入的UB内存空间地址。
asc_intlv(dst0, dst1, src0, src1);
```
