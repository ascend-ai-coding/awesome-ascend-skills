> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_select.md

# asc_select

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| Ascend 950PR/Ascend 950DT | √ |
 
## 功能说明

根据mask的比特位值，从源操作数src0，src1中选择元素，得到目的操作数dst。选择的规则为：当mask的比特位为1时，从src0中选取对应位置的数；当mask的比特位为0时，从src1中选取对应位置的数。

计算公式如下：

$$
dst_i = (mask_i == 1) ? src0_i : src1_i
$$

## 函数原型

```cpp
__simd_callee__ inline void asc_select(vector_bool& dst, vector_bool src0, vector_bool src1, vector_bool mask)
__simd_callee__ inline void asc_select(vector_uint8_t& dst, vector_uint8_t src0, vector_uint8_t src1, vector_bool mask)
__simd_callee__ inline void asc_select(vector_int8_t& dst, vector_int8_t src0, vector_int8_t src1, vector_bool mask)
__simd_callee__ inline void asc_select(vector_fp8_e4m3fn_t& dst, vector_fp8_e4m3fn_t src0, vector_fp8_e4m3fn_t src1, vector_bool mask)
__simd_callee__ inline void asc_select(vector_hifloat8_t& dst, vector_hifloat8_t src0, vector_hifloat8_t src1, vector_bool mask)
__simd_callee__ inline void asc_select(vector_fp8_e5m2_t& dst, vector_fp8_e5m2_t src0, vector_fp8_e5m2_t src1, vector_bool mask)
__simd_callee__ inline void asc_select(vector_uint16_t& dst, vector_uint16_t src0, vector_uint16_t src1, vector_bool mask)
__simd_callee__ inline void asc_select(vector_int16_t& dst, vector_int16_t src0, vector_int16_t src1, vector_bool mask)
__simd_callee__ inline void asc_select(vector_half& dst, vector_half src0, vector_half src1, vector_bool mask)
__simd_callee__ inline void asc_select(vector_bfloat16_t& dst, vector_bfloat16_t src0, vector_bfloat16_t src1, vector_bool mask)
__simd_callee__ inline void asc_select(vector_uint32_t& dst, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask)
__simd_callee__ inline void asc_select(vector_int32_t& dst, vector_int32_t src0, vector_int32_t src1, vector_bool mask)
__simd_callee__ inline void asc_select(vector_float& dst, vector_float src0, vector_float src1, vector_bool mask)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（掩码寄存器或矢量数据寄存器）。 |
| src0 | 输入 | 源操作数0（掩码寄存器或矢量数据寄存器）。 |
| src1 | 输入 | 源操作数1（掩码寄存器或矢量数据寄存器）。 |
| mask | 输入 | 源操作数掩码（掩码寄存器）。指定选择src0或src1为有效数据。mask的比特位为1时，选取src0；mask的比特位位0时，选取src1。 |

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
asc_select(dst, src0, src1, mask);
```