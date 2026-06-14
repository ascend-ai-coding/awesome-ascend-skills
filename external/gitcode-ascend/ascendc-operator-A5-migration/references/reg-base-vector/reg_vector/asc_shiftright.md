> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_shiftright.md

# asc_shiftright

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| <cann-filter npu_type="950"><term>Ascend 950PR/Ascend 950DT</term>  | √ </cann-filter>|

## 功能说明

根据掩码mask对输入数据src0，按照src1对应元素进行右移操作，完成后将结果写入dst中。

## 函数原型

```cpp
__simd_callee__ inline void asc_shiftright(vector_uint8_t& dst, vector_uint8_t src0, vector_int8_t src1, vector_bool mask)
__simd_callee__ inline void asc_shiftright(vector_int8_t& dst, vector_int8_t src0, vector_int8_t src1, vector_bool mask)
__simd_callee__ inline void asc_shiftright(vector_uint16_t& dst, vector_uint16_t src0, vector_int16_t src1, vector_bool mask)
__simd_callee__ inline void asc_shiftright(vector_int16_t& dst, vector_int16_t src0, vector_int16_t src1, vector_bool mask)
__simd_callee__ inline void asc_shiftright(vector_uint32_t& dst, vector_uint32_t src0, vector_int32_t src1, vector_bool mask)
__simd_callee__ inline void asc_shiftright(vector_int32_t& dst, vector_int32_t src0, vector_int32_t src1, vector_bool mask)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量数据寄存器）。|
| src0 | 输入 | 源操作数（矢量数据寄存器）。|
| src1 | 输入 | 源操作数（矢量数据寄存器）。|
| mask | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义.md](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 数据类型为无符号类型：执行逻辑右移。逻辑右移会将二进制数整体向右移动指定的位数，最低位被丢弃，最高位用0填充。
- 对于逻辑移位（无符号数据类型），如果位移量大于等于数据类型位宽，则输出为0。
- 数据类型为有符号类型：执行算术右移。算术右移会将二进制数整体向右移动指定的位数，最低位被丢弃，最高位用原符号位填充。
- 对于算术移位（有符号数据类型），如果src0小于0，src1大于0，且src1的位移量大于数据类型位宽，则输出为-1。
- 对于算术移位（有符号数据类型），如果src0大于0，且src1的位移量大于数据类型位宽，则输出为0。

## 调用示例

```cpp
vector_uint32_t dst;
vector_uint32_t src0;
vector_int32_t src1;
vector_bool mask = asc_create_mask_b32(PAT_ALL);
asc_loadalign(src0, src0_addr); // src0_addr是外部输入的UB内存空间地址。
asc_loadalign(src1, src1_addr); // src1_addr是外部输入的UB内存空间地址。
asc_shiftright(dst, src0, src1, mask);
```