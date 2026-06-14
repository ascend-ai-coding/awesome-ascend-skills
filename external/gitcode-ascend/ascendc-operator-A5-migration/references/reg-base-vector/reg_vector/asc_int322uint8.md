> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_int322uint8.md

# asc_int322uint8
## 产品支持情况

| 产品     | 是否支持 |
| ----------- | :----: |
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

将int32_t类型转化为uint8_t类型。

溢出模式：
- 非饱和模式：结果被截断至目标格式的数据宽度，且保留最低有效位（LSB）。
- 饱和模式： 结果被限制在目标格式的可表示范围内。

由于源操作数与目的操作数类型位宽比为4:1，写入数据时需要将一个VL大小的数据分为四部分，根据不同接口选取索引0、索引1、索引2或者索引3。

## 函数原型

```cpp
// 非饱和模式，数据写入索引0的位置
__simd_callee__ inline void asc_int322uint8(vector_uint8_t& dst, vector_int32_t src, vector_bool mask)
// 饱和模式，数据写入索引0的位置
__simd_callee__ inline void asc_int322uint8_sat(vector_uint8_t& dst, vector_int32_t src, vector_bool mask)
// 非饱和模式，数据写入索引1的位置
__simd_callee__ inline void asc_int322uint8_v2(vector_uint8_t& dst, vector_int32_t src, vector_bool mask)
// 饱和模式，数据写入索引1的位置
__simd_callee__ inline void asc_int322uint8_sat_v2(vector_uint8_t& dst, vector_int32_t src, vector_bool mask)
// 非饱和模式，数据写入索引2的位置
__simd_callee__ inline void asc_int322uint8_v3(vector_uint8_t& dst, vector_int32_t src, vector_bool mask)
// 饱和模式，数据写入索引2的位置
__simd_callee__ inline void asc_int322uint8_sat_v3(vector_uint8_t& dst, vector_int32_t src, vector_bool mask)
// 非饱和模式，数据写入索引3的位置
__simd_callee__ inline void asc_int322uint8_v4(vector_uint8_t& dst, vector_int32_t src, vector_bool mask)
// 饱和模式，数据写入索引3的位置
__simd_callee__ inline void asc_int322uint8_sat_v4(vector_uint8_t& dst, vector_int32_t src, vector_bool mask)
```

## 参数说明

| 参数名    | 输入/输出 | 描述                |
| :------ | :----- | :----------------- |
| dst    | 输出    | 目的操作数（矢量数据寄存器）。            |
| src    | 输入    | 源操作数（矢量数据寄存器）。             |
| mask | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义.md](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
vector_int32_t src;
vector_uint8_t dst;
vector_bool mask = asc_create_mask_b32(PAT_ALL);
asc_loadalign(src, src_addr); // src_addr是外部输入的UB内存空间地址。
asc_int322uint8(dst, src, mask);
```
