> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_arange.md

# asc_arange

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <cann-filter npu_type="950"><term>Ascend 950PR/Ascend 950DT</term>  | √ </cann-filter>|

## 功能说明

以传入的value为起始值，生成递增/递减的索引，并将生成的索引保存在dst中。算法逻辑表示如下：
  ```cpp
  // 递增
  {value, value + 1, value + 2, ... value + VL_T - 2, value + VL_T - 1}
  // 递减
  {value + VL_T - 1, value + VL_T - 2, value + VL_T - 3, ... value + 1, value}
  ```

以int16_t数据类型，起始值value=10为例：
递增索引为{10, 11, 12, 13, ... 135, 136, 137}, 递减索引为{137, 136, 135, 134, ... 12, 11, 10}。

## 函数原型

  ```cpp
  // 递增模式
  __simd_callee__ inline void asc_arange(vector_int8_t& dst, int8_t value)
  __simd_callee__ inline void asc_arange(vector_int16_t& dst, int16_t value)
  __simd_callee__ inline void asc_arange(vector_half& dst, half value)
  __simd_callee__ inline void asc_arange(vector_int32_t& dst, int32_t value)
  __simd_callee__ inline void asc_arange(vector_float& dst, float value)
  // 递减模式
  __simd_callee__ inline void asc_arange_descend(vector_int8_t& dst, int8_t value)
  __simd_callee__ inline void asc_arange_descend(vector_int16_t& dst, int16_t value)
  __simd_callee__ inline void asc_arange_descend(vector_half& dst, half value)
  __simd_callee__ inline void asc_arange_descend(vector_int32_t& dst, int32_t value)
  __simd_callee__ inline void asc_arange_descend(vector_float& dst, float value)
  ```

## 参数说明

| 参数名       | 输入/输出 | 描述                |
| --------- | ----- | ----------------- |
| dst       | 输出    | 目的操作数（矢量数据寄存器）。 |
| value     | 输入    | 源操作数（标量）。 |

矢量数据寄存器的详细说明请参见[reg数据类型定义.md](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
vector_int8_t dst;
int8_t value;
asc_arange(dst, value);
```
