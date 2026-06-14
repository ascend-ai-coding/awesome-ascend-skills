> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_copy.md

# asc_copy

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| <cann-filter npu_type="950"><term>Ascend 950PR/Ascend 950DT</term> | √ </cann-filter> |

## 功能说明

完成数据在寄存器内的搬运。

## 函数原型

- 由矢量数据寄存器搬运至矢量数据寄存器

    ```cpp
    __simd_callee__ inline void asc_copy(vector_int8_t& dst, vector_int8_t src, vector_bool mask)
    __simd_callee__ inline void asc_copy(vector_uint8_t& dst, vector_uint8_t src, vector_bool mask)
    __simd_callee__ inline void asc_copy(vector_int16_t& dst, vector_int16_t src, vector_bool mask)
    __simd_callee__ inline void asc_copy(vector_uint16_t& dst, vector_uint16_t src, vector_bool mask)
    __simd_callee__ inline void asc_copy(vector_half& dst, vector_half src, vector_bool mask)
    __simd_callee__ inline void asc_copy(vector_bfloat16_t& dst, vector_bfloat16_t src, vector_bool mask)
    __simd_callee__ inline void asc_copy(vector_int32_t& dst, vector_int32_t src, vector_bool mask)
    __simd_callee__ inline void asc_copy(vector_uint32_t& dst, vector_uint32_t src, vector_bool mask)
    __simd_callee__ inline void asc_copy(vector_float& dst, vector_float src, vector_bool mask)
    __simd_callee__ inline void asc_copy(vector_bool& dst, vector_bool src, vector_bool mask)
    __simd_callee__ inline void asc_copy(vector_bool& dst, vector_bool src)
    ```

- 由矢量数据寄存器搬运至掩码数据寄存器
    ```cpp
    __simd_callee__ inline void asc_copy(vector_bool& dst, vector_uint16_t src, int16_t part)
    __simd_callee__ inline void asc_copy(vector_bool& dst, vector_uint32_t src, int16_t part)
    ```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量数据寄存器）。 |
| src | 输入 | 源操作数（矢量数据寄存器）。 |
| mask | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |
| part | 输入 | 数据搬运阶段的偏移量。 |

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
vector_half src;
vector_bool mask;
asc_copy(dst, src, mask);
```