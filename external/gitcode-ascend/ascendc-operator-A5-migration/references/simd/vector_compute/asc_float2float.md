> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_float2float.md

# asc_float2float

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

对float类型数据进行精度转换处理, 支持多种舍入模式：

- RINT舍入模式：四舍六入五成双舍入
- ROUND舍入模式：四舍五入舍入
- FLOOR舍入模式：向负无穷舍入
- CEIL舍入模式：向正无穷舍入
- TRUNC舍入模式：向零舍入

## 函数原型

- 前n个数据计算

    ```cpp
    // RINT模式
    __aicore__ inline void asc_float2float_rn(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
    // FLOOR模式
    __aicore__ inline void asc_float2float_rd(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
    // CEIL模式
    __aicore__ inline void asc_float2float_ru(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
    // ROUND模式
    __aicore__ inline void asc_float2float_rna(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
    // TRUNC模式
    __aicore__ inline void asc_float2float_rz(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
    ```

- 高维切分计算

    ```cpp
    // RINT模式
    __aicore__ inline void asc_float2float_rn(__ubuf__ float* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    // FLOOR模式
    __aicore__ inline void asc_float2float_rd(__ubuf__ float* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    // CEIL模式
    __aicore__ inline void asc_float2float_ru(__ubuf__ float* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    // ROUND模式
    __aicore__ inline void asc_float2float_rna(__ubuf__ float* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    // TRUNC模式
    __aicore__ inline void asc_float2float_rz(__ubuf__ float* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    ```

- 同步模式

    ```cpp
    // RINT模式
    __aicore__ inline void asc_float2float_rn_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
    // FLOOR模式
    __aicore__ inline void asc_float2float_rd_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
    // CEIL模式
    __aicore__ inline void asc_float2float_ru_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
    // ROUND模式
    __aicore__ inline void asc_float2float_rna_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
    // TRUNC模式
    __aicore__ inline void asc_float2float_rz_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
    ```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数地址（矢量）的起始地址。 |
| src | 输入 | 源操作数地址（矢量）的起始地址。 |
| count | 输入 | 参与计算的元素个数。 |
| repeat |输入 | 迭代次数。 |
| dst_block_stride |输入 | 目的操作数单次迭代内不同DataBlock间地址步长。 |
| src_block_stride |输入 | 源操作数单次迭代内不同DataBlock间地址步长。 |
| dst_repeat_stride |输入 | 目的操作数相邻迭代间相同DataBlock的地址步长。 |
| src_repeat_stride |输入 | 源操作数相邻迭代间相同DataBlock的地址步长。 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- dst、src的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。

## 调用示例

```cpp
constexpr uint32_t total_length = 256;
__ubuf__ float src[total_length];
__ubuf__ float dst[total_length];
asc_float2float_rn(dst, src, total_length);
```