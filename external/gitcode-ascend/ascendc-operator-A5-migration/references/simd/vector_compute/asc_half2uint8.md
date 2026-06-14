> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_half2uint8.md
# asc_half2uint8

## 产品支持情况

| 产品 | 是否支持  |
| :----------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

将half类型数据转换为uint8_t类型，支持多种舍入模式：

- NONE舍入模式: 在转换有精度损失时表示RINT舍入模式，不涉及精度损失时代表不舍入
- RINT舍入模式：四舍六入五成双舍入
- ROUND舍入模式：四舍五入舍入
- FLOOR舍入模式：向负无穷舍入
- CEIL舍入模式：向正无穷舍入
- TRUNC舍入模式：向零舍入

## 函数原型

- 前n个数据计算

    ```c++
    // NONE舍入模式
    __aicore__ inline void asc_half2uint8(__ubuf__ uint8_t* dst, __ubuf__ half* src, uint32_t count)
    // RINT舍入模式
    __aicore__ inline void asc_half2uint8_rn(__ubuf__ uint8_t* dst, __ubuf__ half* src, uint32_t count)
    // FLOOR舍入模式
    __aicore__ inline void asc_half2uint8_rd(__ubuf__ uint8_t* dst, __ubuf__ half* src, uint32_t count)
    // ROUND舍入模式
    __aicore__ inline void asc_half2uint8_rna(__ubuf__ uint8_t* dst, __ubuf__ half* src, uint32_t count)
    // CEIL舍入模式
    __aicore__ inline void asc_half2uint8_ru(__ubuf__ uint8_t* dst, __ubuf__ half* src, uint32_t count)
    // TRUNC舍入模式
    __aicore__ inline void asc_half2uint8_rz(__ubuf__ uint8_t* dst, __ubuf__ half* src, uint32_t count)
    ```

- 高维切分计算

    ```c++
    // NONE舍入模式
    __aicore__ inline void asc_half2uint8(__ubuf__ uint8_t* dst, __ubuf__ half* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    // RINT舍入模式
    __aicore__ inline void asc_half2uint8_rn(__ubuf__ uint8_t* dst, __ubuf__ half* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    // FLOOR舍入模式
    __aicore__ inline void asc_half2uint8_rd(__ubuf__ uint8_t* dst, __ubuf__ half* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    // ROUND舍入模式
    __aicore__ inline void asc_half2uint8_rna(__ubuf__ uint8_t* dst, __ubuf__ half* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    // CEIL舍入模式
    __aicore__ inline void asc_half2uint8_ru(__ubuf__ uint8_t* dst, __ubuf__ half* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    // TRUNC舍入模式
    __aicore__ inline void asc_half2uint8_rz(__ubuf__ uint8_t* dst, __ubuf__ half* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    ```

- 同步计算

    ```c++
    // NONE舍入模式
    __aicore__ inline void asc_half2uint8_sync(__ubuf__ uint8_t* dst, __ubuf__ half* src, uint32_t count)
    // RINT舍入模式
    __aicore__ inline void asc_half2uint8_rn_sync(__ubuf__ uint8_t* dst, __ubuf__ half* src, uint32_t count)
    // FLOOR舍入模式
    __aicore__ inline void asc_half2uint8_rd_sync(__ubuf__ uint8_t* dst, __ubuf__ half* src, uint32_t count)
    // ROUND舍入模式
    __aicore__ inline void asc_half2uint8_rna_sync(__ubuf__ uint8_t* dst, __ubuf__ half* src, uint32_t count)
    // CEIL舍入模式
    __aicore__ inline void asc_half2uint8_ru_sync(__ubuf__ uint8_t* dst, __ubuf__ half* src, uint32_t count)
    // TRUNC舍入模式
    __aicore__ inline void asc_half2uint8_rz_sync(__ubuf__ uint8_t* dst, __ubuf__ half* src, uint32_t count)
    ```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src | 输入 | 源操作数（矢量）的起始地址。 |
| count | 输入 | 参与计算的元素个数。 |
| repeat | 输入 | 迭代次数。 |
| dst_block_stride | 输入 | 目的操作数单次迭代内不同DataBlock间地址步长。 |
| src_block_stride | 输入 | 源操作数单次迭代内不同DataBlock间地址步长。 |
| dst_repeat_stride | 输入 | 目的操作数相邻迭代间相同DataBlock的地址步长。 |
| src_repeat_stride | 输入 | 源操作数相邻迭代间相同DataBlock的地址步长。 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。
- dst、src的起始地址需要32字节对齐。

## 调用示例

```cpp
//total_length指参与转换的数据总长度
constexpr uint32_t total_length = 128;
__ubuf__ half src[total_length];
__ubuf__ uint8_t dst[total_length];
asc_half2uint8(dst, src, total_length);
```
