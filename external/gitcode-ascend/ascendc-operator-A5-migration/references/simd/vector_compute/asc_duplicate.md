> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_duplicate.md
# asc_duplicate

## 产品支持情况

| 产品 | 是否支持  |
| :-----------| :------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

执行矢量复制（Duplicate）操作，将标量值复制填充到矢量中。

## 函数原型

- 前n个数据计算

    ```cpp
    __aicore__ inline void asc_duplicate(__ubuf__ half* dst, half src, uint32_t count)
    __aicore__ inline void asc_duplicate(__ubuf__ int16_t* dst, int16_t src, uint32_t count)
    __aicore__ inline void asc_duplicate(__ubuf__ uint16_t* dst, uint16_t src, uint32_t count)
    __aicore__ inline void asc_duplicate(__ubuf__ bfloat16_t* dst, bfloat16_t src, uint32_t count)
    __aicore__ inline void asc_duplicate(__ubuf__ float* dst, float src, uint32_t count)
    __aicore__ inline void asc_duplicate(__ubuf__ int32_t* dst, int32_t src, uint32_t count)
    __aicore__ inline void asc_duplicate(__ubuf__ uint32_t* dst, uint32_t src, uint32_t count)
    ```

- 高维切分计算

    ```cpp
    __aicore__ inline void asc_duplicate(__ubuf__ half* dst, half src, uint8_t repeat, uint16_t dst_block_stride, uint16_t dst_repeat_stride)
    __aicore__ inline void asc_duplicate(__ubuf__ int16_t* dst, int16_t src, uint8_t repeat, uint16_t dst_block_stride, uint16_t dst_repeat_stride)
    __aicore__ inline void asc_duplicate(__ubuf__ uint16_t* dst, uint16_t src, uint8_t repeat, uint16_t dst_block_stride, uint16_t dst_repeat_stride)
    __aicore__ inline void asc_duplicate(__ubuf__ bfloat16_t* dst, bfloat16_t src, uint8_t repeat, uint16_t dst_block_stride, uint16_t dst_repeat_stride)
    __aicore__ inline void asc_duplicate(__ubuf__ float* dst, float src, uint8_t repeat, uint16_t dst_block_stride, uint16_t dst_repeat_stride)
    __aicore__ inline void asc_duplicate(__ubuf__ int32_t* dst, int32_t src, uint8_t repeat, uint16_t dst_block_stride, uint16_t dst_repeat_stride)
    __aicore__ inline void asc_duplicate(__ubuf__ uint32_t* dst, uint32_t src, uint8_t repeat, uint16_t dst_block_stride, uint16_t dst_repeat_stride)
    ```

- 同步计算

    ```cpp
    __aicore__ inline void asc_duplicate_sync(__ubuf__ half* dst, half src, uint32_t count)
    __aicore__ inline void asc_duplicate_sync(__ubuf__ int16_t* dst, int16_t src, uint32_t count)
    __aicore__ inline void asc_duplicate_sync(__ubuf__ uint16_t* dst, uint16_t src, uint32_t count)
    __aicore__ inline void asc_duplicate_sync(__ubuf__ bfloat16_t* dst, bfloat16_t src, uint32_t count)
    __aicore__ inline void asc_duplicate_sync(__ubuf__ float* dst, float src, uint32_t count)
    __aicore__ inline void asc_duplicate_sync(__ubuf__ int32_t* dst, int32_t src, uint32_t count)
    __aicore__ inline void asc_duplicate_sync(__ubuf__ uint32_t* dst, uint32_t src, uint32_t count)
    ```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src | 输入 | 源操作数（标量）。 |
| count | 输入 | 参与连续复制的元素个数。 |
| repeat | 输入  | 迭代次数。 |
| dst_block_stride | 输入 | 目的操作数单次迭代内不同DataBlock间地址步长。 |
| dst_repeat_stride | 输入 | 目的操作数相邻迭代间相同DataBlock的地址步长。 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- dst的起始地址需要32字节对齐。

## 调用示例

```cpp
constexpr uint32_t dst_length = 128;
__ubuf__ uint16_t dst[dst_length];
uint16_t val = 18;
asc_duplicate(dst, val, dst_length);
```

结果示例：

```
输出数据dst：[18 18 18 ... 18]
```