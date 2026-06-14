> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_shiftleft.md

# asc_shiftleft

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

### 功能说明

将所有元素左移distance位。计算公式如下：
$$
dst_i = src_i << distance
$$

### 函数原型

- 前n个数据计算

    ```cpp
    __aicore__ inline void asc_shiftleft(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, uint32_t distance, uint32_t count)
    __aicore__ inline void asc_shiftleft(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, uint32_t distance, uint32_t count)
    __aicore__ inline void asc_shiftleft(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, uint32_t distance, uint32_t count)
    __aicore__ inline void asc_shiftleft(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src, uint32_t distance, uint32_t count)
    ```

- 高维切分计算

    ```cpp
    __aicore__ inline void asc_shiftleft(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, uint32_t distance, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    __aicore__ inline void asc_shiftleft(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, uint32_t distance, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    __aicore__ inline void asc_shiftleft(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, uint32_t distance, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    __aicore__ inline void asc_shiftleft(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src, uint32_t distance, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    ```

- 同步计算

    ```cpp
    __aicore__ inline void asc_shiftleft_sync(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, uint32_t distance, uint32_t count)
    __aicore__ inline void asc_shiftleft_sync(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, uint32_t distance, uint32_t count)
    __aicore__ inline void asc_shiftleft_sync(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, uint32_t distance, uint32_t count)
    __aicore__ inline void asc_shiftleft_sync(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src, uint32_t distance, uint32_t count)
    ```

### 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src | 输入 | 源操作数（矢量）的起始地址。 |
| distance | 输入 | 源操作数（标量），表示左移的位数。 |
| count | 输入 | 参与计算的元素个数。 |
| repeat | 输入 | 迭代次数。 |
| dst_block_stride | 输入 | 目的操作数单次迭代内不同DataBlock间地址步长。 |
| src_block_stride | 输入 | 源操作数单次迭代内不同DataBlock间地址步长。 |
| dst_repeat_stride | 输入 | 目的操作数相邻迭代间相同DataBlock的地址步长。 |
| src_repeat_stride | 输入 | 源操作数相邻迭代间相同DataBlock的地址步长。 |

### 返回值说明

无

### 流水类型

PIPE_V

### 约束说明

- distance不支持设置为负数。
- distance取值不能超过src数据类型的位宽。
- dst、src的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。

### 调用示例

```cpp
constexpr uint64_t total_length = 128;
uint32_t distance = 1;
__ubuf__ int16_t src[total_length];
__ubuf__ int16_t dst[total_length];
asc_shiftleft(dst, src, distance, total_length);
```