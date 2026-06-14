> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_repeat_reduce_sum.md
# asc_repeat_reduce_sum

## 产品支持情况

| 产品            | 是否支持 |
| :-----------------------   | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>   |    √    |

## 功能说明

对每个Repeat内所有元素求和。

源操作数相加采用二叉树的方式，两两相加。以256个half类型数据求和为例，每个Repeat可以计算128个half类型数据，分成2个Repeat计算；每个Repeat内，通过二叉树的方式，两两相加。

需要注意的是，两两相加计算过程中，计算结果大于65504时结果保存为65504。例如，源操作数为[60000,60000,-30000,100]，首先60000+60000溢出，结果为65504，然后计算-30000+100=-29900，最后计算65504-29900=35604。

## 函数原型

- 前n个数据计算

    ```cpp
    __aicore__ inline void asc_repeat_reduce_sum(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count)
    __aicore__ inline void asc_repeat_reduce_sum(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
    ```

- 高维切分计算

    ```cpp
    __aicore__ inline void asc_repeat_reduce_sum(__ubuf__ half* dst, __ubuf__ half* src, uint8_t repeat, uint16_t dst_repeat_stride, uint16_t src_block_stride, uint16_t src_repeat_stride)
    __aicore__ inline void asc_repeat_reduce_sum(__ubuf__ float* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_repeat_stride, uint16_t src_block_stride, uint16_t src_repeat_stride)
    ```

- 同步计算

    ```cpp
    __aicore__ inline void asc_repeat_reduce_sum_sync(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count)
    __aicore__ inline void asc_repeat_reduce_sum_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
    ```

## 参数说明

表1 参数说明

| 参数名 | 输入/输出 | 描述 |
|----|-----|-----|
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src  | 输入 | 源操作数（矢量）的起始地址。|
| count | 输入 | 参与计算的元素个数。 |
| repeat | 输入 | 迭代次数。|
| dst_repeat_stride | 输入 | 目的操作数相邻迭代间的地址步长，单位为一个repeat计算后的长度，具体如下：<br>若输出数据类型为half，则单位为2字节。<br> 若输出数据类型为float，则单位为4字节。 |
| src_block_stride | 输入 | 源操作数单次迭代内不同DataBlock间地址步长。|
| src_repeat_stride | 输入 | 源操作数相邻迭代间相同DataBlock的地址步长。|

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。
- dst的起始地址，输入类型为half时需要2字节对齐，输入类型为float时需要4字节对齐。
- src的起始地址需要32字节对齐。


## 调用示例


```cpp
// 示例计算256个half的最大值和索引，分2次迭代完成，计算结果为2组求和的值
constexpr uint32_t total_length = 256;
__ubuf__ half src[total_length];
constexpr uint32_t dst_length = 2;
__ubuf__ half dst[dst_length];

uint8_t repeat = 2;
// 单位为2字节，连续存放
uint16_t dst_repeat_stride = 1;
uint16_t src_block_stride = 1;
uint16_t src_repeat_stride = 8;
asc_repeat_reduce_sum(dst, src, repeat, dst_repeat_stride, src_block_stride, src_repeat_stride);
```

结果示例：

```
输入数据src：[1, 1, 1, ..., 1]
输出数据dst：[128, 128]
```