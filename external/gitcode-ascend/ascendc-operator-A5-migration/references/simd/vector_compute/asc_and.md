> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_and.md
# asc_and

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

执行矢量与运算。计算公式如下：

$$
dst_i = src0_i \quad \&  \quad src1_i
$$

## 函数原型

- 前n个数据计算

    ```c++
    __aicore__ inline void asc_and(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1, uint32_t count)
    __aicore__ inline void asc_and(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src0, __ubuf__ uint16_t* src1, uint32_t count)
    ```
    
- 高维切分计算

    ```cpp
    __aicore__ inline void asc_and(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1, uint8_t repeat, uint8_t dst_block_stride, uint8_t src0_block_stride, uint8_t src1_block_stride, uint8_t dst_repeat_stride, uint8_t src0_repeat_stride, uint8_t src1_repeat_stride)
    __aicore__ inline void asc_and(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src0, __ubuf__ uint16_t* src1, uint8_t repeat, uint8_t dst_block_stride, uint8_t src0_block_stride, uint8_t src1_block_stride, uint8_t dst_repeat_stride, uint8_t src0_repeat_stride, uint8_t src1_repeat_stride)
    ```

- 同步计算

    ```cpp
    __aicore__ inline void asc_and_sync(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1, uint32_t count)
    __aicore__ inline void asc_and_sync(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src0, __ubuf__ uint16_t* src1, uint32_t count)
    ```

## 参数说明

表1 参数说明

| 参数名       | 输入/输出 | 描述                |
| --------- | ----- | ----------------- |
| dst       | 输出    | 目的操作数。            |
| src0、src1 | 输入    | 源操作数（矢量）的起始地址。             |
| count     | 输入    | 参与计算的元素个数。        |
| dst_block_stride   | 输入 | 目的操作数单次迭代内不同DataBlock间地址步长。 |
| src0_block_stride  | 输入 | 源操作数0单次迭代内不同DataBlock间地址步长。 |
| src1_block_stride  | 输入 | 源操作数1单次迭代内不同DataBlock间地址步长。 |
| dst_repeat_stride  | 输入 | 目的操作数相邻迭代间相同DataBlock的地址步长。 |
| src0_repeat_stride | 输入 | 源操作数0相邻迭代间相同DataBlock的地址步长。 |
| src1_repeat_stride | 输入 | 源操作数1相邻迭代间相同DataBlock的地址步长。 |
| repeat             | 输入 | 迭代次数。 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。
- dst、src的起始地址需要32字节对齐。

## 调用示例

```c++
constexpr uint64_t total_length = 64;
//total_length 指参与计算的数据长度
__ubuf__ int16_t src0[total_length];
__ubuf__ int16_t src1[total_length];
__ubuf__ int16_t dst[total_length];
asc_and(dst, src0, src1, total_length);
```