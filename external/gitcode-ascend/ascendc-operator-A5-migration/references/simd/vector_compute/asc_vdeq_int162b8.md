> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_vdeq_int162b8.md

# asc_vdeq_int162b8_h

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

将int16_t类型转化为int8_t或uint8_t类型，并将数据存放在每个DataBlock的上半块或下半块。使用该接口前需要调用asc_set_deqscale设置量化参数。

- asc_vdeq_int162b8_h：将数据存放在每个DataBlock的上半块。
- asc_vdeq_int162b8_l：将数据存放在每个DataBlock的下半块。
计算公式如下:

$$
dst_i = (src_i*scale_j)+offset_j
$$

## 函数原型

- 前n个数据计算

    ```cpp
   __aicore__ inline void asc_vdeq_int162b8_h(__ubuf__ int8_t* dst, __ubuf__ int16_t* src, uint32_t count)
   __aicore__ inline void asc_vdeq_int162b8_h(__ubuf__ uint8_t* dst, __ubuf__ int16_t* src, uint32_t count)
   __aicore__ inline void asc_vdeq_int162b8_l(__ubuf__ int8_t* dst, __ubuf__ int16_t* src, uint32_t count)
   __aicore__ inline void asc_vdeq_int162b8_l(__ubuf__ uint8_t* dst, __ubuf__ int16_t* src, uint32_t count)
    ```

- 高维切分计算

    ```cpp
    __aicore__ inline void asc_vdeq_int162b8_h(__ubuf__ int8_t* dst, __ubuf__ int16_t* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint8_t dst_repeat_stride, uint8_t src_repeat_stride)
    __aicore__ inline void asc_vdeq_int162b8_h(__ubuf__ uint8_t* dst, __ubuf__ int16_t* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint8_t dst_repeat_stride, uint8_t src_repeat_stride)
    __aicore__ inline void asc_vdeq_int162b8_l(__ubuf__ int8_t* dst, __ubuf__ int16_t* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint8_t dst_repeat_stride, uint8_t src_repeat_stride)
    __aicore__ inline void asc_vdeq_int162b8_l(__ubuf__ uint8_t* dst, __ubuf__ int16_t* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint8_t dst_repeat_stride, uint8_t src_repeat_stride)
    ```

- 同步计算

    ```cpp
    __aicore__ inline void asc_vdeq_int162b8_h_sync(__ubuf__ int8_t* dst, __ubuf__ int16_t* src, uint32_t count)
    __aicore__ inline void asc_vdeq_int162b8_h_sync(__ubuf__ uint8_t* dst, __ubuf__ int16_t* src, uint32_t count)
    __aicore__ inline void asc_vdeq_int162b8_l_sync(__ubuf__ int8_t* dst, __ubuf__ int16_t* src, uint32_t count)
    __aicore__ inline void asc_vdeq_int162b8_l_sync(__ubuf__ uint8_t* dst, __ubuf__ int16_t* src, uint32_t count)
    ```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
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

- dst、src的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。

## 调用示例

```cpp
// total_length指参与计算的数据长度
constexpr uint32_t total_length = 32;
__ubuf__ int8_t dst[total_length];
__ubuf__ int16_t src[total_length];
asc_vdeq_int162b8_h(dst, src, total_length);
asc_vdeq_int162b8_l(dst, src, total_length);
```