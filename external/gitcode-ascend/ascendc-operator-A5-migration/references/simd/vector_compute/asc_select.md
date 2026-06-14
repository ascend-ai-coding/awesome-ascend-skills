> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_select.md
# asc_select

## 产品支持情况

| 产品 | 是否支持  |
| :----------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

执行矢量选择操作，给定两个源操作数src0和src1，根据条件选择元素，得到目的操作数dst。此接口配合[asc_set_cmp_mask(sel_mask)](asc_set_cmp_mask.md)设置，规则为：当sel_mask的比特位是1时，从src0选取；当比特位是0时，从src1选择。

## 函数原型

- 前n个数据计算

    ```c++
    __aicore__ inline void asc_select(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count)
    __aicore__ inline void asc_select(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t count)
    ```

- 高维切分计算

    ```c++
    __aicore__ inline void asc_select(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint8_t repeat, uint8_t dst_block_stride, uint8_t src0_block_stride, uint8_t src1_block_stride, uint8_t dst_repeat_stride, uint8_t src0_repeat_stride, uint8_t src1_repeat_stride)
    __aicore__ inline void asc_select(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint8_t repeat, uint8_t dst_block_stride, uint8_t src0_block_stride, uint8_t src1_block_stride, uint8_t dst_repeat_stride, uint8_t src0_repeat_stride, uint8_t src1_repeat_stride)
    ```

- 同步计算

    ```c++
    __aicore__ inline void asc_select_sync(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count)
    __aicore__ inline void asc_select_sync(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t count)
    ```

## 参数说明

|参数名|输入/输出|描述|
| :--- | :--- | :--- |
| dst  | 输出  | 目的操作数（矢量）的起始地址。 |
| src0 | 输入  | 源操作数0（矢量）的起始地址。 |
| src1 | 输入  | 源操作数1（矢量）的起始地址。 |
| count | 输入    | 参与计算的元素个数。 |
| repeat | 输入    | 迭代次数。 |
| dst_block_stride | 输入    | 目的操作数单次迭代内不同DataBlock间地址步长。 |
| src0_block_stride | 输入    | 源操作数0单次迭代内不同DataBlock间地址步长。 |
| src1_block_stride | 输入    | 源操作数1单次迭代内不同DataBlock间地址步长。 |
| dst_repeat_stride | 输入    | 目的操作数相邻迭代间相同DataBlock的地址步长。 |
| src0_repeat_stride | 输入    | 源操作数0相邻迭代间相同DataBlock的地址步长。 |
| src1_repeat_stride | 输入    | 源操作数1相邻迭代间相同DataBlock的地址步长。 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 需要配合[asc_set_cmp_mask](asc_set_cmp_mask.md)接口使用。
- dst、src0、src1的起始地址需要32字节对齐
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。

## 调用示例

```c++
constexpr uint32_t total_length = 256;
constexpr uint32_t sel_length = 8;
__ubuf__ uint8_t sel_mask[sel_length];
__ubuf__ float dst[total_length];
__ubuf__ float src0[total_length];
__ubuf__ float src1[total_length];
asc_set_cmp_mask(sel_mask);
asc_sync_pipe(PIPE_V);
asc_select(dst, src0, src1, total_length);
```
