> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_uint82half.md
# asc_uint82half

## 产品支持情况

| 产品 | 是否支持  |
| :----------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

将uint8_t类型数据转换为half类型。

## 函数原型

- 前n个数据计算

    ```c++
    __aicore__ inline void asc_uint82half(__ubuf__ half* dst, __ubuf__ uint8_t* src, uint32_t count)
    ```

- 高维切分计算

    ```c++
    __aicore__ inline void asc_uint82half(__ubuf__ half* dst, __ubuf__ uint8_t* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    ```

- 同步计算

    ```c++
    __aicore__ inline void asc_uint82half_sync(__ubuf__ half* dst, __ubuf__ uint8_t* src, uint32_t count)
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

- dst、src的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。

## 调用示例

```cpp
constexpr uint64_t total_length = 128;    // total_length指参与计算的数据长度
__ubuf__ uint8_t src[total_length];
__ubuf__ half dst[total_length];
asc_uint82half(dst, src, total_length);    // 将src转换为half类型并存放到dst中
```
