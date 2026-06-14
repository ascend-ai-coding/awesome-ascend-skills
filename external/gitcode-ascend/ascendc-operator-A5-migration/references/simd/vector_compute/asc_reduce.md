> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_reduce.md
# asc_reduce

## 产品支持情况

| 产品 | 是否支持  |
| :----------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

以内置固定模式对应的二进制或者用户自定义输入的数值对应的gather mask（数据收集的掩码），从源操作数中选取元素写入目的操作数中。

## 函数原型

- 前n个数据计算

    ```c++
    __aicore__ inline void asc_reduce(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src0, __ubuf__ uint16_t* src1, uint32_t count)
    __aicore__ inline void asc_reduce(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src0, __ubuf__ uint32_t* src1, uint32_t count)
    ```

- 高维切分计算

  ```c++
  __aicore__ inline void asc_reduce(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src0, __ubuf__ uint16_t* src1,
    uint16_t repeat, uint8_t src0_block_stride, uint8_t pattern_mode, uint16_t src0_repeat_stride, uint8_t src1_repeat_stride)
  __aicore__ inline void asc_reduce(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src0, __ubuf__ uint32_t* src1,
    uint16_t repeat, uint8_t src0_block_stride, uint8_t pattern_mode, uint16_t src0_repeat_stride, uint8_t src1_repeat_stride)
  ```

- 同步计算

  ```c++
  __aicore__ inline void asc_reduce_sync(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src0, __ubuf__ uint16_t* src1, uint32_t count)
  __aicore__ inline void asc_reduce_sync(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src0, __ubuf__ uint32_t* src1, uint32_t count)
  ```

## 参数说明

表1 参数说明

| 参数名                | 输入/输出 | 描述                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|:-------------------| :-----|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dst                | 输出 | 目的操作数（矢量）的起始地址。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| src0               | 输入 | 源操作数（矢量）的起始地址。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| src1               | 输入 | 在用户自定义模式下，由用户自定义的数据收集的掩码。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| count              | 输入 | 参与计算的元素个数。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| repeat             | 输入 | 迭代次数。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| src0_block_stride  | 输入 | 源操作数0单次迭代内不同DataBlock间地址步长。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| pattern_mode       | 输入 | gather mask（数据收集的掩码），分为内置固定模式和用户自定义模式两种，根据内置固定模式对应的二进制或者用户自定义输入的数值对应的二进制从源操作数中选取元素写入目的操作数中。<br/>&bull; 内置固定模式：pattern_mode数据类型为uint8_t，取值范围为[1, 7]，所有迭代使用相同的gather mask。不支持配置src1。<br/>&nbsp;&nbsp;&bull; 1：01010101...0101 # 每个repeat取偶数索引元素<br/>&nbsp;&nbsp;&bull; 2：10101010...1010 # 每个repeat取奇数索引元素<br/>&nbsp;&nbsp;&bull; 3：00010001...0001 # 每个repeat内每四个元素取第一个元素<br/>&nbsp;&nbsp;&bull; 4：00100010...0010 # 每个repeat每四个元素取第二个元素<br/>&nbsp;&nbsp;&bull; 5：01000100...0100 # 每个repeat内每四个元素取第三个元素<br/>&nbsp;&nbsp;&bull; 6：10001000...1000 # 每个repeat内每四个元素区第四个元素<br/>&nbsp;&nbsp;&bull; 7：11111111...1111 # 每个repeat内取全部元素<br/>&bull;用户自定义模式：pattern_mode取值为0时为用户自定义模式，用户通过src1参数输入自定义的gather mask，迭代间隔由src1_repeat_stride决定。 |
| src0_repeat_stride | 输入 | 源操作数0相邻迭代间相同DataBlock的地址步长。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| src1_repeat_stride | 输入 | 源操作数1相邻迭代间相同DataBlock的地址步长。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 接口不支持通过[asc_set_vector_mask](asc_set_vector_mask.md)配置掩码。
- dst、src的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。

## 调用示例

```cpp
//total_length指参与计算的数据总长度
constexpr uint64_t total_length = 128;
__ubuf__ uint_16 src0[total_length];
__ubuf__ uint_16 src1[total_length/16];
__ubuf__ uint_16 dst[total_length];
asc_reduce(dst, src0, src1, total_length);
```
