> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_shiftright.md

# asc_shiftright

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

对src中的每个元素执行右移，右移大小为value。

- 对于int16_t/int32_t，做算术右移（去掉最低位，最高位复制符号位）。
- 对于uint16_t/uint32_t，做逻辑右移（去掉最低位，最高位填充为0）。

## 函数原型

- 前n个数据计算

    ```cpp
    __aicore__ inline void asc_shiftright(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int32_t value, uint32_t count)
    __aicore__ inline void asc_shiftright(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, uint32_t value, uint32_t count)
    __aicore__ inline void asc_shiftright(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t value, uint32_t count)
    __aicore__ inline void asc_shiftright(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src, uint32_t value, uint32_t count)
    __aicore__ inline void asc_shiftright_round(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int32_t value, uint32_t count)
    __aicore__ inline void asc_shiftright_round(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t value, uint32_t count)
    ```

- 高维切分计算

    ```cpp
    __aicore__ inline void asc_shiftright(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int32_t value, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    __aicore__ inline void asc_shiftright(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, uint32_t value, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    __aicore__ inline void asc_shiftright(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t value, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    __aicore__ inline void asc_shiftright(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src, uint32_t value, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    __aicore__ inline void asc_shiftright_round(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int32_t value, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    __aicore__ inline void asc_shiftright_round(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t value, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    ```

- 同步计算

    ```cpp
    __aicore__ inline void asc_shiftright_sync(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int32_t value, uint32_t count)
    __aicore__ inline void asc_shiftright_sync(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, uint32_t value, uint32_t count)
    __aicore__ inline void asc_shiftright_sync(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t value, uint32_t count)
    __aicore__ inline void asc_shiftright_sync(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src, uint32_t value, uint32_t count)
    __aicore__ inline void asc_shiftright_round_sync(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int32_t value, uint32_t count)
    __aicore__ inline void asc_shiftright_round_sync(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t value, uint32_t count)
    ```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src | 输入 | 源操作数（矢量）的起始地址。 |
| value | 输入 | 源操作数（标量），表示右移的位数。 |
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

- value不支持设置为负数。
- value取值不能超过src数据类型的位宽。
- dst、src的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。

## 调用示例

```cpp
constexpr uint64_t total_length = 128;    // total_length指参与计算的数据长度
int16_t value = 1;
__ubuf__ int16_t src[total_length];
__ubuf__ int16_t dst[total_length];
asc_shiftright(dst, src, value, total_length);    // 对src中的每个元素做算术右移一位，并存放到dst中
// src[0] = 0b1010, dst[0] = 0b0101

asc_shiftright_round(dst, src, value, total_length);    // 对src中的每个元素做算术右移一位并进行舍入，将结果存放到dst中
// src[0] = 0b1010, dst[0] = 0b0101  被舍入最后一位为0，不进1
// src[0] = 0b1001, dst[0] = 0b0101  被舍入最后一位为1，进1
```