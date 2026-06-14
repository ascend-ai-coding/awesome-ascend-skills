> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_ne_scalar.md
# asc_ne_scalar

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

src中的每个元素逐个与标量value比较大小，如果某个位置上的元素不等于value，则输出结果dst对应比特位为1，否则为0。

## 函数原型

- 高维切分计算

    ```cpp
    __aicore__ inline void asc_ne_scalar(__ubuf__ uint8_t *dst, __ubuf__ half *src, half value, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    __aicore__ inline void asc_ne_scalar(__ubuf__ uint8_t *dst, __ubuf__ float *src, float value, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    ```

- 同步计算

    ```cpp
    __aicore__ inline void asc_ne_scalar_sync(__ubuf__ uint8_t *dst, __ubuf__ half *src, half value, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    __aicore__ inline void asc_ne_scalar_sync(__ubuf__ uint8_t *dst, __ubuf__ float *src, float value, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
    ```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src | 输入 | 源操作数（矢量）的起始地址。 |
| value | 输入 | 源操作数（标量）。 |
| repeat | 输入    | 迭代次数。 |
| dst_block_stride | 输入    | 目的操作数单次迭代内不同DataBlock间地址步长。 |
| src_block_stride | 输入    | 源操作数单次迭代内不同DataBlock间地址步长。 |
| dst_repeat_stride | 输入    | 目的操作数相邻迭代间相同DataBlock的地址步长。 |
| src_repeat_stride | 输入    | 源操作数相邻迭代间相同DataBlock的地址步长。 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。
- dst、src的起始地址需要32字节对齐。
- dst按照小端顺序排序成二进制结果，对应src中相应位置的数据比较结果。
- 当结果输出到寄存器中时，固定处理256B的数据，repeat字段不生效。
- 当结果输出到目的地址中时，mask参数不生效。

## 调用示例

```cpp
// 结果输出到目标地址中，固定为128个元素
constexpr uint32_t total_length = 128;
__ubuf__ uint8_t dst[total_length / 8];
__ubuf__ half src[total_length];
half scalar = 20;
uint8_t repeat = 1;
uint8_t dst_block_stride = 1;
uint8_t src_block_stride = 1;
uint8_t dst_repeat_stride = 8;
uint8_t src_repeat_stride = 8;
…… // 数据搬运及同步操作
asc_ne_scalar(dst, src, scalar, repeat, dst_block_stride, src_block_stride, dst_repeat_stride, src_repeat_stride);
…… // 同步操作
```