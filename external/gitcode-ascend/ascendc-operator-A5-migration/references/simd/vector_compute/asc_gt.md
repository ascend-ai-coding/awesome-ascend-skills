> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_gt.md
# asc_gt

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

按元素比较两个矢量的大小关系，若比较后的结果为真，则输出结果的对应比特位为1，否则为0。计算公式如下：

$$
\overrightarrow{\text{result}} = \text{asc\_gt}(\overrightarrow{src0}, \overrightarrow{src1})
$$
$$
\begin{bmatrix}
\text{result}_0 \\
\text{result}_1 \\
\vdots \\
\text{result}_{n-1}
\end{bmatrix}
=
\begin{bmatrix}
\mathbb{I}_{(src0_0 > src1_0)} \\
\mathbb{I}_{(src0_1 > src1_1)} \\
\vdots \\
\mathbb{I}_{(src0_{n-1} > src1_{n-1})}
\end{bmatrix}
$$

## 函数原型


- 高维切分计算

    ```cpp
    // 结果存入寄存器
    __aicore__ inline void asc_gt(__ubuf__ half* src0, __ubuf__ half* src1, uint8_t repeat, uint8_t dst_block_stride, uint8_t src0_block_stride, uint8_t src1_block_stride, uint8_t dst_repeat_stride, uint8_t src0_repeat_stride, uint8_t src1_repeat_stride)
    __aicore__ inline void asc_gt(__ubuf__ float* src0, __ubuf__ float* src1, uint8_t repeat, uint8_t dst_block_stride, uint8_t src0_block_stride, uint8_t src1_block_stride, uint8_t dst_repeat_stride, uint8_t src0_repeat_stride, uint8_t src1_repeat_stride)
    // 结果输出到目的操作数
    __aicore__ inline void asc_gt(__ubuf__ uint8_t* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint8_t repeat, uint8_t dst_block_stride, uint8_t src0_block_stride, uint8_t src1_block_stride, uint8_t dst_repeat_stride, uint8_t src0_repeat_stride, uint8_t src1_repeat_stride)
    __aicore__ inline void asc_gt(__ubuf__ uint8_t* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint8_t repeat, uint8_t dst_block_stride, uint8_t src0_block_stride, uint8_t src1_block_stride, uint8_t dst_repeat_stride, uint8_t src0_repeat_stride, uint8_t src1_repeat_stride)
    ```

- 同步计算

    ```cpp
    // 结果存入寄存器
    __aicore__ inline void asc_gt_sync(__ubuf__ half* src0, __ubuf__ half* src1, uint8_t repeat, uint8_t      dst_block_stride, uint8_t src0_block_stride, uint8_t src1_block_stride, uint8_t dst_repeat_stride, uint8_t src0_repeat_stride, uint8_t src1_repeat_stride)
    __aicore__ inline void asc_gt_sync(__ubuf__ float* src0, __ubuf__ float* src1, uint8_t repeat, uint8_t dst_block_stride, uint8_t src0_block_stride, uint8_t src1_block_stride, uint8_t dst_repeat_stride, uint8_t src0_repeat_stride, uint8_t src1_repeat_stride)
    // 结果输出到目的操作数
    __aicore__ inline void asc_gt_sync(__ubuf__ uint8_t* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint8_t repeat, uint8_t dst_block_stride, uint8_t src0_block_stride, uint8_t src1_block_stride, uint8_t dst_repeat_stride, uint8_t src0_repeat_stride, uint8_t src1_repeat_stride)
    __aicore__ inline void asc_gt_sync(__ubuf__ uint8_t* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint8_t repeat, uint8_t dst_block_stride, uint8_t src0_block_stride, uint8_t src1_block_stride, uint8_t dst_repeat_stride, uint8_t src0_repeat_stride, uint8_t src1_repeat_stride)
    ```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src0 | 输入 | 源操作数（矢量）的起始地址。 |
| src1 | 输入 | 源操作数（矢量）的起始地址。 |
| count | 输入 | 参与计算的元素个数。 |
| repeat | 输入 | 迭代次数。 |
| dst_block_stride | 输入 | 目的操作数单次迭代内不同DataBlock间地址步长。 |
| src0_block_stride | 输入 | 源操作数0单次迭代内不同DataBlock间地址步长。 |
| src1_block_stride | 输入 | 源操作数1单次迭代内不同DataBlock间地址步长。 |
| dst_repeat_stride | 输入 | 目的操作数相邻迭代间相同DataBlock的地址步长。 |
| src0_repeat_stride | 输入 | 源操作数0相邻迭代间相同DataBlock的地址步长。 |
| src1_repeat_stride | 输入 | 源操作数1相邻迭代间相同DataBlock的地址步长。 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。
- dst、src0、src1的起始地址需要32字节对齐。
- dst按照小端顺序排序成二进制结果，对应src中相应位置的数据比较结果。
- 当结果输出到目的地址中时，mask参数不生效。
- 使用寄存器形式的接口时，需要配合[asc_get_cmp_mask](asc_get_cmp_mask.md)获取比较结果。
- 当结果输出到寄存器中时，固定处理256B的数据，repeat字段不生效。
- 当结果输出到寄存器中时，若mask某bit位为0，对应输出寄存器的bit位的值不会改变时，最终结果可能会有上一次计算的残留。如果要初始化寄存器，可以调用[asc_set_cmp_mask](asc_set_cmp_mask.md)接口。

## 调用示例

```cpp
// 输入固定为128个元素
constexpr uint32_t total_length = 128;
__ubuf__ uint8_t dst[total_length / 8];
__ubuf__ half src0[total_length];
__ubuf__ half src1[total_length];
uint8_t repeat = 1;
uint8_t dst_block_stride = 1;
uint8_t src0_block_stride = 1;
uint8_t src1_block_stride = 1;
uint8_t dst_repeat_stride = 8;
uint8_t src0_repeat_stride = 8;
uint8_t src1_repeat_stride = 8;
…… // 数据搬运及同步操作

// 结果存入寄存器，通过寄存器获取比较结果
asc_gt(src0, src1, repeat, dst_block_stride, src0_block_stride, src1_block_stride, dst_repeat_stride, src0_repeat_stride, src1_repeat_stride);
…… // 同步操作
asc_get_cmp_mask(dst); // 读取结果

// 结果输出到目标地址中
asc_gt(dst, src0, src1, repeat, dst_block_stride, src0_block_stride, src1_block_stride, dst_repeat_stride, src0_repeat_stride, src1_repeat_stride);
```