> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_datablock_reduce_min.md
# asc_datablock_reduce_min

## 产品支持情况

| 产品   | 是否支持 |
| ------------|:----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

执行数据块内的求最大值规约（Reduce Min）操作。

## 函数原型

- 前n个数据连续计算

    ```c++
    __aicore__ inline void asc_datablock_reduce_min(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count)
    __aicore__ inline void asc_datablock_reduce_min(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
    ```

- 高维切分计算

    ```c++
    __aicore__ inline void asc_datablock_reduce_min(__ubuf__ half* dst, __ubuf__ half* src, uint8_t repeat, uint16_t dst_repeat_stride, uint16_t src_block_stride, uint16_t src_repeat_stride)
    __aicore__ inline void asc_datablock_reduce_min(__ubuf__ float* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_repeat_stride, uint16_t src_block_stride, uint16_t src_repeat_stride)
    ```

- 同步计算

    ```c++
    __aicore__ inline void asc_datablock_reduce_min_sync(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count)
    __aicore__ inline void asc_datablock_reduce_min_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
    ```

## 参数说明

表1 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| dst     | 输出     | 目的操作数（矢量）的起始地址。 |
| src     | 输入     | 源操作数（矢量）的起始地址。 |
| count   | 输入     | 参与连续计算的元素个数。 |
| repeat | 输入 | 迭代次数。|
| dst_repeat_stride | 输入 | 目的操作数相邻迭代间相同DataBlock的地址步长。<br>输入类型位宽为16bit时，单位为16Byte，输入类型位宽为32bit时，单位为32Byte。|
| src_block_stride | 输入 | 源操作数单次迭代内不同DataBlock间地址步长。|
| src_repeat_stride | 输入 | 源操作数相邻迭代间相同DataBlock的地址步长。|

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。
- dst、src的起始地址需要32字节对齐。

## 调用示例

```c++
constexpr uint32_t src_length = 256;
constexpr uint32_t dst_length = 16;
__ubuf__ half src[src_length];
__ubuf__ half dst[dst_length];
// 每次repeat256B，2次repeat，无间隔
asc_datablock_reduce_min(dst, src, 2, 1, 1, 8);
```

结果示例：

```
输入数据src：[1 2 3 ... 16 17 ... 32 ... 225... 256]
输出数据dst：[1 17 ... 225]
```
