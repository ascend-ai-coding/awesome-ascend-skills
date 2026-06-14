> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_half2float.md
# asc_half2float

## 产品支持情况

| 产品     | 是否支持 |
| ----------- | :----: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

将half类型转化为float类型，无舍入模式。

## 函数原型

* 前n个数据计算
  
  ```cpp
  //在转换有精度损失时表示RINT舍入模式，不涉及精度损失时表示不舍入
  __aicore__ inline void asc_half2float(__ubuf__ float* dst, __ubuf__ half* src, uint32_t count)
  ```
- 高维切分计算
  
  ```cpp
  //在转换有精度损失时表示RINT舍入模式，不涉及精度损失时表示不舍入
  __aicore__ inline void asc_half2float(__ubuf__ float* dst, __ubuf__ half* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
  ```

- 同步计算
  
  ```cpp
  //在转换有精度损失时表示RINT舍入模式，不涉及精度损失时表示不舍入
  __aicore__ inline void asc_half2float_sync(__ubuf__ float* dst, __ubuf__ half* src, uint32_t count)
  ```

## 参数说明

| 参数名    | 输入/输出 | 描述                |
| ------ | ----- | ----------------- |
| dst    | 输出    | 目的操作数（矢量）的起始地址。            |
| src    | 输入    | 源操作数（矢量）的起始地址。             |
| count  | 输入    | 参与连续计算的元素个数。      |
| repeat |输入    | 迭代次数。 |
| dst_block_stride |输入    | 目的操作数单次迭代内不同DataBlock间地址步长。 |
| src_block_stride |输入    | 源操作数单次迭代内不同DataBlock间地址步长。 |
| dst_repeat_stride |输入    | 目的操作数相邻迭代间相同DataBlock的地址步长。 |
| src_repeat_stride |输入    | 源操作数相邻迭代间相同DataBlock的地址步长。 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。
- dst、src的起始地址需要32字节对齐。

## 调用示例

```cpp
constexpr uint32_t total_length = 256;
__ubuf__ half src[total_length];
__ubuf__ float dst[total_length];
asc_half2float(dst, src, total_length);
```
