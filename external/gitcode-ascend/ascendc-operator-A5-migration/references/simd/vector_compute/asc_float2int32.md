> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_float2int32.md
# asc_float2int32

## 产品支持情况

| 产品     | 是否支持 |
| ----------- | ---- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

将float类型转化为int32_t类型，并支持多种舍入模式：

* RINT舍入模式：四舍六入五成双舍入
* ROUND舍入模式：四舍五入舍入
* FLOOR舍入模式：向负无穷舍入
* CEIL舍入模式：向正无穷舍入
* TRUNC舍入模式：向零舍入
  
  

## 函数原型

* 前n个数据计算
  
  ```cpp
  // RINT舍入模式
  __aicore__ inline void asc_float2int32_rn(__ubuf__ int32_t* dst, __ubuf__ float* src, uint32_t count)
  // ROUND舍入模式
  __aicore__ inline void asc_float2int32_rna(__ubuf__ int32_t* dst, __ubuf__ float* src, uint32_t count)
  // FLOOR舍入模式
  __aicore__ inline void asc_float2int32_rd(__ubuf__ int32_t* dst, __ubuf__ float* src, uint32_t count)
  // CEIL舍入模式
  __aicore__ inline void asc_float2int32_ru(__ubuf__ int32_t* dst, __ubuf__ float* src, uint32_t count)
  // TRUNC舍入模式
  __aicore__ inline void asc_float2int32_rz(__ubuf__ int32_t* dst, __ubuf__ float* src, uint32_t count)
  ```

- 高维切分计算
  
  ```cpp
  // RINT舍入模式
  __aicore__ inline void asc_float2int32_rn(__ubuf__ int32_t* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
  // ROUND舍入模式
  __aicore__ inline void asc_float2int32_rna(__ubuf__ int32_t* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
  // FLOOR舍入模式
  __aicore__ inline void asc_float2int32_rd(__ubuf__ int32_t* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
  // CEIL舍入模式
  __aicore__ inline void asc_float2int32_ru(__ubuf__ int32_t* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
  // TRUNC舍入模式
  __aicore__ inline void asc_float2int32_rz(__ubuf__ int32_t* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
  ```

- 同步计算
  
  ```cpp
  // RINT舍入模式
  __aicore__ inline void asc_float2int32_rn_sync(__ubuf__ int32_t* dst, __ubuf__ float* src, uint32_t count)
  // ROUND舍入模式
  __aicore__ inline void asc_float2int32_rna_sync(__ubuf__ int32_t* dst, __ubuf__ float* src, uint32_t count)
  // FLOOR舍入模式
  __aicore__ inline void asc_float2int32_rd_sync(__ubuf__ int32_t* dst, __ubuf__ float* src, uint32_t count)
  // CEIL舍入模式
  __aicore__ inline void asc_float2int32_ru_sync(__ubuf__ int32_t* dst, __ubuf__ float* src, uint32_t count)
  // TRUNC舍入模式
  __aicore__ inline void asc_float2int32_rz_sync(__ubuf__ int32_t* dst, __ubuf__ float* src, uint32_t count)
  ```

## 参数说明

表1 参数说明

| 参数名    | 输入/输出 | 描述                |
| ------ | ----- | ----------------- |
| dst    | 输出    | 目的操作数（矢量）的起始地址。 |
| src    | 输入    | 源操作数（矢量）的起始地址。   |
| count  | 输入    | 参与连续计算的元素个数。      |
| repeat | 输入    | 迭代次数。 |
| dst_block_stride |输入| 目的操作数单次迭代内不同DataBlock间地址步长。 |
| src_block_stride |输入| 源操作数单次迭代内不同DataBlock间地址步长。 |
| dst_repeat_stride |输入| 目的操作数相邻迭代间相同DataBlock的地址步长。 |
| src_repeat_stride |输入| 源操作数相邻迭代间相同DataBlock的地址步长。 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。
- dst、src的起始地址需要32字节对齐。

## 调用示例

```cpp
__ubuf__ int32_t dst[256];
__ubuf__ float src[256];
asc_float2int32_rn(dst, src, 256);
```
