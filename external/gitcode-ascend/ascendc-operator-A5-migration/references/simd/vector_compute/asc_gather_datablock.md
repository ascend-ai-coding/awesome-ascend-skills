> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_compute/asc_gather_datablock.md
# asc_gather_datablock

## 产品支持情况

| 产品     | 是否支持 |
| ----------- | ---- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

### 功能说明

给定一个输入的矢量和一个地址偏移矢量，根据偏移地址按照DataBlock的粒度将输入矢量收集到目的矢量中。

### 函数原型

- 高维切分计算

  ```cpp
  __aicore__ inline void asc_gather_datablock(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, __ubuf__ uint32_t* src_offset, uint16_t dst_repeat_stride, uint8_t dst_block_stride, uint8_t repeat)
  __aicore__ inline void asc_gather_datablock(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src, __ubuf__ uint32_t* src_offset, uint16_t dst_repeat_stride, uint8_t dst_block_stride, uint8_t repeat)
  ```

- 同步计算

  ```cpp
  __aicore__ inline void asc_gather_datablock_sync(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, __ubuf__ uint32_t* src_offset, uint16_t dst_repeat_stride, uint8_t dst_block_stride, uint8_t repeat)
  __aicore__ inline void asc_gather_datablock_sync(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src, __ubuf__ uint32_t* src_offset, uint16_t dst_repeat_stride, uint8_t dst_block_stride, uint8_t repeat)
  ```

### 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src | 输入 | 源操作数（矢量）的起始地址。 |
| src_offset | 输入 | 每个Datablock在源操作数中的地址偏移。 |
| dst_repeat_stride |输入| 目的操作数相邻迭代间相同DataBlock的地址步长。 |
| dst_block_stride |输入| 目的操作数单次迭代内不同DataBlock间地址步长。 |
| repeat |输入| 迭代次数。 |
### 返回值说明

无

### 流水类型

PIPE_V

### 约束说明
- src_offset取值应保证src元素类型位宽对齐。
- src加上src_offset偏移后的地址不能超出Unified Buffer (UB)大小数据的范围。
- dst、src的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。

### 调用示例

```cpp
constexpr uint64_t total_length = 128;    // total_length指参与计算的数据长度
__ubuf__ uint32_t src_offset[total_length * sizeof(uint32_t) / 32];
__ubuf__ uint32_t dst[total_length];
__ubuf__ uint32_t src[total_length];
asc_gather_datablock(dst, src, src_offset, 8, 1, 2);
// src [1, 2, 3, ..., 126, 127, 128] 128个uint32_t类型数据
// src_offset [0, 32, 64, ..., 416, 448, 480] 
// dst [1, 2, 3, ..., 126, 127, 128]
```