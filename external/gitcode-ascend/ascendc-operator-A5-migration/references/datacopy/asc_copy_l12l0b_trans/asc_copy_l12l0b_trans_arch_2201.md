> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_copy_l12l0b_trans/asc_copy_l12l0b_trans_arch_2201.md

# asc_copy_l12l0b_trans

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

该接口实现带转置的2D格式数据从L1 Buffer到L0B Buffer的加载。

下面通过示例来讲解接口功能和关键参数：下文图中一个N形或者一个Z形代表一个分形。

- 对于uint8_t/int8_t数据类型，每次迭代处理32 × 32 × 1B数据，可处理2个分形（一个分形512B），每次迭代中，源操作数中2个连续的16 × 32分形将被合并为1个32 × 32的方块矩阵，基于方块矩阵做转置，转置后分裂为2个16 × 32分形，根据目的操作数分形间隔等参数可以有不同的排布。

如下图示例：

- 共处理3072B的数据，每次迭代处理32 × 32 × 1B数据，需要3次迭代可以完成，repeat_time = 3；
- src_stride = 1，表示相邻迭代间，源操作数前一个方块矩阵与后一个方块矩阵起始地址的间隔为1（单位：32 × 32 × 1B），这里的单位实际上是拼接后的方块矩阵的大小；
- dst_gap = 1，表示相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址的间隔为1（单位：512B）；
- dst_frac_gap = 0，表示每个迭代内目的操作数前一个分形的结束地址与后一个分形起始地址的间隔为0（单位：512B）。

![](../../figures/asc_copy_l12l0b_trans_b8示例一.png)

如下图示例：

- repeat_time和src_stride的解释和上图示例一致。
- dst_gap = 0，表示相邻迭代间，目的操作数前一个迭代第一个分形的结束地址和下一个迭代第一个分形起始地址无间隔。
- dst_frac_gap = 2，表示每个迭代内目的操作数前一个分形的结束地址与后一个分形起始地址的间隔为2（单位：512B）。

![](../../figures/asc_copy_l12l0b_trans_b8示例二.png)

- 对于half/bfloat16_t数据类型，每次迭代处理16 × 16 × 2B数据，可处理1个分形（一个分形512B），每次迭代中，源操作数中1个16 × 16分形将被转置。
  - 因为每次迭代处理16 × 16 × 2B数据，需要3次迭代可以完成，repeat_time = 3；
  - src_stride = 1，表示相邻迭代间，源操作数前一个方块矩阵与后一个方块矩阵起始地址的间隔为1 （单位：16 × 16 × 2B）；
  - dst_gap = 0，表示相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址无间隔；
  - 该场景下，因为其分形即为方块矩阵，每个迭代处理一个分形，不存在迭代内分形的间隔，该参数设置无效。

![](../../figures/asc_copy_l12l0b_trans_b16示例.png)

- 对于float/int32_t/uint32_t数据类型，每次迭代处理16 × 16 × 4B数据，可处理2个分形（一个分形512B），每次迭代中，源操作数2个连续的16 × 8分形将被合并为1个16 × 16的方块矩阵，基于方块矩阵做转置，转置后分裂为2个16 × 8分形，根据目的操作数分形间隔等参数可以有不同的排布。
如下图示例：
  - 因为每次迭代处理16 × 16 × 4B数据，需要3次迭代可以完成，repeat_time = 3；
  - src_stride = 1，表示相邻迭代间，源操作数前一个方块矩阵与后一个方块矩阵起始地址的间隔为1（单位：16 × 16 × 4B），这里的单位实际上是拼接后的方块矩阵的大小；
  - dst_gap = 1，表示相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址的间隔为1（单位：512B）；
  - dst_frac_gap = 0，表示每个迭代内目的操作数前一个分形结束地址与后一个分形起始地址的间隔为0（单位：512B）。

![](../../figures/asc_copy_l12l0b_trans_b32示例一.png)

如下图示例：
- repeat_time和src_stride的解释和上图示例一致。
- dst_gap = 0，表示相邻迭代间，目的操作数前一个迭代第一个分形的结束地址和下一个迭代第一个分形起始地址无间隔。
- dst_frac_gap = 2，表示每个迭代内目的操作数前一个分形结束地址与后一个分形起始地址的间隔为2（单位：512B）。

![](../../figures/asc_copy_l12l0b_trans_b32示例二.png)


## 函数原型

- 高维切分搬运
```cpp
__aicore__ inline void asc_copy_l12l0b_trans(__cb__ uint8_t* dst, __cbuf__ uint8_t* src, uint16_t index_id, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride, bool addrmode, uint64_t dst_frac_stride)
__aicore__ inline void asc_copy_l12l0b_trans(__cb__ int8_t* dst, __cbuf__ int8_t* src, uint16_t index_id, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride, bool addrmode, uint64_t dst_frac_stride)
__aicore__ inline void asc_copy_l12l0b_trans(__cb__ half* dst, __cbuf__ half* src, uint16_t index_id, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride, bool addrmode, uint64_t dst_frac_stride)
__aicore__ inline void asc_copy_l12l0b_trans(__cb__ bfloat16_t* dst, __cbuf__ bfloat16_t* src, uint16_t index_id, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride, bool addrmode, uint64_t dst_frac_stride)
__aicore__ inline void asc_copy_l12l0b_trans(__cb__ uint32_t* dst, __cbuf__ uint32_t* src, uint16_t index_id, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride, bool addrmode, uint64_t dst_frac_stride)
__aicore__ inline void asc_copy_l12l0b_trans(__cb__ int32_t* dst, __cbuf__ int32_t* src, uint16_t index_id, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride, bool addrmode, uint64_t dst_frac_stride)
__aicore__ inline void asc_copy_l12l0b_trans(__cb__ float* dst, __cbuf__ float* src, uint16_t index_id, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride, bool addrmode, uint64_t dst_frac_stride)
```

- 同步高维切分搬运 
```cpp
__aicore__ inline void asc_copy_l12l0b_trans_sync(__cb__ uint8_t* dst, __cbuf__ uint8_t* src, uint16_t index_id, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride, bool addrmode, uint64_t dst_frac_stride)
__aicore__ inline void asc_copy_l12l0b_trans_sync(__cb__ int8_t* dst, __cbuf__ int8_t* src, uint16_t index_id, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride, bool addrmode, uint64_t dst_frac_stride)
__aicore__ inline void asc_copy_l12l0b_trans_sync(__cb__ half* dst, __cbuf__ half* src, uint16_t index_id, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride, bool addrmode, uint64_t dst_frac_stride)
__aicore__ inline void asc_copy_l12l0b_trans_sync(__cb__ bfloat16_t* dst, __cbuf__ bfloat16_t* src, uint16_t index_id, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride, bool addrmode, uint64_t dst_frac_stride)
__aicore__ inline void asc_copy_l12l0b_trans_sync(__cb__ uint32_t* dst, __cbuf__ uint32_t* src, uint16_t index_id, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride, bool addrmode, uint64_t dst_frac_stride)
__aicore__ inline void asc_copy_l12l0b_trans_sync(__cb__ int32_t* dst, __cbuf__ int32_t* src, uint16_t index_id, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride, bool addrmode, uint64_t dst_frac_stride)
__aicore__ inline void asc_copy_l12l0b_trans_sync(__cb__ float* dst, __cbuf__ float* src, uint16_t index_id, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride, bool addrmode, uint64_t dst_frac_stride)
```

## 参数说明

| 参数名       | 输入/输出 | 描述               |
| --------- | ----- | ---------------- |
| dst       | 输出    | 目的操作数。            |
| src | 输入    | 源操作数。            |
| index_id |输入 |方块矩阵的ID，搬运起始位置为源操作数中第几个方块矩阵（0 为源操作数中第1个方块矩阵）。取值范围：index_id∈[0, 65535]。|
| repeat |输入 |迭代重复次数。repeat∈[0, 255]。<br> 对于uint8_t/int8_t数据类型，每次迭代处理32 × 32 × 1B数据；<br> 对于half/bfloat16_t数据类型，每次迭代处理16 × 16 × 2B数据；<br> 对于float/int32_t/uint32_t数据类型，每次迭代处理16 × 16 × 4B数据。|
| src_stride | 输入|表示相邻迭代间，源操作数前一个方块矩阵与后一个方块矩阵起始地址的间隔，单位：（1024B）。取值范围：src_stride∈[0, 65535]。<br> 对于uint8_t/int8_t数据类型，单位是32 × 32 × 1B； <br> 对于half/bfloat16_t数据类型，单位是16 × 16 × 2B；<br> 对于float/int32_t/uint32_t数据类型，单位是16 × 16 × 4B； <br> 对于int4b_t数据类型，每次迭代处理16 × 64 × 0.5B数据。|
| dst_stride | 输入|表示相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址的间隔。取值范围：dst_stride∈[0, 65535]。 |
| addrmode | 输入| 控制地址更新方式。<br>false: 递增，每次迭代在前一个地址的基础上加上src_stride。<br>true: 递减，每次迭代在前一个地址的基础上减去src_stride。 |
| dst_frac_stride | 输入  | 每个迭代内目的操作数转置前一个分形结束地址与后一个分形起始地址的间隔，单位为512B，仅在数据类型为float/int32_t/uint32_t/uint8_t/int8_t时有效。|



## 返回值说明

无

## 流水类型

PIPE_MTE1

## 约束说明

- repeat=0表示不执行搬运操作。
- 开发者需要保证目的操作数转置后的分形没有重叠。

## 调用示例

```cpp
// 设置源操作数和目的操作数，total_length 指参与计算的数据长度
constexpr uint64_t total_length = 512;
__cbuf__ int32_t src[total_length];
__cb__ int32_t dst[total_length];

// 设置搬运过程中的配置
uint8_t n = 64;
uint8_t nBlockSize = 32;
uint8_t repeat = n / nBlockSize;
uint16_t index_id = 0;
uint16_t src_stride = 1;
uint16_t dst_stride = 1;
bool addrmode = false;
uint64_t dst_frac_stride = 0;

// 搬运过程
asc_copy_l12l0b_trans(dst, src, index_id, repeat, src_stride, dst_stride, addrmode, dst_frac_stride);
```
