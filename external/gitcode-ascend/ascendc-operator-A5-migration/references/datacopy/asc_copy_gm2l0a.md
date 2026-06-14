> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_copy_gm2l0a.md

# asc_copy_gm2l0a

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

将数据从GM搬运到L0A Buffer。

## 函数原型

- 常规搬运

    ```cpp
    __aicore__ inline void asc_copy_gm2l0a(__ca__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0a(__ca__ half* dst, __gm__ falf* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0a(__ca__ float* dst, __gm__ float* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0a(__ca__ int32_t* dst, __gm__ int32_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0a(__ca__ uint8_t* dst, __gm__ uint8_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0a(__ca__ int8_t* dst, __gm__ int8_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0a(__ca__ uint32_t* dst, __gm__ uint32_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    ```

- 同步搬运
    
    ```cpp
    __aicore__ inline void asc_copy_gm2l0a_sync(__ca__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0a_sync(__ca__ half* dst, __gm__ half* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0a_sync(__ca__ float* dst, __gm__ float* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0a_sync(__ca__ int32_t* dst, __gm__ int32_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0a_sync(__ca__ uint8_t* dst, __gm__ uint8_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0a_sync(__ca__ int8_t* dst, __gm__ int8_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0a_sync(__ca__ uint32_t* dst, __gm__ uint32_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    ```

## 参数说明

| 参数名 |输入/输出| 描述                                                        |
|-----| ------------ |-----------------------------------------------------------|
| dst | 输出 | 目的操作数起始地址。                                                |
| src | 输入 | 源操作数起始地址。                                                 | 
| base_idx | 输入 | 分形矩阵ID，搬运起始位置为源操作数的第几个分形（0为源操作数中第1个分形矩阵）。取值范围：[0, 65535]。 |
| repeat | 输入 | 迭代次数，每个迭代可以处理512B数据。取值范围：[1, 255]。                        |
| src_stride | 输入 | 相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔， 单位：512B。取值范围：[0, 65535]。    |
| dst_stride | 输入 | 相邻迭代间，目的操作数前一个分形与后一个分形起始地址的间隔， 单位：512B。取值范围：[0, 65535]。   |

## 返回值说明

无

## 流水类型

PIPE_MTE2

## 约束说明

操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。

## 调用示例、
```cpp
__ca__ half dst[256];
__gm__ half src[256];
asc_copy_gm2l0a(dst, src, 0, 1, 64, 64)
```
