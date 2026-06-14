> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_copy_gm2l0b.md

# asc_copy_gm2l0b

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |

## 功能说明

将矩阵数据从Global Memory搬运到L0B Buffer。

## 函数原型

- 常规计算

    ```cpp
    __aicore__ inline void asc_copy_gm2l0b(__cb__ int4b_t* dst, __gm__ int4b_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0b(__cb__ int8_t* dst, __gm__ int8_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0b(__cb__ uint8_t* dst, __gm__ uint8_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0b(__cb__ half* dst, __gm__ half* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0b(__cb__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0b(__cb__ int32_t* dst, __gm__ int32_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0b(__cb__ uint32_t* dst, __gm__ uint32_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0b(__cb__ float* dst, __gm__ float* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    ```

- 同步计算

    ```cpp
    __aicore__ inline void asc_copy_gm2l0b_sync(__cb__ int4b_t* dst, __gm__ int4b_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0b_sync(__cb__ int8_t* dst, __gm__ int8_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0b_sync(__cb__ uint8_t* dst, __gm__ uint8_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0b_sync(__cb__ half* dst, __gm__ half* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0b_sync(__cb__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0b_sync(__cb__ int32_t* dst, __gm__ int32_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0b_sync(__cb__ uint32_t* dst, __gm__ uint32_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_gm2l0b_sync(__cb__ float* dst, __gm__ float* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_stride)
    ```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src | 输入 | 源操作数（矢量）的起始地址。 |
| base_idx | 输入 | 以16*16个数对矩阵进行分块，搬运的起始分块ID。 |
| repeat | 输入 | 迭代次数。 |
| src_stride |输入| 输入数据中两个相邻连续数据块之间的距离。 |
| dst_stride | 输入 | 输出数据中两个相邻连续数据块之间的距离。 |

## 返回值说明

无

## 流水类型

PIPE_MTE2

## 约束说明

无

## 调用示例

```cpp
//搬运的起始分块为1
constexpr uint16_t base_idx = 1;
//搬运的迭代次数为2
constexpr uint8_t repeat = 2; 
//输入的搬运步长为0字节，输出的搬运步长为512字节
constexpr uint16_t src_stride = 0;
constexpr uint16_t dst_stride = 1;
__gm__ half src[256];
__cb__ half dst[256];
asc_copy_gm2l0b(dst, src, base_idx, repeat, src_stride, dst_stride);
```