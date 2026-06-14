> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_copy_l0c2l1/asc_copy_l0c2l1_arch_2201.md

# asc_copy_l0c2l1

## AI处理器支持情况

| AI处理器类型 | 是否支持  |
| :----------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |

## 功能说明

矩阵计算完成后，对结果进行量化处理，之后将处理结果搬运到L1 Buffer中。量化模式共分为8种，分别为：
- DEQF16：int32_t量化成half。量化结果不支持INF_NAN模式。
- F322BF16：float量化成bfloat16_t。量化结果不支持INF_NAN模式。
- F322F16：float量化成half。量化结果不支持INF_NAN模式。
- QF322B8_PRE：float量化成uint8_t/int8_t。scalar量化。
- REQ8：int32_t量化成uint8_t/int8_t。scalar量化。
- VDEQF16：int32_t量化成half。量化结果不支持INF_NAN模式。
- VQF322B8_PRE：float量化成uint8_t/int8_t。scalar量化。
- VREQ8：int32_t量化成uint8_t/int8_t。矢量量化。
此外，该搬运指令还支持int4b_t类型的量化，相关量化类型分为两种：
- int32_t量化为int4b_t
- float量化为int4b_t

## 函数原型

- 常规搬运

    ```c++
    __aicore__ inline void asc_copy_l0c2l1(__cbuf__ half* dst, __cc__ float* src, uint16_t n_size,uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t uint_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool enable_channel_split, bool enable_nd2nz)
    __aicore__ inline void asc_copy_l0c2l1(__cbuf__ bfloat16_t* dst, __cc__ float* src, uint16_t n_size,uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t uint_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool enable_channel_split, bool enable_nd2nz)
    __aicore__ inline void asc_copy_l0c2l1(__cbuf__ int8_t* dst, __cc__ float* src, uint16_t n_size,uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t uint_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool enable_channel_split, bool enable_nd2nz)
    __aicore__ inline void asc_copy_l0c2l1(__cbuf__ half* dst, __cc__ int32_t* src, uint16_t n_size,uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t uint_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool enable_channel_split, bool enable_nd2nz)
    __aicore__ inline void asc_copy_l0c2l1(__cbuf__ int16_t* dst, __cc__ int32_t* src, uint16_t n_size,uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t uint_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool enable_channel_split, bool enable_nd2nz)
    __aicore__ inline void asc_copy_l0c2l1(__cbuf__ int8_t* dst, __cc__ int32_t* src, uint16_t n_size,uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t uint_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool enable_channel_split, bool enable_nd2nz)
    __aicore__ inline void asc_copy_l0c2l1(__cbuf__ uint8_t* dst, __cc__ int32_t* src, uint16_t n_size,uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t uint_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool enable_channel_split, bool enable_nd2nz)
    __aicore__ inline void asc_copy_l0c2l1(__cbuf__ void* dst, __cc__ float* src, uint16_t n_size,uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t uint_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool enable_channel_split, bool enable_nd2nz)
    __aicore__ inline void asc_copy_l0c2l1(__cbuf__ void* dst, __cc__ int32_t* src, uint16_t n_size,uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t uint_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool enable_channel_split, bool enable_nd2nz)
    ```
- 同步搬运

    ```c++
    __aicore__ inline void asc_copy_l0c2l1_sync(__cbuf__ half* dst, __cc__ float* src, uint16_t n_size,uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t uint_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool enable_channel_split, bool enable_nd2nz)
    __aicore__ inline void asc_copy_l0c2l1_sync(__cbuf__ bfloat16_t* dst, __cc__ float* src, uint16_t n_size,uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t uint_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool enable_channel_split, bool enable_nd2nz)
    __aicore__ inline void asc_copy_l0c2l1_sync(__cbuf__ int8_t* dst, __cc__ float* src, uint16_t n_size,uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t uint_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool enable_channel_split, bool enable_nd2nz)
    __aicore__ inline void asc_copy_l0c2l1_sync(__cbuf__ half* dst, __cc__ int32_t* src, uint16_t n_size,uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t uint_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool enable_channel_split, bool enable_nd2nz)
    __aicore__ inline void asc_copy_l0c2l1_sync(__cbuf__ int16_t* dst, __cc__ int32_t* src, uint16_t n_size,uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t uint_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool enable_channel_split, bool enable_nd2nz)
    __aicore__ inline void asc_copy_l0c2l1_sync(__cbuf__ int8_t* dst, __cc__ int32_t* src, uint16_t n_size,uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t uint_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool enable_channel_split, bool enable_nd2nz)
    __aicore__ inline void asc_copy_l0c2l1_sync(__cbuf__ uint8_t* dst, __cc__ int32_t* src, uint16_t n_size,uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t uint_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool enable_channel_split, bool enable_nd2nz)
    __aicore__ inline void asc_copy_l0c2l1_sync(__cbuf__ void* dst, __cc__ float* src, uint16_t n_size,uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t uint_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool enable_channel_split, bool enable_nd2nz)
    __aicore__ inline void asc_copy_l0c2l1_sync(__cbuf__ void* dst, __cc__ int32_t* src, uint16_t n_size,uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t uint_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool enable_channel_split, bool enable_nd2nz)
    ```

## 参数说明


| 参数名                  | 输入/输出 | 描述                                                                                                                               |
|:---------------------|:------|:---------------------------------------------------------------------------------------------------------------------------------|
| dst                  | 输出    | 目的操作数（矢量）的起始地址。                                                                                                                  |
| src                  | 输入    | 源操作数的起始地址。                                                                                                                       |
| n_size               | 输入    | 源NZ矩阵在N方向上的大小。<br/>&bull; 不使能NZ2ND功能，取值范围：[1, 4095]；<br/>&bull; 使能NZ2ND功能，取值范围：[1, 4095]。                                        |
| m_size               | 输入    | 源NZ矩阵在M方向上的大小。<br/>&bull; 不使能NZ2ND功能，取值范围：[1, 65535]；<br/>&bull; 使能NZ2ND功能，取值范围：[1, 8192]。                                       |
| dst_stride           | 输入    | 目的相邻ND矩阵起始地址之间的偏移。                                                                                                               |
| src_stride           | 输入    | 源NZ矩阵中相邻Z排布的起始地址偏移，取值范围：[0, 65535]，单位：C0_Size（16*sizeof(T)，T为src的数据类型）。                                                          |
| uint_flag_mode       | 输入    | 与unit_flag参数相关，取值如下：<br/>&bull;0 保留值；<br/>&bull;2 使能unit_flag，硬件执行完指令之后，不会设置寄存器；<br/>&bull;3 使能unit_flag，硬件执行完指令后，会将unit_flag关闭。 |
| quant_pre            | 输入    | 量化参数。取值见[功能说明](./asc_copy_l0c2l1_arch_2201.md#功能说明)。                                                                             |
| relu_pre             | 输入    | 使能relu。                                                                                                                          |
| enable_channel_split | 输入    | 是否使能通道拆分的功能，默认false，不使能该功能。仅在src和dst都为float时才能使能通道拆分，且不能同时使能channel_split和NZ2ND功能。                                               |
| enable_nd2nz         | 输入    | 使能ND2NZ开关。<br/>&bull;false：不使能；<br/>&bull;true：使能。                                                                               |

## 返回值说明

无

## 流水类型

PIPE_FIX

## 约束说明

- src的起始地址要求按照对应数据类型所占字节数对齐。
- dst的起始地址要求32字节对齐。
- 如果需要执行多条asc_copy_l0c2l1指令，且asc_copy_l0c2l1指令的目的地址存在重叠，需要插入同步指令，保证多个asc_copy_l0c2l1指令的串行化，防止出现异常数据。

## 调用示例

```cpp
uint64_t quant_pre = DEQF16;
// dst src分别对应目的操作数的输出地址和源操作数的输入地址
asc_copy_l0c2l1(dst, src, n_size, m_size, dst_stride, src_stride, unit_flag_mode, quant_pre, relu_pre, enable_channel_split, enable_nd2nz);
```
