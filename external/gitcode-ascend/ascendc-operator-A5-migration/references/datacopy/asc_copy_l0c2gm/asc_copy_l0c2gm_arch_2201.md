> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_copy_l0c2gm/asc_copy_l0c2gm_arch_2201.md


# asc_copy_l0c2gm

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

矩阵计算完成后，对结果进行量化处理，之后将处理结果搬运到GM中。量化模式共分为9种，分别为：
- NoQuant：不使能量化功能。
- F322F16：float量化成half。量化结果不支持INF_NAN模式。
- F322BF16：float量化成bfloat16_t。量化结果不支持INF_NAN模式。
- DEQF16：int32_t量化成half。量化结果不支持INF_NAN模式。
- VDEQF16：int32_t量化成half。量化结果不支持INF_NAN模式。
- QF322B8_PRE：float量化成uint8_t/int8_t。scalar量化。
- VQF32B8_PRE：float量化成uint8_t/int8_t。矢量量化。
- REQ8：int32_t量化成uint8_t/int8_t。scalar量化。
- VREQ8：int32_t量化成uint8_t/int8_t。矢量量化。

## 函数原型

- 常规搬运

    ```cpp
    __aicore__ inline void asc_copy_l0c2gm(__gm__ half* dst, __cc__ float* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en)
    __aicore__ inline void asc_copy_l0c2gm(__gm__ bfloat16_t* dst, __cc__ float* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en)
    __aicore__ inline void asc_copy_l0c2gm(__gm__ int8_t* dst, __cc__ float* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en)
    __aicore__ inline void asc_copy_l0c2gm(__gm__ uint8_t* dst, __cc__ float* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en)
    __aicore__ inline void asc_copy_l0c2gm(__gm__ float* dst, __cc__ float* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en)
    __aicore__ inline void asc_copy_l0c2gm(__gm__ half* dst, __cc__ int32_t* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en)
    __aicore__ inline void asc_copy_l0c2gm(__gm__ int16_t* dst, __cc__ int32_t* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en)
    __aicore__ inline void asc_copy_l0c2gm(__gm__ int8_t* dst, __cc__ int32_t* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en)
    __aicore__ inline void asc_copy_l0c2gm(__gm__ int32_t* dst, __cc__ int32_t* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en)
    ```

- 同步搬运

    ```cpp
    __aicore__ inline void asc_copy_l0c2gm_sync(__gm__ half* dst, __cc__ float* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en)
    __aicore__ inline void asc_copy_l0c2gm_sync(__gm__ bfloat16_t* dst, __cc__ float* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en)
    __aicore__ inline void asc_copy_l0c2gm_sync(__gm__ int8_t* dst, __cc__ float* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en)
    __aicore__ inline void asc_copy_l0c2gm_sync(__gm__ uint8_t* dst, __cc__ float* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en)
    __aicore__ inline void asc_copy_l0c2gm_sync(__gm__ float* dst, __cc__ float* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en)
    __aicore__ inline void asc_copy_l0c2gm_sync(__gm__ half* dst, __cc__ int32_t* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en)
    __aicore__ inline void asc_copy_l0c2gm_sync(__gm__ int16_t* dst, __cc__ int32_t* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en)
    __aicore__ inline void asc_copy_l0c2gm_sync(__gm__ int8_t* dst, __cc__ int32_t* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en)
    __aicore__ inline void asc_copy_l0c2gm_sync(__gm__ int32_t* dst, __cc__ int32_t* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t unit_flag_mode,  uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en)
    ```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src | 输入 | 源操作数（矢量）的起始地址。 |
| n_size | 输入 | 源NZ矩阵在N方向上的大小。<br> - 不使能NZ2ND功能：若使能channel_split功能，n_size必须为8的倍数，取值范围为[1, 4095]。若不使能channel_split功能，n_size必须为16的倍数，取值范围为[1, 4095]。<br>- 使能NZ2ND功能：m_size的取值范围为[1, 4095]。 |
| m_size | 输入 | 源NZ矩阵在M方向上的大小。<br> - 不使能NZ2ND功能：取值范围为[1, 65535]。<br> - 使能NZ2ND功能，m_size的取值范围为[1, 8192]。 |
| dst_stride_dst_d | 输入 |目的相邻ND矩阵起始地址之间的偏移 。 |
| src_stride | 输入 | 源NZ矩阵中相邻Z排布的起始地址偏移，取值范围为[0, 65535]，单位为16*sizeof(数据类型)。 |
| unit_flag_mode | 输入 | 与unit_flag参数相关，取值如下：<br>0：保留值；<br>2：使能unit_flag，硬件执行完指令之后，不会设置寄存器；<br>3：使能unit_flag，硬件执行完指令后，会将unit_flag关闭。 |
| quant_pre |输入|量化参数。取值见[功能说明](./asc_copy_l0c2gm_arch_2201.md#功能说明)。|
| relu_pre | 输入 | 使能relu。 |
| channel_split | 输入 | 是否使能通道拆分的功能，默认false，不使能该功能。仅在src和dst都为float时才能使能通道拆分，且不能同时使能channel_split和NZ2ND功能。 |
| nz2nd_en | 输入 |使能nz2nd开关，false：不使能；true：使能 。 |

## 返回值说明

无

## 流水类型

PIPE_MTE1

## 约束说明

- dst、src的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../../general_instruction.md#通用地址重叠约束)。

## 调用示例

```cpp
// total_length指参与搬运的数据总长度
constexpr uint64_t total_length = 128;
// dst src分别对应目的操作数的输出地址和源操作数的输入地址
__gm__ int32_t dst[total_length];
__cc__ int32_t src[total_length];
// 其余入参均已默认数值传入
uint16_t n_size = 16;
uint16_t m_size = 16;
uint32_t dst_stride_dst_d = 0;
uint16_t src_stride = 8;
uint8_t unit_flag_mode = 0;
uint64_t quant_pre = NoQuant;
uint8_t relu_pre = 0;
bool channel_split = false;
bool nz2nd_en = false;
// 函数调用
asc_copy_l0c2gm_sync(dst, src, n_size, m_size, dst_stride_dst_d, src_stride, unit_flag_mode, quant_pre, relu_pre, channel_split, nz2nd_en);
```
