> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_copy_l0c2l1/asc_copy_l0c2l1_arch_3510.md

# asc_copy_l0c2l1

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

矩阵计算完成后，对结果进行量化处理，之后将处理结果搬运到L1 Buffer中。量化参数共有2个：quant_pre和quant_post，分别对应预处理和后处理阶段。

quant_pre可选量化模式，分别为：
- NoQuant：不使能量化功能。
- F322BF16：float量化成bfloat16_t。量化结果不支持INF_NAN模式。
- F322F16：float量化成half。量化结果支持INF_NAN模式。
- DEQF16：int32_t量化成half。量化结果不支持INF_NAN模式。
- VDEQF16：int32_t量化成half。量化结果不支持INF_NAN模式。
- QF322B8_PRE：float量化成uint8_t/int8_t。scalar量化。
- VQF322B8_PRE：float量化成uint8_t/int8_t。scalar量化。
- REQ8：int32_t量化成uint8_t/int8_t。scalar量化。
- VREQ8：int32_t量化成uint8_t/int8_t。矢量量化。
- QF322FP8_PRE：float量化成fp8_e4m3fn_t，scalar量化。
- VQF322FP8_PRE：float量化成fp8_e4m3fn_t，矢量量化。
- QF322HIF8_PRE：float量化成hifloat8_t（Half to Away Round），scalar量化。
- VQF322HIF8_PRE：float量化成hifloat8_t（Half to Away Round），矢量量化。
- QF322HIF8_PRE_HYBRID：float量化成hifloat8_t（Hybrid Round），scalar量化。
- VQF322HIF8_PRE_HYBRID：float量化成hifloat8_t（Hybrid Round），矢量量化。
- QS322BF16_PRE：int32_t量化成bfloat16_t，scalar量化。
- VQS322BF16_PRE：int32_t量化成bfloat16_t，矢量量化。
- QF322F16_PRE：float量化成half，scalar量化。
- VQF322F16_PRE：float量化成half，矢量量化。
- QF322BF16_PRE：float量化成bfloat16_t，scalar量化。
- VQF322BF16_PRE：float量化成bfloat16_t，矢量量化。
- QF322BF16_PRE：float量化成float，scalar量化。该量化模式精度无法达到双万分之一，可以达到双千分之一。
- VQF322BF16_PRE：float量化成float，矢量量化。该量化模式精度无法达到双万分之一，可以达到双千分之一。

quant_post可选量化模式，分别为：
- NoConv：不使能量化功能。
- QS162B8_POST：int16_t量化成bfloat8_t，scalar量化。
- VQS162B8_POST：int16_t量化成bfloat8_t，矢量量化。
- QF162B8_POST：half量化成bfloat8_t，scalar量化。
- VQF162B8_POST：half量化成bfloat8_t，矢量量化。
- QS162S4_POST：int16_t量化成int4_t，scalar量化。
- VQS162S4_POST：int16_t量化成int4_t，矢量量化。
- QF162S4_POST：half量化成int4_t类型，scalar量化。
- VQF162S4_POST：half量化成int4_t类型，矢量量化。
- QS162S16_POST：int16_t量化成int16_t，scalar量化。
- VQS162S16_POST：int16_t量化成int16_t，矢量量化。
- QF162S16_POST：half量化成int16_t，scalar量化。
- VQF162S16_POST：half量化成int16_t，矢量量化。

## 函数原型

```cpp
__aicore__ inline void asc_copy_l0c2l1(__cbuf__ half* dst, __cc__ float* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t clip_relu_pre, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en, uint64_t quant_post, uint8_t relu_post, bool clip_relu_post, uint8_t eltwise_op, uint8_t eltwise_antq_cfg, bool c0_pad_en)
__aicore__ inline void asc_copy_l0c2l1(__cbuf__ int8_t* dst, __cc__ float* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t clip_relu_pre, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en, uint64_t quant_post, uint8_t relu_post, bool clip_relu_post, uint8_t eltwise_op, uint8_t eltwise_antq_cfg, bool c0_pad_en)
__aicore__ inline void asc_copy_l0c2l1(__cbuf__ uint8_t* dst, __cc__ float* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t clip_relu_pre, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en, uint64_t quant_post, uint8_t relu_post, bool clip_relu_post, uint8_t eltwise_op, uint8_t eltwise_antq_cfg, bool c0_pad_en)
__aicore__ inline void asc_copy_l0c2l1(__cbuf__ float* dst, __cc__ float* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t clip_relu_pre, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en, uint64_t quant_post, uint8_t relu_post, bool clip_relu_post, uint8_t eltwise_op, uint8_t eltwise_antq_cfg, bool c0_pad_en)
__aicore__ inline void asc_copy_l0c2l1(__cbuf__ half* dst, __cc__ int32_t* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t clip_relu_pre, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en, uint64_t quant_post, uint8_t relu_post, bool clip_relu_post, uint8_t eltwise_op, uint8_t eltwise_antq_cfg, bool c0_pad_en)
__aicore__ inline void asc_copy_l0c2l1(__cbuf__ int8_t* dst, __cc__ int32_t* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t clip_relu_pre, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en, uint64_t quant_post, uint8_t relu_post, bool clip_relu_post, uint8_t eltwise_op, uint8_t eltwise_antq_cfg, bool c0_pad_en)
__aicore__ inline void asc_copy_l0c2l1(__cbuf__ uint8_t* dst, __cc__ int32_t* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t clip_relu_pre, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en, uint64_t quant_post, uint8_t relu_post, bool clip_relu_post, uint8_t eltwise_op, uint8_t eltwise_antq_cfg, bool c0_pad_en)
__aicore__ inline void asc_copy_l0c2l1(__cbuf__ int32_t* dst, __cc__ int32_t* src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t clip_relu_pre, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en, uint64_t quant_post, uint8_t relu_post, bool clip_relu_post, uint8_t eltwise_op, uint8_t eltwise_antq_cfg, bool c0_pad_en)
__aicore__ inline void asc_copy_l0c2l1(__cbuf__ void *dst, __cc__ float *src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en, uint64_t quant_post, uint8_t relu_post, bool clip_relu_post, uint8_t eltwise_op, bool eltwise_antq_en, bool C0_pad_en, bool broadcast_en, bool NZ2DN_en)
__aicore__ inline void asc_copy_l0c2l1(__cbuf__ void *dst, __cc__ int32_t *src, uint16_t n_size, uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre, uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre, bool channel_split, bool nz2nd_en, uint64_t quant_post, uint8_t relu_post, bool clip_relu_post, uint8_t eltwise_op, bool eltwise_antq_en, bool C0_pad_en, bool broadcast_en, bool NZ2DN_en)
```

## 参数说明

| 参数名              | 输入/输出 | 描述                                                                                                                               |
|:-----------------|:------|:---------------------------------------------------------------------------------------------------------------------------------|
| dst              | 输出    | 目的操作数（矢量）的起始地址。                                                                                                                  |
| src              | 输入    | 源操作数（矢量）的起始地址。                                                                                                                   |
| n_size           | 输入    | 源NZ矩阵在N方向上的大小。<br/>&bull; 不使能NZ2ND功能，取值范围：[1, 4095]；<br/>&bull; 使能NZ2ND功能，取值范围：[1, 4095]。                                        |
| m_size           | 输入    | 源NZ矩阵在M方向上的大小。<br/>&bull; 不使能NZ2ND功能，取值范围：[1, 65535]；<br/>&bull; 使能NZ2ND功能，取值范围：[1, 8192]。                                       |
| dst_stride       | 输入    | 目的相邻ND矩阵起始地址之间的偏移。                                                                                                               |
| src_stride       | 输入    | 源NZ矩阵中相邻Z排布的起始地址偏移，取值范围：[0, 65535]，单位：C0_Size（16*sizeof(T)，T为src的数据类型）。                                                          |
| l2_cache_ctl | 输入 | 配置数据在L2 Cache中的管理策略。取值说明如下：  <br>&bull; 0：DISABLE模式，适用于仅需访问一次的数据。 <br>&bull; 1：NORMAL模式，适用于重用模式未知或不极端的数据。 <br>&bull; 2：LAST模式，适用于高频重复访问的数据。 <br>&bull; 4：PERSISTENT模式，适用于需要长期驻留在缓存中的数据。 |
| clip_relu_pre    | 输入    | 预处理阶段使能clip_relu，需搭配normal relu（归一化的relu函数）一起使用且需要使能量化功能。                                                                        |
| uint_flag_mode   | 输入    | 与unit_flag参数相关，取值如下：<br/>&bull;0 保留值；<br/>&bull;2 使能unit_flag，硬件执行完指令之后，不会设置寄存器；<br/>&bull;3 使能unit_flag，硬件执行完指令后，会将unit_flag关闭。 |
| quant_pre        | 输入    | 预处理阶段量化参数。取值见[功能说明](./asc_copy_l0c2l1_arch_3510.md#功能说明)。                                                                        |
| relu_pre         | 输入    | 预处理阶段使能relu。                                                                                                                     |
| channel_split    | 输入    | 是否使能通道拆分的功能，默认false，不使能该功能。仅在src和dst都为float时才能使能通道拆分，且不能同时使能channel_split和NZ2ND功能。                                               |
| nz2nd_en         | 输入    | 使能NZ2ND开关。<br/>&bull;false：不使能；<br/>&bull;true：使能。                                                                               |
| quant_post       | 输入    | 后处理阶段量化参数。取值见[功能说明](./asc_copy_l0c2l1_arch_3510.md#功能说明)。                                                                        |
| relu_post        | 输入    | 后处理阶段使能relu。                                                                                                                     |
| clip_relu_post   | 输入    | 后处理阶段使能clip_relu，需搭配normal relu一起使用，且需要使能量化功能。                                                                                   |
| eltwise_op       | 输入    | 定义数据从l0c搬运至l1时的目的操作数地址和通道步长。                                                                                                     |
| eltwise_antq_cfg | 输入    | 按位使能元素的反量化操作。                                                                                                                    |
| C0_pad_en        | 输入    | 使能为C0配置填充位，C0是通道循环的目标步长数。                                                                                                        |
| broadcast_en        | 输入    | 是否使能广播能力。<br/>&bull;false：不使能；<br/>&bull;true：使能，在数据搬运时沿M轴方向进行数据广播。                                                                                                         |
| NZ2DN_en        | 输入    |  使能NZ2DN开关。<br/>&bull;false：不使能；<br/>&bull;true：使能。                                                                                       |

矢量数据寄存器的详细说明请参见[reg数据类型定义.md](../../reg/reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_FIX

## 约束说明

- src的起始地址要求按照对应数据类型所占字节数对齐。
- dst的起始地址要求32字节对齐。

## 调用示例

```cpp
uint64_t quant_pre = DEQF16;
uint64_t quant_post = VQS162B8_POST;
// dst src分别对应目的操作数的输出地址和源操作数的输入地址
asc_copy_l0c2l1(dst, src, n_size, m_size, dst_stride, src_stride, clip_relu_pre, unit_flag_mode, quant_pre, relu_pre, channel_split, nz2nd_en, quant_post, relu_post, clip_relu_post, eltwise_op, eltwise_antq_cfg, c0_pad_en);
```
