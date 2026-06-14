> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_copy_l12l0b/asc_copy_l12l0b_arch_3510.md

# asc_copy_l12l0b

## 产品支持情况

| 产品                       | 是否支持 |
|:--------------------------| :------: |
| Ascend 950PR/Ascend 950DT |    √     |

## 功能说明

用于搬运存放在L1 Buffer里的512B大小的矩阵到L0B Buffer里，包含2D格式搬运、2D格式转置搬运、3D格式搬运。

## 函数原型

- 常规搬运 2D格式
     ```cpp
    __aicore__ inline void asc_copy_l12l0b(__cb__ bfloat16_t* dst, __cbuf__ bfloat16_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b(__cb__ float8_e4m3_t* dst, __cbuf__ float8_e4m3_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b(__cb__ float8_e5m2_t* dst, __cbuf__ float8_e5m2_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b(__cb__ half* dst, __cbuf__ half* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b(__cb__ float* dst, __cbuf__ float* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b(__cb__ hifloat8_t* dst, __cbuf__ hifloat8_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b(__cb__ int16_t* dst, __cbuf__ int16_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b(__cb__ int32_t* dst, __cbuf__ int32_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b(__cb__ int8_t* dst, __cbuf__ int8_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b(__cb__ uint16_t* dst, __cbuf__ uint16_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b(__cb__ uint32_t* dst, __cbuf__ uint32_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b(__cb__ uint8_t* dst, __cbuf__ uint8_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b(__cb__ fp4x2_e2m1_t* dst, __cbuf__ fp4x2_e2m1_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b(__cb__ fp4x2_e1m2_t* dst, __cbuf__ fp4x2_e1m2_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    
    ```

- 同步常规搬运 2D格式
     ```cpp
    __aicore__ inline void asc_copy_l12l0b_sync(__cb__ bfloat16_t* dst, __cbuf__ bfloat16_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_sync(__cb__ float8_e4m3_t* dst, __cbuf__ float8_e4m3_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_sync(__cb__ float8_e5m2_t* dst, __cbuf__ float8_e5m2_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_sync(__cb__ half* dst, __cbuf__ half* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_sync(__cb__ float* dst, __cbuf__ float* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_sync(__cb__ hifloat8_t* dst, __cbuf__ hifloat8_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_sync(__cb__ int16_t* dst, __cbuf__ int16_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_sync(__cb__ int32_t* dst, __cbuf__ int32_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_sync(__cb__ int8_t* dst, __cbuf__ int8_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_sync(__cb__ uint16_t* dst, __cbuf__ uint16_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_sync(__cb__ uint32_t* dst, __cbuf__ uint32_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_sync(__cb__ uint8_t* dst, __cbuf__ uint8_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_sync(__cb__ fp4x2_e2m1_t* dst, __cbuf__ fp4x2_e2m1_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_sync(__cb__ fp4x2_e1m2_t* dst, __cbuf__ fp4x2_e1m2_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    
    ```

- 转置搬运 2D格式
     ```cpp
    __aicore__ inline void asc_copy_l12l0b_transpose(__cb__ bfloat16_t* dst, __cbuf__ bfloat16_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose(__cb__ float8_e4m3_t* dst, __cbuf__ float8_e4m3_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose(__cb__ float8_e5m2_t* dst, __cbuf__ float8_e5m2_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose(__cb__ half* dst, __cbuf__ half* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose(__cb__ float* dst, __cbuf__ float* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose(__cb__ hifloat8_t* dst, __cbuf__ hifloat8_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose(__cb__ int16_t* dst, __cbuf__ int16_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose(__cb__ int32_t* dst, __cbuf__ int32_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose(__cb__ int8_t* dst, __cbuf__ int8_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose(__cb__ uint16_t* dst, __cbuf__ uint16_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose(__cb__ uint32_t* dst, __cbuf__ uint32_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose(__cb__ uint8_t* dst, __cbuf__ uint8_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose(__cb__ fp4x2_e2m1_t* dst, __cbuf__ fp4x2_e2m1_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose(__cb__ fp4x2_e1m2_t* dst, __cbuf__ fp4x2_e1m2_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    
    ```
  
- 同步转置搬运 2D格式
     ```cpp
    __aicore__ inline void asc_copy_l12l0b_transpose_sync(__cb__ bfloat16_t* dst, __cbuf__ bfloat16_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose_sync(__cb__ float8_e4m3_t* dst, __cbuf__ float8_e4m3_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose_sync(__cb__ float8_e5m2_t* dst, __cbuf__ float8_e5m2_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose_sync(__cb__ half* dst, __cbuf__ half* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose_sync(__cb__ float* dst, __cbuf__ float* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose_sync(__cb__ hifloat8_t* dst, __cbuf__ hifloat8_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose_sync(__cb__ int16_t* dst, __cbuf__ int16_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose_sync(__cb__ int32_t* dst, __cbuf__ int32_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose_sync(__cb__ int8_t* dst, __cbuf__ int8_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose_sync(__cb__ uint16_t* dst, __cbuf__ uint16_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose_sync(__cb__ uint32_t* dst, __cbuf__ uint32_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose_sync(__cb__ uint8_t* dst, __cbuf__ uint8_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose_sync(__cb__ fp4x2_e2m1_t* dst, __cbuf__ fp4x2_e2m1_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    __aicore__ inline void asc_copy_l12l0b_transpose_sync(__cb__ fp4x2_e1m2_t* dst, __cbuf__ fp4x2_e1m2_t* src, uint16_t m_start_position, uint16_t k_start_position, uint8_t m_step, uint8_t k_step, int16_t src_stride, uint16_t dst_stride)
    
    ```

- 高维切分搬运 3D格式

    ```cpp
    __aicore__ inline void asc_copy_l12l0b(__cb__ half* dst, __cbuf__ half* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0b(__cb__ bfloat16_t* dst, __cbuf__ bfloat16_t* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0b(__cb__ uint32_t* dst, __cbuf__ uint32_t* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0b(__cb__ int32_t* dst, __cbuf__ int32_t* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0b(__cb__ float* dst, __cbuf__ float* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    ```

- 同步高维切分搬运 3D格式

    ```cpp
    __aicore__ inline void asc_copy_l12l0b_sync(__cb__ half* dst, __cbuf__ half* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0b_sync(__cb__ bfloat16_t* dst, __cbuf__ bfloat16_t* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0b_sync(__cb__ uint32_t* dst, __cbuf__ uint32_t* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0b_sync(__cb__ int32_t* dst, __cbuf__ int32_t* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0b_sync(__cb__ float* dst, __cbuf__ float* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    ```

## 参数说明

表1 2D格式参数说明
| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的L0B Buffer地址。 |
| src | 输入 | 源L1 Buffer地址。  |
| m_start_position | 输入 | 以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16个元素。 |
| k_start_position | 输入 | 以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B。 |
| m_step | 输入 | 以M*K矩阵为例，源矩阵M轴方向搬运长度，单位为16个元素。取值范围：[0, 255]。进行转置搬运时，还需满足以下额外约束：<br>数据位宽为4时，m_step必须为4的倍数；<br>数据位宽为8时，m_step必须为2的倍数；<br>数据位宽为16时，m_step必须为1的倍数；<br>数据位宽为32时，m_step无额外约束。 |
| k_step | 输入 | 以M*K矩阵为例，源矩阵M轴方向搬运长度，单位为32B。取值范围：[0. 255]。进行转置搬运时，还需满足以下额外约束：<br>数据位宽为4、8或16时，k_step没有额外约束；<br>数据位宽为32时，k_step必须是2的倍数。 |
| src_stride | 输入 | 以M*K矩阵为例，源矩阵K方向前一个分形起始地址与后一个分形起始地址的间隔，单位为512B。 |
| dst_stride | 输入 | 以M*K矩阵为例，目标矩阵K方向前一个分形起始地址与后一个分形起始地址的间隔，单位为512B。 |

表2 3D格式参数说明
| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的L0B Buffer地址。 |
| src | 输入 | 源L1 Buffer地址。 |
| k_extension | 输入 | 该指令在目的操作数width维度的传输长度，取值范围：[1, 65535]。<br>如果不覆盖最右侧的分形，对于half类型，应为16的倍数，对于int8_t/uint8_t类型，应为32的倍数；<br>如果覆盖最右侧的分形，则无倍数要求。 |
| m_extension | 输入 | 该指令在目的操作数height维度的传输长度，取值范围：[1, 65535]。<br>如果不覆盖最下侧的分形，对于half/int8_t/uint8_t类型，应为16的倍数；<br>如果覆盖最下侧的分形，则无倍数要求。 |
| k_start_pt | 输入 | 该指令在目的操作数width维度的起点，取值范围：[0, 65535]。<br>对于half类型，应为16的倍数；<br>对于int8_t/uint8_t类型，应为32的倍数。 |
| m_start_pt | 输入 | 该指令在目的操作数height维度的起点，取值范围：[0, 65535] 。<br>如果不覆盖最下侧的分形，对于half/int8_t/uint8_t，应为16的倍数；<br>如果覆盖最下侧的分形，则无倍数要求。|
| stride_w | 输入 | 卷积核在源操作数width维度滑动的步长，取值范围：[1, 63]。 |
| stride_h | 输入 | 卷积核在源操作数height维度滑动的步长，取值范围：[1, 63]。 |
| filter_w | 输入 | 卷积核width，取值范围：[1, 255]。 |
| filter_h | 输入 | 卷积核height，取值范围：[1, 255]。 |
| dilation_filter_w | 输入 | 卷积核width膨胀系数，取值范围：[1, 255]。 |
| dilation_filter_h | 输入 | 卷积核height膨胀系数，取值范围：[1, 255]。 |
| filter_size_w | 输入 | 是否在filter_w的基础上将卷积核width增加256个元素。true表示增加；false表示不增加。 |
| filter_size_h | 输入 | 是否在filter_h的基础上将卷积核height增加256个元素。true表示增加；false表示不增加。 |
| transpose | 输入 | 是否启用转置功能，对整个目标矩阵进行转置，仅在源操作数为half类型时有效。true表示启用；false表示不启用。 |
| f_matrix_ctrl | 输入 | 表示asc_copy_l12l0b指令从左矩阵还是右矩阵获取FeatureMap的属性描述，当前只支持设置为false。 |
| channel_size | 输入 | 源操作数的通道数，取值范围：[1, 63]。<br>N为正整数。 <br>对于uint32_t/int32_t/float，channelSize可取值为4，N * 8，N * 8 + 4；<br>对于half/bfloat16，channelSize可取值为4，8，N * 16，N * 16 + 4，N * 16 + 8；<br>对于int8_t/uint8_t，channelSize可取值为4，8，16， 32 * N，N * 32 + 4，N * 32 + 8，N * 32 + 16；<br>对于int4b_t，ChannelSize可取值为8，16，32，N * 64，N * 64 + 8，N * 64 + 16，N * 64 + 32。|

## 返回值说明

无

## 流水类型

PIPE_MTE1

## 约束说明

- dst的起始地址需要512字节对齐，src的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../../general_instruction.md#通用地址重叠约束)。

## 3D数据格式说明

要求输入的feature map和filter的格式是 NC1HWC0，其中 C0 是最低维度而且 C0 是固定值为 16（对于u8/s8类型为32），C1=C/C0。

为了简化场景，以下场景假设输入的 feature map 的 channel 为4，即 Ci=4。输入 feature maps 在 A1 中的形状为 (Hi,Wi,Ci)，经过 load3dv1 处理后在 A2 的数据形状为(Wo*Ho, Hk*Wk*Ci)。其中 Wo 和 Ho 是卷积后输出的shape，Hk 和 Wk 是 filter 的 shape。

直观的来看，img2col 的过程就是 filter 在 feature map 上扫过，将对应 feature map 的数据展开成输出数据的每一行的过程。filter 首先在W方向上滑动 Wo 步，然后在 H 方向上走一步然后重复以上过程，最终输出 Wo * Ho 行数据。下图中红色和黄色的数据分别代表第一行和第二行。数字表示原始输入数据，filter 和输出数据三者之间的关联关系。可以看到，load3dv1 首先在输入数据的 Ci 维度搬运对应于 00 的 4 个数，然后搬运对应于 01 的四个数，最终这一行的大小为 Hk*Wk*Ci 即 3*3*4=36 个数。

- 对应的feature map格式如下图：

![ ](../../figures/load3d_01.png)

- 对应的 filter 的格式如下图：

![ ](../../figures/load3d_02.png)

其中 n 为 filter 的个数，可以看出维度排布为 (Hk,Wk,Ci,n)，但是需要注意的是下图的格式还需要根据Mmad中B矩阵的格式转换。

实际操作中，由于存储空间或者计算能力限制，我们通常会将整个卷积计算分块，一次只搬运并计算一小块数据。

![ ](../../figures/load3d_03.png)

对于 A2 的 feature map 来说有两种方案，水平分块和垂直分块。分别对应参数中 repeatMode 的 0 和 1。

注：下图中的分形矩阵大小为 4x4，实际应该为 16x16 (对于 u8/s8 类型为 16x32)

repeatMode =0 时，每次 repeat 会改变在 filter 窗口中读取数据点的位置，然后跳到下一个 C0 的位置。

![ ](../../figures/load3d_04.png)

repeatMode =1 的时候 filter 窗口中读取数据的位置保持不变，每个 repeat 在 feature map 中前进 C0 个元素。

![ ](../../figures/load3d_05.png)

## 调用示例

```cpp
__cb__ bfloat16_t dst[256];
__cbuf__ bfloat16_t src[256];
uint16_t m_start_position = 8;
uint16_t k_start_position = 2;
uint8_t m_step = 4;
uint8_t k_step = 4;
int16_t src_stride = 1;
uint16_t dst_stride = 1;
asc_copy_l12l0b(dst, src, m_start_position, k_start_position, m_step, k_step, src_stride, dst_stride);
```