> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/copy_gm2l1_nd2nz.md

# asc_copy_gm2l1_nd2nz

## 产品支持情况

| 产品 | 是否支持  |
| :----------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

将数据从Global Memory (GM) 搬运到 Level 1 cache (L1)，支持在数据搬运时进行ND格式到NZ格式的转换。

## 函数原型

- 常规搬运

    ```c++
    __aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ int8_t* dst, __gm__ int8_t* src, uint16_t nd_num,
        uint16_t n_value, uint16_t d_value, uint16_t src_nd_matrix_stride, uint16_t src_d_value, uint16_t dst_nz_c0_stride,
        uint16_t dst_nz_n_stride, uint16_t dst_nz_matrix_stride)
    __aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ uint8_t* dst, __gm__ uint8_t* src, uint16_t nd_num,
        uint16_t n_value, uint16_t d_value, uint16_t src_nd_matrix_stride, uint16_t src_d_value, uint16_t dst_nz_c0_stride,
        uint16_t dst_nz_n_stride, uint16_t dst_nz_matrix_stride)
    __aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint16_t nd_num,
        uint16_t n_value, uint16_t d_value, uint16_t src_nd_matrix_stride, uint16_t src_d_value, uint16_t dst_nz_c0_stride,
        uint16_t dst_nz_n_stride, uint16_t dst_nz_matrix_stride)
    __aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ half* dst, __gm__ half* src, uint16_t nd_num,
        uint16_t n_value, uint16_t d_value, uint16_t src_nd_matrix_stride, uint16_t src_d_value, uint16_t dst_nz_c0_stride,
        uint16_t dst_nz_n_stride, uint16_t dst_nz_matrix_stride)
    __aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ int16_t* dst, __gm__ int16_t* src, uint16_t nd_num,
        uint16_t n_value, uint16_t d_value, uint16_t src_nd_matrix_stride, uint16_t src_d_value, uint16_t dst_nz_c0_stride,
        uint16_t dst_nz_n_stride, uint16_t dst_nz_matrix_stride)
    __aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ float* dst, __gm__ float* src, uint16_t nd_num,
        uint16_t n_value, uint16_t d_value, uint16_t src_nd_matrix_stride, uint16_t src_d_value, uint16_t dst_nz_c0_stride,
        uint16_t dst_nz_n_stride, uint16_t dst_nz_matrix_stride)
    __aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ int32_t* dst, __gm__ int32_t* src, uint16_t nd_num,
        uint16_t n_value, uint16_t d_value, uint16_t src_nd_matrix_stride, uint16_t src_d_value, uint16_t dst_nz_c0_stride,
        uint16_t dst_nz_n_stride, uint16_t dst_nz_matrix_stride)
    __aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ uint32_t* dst, __gm__ uint32_t* src, uint16_t nd_num,
        uint16_t n_value, uint16_t d_value, uint16_t src_nd_matrix_stride, uint16_t src_d_value, uint16_t dst_nz_c0_stride,
        uint16_t dst_nz_n_stride, uint16_t dst_nz_matrix_stride)
    ```

- 同步搬运

    ```c++
    __aicore__ inline void asc_copy_gm2l1_nd2nz_sync(__cbuf__ int8_t* dst, __gm__ int8_t* src, uint16_t nd_num,
        uint16_t n_value, uint16_t d_value, uint16_t src_nd_matrix_stride, uint16_t src_d_value, uint16_t dst_nz_c0_stride,
        uint16_t dst_nz_n_stride, uint16_t dst_nz_matrix_stride)
    __aicore__ inline void asc_copy_gm2l1_nd2nz_sync(__cbuf__ uint8_t* dst, __gm__ uint8_t* src, uint16_t nd_num,
        uint16_t n_value, uint16_t d_value, uint16_t src_nd_matrix_stride, uint16_t src_d_value, uint16_t dst_nz_c0_stride,
        uint16_t dst_nz_n_stride, uint16_t dst_nz_matrix_stride)
    __aicore__ inline void asc_copy_gm2l1_nd2nz_sync(__cbuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint16_t nd_num,
        uint16_t n_value, uint16_t d_value, uint16_t src_nd_matrix_stride, uint16_t src_d_value, uint16_t dst_nz_c0_stride,
        uint16_t dst_nz_n_stride, uint16_t dst_nz_matrix_stride)
    __aicore__ inline void asc_copy_gm2l1_nd2nz_sync(__cbuf__ half* dst, __gm__ half* src, uint16_t nd_num,
        uint16_t n_value, uint16_t d_value, uint16_t src_nd_matrix_stride, uint16_t src_d_value, uint16_t dst_nz_c0_stride,
        uint16_t dst_nz_n_stride, uint16_t dst_nz_matrix_stride)
    __aicore__ inline void asc_copy_gm2l1_nd2nz_sync(__cbuf__ int16_t* dst, __gm__ int16_t* src, uint16_t nd_num,
        uint16_t n_value, uint16_t d_value, uint16_t src_nd_matrix_stride, uint16_t src_d_value, uint16_t dst_nz_c0_stride,
        uint16_t dst_nz_n_stride, uint16_t dst_nz_matrix_stride)
    __aicore__ inline void asc_copy_gm2l1_nd2nz_sync(__cbuf__ float* dst, __gm__ float* src, uint16_t nd_num,
        uint16_t n_value, uint16_t d_value, uint16_t src_nd_matrix_stride, uint16_t src_d_value, uint16_t dst_nz_c0_stride,
        uint16_t dst_nz_n_stride, uint16_t dst_nz_matrix_stride)
    __aicore__ inline void asc_copy_gm2l1_nd2nz_sync(__cbuf__ int32_t* dst, __gm__ int32_t* src, uint16_t nd_num,
        uint16_t n_value, uint16_t d_value, uint16_t src_nd_matrix_stride, uint16_t src_d_value, uint16_t dst_nz_c0_stride,
        uint16_t dst_nz_n_stride, uint16_t dst_nz_matrix_stride)
    __aicore__ inline void asc_copy_gm2l1_nd2nz_sync(__cbuf__ uint32_t* dst, __gm__ uint32_t* src, uint16_t nd_num,
        uint16_t n_value, uint16_t d_value, uint16_t src_nd_matrix_stride, uint16_t src_d_value, uint16_t dst_nz_c0_stride,
        uint16_t dst_nz_n_stride, uint16_t dst_nz_matrix_stride)
    ```

## 参数说明


| 参数名                  | 输入/输出 | 描述                                                                                                                                    |
|:---------------------|:------|:--------------------------------------------------------------------------------------------------------------------------------------|
| dst                  | 输出    | 目的操作数（矢量）的起始地址。                                                                                                                       |
| src                  | 输入    | 源操作数的起始地址。                                                                                                                            |
| nd_num               | 输入    | 传输ND矩阵的数目，取值范围：n_num∈[0, 4095]。                                                                                                       |
| n_value              | 输入    | ND矩阵的行数，取值范围：n_value∈[0, 16384]。                                                                                                      |
| d_value              | 输入    | ND矩阵的列数，取值范围：d_value∈[0, 65535]。                                                                                                      |
| src_nd_matrix_stride | 输入    | 源操作数相邻ND矩阵起始地址间的偏移，取值范围：src_nd_matrix_stride∈[0, 65535]，单位为元素。                                                                        |
| src_d_value          | 输入    | 源操作数同一ND矩阵的相邻行起始地址间的偏移，取值范围：src_d_value∈[1, 65535]，单位为元素。                                                                             |
| dst_nz_c0_stride     | 输入    | ND转换到NZ格式后，源操作数中的一行会转换为目的操作数的多行。dst_nz_c0_stride表示，目的NZ矩阵中，来自源操作数同一行的多行数据相邻起始地址间的偏移，取值范围：dst_nz_c0_stride∈[1, 16384]，单位：C0_SIZE（32B）。 |
| dst_nz_n_stride      | 输入    | 目的NZ矩阵中，Z型矩阵相邻行起始地址之间的偏移。取值范围：dst_nz_n_stride∈[1, 16384]，单位：C0_SIZE（32B）。                                                             |
| dst_nz_matrix_stride | 输入    | 目的NZ矩阵中，相邻NZ矩阵起始地址间的偏移，取值范围：dst_nz_matrix_stride∈[1, 65535]，单位为元素。                                                                    |

## 返回值说明

无

## 流水类型

PIPE_MTE2

## 约束说明

- src的起始地址要求按照对应数据类型所占字节数对齐。
- dst的起始地址要求32字节对齐。
- 如果需要执行多条asc_copy_gm2l1_nd2nz指令，且asc_copy_gm2l1_nd2nz指令的目的地址存在重叠，需要插入同步指令，保证多个asc_copy_gm2l1_nd2nz指令的串行化，防止出现异常数据。

## 调用示例

```cpp
//total_length指参与搬运的数据总长度。
constexpr uint64_t total_length = 128;
__cbuf__ int8_t dst[total_length];
asc_copy_gm2l1_nd2nz(dst, src, nd_num, n_value, d_value, src_nd_matrix_stride, src_d_value, dst_nz_c0_stride, dst_nz_n_stride, dst_nz_matrix_stride);
```
