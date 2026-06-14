> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_datamove/asc_copy_ub2gm_align/asc_copy_ub2gm_align_arch_2201.md
# asc_copy_ub2gm_align

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

将数据从Unified Buffer (UB) 搬运到 Global Memory (GM)，支持8位/16位/32位分块拷贝操作。

## 函数原型

- 前n个数据搬运

    ```c++
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ uint8_t* dst, __ubuf__ uint8_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ int8_t* dst, __ubuf__ int8_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ half* dst, __ubuf__ half* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ uint16_t* dst, __ubuf__ uint16_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ int16_t* dst, __ubuf__ int16_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ bfloat16_t* dst, __ubuf__ bfloat16_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ uint32_t* dst, __ubuf__ uint32_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ float* dst, __ubuf__ float* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ int32_t* dst, __ubuf__ int32_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ double* dst, __ubuf__ double* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ int64_t* dst, __ubuf__ int64_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ uint64_t* dst, __ubuf__ uint64_t* src, uint32_t size)
    ```

- 高维切分搬运

    ```c++
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ uint8_t* dst, __ubuf__ uint8_t* src, uint16_t n_burst, uint32_t len_burst,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ int8_t* dst, __ubuf__ int8_t* src, uint16_t n_burst, uint32_t len_burst,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ half* dst, __ubuf__ half* src, uint16_t n_burst, uint32_t len_burst,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ uint16_t* dst, __ubuf__ uint16_t* src, uint16_t n_burst, uint32_t len_burst,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ int16_t* dst, __ubuf__ int16_t* src, uint16_t n_burst, uint32_t len_burst,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ bfloat16_t* dst, __ubuf__ bfloat16_t* src, uint16_t n_burst, uint32_t len_burst,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ uint32_t* dst, __ubuf__ uint32_t* src, uint16_t n_burst, uint32_t len_burst,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ float* dst, __ubuf__ float* src, uint16_t n_burst, uint32_t len_burst,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ int32_t* dst, __ubuf__ int32_t* src, uint16_t n_burst, uint32_t len_burst,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ double* dst, __ubuf__ double* src, uint16_t n_burst, uint32_t len_burst,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ int64_t* dst, __ubuf__ int64_t* src, uint16_t n_burst, uint32_t len_burst,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ uint64_t* dst, __ubuf__ uint64_t* src, uint16_t n_burst, uint32_t len_burst,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    ```

- 同步计算

    ```c++
    __aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ uint8_t* dst, __ubuf__ uint8_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ int8_t* dst, __ubuf__ int8_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ half* dst, __ubuf__ half* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ uint16_t* dst, __ubuf__ uint16_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ int16_t* dst, __ubuf__ int16_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ bfloat16_t* dst, __ubuf__ bfloat16_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ uint32_t* dst, __ubuf__ uint32_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ float* dst, __ubuf__ float* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ int32_t* dst, __ubuf__ int32_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ double* dst, __ubuf__ double* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ int64_t* dst, __ubuf__ int64_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ uint64_t* dst, __ubuf__ uint64_t* src, uint32_t size)
    ```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的GM的起始地址。 |
| src | 输入 | 源UB的起始地址。 |
| size | 输入 | 搬运数据大小（字节）。 |
| n_burst | 输入 | 待搬运的连续传输数据块个数。取值范围：[1, 4095]。 |
| len_burst | 输入 | 待搬运的每个连续传输数据块的长度，单位为字节。取值范围：[1, 2097151]。 |
| left_padding_num | 输入 | 连续搬运数据块左侧需要补充的数据范围，单位为元素个数，left_padding_num的字节数不能超过32。|
| right_padding_num | 输入 | 连续搬运数据块右侧需要补充的数据范围，单位为元素个数，right_padding_num的字节数不能超过32。|
| src_gap | 输入 | 源操作数相邻连续数据块的间隔（前面一个数据块的尾与后面一个数据块的头的间隔）。<br>单位为DataBlock（32字节）。|
| dst_gap | 输入 | 目的操作数相邻连续数据块的间隔（前面一个数据块的尾与后面一个数据块的头的间隔）。<br>单位为DataBlock（32字节）。 |

## 返回值说明

无

## 流水类型

PIPE_MTE2

## 约束说明

- dst的起始地址要求按照对应数据类型所占字节数对齐。
- src的起始地址要求32字节对齐。
- 如果需要执行多条asc_copy_ub2gm_align指令，且asc_copy_ub2gm_align指令的目的地址存在重叠，需要插入同步指令，保证多个asc_copy_ub2gm指令的串行化，防止出现异常数据。
- left_padding_num、right_padding_num的字节数均不能超过32Bytes。

## 调用示例

```cpp
//total_length指参与搬运的数据总长度。dst是外部输入的uint8_t类型的GM内存。
constexpr uint64_t total_length = 128;
__ubuf__ uint8_t src[total_length];
asc_copy_ub2gm_align(dst, src, total_length * sizeof(uint8_t));
```
