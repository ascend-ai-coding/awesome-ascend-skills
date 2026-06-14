> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_copy_gm2l1_align.md

# asc_copy_gm2l1_align

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

对矩阵数据进行实时padding，完成padding后将数据从Global Memory搬运到L1 Buffer。

## 函数原型

- 常规计算

    ```cpp
    __aicore__ inline void asc_copy_gm2l1_align(__cbuf__ int8_t* dst, __gm__ int8_t* src, uint32_t n_burst, uint32_t len_burst, uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
    __aicore__ inline void asc_copy_gm2l1_align(__cbuf__ uint8_t* dst, __gm__ uint8_t* src, uint32_t n_burst, uint32_t len_burst, uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
    __aicore__ inline void asc_copy_gm2l1_align(__cbuf__ hifloat8_t* dst, __gm__ hifloat8_t* src, uint32_t n_burst, uint32_t len_burst, uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
    __aicore__ inline void asc_copy_gm2l1_align(__cbuf__ int16_t* dst, __gm__ int16_t* src, uint32_t n_burst, uint32_t len_burst, uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
    __aicore__ inline void asc_copy_gm2l1_align(__cbuf__ uint16_t* dst, __gm__ uint16_t* src, uint32_t n_burst, uint32_t len_burst, uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
    __aicore__ inline void asc_copy_gm2l1_align(__cbuf__ half* dst, __gm__ half* src, uint32_t n_burst, uint32_t len_burst, uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
    __aicore__ inline void asc_copy_gm2l1_align(__cbuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint32_t n_burst, uint32_t len_burst, uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
    __aicore__ inline void asc_copy_gm2l1_align(__cbuf__ int32_t* dst, __gm__ int32_t* src, uint32_t n_burst, uint32_t len_burst, uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
    __aicore__ inline void asc_copy_gm2l1_align(__cbuf__ uint32_t* dst, __gm__ uint32_t* src, uint32_t n_burst, uint32_t len_burst, uint8_t left_padding_count,uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
    __aicore__ inline void asc_copy_gm2l1_align(__cbuf__ float* dst, __gm__ float* src, uint32_t n_burst, uint32_t len_burst, uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
    ```

- 同步计算

    ```cpp
    __aicore__ inline void asc_copy_gm2l1_align_sync(__cbuf__ int8_t* dst, __gm__ int8_t* src, uint32_t n_burst, uint32_t len_burst, uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
    __aicore__ inline void asc_copy_gm2l1_align_sync(__cbuf__ uint8_t* dst, __gm__ uint8_t* src, uint32_t n_burst, uint32_t len_burst, uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
    __aicore__ inline void asc_copy_gm2l1_align_sync(__cbuf__ hifloat8_t* dst, __gm__ hifloat8_t* src, uint32_t n_burst, uint32_t len_burst, uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
    __aicore__ inline void asc_copy_gm2l1_align_sync(__cbuf__ int16_t* dst, __gm__ int16_t* src, uint32_t n_burst, uint32_t len_burst, uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
    __aicore__ inline void asc_copy_gm2l1_align_sync(__cbuf__ uint16_t* dst, __gm__ uint16_t* src, uint32_t n_burst, uint32_t len_burst, uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
    __aicore__ inline void asc_copy_gm2l1_align_sync(__cbuf__ half* dst, __gm__ half* src, uint32_t n_burst, uint32_t len_burst, uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
    __aicore__ inline void asc_copy_gm2l1_align_sync(__cbuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint32_t n_burst, uint32_t len_burst, uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
    __aicore__ inline void asc_copy_gm2l1_align_sync(__cbuf__ int32_t* dst, __gm__ int32_t* src, uint32_t n_burst, uint32_t len_burst, uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
    __aicore__ inline void asc_copy_gm2l1_align_sync(__cbuf__ uint32_t* dst, __gm__ uint32_t* src, uint32_t n_burst, uint32_t len_burst, uint8_t left_padding_count,uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
    __aicore__ inline void asc_copy_gm2l1_align_sync(__cbuf__ float* dst, __gm__ float* src, uint32_t n_burst, uint32_t len_burst, uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
    ```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src | 输入 | 源操作数（矢量）的起始地址。 |
| n_burst | 输入 | 待搬运的连续传输数据块个数。 |
| len_burst | 输入 | 待搬运的每个连续传输数据块的长度，单位为32个字节。对位宽为16的数据类型，该参数取值应为2的倍数，对位宽为32的数据类型，该参数取值应为4的倍数。 |
| left_padding_count | 输入 | 数据左侧的padding元素数。对位宽为8的数据类型，该参数的最大值为32，对位宽为16的数据类型，该参数的最大值为16，对位宽为32的数据类型，该参数的最大值为8。 |
| right_padding_count | 输入 | 数据右侧的padding元素数。对位宽为8的数据类型，该参数的最大值为32，对位宽为16的数据类型，该参数的最大值为16，对位宽为32的数据类型，该参数的最大值为8。 |
| data_select_bit | 输入 | padding数据选择位，当前只支持设置为false，将padding值设置为数据块的第一个元素。 |
| l2_cache_ctl | 输入 | 配置数据在L2 Cache中的管理策略。该参数取值说明如下：  <br>&bull; 0：DISABLE模式，适用于仅需访问一次的数据。 <br>&bull; 1：NORMAL模式，适用于重用模式未知或不极端的数据。 <br>&bull; 2：LAST模式，适用于高频重复访问的数据。 <br>&bull; 4：PERSISTENT模式，适用于需要长期驻留在缓存中的数据。 |
| burst_src_stride |输入| 输入数据中两个相邻的burst所对应的连续数据块头与头之间的距离。 |
| burst_dst_stride | 输入 | 输出数据中两个相邻的burst所对应的连续数据块头与头之间的距离。 |

## 返回值说明

无

## 流水类型

PIPE_MTE2

## 约束说明

- 目的操作数不得存在地址重叠。对于存在地址重叠的操作，硬件将不返回任何警告或错误，也无法保证地址重叠数据的正确性。

## 调用示例

```cpp
//待搬运的连续传输数据块个数为2
constexpr uint32_t n_burst = 2;
//待搬运的每个连续传输数据块的长度为64个字节
constexpr uint32_t len_burst = 2;
//数据左右侧padding的元素数为0
constexpr uint8_t left_padding_count = 0;
constexpr uint8_t right_padding_count = 0;
//padding值取数据块的第一个元素
constexpr bool data_select_bit = false;
//l2 cache采用DISABLE模式
constexpr uint8_t l2_cache_ctl = 0;
//输入输出数据中两个相邻连续数据块之间的距离为0
constexpr uint64_t burst_src_stride = 0;
constexpr uint32_t burst_dst_stride = 0;
__gm__ half src[256];
__cbuf__ half dst[256];
asc_copy_gm2l1_align(dst, src, n_burst, len_burst,  left_padding_count, right_padding_count, data_select_bit, l2_cache_ctl, burst_src_stride, burst_dst_stride);
```