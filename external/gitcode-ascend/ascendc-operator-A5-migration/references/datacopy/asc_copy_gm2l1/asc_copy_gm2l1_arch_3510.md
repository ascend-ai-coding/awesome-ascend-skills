> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_copy_gm2l1/asc_copy_gm2l1_arch_3510.md

# asc_copy_gm2l1

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| <cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |

## 功能说明

将矩阵数据从Global Memory搬运到L1 Buffer中，支持以下两种搬运模式：
- **填充模式**：搬运过程中可以对数据进行填充或移除填充值。
- **2D搬运模式**：支持2D矩阵数据搬运，支持压缩能力。（NPU_ARCH 220x版本不支持.）

## 函数原型

- 填充模式

    ```cpp
    __aicore__ inline void asc_copy_gm2l1(__cbuf__ void* dst, __gm__ void* src, uint32_t n_burst, uint32_t len_burst, uint8_t pad_func_mode, uint64_t src_stride, uint32_t dst_stride)
    ```

- 填充模式（同步）

    ```cpp
    __aicore__ inline void asc_copy_gm2l1_sync(__cbuf__ void* dst, __gm__ void* src, uint32_t n_burst, uint32_t len_burst, uint8_t pad_func_mode, uint64_t src_stride, uint32_t dst_stride)
    ```

- 2D搬运模式

    ```cpp
    __aicore__ inline void asc_copy_gm2l1(__cbuf__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1(__cbuf__ float *dst, __gm__ float *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1(__cbuf__ float8_e4m3_t *dst, __gm__ float8_e4m3_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1(__cbuf__ float8_e5m2_t *dst, __gm__ float8_e5m2_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1(__cbuf__ half *dst, __gm__ half *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1(__cbuf__ hifloat8_t *dst, __gm__ hifloat8_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1(__cbuf__ int16_t *dst, __gm__ int16_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1(__cbuf__ int32_t *dst, __gm__ int32_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1(__cbuf__ int8_t *dst, __gm__ int8_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1(__cbuf__ uint16_t *dst, __gm__ uint16_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1(__cbuf__ uint32_t *dst, __gm__ uint32_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1(__cbuf__ uint8_t *dst, __gm__ uint8_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1(__cbuf__ float4_e1m2x2_t *dst, __gm__ float4_e1m2x2_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1(__cbuf__ float4_e2m1x2_t *dst, __gm__ float4_e2m1x2_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1(__cbuf__ void *dst, __gm__ void *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    ```

- 2D搬运模式（同步）

    ```cpp
    __aicore__ inline void asc_copy_gm2l1_sync(__cbuf__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1_sync(__cbuf__ float *dst, __gm__ float *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1_sync(__cbuf__ float8_e4m3_t *dst, __gm__ float8_e4m3_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1_sync(__cbuf__ float8_e5m2_t *dst, __gm__ float8_e5m2_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1_sync(__cbuf__ half *dst, __gm__ half *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1_sync(__cbuf__ hifloat8_t *dst, __gm__ hifloat8_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1_sync(__cbuf__ int16_t *dst, __gm__ int16_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1_sync(__cbuf__ int32_t *dst, __gm__ int32_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1_sync(__cbuf__ int8_t *dst, __gm__ int8_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1_sync(__cbuf__ uint16_t *dst, __gm__ uint16_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1_sync(__cbuf__ uint32_t *dst, __gm__ uint32_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1_sync(__cbuf__ uint8_t *dst, __gm__ uint8_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1_sync(__cbuf__ float4_e1m2x2_t *dst, __gm__ float4_e1m2x2_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1_sync(__cbuf__ float4_e2m1x2_t *dst, __gm__ float4_e2m1x2_t *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    __aicore__ inline void asc_copy_gm2l1_sync(__cbuf__ void *dst, __gm__ void *src, uint32_t m_start_position, uint32_t k_start_position, uint16_t dst_stride, uint16_t m_step, uint16_t k_step, uint8_t decomp_mode, uint8_t l2_cache_ctl)
    ```

## 参数说明

### 填充模式参数

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src | 输入 | 源操作数（矢量）的起始地址。 |
| n_burst | 输入 | 待搬运的连续传输数据块个数。 |
| len_burst | 输入 | 待搬运的每个连续传输数据块的长度，单位为32个字节。 |
| pad_func_mode | 输入 | 一个4Bit的值，用于指示通道上的padding功能，他会在32个字节的数据中填充数据；或从每个32个字节的读取数据中移除padding值。<br>&bull; 0：不做padding。 <br>&bull; 1：每有1个字节的数据，插入31个padding值。 <br>&bull; 2：每有2个字节的数据，插入15个padding值。 <br>&bull; 3：每有4个字节的数据，插入14个padding值。<br>&bull; 4：每有8个字节的数据，插入12个padding值。 <br>&bull; 5：每有16个字节的数据，插入8个padding值。<br>&bull; 6：每有32个字节的数据，移除28个最高有效位数据，只保留4个最低有效位元素。<br>&bull; 7：每有32个字节的数据，移除24个最高有效位数据，只保留8个最低有效位元素。<br>&bull; 8：每有32个字节的数据，移除16个最高有效位数据，只保留16个最低有效位元素。 |
| src_stride | 输入 | 输入数据中两个相邻连续数据块之间的距离。 |
| dst_stride | 输入 | 输出数据中两个相邻连续数据块之间的距离。 |

### 2D搬运模式参数

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src | 输入 | 源操作数（矢量）的起始地址。 |
| m_start_position | 输入 | M方向上的起始位置。 |
| k_start_position | 输入 | K方向上的起始位置。 |
| dst_stride | 输入 | 目的操作数步长。 |
| m_step | 输入 | M方向上的步长。 |
| k_step | 输入 | K方向上的步长。 |
| decomp_mode | 输入 | 设置不同的压缩模式。 |
| l2_cache_ctl | 输入 | 配置数据在L2 Cache中的管理策略。取值说明如下：  <br>&bull; 0：DISABLE模式，适用于仅需访问一次的数据。 <br>&bull; 1：NORMAL模式，适用于重用模式未知或不极端的数据。 <br>&bull; 2：LAST模式，适用于高频重复访问的数据。 <br>&bull; 4：PERSISTENT模式，适用于需要长期驻留在缓存中的数据。 |

## 返回值说明

无

## 流水类型

PIPE_MTE2

## 约束说明

### 填充模式约束

- 对于插入padding值的场景：
    - len_burst的单位为32个字节的指的是写入L1 Buffer的实际数据数，它必须设置为1，此时一次仅从GM里读取1/2/4/8/16个字节的数据。
    - 每一次burst都是从GM中连续读取，因此不需要设置src_stride。
    - dst_stride以32个字节为单位。
- 对于移除padding值的场景：
    - 源操作数中一次burst操作的数据为len_burst * 32个字节，目的操作数中一次burst的实际数据为len_burst * 4/8/16个字节。
    - 每一次burst后都连续地往L1 Buffer中写入，因此不需要设置dst_stride。
    - src_stride以32个字节为单位。

### 2D搬运模式约束

- dst、src的起始地址需要32字节对齐。
- 为了兼容性，需要设置m_step为1。

## 调用示例

```cpp
//待搬运的连续传输数据块个数为1
constexpr uint32_t n_burst = 1;
//待搬运的每个连续传输数据块的长度为128个字节
constexpr uint32_t len_burst = 4;
//不对数据做padding
constexpr uint8_t pad_func_mode = 0;
//输入输出数据中两个相邻连续数据块之间的距离为0
constexpr uint64_t src_stride = 0;
constexpr uint32_t dst_stride = 0;
__gm__ half src[256];
__cbuf__ half dst[256];
asc_copy_gm2l1(dst, src, n_burst, len_burst, pad_func_mode, src_stride, dst_stride);
```
