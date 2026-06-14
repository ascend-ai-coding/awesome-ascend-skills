> **原始文档路径**: asc-devkit/docs/api/context/c_api/vector_datamove/asc_ndim_copy_gm2ub.md
# asc_ndim_copy_gm2ub

## 产品支持情况

| 产品 | 是否支持  |
| :----------------------- | :------: |
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

多维数据搬运接口，将数据从Global Memory (GM) 搬运到 Unified Buffer (UB)。相比于基础数据搬移接口，可以更加自由配置搬入的维度信息。
本功能最多能处理5个维度，每个维度都可以单独配置Stride，且支持随路Padding功能。相关配置参数有很多，需要通过不同的接口传入。

表1 Stride配置涉及的所有参数及其传入接口，其中i表示第i维，可取值[0,4]
| 参数 | 传入接口 | 描述 |
| :--- | :------- | :--- |
| loopi_size | 本接口 | 表示每个维度内，处理的元素个数（不包含Padding元素）。<br> 单位为元素个数，取值范围为[0, 2^20-1]。 |
| loopi_src_stride | [asc_set_ndim_loopi_stride](./asc_set_ndim_loop_stride.md) | 表示每个维度内，该源操作数元素与下一个元素间的间隔。<br> 单位为元素个数，取值范围为[0, 2^40-1]。 |
| loopi_dst_stride | [asc_set_ndim_loopi_stride](./asc_set_ndim_loop_stride.md) | 表示每个维度内，该目的操作数元素与下一个元素间的间隔。<br> 单位为元素个数，取值范围为[0, 2^40-1]。 |

表2 Padding配置涉及的所有参数及其传入接口其中i表示第i维，可取值[0,4]
| 参数 | 传入接口 | 描述 |
| :--- | :------- | :--- |
| padding_mode | 本接口 | 表示Padding值填取方式，类型为bool。<br>&bull; True：使能常数填充方式，即所有Padding值为固定值pad_value。<br>&bull; False：使能最近值填充方式，即左右Padding值会选取当前维度最左或最右的值进行填充。 |
| pad_value | [asc_set_ndim_pad_value](./asc_set_ndim_pad_value.md) | 当padding_mode为True时，该值将作为Padding的填充固定值。当padding_mode为False时，该值不生效。 |
| loopi_lp_count | <br>&bull; 0维：本接口 <br>&bull; 1至4维：[asc_set_ndim_pad_count](./asc_set_ndim_pad_count.md) | 表示每个维度内，左侧需要补齐的元素个数。<br> 单位为元素个数，取值范围为[0, 255]。 |
| loopi_rp_count | <br>&bull; 0维：本接口 <br>&bull; 1至4维：[asc_set_ndim_pad_count](./asc_set_ndim_pad_count.md) | 表示每个维度内，右侧需要补齐的元素个数。<br> 单位为元素个数，取值范围为[0, 255]。 |

以下以2维的例子，介绍几个典型使用场景。

- 2D Padding场景，使能固定值填充方式
![ ](../figures/asc_ndim_copy_gm2ub典型使用场景：2D_Padding场景，使能固定值填充方式.png)

- 2D Padding场景，使能最近值填充方式
![ ](../figures/asc_ndim_copy_gm2ub典型使用场景：2D_Padding场景，使能最近值填充方式.png)

- 2D Transpose场景
![ ](../figures/asc_ndim_copy_gm2ub典型使用场景：2D_Transpose场景.png)

- 2D Slice场景
![ ](../figures/asc_ndim_copy_gm2ub典型使用场景：2D_Slice场景.png)

- 2D BroadCast场景
![ ](../figures/asc_ndim_copy_gm2ub典型使用场景：2D_BroadCast场景.png)


## 函数原型

```c++
__aicore__ inline void asc_ndim_copy_gm2ub(__ubuf__ int8_t* dst, __gm__ int8_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size, uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode, uint8_t cache_mode)
__aicore__ inline void asc_ndim_copy_gm2ub(__ubuf__ uint8_t* dst, __gm__ uint8_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size, uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode, uint8_t cache_mode)
__aicore__ inline void asc_ndim_copy_gm2ub(__ubuf__ fp4x2_e2m1_t* dst, __gm__ fp4x2_e2m1_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size, uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode, uint8_t cache_mode)
__aicore__ inline void asc_ndim_copy_gm2ub(__ubuf__ fp4x2_e1m2_t* dst, __gm__ fp4x2_e1m2_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size, uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode, uint8_t cache_mode)
__aicore__ inline void asc_ndim_copy_gm2ub(__ubuf__ fp8_e8m0_t* dst, __gm__ fp8_e8m0_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size, uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode, uint8_t cache_mode)
__aicore__ inline void asc_ndim_copy_gm2ub(__ubuf__ fp8_e5m2_t* dst, __gm__ fp8_e5m2_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size, uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode, uint8_t cache_mode)
__aicore__ inline void asc_ndim_copy_gm2ub(__ubuf__ fp8_e4m3fn_t* dst, __gm__ fp8_e4m3fn_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size, uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode, uint8_t cache_mode)
__aicore__ inline void asc_ndim_copy_gm2ub(__ubuf__ int16_t* dst, __gm__ int16_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size, uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode, uint8_t cache_mode)
__aicore__ inline void asc_ndim_copy_gm2ub(__ubuf__ uint16_t* dst, __gm__ uint16_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size, uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode, uint8_t cache_mode)
__aicore__ inline void asc_ndim_copy_gm2ub(__ubuf__ half* dst, __gm__ half* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size, uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode, uint8_t cache_mode)
__aicore__ inline void asc_ndim_copy_gm2ub(__ubuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size, uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode, uint8_t cache_mode)
__aicore__ inline void asc_ndim_copy_gm2ub(__ubuf__ int32_t* dst, __gm__ int32_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size, uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode, uint8_t cache_mode)
__aicore__ inline void asc_ndim_copy_gm2ub(__ubuf__ uint32_t* dst, __gm__ uint32_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size, uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode, uint8_t cache_mode)
__aicore__ inline void asc_ndim_copy_gm2ub(__ubuf__ float* dst, __gm__ float* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size, uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode, uint8_t cache_mode)
```

## 参数说明

参数说明
| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的UB地址。 |
| src | 输入 | 源GM地址。 |
| loop0_size | 输入 | 表示0维处理的元素个数（不包含Padding元素）。<br> 单位为元素个数，取值范围为[1, 2^20-1]。 |
| loop1_size | 输入 | 表示1维处理的元素个数（不包含Padding元素）。<br> 单位为元素个数，取值范围为[1, 2^20-1]。 |
| loop2_size | 输入 | 表示2维处理的元素个数（不包含Padding元素）。<br> 单位为元素个数，取值范围为[1, 2^20-1]。 |
| loop3_size | 输入 | 表示3维处理的元素个数（不包含Padding元素）。<br> 单位为元素个数，取值范围为[1, 2^20-1]。 |
| loop4_size | 输入 | 表示4维处理的元素个数（不包含Padding元素）。<br> 单位为元素个数，取值范围为[1, 2^20-1]。 |
| loop0_lp_count | 输入 | 表示0维左侧需要补齐的元素个数。<br> 单位为元素个数，取值范围为[0, 255]。 |
| loop0_rp_count | 输入 | 表示0维右侧需要补齐的元素个数。<br> 单位为元素个数，取值范围为[0, 255]。 |
| padding_mode | 输入 | 表示Padding值填取方式。<br>&bull; True：使能常数填充方式，即所有Padding值为固定值pad_value。<br>&bull; False：使能最近值填充方式，即左右Padding值会选取当前维度最左或最右的值进行填充。 |
| cache_mode | 输入 | L2 cache模式。 |

## 返回值说明

无

## 流水类型

PIPE_MTE2

## 约束说明

- src的起始地址要求按照对应数据类型所占字节数对齐。
- dst的起始地址要求32字节对齐。
- 如果需要执行多条asc_ndim_copy_gm2ub指令，且asc_ndim_copy_gm2ub指令的目的地址存在重叠，需要插入同步指令，保证多个asc_ndim_copy_gm2ub指令的串行化，防止出现异常数据。
- 一条指令所能获取的所有数据的地址范围宽度不能超过40位（1TB），即：源操作数的i维大小为（loopi_lp_size + loopi_size + loopi_rp_size - 1） * loop0_src_stride，目的操作数的i维大小为：（loopi_lp_size + loopi_size + loopi_rp_size - 1） * loop0_dst_stride，所有的维度的大小加起来不超过2^40位。
- 当每维的loopi_dst_stride为升序序列，则不同循环间的地址空间不能交织或重叠。

## 调用示例

2D Padding场景，使能固定值填充方式
![ ](../figures/asc_ndim_copy_gm2ub典型使用场景：2D_Padding场景，使能固定值填充方式.png)

```cpp
//total_length指参与搬运的数据总长度。src是外部输入的half类型的GM内存。
constexpr uint64_t total_length = 54;
__ubuf__ half dst[total_length];

uint32_t loop0_size = 3;
uint32_t loop1_size = 4;
uint32_t loop2_size = 1;
uint32_t loop3_size = 1;
uint32_t loop4_size = 1;
uint64_t loop0_src_stride = 1;
uint64_t loop1_src_stride = 3;
uint64_t loop2_src_stride = 0;
uint64_t loop3_src_stride = 0;
uint64_t loop4_src_stride = 0;
uint64_t loop0_dst_stride = 1;
uint64_t loop1_dst_stride = 6;
uint64_t loop2_dst_stride = 0;
uint64_t loop3_dst_stride = 0;
uint64_t loop4_dst_stride = 0;

bool padding_mode = false;
half pad_value = 88.0;
uint8_t loop0_lp_count = 2;
uint8_t loop1_lp_count = 2;
uint8_t loop0_rp_count = 1;
uint8_t loop1_rp_count = 3;

asc_set_ndim_loop0_stride(loop0_dst_stride, loop0_src_stride);
asc_set_ndim_loop1_stride(loop1_dst_stride, loop1_src_stride);
asc_set_ndim_loop2_stride(loop2_dst_stride, loop2_src_stride);
asc_set_ndim_loop3_stride(loop3_dst_stride, loop3_src_stride);
asc_set_ndim_loop4_stride(loop4_dst_stride, loop4_src_stride);

asc_set_ndim_pad_value(pad_value);

asc_ndim_pad_count_config pad_count_config;
pad_count_config.loop1_lp_count = loop1_lp_count;
pad_count_config.loop1_rp_count = loop1_rp_count;
asc_set_ndim_pad_count(pad_count_config);

asc_ndim_copy_gm2ub(dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
    loop0_lp_count, loop0_rp_count, padding_mode, 0);
```