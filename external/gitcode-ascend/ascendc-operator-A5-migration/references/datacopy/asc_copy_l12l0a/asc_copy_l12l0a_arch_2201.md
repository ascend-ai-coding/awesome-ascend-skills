> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_copy_l12l0a/asc_copy_l12l0a_arch_2201.md

# asc_copy_l12l0a

## 产品支持情况

| 产品 | 是否支持  |
| :----------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

用于搬运存放在L1 Buffer里的512B大小的矩阵到L0A Buffer里。包含2D格式搬运、3D格式搬运。

## 函数原型

- 高维切分搬运 2D格式
     ```c++
    __aicore__ inline void asc_copy_l12l0a(__ca__ int4b_t* dst, __cbuf__ int4b_t* src, uint16_t start_index, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    __aicore__ inline void asc_copy_l12l0a(__ca__ uint8_t* dst, __cbuf__ uint8_t* src, uint16_t start_index, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    __aicore__ inline void asc_copy_l12l0a(__ca__ int8_t* dst, __cbuf__ int8_t* src, uint16_t start_index, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    __aicore__ inline void asc_copy_l12l0a(__ca__ half* dst, __cbuf__ half* src, uint16_t start_index, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    __aicore__ inline void asc_copy_l12l0a(__ca__ bfloat16_t* dst, __cbuf__ bfloat16_t* src, uint16_t start_index, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    __aicore__ inline void asc_copy_l12l0a(__ca__ uint32_t* dst, __cbuf__ uint32_t* src, uint16_t start_index, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    __aicore__ inline void asc_copy_l12l0a(__ca__ int32_t* dst, __cbuf__ int32_t* src, uint16_t start_index, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    __aicore__ inline void asc_copy_l12l0a(__ca__ float* dst, __cbuf__ float* src, uint16_t start_index, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    ```

- 同步高维切分搬运 2D格式

    ```c++
    __aicore__ inline void asc_copy_l12l0a_sync(__ca__ int4b_t* dst, __cbuf__ int4b_t* src, uint16_t start_index, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    __aicore__ inline void asc_copy_l12l0a_sync(__ca__ uint8_t* dst, __cbuf__ uint8_t* src, uint16_t start_index, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    __aicore__ inline void asc_copy_l12l0a_sync(__ca__ int8_t* dst, __cbuf__ int8_t* src, uint16_t start_index, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    __aicore__ inline void asc_copy_l12l0a_sync(__ca__ half* dst, __cbuf__ half* src, uint16_t start_index, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    __aicore__ inline void asc_copy_l12l0a_sync(__ca__ bfloat16_t* dst, __cbuf__ bfloat16_t* src, uint16_t start_index, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    __aicore__ inline void asc_copy_l12l0a_sync(__ca__ uint32_t* dst, __cbuf__ uint32_t* src, uint16_t start_index, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    __aicore__ inline void asc_copy_l12l0a_sync(__ca__ int32_t* dst, __cbuf__ int32_t* src, uint16_t start_index, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    __aicore__ inline void asc_copy_l12l0a_sync(__ca__ float* dst, __cbuf__ float* src, uint16_t start_index, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    ```

- 高维切分搬运 3D格式

    ```c++
    __aicore__ inline void asc_copy_l12l0a(__ca__ int4b_t* dst, __cbuf__ int4b_t* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0a(__ca__ uint8_t* dst, __cbuf__ uint8_t* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0a(__ca__ int8_t* dst, __cbuf__ int8_t* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0a(__ca__ half* dst, __cbuf__ half* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0a(__ca__ bfloat16_t* dst, __cbuf__ bfloat16_t* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0a(__ca__ uint32_t* dst, __cbuf__ uint32_t* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0a(__ca__ int32_t* dst, __cbuf__ int32_t* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0a(__ca__ float* dst, __cbuf__ float* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    ```

- 同步高维切分搬运 3D格式

    ```c++
    __aicore__ inline void asc_copy_l12l0a_sync(__ca__ int4b_t* dst, __cbuf__ int4b_t* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0a_sync(__ca__ uint8_t* dst, __cbuf__ uint8_t* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0a_sync(__ca__ int8_t* dst, __cbuf__ int8_t* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0a_sync(__ca__ half* dst, __cbuf__ half* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0a_sync(__ca__ bfloat16_t* dst, __cbuf__ bfloat16_t* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0a_sync(__ca__ uint32_t* dst, __cbuf__ uint32_t* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0a_sync(__ca__ int32_t* dst, __cbuf__ int32_t* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    __aicore__ inline void asc_copy_l12l0a_sync(__ca__ float* dst, __cbuf__ float* src, uint16_t k_extension, uint16_t m_extension, uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w, uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w, bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    ```

## 参数说明

表1 2D格式参数说明
| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的L0A Buffer地址。 |
| src | 输入 | 源L1 Buffer地址。 |
| start_index | 输入 | 分形矩阵ID，说明搬运起始位置为源操作数中第几个分形（0为源操作数中第1个分形矩阵）。取值范围：[0, 65535]。单位为512字节。 |
| repeat | 输入 | 迭代次数，每个迭代可以处理512B数据。取值范围：[1, 255]。 |
| src_stride | 输入 | 相邻迭代间，源操作数前一个分形与后一个分型起始地址的间隔。取值范围：[0, 65535]。单位为512字节。 |
| dst_gap | 输入 | 目的操作数相邻连续数据块的间隔（前面一个数据块的尾与后面一个数据块的头的间隔）。取值范围：[0, 65535]。单位为512字节。 |

表2 3D格式参数说明
| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的L0A Buffer地址。 |
| src | 输入 | 源L1 Buffer地址。 |
| k_extension | 输入 | 该指令在目的操作数width维度的传输长度。如果不覆盖最右侧的分形，对于half类型，应为16的倍数，对于int8_t/uint8_t类型，应为32的倍数；如果覆盖最右侧的分形，则无倍数要求。取值范围：[1, 65535]。 |
| m_extension | 输入 | 该指令在目的操作数height维度的传输长度。如果不覆盖最下侧的分形，对于half/int8_t/uint8_t类型，应为16的倍数；如果覆盖最下侧的分形，则无倍数要求。取值范围：[1, 65535]。 |
| k_start_pt | 输入 | 该指令在目的操作数width维度的起点。对于half类型，应为16的倍数，对于int8_t/uint8_t类型，应为32的倍数。取值范围：[0, 65535] |
| m_start_pt | 输入 | 该指令在目的操作数height维度的起点，如果不覆盖最下侧的分形，对于half/int8_t/uint8_t，应为16的倍数；如果覆盖最下侧的分形，则无倍数要求。取值范围：[0, 65535] 。|
| stride_w | 输入 | 卷积核在源操作数width维度滑动的步长，取值范围：[1, 63]。 |
| stride_h | 输入 | 卷积核在源操作数height维度滑动的步长，取值范围：[1, 63]。 |
| filter_w | 输入 | 卷积核width，取值范围：[1, 255]。 |
| filter_h | 输入 | 卷积核height，取值范围：[1, 255]。 |
| dilation_filter_w | 输入 | 卷积核width膨胀系数，取值范围：[1, 255]。 |
| dilation_filter_h | 输入 | 卷积核height膨胀系数，取值范围：[1, 255]。 |
| filter_size_w | 输入 | 是否在filter_w的基础上将卷积核width增加256个元素。true表示增加；false表示不增加。 |
| filter_size_h | 输入 | 是否在filter_h的基础上将卷积核height增加256个元素。true表示增加；false表示不增加。 |
| transpose | 输入 | 是否启用转置功能，对整个目标矩阵进行转置，仅在源操作数为half类型时有效。true表示启用；false表示不启用。 |
| f_matrix_ctrl | 输入 | 表示asc_copy_l12l0a指令从左矩阵还是右矩阵获取FeatureMap的属性描述，当前只支持设置为false。 |
| channel_size | 输入 | 源操作数的通道数，取值范围：[1, 63]。 对于uint32_t/int32_t/float，channelSize可取值为4，N * 8，N * 8 + 4；对于half/bfloat16，channelSize可取值为4，8，N * 16，N * 16 + 4，N * 16 + 8；对于int8_t/uint8_t，channelSize可取值为4，8，16， 32 * N，N * 32 + 4，N * 32 + 8，N * 32 + 16；对于int4b_t，ChannelSize可取值为8，16，32，N * 64，N * 64 + 8，N * 64 + 16，N * 64 + 32。N为正整数。|

## 返回值说明

无

## 流水类型

PIPE_MTE1

## 约束说明

- dst的起始地址需要512字节对齐，src的起始地址需要32字节对齐。
- 本接口不支持2D格式搬运的转置场景。

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
__cbuf__ half src[256];
__ca__ half dst[256];
asc_copy_l12l0a(dst, src, 1, 1, 1, 0);
```
