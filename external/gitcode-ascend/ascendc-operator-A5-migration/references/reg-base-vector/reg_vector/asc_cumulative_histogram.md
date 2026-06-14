> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_cumulative_histogram.md

# asc_cumulative_histogram

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <cann-filter npu_type="950"><term>Ascend 950PR/Ascend 950DT</term>  | √ </cann-filter>|

## 功能说明

对直方图数据进行累计统计。根据低位/高位模式分别统计[0, 127]和[128, 255]区间内的数据，dst数据中的第n位数据代表src中0到n内所有数据出现的频率，并在dst源数据基础上累加所统计出的数据。

Vector Length长度为256Byte，dst数据类型为uint16_t，一个dst可以存储128个数据，因此需要两个dst。BIN0表示低位模式，统计src中[0, 127]范围内的数据写入；BIN1表示高位模式，统计src中[128, 255]范围内的数据写入。

统计方式如下图所示，其中dst0表示低位模式，dst1表示高位模式。

![累计统计](../../figures/累计统计.png)

## 函数原型

  ```cpp
  // 低位模式
  __simd_callee__ inline void asc_cumulative_histogram_bin0(vector_uint16_t& dst, vector_uint8_t src, vector_bool mask)
  // 高位模式
  __simd_callee__ inline void asc_cumulative_histogram_bin1(vector_uint16_t& dst, vector_uint8_t src, vector_bool mask)
  ```

## 参数说明

| 参数名       | 输入/输出 | 描述                |
| --------- | ----- | ----------------- |
| dst       | 输出    | 目的操作数（矢量数据寄存器）。 |
| src       | 输入    | 源操作数（矢量数据寄存器）。 |
| mask      | 输入    | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义.md](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
vector_uint16_t dst;
vector_uint8_t src;
vector_bool mask = asc_create_mask_b8(PAT_ALL);
asc_loadalign(src, src_addr); // src_addr是外部输入的UB内存空间地址。
asc_cumulative_histogram_bin0(dst, src, mask);
```
