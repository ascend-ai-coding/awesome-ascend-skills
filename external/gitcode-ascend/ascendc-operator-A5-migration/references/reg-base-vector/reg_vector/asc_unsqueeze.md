> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_unsqueeze.md

# asc_unsqueeze

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <cann-filter npu_type="950"><term>Ascend 950PR/Ascend 950DT</term>  | √ </cann-filter>|

## 功能说明

根据mask进行解压缩，将生成的数据输出到dst。解压缩方式：dst中的第0个元素置为0，dst中的第i个元素等于mask中从第0个到第(i-1)个元素中1的数量。mask最高位被忽略不参与统计。算法逻辑表示如下：
  ```cpp
  dst[0] = 0;
  // dstLen为dst中包含的元素个数
  for (int i = 1; i < dstLen; i++) {
    dst[i] = mask[i - 1] ? (dst[i - 1] + 1) : dst[i - 1];
  }
  ```

## 函数原型

  ```cpp
  __simd_callee__ inline void asc_unsqueeze(vector_int8_t& dst, vector_bool mask)
  __simd_callee__ inline void asc_unsqueeze(vector_uint8_t& dst, vector_bool mask)
  __simd_callee__ inline void asc_unsqueeze(vector_int16_t& dst, vector_bool mask)
  __simd_callee__ inline void asc_unsqueeze(vector_uint16_t& dst, vector_bool mask)
  __simd_callee__ inline void asc_unsqueeze(vector_int32_t& dst, vector_bool mask)
  __simd_callee__ inline void asc_unsqueeze(vector_uint32_t& dst, vector_bool mask)
  ```

## 参数说明

| 参数名       | 输入/输出 | 描述                |
| --------- | ----- | ----------------- |
| dst       | 输出    | 目的操作数（矢量数据寄存器）。 |
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
vector_int8_t dst;
vector_bool mask = asc_create_mask_b8(PAT_ALL);
asc_squeeze(dst, mask);
```
