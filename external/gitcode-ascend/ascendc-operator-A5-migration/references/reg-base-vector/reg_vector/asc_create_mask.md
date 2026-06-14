> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_create_mask.md

# asc_create_mask

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <cann-filter npu_type="950">Ascend 950PR/Ascend 950DT  | √ </cann-filter>|

## 功能说明

根据入参生成相应的掩码寄存器。

## 函数原型

  ```cpp
  asc_create_mask_b8(pat_mode)
  asc_create_mask_b16(pat_mode)
  asc_create_mask_b32(pat_mode)
  ```

## 参数说明

| 参数名       | 输入/输出 | 描述                |
| --------- | ----- | ----------------- |
| pat_mode  | 输入 | 创建掩码寄存器的模式，定义如下：<pre><code>PAT_ALL, // All elements are set to True<br>PAT_VL1, // The lowest element<br>PAT_VL2, // The lowest 2 element<br>PAT_VL3, // The lowest 3 element<br>PAT_VL4, // The lowest 4 element<br>PAT_VL8, // The lowest 8 element<br>PAT_VL16, // The lowest 16 element<br>PAT_VL32, // The lowest 32 element<br>PAT_VL64, // The lowest 64 element<br>PAT_VL128, // The lowest 128 element<br>PAT_M3, // Multiples of 3<br>PAT_M4, // Multiples of 4<br>PAT_H, // The lowest half elements<br>PAT_Q, // The lowest quarter elements<br>PAT_ALLF = 15 // All elements are set to False</pre> |

## 返回值说明

根据上述提供的模式生成相应的掩码寄存器。

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

见[掩码寄存器调用示例](../reg数据类型定义.md#调用示例-1)。
