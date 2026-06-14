> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_mull.md

# asc_mull

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| <cann-filter npu_type="950"><term>Ascend 950PR/Ascend 950DT</term>  | √ </cann-filter>|

### 功能说明

根据mask对输入数据src0、src1按元素相乘，将结果写入dst0，溢出部分写入dst1。

![asc_mull示例](../../figures/asc_mull示例.png)

### 函数原型

  ```cpp
  __simd_callee__ inline void asc_mull(vector_int32_t& dst0, vector_int32_t& dst1, vector_int32_t src0, vector_int32_t src1, vector_bool mask)
  __simd_callee__ inline void asc_mull(vector_uint32_t& dst0, vector_uint32_t& dst1, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask)
  ```

### 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst0       | 输出    | 目的操作数（矢量数据寄存器）。 |
| dst1       | 输出    | 目的操作数（矢量数据寄存器）。 |
| src0       | 输入    | 源操作数（矢量数据寄存器）。 |
| src1       | 输入    | 源操作数（矢量数据寄存器）。 |
| mask       | 输入    | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义.md](../reg数据类型定义.md)。

### 返回值说明

无

### 流水类型

PIPE_V

### 约束说明

无

### 调用示例

```cpp
vector_int32_t dst0;
vector_int32_t dst1;
vector_int32_t src0;
vector_int32_t src1;
vector_bool mask = asc_create_mask_b32(PAT_ALL);
asc_loadalign(src0, src0_addr); // src0_addr是外部输入的UB内存空间地址。
asc_loadalign(src1, src1_addr); // src1_addr是外部输入的UB内存空间地址。
asc_mull(dst0, dst1, src0, src1, mask);
```
