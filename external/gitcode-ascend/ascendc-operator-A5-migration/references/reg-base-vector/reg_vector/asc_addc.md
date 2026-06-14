> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_addc.md

# asc_addc

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| <cann-filter npu_type="950"><term>Ascend 950PR/Ascend 950DT</term>  | √ </cann-filter>|

## 功能说明

基于掩码mask，对输入数据src0、src1及进位数据src2执行元素逐位相加操作，i为元素索引，相加结果写入dst1。若src0和src1的数据类型为uint32_t，且二者与进位值src2的累加结果超出uint32_t的最值时，在dst0的对应位置上，每4个比特位写入1，否则写入0。

计算公式如下：

$$
\{dst0_i, dst1_i\} = src0_i + src1_i + src2_i
$$

## 函数原型

```cpp
__simd_callee__ inline void asc_addc(vector_bool& dst0, vector_uint32_t& dst1, vector_uint32_t src0, vector_uint32_t src1, vector_bool src2, vector_bool mask)
__simd_callee__ inline void asc_addc(vector_bool& dst0, vector_int32_t& dst1, vector_int32_t src0, vector_int32_t src1, vector_bool src2, vector_bool mask)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst0 | 输出 | 目的操作数（掩码寄存器）。存储加法计算后的进位数据。 |
| dst1 | 输出 | 目的操作数（矢量数据寄存器）。 存储加法计算后的结果。|
| src0 | 输入 | 源操作数（矢量数据寄存器）。|
| src1 | 输入 | 源操作数（矢量数据寄存器）。|
| src2 | 输入 | 源操作数（掩码寄存器），进位数据。|
| mask | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。  |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义.md](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
vector_bool dst0;
vector_int32 dst1;
vector_int32 src0, src1;
vector_bool src2;
vector_bool mask = asc_create_mask_b32(PAT_ALL);
asc_loadalign(src0, src0_addr); // src0_addr是外部输入的UB内存空间地址。
asc_loadalign(src1, src1_addr); // src1_addr是外部输入的UB内存空间地址。
asc_loadalign(src2, src2_addr); // src2_addr是外部输入的UB内存空间地址。
asc_addc(dst0, dst1, src0, src1, src2, mask);
```