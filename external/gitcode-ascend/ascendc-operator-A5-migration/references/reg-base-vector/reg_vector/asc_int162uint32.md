> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_int162uint32.md

# asc_int162uint32

## 产品支持情况

| 产品     | 是否支持 |
| ----------- | :----: |
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

将int16_t类型转化为uint32_t类型，无舍入模式。

由于源操作数与目的操作数类型位宽比为1:2，读取数据时需要将一个VL大小的数据分为两部分，根据不同接口选取索引0或者索引1。

## 函数原型

```cpp
// 数据读取索引为偶数的位置
__simd_callee__ inline void asc_int162uint32(vector_uint32_t& dst, vector_int16_t src, vector_bool mask)
// 数据读取索引为奇数的位置
__simd_callee__ inline void asc_int162uint32_v2(vector_uint32_t& dst, vector_int16_t src, vector_bool mask)
```

## 参数说明

| 参数名    | 输入/输出 | 描述                |
| :------ | :----- | :----------------- |
| dst    | 输出    | 目的操作数（矢量数据寄存器）。            |
| src    | 输入    | 源操作数（矢量数据寄存器）。             |
| mask | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义.md](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
vector_int16_t src;
vector_uint32_t dst;
vector_bool mask = asc_create_mask_b16(PAT_ALL);
asc_loadalign(src, src_addr); // src_addr是外部输入的UB内存空间地址。
asc_int162uint32(dst, src, mask);
```
