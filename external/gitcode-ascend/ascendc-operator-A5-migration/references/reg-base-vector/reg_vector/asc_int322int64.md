> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_int322int64.md

# asc_int322int64

## 产品支持情况

| 产品     | 是否支持 |
| ----------- | :----: |
| Ascend 950PR/Ascend 950DT | √    |

## 功能说明

将vector_int32_t类型的源操作数以256B为单位分为两部分，读取其中一部分元素，将其转换成vector_int64_t类型并写入目的操作数。

- asc_int322int64：将源操作数的上半部分写入目的操作数。

- asc_int322int64_v2：将源操作数的下半部分写入目的操作数。

## 函数原型

```cpp
__simd_callee__ inline void asc_int322int64(vector_int64_t& dst, vector_int32_t src, vector_bool mask)
__simd_callee__ inline void asc_int322int64_v2(vector_int64_t& dst, vector_int32_t src, vector_bool mask)
```

## 参数说明

| 参数名       | 输入/输出 | 描述               |
| --------- | ----- | ---------------- |
| dst       | 输出    | 目的操作数（矢量数据寄存器）。            |
| src | 输入    | 源操作数（矢量数据寄存器）。            |
| mask     | 输入    | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。        |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义.md](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
vector_int64_t dst;
vector_int32_t src;
vector_bool mask;
asc_int322int64(dst, src, mask);    // 将src的上半部分转换成vector_int64_t类型并写入dst
asc_int322int64_v2(dst, src, mask);    // 将src的下半部分转换成vector_int64_t类型并写入dst
```