> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_muls.md

# asc_muls

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## 功能说明

对源操作数src和value进行乘法运算再按照数据类型转换的ROUND舍入模式转成half类型，根据mask将结果写入目的操作数dst。计算公式如下：

$$
dst_i = round_to_half(src_i * value)
$$

## 函数原型

```cpp
// 数据写入索引为偶数的位置
__simd_callee__ inline void asc_muls(vector_half& dst, vector_float src, float value, vector_bool mask)
// 数据写入索引为奇数的位置
__simd_callee__ inline void asc_muls_v2(vector_half& dst, vector_float src, float value, vector_bool mask)
```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量数据寄存器）。 |
| src | 输入 | 源操作数（矢量数据寄存器）。 |
| value | 输入 | 源操作数（标量）。 |
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
vector_float src;
vector_half dst;
float value;
vector_bool mask;
asc_muls(dst, src, value, mask);
```