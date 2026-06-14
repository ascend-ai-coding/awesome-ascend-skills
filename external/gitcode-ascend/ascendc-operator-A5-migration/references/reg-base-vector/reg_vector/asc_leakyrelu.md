> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_leakyrelu.md

# asc_leakyrelu

## 产品支持情况

| 产品                  | 是否支持  |
|:-------------------------| :------: |
| Ascend 950PR/Ascend 950DT | √    |

## 功能说明

按元素执行Leaky ReLU（Leaky Rectified Linear Unit）操作。Leaky ReLU带泄露性整流函数是一种人工神经网络中常用的激活函数，其数学表达式为：

$$
f(x_i) =
\begin{cases}
x_i &  \text{if } x_i ≥ \text{ }0  \\
                                   \\
ax_i &  \text{if } x_i < \text{ }0 \\
\end{cases}
$$

## 函数原型

```cpp
__simd_callee__ inline void asc_leakyrelu(vector_float& dst, vector_float src, float value, vector_bool mask)
__simd_callee__ inline void asc_leakyrelu(vector_half& dst, vector_half src, half value, vector_bool mask)
```

## 参数说明

| 参数名  | 输入/输出 | 描述                                                                      |
|:-----| :--- |:------------------------------------------------------------------------|
| dst | 输出 | 目的操作数（矢量数据寄存器）。                                                           |
| src | 输入 | 源操作数（矢量数据寄存器）。                                                          |
| value | 输入 | 源操作数（标量）的起始地址。                                                          |
| mask | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义](../reg数据类型定义.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

 ```cpp
vector_half dst;
vector_half src;
half value;
vector_bool mask = asc_create_mask_b16(PAT_ALL);
asc_loadalign(src, src_addr); // src_addr是外部输入的UB内存空间地址。
asc_leakyrelu(dst, src, value, mask);
```