> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/layout/MakeScaleADNLayout.md
# MakeScaleADNLayout

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 函数原型

```cpp
template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleADNLayout(U row, S column)
```

## 功能描述

创建ScaleADN格式的Layout对象。ScaleADN格式用于矩阵A的转置布局（m, scaleK），常用于矩阵乘法中A矩阵转置后的存储。

## 参数说明

**表 1** 模板参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| T | 输入 | 数据类型，支持数据类型为`fp8_e8m0_t`。 |
| U | 输入 | 行参数类型，可以是编译时常量或运行时变量。 |
| S | 输入 | 列参数类型，可以是编译时常量或运行时变量。 |

**表 2** 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| row | 输入 | 矩阵的行数（`m`）。 |
| column | 输入 | 矩阵的列数（`scaleK`）。 |

## 返回值

- 输入为编译时常量时，返回ScaleADNLayout格式的Layout类型。
- 输入为整型变量时，返回ScaleADNLayout格式的Layout对象。
- 返回对齐后的Layout，对齐方式及对应位置的参数大小说明详见[Layout和层次化表述法](../../../Layout和层次化表述法.md)。

## 约束说明

- 参数row和column需为size_t类型或Int整型常量。
- 对于T为fp8_e8m0_t时，column需为2的倍数。

## 使用示例

```cpp
using namespace AscendC::Te;

// 使用编译时常量
auto staticLayout = MakeScaleADNLayout<fp8_e8m0_t>(Std::Int<32>{}, Std::Int<64>{});

// 使用运行时变量
size_t m = 32;
size_t scaleK = 64;
auto layout2 = MakeScaleADNLayout<fp8_e8m0_t>(m, scaleK);
```

