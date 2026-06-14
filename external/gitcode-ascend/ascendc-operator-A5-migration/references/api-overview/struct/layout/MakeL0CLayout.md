> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/layout/MakeL0CLayout.md
# MakeL0CLayout

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

创建L0C格式的布局，L0CLayout是Nz分形的一种，用于描述矩阵计算的结果在物理位置为L0C Buffer的布局格式。

## 函数原型

```cpp
template <typename U, typename S>
__aicore__ inline decltype(auto) MakeL0CLayout(U row, S column)
```

## 参数说明

**表 1** 模板参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| U | 输入 | 行数类型，size_t或Int整型常量。 |
| S | 输入 | 列数类型，size_t或Int整型常量。 |

**表 2** 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| row | 输入 | 矩阵的总行数。 |
| column | 输入 | 矩阵的总列数。 |

## 返回值说明

- 输入为编译时常量时，返回L0C格式的Layout类型。
- 输入为整型变量时，返回L0C格式的Layout对象。
- 返回对齐后的Layout，对齐方式及对应位置的参数大小说明详见[Layout和层次化表述法](../../../Layout和层次化表述法.md)。

## 约束说明

参数row和column需为size_t类型或Int整型常量。

## 调用示例

```cpp
// 创建L0C格式Layout
using namespace AscendC::Te;
// 编译时常量传参构造Layout
auto staticLayout = MakeL0CLayout(Std::Int<32>{}, Std::Int<32>{});

// 运行时变量传参构造Layout
auto layout = MakeL0CLayout(32, 32);
```
