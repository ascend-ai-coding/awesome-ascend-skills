> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/layout/MakeNzLayout.md
# MakeNzLayout

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

创建NZ格式的布局，NZ格式是昇腾AI处理器常用的分块存储格式，外层矩阵按行存储，内层矩阵按列存储。

## 函数原型

```cpp
template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeNzLayout(U row, S column)
```

## 参数说明

**表 1** 模板参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| T | 输入 | 数据类型，支持int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、complex32、int64_t、uint64_t。 |
| U | 输入 | 行数类型，size_t或Int整型常量。 |
| S | 输入 | 列数类型，size_t或Int整型常量。 |

**表 2** 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| row | 输入 | 矩阵的总行数。 |
| column | 输入 | 矩阵的总列数。 |

## 返回值说明

- 输入为编译时常量时，返回NZ格式的Layout类型。
- 输入为整型变量时，返回NZ格式的Layout对象。
- 返回对齐后的Layout，对齐方式及对应位置的参数大小说明详见[Layout和层次化表述法](../../../Layout和层次化表述法.md)。

## 约束说明

参数row和column需为size_t类型或Int整型常量。

## 调用示例

```cpp
// 创建NZ格式Layout
using namespace AscendC::Te;
// 编译时常量传参构造Layout
auto staticLayout = MakeNzLayout<half>(Std::Int<32>{}, Std::Int<32>{});

// 运行时变量传参构造Layout
auto layout = MakeNzLayout<half>(32, 32);
```
