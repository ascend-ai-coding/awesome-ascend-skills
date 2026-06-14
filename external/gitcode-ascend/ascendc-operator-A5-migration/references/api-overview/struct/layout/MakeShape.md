> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/layout/MakeShape.md
# MakeShape

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

构造Shape对象，用于描述数据的逻辑形状。支持传入多个维度值，也支持传入嵌套的Shape子结构以构造层次化Shape。

## 函数原型

```cpp
template <typename... Ts>
__aicore__ inline constexpr Shape<Ts...> MakeShape(const Ts&... t)
```

## 参数说明

**表 1** 模板参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| Ts... | 输入 | Shape各元素的类型，可以是整型变量、`Std::Int`整型常量，或嵌套的Shape子结构类型。 |

**表 2** 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| t | 输入 | Shape的各维度取值或子结构，可变参数。 |

## 返回值说明

返回`Shape<Ts...>`对象。

## 约束说明

参数`t`需满足可用于构造`Shape<Ts...>`对象。

## 调用示例

```cpp
using namespace AscendC::Te;

// 构造普通Shape
auto shape = MakeShape(10, 20, 30);

// 构造层次化Shape
auto fractalShape = MakeShape(MakeShape(16, 8), MakeShape(32, 4));

auto dim0 = Std::get<0>(shape);                // dim0 = 10
auto innerRow = Std::get<0>(Std::get<0>(fractalShape)); // innerRow = 16
```
