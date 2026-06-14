> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/layout/MakeLayout.md
# MakeLayout

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

将传入的Shape和Stride数据打包成Layout数据结构。

## 函数原型

```cpp
template <typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto MakeLayout(const ShapeType& shape, const StrideType& stride)
```

## 参数说明

**表 1** 模板参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| ShapeType | 输入 | Shape类型，需为`Std::tuple`结构类型。 |
| StrideType | 输入 | Stride类型，需为`Std::tuple`结构类型。 |

**表 2** 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| shape | 输入 | `Std::tuple`结构类型，用于定义数据的逻辑形状，例如二维矩阵的行数和列数或多维张量的各维度大小。 |
| stride | 输入 | `Std::tuple`结构类型，用于定义各维度在内存中的步长，即同维度相邻元素在内存中的间隔，间隔的单位为元素，与Shape的维度信息一一对应。 |

## 返回值说明

返回Layout对象。

## 约束说明

构造Layout对象时传入的Shape和Stride结构，需是[`Std::tuple`](../../../../容器函数.md)结构类型，且满足Std::tuple结构类型的使用约束。

## 调用示例

```cpp
// 初始化Layout数据结构，获取对应数值
using namespace AscendC::Te;

auto shape = MakeShape(10, 20, 30);
auto stride = MakeStride(1, 100, 200);
auto layoutMake = MakeLayout(shape, stride);
Layout<decltype(shape), decltype(stride)> layoutInit(shape, stride);

int value = Std::get<0>(layoutMake.Shape()); // value = 10
value = Std::get<1>(layoutMake.Shape()); // value = 20
value = Std::get<2>(layoutMake.Shape()); // value = 30

value = Std::get<0>(layoutInit.Stride()); // value = 1
value = Std::get<1>(layoutInit.Stride()); // value = 100
value = Std::get<2>(layoutInit.Stride()); // value = 200
```

