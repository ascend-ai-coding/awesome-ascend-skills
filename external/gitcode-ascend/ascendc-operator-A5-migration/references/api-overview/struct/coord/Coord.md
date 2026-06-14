> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/coord/Coord.md
# Coord

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

Coord用于定义张量的坐标，用于访问张量中特定位置的元素。

## 类型定义

Coord是一个模板别名，用于表示张量的坐标：

```cpp
template <typename... Coords>
using Coord = Std::tuple<Coords...>;
```

其中：
- `Coords...` 是可变参数模板，表示各维度的坐标
- 实际类型为 `Std::tuple<Coords...>`

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|----------|------|
| Coords... | 输入 | 各维度的坐标，类型为size_t等整数类型或者Std::Int类型。 |

## API映射关系

Coord通常通过[MakeCoord](MakeCoord.md)函数创建。

## 约束说明

- Coord的维度数量必须与对应的Shape维度数量一致。
- 各维度的坐标值必须在对应Shape维度的有效范围内。
- 支持的数据类型包括：size_t、int等整数类型或者Std::Int类型。

## 调用示例

```cpp
// 使用整数类型创建一个3维张量的坐标
auto coord = AscendC::Te::MakeCoord(5, 10, 15);

// 获取各维度的坐标
auto coord0 = AscendC::Std::get<0>(coord); // coord0 = 5
auto coord1 = AscendC::Std::get<1>(coord); // coord1 = 10
auto coord2 = AscendC::Std::get<2>(coord); // coord2 = 15

// 使用Std::Int类型创建一个3维张量的坐标
auto coordInt = AscendC::Te::MakeCoord(AscendC::Std::Int<5>{}, AscendC::Std::Int<10>{}, AscendC::Std::Int<15>{});

// 获取各维度的坐标
auto coordInt0 = AscendC::Std::get<0>(coordInt); // coordInt0 = 5
auto coordInt1 = AscendC::Std::get<1>(coordInt); // coordInt1 = 10
auto coordInt2 = AscendC::Std::get<2>(coordInt); // coordInt2 = 15
```