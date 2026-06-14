> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/tile/Tile.md
# Tile

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

Tile用于定义张量的分块，用于将大张量分割成多个小块进行并行处理。

## 类型定义

Tile是一个模板别名，用于表示张量的分块：

```cpp
template <typename... Layouts>
using Tile = Std::tuple<Layouts...>;
```

其中：
- `Layouts...`是可变参数模板，表示各维度的分块大小
- 实际类型为`Std::tuple<Layouts...>`

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|----------|------|
| Layouts... | 输入 | 各维度分块大小，类型为size_t等整数类型或者Std::Int类型。 |

## API映射关系

Tile通常通过[MakeTile](MakeTile.md)函数创建

## 约束说明

- Tile的维度数量必须与对应的Shape维度数量一致。
- 各维度的分块大小必须为正整数。
- 分块大小不能超过对应Shape维度的大小。
- 支持的数据类型包括：size_t、int等整数类型或者Std::Int类型。

## 调用示例

```cpp
// 使用整数类型创建一个3维张量的分块
auto tile = AscendC::Te::MakeTile(2, 5, 3);

// 获取各维度的分块大小
auto tile0 = AscendC::Std::get<0>(tile); // tile0 = 2
auto tile1 = AscendC::Std::get<1>(tile); // tile1 = 5
auto tile2 = AscendC::Std::get<2>(tile); // tile2 = 3

// 使用Std::Int创建一个3维张量的分块
auto tileInt = AscendC::Te::MakeTile(AscendC::Std::Int<2>{}, AscendC::Std::Int<5>{}, AscendC::Std::Int<3>{});

// 获取各维度的分块大小
auto tileInt0 = AscendC::Std::get<0>(tileInt); // tileInt0 = 2
auto tileInt1 = AscendC::Std::get<1>(tileInt); // tileInt1 = 5
auto tileInt2 = AscendC::Std::get<2>(tileInt); // tileInt2 = 3
```