> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/tile/MakeTile.md
# MakeTile

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

构造Tile对象，用于定义张量的分块。

## 函数原型

```cpp
template <typename... Ts>
__aicore__ inline constexpr Tile<Ts...> MakeTile(const Ts&... t)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| t | 输入 | 各维度的分块大小，可变参数。 |

## 返回值说明

返回Tile<Ts...>对象。

## 约束说明

- 参数数量必须与对应的Shape维度数量一致。
- 各参数必须为正整数。
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