> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/tensor/Crd2Idx.md
# Crd2Idx

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

Crd2Idx函数用于将多维坐标（Coordinate）通过布局（Layout）转换为内存位置索引（Index），这里的Layout包含了Shape和Stride信息。

对于一个布局Layout，其Shape为\(d<sub>0</sub>, d<sub>1</sub>, ..., d<sub>n</sub>\)，Stride为\(s<sub>0</sub>, s<sub>1</sub>, ..., s<sub>n</sub>\)，Coordinate为\(c<sub>0</sub>, c<sub>1</sub>, ..., c<sub>n</sub>\)到线性索引Index的转换公式为：

![](../../../figures/zh-cn_formulaimage_0000002372135864.png)

例如，对于Shape \(3, 4, 5\)，Stride \(20, 5, 1\)和Coordinate \(1, 2, 3\)：

```
维度0：c₀ * s₀ = 1 * 20 = 20 
维度1：c₁ * s₁ = 2 * 5  = 10 
维度2：c₂ * s₂ = 3 * 1  = 3
Index = 20 + 10 + 3 = 33
```

当Coordinate维度和Stride维度不相同时，可以采用去线性化（delinearize）的方法，使得Coordinate维度和Stride维度相同，再使用上述公式计算得到最终结果。

去线性化的方法介绍如下：对于一个n维数组，形状为\(d<sub>0</sub>, d<sub>1</sub>, ..., d<sub>n</sub>\)，线性坐标c对应的多维坐标\(c<sub>0</sub>, c<sub>1</sub>, ..., c<sub>n</sub>\)，可以通过以下公式进行转换：

![](../../../figures/zh-cn_formulaimage_0000002405659569.png)

例如：对于Shape \(\(2, 4\), \(3, 5\)\)，`Stride`\(\(3, 6\), \(1, 24\)\)，`Layout` \(\(2, 4\), \(3, 5\)\) : \(\(3, 6\), \(1, 24\)\)，Coordinate（11, 12），按照列优先原则，Crd2Idx的结果为：

```
crd2idx = delinearize(11, 12) * stride 
= ((11 % 2, 11 / 2), (12 % 3, 12 / 3)) *  ((3, 6), (1, 24))
= ((1, 5), (0, 4)) *  ((3, 6), (1, 24))
= 1 * 3 + 5 * 6 + 0 * 1 + 4 * 24 
= 129
```

总结上述过程，计算公式如下：

![](../../../figures/zh-cn_formulaimage_0000002404172833.png)

![](../../../figures/zh-cn_formulaimage_0000002426040106.png)

![](../../../figures/zh-cn_formulaimage_0000002370455372.png)

![](../../../figures/zh-cn_formulaimage_0000002370616000.png)

其中\(d<sub>0</sub>, d<sub>1</sub>, ..., d<sub>n</sub>\)为Shape，\(s<sub>0</sub>, s<sub>1</sub>, ..., s<sub>n</sub>\)为Stride，delinearize公式展开如下：

![](../../../figures/zh-cn_formulaimage_0000002370590770.png)

## 函数原型

```cpp
// Layout输入，Coordinate转换为Index
template <typename T, typename U, typename S>
__aicore__ inline constexpr auto Crd2Idx(const T& coord, const Layout<U, S>& layout)

// Shape和Stride输入，Coordinate转换为Index
template <typename T, typename Shape, typename Stride>
__aicore__ inline constexpr auto Crd2Idx(const T& coord, const Shape& shape, const Stride& stride)
```

## 参数说明

**表 1**  模板参数说明

|参数名|描述|
|--|--|
| T | 张量坐标coord类型 |
| U/Shape | 张量逻辑形状shape类型 |
| S/Stride | 张量步长stride类型 |


**表 2** 参数说明
| 参数名 | 输入/输出 | 描述 |
|--------|----------|------|
| coord | 输入 | Std::tuple结构类型，用于表示张量在不同维度上的坐标值。<br/>输入的数据类型支持size_t和Std::Int。 |
| layout | 输入 | 输入的Layout对象。<br/>输入的数据类型支持Layout类型。 |
| shape | 输入 | Std::tuple结构类型，用于定义数据的逻辑形状，例如二维矩阵的行数和列数或多维张量的各维度大小。<br/>输入的数据类型支持size_t和Std:Int。 |
| stride | 输入 | Std::tuple结构类型，用于定义各维度在内存中的步长，即同维度相邻元素在内存中的间隔，间隔的单位为元素，与Shape的维度信息一一对应。<br/>输入的数据类型支持size_t和Std::Int。 |

## 返回值说明

返回根据Coordinate信息转换之后的索引值。

## 约束说明

输入参数需满足对应的数据类型要求。

## 调用示例

```cpp
using namespace AscendC::Te;

// Layout形式入参计算索引值
constexpr int M = 11;
constexpr int N = 12;
constexpr int blockM = 13;
constexpr int blockN = 14;


auto coord = MakeCoord(AscendC::Std::Int<20>{}, AscendC::Std::Int<30>{});
auto shape = MakeShape(MakeShape(AscendC::Std::Int<blockM>{}, AscendC::Std::Int<M/blockM>{}), MakeShape(AscendC::Std::Int<blockN>{}, AscendC::Std::Int<N/blockN>{}));
auto stride = MakeStride(MakeStride(AscendC::Std::Int<blockN>{}, AscendC::Std::Int<blockM*blockN>{}),MakeStride(AscendC::Std::Int<1>{}, AscendC::Std::Int<M*blockN>{}));


auto layout = MakeLayout(shape, stride);
auto index = layout(coord); // decltype(index)::value = 590
index = Crd2Idx(coord, layout);  // decltype(index)::value = 590

// Shape和Stride形式入参计算索引值
auto blockCoordM    = AscendC::Std::Int<11>{};
auto blockCoordN    = AscendC::Std::Int<12>{};
auto baseShapeM     = AscendC::Std::Int<13>{};
auto baseShapeN     = AscendC::Std::Int<14>{};
auto basestrideM    = AscendC::Std::Int<15>{};
auto basestrideN    = AscendC::Std::Int<16>{};
auto coord = MakeCoord(AscendC::Std::Int<0>{}, blockCoordN);
auto shape = MakeShape(MakeShape(baseShapeM, baseShapeM), MakeShape(baseShapeN, baseShapeN));
auto stride = MakeStride(MakeStride(basestrideM, basestrideM), MakeStride(basestrideN, basestrideN));

auto index = Crd2Idx(coord, shape, stride); // decltype(index)::value = 192
```

