> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/layout/ScaleBNDLayoutFormat.md
# ScaleBNDLayoutFormat

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

ScaleBNDLayoutFormat用于定义ScaleBND格式的布局，ScaleBND格式是一种支持缩放的布局。

## 结构体定义

```cpp
template <typename T>
struct ScaleBNDLayoutFormat {
    template <size_t row, size_t column>
    using type = ScaleBNDFormatLayout<T, row, column>;

    template <typename U, typename S>
    __aicore__ inline decltype(auto) operator()(U row, S column) {
        return MakeScaleBNDLayout<T, U, S>(row, column);
    }  
};
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| T | 输入 | 数据类型模板参数。<br>支持的数据类型为：fp8_e8m0_t。 |
| row | 输入 | 矩阵的总行数。 |
| column | 输入 | 矩阵的总列数。 |

## 返回值

- 输入为编译时常量时，返回ScaleBND格式的Layout类型。
- 输入为整型变量时，返回ScaleBND格式的Layout对象。
- 返回对齐后的Layout，对齐方式及对应位置的参数大小说明详见[Layout和层次化表述法](../../../Layout和层次化表述法.md)。

## 约束说明

- 参数row和column需为size_t类型或Int整型常量。
- 对于T为fp8_e8m0_t时，row需为2的倍数。

## 调用示例

```cpp
// 创建ScaleBND格式Layout
using namespace AscendC::Te;

// 根据flag的值，选择ScaleBND格式或ScaleBDN格式的类型
constexpr bool flag = true;
using MyLayoutType = conditional_t<flag, ScaleBNDFormatLayout<fp8_e8m0_t>, ScaleBDNFormatLayout<fp8_e8m0_t>>;
size_t scaleK = 128;
size_t n = 128;
auto layoutAL1 = MyLayoutType{}(scaleK, n);

// 编译时常量
using MyScaleBNDLayout = ScaleBNDLayoutFormat<fp8_e8m0_t>::type<Std::Int<64>, Std::Int<32>>;
constexpr MyScaleBNDLayout staticLayout1;
auto staticLayout2 = MyScaleBNDLayout{};

// 运行时值
ScaleBNDLayoutFormat<fp8_e8m0_t> scaleBNDFormat;
auto layout = scaleBNDFormat(64, 32);
```
