> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/layout/ScaleADNLayoutFormat.md
# ScaleADNLayoutFormat

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

ScaleADNLayoutFormat用于定义ScaleADN格式的布局，ScaleADN格式是一种支持缩放的布局。

## 结构体定义

```cpp
template <typename T>
struct ScaleADNLayoutFormat {
    template <size_t row, size_t column>
    using type = ScaleADNFormatLayout<T, row, column>;

    template <typename U, typename S>
    __aicore__ inline decltype(auto) operator()(U row, S column) {
        return MakeScaleADNLayout<T, U, S>(row, column);
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

- 输入为编译时常量时，返回ScaleADN格式的Layout类型。
- 输入为整型变量时，返回ScaleADN格式的Layout对象。
- 返回对齐后的Layout，对齐方式及对应位置的参数大小说明详见[Layout和层次化表述法](../../../Layout和层次化表述法.md)。

## 约束说明

- 参数row和column需为size_t类型或Int整型常量。
- 对于T为fp8_e8m0_t时，column需为2的倍数。

## 调用示例

```cpp
// 创建ScaleADN格式Layout
using namespace AscendC::Te;
// 根据flag的值，选择ScaleAND格式或ScaleADN格式的类型
constexpr bool flag = true;
using MyLayoutType = conditional_t<flag, ScaleANDFormatLayout<fp8_e8m0_t>, ScaleADNFormatLayout<fp8_e8m0_t>>;
size_t m = 128;
size_t scaleK = 128;
auto layoutAL1 = MyLayoutType{}(m, scaleK);

// 编译时常量传参构造Layout
using MyScaleADNLayout = ScaleADNLayoutFormat<fp8_e8m0_t>::type<Std::Int<32>, Std::Int<64>>;
auto staticLayout = MyScaleADNLayout{};

// 运行时变量传参构造Layout
ScaleADNLayoutFormat<fp8_e8m0_t> scaleADNFormat;
auto layout = scaleADNFormat(32, 64);
```
