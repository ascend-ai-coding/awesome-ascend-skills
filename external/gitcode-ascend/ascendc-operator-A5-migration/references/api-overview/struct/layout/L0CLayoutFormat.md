> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/layout/L0CLayoutFormat.md
# L0CLayoutFormat

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

L0CLayoutFormat用于描述矩阵计算的结果在物理位置为L0C Buffer的布局格式，L0CLayout是Nz分形的一种。

## 结构体定义

```cpp
template <typename T>
struct L0CLayoutFormat {
    template <size_t row, size_t column>
    using type = L0CFormatLayout<row, column>;

    template <typename U, typename S>
    __aicore__ inline decltype(auto) operator()(U row, S column) {
        return MakeL0CLayout<U, S>(row, column);
    }  
};
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| T | 输入 | 数据类型模板参数。<br>支持的数据类型为：int32_t、float。 |
| row | 输入 | 矩阵的总行数。 |
| column | 输入 | 矩阵的总列数。 |

## 返回值

- 输入为编译时常量时，返回L0C格式的Layout类型。
- 输入为整型变量时，返回L0C格式的Layout对象。
- 返回对齐后的Layout，对齐方式及对应位置的参数大小说明详见[Layout和层次化表述法](../../../Layout和层次化表述法.md)。

## 约束说明

参数row和column需为size_t类型或Int整型常量。

## 调用示例

```cpp
// 创建L0C格式Layout
using namespace AscendC::Te;
// 根据flag的值，选择L0C格式或DN格式的类型
constexpr bool flag = true;
using MakeLayoutGM = conditional_t<flag, DNLayoutFormat<half>, L0CLayoutFormat<half>>;
size_t m = 128;
size_t n = 128;
auto layoutAL1 = MakeLayoutGM{}(m, n);

// 编译时常量传参构造Layout
using MyL0CLayout = L0CLayoutFormat<uint16_t>::type<Std::Int<32>, Std::Int<32>>;
auto staticLayout = MyL0CLayout{};

// 运行时变量传参构造Layout
L0CLayoutFormat<uint16_t> l0cFormat;
auto layout = l0cFormat(32, 32);
```
