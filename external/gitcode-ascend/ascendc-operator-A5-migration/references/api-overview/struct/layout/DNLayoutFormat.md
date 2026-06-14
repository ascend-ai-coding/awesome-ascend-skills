> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/layout/DNLayoutFormat.md
# DNLayoutFormat

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

DNLayoutFormat用于定义DN格式的布局，DN格式是标准的列主序存储格式，数据按列连续存储。

## 结构体定义

```cpp
template <typename T>
struct DNLayoutFormat {
    template <size_t row, size_t column>
    using type = DNFormatLayout<T, row, column>;

    template <typename U, typename S>
    __aicore__ inline decltype(auto) operator()(U row, S column) {
        return MakeDNLayout<T, U, S>(row, column);
    }  
};

```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| T | 输入 | 数据类型模板参数。<br>支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、complex32、int64_t、uint64_t。 |
| row | 输入 | 矩阵的总行数。 |
| column | 输入 | 矩阵的总列数。 |

## 返回值

- 输入为编译时常量时，返回DN格式的Layout类型。
- 输入为整型变量时，返回DN格式的Layout对象。
- 返回Layout对应位置的参数大小说明详见[Layout和层次化表述法](../../../Layout和层次化表述法.md)。

## 约束说明

参数row和column需为size_t类型或Int整型常量。

## 调用示例

```cpp
// 创建DN格式Layout
using namespace AscendC::Te;
// 根据flag的值，选择DN格式或ND格式的类型
constexpr bool flag = true;
using MyLayoutType = conditional_t<flag, DNLayoutFormat<half>, NDLayoutFormat<half>>;
size_t m = 128;
size_t n = 128;
auto layoutAL1 = MyLayoutType{}(m, n);

// 编译时常量传参构造Layout
using MyDNLayout = DNLayoutFormat<half>::type<Std::Int<32>, Std::Int<32>>;
auto staticLayout = MyDNLayout{};

// 运行时变量传参构造Layout
DNLayoutFormat<half> dnFormat;
auto layout = dnFormat(32, 32);
```
