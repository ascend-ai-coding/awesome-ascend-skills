> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/layout/ZzLayoutFormat.md
# ZzLayoutFormat

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

ZzLayoutFormat用于定义Zz格式的布局，Zz格式是一种特殊的分块存储格式。

## 结构体定义

```cpp
template <typename T>
struct ZzLayoutFormat {
    template <size_t row, size_t column>
    using type = ZZFormatLayout<T, row, column>;

    template <typename U, typename S>
    __aicore__ inline decltype(auto) operator()(U row, S column) {
        return MakeZzLayout<T, U, S>(row, column);
    }  
};
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| T | 输入 | 数据类型模板参数。<br>支持的数据类型为：fp8_e8m0_t、int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、complex32、int64_t、uint64_t。 |
| row | 输入 | 矩阵的总行数。 |
| column | 输入 | 矩阵的总列数。 |

## 返回值

- 输入为编译时常量时，返回Zz格式的Layout类型。
- 输入为整型变量时，返回Zz格式的Layout对象。
- 返回对齐后的Layout，对齐方式及对应位置的参数大小说明详见[Layout和层次化表述法](../../../Layout和层次化表述法.md)。

## 约束说明

- 参数row和column需为size_t类型或Int整型常量。
- 对于T为fp8_e8m0_t时，column需为2的倍数。

## 调用示例

```cpp
// 创建Zz格式Layout
using namespace AscendC::Te;
// 根据flag的值，选择Nn格式或Zz格式的类型
constexpr bool flag = true;
using MyLayoutType = conditional_t<flag, NnLayoutFormat<fp8_e8m0_t>, ZzLayoutFormat<fp8_e8m0_t>>;
size_t m = 128;
size_t n = 128;
auto layoutAL1 = MyLayoutType{}(m, n);

// 编译时常量传参构造Layout
using MyZZLayout = ZzLayoutFormat<half>::type<Std::Int<32>, Std::Int<32>>;
auto staticLayout = MyZZLayout{};

// 运行时变量传参构造Layout
ZzLayoutFormat<half> zzFormat;
auto layout = zzFormat(32, 32);
```
