> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/layout/MakeStride.md
# MakeStride

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

构造Stride对象，用于描述各维度在内存中的步长。支持传入多个步长值，也支持传入嵌套的Stride子结构以构造层次化Stride。

## 函数原型

```cpp
template <typename... Ts>
__aicore__ inline constexpr Stride<Ts...> MakeStride(const Ts&... t)
```

## 参数说明

**表 1** 模板参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| Ts... | 输入 | Stride各元素的类型，可以是整型变量、`Std::Int`整型常量，或嵌套的Stride子结构类型。 |

**表 2** 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| t | 输入 | Stride的各维度步长或子结构，可变参数。 |

## 返回值说明

返回`Stride<Ts...>`对象。

## 约束说明

参数`t`需满足可用于构造`Stride<Ts...>`对象。

## 调用示例

```cpp
using namespace AscendC::Te;

// 构造普通Stride
auto stride = MakeStride(1, 100, 200);

// 构造层次化Stride
auto fractalStride = MakeStride(MakeStride(1, 16), MakeStride(32, 512));

auto stride0 = Std::get<0>(stride);                    // stride0 = 1
auto innerStride = Std::get<1>(Std::get<0>(fractalStride)); // innerStride = 16
```
