> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/layout/is_layout.md
# is\_layout

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

判断输入的数据结构是否为Layout数据结构，可通过检查其成员常量value的值来判断。当value为true时，表示输入的数据结构是Layout类型；反之则为非Layout类型。

## 函数原型

```cpp
template <typename T> struct is_layout
```

## 参数说明

**表 1**  模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 根据输入的数据类型，判断是否为Layout数据结构。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
using namespace AscendC::Te;

// 初始化Layout数据结构并判断其类型
auto shape = MakeShape(10, 20, 30);
auto stride = MakeStride(1, 100, 200);

auto layoutMake = MakeLayout(shape, stride);
Layout<decltype(shape), decltype(stride)> layoutInit(shape, stride);

bool value = is_layout<decltype(shape)>::value; // value = false
value = is_layout<decltype(stride)>::value; // value = false

value = is_layout<decltype(layoutMake)>::value; // value = true
value = is_layout<decltype(layoutInit)>::value; // value = true
```

