> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/pointer/MakeGMmemPtr.md
# MakeGMmemPtr

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

创建GM上的指针，用于访问AI Core的GM的内存空间。

## 函数原型

```cpp
template <typename Iterator>
__aicore__ inline constexpr auto MakeGMmemPtr(Iterator iter)
```

## 参数说明

**表 1** 模板参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| Iterator | 输入 | 迭代器或指针类型。 |

**表 2** 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| iter | 输入 | 指针，指向GM的起始地址，类型为`__gm__ T*`。 |

## 返回值说明

返回GM内存指针对象，类型为`HardwareMemPtr<Hardware::GM, Iterator>`。

## 约束说明

iter必须是有效的迭代器类型。

## 调用示例

```cpp
using namespace AscendC::Te;

// 示例1： 使用指针创建
constexpr uint32_t TILE_LENGTH = 128;
__gm__ float data[TILE_LENGTH];
auto ptr = MakeGMmemPtr(data);
```
