> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/pointer/MakeBiasmemPtr.md
# MakeBiasmemPtr

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

创建BiasTable Buffer上的指针，用于访问AI Core的BiasTable Buffer的内存空间。

## 函数原型

```cpp
template <typename Iterator>
__aicore__ inline constexpr auto MakeBiasmemPtr(Iterator iter)

template <typename T, typename U>
__aicore__ inline auto MakeBiasmemPtr(const U& byteOffset)
```

## 参数说明

**表 1** 模板参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| Iterator | 输入 | 迭代器或指针类型。 |
| T | 输入 | 元素类型。 |
| U | 输入 | 字节偏移量类型。 |

**表 2** 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| iter | 输入 | 指针，指向BiasTable Buffer的起始地址，类型为`__biasbuf__ T*`。 |
| byteOffset | 输入 | BiasTable Buffer内存起始地址的字节偏移量。 |

## 返回值说明

返回BiasTable Buffer内存指针对象，类型为`HardwareMemPtr<Hardware::BIAS, Iterator>`。

## 约束说明

- iter必须是有效的迭代器类型。
- 偏移地址必须在BiasTable Buffer内存范围内。

## 调用示例

```cpp
using namespace AscendC::Te;

// 示例1： 使用指针创建
constexpr uint32_t Tile_LENGTH = 128;
__biasbuf__ float data[Tile_LENGTH];
auto ptr = MakeBiasmemPtr(data);

// 示例2： 使用地址字节偏移创建
uint32_t byteOffset = 256;
uint32_t offset = 128;
auto basePtr = MakeBiasmemPtr<float>(byteOffset);
auto offsetPtr = basePtr + offset;
```
