> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/pointer/MakeFixbufmemPtr.md
<!-- codespell:ignore MakeFixbufmemPtr FIXBUF Fixbufmem -->
# MakeFixbufmemPtr

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

创建FIXBUF上的指针，用于访问AI Core的FIXBUF的内存空间。

## 函数原型

```cpp
template <typename Iterator>
__aicore__ inline constexpr auto MakeFixbufmemPtr(Iterator iter)

template <typename T, typename U>
__aicore__ inline auto MakeFixbufmemPtr(const U& byteOffset)
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
| iter | 输入 | 指针，指向FIXBUF的起始地址，类型为`__fbuf__ T`。 |
| byteOffset | 输入 | FIXBUF内存起始地址的字节偏移量。 |

## 返回值说明

返回FIXBUF内存指针对象，类型为`HardwareMemPtr<Hardware::FIXBUF, Iterator>`。

## 约束说明

- iter必须是有效的迭代器类型。
- 偏移地址必须在FIXBUF内存范围内。

## 调用示例

```cpp
using namespace AscendC::Te;

// 示例1： 使用指针创建
constexpr uint32_t Tile_LENGTH = 128;
__fbuf__ float data[Tile_LENGTH];
auto ptr = MakeFixbufmemPtr(data);

// 示例2： 使用地址字节偏移创建
uint32_t byteOffset = 128;
auto ptr = MakeFixbufmemPtr<float>(byteOffset);
```
