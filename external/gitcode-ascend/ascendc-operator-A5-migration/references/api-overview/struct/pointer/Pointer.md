> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/pointer/Pointer.md
# Pointer

## 功能说明

Pointer用于定义指针迭代器，用于遍历和访问张量数据。

## 结构体定义

```cpp
template <typename T, typename P>
struct Pointer {
    T* ptr;
    P position;
};
```

## 字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| ptr | T* | 指向数据的指针。 |
| position | P | 指针的位置信息。 |

## 约束说明

- ptr指针必须指向有效的内存空间。
- position必须正确描述指针的当前位置。
- Pointer支持多种内存空间的指针类型。

## 调用示例

```cpp
// 创建Global Memory指针
auto gmPtr = AscendC::MakeGMmemPtr(gmTensor);

// 创建Unified Buffer指针
auto ubPtr = AscendC::MakeUBmemPtr(ubTensor);

// 创建L1 Buffer指针
auto l1Ptr = AscendC::MakeL1memPtr(l1Tensor);
```
