> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/atom/CopyTraits.md
# CopyTraits

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

CopyTraits用于定义搬运操作的特性，包括搬运操作类型和相关参数。

## 结构体定义

```cpp
template <typename CopyOperation, typename... CopyOpArgs>
struct CopyTraits {
    using OperationType = CopyOperation;
    std::tuple<CopyOpArgs...> args;
};
```

## 字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| OperationType | CopyOperation | 复制操作类型。 |
| args | std::tuple<CopyOpArgs...> | 复制操作参数。 |

## 约束说明

- CopyOperation必须是有效的复制操作类型。
- CopyOpArgs的数量和类型必须与CopyOperation的要求匹配。

## 调用示例

```cpp
using namespace AscendC::Te;
// 以CopyGM2L1操作类型为例，创建CopyTraits
using copyTraits = CopyTraits<CopyGM2L1, DataCopyTraitDefault>;
// 使用copyTraits声明CopyAtom对象，调用Copy接口实现CopyGM2L1，其中l1ATensor是位于L1上的目的操作数，globalA是位于GM上的源操作数。
Copy(CopyAtom<copyTraits>{}, l1ATensor, globalA);
```