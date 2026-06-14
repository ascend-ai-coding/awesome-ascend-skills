> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/atom/MmadAtom.md
# MmadAtom

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

MmadAtom用于定义矩阵乘加原子操作，封装了矩阵乘加操作的所有信息。

## 结构体定义

```cpp
template <typename... Args>
struct MmadAtom {
    std::tuple<Args...> value;
};
```

## 字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| value | std::tuple<Args...> | 存储矩阵矩阵乘加操作参数的元组。 |

## 约束说明

- Args的数量和类型必须与矩阵乘加操作的要求匹配。
- MmadAtom通常与Mmad函数配合使用。

## 调用示例

```cpp
// 创建MmadAtom
auto mmadAtom = AscendC::Te::MakeMmad(arg1, arg2, arg3);

// 执行矩阵乘加操作
AscendC::Te::Mmad(mmadAtom, dst, src0, src1);
```
