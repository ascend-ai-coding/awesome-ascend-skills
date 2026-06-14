> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/cube_compute/Mmad.md

# Mmad

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

执行矩阵乘加（C += A * B或C = A * B + Bias）操作。支持`MmadOperation`操作，用于执行基本计算。

## 函数原型

```cpp
template <typename Tp, const Tp& traits, typename T, typename... Params>
__aicore__ inline void Mmad(const MmadAtom<T>& atomMmad, const Params& ...params)

template <typename T, typename... Params>
__aicore__ inline void Mmad(const MmadAtom<T>& atomMmad, const Params& ...params)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| atomMmad | 输入 | [MmadAtom](../struct/atom/MmadAtom.md) 对象，封装了具体的矩阵乘加操作和traits配置。 |
| params | 输入 | 矩阵乘加操作的参数，可变参数。通常包括：<br>- 目的张量C（计算结果矩阵，位于L0C）<br>- 源张量A（左矩阵，位于L0A）<br>- 源张量B（右矩阵，位于L0B）<br>- 参数（如defaultMmadParams）<br>- 可选：源张量Bias（偏置张量，位于L0C或者BiasTable Buffer） |

## 返回值说明

无

## 约束说明

- atomMmad必须是有效的MmadAtom对象；
- params的数量和类型必须与矩阵乘加操作的要求匹配；
- 源操作数和目的操作数必须位于支持的存储空间。

## 调用示例

```cpp
using namespace AscendC::Te;
// 下面为接口3种调用方式的示例
// 调用方式1
Mmad(MmadAtom<MmadTraits<MmadOperation, MmadTraitDefault>>{}, l0CTensor, l0ATensor, l0BTensor, para);
// 调用方式2
MmadAtom<MmadTraits<MmadOperation, MmadTraitDefault>>{}.Call(l0CTensor, l0ATensor, l0BTensor, para);
// 调用方式3
auto atomMmad = MakeMmad(MmadOperation{}, MmadTraitDefault{});
atomMmad.Call(l0CTensor, l0ATensor, l0BTensor, para);

// 带Bias的调用方式
Mmad(MmadAtom<MmadTraits<MmadOperation, MmadTraitDefault>>{}, l0CTensor, l0ATensor, l0BTensor, biasTensor, para);

```

## 相关参考

- [MmadAtom](../struct/atom/MmadAtom.md)
- [MmadTraits](../struct/atom/MmadTraits.md)
