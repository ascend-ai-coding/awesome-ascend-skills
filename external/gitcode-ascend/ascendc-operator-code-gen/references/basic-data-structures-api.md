# AscendC 基础数据结构接口总结

## 一、LocalTensor

**用途**: 存放AI Core内部Local Memory数据，逻辑位置包括VECIN、VECOUT、VECCALC、A1、A2、B1、B2、CO1、CO2。

### 构造与初始化

```cpp
// Pipe框架（不直接调用）
AscendC::LocalTensor<T>() {}

// 静态Tensor编程
AscendC::LocalTensor<T>(TPosition pos, uint32_t addr, uint32_t tileSize)
AscendC::LocalTensor<T>(uint32_t addr)  // 仅支持TensorTrait类型
```

### 核心接口

| 接口 | 功能 | 示例 |
|------|------|------|
| `SetValue(index, value)` | 设置元素值 | `local.SetValue(0, 100)` |
| `GetValue(index)` | 获取元素值 | `auto val = local.GetValue(0)` |
| `operator()(offset)` | 获取元素引用 | `local(0) = 100` |
| `operator[](offset)` | 偏移获取新Tensor | `local[16]` |
| `GetSize()` | 获取元素个数 | `uint32_t size = local.GetSize()` |
| `SetSize(size)` | 设置元素个数 | `local.SetSize(256)` |
| `GetPhyAddr()` | 获取物理地址 | `uint64_t addr = local.GetPhyAddr()` |
| `GetPosition()` | 获取逻辑位置 | `TPosition pos = local.GetPosition()` |
| `ReinterpretCast<T>()` | 类型重解释 | `auto t = local.ReinterpretCast<half>()` |
| `SetShapeInfo(shapeInfo)` | 设置形状信息 | `local.SetShapeInfo(ShapeInfo(...))` |
| `GetShapeInfo()` | 获取形状信息 | `ShapeInfo info = local.GetShapeInfo()` |
| `SetUserTag(tag)` | 设置用户标签 | `local.SetUserTag(10)` |
| `GetUserTag()` | 获取用户标签 | `TTagType tag = local.GetUserTag()` |

### 示例

```cpp
// 分配与使用
AscendC::LocalTensor<half> srcLocal = inQueue.AllocTensor<half>();
AscendC::DataCopy(srcLocal, srcGlobal, 512);
inQueue.EnQue(srcLocal);

// 元素访问
srcLocal.SetValue(0, 1.0f);
auto val = srcLocal.GetValue(0);

// 偏移操作
AscendC::LocalTensor<half> offsetTensor = srcLocal[16];

// 类型转换
AscendC::LocalTensor<int16_t> castTensor = srcLocal.ReinterpretCast<int16_t>();
```

---

## 二、GlobalTensor

**用途**: 存放Global Memory全局数据。

### 核心接口

| 接口 | 功能 | 示例 |
|------|------|------|
| `SetGlobalBuffer(buffer, size)` | 设置缓冲区 | `gm.SetGlobalBuffer((__gm__ half*)ptr, 1024)` |
| `SetGlobalBuffer(buffer)` | 设置缓冲区（无size） | `gm.SetGlobalBuffer((__gm__ half*)ptr)` |
| `GetPhyAddr()` | 获取地址 | `const __gm__ T* addr = gm.GetPhyAddr()` |
| `GetValue(offset)` | 获取元素值 | `auto val = gm.GetValue(0)` |
| `SetValue(offset, value)` | 设置元素值 | `gm.SetValue(0, 1.0f)` |
| `operator()(offset)` | 获取元素引用 | `gm(0) = 1.0f` |
| `operator[](offset)` | 偏移获取新Tensor | `gm[256]` |
| `GetSize()` | 获取元素个数 | `uint64_t size = gm.GetSize()` |
| `SetShapeInfo(shapeInfo)` | 设置形状信息 | `gm.SetShapeInfo(...)` |
| `GetShapeInfo()` | 获取形状信息 | `ShapeInfo info = gm.GetShapeInfo()` |
| `SetL2CacheHint(mode)` | 设置L2缓存提示 | `gm.SetL2CacheHint<CacheRwMode::RW>(mode)` |

### 示例

```cpp
AscendC::GlobalTensor<half> srcGlobal;
srcGlobal.SetGlobalBuffer((__gm__ half*)srcGm, dataSize);

// 读取
auto val = srcGlobal.GetValue(0);

// 偏移访问
AscendC::GlobalTensor<half> offsetGlobal = srcGlobal[128];

// DataCopy
AscendC::DataCopy(srcLocal, srcGlobal, dataSize);
```

---

## 三、Layout

**用途**: 描述多维张量内存布局，包含Shape和Stride。

### 原型

```cpp
template <typename ShapeType, typename StrideType>
struct Layout {
    __aicore__ inline constexpr Layout(const ShapeType& shape = {}, const StrideType& stride = {});
    __aicore__ inline constexpr decltype(auto) GetShape();
    __aicore__ inline constexpr decltype(auto) GetStride();
    template <typename CoordType>
    __aicore__ inline constexpr auto operator()(const CoordType& coord) const;
};
```

### 构造方法

```cpp
#include "kernel_operator_layout.h"

// Shape构造
auto shape = AscendC::MakeShape(4, 2);  // 4行2列

// Stride构造
auto stride = AscendC::MakeStride(4, 1);  // 行步长4, 列步长1

// Layout构造
auto layout = AscendC::MakeLayout(shape, stride);

// 通过坐标计算内存索引
auto coord = AscendC::MakeCoord(1, 0);  // 第1行第0列
auto idx = layout(coord);  // 计算得到地址索引
```

### 示例：4行2列矩阵

| 地址 | 0 | 1 | 2,3 | 4 | 5 | 6,7 | 8 | 9 |
|------|---|---|-----|---|---|------|---|---|
| 元素 | a00 | a01 | - | a10 | a11 | - | a20 | a21 |

Shape: (4, 2), Stride: (4, 1)

---

## 四、Coordinate

**用途**: 表示张量多维坐标，配合Layout使用计算内存索引。

### 原型

```cpp
template <typename... Coords>
using Coord = Std::tuple<Coords...>;
```

### 接口

```cpp
// 构造坐标
auto coord = AscendC::MakeCoord(row, col);

// 坐标转内存索引
template <typename CoordType, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto Crd2Idx(const CoordType& coord,
                                         const Layout<ShapeType, StrideType>& layout);
```

### 示例

```cpp
auto shape = AscendC::MakeShape(4, 2);
auto stride = AscendC::MakeStride(4, 1);
auto layout = AscendC::MakeLayout(shape, stride);

auto coord = AscendC::MakeCoord(1, 0);  // row=1, col=0
auto idx = AscendC::Crd2Idx(coord, layout);  // idx = 4
```

---

## 五、TensorTrait

**用途**: 描述Tensor的完整信息（数据类型、逻辑位置、Layout），用于编译期优化。

### 原型

```cpp
template <typename T, TPosition pos = TPosition::GM,
          typename LayoutType = Layout<Shape<>, Stride<>>>
struct TensorTrait {
    using LiteType = T;
    using LiteLayoutType = LayoutType;
    static constexpr const TPosition tPos = pos;

    __aicore__ inline LayoutType& GetLayout();
    __aicore__ inline void SetLayout(const LayoutType& t);
};
```

### 构造方法

```cpp
#include "kernel_operator_tensor_trait.h"

auto shape = AscendC::MakeShape(16, 16, 16);
auto stride = AscendC::MakeStride(0, 0, 0);
auto layout = AscendC::MakeLayout(shape, stride);

// 构造TensorTrait
auto tensorTrait = AscendC::MakeTensorTrait<float, AscendC::TPosition::VECIN>(layout);

// 使用TensorTrait类型构造LocalTensor
AscendC::LocalTensor<decltype(tensorTrait)> tensor(addr);
```

### 支持的数据类型

`int4b_t`, `uint8_t`, `int8_t`, `int16_t`, `uint16_t`, `bfloat16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `float`, `half`

### 约束

- 同一接口不支持同时输入TensorTrait类型和非TensorTrait类型的Tensor
- TensorTrait类型的Tensor不包含ShapeInfo信息
- DataCopy切片接口不支持TensorTrait类型

---

## TPosition 逻辑位置

| 位置 | 说明 |
|------|------|
| VECIN | 向量计算输入 |
| VECOUT | 向量计算输出 |
| VECCALC | 向量计算中间结果 |
| A1 | 矩阵计算A矩阵输入L1 |
| A2 | 矩阵计算A矩阵L0A |
| B1 | 矩阵计算B矩阵输入L1 |
| B2 | 矩阵计算B矩阵L0B |
| CO1 | 矩阵计算输出L0C |
| CO2 | 矩阵计算输出UB |
| GM | 全局内存 |
