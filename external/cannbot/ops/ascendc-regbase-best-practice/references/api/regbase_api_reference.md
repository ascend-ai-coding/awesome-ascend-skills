---
title: RegBase API 参考
purpose: API 可用性和 VF 边界已缩小后，提供 RegBase 核心类型、作用域、调用 family 和数据搬运 family 的紧凑签名地图。
read_when:
  - 已确认任务进入 RegBase，需要家族级签名概览。
  - 需要在查 header 前映射名称、作用域和调用家族。
not_for:
  - 替代 SDK header 做穷尽签名验证。
  - 顶层路线选择。
keywords:
  - api reference
  - signatures
  - RegTensor
  - LoadDist
next_reads:
  - regbase_api_whitelist.md
  - compute_api_membase_vs_regbase.md
  - arithmetic_and_reduce.md
  - ../dev-experience/regbase_programming_notes.md
depth: foundation
topic_type: api
---

# Ascend C RegBase API 参考

本文件是直接调用工作中的 RegBase 签名地图。它用于建立方向感；最终可用性、参数顺序、模板参数和 dtype 限制仍必须回到白名单、SDK header 和真实参考实现验证。

## 范围

- 覆盖 `DAV_3510` RegBase 优先的直接调用工作中常见的 kernel 侧 API。
- 区分普通 kernel 侧 API 与 VF 函数安全的 `AscendC::Reg::*` API。
- 说明开源 kernel 经常把同一组 VF-safe surface 写成 `AscendC::MicroAPI::*`，因为 `MicroAPI` 是 `Reg` 的 SDK alias。
- 把 MemBase 兼容作为上下文，不把 `LocalTensor` / `TQue` workflow 当作 RegBase API 的来源。
- 聚焦设计和实现审查中最常出现的 core types 与 call families。

## 核心类型和作用域标记

| 名称 | 定义 / 用途 | 关键边界 |
|---|---|---|
| `RegTensor<T>` | 寄存器级 tensor storage，用于 VF 内部操作数和中间值 | 生命周期应局限在 VF body 或 reg-compute helper 内 |
| `MaskReg` | vector operation 的 mask register | 控制 active lanes、tail、compare/select，不是同步 primitive |
| `__VEC_SCOPE__ { ... }` | RegBase compute scope | 表明进入 VF / register compute 语义 |
| `LoadDist` / `StoreDist` | UB 与 register object 之间的 distributed load/store 策略 | 描述 load/store distribution，不是 barrier |
| `CastTrait` | cast 行为控制，包含 `RegLayout`、`SatMode`、`MaskMergeMode`、`RoundMode` 等 | 用于 dtype route 和 cast 语义 |
| `MemType` | `LocalMemBar` 的本地 memory 访问类别 | 区分 `VEC_STORE`、`VEC_LOAD`、`SCALAR_STORE`、`SCALAR_LOAD` 等本地访问顺序 |

重要 header 位置：

- CANN 9.0.0 中，VF 函数安全的可调用家族位于 `<cann-root>/aarch64-linux/asc/include/basic_api/reg_compute/`。
- 普通 kernel 侧可调用家族位于 `<cann-root>/aarch64-linux/asc/include/basic_api/`。
- 高阶 helper 位于 `<cann-root>/aarch64-linux/asc/include/adv_api/`。
- 很多真实 kernel include `kernel_operator.h` 并写 `AscendC::MicroAPI::*`；alias 在 `<cann-root>/aarch64-linux/asc/impl/basic_api/kernel_macros.h` 中定义。
- 不同 SDK 包可能暴露不同逻辑 include root；结论应以当前 SDK 实际 header 为准，不要只按旧路径字符串判断 API 是否存在。

## 计算 API 家族

常见 RegBase compute call 包括：

- unary math：`Abs`、`Exp`、`Relu`、`Sqrt`、`Ln`、`Log`
- binary math：`Add`、`Sub`、`Mul`、`Div`、`Max`、`Min`
- scalar helpers：`Adds`、`Muls`
- compare / select：`Compare`、`Compares`、`Select`
- type conversion / packing：`Cast`、`Pack`、`UnPack`
- VF 归约 helper：`Reduce`、`ReduceDataBlock`、`PairReduceElem`

使用顺序是：先查 [[regbase_api_whitelist]]，再用本文件确认大类调用家族。特别要通过白名单判断当前路径需要普通 kernel 侧 API，还是 VF 函数安全的 `AscendC::Reg::*` API；源码中后者可能写作 `AscendC::MicroAPI::*`。

## Reg Compute 代表签名

以下签名来自 CANN 9.0.0 `basic_api/reg_compute/` header，用于提示参数形态。它们不是穷尽清单；写代码前仍要回到当前 SDK header 检查 dtype、模板默认值和平台条件。

这里的 `__simd_callee__` 是 SDK header 里 RegBase primitive / helper 的声明宏，不是新算子 VF 入口的推荐写法。编写算子时，外层 `Compute` 通常通过 `asc_vf_call<...>` 调用一个 `__simd_vf__` 函数；该 `__simd_vf__` 函数内部再调用下面这些 `AscendC::Reg::*` API。只有在拆内部 VF helper 时，才会看到类似 SDK 参考实现里的 `__simd_callee__` helper。

典型层级：

```cpp
__simd_vf__ inline void MyOpVf(__ubuf__ float* in, __ubuf__ float* out, uint32_t count)
{
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(count);
    AscendC::Reg::LoadAlign(xReg, in);
    // AscendC::Reg::* API 在 SDK header 中声明为 __simd_callee__。
    AscendC::Reg::StoreAlign(out, xReg, mask);
}
```

### 二元 / 标量

来源：`kernel_reg_compute_vec_binary_intf.h`、`kernel_reg_compute_vec_binary_scalar_intf.h`。

```cpp
template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Add(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask);

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Sub(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask);

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Mul(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask);

template <typename T = DefaultType, auto mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Div(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask);

template <typename T = DefaultType, typename U, MaskMergeMode mode = MaskMergeMode::ZEROING, typename S>
__simd_callee__ inline void Adds(S& dstReg, S& srcReg, U scalarValue, MaskReg& mask);

template <typename T = DefaultType, typename U, MaskMergeMode mode = MaskMergeMode::ZEROING, typename S>
__simd_callee__ inline void Muls(S& dstReg, S& srcReg, U scalarValue, MaskReg& mask);
```

### 比较 / 选择

来源：`kernel_reg_compute_vec_cmpsel_intf.h`。

```cpp
template <typename T = DefaultType, CMPMODE mode = CMPMODE::EQ, typename U>
__simd_callee__ inline void Compare(MaskReg& dst, U& srcReg0, U& srcReg1, MaskReg& mask);

template <typename T = DefaultType, CMPMODE mode = CMPMODE::EQ, typename U, typename S>
__simd_callee__ inline void Compares(MaskReg& dst, U& srcReg, S scalarValue, MaskReg& mask);

template <typename T = DefaultType, typename U>
__simd_callee__ inline void Select(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask);
```

### 转换 / 归约

来源：`kernel_reg_compute_vec_vconv_intf.h`、`kernel_reg_compute_vec_reduce_intf.h`。

```cpp
template <typename T = DefaultType, typename U = DefaultType,
          const CastTrait& trait = castTrait, typename S, typename V>
__simd_callee__ inline void Cast(S& dstReg, V& srcReg, MaskReg& mask);

template <typename T = DefaultType, RoundMode roundMode = RoundMode::CAST_NONE,
          MaskMergeMode mode = MaskMergeMode::ZEROING, typename S>
__simd_callee__ inline void Truncate(S& dstReg, S& srcReg, MaskReg& mask);

template <ReduceType type = ReduceType::SUM, typename T = DefaultType, typename U = DefaultType,
          MaskMergeMode mode = MaskMergeMode::ZEROING, typename S, typename V>
__simd_callee__ inline void Reduce(S& dstReg, V srcReg, MaskReg mask);

template <ReduceType type = ReduceType::SUM, typename T = DefaultType,
          MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void ReduceDataBlock(U& dstReg, U srcReg, MaskReg mask);

template <PairReduce type = PairReduce::SUM, typename T = DefaultType,
          MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void PairReduceElem(U& dstReg, U srcReg, MaskReg mask);
```

### 本地 memory barrier

来源：`kernel_reg_compute_membar_intf.h`。

```cpp
template <MemType src, MemType dst>
__simd_callee__ inline void LocalMemBar();
```

`LocalMemBar` 是 VF 内部本地 memory 顺序接口，常见于 `RegTensor -> local scratch -> RegTensor` 或非对齐 VF load/store 辅助状态复用场景。它不是 queue handoff、event flag 或跨核同步。使用规则见 [[regbase_api_sync]]。

## 数据搬运 API 家族

RegBase 直接调用路径也会用到少量 VF 侧 UB/寄存器数据搬运家族：

- `LoadDist`
- `StoreDist`
- `LoadAlign` / `StoreAlign`
- `LoadUnAlignPre` / `LoadUnAlign`
- `StoreUnAlign` / `StoreUnAlignPost`
- `Load` / `Store`
- `Gather` / `GatherB` / `Scatter`

来源：`kernel_reg_compute_datacopy_intf.h`。

```cpp
template <typename T = DefaultType, LoadDist dist = LoadDist::DIST_NORM, typename U>
__simd_callee__ inline void LoadAlign(U& dstReg, __ubuf__ T* srcAddr);

template <typename T = DefaultType, StoreDist dist = StoreDist::DIST_NORM, typename U>
__simd_callee__ inline void StoreAlign(__ubuf__ T* dstAddr, U& srcReg, MaskReg& mask);

template <typename T = DefaultType, DataCopyMode dataMode, typename U>
__simd_callee__ inline void LoadAlign(U& dstReg, __ubuf__ T* srcAddr,
                                      uint32_t dataBlockStride, MaskReg& mask);

template <typename T = DefaultType, DataCopyMode dataMode, typename U>
__simd_callee__ inline void StoreAlign(__ubuf__ T* dstAddr, U& srcReg,
                                       uint32_t dataBlockStride, MaskReg& mask);

template <typename T = DefaultType, typename U>
__simd_callee__ inline void Load(U& dstReg, __ubuf__ T* srcAddr);

template <typename T = DefaultType, typename U>
__simd_callee__ inline void Store(__ubuf__ T* dstAddr, U& srcReg);

template <typename T = DefaultType, typename U>
__simd_callee__ inline void Store(__ubuf__ T* dstAddr, U& srcReg, uint32_t count);

template <typename T0 = DefaultType, typename T1, typename T2 = DefaultType, typename T3, typename T4>
__simd_callee__ inline void Gather(T3& dstReg, __ubuf__ T1* baseAddr, T4& index, MaskReg& mask);

template <typename T = DefaultType, typename U = DefaultType, typename S, typename V>
__simd_callee__ inline void Scatter(__ubuf__ T* baseAddr, S& srcReg, V& index, MaskReg& mask);
```

GM/UB 搬运细节优先查 [[datacopy_best_practices]]，不要把 kernel 侧 `DataCopyPad` 与 VF 侧 UB/寄存器 load/store 混成同一层。

## 非可编译结构示例

以下片段只说明层级，不保证参数顺序或模板签名：

```cpp
// UB staging 已经准备好 inAddr/outAddr。
__VEC_SCOPE__ {
    RegTensor<float> vin;
    RegTensor<float> vout;
    MaskReg mask;

    // load: UB address -> RegTensor，LoadDist 决定分布策略。
    // AscendC::Reg::LoadAlign<float, LoadDist::DIST_NORM>(vin, inAddr);

    // compute: 使用 AscendC::Reg::* 或源码中的 AscendC::MicroAPI::*。
    // AscendC::Reg::Abs(vout, vin, mask);

    // store: RegTensor -> UB address，StoreDist 决定写回策略。
    // AscendC::Reg::StoreAlign<float, StoreDist::DIST_NORM>(outAddr, vout, mask);
}
```

在 DESIGN.md 中，不能只写“调用 RegBase API”。至少写出：输入寄存器、输出寄存器、mask 来源、load/store 策略和文档依据。

## 决策规则

1. 先检查 [[regbase_api_whitelist]]。
2. 用本文件确认 broad RegBase call family 和正确心智模型。
3. 问题涉及 VF 内部本地 memory 顺序、readiness、stall 或 cross-core coordination 时，读 [[regbase_api_sync]]。
4. 任务可能改变 precision 或 cast timing 时，读 [[../pitfalls/precision_guide]]。
5. 如果任务实际依赖 MemBase-only workflow，把它视为 scope change，不要强行套 RegBase 语义。

## 相关文档

- [[regbase_api_whitelist]]
- [[compute_api_membase_vs_regbase]]
- [[regbase_api_sync]]
- [[datacopy_best_practices]]
- [[pipeline_and_buffer]]
- [[arithmetic_and_reduce]]
- [[../pitfalls/precision_guide]]
