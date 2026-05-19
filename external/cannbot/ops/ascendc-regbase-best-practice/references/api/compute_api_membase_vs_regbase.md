---
title: MemBase 与 RegBase 计算 API 对照
purpose: 对比 LocalTensor 中心的 MemBase 写法与 RegTensor 中心的 RegBase 写法，帮助选择正确 API 层级。
read_when:
  - 设计或代码需要把 MemBase 计算链迁移到 RegBase。
  - 不确定某个 compute API 应该写成 `LocalTensor` vector API 还是 `AscendC::Reg::*` API。
  - 审查发现 `LocalTensor` compute 和 `RegTensor` compute 混用。
not_for:
  - 只判断 GM/UB 搬运对齐。
  - 纯 Host tiling 结构设计。
keywords:
  - membase
  - regbase
  - LocalTensor
  - RegTensor
  - AscendC::Reg
next_reads:
  - regbase_api_reference.md
  - arithmetic_and_reduce.md
  - datacopy_best_practices.md
  - ../pitfalls/regbase_vs_membase_confusions.md
depth: foundation
topic_type: api
---

# MemBase 与 RegBase 计算 API 对照

MemBase 和 RegBase 可以共享 Host、tiling、queue、UB staging 的部分结构，但 compute 核心不是同一套 API surface。迁移或审查时，先判断当前代码在哪一层计算。

| 维度 | MemBase / LocalTensor 写法 | RegBase / RegTensor 写法 |
|---|---|---|
| 计算对象 | `LocalTensor<T>` | `RegTensor<T>` |
| 入口形态 | kernel 普通成员函数或 `Process/Compute` 中调用 vector API | `Compute` 通过 `asc_vf_call` 进入 `__simd_vf__` |
| 输入来源 | `LocalTensor` 已经代表 UB buffer | `__ubuf__ T*` 先经 `Reg::Load*` 进入 `RegTensor` |
| 计算 API | `AscendC::Add/Mul/Exp/...` 等 kernel-side vector API | `AscendC::Reg::Add/Mul/Exp/...` 或 `AscendC::MicroAPI::*` |
| tail 控制 | API 的 count / mask / repeat 参数，依具体接口而定 | `MaskReg` 控制 active lanes |
| 中间值 | 常物化为 UB 上的 `LocalTensor` 临时区 | 尽量保留在短生命周期 `RegTensor` 中 |
| 输出 | 写到 `LocalTensor` 后再 copy out | `Reg::Store*` 写回 UB address 后再 copy out |

## 逐元素计算对照

MemBase 写法通常围绕 `LocalTensor`：

```cpp
// 示意：LocalTensor 上直接调用 vector API。
AscendC::LocalTensor<float> x = inQueue.DeQue<float>();
AscendC::LocalTensor<float> y = outQueue.AllocTensor<float>();
AscendC::Muls(y, x, scale, count);
```

RegBase 写法先进入 VF body，再在 `RegTensor` 上计算：

```cpp
__simd_vf__ inline void ScaleVf(__ubuf__ float* in, __ubuf__ float* out, uint32_t count)
{
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::RegTensor<float> yReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(count);

    AscendC::Reg::LoadAlign(xReg, in);
    AscendC::Reg::Muls(yReg, xReg, scale, mask);
    AscendC::Reg::StoreAlign(out, yReg, mask);
}
```

关键区别：MemBase 的 API 参数通常是 `LocalTensor` 和元素数；RegBase 的 API 参数是 `RegTensor` 和 `MaskReg`。

## 常见 API 映射

这些名称可能相同，但对象和签名不同。不能只按函数名迁移。

| 需求 | MemBase 倾向 | RegBase 倾向 | 审查点 |
|---|---|---|---|
| unary | `Abs(dstLocal, srcLocal, count)`、`Exp(...)` | `Reg::Abs(dstReg, srcReg, mask)`、`Reg::Exp(...)` | RegBase 输入输出必须是 `RegTensor` |
| binary | `Add(dstLocal, aLocal, bLocal, count)` | `Reg::Add(dstReg, aReg, bReg, mask)` | 两个输入寄存器 lane layout 一致 |
| scalar | `Adds` / `Muls` kernel-side vector API | `Reg::Adds(dstReg, srcReg, scalar, mask)` / `Reg::Muls(...)` | 不要为了 scalar 物化 broadcast UB buffer |
| compare | `Compare` 产生 LocalTensor / mask-like 结果，依接口而定 | `Reg::Compare(maskOut, aReg, bReg, mask)` | compare 结果是 `MaskReg` |
| select | `Select` 消费 LocalTensor 或 mask tensor | `Reg::Select(dstReg, trueReg, falseReg, mask)` | true/false source 要同 dtype route |
| cast | kernel-side `Cast` 处理 `LocalTensor` | `Reg::Cast(dstReg, srcReg, mask)` | cast trait、round、saturation 必须验证 |
| reduce | `ReduceSum` / `WholeReduceSum` 等 LocalTensor API | `Reg::Reduce` / `ReduceDataBlock` / `PairReduceElem` | 不要把 LocalTensor reduce 签名套到 VF-side reduce |

## 迁移规则

从 MemBase 迁到 RegBase 时，不是给原 API 加 namespace，而是重写 compute 层：

1. 保留可复用的 Host tiling、GM/UB staging 和 queue 结构。
2. 把原 `LocalTensor` compute 链拆成 `__simd_vf__` 函数。
3. 在 VF 入口把 `__ubuf__` 地址 load 到 `RegTensor`。
4. 使用 `AscendC::Reg::*` / `AscendC::MicroAPI::*` 完成计算链。
5. 用 `MaskReg` 表示 tail，不把 padding 当作有效 lane。
6. store 回 UB，再由外层 `CopyOut` 写回 GM。

## 不要这样写

- 不要在 `__simd_vf__` 里直接调用只接受 `LocalTensor` 的 vector API。
- 不要把 `AscendC::Reg::Add` 当成 `AscendC::Add` 的命名空间替换；它们的对象、mask 和签名不同。
- 不要让 `RegTensor` 跨 tile 或跨 VF 作用域保存状态。
- 不要因为 outer shell 使用 `TQue` / `LocalTensor` 就判定它不是 RegBase；关键看热计算链是否进入 `RegTensor`。
- 不要把设计伪代码中的 `Reg::` API 当成无需验证的可编译签名。

## 相关文档

- [[regbase_api_reference]]
- [[arithmetic_and_reduce]]
- [[datacopy_best_practices]]
- [[../pitfalls/regbase_vs_membase_confusions]]
- [[../dev-experience/regbase_programming_notes]]
