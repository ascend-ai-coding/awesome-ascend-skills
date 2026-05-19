---
title: 算术与归约
purpose: 选择标量、广播式、归约 API 家族，并记录 repeat 和 shape 约束。
read_when:
  - 正在 `Duplicate + Add`、标量 helper、repeat 广播形态或归约家族之间选择。
  - 算子数学已经明确，但算术或归约 API 家族不清楚。
not_for:
  - 构建或打包问题。
  - 高层 kernel structuring。
keywords:
  - arithmetic
  - reduce
  - repeat limits
  - scalar helpers
next_reads:
  - compute_api_membase_vs_regbase.md
  - regbase_api_whitelist.md
  - ../pitfalls/precision_guide.md
depth: intermediate
topic_type: api
---

# 算术与归约

本卡片合并两类通常决定 RegBase kernel 形状的 API 家族：标量 / 广播算术，以及本地或模式化归约。

## 算术选择

单行或标量化工作中，优先使用标量 helper：

- 使用 `Adds`，不要默认写成 `Duplicate + Add`。
- 使用 `Muls`，不要默认写成 `Duplicate + Mul`。

重复行广播时，使用带 `BinaryRepeatParams` 的 repeat 形态，并设置 `src1RepStride = 0`，让一个源操作数被复用，而不是物化广播缓冲。

## Reg Compute 代表签名

本节只列 RegBase VF 侧常用形态，签名来自 CANN 9.0.0 `basic_api/reg_compute/` header。它用于避免只写 API 名称；最终 dtype 支持、模板默认值和平台条件仍以当前 SDK header 为准。

签名中的 `__simd_callee__` 保留 SDK 原样，表示这些 API 可被 VF body 调用；新算子的 VF 入口仍应按工程模板写成 `__simd_vf__` 并通过 `asc_vf_call` 进入。

### 标量 helper

来源：`kernel_reg_compute_vec_binary_scalar_intf.h`。

```cpp
template <typename T = DefaultType, typename U, MaskMergeMode mode = MaskMergeMode::ZEROING, typename S>
__simd_callee__ inline void Adds(S& dstReg, S& srcReg, U scalarValue, MaskReg& mask);

template <typename T = DefaultType, typename U, MaskMergeMode mode = MaskMergeMode::ZEROING, typename S>
__simd_callee__ inline void Muls(S& dstReg, S& srcReg, U scalarValue, MaskReg& mask);
```

用法边界：

- `dstReg` 与 `srcReg` 是 register object，通常是 `RegTensor<T>` 或同类 reg container。
- `scalarValue` 是标量操作数，不要为了标量加/乘先物化广播缓冲。
- `mask` 控制 active lanes；tail 应在调用前通过 `MaskReg` 准备好。

### 二元算术

来源：`kernel_reg_compute_vec_binary_intf.h`。

```cpp
template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Add(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask);

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Sub(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask);

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Mul(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask);

template <typename T = DefaultType, auto mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Div(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask);
```

用法边界：

- 两个输入寄存器必须来自同一 VF lane 语义，不要把 `LocalTensor` 直接传入 VF 侧 API。
- `MaskMergeMode::ZEROING` 这类模板参数影响 inactive lane 的合并语义，设计文档应写清楚 tail 结果是否需要保留旧值。

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

用法边界：

- `Compare` / `Compares` 的结果是 `MaskReg`，它可以继续控制后续 `Select` 或 store，不是 event sync。
- `CMPMODE` 必须在设计中明确，例如等于、大于、小于等；不要只写“比较得到 mask”。
- `Select` 的两个源寄存器要有同一 dtype 路线和 lane 布局。

## Repeat 限制

很多 vector repeat 形态 API 的 `repeatTime` 是 `uint8_t`，实际限制是 255。

如果某个 tile 可能超过这个限制，应在进入 kernel 循环前拆分工作，而不是让 repeat 计数器溢出。

## Reduce 选择

Reduction 在 RegBase 中需要同时考虑 reduce 维度、tile 切分、partial result 和 dtype accumulation。不要只写一个 `ReduceSum` 名称就结束设计。

设计前先回答四个问题：

1. reduce 维度是哪一个，内存是否连续。
2. 一个 tile 能否覆盖完整 reduce 维度。
3. partial result 存在 register、UB 还是 GM。
4. accumulation dtype 是什么，最终何时 cast。

根据 shape 和 alignment 选择 reduce family：

- row-local reduction：例如 softmax、layernorm 风格的行内归约，选择 level-2 reduce。
- cross-row 或 batch-oriented reduction：输入已经对齐且 pattern 合适时，选择 pattern reduce。

RegBase VF 侧归约的代表签名来自 `kernel_reg_compute_vec_reduce_intf.h`：

```cpp
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

`ReduceType` / `PairReduce` 决定 sum、max、min 等归约语义；`MaskReg` 决定参与归约的有效 lanes。不要把普通 kernel 侧 `ReduceSum` / `WholeReduceSum` 的 `LocalTensor` 形态直接套到 VF 侧 `Reduce` 上。

常见方案：

| 场景 | 方案 |
|---|---|
| reduce 维度在一个 tile 内 | VF 内 reduce，结果写回 UB |
| reduce 维度跨多个 tile | 每 tile 生成 partial，再在 UB 或后续 pass 合并 |
| reduce 后继续 elementwise | 尽量在寄存器或 UB 中保留 partial，减少 GM 往返 |
| mean / variance | sum 与 count 要同时明确，除法和 epsilon 在精度策略里说明 |

tail 和 mask 要求：

- reduce tail 必须用 `MaskReg` 控制无效通道。
- padding 值必须符合 reduce 语义，例如 max 的 padding 不能影响结果。
- partial result 合并时要说明是否包含 tail count。

## Level-2 Reduce 规则

- temporary buffer type 必须与被 reduce 的数据类型匹配。
- `count` 是 effective elements 数量，不是 padded row length。
- row offset 应使用 padded row stride，不是 logical row length。

## Pattern Reduce 规则

- source shape 必须使用 aligned columns。
- 优先使用 explicit shared temporary buffer form。
- 如果接口要求 temporary space，调用前必须预留。
- 只有 shape 与 alignment 已经受控时才使用 pattern reduce。

## Compare 约束

`Compare` 不是任意 shape 的自由回退。它用于兼容路径时，工作区必须 padding，使被比较的 `count` 占据 256-byte 对齐区间。

- 调用 `Compare` 前，把 working buffer pad 到 256-byte boundary。
- compare span 按 padded size 对齐，不按 logical unpadded count。
- compare 结果产生后，只 copy out effective elements。
- 当 compare 语义依赖 neutral element 时，padding 要使用明确 extreme values。

经验规则：

- ArgMax-style compare：padding 使用 active dtype 的最小可表示值。
- ArgMin-style compare：padding 使用 active dtype 的最大可表示值。

## 非可编译结构示例

```cpp
// scalar adjustment: 优先找 scalar helper，mask 覆盖有效 lanes。
// AscendC::Reg::Adds<float>(dstReg, srcReg, alpha, mask);

// repeated row-wise broadcast: 用 repeat form 复用一侧 operand。
// BinaryRepeatParams params;
// params.src1RepStride = 0;
// Add(dst, src0, src1, repeatTime, params);

// VF 侧 reduction: mask 表示有效 lanes，ReduceType 表示归约语义。
// AscendC::Reg::Reduce<ReduceType::SUM>(dstReg, srcReg, mask);
```

具体参数顺序、模板参数和 mask 约定必须回到当前 CANN 版本的 SDK 文档或 header 验证。

## 实用决策树

1. 只是对单行做 scalar adjustment？用 `Adds` 或 `Muls`。
2. 是重复 row-wise broadcast？用 repeat form，并保持 `src1RepStride = 0`。
3. 是 row-local reduction？用 level-2 reduce API。
4. 是 cross-row 或 patterned batch reduction？在 shape aligned 且 temporary buffer 明确时用 pattern reduce。

## 常见错误

- scalar helper 一条指令能完成时，仍使用 `Duplicate`。
- 忘记 `repeatTime` 可能溢出。
- reduce API 期望 effective count，却传入 padded count。
- 在 unaligned shape 上使用 pattern reduce，并期待 API 自动修复。
- API 要求 `dst` 与 `tmpBuffer` 分离时，复用同一 buffer。
- 没有说明 reduce 是否跨 tile。
- accumulation dtype 不明确。
- padding 值可能改变 max/min/sum 语义。
- 把 reduction API 当作已存在但没有文档依据。

## 相关文档

- [[regbase_api_reference]]
- [[compute_api_membase_vs_regbase]]
- [[regbase_api_whitelist]]
- [[pipeline_and_buffer]]
- [[../pitfalls/precision_guide]]
