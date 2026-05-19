---
title: RegBase API 白名单
purpose: 作为 RegBase API 的进入门禁，区分普通 kernel-side API 和 VF-function-safe `AscendC::Reg::*` 调用。
read_when:
  - 编码或审查前必须确认某个 API family 是否允许。
  - 主要问题是某个调用属于 kernel shell、UB staging 还是 VF function body。
not_for:
  - 性能调优。
  - 选择现有参考算子。
keywords:
  - whitelist
  - vf function
  - AscendC::Reg
  - AscendC::MicroAPI
  - allowed api
next_reads:
  - regbase_api_reference.md
  - compute_api_membase_vs_regbase.md
  - ../dev-experience/regbase_programming_notes.md
  - ../pitfalls/api_misuse.md
depth: foundation
topic_type: api
---

# RegBase API 白名单

本文件是 RegBase direct-invoke 工作的 API 门禁。它把普通 RegBase kernel-side API 与 VF-function-safe API 分开，避免设计和审查把 `LocalTensor` 路径假设混进 `AscendC::Reg::*` reg-compute 语义。开源 kernel 中同一组 VF-safe API 经常写成 `AscendC::MicroAPI::*`；如果某个 API 不在下面的列表中，先按“不可用”处理，直到在安装的 SDK header 或真实参考实现中验证过。

## Header 参考根目录

使用这些 SDK 逻辑根目录做首要验证来源。不要把环境相关绝对路径写进设计或审查结论。

- `<cann-root>/aarch64-linux/asc/include/basic_api/`
  普通 kernel-side AscendC 接口，例如 `DataCopy`、vector `LocalTensor` compute、barrier、event sync。
- `<cann-root>/aarch64-linux/asc/include/basic_api/reg_compute/`
  `AscendC::Reg::*` 公开接口，以及可在 VF function 中调用的 support types。
- `<cann-root>/aarch64-linux/asc/include/adv_api/`
  任务相关的高阶 RegBase helper，例如 norm、pad、transpose、select、quantization 等 family。
- `<cann-root>/aarch64-linux/asc/impl/basic_api/kernel_macros.h`
  定义 `namespace MicroAPI = Reg;`。很多开源 kernel 因此把同一组 VF-safe API 写成 `AscendC::MicroAPI::*`。

相关文档可能用环境变量或逻辑 include root 作为简写。把它视为同一组 header，不要理解成第二套 API 表面。

## 作用域规则

- 普通 RegBase kernel-side API 和 VF-function API 不能互换，即使名字相似。
- 设计进入 VF function 后，优先把 `AscendC::Reg::*` 和 `reg_compute/` 下的接口作为 callable surface。
- 阅读开源代码时，除非本地代码另加 wrapper，否则把 `AscendC::MicroAPI::*` 视为与 `AscendC::Reg::*` 相同的 VF-safe API family。
- 新算子的 VF 入口通常写成 `__simd_vf__` 并由 `asc_vf_call` 调用；`__simd_callee__` 主要出现在 SDK `reg_compute/` API 声明或内部 VF helper 上。
- `__simd_vf__` 内可以调用 `reg_compute/` 下声明为 `__simd_callee__` 的 `AscendC::Reg::*` 接口，但不能把任意 VF helper 当成普通可调用工具随意嵌套。
- 传入 VF helper 的 pointer 参数必须保持在 `__ubuf__` 地址空间。
- 同一能力如果同时有 kernel-side 与 VF-function-safe 两种形态，编码前必须确认 namespace、参数样式和 mask/control contract。

## 普通 RegBase Kernel-Side API

这些 family 用于普通 RegBase kernel 代码，不是 VF function 的默认白名单。

| Family | 代表 API | VF Function? | 主要 Header Root |
|---|---|---|---|
| Vector compute | `Abs`, `Exp`, `Relu`, `Sqrt`, `Rsqrt`, `Reciprocal`, `Ln`, `Log`, `Sigmoid`, `Tanh`, `LeakyRelu`, `Ceil`, `Floor`, `Round` | No | `basic_api/` |
| Binary compute | `Add`, `Sub`, `Mul`, `Div`, `Max`, `Min`, `And`, `Or`, `Xor`, `Axpy`, `FusedMulAdd`, `AddRelu`, `SubRelu` | No | `basic_api/` |
| Scalar helpers | `Adds`, `Muls`, `Divs`, `Maxs`, `Mins`, `ShiftLefts`, `ShiftRights` | No | `basic_api/` |
| Compare and select | `Compare`, `Compares`, `Select` | No | `basic_api/` |
| Reduction | `ReduceSum`, `WholeReduceSum`, `BlockReduceSum`, `ReduceMax`, `WholeReduceMax`, `BlockReduceMax`, `ReduceMin`, `WholeReduceMin`, `BlockReduceMin`, `ReduceMean`, `ReduceProd`, `ReduceAll`, `ReduceAny` | No | `basic_api/` |
| Conversion and utility | `Cast`, `ReinterpretCast`, `Duplicate`, `Arange`, `CreateVecIndex`, `Interleave`, `DeInterleave`, `Move` | No | `basic_api/` |
| Data movement | `DataCopy`, `DataCopyPad`, `DataCopyGather`, `DataCopyUnAlignPre`, `DataCopyUnAlign`, `DataCopyUnAlignPost`, `Copy` | No | `basic_api/` |
| Sync and event | `PipeBarrier`, `SetFlag`, `WaitFlag`, `CrossCoreSetFlag`, `CrossCoreWaitFlag`, `SyncAll` | No | `basic_api/` |

## VF-Function-Safe API (`AscendC::Reg::*`)

进入 VF function 或 reg-compute helper 路径后，这些 family 是优先 callable surface。权威来源是 `reg_compute/`。

开源 RegBase kernel 经常把同一组 API 写成 `AscendC::MicroAPI::*`。把它看作 alias 风格差异，不要当成第二套白名单。

| Family | 代表 API | VF Function? | 主要 Header Root |
|---|---|---|---|
| Unary reg compute | `Abs`, `Relu`, `Exp`, `Sqrt`, `Ln`, `Log`, `Log2`, `Log10`, `Neg`, `Not` | Yes | `basic_api/reg_compute/` |
| Binary reg compute | `Add`, `Sub`, `Mul`, `Div`, `Max`, `Min`, `And`, `Or`, `Xor`, `MulAddDst`, `Mull` | Yes | `basic_api/reg_compute/` |
| Scalar reg compute | `Adds`, `Muls`, `Divs`, `Maxs`, `Mins`, `ShiftLefts`, `ShiftRights` | Yes | `basic_api/reg_compute/` |
| Compare and select | `Compare`, `Compares`, `Select` | Yes | `basic_api/reg_compute/` |
| Reg reduction | `Reduce`, `ReduceDataBlock`, `PairReduceElem` | Yes | `basic_api/reg_compute/` |
| Reg load and store | `LoadAlign`, `StoreAlign`, `LoadUnAlignPre`, `LoadUnAlign`, `StoreUnAlign`, `StoreUnAlignPost`, `Load`, `Store`, `Gather`, `GatherB`, `Scatter` | Yes | `basic_api/reg_compute/` |
| Mask and copy helpers | `MaskReg`, `CreateMask`, `UpdateMask`, `Pack`, `UnPack`, `Move`, `LocalMemBar` | Yes | `basic_api/reg_compute/` |
| VF 路径 traits / enums | `LoadDist`, `StoreDist`, `MemType`, `CastTrait`, `CMPMODE`, `RoundMode`, `MaskMergeMode`, `PostLiteral`, `DataCopyMode`, `ReduceType`, `PairReduce` | Yes | `basic_api/reg_compute/` |

## 高阶任务相关 RegBase API

这些 family 存在于 SDK 中，RegBase 工作中可能有用，但不是默认 primitive。使用前必须确认任务适配性、tiling 假设和精确签名。

| Family | 代表 API | VF Function? | 主要 Header Root |
|---|---|---|---|
| Norm and mean | `LayerNorm`, `RmsNorm`, `GroupNorm`, `Normalize`, `Mean` | No | `adv_api/` |
| Pad and broadcast | `Pad`, `Broadcast`, `Brcb` | No | `adv_api/` |
| Structured select | `Select` bytes-mask overload | No | `adv_api/` |
| Transpose and layout change | `TransData`, `Transpose` confusion-transpose overload | No | `adv_api/` |
| Quantization | `AscendQuant`, `AscendDequant`, `AscendAntiQuant`, `Quantize`, `Dequantize`, `AntiQuantize` | No | `adv_api/` |

## VF 专项护栏

- 进入 VF function 后，把 `AscendC::Reg::*` 作为默认 API surface；源码使用 `AscendC::MicroAPI::*` 时，按同一 VF-safe family 理解。
- 不要假设 kernel-side `LocalTensor` API 与 VF-side API 拥有相同签名、mask contract 或地址假设。
- 代码路径出现 `RegTensor`、`MaskReg`、`LoadDist`、`StoreDist` 时，先检查 VF-capable header。
- 在 VF helper 内使用 `CreateMask`、`UpdateMask`、`Pack`、`UnPack` 前，重新确认 mask 创建和更新规则。
- 如果一个 helper 需要再调用另一个 VF helper，暂停并回查设计。新算子的主 VF body 优先保持 `__simd_vf__` + `asc_vf_call` 结构；VF body 内稳定的 callable surface 是 `reg_compute/` 下声明为 `__simd_callee__` 的 `AscendC::Reg::*` API family。

## 非可编译结构示例

以下只表达层级和 API surface，不替代 SDK 签名：

```cpp
// kernel shell / UB staging: 普通 kernel-side API
DataCopy(ubIn, gmIn, copyLen);

// VF body: 进入 reg_compute callable surface
__VEC_SCOPE__ {
    RegTensor<float> x;
    MaskReg mask;
    // 优先使用 AscendC::Reg::*；开源代码中可能写作 AscendC::MicroAPI::*。
    // 具体参数顺序、模板参数和 mask contract 必须查 reg_compute header。
}
```

## 使用护栏

- 编写代码或生成设计前先检查本文件。
- 白名单检查后，用 [[regbase_api_reference]] 确认 broad family 和心智模型。
- 需要判断 `LocalTensor` compute 是否应迁移成 `RegTensor` compute 时，用 [[compute_api_membase_vs_regbase]] 做层级对照。
- 问题属于 barrier、flag、cross-core coordination 时，用 [[regbase_api_sync]]，不要只看 API 可用性。
- alignment 不确定时优先考虑 `DataCopyPad`；不要假设 `DataCopy` 能吸收 misaligned GM transfer。
- 如果任务需要未列出的 API，先验证 SDK header family 并更新设计，不要补猜测 API。

## 相关文档

- [[regbase_api_reference]]
- [[compute_api_membase_vs_regbase]]
- [[regbase_api_sync]]
- [[datacopy_best_practices]]
- [[arithmetic_and_reduce]]
- [[../pitfalls/precision_guide]]
