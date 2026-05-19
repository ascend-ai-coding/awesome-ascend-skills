---
title: RegBase 编程笔记
purpose: 编写 RegBase 代码前的实用提醒，覆盖参考实现阅读、数据流、buffer 分域、VF 边界、同步边界和伪代码到可编译代码的差异。
read_when:
  - 开发者即将根据 DESIGN.md 实现 RegBase 算子。
  - 需要确认 `__simd_vf__` / `asc_vf_call` 结构。
  - 需要把 Host/Kernel/UB/VF 分层落到可实现的代码结构。
keywords:
  - programming
  - dataflow
  - buffer
  - sync
  - __simd_vf__
  - asc_vf_call
next_reads:
  - regbase_build_notes.md
  - ../api/regbase_api_reference.md
  - ../api/datacopy_best_practices.md
  - ../api/regbase_api_sync.md
  - ../reference-ops/open_source_operator_table.md
depth: practice
topic_type: experience
---

# RegBase 编程笔记

## 1. 从真实实现落到工程模板

DESIGN.md 中的 RegBase 伪代码只表达层级、API 家族和数据流。实现时必须按工程模板、真实 API 签名和参考实现调整。

编码前先通过 [[../reference-ops/open_source_operator_table]] 定位候选目录，并确认目录中存在 `op_kernel/arch35/*regbase*`、`*_apt.cpp`、`__VEC_SCOPE__`、`__simd_vf__`、`RegTensor` 或 `AscendC::MicroAPI::*` 等 RegBase 证据。阅读顺序建议固定为：

1. kernel entry：确认 `TILING_KEY_IS(...)`、dtype route 和 kernel 参数列表。
2. `Init`：确认 GM/UB binding、pipe、queue 或 buffer 分配。
3. `Process`：确认 tile loop、tail 和 `CopyIn -> Compute -> CopyOut` 顺序。
4. VF body：确认 `RegTensor`、`MaskReg`、`LoadDist`、`StoreDist` 和 API family。

参考实现只能证明某种结构可行，不能替代当前任务的 shape、dtype、tail 和 API 适配判断。

## 2. 推荐结构

- Kernel outer shell 管理 `Init`、`Process`、`CopyIn`、`Compute`、`CopyOut`。
- UB `Compute` 层准备 address、count、mask。
- VF 函数通过 `__simd_vf__` 表达寄存器级 body。
- UB `Compute` 层通过 `asc_vf_call` 调用 VF 函数。
- `AscendC::Reg::*` API 在 SDK header 中常声明为 `__simd_callee__`；这是被 VF body 调用的 primitive / helper 形态，不应替代新算子的 `__simd_vf__` 入口。

标准数据流分两层：

```text
GM input
  -> CopyIn: DataCopy/DataCopyPad 到 UB tile
UB input tile
  -> Compute: 取 UB address，调用 asc_vf_call 或进入 __VEC_SCOPE__
RegTensor input
  -> AscendC::Reg::* / AscendC::MicroAPI::* 计算
RegTensor output
  -> Store* 回 UB output tile
UB output tile
  -> CopyOut: DataCopy/DataCopyPad 回 GM output
```

`DataCopy` / `DataCopyPad` 属于 GM/UB 搬运，不能写进 `__simd_vf__`。VF 内的 `LoadAlign` / `StoreAlign` 是 UB address 与 `RegTensor` 之间的搬运，具体签名和边界见 [[../api/datacopy_best_practices]]。

## 3. Buffer 分域

RegBase buffer 规划按域写清楚，不按“看起来都是 tensor”混在一起：

| 域 | 常见对象 | 用途 |
|---|---|---|
| GM | `GlobalTensor`、workspace、tiling | 输入、输出、全局临时数据 |
| UB | `TQue`、`TBuf`、`LocalTensor`、raw `__ubuf__` pointer | tile staging、copy、临时 scratch |
| 寄存器 | `RegTensor<T>`、`MaskReg` | VF 操作数、mask、cast/compare/select 中间值 |

规划顺序：

1. 先确定每个 tile 的输入和输出 UB 大小。
2. 再确认中间 UB scratch 是否真的需要。
3. 再列出 VF 内 `RegTensor` 临时变量。
4. 最后计算总 UB 使用量、对齐方式和 double buffer 开销。

只在一个 VF 链内使用的中间值优先留在 `RegTensor`。需要跨 VF loop、跨 stage 或在 `CopyOut` 前以 tile 形式存在的数据才放 UB。

## 4. VF 内常见对象

- `RegTensor<T>`：输入、输出和中间寄存器，生命周期不跨 VF 作用域或 tile。
- `MaskReg`：tail、compare 和 select mask，不是同步 primitive。
- `LoadDist` / `StoreDist`：UB/寄存器之间的 load/store 分布，不是 barrier。
- `AscendC::Reg::*` 或 `AscendC::MicroAPI::*`：VF-safe 算术和辅助 API。
- raw UB pointer 在文档或 SDK 声明里常见为 `__ubuf__ T*`，在部分参考实现里也会看到 `__local_mem__ T*`；后者在 CANN 9.0.0 `reg_compute` 公共头中是 `__ubuf__` 的别名。

## 5. 同步边界

同步先按层判断：

- UB pipeline 层：`EnQue` / `DeQue`、`CopyIn -> Compute -> CopyOut` 顺序或必要的 pipe fence 表达 stage handoff。
- VF 层：只有 VF local scratch 出现真实 producer / consumer 顺序风险时才考虑 `AscendC::Reg::LocalMemBar<src, dst>()`。
- 高级 pipeline 层：`SetFlag` / `WaitFlag`、`PipeBarrier`、`SyncAll` 或 cross-core flag 只能作为 outer shell 机制，不能写进 `__simd_vf__` 的默认 callable surface。

普通 `LoadAlign -> Add/Mul/Exp -> StoreAlign` 的寄存器链不需要 barrier。`LocalMemBar` 的 `MemType` 组合和使用边界见 [[../api/regbase_api_sync]]。

## 6. 编码检查

- `CopyIn`、`Compute`、`CopyOut` 的 ownership 清楚。
- VF body 只处理 UB/寄存器，不直接处理 GM。
- `RegTensor` 不跨 VF 作用域或 tile 保存。
- mask 生成与当前 tile 有效长度一致。
- cast 链条和输出 dtype 一致。
- store 回 UB 后再由 `CopyOut` 写 GM。
- UB 大小计算包含 scratch、对齐和 double buffer 开销。
- 同步机制能对应真实层级和生产者 / 消费者关系。
- 如果 API 参数不确定，先查文档或参考实现，不试错编造。

## 相关文档

- [[regbase_build_notes]]
- [[../api/regbase_api_reference]]
- [[../api/datacopy_best_practices]]
- [[../api/regbase_api_sync]]
- [[../reference-ops/open_source_operator_table]]
