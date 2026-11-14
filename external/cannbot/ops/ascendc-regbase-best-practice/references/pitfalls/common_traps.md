---
title: 常见陷阱
purpose: 汇总 RegBase 设计和实现中的高频错误，作为审查前快速检查表。
read_when:
  - 设计缺少分层或实现出现不稳定问题。
  - 需要快速判断是否偏离 RegBase 主路径。
keywords:
  - traps
  - review
  - layer
next_reads:
  - api_misuse.md
  - regbase_vs_membase_confusions.md
  - precision_failures.md
depth: foundation
topic_type: pitfall
---

# 常见陷阱

## 1. 分层陷阱

- 把整个 kernel 写成一个大 VF 函数。
- 把 GM/UB copy 和寄存器计算写在同一层，不说明 ownership。
- 在 `Process` 里直接堆寄存器级数学，导致 `CopyIn -> Compute -> CopyOut` 不可读。
- 只写 “RegBase path”，没有说明 Host / Kernel / UB / VF 边界。

## 2. API 陷阱

- 编造 `AscendC::Reg::*` 签名。
- 把 MemBase / LocalTensor API 当成 RegBase VF API。
- 没有检查 header 或 SDK 文档。
- API 参数顺序和 mask 位置靠猜。

## 3. tail / mask 陷阱

- copy 层处理了 tail，但 VF store 没有 mask。
- mask 使用对齐长度而不是有效长度。
- compare mask 和 store mask 混用但语义不同。
- padding 值进入数学结果。
- `UpdateMask<T>(remaining)` 通过引用推进 `remaining`；随后手工递减同一变量会
  double-decrement，导致跳过 slice 或 tail。
- 使用 `UpdateMask<T>(active)` 时，`active` 也可能被更新；索引和
  `remaining` 必须基于调用前保存的 `step` 或独立的循环索引，不能读取更新后的
  局部值来推导下一次偏移。
- 一个进度变量只能有一个推进者；先查当前 SDK 的 `UpdateMask` header，再确定
  由 API 或外层循环负责推进。

## 4. dtype / precision 陷阱

- fp16/bf16 不说明是否升 fp32。
- cast 回输出 dtype 的位置不清楚。
- reduce accumulation dtype 不明确。
- quant/dequant 缺少 scale、rounding、saturate 说明。

## 5. 同步陷阱

- 用 `SyncAll` 修本地 stage ordering。
- 把 `MaskReg` 当同步。
- 没有参考实现就加 `SetFlag` / `WaitFlag`。
- cross-core flag 用来补本地 UB 交接。

## 相关文档

- [[api_misuse]]
- [[regbase_vs_membase_confusions]]
- [[precision_failures]]
