---
title: RegBase 与 MemBase 混淆
purpose: 区分 RegBase VF/寄存器数据流与 MemBase/LocalTensor 数据流，避免混用 API 和设计语言。
read_when:
  - 设计或代码同时出现 `LocalTensor` compute 和 `RegTensor` compute。
  - 审查怀疑工作流走错分支。
keywords:
  - RegBase
  - MemBase
  - LocalTensor
next_reads:
  - ../api/regbase_api_reference.md
  - ../dev-experience/regbase_programming_notes.md
  - common_traps.md
depth: foundation
topic_type: pitfall
---

# RegBase 与 MemBase 混淆

## 1. 核心区别

| 维度 | RegBase | MemBase / LocalTensor 路径 |
|---|---|---|
| compute 核心 | `__VEC_SCOPE__` 内 `RegTensor` / `MaskReg` | `LocalTensor` 上调用 vector API |
| 中间值 | 尽量留在寄存器 | 常物化到 UB tensor |
| 有效通道控制 | `MaskReg` | API count / mask 语义依具体 API |
| 数据流语言 | UB address -> 寄存器 -> UB address | LocalTensor -> LocalTensor |

## 2. 可以共存的部分

RegBase outer shell 可以使用 `TPipe`、`TQue`、`LocalTensor` 做 UB staging。这不等于 MemBase。关键在于 `Compute` 的数学核心是否进入寄存器级 VF body。

## 3. 不应混用的部分

- 在 VF body 内调用只适用于 `LocalTensor` 的 API。
- 用 `LocalTensor` 中间值替代本应短生命周期的 `RegTensor`。
- 设计文档一边说 `asc_vf_call`，一边给出 MemBase API 伪代码。
- 审查时看到 `TQue` 就否定 RegBase。

## 4. 修复方式

- 重新画 Host / Kernel / UB / VF 分层。
- 把 UB staging 和 VF compute 分开写。
- API 映射表标注每个 API 所在层。
- 对不确定 API 回到白名单和参考实现验证。

## 相关文档

- [[../api/regbase_api_reference]]
- [[../dev-experience/regbase_programming_notes]]
- [[common_traps]]
