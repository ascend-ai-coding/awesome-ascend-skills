---
title: RegBase 构建笔记
purpose: 汇总 RegBase 构建阶段常见 header、宏、模板和路径问题。
read_when:
  - 编译失败但设计看起来合理。
  - 新增 RegBase body 或 VF 函数后出现构建错误。
keywords:
  - build
  - header
  - compile
next_reads:
  - regbase_programming_notes.md
  - ../api/regbase_api_whitelist.md
  - ../pitfalls/api_misuse.md
depth: practice
topic_type: experience
---

# RegBase 构建笔记

## 1. 先判断失败层级

| 失败位置 | 常见原因 |
|---|---|
| Host 编译 | tiling struct、注册宏或 include 路径不匹配 |
| kernel entry | `TILING_KEY_IS`、模板实例或参数列表不一致 |
| RegBase header | API 家族不存在、模板参数错误、dtype 不支持 |
| VF body | `RegTensor` 类型、mask 参数、load/store dist 不匹配 |

## 2. 构建前检查

- RegBase body 所在路径是否与 arch 目录一致，例如 `arch35`。
- include 是否能找到 RegBase API header。
- `__VEC_SCOPE__`、`__simd_vf__`、`asc_vf_call` 的位置是否符合工程模板。
- 模板 dtype 与 tiling route 是否一致。
- Host 和 kernel 的 tiling data struct 是否一致。

## 3. 常见处理方式

- API 不存在：回到 `regbase_api_whitelist.md` 和 SDK 文档确认，不要改成猜测名称。
- 模板报错：先缩小到单 dtype route。
- mask 类型不匹配：检查 `MaskReg` 生成和 API 参数顺序。
- store 报错：检查 `StoreDist`、输出 dtype 和 UB address 类型。

## 相关文档

- [[regbase_programming_notes]]
- [[../api/regbase_api_whitelist]]
- [[../pitfalls/api_misuse]]
