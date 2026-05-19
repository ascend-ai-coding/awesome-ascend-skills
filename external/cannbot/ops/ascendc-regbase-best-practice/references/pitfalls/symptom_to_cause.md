---
title: 症状到原因
purpose: 在修复循环中从可观察症状快速定位 RegBase 检查路径。
read_when:
  - 构建、运行或精度失败，但原因不清。
keywords:
  - symptom
  - cause
  - debug
next_reads:
  - api_misuse.md
  - precision_failures.md
  - ../dev-experience/regbase_build_notes.md
depth: foundation
topic_type: pitfall
---

# 症状到原因

| 症状 | 优先原因 | 下一步 |
|---|---|---|
| 编译找不到 API | API 名或 header 不正确 | [[api_misuse]]、[[../api/regbase_api_whitelist]] |
| 模板实例化失败 | dtype route 或 `RegTensor<T>` 不匹配 | 查 API 签名和 dtype 限制 |
| 运行越界 | tiling count、UB offset 或 tail store 错 | 查 [[../dev-experience/tiling_review_notes]] |
| 输出全 0 | store 未执行、mask 全 `false`、输出 UB 未入队 | 查数据流和 mask |
| 只有 tail 错 | `MaskReg` 或有效长度错 | 查 [[precision_failures]] |
| 性能异常差 | GM/UB 往返多、barrier 过重、fusion 未生效 | 查 [[../dev-experience/regbase_performance_practices]] |
| 代码看似 RegBase 但审查不过 | MemBase / RegBase 混用 | 查 [[regbase_vs_membase_confusions]] |

## 使用方式

先用症状表确定第一条检查路径，再回到 API、模式或开发经验文档。不要在没有定位层级前盲目改代码。

## 相关文档

- [[api_misuse]]
- [[precision_failures]]
- [[../dev-experience/regbase_build_notes]]
