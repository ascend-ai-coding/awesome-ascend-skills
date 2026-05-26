---
name: external-cannbot-ops-ascendc-runtime-debug
description: Ascend C 算子运行时错误调试技能。用于处理算子运行时问题：(1) aclnn 返回错误码（161xxx/361xxx/561xxx，包括环境配置、Tiling、Kernel
  查找等错误），(2) 需要解析 plog 日志定位问题。触发关键词：运行时错误、错误码、Tiling错误、Kernel查找失败、环境变量、plog。
original-name: ascendc-runtime-debug
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-26'
synced-commit: ac5bbd2b4cf427d011874e11f8d1e8b1bef66eda
license: UNKNOWN
---

# Ascend C 算子运行时错误调试

系统化调试 Ascend C 算子运行时错误，包括错误码处理、环境检查。

## 快速诊断

```
运行时错误
    │
    ├─ 返回码非0？
    │   ├─ 161xxx → 参数错误 → 见高频错误速查
    │   ├─ 361xxx → Runtime错误 → 见 debug_workflow.md
    │   ├─ 507035 → 向量核异常 → 见 debug_workflow.md#507035-向量核异常
    │   └─ 561xxx → 内部错误 → 见高频错误速查 / debug_workflow.md
    │
    └─ 复杂场景 → 开启日志调试
```

## 高频错误速查

| 错误码 | 类型 | 排查方向 | 详细方案 |
|-------|------|---------|---------|
| 507035 | 向量核异常 | 检查 DataCopyPad 32B对齐 / UB溢出 | [debug_workflow.md](references/debug_workflow.md#507035-向量核异常) |
| 161xxx | 参数错误 | 检查 dtype/shape/nullptr | [debug_workflow.md](references/debug_workflow.md#161xxx---参数错误) |
| 561002 | Tiling错误 | 检查 TilingKey/TilingFunc | [debug_workflow.md](references/debug_workflow.md#561002---tiling错误) |
| 561003 | Kernel未找到 | 检查算子安装/环境配置 | [debug_workflow.md](references/debug_workflow.md#561003---kernel查找失败) |

更多错误码见 [error_codes.md](references/error_codes.md)

## 未知错误码处理

遇到速查表中未列出的错误码时，见 [debug_workflow.md](references/debug_workflow.md#未知错误码处理)

## 调试工具速查

| 工具 | 用途 | 使用 |
|-----|------|------|
| `aclGetRecentErrMsg()` | 获取错误详情 | 返回码非0时调用 |
| 环境检查 | ascendc-env-check skill | 见 skill:ascendc-env-check |
| `parse_plog.py` | 日志解析 | `python3 scripts/parse_plog.py [plog_file]` |
| `ASCEND_SLOG_PRINT_TO_STDOUT=1` | 日志打屏 | 实时查看日志 |

详细调试方法见 [debug_workflow.md](references/debug_workflow.md#调试工具速查)

## 详细资源

- **[debug_workflow.md](references/debug_workflow.md)**：详细调试流程（错误码处理、环境检查）
- **[kernel_binary_debug.md](references/kernel_binary_debug.md)**：Kernel 二进制构建调试（编译缓存、SEL匹配、多版本冲突、opParaSize）
- **[error_codes.md](references/error_codes.md)**：完整错误码表（基本状态码、内部异常状态码）
- **[ascendc-env-check skill](skill:ascendc-env-check)**：环境变量配置指南
