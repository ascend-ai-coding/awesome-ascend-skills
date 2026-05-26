---
name: external-cannbot-ops-ascendc-crash-debug
description: Ascend C 算子卡死/崩溃调试技能。用于处理程序无法运行完的场景：(1) 程序卡死/挂起/超时，Kernel 无响应，(2) 程序崩溃（Segmentation
  Fault、Abort），(3) Buffer 冲突/死锁导致的核心挂起，(4) 需要解析 plog 日志定位卡死/崩溃位置。触发关键词：卡死、挂起、超时、崩溃、hang、crash、deadlock、Segmentation
  Fault、Abort、Kernel hang、内存越界、plog。
original-name: ascendc-crash-debug
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-26'
synced-commit: ac5bbd2b4cf427d011874e11f8d1e8b1bef66eda
license: UNKNOWN
---

# Ascend C 算子卡死/崩溃调试

系统化调试 Ascend C 算子无法运行完的问题，包括程序卡死/挂起、Kernel超时、程序崩溃。

## 快速诊断

```
程序无法运行完
    │
    ├─ 程序崩溃？
    │   ├─ 启用 coredump → ulimit -c unlimited
    │   ├─ 生成 core 文件 → 运行程序
    │   └─ GDB 分析 → gdb <exe> <core> → bt / bt full / info locals
    │
    └─ 程序卡死？
        ├─ 查看 plog 日志 → 定位卡死位置
        ├─ 检查 Buffer 配对 → AllocTensor/FreeTensor
        ├─ 检查同步 → EnQue/DeQue 配对
        └─ Kernel 调试 → AscendC::PRINTF / DumpTensor / msDebug
```

## 症状-原因速查表

| 症状 | 可能原因 | 诊断方向 |
|------|----------|----------|
| **程序卡死/超时** | Buffer 未释放 / 死锁 | 检查 AllocTensor/FreeTensor 配对、EnQue/DeQue 配对 |
| **核心超时/挂起** | Buffer 冲突/死锁 | 检查 Alloc / Free 配对 |
| **Segmentation Fault** | 空指针解引用 / 内存越界 | GDB 分析 coredump → bt 查看调用栈 |
| **Abort** | 断言失败 / 异常终止 | GDB 分析 coredump → 检查断言条件 |
| **栈溢出** | 递归过深 / 大数组 | 检查 Kernel 内数组大小 |
| **aic error** | 内存访问越界 / 非对齐访问 | 检查 DataCopy 长度、32B 对齐 |
| **AIV 卡死** | 跨核同步缺失 | 检查 CrossCoreWaitFlag 对应 SetFlag |
| **drain 卡死** | 未发送 AIV→AIC flag | 检查 flag 发送链路 |

## 按场景进入详细流程

| 场景 | 说明 | 详细步骤 |
|------|------|----------|
| 程序崩溃 | Segmentation Fault / Abort | [Coredump 调试](references/crash_workflow.md) |
| 程序卡死/超时 | Kernel 无响应、挂起 | [Kernel 挂起调试](references/crash_workflow.md) |
| Buffer 冲突/死锁 | Alloc/Free 不配对、同步缺失 | [Buffer 问题排查](references/crash_workflow.md) |

## 详细资源

- **[crash_workflow.md](references/crash_workflow.md)**：完整调试流程 + 调试工具速查
- **[parse_plog.py](scripts/parse_plog.py)**：plog 日志解析脚本（自动识别卡死/崩溃/硬件异常信号）
- **[ascendc-env-check skill](skill:ascendc-env-check)**：环境变量配置指南
