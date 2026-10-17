---
name: external-cannbot-ops-ascendc-crash-debug
description: Ascend C 算子卡死/崩溃/内存错误调试路由技能。用于处理程序无法运行完或执行异常崩溃的场景：(1) 程序卡死/挂起/超时，Kernel
  无响应，(2) 程序崩溃（Segmentation Fault、Abort），(3) Buffer 冲突/死锁导致的核心挂起，(4) 需要解析 plog 日志定位卡死/崩溃位置，(5)
  偶发崩溃/结果异常（怀疑内存越界踩踏），需要主动检测内存错误。触发关键词：卡死、挂起、超时、崩溃、hang、crash、deadlock、Segmentation
  Fault、Abort、Kernel hang、内存越界、plog、memcheck、内存检查、内存错误、偶发崩溃、aic error、非法读写。
original-name: ascendc-crash-debug
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-17'
synced-commit: fe65c77485d42a93c7e45f3e67ab96d95504b9dc
license: UNKNOWN
---

# Ascend C 算子卡死/崩溃调试

系统化调试 Ascend C 算子无法运行完的问题，包括程序卡死/挂起、Kernel超时、程序崩溃。

## 快速诊断

```
程序无法运行完或偶发崩溃
    │
    ├─ 程序崩溃（Segmentation Fault / Abort）？
    │   ├─ 启用 coredump → ulimit -c unlimited
    │   ├─ 生成 core 文件 → 运行程序
    │   └─ GDB 分析 → gdb <exe> <core> → bt / bt full / info locals
    │
    ├─ 程序卡死/超时？
    │   ├─ 查看 plog 日志 → 定位卡死位置
    │   ├─ 检查 Buffer 配对 → AllocTensor/FreeTensor
    │   ├─ 检查同步 → EnQue/DeQue 配对
    │   └─ Kernel 调试 → AscendC::PRINTF / DumpTensor / msDebug
    │
    ├─ 日志包含 AIC/AIV Error、EXCP DS、ERR_INFO 或异常 PC？
    │   ├─ 先保留同一次故障的 kernelName、core type 和完整寄存器；TilingKey 已知时记录
    │   └─ 需要定位具体指令 → npu-objdump-toolkit 异常 PC 精确反汇编
    │
    └─ 偶发崩溃/怀疑内存错误导致异常？
        └─ 内存错误主动检测 → mssanitizer
            ├─ 越界读写（Illegal Read/Write）
            ├─ 多核踩踏（Multi-core Overwrite）
            ├─ 非对齐访问（Misaligned Access）
            └─ 内存泄漏（Memory Leak）
```

## 症状-原因速查表

| 症状 | 可能原因 | 诊断方向 |
|------|----------|----------|
| **程序卡死/超时** | Buffer 未释放 / 死锁 | 检查 AllocTensor/FreeTensor 配对、EnQue/DeQue 配对 |
| **核心超时/挂起** | Buffer 冲突/死锁 | 检查 Alloc / Free 配对 |
| **Segmentation Fault** | 空指针解引用 / 内存越界 | GDB 分析 coredump → bt 查看调用栈；如堆栈不清晰 → [mssanitizer 内存检测](references/memcheck/) |
| **Abort** | 断言失败 / 异常终止 | GDB 分析 coredump → 检查断言条件 |
| **栈溢出** | 递归过深 / 大数组 | 检查 Kernel 内数组大小 |
| **aic/aiv error** | 内存访问越界 / 非对齐访问 / 非法指令 | 检查 DataCopy 长度、32B 对齐；已有 EXCP DS、ERR_INFO 或异常 PC 且需要具体指令时，使用 [npu-objdump-toolkit](skill:npu-objdump-toolkit) 的异常 PC 精确反汇编；需要主动检测内存错误时使用 [mssanitizer](references/memcheck/) |
| **AIV 卡死** | 跨核同步缺失 | 检查 CrossCoreWaitFlag 对应 SetFlag |
| **drain 卡死** | 未发送 AIV→AIC flag | 检查 flag 发送链路 |
| **偶发崩溃无法复现** | 内存踩踏 / 竞争条件 | [mssanitizer 内存检测](references/memcheck/)（加 -g 编译选项定位） |

## 按场景进入详细流程

| 场景 | 说明 | 详细步骤 |
|------|------|----------|
| 程序崩溃 | Segmentation Fault / Abort | [Coredump 调试](references/crash_workflow.md) |
| 程序卡死/超时 | Kernel 无响应、挂起 | [Kernel 挂起调试](references/crash_workflow.md) |
| Buffer 冲突/死锁 | Alloc/Free 不配对、同步缺失 | [Buffer 问题排查](references/crash_workflow.md) |
| 内存错误主动检测 | 越界读写/多核踩踏/非对齐访问/内存泄漏/偶发崩溃 | [mssanitizer 内存检测](references/memcheck/) |
| AIC/AIV 异常指令定位 | 已有 AIC/AIV Error 日志、EXCP DS/ERR_INFO 或异常 PC，需要查看对应指令 | [npu-objdump-toolkit](skill:npu-objdump-toolkit) 的异常 PC 精确反汇编 |

## 执行检查点

1. **确认故障身份**：固定同一次失败的 device、进程、kernelName、时间戳和输入摘要，TilingKey 已知时一并记录，避免混用不同故障记录。
2. **按症状选择流程**：主机进程崩溃走 coredump，Kernel 卡死走 plog/同步检查，疑似踩踏走 mssanitizer，AIC/AIV Error 需要具体指令时调用反汇编工具。
3. **控制写操作范围**：日志和已有 coredump 默认只读；新建结果目录、重新构建、插桩或运行检测脚本时只使用用户指定的工程、device 和输出目录，并记录改动。
4. **复核结论**：修复后使用原始最小复现输入回归，确认故障消失且没有新增精度或同步问题；单次未复现只能记为结果，不能证明根因成立。

故障日志、coredump、输入输出和内存报告可能包含业务数据。验证报告只保留定位所需的最小片段、路径、shape 和哈希，不提交原始数据、token、密钥或凭据。

## 修改验证

```bash
bash -n ops/ascendc-crash-debug/scripts/run_memcheck_pre.sh
python3 -m py_compile ops/ascendc-crash-debug/scripts/parse_plog.py
INCREMENTAL_SKILLS=ascendc-crash-debug bash tests/unit/skills/test-structure.sh
INCREMENTAL_SKILLS=ascendc-crash-debug bash tests/unit/skills/test-content.sh
git diff --check
```

验证应覆盖 Segmentation Fault、Kernel 卡死、内存踩踏和 AIC/AIV Error 四类路由；其中 AIC/AIV Error 用例需要确认会在已有 ERR_INFO/异常 PC 且用户要求具体指令时进入 `npu-objdump-toolkit`，普通稳定精度误差和 561xxx Kernel 查找错误不进入该分支。

## 详细资源

- **[crash_workflow.md](references/crash_workflow.md)**：完整调试流程 + 调试工具速查
- **[parse_plog.py](scripts/parse_plog.py)**：plog 日志解析脚本（自动识别卡死/崩溃/硬件异常信号）
- **[mssanitizer 内存检测](references/memcheck/)**：内存错误检测完整工作流
  - [自动化工作流指南](references/memcheck/automated_workflow.md) — 3步完成编译-安装-检测
  - [用户使用指南](references/memcheck/README.md) — 配置文件模板 + 常见问题
  - [msSanitizer 工具指南](references/memcheck/mssanitizer_guide.md) — 6类内存异常详解
- **[自动化检测脚本](scripts/run_memcheck_pre.sh)**：一键执行内存检测
- **[配置文件模板](scripts/memcheck_input.json.template)**：memcheck_input.json 模板
- **[ascendc-env-check skill](skill:ascendc-env-check)**：环境变量配置指南
