---
name: external-cannbot-tools-tool-npu-check
description: npu-check NPU 运行正确性检查工具使用指导，覆盖检查能力、运行配置、命令用法和报告结论边界。触发：用户明确询问 npu-check、memcheck、synccheck，或要求解读
  npu-check 检查报告。
original-name: tool-npu-check
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-26'
synced-commit: edcd4b94390c2a6b664f0935fbdd8a65e3644e9f
license: UNKNOWN
---

# npu-check 工具能力与报告分析

本 Skill 说明 `npu-check` 的检查能力和运行配置，指导命令执行和检查报告解读。

## 基本术语

- **设备全局内存（GM）**：NPU 核函数能够访问的全局内存。
- **Host/Device**：Host 指运行应用的 CPU 侧；Device 指执行核函数的 NPU 侧。
- **核函数（kernel）**：在 NPU 上执行的函数；一次核函数调用称为一次下发（launch）。
- **输入形状（shape）**：本次执行所使用的输入维度和大小。

## 检查能力

- `memcheck` 检查本次运行中数据搬运指令对 GM 的读写，报告读写方向、访问大小、源码位置、执行核、任务块、
  搬运流水、下发批次和异常地址，可用于定位越界访问及已释放内存访问。
- `synccheck` 检查核内的事件设置与等待（`SET_FLAG -> WAIT_FLAG`），以及缓冲区获取与释放
  （`GET_BUF -> RLS_BUF`）是否精确配对。报告可区分前置操作重复执行、缺少配对操作和两侧执行次数不一致。

两种检查能力的详细报告字段和命令行选项见[检查能力](references/check-capabilities.md)。

## 使用限制

| 项目 | 当前边界 |
| --- | --- |
| 设备与环境 | 仅支持 Ascend 950（芯片架构 `dav-3510`），使用与目标 NPU 和驱动匹配且包含 `npu-check` 的 CANN 包 |
| 应用形态 | 目标程序是可独立运行的可执行文件；PyTorch 场景暂不支持 |
| 调用形态 | 通过 `<<<>>>` NPU 核函数调用语法直接调用一个算子，且只下发一次运行 |
| 执行模型 | 单进程、单线程运行 |
| `memcheck` | 仅检查数据搬运指令的 GM 访问，不支持标量直接访问（scalar）检测 |
| `synccheck` | 仅支持核内同步检查，覆盖 `SET_FLAG/WAIT_FLAG` 和 `GET_BUF/RLS_BUF` 配对 |
| 检测数据导出 | 被测应用需要在核函数下发后调用 `aclrtSynchronizeStream` 或 `aclrtSynchronizeStreamWithTimeout`，以导出并处理检测数据 |
| 非检查范围 | 不检查算子精度、性能、CPU 侧（Host）逻辑和构建配置 |

详细的检查边界、GM 越界判定依据和结论边界见
[能力范围与限制](references/capability-boundaries.md)。

## 常用命令

```bash
# 查看当前版本支持的参数
npu-check --help

# 检查 GM 访问
npu-check --tool memcheck -- <application> [application-args...]

# 检查同步指令配对
npu-check --tool synccheck -- <application> [application-args...]

# 同时启用两种检查
npu-check --tool memcheck --tool synccheck -- <application> [application-args...]

# 将工具报告写入指定文件
npu-check --tool memcheck --log-file <report-path> -- <application> [application-args...]
```

| 参数 | 含义 |
| --- | --- |
| `--tool <name>` | 选择 `memcheck` 或 `synccheck`，可重复指定；未指定时默认启用 `memcheck` |
| `--log-file <path>` | 将检查报告和应用输出写入指定文件；检查报告不再在终端显示，应用输出仍显示；父目录必须已存在，同名文件会被覆盖 |
| `--` | 分隔 `npu-check` 选项与被检查应用命令；应用路径或参数以 `-` 开头时不可省略 |
| `-h`、`--help` | 显示当前版本的命令帮助 |
| `<application> [application-args...]` | 可独立运行的目标应用及其原有参数 |

完整的环境检查、原始日志保存和超时结果处理见[运行 npu-check](references/execution.md)。

## GM 越界检测方式

`memcheck` 根据成功执行的 `aclrtMalloc` 和 `aclrtFree` 记录当前有效的 NPU 内存分配范围，并将申请大小向上按
32 字节对齐后作为检查边界。分配或释放失败时，不更新记录。只有搬运指令的访问范围完整落入一个当前有效的
内存分配范围时，才视为有效访问。
详细的区间定义、释放后访问和覆盖限制见
[GM 越界判定基准](references/capability-boundaries.md#gm-越界判定基准)。

检测数据在被测应用调用 `aclrtSynchronizeStream` 或 `aclrtSynchronizeStreamWithTimeout` 时导出并处理。因此，被测
应用仅下发核函数而不调用任一同步接口时，不能根据没有诊断得出检查通过结论。

## 工作流程

1. **确认能力边界。** 读取[能力范围与限制](references/capability-boundaries.md)，记录目标设备、CANN 环境、应用调用
   形态、启用的检查类型和输入。
2. **确认检查能力。** 读取[检查能力](references/check-capabilities.md)，说明 `memcheck`、`synccheck` 或两者实际
   启用的能力和未覆盖边界。用户已经指定检查类型时遵循其选择；未指定时说明各选项及命令行默认行为，不根据一般
   开发现象主动推荐检查类型。
3. **受控运行。** 检查 `command -v npu-check` 和 `npu-check --help`，保留应用原有工作目录、参数、环境和
   NPU 设备选择，按照[运行 npu-check](references/execution.md)保存合并后的原始捕获日志、执行命令和进程退出
   状态。应用被外部超时或信号终止且 `npu-check` 未收到完整报告时，按运行参考保留原始日志和终止
   信息，将本次检查判定为未完成，不给出检查通过或已发现问题的结论。
4. **判断诊断可信度。** 将原始日志与对应的命令、环境、检查类型和退出状态一起核对，检查来源对应关系、上下文
   完整性、诊断与汇总是否自洽，以及是否存在截断或消息丢失迹象，据此判断为“可靠”“部分可靠”或“不可靠”并
   说明理由。不得通过固定文字或字段是否存在机械核对报告。部分可靠的报告可以保留上下文足以支撑的
   明确诊断，但不能扩展为完整通过结论。具体方法见
   [报告阅读](references/result-interpretation.md)。
5. **处理受限结果。** 报告不完整、自相矛盾、来源不明或命令无法启动时，读取
   [故障排查](references/troubleshooting.md)。说明哪些信息仍可作为证据、哪些结论无法确认；只有改变了明确的
   失败条件后才重试。

## 报告输出

面向用户的结果按以下顺序组织：

1. **执行上下文**：算子、输入形状、应用命令、工作目录、检查类型，以及工具能检查和不能检查的范围。
2. **诊断可信度**：可靠、部分可靠或不可靠，以及支持该判断的来源、完整性、自洽性和缺失信息。
3. **检查结论与关键证据**：明确发现的问题或当前无法确认的结论、应用自身状态、首个相关诊断、源码位置、
   访问或同步上下文和原始日志路径。

```text
执行上下文：<算子、输入形状、应用命令、工作目录和检查类型>
检查范围：<应用命令、检查类型、已验证能力和未覆盖边界>
诊断可信度：<可靠/部分可靠/不可靠及判断理由>
检查结论：<明确问题、未发现明确问题或无法确认，以及应用状态>
关键证据：<检查报告中的诊断、源码位置、访问/同步上下文和原始捕获日志路径>
```

“未发现明确问题”的表述按[能力范围与限制](references/capability-boundaries.md)收敛。只有可靠报告才能支撑本次执行
范围内的通过结论；未实际执行时，不得声称完成了真实 NPU 验证。

## 信息来源

命令参数以当前环境的 `npu-check --help` 为准；检查能力和报告字段以当前版本的 CANN 官方文档及实际输出为准。
不同版本存在差异时，记录工具版本和实际输出；官方资料与本次报告未提供的信息一律标记为未知，不通过分析
`npu-check`、开发工具包或编译器的内部源码补充用户使用结论，也不得补造参数、字段或检查能力。
