---
name: external-cannbot-tools-tool-npu-check
description: 使用 npu-check 检查 Ascend 算子的内存访问、同步配对、初始化和访问竞争，运行工具并解读报告。适用于明确请求 npu-check、memcheck、synccheck、initcheck、racecheck，或
  Ascend 算子定位任务中已有非法访问、同步不配对、未初始化读取或冲突访问线索的场景。仅有普通数值精度偏差、性能问题或 Host 侧异常时不直接触发。
original-name: tool-npu-check
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-11-21'
synced-commit: 21b65a874d1827cff8ecdeee79288fc60ab64095
license: UNKNOWN
---

# npu-check 工具能力与报告分析

本 Skill 说明 `npu-check` 的检查能力和运行配置，指导命令执行和检查报告解读。

## 任务入口

- 仅询问用法或解释已有报告时，按问题读取对应参考，不启动应用。
- 用户要求算子检查，或定位任务中已有线索指向受支持的问题时，依据
  [检查类型选择](references/check-capabilities.md#检查类型选择)确定工具；输入和运行条件齐备后，在当前任务范围内执行。
- 只有工程目录时，先只读查看 README 和现有构建、运行入口。应用应能独立启动并执行到目标算子，
  不要求待诊断算子成功结束；构建、依赖或启动失败与算子执行异常分别处理。
- 单凭“结果异常”或“卡死”不能确定检查类型；先核对异常上下文，不自动运行全部检查。

## 基本术语

- **设备全局内存（GM）**：NPU 核函数能够访问的全局内存。
- **片上内存（UB/L1/L0）**：核内的存储空间；UB 按执行模式及本次 launch 布局检查，L1、L0A、L0B、L0C 按对应空间容量检查。
- **Host/Device**：Host 指运行应用的 CPU 侧；Device 指执行核函数的 NPU 侧。
- **核函数（kernel）**：在 NPU 上执行的函数；一次核函数调用称为一次下发（launch）。
- **输入形状（shape）**：本次执行所使用的输入维度和大小。

## 检查能力

- `memcheck` 检查本次运行中已支持指令的 GM 访问、L1/L0 容量边界，以及普通 SIMD、SIMD VF 的 UB 分区边界和 SIMT VF 的 UB 写入。
  GM 以有效分配范围为基准，可定位越界、无效地址和释放后访问；L1/L0 以对应容量为基准，UB 使用分区规则。
  当前也支持部分 scalar 访问，不能概括为“仅 GM 搬运”或“全部 scalar 均不支持”。
- `synccheck` 检查核内的事件设置与等待（`SET_FLAG -> WAIT_FLAG`），以及缓冲区获取与释放
  （`GET_BUF -> RLS_BUF`）是否精确配对，并支持受支持的跨核同步模式。报告可区分前置操作重复执行、
  缺少配对操作和两侧执行次数不一致；跨核检查仍受同步模式和核拓扑限制。
- `initcheck` 检查受支持指令的持久化寄存器依赖，以及 GM、UB、L1、L0A、L0B、L0C 的未初始化读取；UB 包括普通 SIMD 和 SIMD VF，按实际字节访问跟踪。
- `racecheck` 检查同一次核函数下发内缺少必要同步的冲突内存访问，包括读写竞争和受支持的原子写语义竞争；普通 SIMD 与 SIMD VF 的 UB 潜在竞争以 WARNING 报告。

四种检查能力的详细边界和命令行选项见[检查能力](references/check-capabilities.md)。

## 使用限制

| 项目 | 当前边界 |
| --- | --- |
| 设备与环境 | Ascend 950PR/950DT（芯片架构 `dav-3510`），使用与目标 NPU 和驱动匹配且包含 `npu-check` 的 CANN 包 |
| 应用形态 | 目标程序是可独立运行的可执行文件；PyTorch 场景暂不支持 |
| 调用形态 | 支持 `<<<>>>` NPU 核函数调用语法和 aclnn 的单算子单次调用 |
| 执行模型 | 单进程、单线程运行 |
| `memcheck` | GM 分配范围、L1/L0 容量、普通 SIMD/SIMD VF 的 UB 分区；SIMT VF 仅 UB 写入，需可靠布局与记录；不保证检查所有指令或逻辑 buffer 边界 |
| `synccheck` | 核内配对及受支持的跨核同步模式，不代表任意模式和核拓扑均可检查 |
| `initcheck` | 受支持的持久化寄存器依赖及 GM/UB/L1/L0A/L0B/L0C 初始化状态；UB 不按分区过滤实际读写，也不以 memcheck 越界抑制状态更新 |
| `racecheck` | 仅同一次下发内的竞争；不覆盖不同下发之间的竞争，未知原子配置会限制分析完整性 |
| VF、子函数及同步方式 | 区分 SIMD VF 与 SIMT VF，不能笼统判为不支持或全部支持；SIMT 不扩展到 initcheck/racecheck。非 inline 子函数、软件同步及高阶 API 内部诊断的既有限制仍适用 |
| UB 布局 | UB 分区 memcheck 使用 dav-3510 默认 8 KiB 预留配置；关闭 VF 栈或 ASC 预留的编译配置不在支持范围。详细规则见 [UB 检查](references/ub-checks.md) |
| 检测数据导出 | 被测应用需要在核函数下发后调用 `aclrtSynchronizeStream` 或 `aclrtSynchronizeStreamWithTimeout`，以导出并处理检测数据 |
| 非检查范围 | 不检查算子精度、性能、CPU 侧（Host）逻辑和构建配置 |

详细的检查边界、GM 分配范围与片上容量判定依据和结论边界见
[能力范围与限制](references/capability-boundaries.md)。

## 常用命令

```bash
# 查看当前版本支持的参数
npu-check --help

# 检查 GM、L1/L0 及受支持的 UB 访问
npu-check --tool memcheck -- <application> [application-args...]

# 检查同步指令配对
npu-check --tool synccheck -- <application> [application-args...]

# 检查初始化状态
npu-check --tool initcheck -- <application> [application-args...]

# 检查内存访问竞争
npu-check --tool racecheck -- <application> [application-args...]

# 同时启用四种检查
npu-check --tool memcheck --tool synccheck --tool initcheck --tool racecheck -- <application> [application-args...]

# 将工具报告写入指定文件
npu-check --tool memcheck --log-file <report-path> -- <application> [application-args...]
```

| 参数 | 含义 |
| --- | --- |
| `--tool <name>` | 选择 `memcheck`、`synccheck`、`initcheck` 或 `racecheck`，可多次指定；重复同一工具不重复检查；未指定时默认启用 `memcheck` |
| `--log-file <path>` | 将检查报告和应用输出写入指定文件；检查报告不再在终端显示，应用输出仍显示；父目录必须已存在，同名文件会被覆盖 |
| `--deadlock-timeout <seconds>` | 为所有 checker 设置算子执行超时，请求值为 1..INT_MAX 整数秒；实际生效受 Runtime 与 launch 配置影响，使用前核对当前帮助 |
| `--` | 分隔 `npu-check` 选项与被检查应用命令；应用路径或参数以 `-` 开头时不可省略 |
| `-h`、`--help` | 显示当前版本的命令帮助 |
| `<application> [application-args...]` | 可独立运行的目标应用及其原有参数 |

完整的环境检查、原始日志保存和超时结果处理见[运行 npu-check](references/execution.md)。

## 内存越界检测方式

`memcheck` 区分 GM 分配范围、L1/L0 容量和 UB 分区检查，不能混用判定基准。

`memcheck` 根据受支持的分配接口和 `aclrtFree` 的成功结果记录当前有效的 NPU 内存分配范围。
`aclrtMalloc`、`aclrtMallocAlign32` 的申请大小向上按 32 字节对齐后作为检查边界；`aclrtMallocWithCfg` 按请求字节数
记录。分配或释放失败时，不更新记录。只有受支持指令的访问范围完整落入一个当前有效的内存分配范围时，才视为有效访问。
详细的区间定义、释放后访问和覆盖限制见
[GM 越界判定基准](references/capability-boundaries.md#gm-越界判定基准)。

L1、L0A、L0B、L0C 按 Ascend 950（dav-3510）对应空间的容量检查，而不是按 `aclrtMalloc` 记录检查。
访问未超过空间容量，并不证明它没有越过用户划分的某个片上 buffer。容量及示例见
[片上容量判定基准](references/capability-boundaries.md#片上容量判定基准)。

UB 先区分普通 SIMD/SIMD VF 与 SIMT VF，再读取 [UB 检查](references/ub-checks.md)。前者按本次 kernel 类型划分区域，
同一操作数的实际片段必须留在起始区域；SIMT VF 逐线程检查写入所在的同一已分配区域。initcheck 独立跟踪实际字节，
racecheck 按物理 UB 归属及同步关系判定潜在竞争，不能照搬 memcheck 分区规则。

检测数据在被测应用调用 `aclrtSynchronizeStream` 或 `aclrtSynchronizeStreamWithTimeout` 时导出并处理。因此，被测
应用仅下发核函数而不调用任一同步接口时，不能根据没有诊断得出检查通过结论。

## 工作流程

1. **确认能力边界。** 读取[能力范围与限制](references/capability-boundaries.md)，记录目标设备、CANN 环境、应用调用
   形态、启用的检查类型和输入。
2. **选择检查类型。** 读取[检查能力](references/check-capabilities.md)。用户或调用方已指定类型时遵循其选择，
   不自动替换或追加；未指定时根据已确认的线索选择并说明理由。多个目标均有依据时可组合检查；
   普通内存检查且没有更具体目标时使用默认 memcheck。说明所选能力及未覆盖边界，线索不足时先澄清目标。
3. **受控运行。** 检查 `command -v npu-check` 和 `npu-check --help`，保留应用原有工作目录、参数、环境和
   NPU 设备选择，按照[运行 npu-check](references/execution.md)保存合并后的原始捕获日志、执行命令和进程退出
   状态。应用被外部超时或信号终止且 `npu-check` 未收到完整报告时，按运行参考保留原始日志和终止
   信息，将本次检查判定为未完成，不给出检查通过或已发现问题的结论。
4. **判断诊断可信度。** 将原始日志与对应的命令、环境、检查类型和退出状态一起核对，检查来源对应关系、上下文
   完整性、诊断与汇总是否自洽，以及是否存在截断或消息丢失迹象，据此判断为“可靠”“部分可靠”或“不可靠”并
   说明理由。不得通过固定文字或字段是否存在机械核对报告。部分可靠的报告可以保留上下文足以支撑的
   明确诊断，但不能扩展为完整通过结论。具体方法见
   [报告阅读](references/result-interpretation.md)。
5. **解读具体诊断。** 在完成可信度判断后，遇到具体报错或分析不完整原因时，只读取对应的诊断参考：
   [Memcheck](references/memcheck-diagnostics.md)、[Synccheck](references/synccheck-diagnostics.md)、
   [Initcheck](references/initcheck-diagnostics.md)、[Racecheck](references/racecheck-diagnostics.md)。结合报告原文
   说明诊断含义、需要核对的证据和不能推出的结论；组合检查只读取实际涉及的 checker 参考。
6. **处理受限结果。** 报告不完整、自相矛盾、来源不明或命令无法启动时，读取
   [故障排查](references/troubleshooting.md)。说明哪些信息仍可作为证据、哪些结论无法确认；只有改变了明确的
   失败条件后才重试。

## 报告输出

面向用户的结果按以下顺序组织：

1. **执行上下文**：算子、输入形状、应用命令、工作目录、检查类型，以及工具能检查和不能检查的范围。
2. **诊断可信度**：可靠、部分可靠或不可靠，以及支持该判断的来源、完整性、自洽性和缺失信息。
3. **检查结论与关键证据**：明确问题、潜在竞争或当前无法确认的结论、应用自身状态、首个相关诊断、源码位置、
   访问空间、分配或容量边界、同步或初始化上下文和原始日志路径。

```text
执行上下文：<算子、输入形状、应用命令、工作目录和检查类型>
检查范围：<应用命令、检查类型、检查覆盖范围和限制>
诊断可信度：<可靠/部分可靠/不可靠及判断理由>
检查结论：<明确问题、潜在竞争、本次范围内未发现问题或无法确认/未完成，以及应用状态>
关键证据：<检查报告中的诊断、源码位置、访问/同步上下文和原始捕获日志路径>
```

“未发现明确问题”的表述按[能力范围与限制](references/capability-boundaries.md)收敛。只有可靠报告才能支撑本次执行
范围内的通过结论；仅咨询或解读已有日志时，不声称执行了本次检查。

## 信息来源

命令参数以当前环境的 `npu-check --help` 为准；检查范围见[能力范围与限制](references/capability-boundaries.md)，报告字段结合配套资料及实际输出解读。
不同版本存在差异时，记录工具版本和实际输出；本技能能力说明、配套资料与本次报告均未提供的信息标记为未知，不通过分析
`npu-check`、开发工具包或编译器的内部源码补充用户使用结论，也不得补造参数、字段或检查能力。

通用命令及环境配置可参考 [npu-check 使用手册](https://gitcode.com/cann/asc-tools/blob/master/docs/zh/02_npu_check.md)。
