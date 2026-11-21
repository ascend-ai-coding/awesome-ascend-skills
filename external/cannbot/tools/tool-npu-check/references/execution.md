# 运行 npu-check

按[检查类型选择](check-capabilities.md#检查类型选择)确定本次检查后读取本参考。保留应用原有的工作目录、环境、NPU 设备选择、参数和测试超时时间。不要编造应用
参数。如果项目已有 CANN 环境加载命令，运行前检查和 `npu-check` 必须复用该命令；否则使用
`source <CANN-install-dir>/cann/set_env.sh`。

## 运行前检查

运行前确认已安装工具和目标应用：

```bash
command -v npu-check
npu-check --help
test -x <application>
```

以当前工具的帮助输出作为命令行契约的权威依据。运行前还要确认：

- 应用属于[能力范围与限制](capability-boundaries.md)列出的适用运行配置；
- 不使用 npu-check 时，应用能够在相同工作目录、参数和 NPU 设备下独立启动并执行到需要检查的算子；
  不要求待诊断算子成功结束。算子报错、崩溃或超时不作为拒绝检查的理由，报告能否形成需在执行后判断；
- 需要源码行号时，算子已按项目现有构建方式保留行号信息。

超出适用运行配置时，将对应能力标记为不支持或未知，不把运行结果解释为有效检查结论。环境或命令异常时，
保留原始报错并按[故障排查](troubleshooting.md)处理。

只有工程目录时，先只读查看 README、构建和运行脚本，确认现有入口及参数。编译失败、依赖缺失或启动前失败
属于准备问题，应先明确具体缺项；不要与已经执行到目标算子的运行异常混淆，也不要为使基线通过而修改算子逻辑。

## 标准调用

使用已有应用命令替换 `<application>` 及其参数。下面是 Bash 模板，使用数组保存参数，避免带空格的参数被错误
拆分。模板统一添加 `--` 分隔符；当应用路径或参数以 `-` 开头时，该分隔符不可省略。

```bash
# 创建本次运行专用的临时目录，避免覆盖已有日志。
capture_dir="$(mktemp -d)"
raw_log="${capture_dir}/npu-check.log"
application="<application>"
application_args=(<existing arguments>)

# 即使 npu-check 返回非零状态，也继续保存退出状态和日志路径。
set +e
npu-check --tool memcheck \
  -- "${application}" "${application_args[@]}" >"${raw_log}" 2>&1
# `$?` 是上一条命令的退出状态；`2>&1` 已将标准错误合并到标准输出。
run_status=$?
set -e

printf 'npu-check exit status: %s\n' "${run_status}"
printf 'npu-check log: %s\n' "${raw_log}"
```

这里的“原始捕获日志”是未经修改的终端输出，可能同时包含 `npu-check` 检查报告和被测应用自身的输出；它不是一份
只包含检查报告的文件。

仅检查同步问题时，将检查类型参数替换为：

```bash
npu-check --tool synccheck
```

用户指定组合检查，或未指定类型且已确认需要同时核验内存范围和同步关系时，检查类型参数使用：

```bash
npu-check --tool memcheck --tool synccheck
```

初始化和竞争检查分别使用 `--tool initcheck`、`--tool racecheck`。仅在请求全部检查或全部目标均有依据时使用：

```bash
npu-check --tool memcheck --tool synccheck --tool initcheck --tool racecheck
```

用户或调用方已指定类型时不自动追加或替换；没有指定时，记录选择依据，不因改动涉及多类操作就组合运行。

所有变体都必须保留相同的应用命令、终端重定向和退出状态捕获。不要改写、摘录后覆盖或手工拼接原始日志；如果
需要摘录诊断，保留原始文件并在回答中注明摘录位置。

被测应用在 NPU 核函数下发后必须调用 `aclrtSynchronizeStream` 或
`aclrtSynchronizeStreamWithTimeout`。工具会在这些流同步接口调用时导出并处理当前流的检测数据；应用只下发
核函数而不调用任一接口时，不能把没有诊断解释为检查通过。

当前公开命令还支持 `--log-file <path>`。指定后，检查报告只写入该文件，不再显示在终端；应用的标准输出和标准
错误仍会显示在终端，并同时写入该文件。因此，`--log-file` 文件是检查报告与应用输出按工具采集顺序形成的混合
日志，终端捕获只是应用输出的重复副本，不是另一份独立检查报告。应分别保留日志文件、终端输出、执行命令和
退出状态，并明确它们属于同一次执行；不得重新拼接两个输出，也不得重复计算其中的应用输出。

## FIFO 采集与同步收尾

SIMD 采集链路使用 FIFO，Host 可在 kernel 运行期间持续搬出记录，但 checker 的解析与分发仍在流同步时完成。
**FIFO 不取消应用的流同步要求，也不新增公开开关或环境变量。** 此 FIFO 不处理 SIMT 采集段，不能据此承诺 SIMT 覆盖。

FIFO、Host 缓存和预算均有上限，不能宣称长 kernel 可无限采集；背压持续时可能丢弃新记录。
FIFO 约 4.5 秒的满队列等待属于采集写入重试策略，不是整个 kernel 的执行期限，更不是 `--deadlock-timeout`。
不同核的出队、回调或日志到达顺序不等于设备全局执行顺序。

正常同步后执行最终回收；同步失败时可能继续交付已采集的局部诊断，但不能据此宣布整个检查通过。
报告出现 `trace_fifo_dropped`、`trace_collection_failed` 或 `analysis_complete=false/status=incomplete` 时，
按[报告阅读](result-interpretation.md)区分局部充分证据和整体分析缺口，保存失败阶段及丢失信息。

## 超时结果

`--deadlock-timeout <seconds>` 是**所有 checker 共用的算子执行超时配置**，不是 synccheck 专用开关。
只有当前 `--help` 支持且用户或项目明确给出该要求时才使用；未指定时保留 Runtime 原有策略。

- 输入为 1..INT_MAX 的整数秒；缺值、0、负数、小数、非数字或超范围均为参数错误，不启动被检查应用。
- 重复指定使用最后一个值；`--` 后面的参数属于应用，不再作为 npu-check 选项解析。
- 工具在每次受支持的实际 launch 前请求 Runtime 设置执行超时。实际阈值受产品粒度、最大值及 Runtime 规则影响；
  launch 自带的显式超时配置优先于此进程级设置，不能把 CLI 请求值当作精确截止时间。
- `operator execution timeout setup failed: timeout_sec=... result=...` 表示 Runtime 接口获取或设置失败。
  该失败会记录日志，原 launch 继续，不直接生成 checker 诊断，也不增加该分支的 framework error；不能声称请求已生效。

它不是进程墙钟超时，不保证在指定秒数内杀死应用，也不检测或证明源码死锁；不保证超时后完整生成报告。
示例（仅在任务要求 5 秒算子超时且当前包支持时）：

```bash
npu-check --tool memcheck --deadlock-timeout 5 -- <application> [application-args...]
```

`npu-check` 在应用正常退出时汇总检查报告。应用被外部超时或信号终止而未执行正常退出流程时，工具可能收不到
完整报告；缺少完整报告时，工具不能给出有效检查结论。应保留原始捕获日志、
`npu-check` 退出状态以及超时或信号信息，将本次检查判定为未完成。终止前的应用输出只能说明应用执行状态，不能
代替 `npu-check` 检查报告，也不能据此声明检查通过或已发现检查问题。只有工具已经输出完整报告时，才按
[报告阅读](result-interpretation.md)判断其可信度。

## 结果判定规范

进程退出状态只能说明命令如何结束，不能单独代表检查结论。应共同保留：

- 未使用 `--log-file` 时，未经改写的原始捕获日志；
- 使用 `--log-file` 时，工具生成的混合日志文件和终端侧应用输出重复副本；
- 完整执行命令、工作目录、检查类型、CANN/NPU 环境和工具版本；
- `run_status`，以及外层超时或信号终止信息；

根据这些原始证据判断诊断可信度，不使用固定文字或字段是否存在进行机械校验。报告来源不明、被手工
拼接、跨版本混用，或关键信息相互冲突时，应说明具体疑点并判断为不可靠。工具已完整发送报告并明确标记
`report_truncated=true` 时，其中上下文充分的明确诊断可判断为部分可靠并予以保留，但不能据此声明完整通过；
应用被终止且工具未收到完整报告时，本次检查没有有效结论。
