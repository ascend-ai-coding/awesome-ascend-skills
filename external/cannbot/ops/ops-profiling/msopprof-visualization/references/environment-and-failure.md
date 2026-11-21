# 环境、预检与失败处理

## 环境档案

环境档案用于复用稳定主机上的：

- 经过筛选的 CANN 环境变量；
- `msprof` 绝对路径和文件身份；
- `msprof op --help` 的完整能力清单；
- 主机和来源脚本元数据。

以下变化使档案失效：

- 主机身份变化；
- 来源脚本路径、时间戳或内容变化；
- `msprof` 文件身份变化；
- 档案缺少必需字段。

档案不得缓存算子相关检查。

## BasicInfo canary

正式采集前执行短时 canary：

- 独立输出目录；
- `--launch-count=1`；
- 解析后的 `--app-cwd`；
- 默认短时超时；
- 校验 OpBasicInfo 或可用 visualize payload。

以下任一情况立即中止：

- 超时；
- 返回码非零；
- 返回码为 0 但没有可用产物；
- 设备或目录权限错误。

## 进程隔离

每个 msprof 命令必须在新 session/进程组中执行。超时处理顺序：

1. 标记命令超时；
2. 向完整进程组发送终止信号；
3. 等待短暂清理窗口；
4. 强制清理仍存活的后代进程；
5. 写入 stdout、stderr、heartbeat 和进程诊断；
6. 默认触发 circuit breaker，跳过后续昂贵模块。

## 输出目录权限

msprof 可能在输出目录对 group/other 可写时返回 0 但不生成产物。创建输出目录后，应移除 group/other write 位，并将修正写入：

```text
_internal/output_permission_fixes.json
```

## 诊断内容

共享设备或锁问题发生时，尽可能收集：

- `npu-smi info`；
- `/proc/locks`；
- Ascend `sink_file_mutex_*` 文件身份；
- holder/waiter PID、状态、wchan、命令行、父 PID、工作目录；
- 进程快照；
- 当前命令、工作目录和完整日志片段。

诊断保存到：

```text
_internal/diagnostics/{stage}/
```

## 状态与复用

顶层 `_internal/run_state.json` 仅允许：

- `running`；
- `completed`；
- `aborted`。

只有 `completed` 可作为复用来源。发现上次状态为 `running` 或 `aborted` 时，即使 per-block state 文件存在，也不得自动复用。
