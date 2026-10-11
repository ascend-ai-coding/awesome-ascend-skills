# CLAUDE.md

## OpenCode 子 Agent 监控经验

### 关键发现

当使用 `opencode run` 运行检视任务时，会派发多个子 agent 并行执行。每个子 agent 有独立的 `sessionID`。

### 日志输出配置

```bash
# 运行检视脚本（带详细日志）
opencode run --dangerously-skip-permissions --print-logs --log-level DEBUG "全量检视代码 $FILE_PATH"
```

关键参数：
- `--dangerously-skip-permissions`: 跳过权限检查，允许访问外部目录
- `--print-logs`: 打印日志到 stderr
- `--log-level DEBUG`: 设置日志级别为 DEBUG

### 子 Agent 日志格式

```
INFO  service=session.prompt step=N sessionID=ses_XXXXXXXXX loop
INFO  service=llm providerID=xxx modelID=glm-5 sessionID=ses_XXXXXXXXX agent=ascendc-ops-reviewer mode=subagent stream
✓ 检视组X：条例范围 Ascendc-Ops-Reviewer Agent
```

**区分子 Agent 的方法：**

1. **主 Agent**: `mode=primary`
2. **子 Agent**: `mode=subagent`
3. **每个子 agent 有唯一 sessionID**: `sessionID=ses_XXXXXXXXX`

### 监控脚本使用

```bash
# 运行检视并监控
./review_code.sh 2>&1 | tee /tmp/review_output.log &
./monitor_subagents.sh /tmp/review_output.log
```

### Claude 自动监控流程

当 Claude 监控 opencode 子 agent 运行时：

1. **每30秒检查**：
   - 活跃子 agent 数量
   - 已完成检视组数
   - 错误日志

2. **异常检测条件**：
   - 错误计数超过阈值（默认5次）
   - 子 agent 无响应（超过2分钟无新日志）
   - 主 agent 进入后期阶段

3. **异常处理权利**：
   - 使用 `TaskStop` 关闭运行任务
   - 分析日志定位问题：
     ```bash
     grep -i "error\|failed" $LOG_FILE | head -20
     grep "sessionID=ses_xxx" $LOG_FILE  # 特定子 agent
     ```

### 日志文件位置

- 实时日志: `/Users/mac/.local/share/opencode/log/*.log`
- 运行输出: 使用 `tee` 命令保存到指定文件

### 常用分析命令

```bash
# 查看所有子 agent sessionID
grep "mode=subagent" output.log | grep -o "sessionID=ses_[a-zA-Z0-9]*" | sort -u

# 查看特定子 agent 活动
grep "sessionID=ses_xxx" output.log | tail -20

# 查看检视组完成状态
grep "检视组" output.log

# 统计错误
grep -i "error" output.log | wc -l
```

---

## 项目脚本说明

### review_code.sh

运行代码检视脚本，默认检视 `split_core.cpp`。

```bash
./review_code.sh                    # 检视默认文件
./review_code.sh /path/to/file.cpp  # 检视指定文件
```

### monitor_subagents.sh

子 agent 监控脚本，每30秒检查运行状态。

```bash
./monitor_subagents.sh /tmp/review_output.log
```