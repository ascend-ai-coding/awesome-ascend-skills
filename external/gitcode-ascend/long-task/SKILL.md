---
name: external-gitcode-ascend-long-task
description: 长耗时任务管理，支持checkpoint记忆、agent休息与恢复
original-name: ssh-dev-suite/long-task
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# ssh-long-task

## 概述
管理预计耗时>2分钟的远程任务，通过checkpoint机制支持agent休息和跨会话恢复。

## 适用场景
- 模型训练（>10分钟）
- 大规模数据处理
- 编译构建任务
- 数据库迁移

## 流程步骤

### 1. 启动任务并写入checkpoint

使用`connect/scripts/ssh-job.sh start`启动任务，获取job_id后立即写入checkpoint：

```bash
bash connect/scripts/ssh-job.sh start <profile> "<command>"
# 输出: 任务已启动: job-20260306-143022-1234

bash long-task/scripts/checkpoint.sh write <profile> <job_id> \
  --task "模型训练 epoch 50" \
  --duration "~2h" \
  --next "检查训练完成,下载模型权重,运行验证" \
  --context "lr=0.001,batch=32"
```

**产出**: checkpoint文件 `~/.ssh-jobs/<job_id>/checkpoint.md`

### 2. 告知用户并休息

Agent输出：
```
✓ 任务已启动: job-20260306-143022-1234
✓ Checkpoint已保存，预计耗时 ~2h

你可以：
1. 稍后输入"检查任务 <job_id>"查看状态
2. 关闭会话，任务继续运行
3. 新会话中我会自动读取checkpoint恢复上下文
```

### 3. 恢复时读取checkpoint

新会话或用户请求检查时：

```bash
bash long-task/scripts/checkpoint.sh read <profile> <job_id>
```

**产出**: 解析checkpoint内容（task, started, duration, next_steps, context）

### 4. 智能状态检查

根据预计耗时决定检查策略：

```bash
bash connect/scripts/ssh-job.sh status <profile> <job_id>
```

- **<5分钟**: 每30秒检查一次
- **5-30分钟**: 每2分钟检查一次
- **>30分钟**: 每10分钟检查一次，或用户手动触发

### 5. 输出检查（Token优化）

**先查大小**:
```bash
bash connect/scripts/ssh-job.sh status <profile> <job_id>
# 输出: stdout: 245KB, stderr: 1.2KB
```

**按大小决策**:
- **<10KB**: 读取全部 `bash connect/scripts/ssh-job.sh output <profile> <job_id>`
- **10KB-100KB**: 只读尾部 `bash connect/scripts/ssh-job.sh output <profile> <job_id> --tail 50`
- **>100KB**: 先搜索错误 `bash connect/scripts/ssh-job.sh output <profile> <job_id> --grep 'error|fail|exception'`

### 6. 执行next_steps

任务完成后，按checkpoint中的next_steps继续：
逐步执行，每步完成后更新checkpoint或删除。

## Checkpoint格式

存储在 `~/.ssh-jobs/<job_id>/checkpoint.md`:

```yaml
job_id: job-20260306-143022-1234
profile: dev-server
task: "模型训练 epoch 50"
started: 2026-03-06T14:30:22
expected_duration: ~2h
next_steps:
  - 检查训练完成状态
  - 下载模型权重到本地
  - 运行验证脚本
context: "learning_rate=0.001, batch_size=32"
```

## Token约束
- 输出检查：先查大小，按大小分级读取（全部/尾部/grep）
- 状态轮询：根据预计耗时调整频率，避免频繁检查
- Checkpoint内容：精简关键信息，不存储完整日志
