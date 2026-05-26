---
name: external-gitcode-ascend-connect
description: SSH连接管理、远程命令执行、文件传输、后台任务
original-name: ssh-dev-suite/connect
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# ssh-connect

## 功能
SSH连接测试、远程命令执行、文件上传下载、后台任务管理。容器场景自动包装docker/kubectl。

## 运行环境
- **客户端**：Windows（Git Bash/MSYS2）、macOS、Linux
- **服务端**：Linux

## 使用
```bash
bash connect/scripts/ssh-test.sh <profile>
bash connect/scripts/ssh-exec.sh <profile> <command>
bash connect/scripts/ssh-upload.sh <profile> <local> <remote>
bash connect/scripts/ssh-download.sh <profile> <remote> <local>
bash connect/scripts/ssh-job.sh start|status|output|kill|list|stream <profile> [args]
```

## 引号与特殊字符

远程命令经过多层 shell 解释（本地 shell → ssh → 远端 shell → docker exec → 容器内 shell），引号嵌套极易出错。

- **复杂命令**（含引号、变量、管道等）：推荐写成脚本文件，通过 `ssh-upload.sh` 上传到服务器后执行，避免内联命令的多层引号嵌套
- **简单命令**：可直接内联传递，脚本已处理基本转义
- 密码认证通过环境变量（`sshpass -e`）传递，不在命令行暴露

## Windows 客户端注意事项

- ControlMaster 会话复用在 Windows 上不可用（脚本已自动检测并跳过）
- `$PATH` 等环境变量在 Git Bash 中可能被 Windows 路径展开，导致传递到远端时出错。建议用上传脚本方式代替内联命令中的 `$PATH` 引用
- identity_file 路径含空格时已自动处理引用

## Token约束
- 远程命令输出：优先使用过滤参数（--tail/--head/--grep），禁止读取完整大日志
- 后台任务输出：先用status查看大小，>10KB只读tail，>100KB先grep错误
- 状态检查：先查摘要，按需深入
- 重复信息：不复述已知内容
