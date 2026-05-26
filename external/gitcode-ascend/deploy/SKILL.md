---
name: external-gitcode-ascend-deploy
description: 本地项目部署到远程服务器，支持增量同步、部署钩子、回滚
original-name: ssh-dev-suite/deploy
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# ssh-deploy

## 功能
通过ssh-dev-suite将本地项目同步到远程服务器，支持部署钩子和回滚。

## 配置
config.yaml deploy段:
- `profile`: 连接配置名
- `local_path`/`remote_path`: 本地和远程路径
- `sync_mode`: full(完整推送删除多余) | incremental(只传变更)
- `exclude`: 排除文件列表
- `pre_deploy`/`post_deploy`: 部署前后远程命令
- `rollback`: 回滚命令列表

## 使用
部署: `bash deploy/scripts/ssh-deploy.sh deploy`
回滚: `bash deploy/scripts/ssh-deploy.sh rollback`

## 注意事项
- full模式删除远程多余文件，确认remote_path正确
- rollback命令需用户按场景自行配置

## Token约束
- 部署日志：rsync输出较长时只关注错误和最终状态
- 钩子命令输出：超过50行时截断显示头尾
- 重复信息：不复述已知内容
