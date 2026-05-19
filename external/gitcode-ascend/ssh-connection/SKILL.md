---
name: external-gitcode-ascend-ssh-connection
description: SSH远程开发套件，连接管理、命令执行、文件传输、部署、隧道、调试
original-name: ssh-dev-suite
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-19'
synced-commit: 61a50580836017810c0ff005bc53c940ca059f06
license: UNKNOWN
---

# ssh-dev-suite

## 功能
多profile SSH远程开发套件。ControlMaster会话复用，容器感知（docker/kubectl）。

## 运行环境
- **客户端**：Windows（Git Bash/MSYS2）、macOS、Linux
- **服务端**：Linux
- Windows 客户端注意：ControlMaster 不可用（脚本已自动跳过）、PATH 环境变量展开差异、换行符问题

## 模块
- **connect**：连接、执行、传输、后台任务 → `connect/SKILL.md`
- **deploy**：部署与回滚 → `deploy/SKILL.md`
- **tunnel**：端口转发与代理 → `tunnel/SKILL.md`
- **debug**：远程排查 → `debug/SKILL.md`
- **long-task**：长耗时任务管理、checkpoint恢复 → `long-task/SKILL.md`

## 配置
`config.yaml`由agent引导生成，参考`config.example.yaml`。
密码通过环境变量设置。

## 注意事项
- 认证优先级：密钥 > 环境变量密码 > 交互
- container字段自动包装docker/kubectl exec

## Token约束
所有子模块遵循统一原则：
- 远程命令输出：优先使用过滤参数（--tail/--head/--grep），禁止读取完整大日志
- 状态检查：先查摘要，按需深入
- 重复信息：不复述已知内容
