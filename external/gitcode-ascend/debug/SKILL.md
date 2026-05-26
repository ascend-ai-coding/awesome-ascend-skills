---
name: external-gitcode-ascend-debug
description: 结构化远程服务器问题排查流程，支持上下文感知的环境检查和容器内调试
original-name: ssh-dev-suite/debug
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# ssh-debug

## 概述
引导agent按结构化流程排查远程服务器问题，避免盲目执行命令。所有远程命令通过`connect/scripts/ssh-exec.sh`执行。

## 适用场景
- 服务异常（进程崩溃、服务不可用）
- 性能问题（CPU/内存/磁盘/GPU）
- 部署失败排查
- 容器内应用调试

## 流程步骤

### 1. 确认目标
- 获取问题描述，确认profile和排查方向
- **产出**：profile名称 + 问题分类

### 2. 上下文感知环境检查
- 读取本地工程文件推断依赖，动态生成检查项
- 基础检查：`uptime`, `df -h`, `free -m`, `top -bn1 | head -20`
- 按规则表执行额外检查
- **产出**：环境状态概览

### 3. 日志分析
- 定位日志：`journalctl -u <service> --since "1h ago"`, `/var/log/`
- 搜索关键词：`grep -i "error\|fatal\|exception" <log> | tail -50`
- 容器：`docker logs <container> --tail 200`

**Token优化策略**:
- 日志分析：先用grep搜索错误关键词，找到问题再扩展上下文
- 大日志文件：只读tail -50，不读全文
- 环境检查：输出摘要，不粘贴完整命令结果

- **产出**：错误信息摘要

### 4. 深入排查
- 根据线索执行针对性命令，每次验证一个假设
- 检查配置文件、环境变量、依赖版本
- **产出**：根因或最可能原因

### 5. 总结报告
- 输出发现（现象、根因、影响）和修复建议
- **产出**：结构化排查报告

## 上下文感知规则表

| 检测到的依赖 | 额外检查命令 |
|---|---|
| PyTorch / TensorFlow | `nvidia-smi`, GPU显存占用 |
| 华为 CANN / MindSpore | `npu-smi info`, NPU状态 |
| Docker / docker-compose | `docker ps -a`, `docker stats --no-stream` |
| 数据库连接配置 | 连通性测试、活跃连接数 |

## 常用排查命令速查

| 场景 | 命令 |
|---|---|
| 进程 | `ps aux \| grep <app>`, `systemctl status <svc>` |
| 资源 | `df -h`, `free -m`, `top -bn1 \| head -20` |
| 日志 | `tail -100 <log>`, `journalctl -u <svc> --since "1h ago"` |
| 网络 | `ss -tlnp`, `curl -sv localhost:<port>` |
| 容器 | `docker logs <c> --tail 100`, `docker stats --no-stream` |
| GPU | `nvidia-smi`, `npu-smi info` |

## 容器内调试

profile配置`container`字段时，`ssh-exec.sh`自动路由命令到容器内。注意：
- 容器内工具可能缺失，优先用基础命令（`cat`, `ls`, `env`）
- 需宿主机信息时，用不带container的profile

## Token约束
- 日志分析：先grep错误关键词，再按需扩展上下文
- 环境检查：输出摘要而非完整命令结果
- 深入排查：每次验证一个假设，不批量执行命令
- 重复信息：不复述已知内容
