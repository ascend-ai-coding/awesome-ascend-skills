---
name: external-gitcode-ascend-tunnel
description: SSH通道管理，支持本地/远程端口转发、SOCKS代理、反向代理
original-name: ssh-dev-suite/tunnel
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# ssh-tunnel

## 功能
管理SSH端口转发与代理通道，含生命周期管理。

## 配置
参考`config.example.yaml`的tunnels段，profile引用connect配置。

## 使用
```bash
bash tunnel/scripts/ssh-tunnel.sh forward <profile> <lport>:<rhost>:<rport>
bash tunnel/scripts/ssh-tunnel.sh reverse <profile> <rport>:<lhost>:<lport>
bash tunnel/scripts/ssh-tunnel.sh socks <profile> <local_port>
bash tunnel/scripts/ssh-tunnel.sh proxy <profile> [--local-proxy-port 7890]
bash tunnel/scripts/ssh-tunnel.sh list
bash tunnel/scripts/ssh-tunnel.sh stop <tunnel_id>
bash tunnel/scripts/ssh-tunnel.sh stop-all
```

## 注意事项
- 隧道记录在`~/.ssh/tunnels.json`，list自动清理失效进程
- proxy输出远程端需执行的环境变量命令
- 成功返回0，失败返回非0

## Token约束
- 隧道列表：已足够精简，无需优化
- 状态检查：只输出关键信息（端口、PID、状态）
- 重复信息：不复述已知内容
