# 故障排查手册

## 目录

- [快速诊断](#快速诊断)
- [常见问题](#常见问题)
- [日志关键词](#日志关键词)
- [启动监控](#启动监控)

---

## 快速诊断

```bash
docker ps | grep vllm                      # 检查容器状态
docker logs <container> --tail 50          # 查看日志
docker exec <container> npu-smi info       # 检查 NPU 使用
curl http://localhost:8000/v1/models       # 测试服务
```

---

## 常见问题

### SSH 连接失败

```
Permission denied (publickey,password)
```

**解决：** 使用 paramiko 配置免密登录

```python
import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('<SERVER>', username='root', password='<PASSWORD>')
pub_key = open('~/.ssh/id_rsa.pub').read()
ssh.exec_command(f'echo "{pub_key}" >> ~/.ssh/authorized_keys')
```

### vllm 命令失败

```
No module named 'vllm.benchmarks.latency'
```

**原因：** 镜像内 vllm 是 editable install，CLI 不完整

**解决：** 使用 Python 模块方式启动

```bash
# ❌ CLI
vllm serve /path/to/model ...

# ✅ Python 模块
python -m vllm.entrypoints.openai.api_server --model /path/to/model ...
```

### 容器创建失败

```
device not found
```

**解决：** 检查 NPU 驱动和设备映射

```bash
npu-smi info  # 确认 NPU 正常
```

### 服务启动失败

```
HCCL initialization failed
```

**解决：** 检查环境变量

```bash
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
```

### OOM 错误

**解决：**

```bash
--gpu-memory-utilization 0.85
--max-model-len 4096
--tensor-parallel-size 8
```

### 推理超时

**原因：** 首次推理需要预热，可能较慢

**解决：** 等待 1-2 分钟，或检查 NPU 状态

---

## 日志关键词

| 关键词 | 含义 | 处理 |
|--------|------|------|
| `Application startup complete` | 启动成功 | ✅ 继续 |
| `Loading model weights` | 加载中 | ⏳ 等待 |
| `Graph capturing` | 图捕获中 | ⏳ 等待 |
| `Error/Exception/Traceback` | 错误 | ❌ 排查 |
| `OOM` | 显存不足 | ❌ 降低配置 |
| `No module named` | 模块缺失 | ❌ 改用 Python 模块 |

---

## 启动监控

### 核心问题

**后台启动服务后，如何监控成功/失败？**

```
我启动后台服务 → exec 超时断开 → 无法持续监控 → 不知道成功还是失败
```

### 解决方案：cron 轮询 + isolated session

```
┌─────────────────────────────────────────────────────────────┐
│  用户 Main Session (agent:main:main)                        │
│                                                              │
│  用户: "部署 qwen3.5"                                        │
│      ↓                                                       │
│  我启动服务 → 创建 cron job → 告知用户"启动中"              │
│      ↓                                                       │
│  会话继续，用户可以做其他事                                  │
│      ↓                                                       │
│  收到通知：✅ 启动成功                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         ↑
         │ sessions_send(sessionKey="agent:main:main", ...)
         │
┌─────────────────────────────────────────────────────────────┐
│  Cron Job (isolated session)                                │
│                                                              │
│  每分钟检查一次服务器状态                                    │
│  检测到结果 → 通知用户 → 删除 job                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**关键机制：**

| 组件 | 说明 |
|------|------|
| Main Session | 用户会话，始终存在，保持不变 |
| Isolated Session | Cron job 独立运行，与 main session 隔离 |
| sessions_send | Isolated session 通过此工具通知 main session |

### 实现代码

**Step 1: 启动服务并创建 cron job**

```python
# 启动服务（后台）
exec("ssh root@server 'nohup vllm serve ... > /tmp/vllm.log 2>&1 &'")

# 创建 cron job 检查状态
cron(action="add", job={
    "name": "vllm-status-check",
    "schedule": {"kind": "every", "everyMs": 60000},  # 每分钟
    "payload": {
        "kind": "agentTurn",
        "message": "检查 vllm 服务状态..."
    },
    "sessionTarget": "isolated",
    "delivery": {"mode": "announce"}
})

# 告知用户
"服务启动中，预计 3-5 分钟，完成后会自动通知你"
```

**Step 2: Cron job 执行逻辑（isolated session）**

```python
# 检查服务器状态
logs = ssh("docker exec <container> tail -50 /tmp/vllm.log")

# 检测成功
if "Application startup complete" in logs:
    sessions_send(
        sessionKey="agent:main:main",  # 通知 main session
        message="✅ Qwen3.5-27B 启动成功！服务地址：http://...",
        timeoutSeconds=0
    )
    cron(action="remove", jobId="vllm-status-check")  # 删除 job

# 检测失败
elif "Error" in logs or "Exception" in logs:
    sessions_send(
        sessionKey="agent:main:main",
        message="❌ 启动失败：...",
        timeoutSeconds=0
    )
    cron(action="remove", jobId="vllm-status-check")

# 超时（10分钟）
elif elapsed > 600:
    sessions_send(
        sessionKey="agent:main:main",
        message="⏠ 启动超时，请检查日志",
        timeoutSeconds=0
    )
    cron(action="remove", jobId="vllm-status-check")
```

### 为什么用 sessions_send 通知

| 方式 | 响应速度 | 实现 | 推荐 |
|------|----------|------|------|
| `sessions_send` | ⚡ 立即 | 1 行代码 | ✅ |
| 服务器回调 | 需要网络可达 | 复杂 | ❌ |

- `sessions_send` 消息立即显示在 webchat
- 服务器回调需要 OpenClaw Gateway 可被服务器访问（网络隔离时不可行）
