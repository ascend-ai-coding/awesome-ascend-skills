---
name: external-gitcode-ascend-vllm-ascend-deploy
description: 昇腾 NPU 平台 vLLM 大模型推理服务一键部署。触发：用户说'部署 模型名'、'NPU 部署模型'、'vllm serve'。流程：SSH检查
  → NPU检查 → 配置发现(必须验证) → 用户确认 → 部署 → cron监控 → 验证。约束：(1) 配置必须从官方文档验证，禁止猜测；(2) 后台启动必须用cron监控，禁止手动轮询。支持
  Qwen/Qwen3.5、GLM、DeepSeek、Kimi。
original-name: vllm-ascend-deploy
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-05'
synced-commit: 03b4bbd870d13fcd48044b285e84dd80e955214f
license: UNKNOWN
---

# vllm-ascend-deploy

昇腾 NPU 平台 vLLM 大模型推理服务一键部署工具。

**核心工作流：** SSH 检查 → 解析模型 → NPU 检查 → 配置发现 → 用户确认 → 执行部署 → 监控启动 → 验证服务

---

## 部署前准备

### 1. SSH 连接检查（远程部署时）

```bash
ssh -o BatchMode=yes <server> "echo ok"
```

**连接失败时：**
1. 询问用户 SSH 密码
2. 使用 paramiko 配置免密登录
3. 验证成功后继续

### 2. 询问用户配置（优先）⚠️

**先问用户：**
- 是否有指定的容器镜像路径？
- 是否有指定的模型权重路径？
- 是否有特定的启动参数？

**判断标准：**

| 用户回复 | 含义 | 后续动作 |
|----------|------|----------|
| 具体路径/版本 | 明确提供 | 直接使用，跳过搜索 |
| "你自己确定"、"随便" | 模糊授权 | 必须搜索文档验证 |
| 未回复/跳过 | 未提供 | 必须搜索文档验证 |

**禁止：** 将模糊授权当作明确提供，跳过验证步骤

### 3. NPU 资源检查

```bash
npu-smi info          # 检查 NPU 状态和卡数
npu-smi info -t pid   # 检查占用进程
```

**占用处理：** NPU 被占用 → 列出进程 → 询问用户（杀进程/等待/取消）

### 4. 配置发现 ⚠️ 必须执行

**关键规则（完整流程见 [model-discovery.md](references/model-discovery.md)）：**

| 规则 | 说明 |
|------|------|
| 镜像仓库深度搜索 | 文档示例通常只展示一种架构（如 A3），**必须进入镜像仓库目录查看完整文件列表** |
| NPU 架构判断 | `npu-smi info` 显示 8×910B3 = **A2**，16×910B3 = **A3** |
| 镜像匹配 | A2 用 `-a2` 后缀，A3 用 `-a3` 后缀，不匹配可能无法启动 |
| 确认清单 | 先展示官方推荐配置，再告知现有资源，让用户选择 |

**信息源优先级：**

| 优先级 | 来源 | URL |
|--------|------|-----|
| 1️⃣ | GitCode 教程 | https://gitcode.com/org/vLLM_Ascend/ |
| 2️⃣ | 官方文档 | https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/tutorials/models/ |
| 3️⃣ | Modelers 在线库 | https://modelers.cn/user/vLLM_Ascend |

**提取内容：** 镜像 + TP 推荐值 + 启动参数 + 环境变量

**网络受限时：** 使用 browser 独立浏览器工具访问，不猜测参数

**禁止：**
- 第一个来源没找到就下结论，必须按优先级继续搜索
- 看到现有容器就直接用，忽略官方推荐

### 5. 用户确认 ⚠️ 阻断点

**前置条件检查：**
- [ ] 配置发现步骤已完成
- [ ] 镜像版本已从文档确认
- [ ] TP 推荐值已从文档确认

**任一条件未满足 → 禁止继续，返回 Step 4 搜索文档**

**确认清单必须包含：**
1. **官方推荐配置**（优先展示）
2. **NPU 架构判断**（8 × 910B3→A2, 16 × 910B3→A3）
3. **现有资源检查**（已有容器/镜像，仅供参考）
4. **选择确认**：让用户选择使用官方推荐还是复用现有资源

**禁止：**
- 看到现有容器就直接用，忽略官方推荐
- 不告知用户有选择余地

**确认后才开始部署，绝不假设或猜测参数。**

---

## 执行部署

### Step 1: 加载镜像（如需要）

```bash
docker load -i <IMAGE_TAR_PATH>
```

### Step 2: 创建容器

```bash
bash scripts/create_container.sh \
  --mode local \
  --image <IMAGE> \
  --model-path <MODEL_PATH> \
  --container-name <CONTAINER_NAME>
```

**远程部署：** 添加 `--mode remote --server <IP> --user <SSH_USER>`

详见 [deployment-procedure.md](references/deployment-procedure.md)

### Step 3: 启动服务

**优先尝试 vllm CLI：**

```bash
vllm serve <MODEL_PATH> --tensor-parallel-size <TP> ...
```

**CLI 失败时（如 ModuleNotFoundError），使用 Python 模块：**

```bash
python -m vllm.entrypoints.openai.api_server --model <MODEL_PATH> ...
```

### Step 4: 监控启动 ⚠️ 必须执行

**问题：** 后台启动服务后，exec 会超时，无法持续监控

**方案：** 创建 cron job 定期检查，结果通过 `sessions_send` 通知用户

```
流程：
1. 启动服务（后台）
2. 创建 cron job（每分钟检查）
3. 告知用户"启动中，完成后自动通知"
4. Cron job 检测到结果 → sessions_send 通知 → 删除 job
```

**创建 cron job：** ⚠️ 必须包含完整上下文

Cron job 在 isolated session 运行，没有父会话上下文，**必须明确指定所有信息**：

```python
cron(action="add", job={
    "name": "vllm-status-check",
    "schedule": {"kind": "every", "everyMs": 60000},
    "payload": {"kind": "agentTurn", "message": """检查 vLLM 服务状态：

服务器: <SERVER_IP>
容器名: <CONTAINER_NAME>
端口: <PORT>
日志路径: <LOG_PATH>  # 通常是 /tmp/vllm.log 或其他指定路径
服务地址: http://<SERVER_IP>:<PORT>/health
开始时间: <START_TIME>

检查步骤：
1. SSH 到服务器，执行 docker exec <CONTAINER_NAME> tail -50 <LOG_PATH>
2. curl http://<SERVER_IP>:<PORT>/health
3. 成功 → sessions_send 通知用户 → 删除此 job
4. 失败 → 检查日志错误，超时则报错
"""},
    "sessionTarget": "isolated"
})
```

**禁止：** 任务消息中省略日志路径、容器名、服务地址，会导致检查失败

**Cron job 检测到结果后通知用户：**

```python
sessions_send(
    sessionKey="agent:main:main",  # 用户会话保持不变
    message="✅ 启动成功！服务地址：...",
    timeoutSeconds=0
)
```

详见 [troubleshooting.md](references/troubleshooting.md)

### Step 5: 验证部署

```bash
curl http://<SERVER_IP>:<PORT>/health
curl http://<SERVER_IP>:<PORT>/v1/models
```

---

## 部署后调用

服务地址：`http://<SERVER_IP>:<PORT>`

**Python SDK：**

```python
from openai import OpenAI
client = OpenAI(base_url="http://<IP>:<PORT>/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="<model-name>",
    messages=[{"role": "user", "content": "你好"}]
)
```

---

## 故障排查

详见 [troubleshooting.md](references/troubleshooting.md)

**常见问题：**
- `No module named 'vllm.benchmarks.latency'` → 使用 Python 模块方式启动
- 容器创建失败 → 检查 NPU 设备映射
- 服务启动失败 → 检查 HCCL 环境变量
- OOM → 降低 gpu-memory-utilization 或增加 TP

---

## 脚本说明

| 脚本 | 功能 |
|------|------|
| `scripts/create_container.sh` | 创建容器（完整 NPU 配置） |
| `scripts/start_service.sh` | 启动 vLLM 服务 |
| `scripts/monitor.sh` | 服务监控 + WeLink 通知 |
| `scripts/api_test.py` | API 测试脚本 |
