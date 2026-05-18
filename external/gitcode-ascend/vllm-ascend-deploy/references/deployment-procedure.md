# 昇腾 NPU vLLM 部署完整流程

## 目录

- [部署前确认](#部署前确认必须执行)
- [完整部署流程](#完整部署流程)
- [服务启动监控](#服务启动监控)
- [部署检查清单](#部署检查清单)
- [模型配置参考](#模型配置参考)

---

## 部署前确认（必须执行）

### 确认原则

- **先问用户** - 用户提供的配置直接使用
- **不假设、不猜测** - 未提供的参数从官方文档查找
- **动态生成** - 根据模型类型和 NPU 型号生成确认清单

### 确认参数框架

| 参数 | 获取方式 | 说明 |
|------|----------|------|
| **服务器 IP** | 用户指定 | 部署目标服务器 |
| **容器镜像** | 用户指定或文档推荐 | 注意 A2/A3 型号匹配 |
| **模型权重路径** | 用户指定 | 必须确认存在 |
| **TP 并行数** | 从官方文档提取 | 不同模型推荐值不同 |
| **服务端口** | 默认 8000 | 可自定义 |

---

## 完整部署流程

### Step 1: 环境确认

```bash
npu-smi info                    # 检查 NPU 状态
docker images | grep vllm       # 检查镜像
ls <MODEL_PATH>/config.json     # 检查模型路径
```

### Step 2: 创建容器

**关键配置：**

| 配置项 | 说明 |
|--------|------|
| `--device=/dev/davinci0-N` | NPU 核心，根据检测卡数动态指定 |
| `--device=/dev/davinci_manager` | NPU 管理设备 |
| `--device=/dev/devmm_svm` | 共享虚拟内存 |
| `--device=/dev/hisi_hdc` | 调试设备 |
| `-v /usr/local/Ascend/driver` | 昇腾驱动 |
| `--network host` | 主机网络模式 |
| `--shm-size 100g` | 共享内存（大模型推荐 100g） |
| `--privileged` | 特权模式 |

详见 `scripts/create_container.sh`

### Step 3: 启动 vLLM 服务

**方式 1：vllm CLI（优先）**

```bash
vllm serve <MODEL_PATH> \
  --tensor-parallel-size <TP> \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.94 \
  --trust-remote-code \
  --async-scheduling
```

**方式 2：Python 模块（CLI 失败时）**

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <MODEL_PATH> \
  --tensor-parallel-size <TP> \
  ...
```

> **注意：** 如果遇到 `No module named 'vllm.benchmarks.latency'`，使用方式 2

### Step 4: 验证部署

```bash
curl http://<IP>:<PORT>/health
curl http://<IP>:<PORT>/v1/models
```

---

## 服务启动监控

### 监控逻辑

启动服务后，执行后台监控检测成功/失败：

```bash
while true; do
    logs=$(docker exec <container> tail -50 /tmp/vllm.log 2>&1)
    
    # 检测错误（立即通知）
    if echo "$logs" | grep -qiE "error|exception|traceback|failed|oom"; then
        # 通知用户启动失败
        break
    fi
    
    # 检测成功（立即通知）
    if echo "$logs" | grep -q "Application startup complete"; then
        # 通知用户启动成功
        break
    fi
    
    # 超时（10分钟）
    if [ $elapsed -gt 600 ]; then
        # 通知用户启动超时
        break
    fi
    
    sleep 30
done
```

### 通知用户

使用 `sessions_send(timeoutSeconds=0)` 在 webchat 中主动通知用户结果

---

## 部署检查清单

- [ ] SSH 连接正常（远程部署）
- [ ] NPU 资源充足
- [ ] 镜像已加载（注意 A2/A3 匹配）
- [ ] 模型权重路径存在
- [ ] TP 并行数已确认（从文档提取）
- [ ] 容器创建成功
- [ ] 服务启动成功（监控确认）
- [ ] 健康检查通过

---

## 模型配置参考

| 模型 | A2 TP | A3 TP | max-model-len | 来源 |
|------|-------|-------|---------------|------|
| Qwen3.5-27B | 4 | - | 32768 | GitCode 官方教程 |
| Qwen3.5-35B-A3B | - | 8 | 32768 | GitCode 官方教程 |
| GLM-5 | - | 16 | 8192 | 官方文档 |

**注意：** 具体配置需从搜索到的官方文档中提取，不硬编码
