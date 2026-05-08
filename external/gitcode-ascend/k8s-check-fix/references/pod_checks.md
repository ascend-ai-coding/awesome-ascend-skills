# Pod 通用检查细则

本文档提供 Pod 诊断的系统化方法，适用于未明确故障类型或需要深入排查 Pod 状态的场景。  
**使用建议**：当通过 `sweep` 发现问题 Pod，或用户直接询问某个 Pod 的健康状况时，可参考以下步骤进行分析。

---

## 1. 快速概览：获取 Pod 基本信息

### 1.1 使用 `pod` 子命令
```
k8s_check_fix(subcommand="pod", target="<pod-name>", namespace="<namespace>")
```
返回的 JSON 包含：
- `pod`：Pod 详细信息（状态、节点、容器、资源等）
- `image_mismatch`：镜像版本差异
- `current_logs`：当前日志
- `previous_logs`：上一次容器日志（崩溃时关键）
- `events`：该 Pod 相关事件

### 1.2 关注的关键字段
- `phase`：Pod 当前阶段（Running、Pending、Failed、Unknown 等）
- `conditions`：Pod 就绪、调度等条件
- `container_statuses[].state`：容器状态（running、waiting、terminated）
- `restart_count`：重启次数（频繁重启通常表示问题）
- `reason` / `message`：终止原因（如 OOMKilled、Error）

---

## 2. 根据 Pod 状态分类诊断

### 2.1 Pending（待调度）
**常见原因**：
- 资源不足（CPU/内存）
- 节点选择器/亲和性无法匹配
- PVC 未绑定
- 节点污点与容忍度不匹配

**诊断步骤**：
1. 查看 `pod.conditions` 中 `PodScheduled` 的 `reason` 和 `message`。
2. 检查节点资源：`kubectl describe nodes` 查看可分配资源。
3. 检查 PVC：`kubectl get pvc -n <namespace>` 确认是否 Bound。
4. 检查节点污点：`kubectl describe node <node>` 查看 Taints。
5. 检查 Pod 的 nodeSelector 和 tolerations。

### 2.2 CrashLoopBackOff
**常见原因**：
- 应用启动失败（配置错误、依赖缺失）
- 内存不足（OOMKilled）
- 健康检查失败
- 镜像问题

**诊断步骤**：
1. 查看 `previous_logs`（上一次容器日志）获取崩溃时的输出。
2. 查看 `container_statuses[].state.terminated.reason` 和 `exitCode`。
3. 如果是 OOMKilled → 检查内存 limit 和实际使用（参考资源部分）。
4. 检查健康检查配置：`livenessProbe`、`readinessProbe` 是否合理。
5. 检查 ConfigMap、Secret 挂载内容是否正确。

### 2.3 ImagePullBackOff / ErrImagePull
**常见原因**：
- 镜像名称错误
- 镜像仓库认证失败
- 网络问题无法拉取

**诊断步骤**：
1. 查看 `container_statuses[].state.waiting.reason`（通常为 ImagePullBackOff）。
2. 查看 `events` 中与镜像相关的错误信息。
3. 检查 `pod.spec.containers[].image` 是否正确。
4. 检查 imagePullSecrets 是否配置且有效。

### 2.4 OOMKilled
**常见原因**：
- 内存 limit 设置过低
- 应用内存泄漏
- 突发内存使用高峰

**诊断步骤**：
1. 确认 `container_statuses[].state.terminated.reason` 为 `OOMKilled`。
2. 查看 `previous_logs` 是否显示内存相关错误。
3. 检查 `pod.spec.containers[].resources.limits.memory`。
4. 评估应用正常内存使用量，建议适当增加 limit 或优化应用。

### 2.5 Running 但未就绪（Ready 为 False）
**常见原因**：
- 健康检查失败（readinessProbe）
- 容器内部服务未启动
- 端口监听错误

**诊断步骤**：
1. 查看 `pod.conditions` 中 `Ready` 的 `message`。
2. 检查 `readinessProbe` 配置（路径、端口、命令）。
3. 进入容器（如果允许）测试探测端点或命令。

### 2.6 Unknown / Terminating
**常见原因**：
- 节点宕机或网络分区
- kubelet 无法上报状态

**诊断步骤**：
1. 检查节点状态（`kubectl get nodes`）。
2. 查看节点事件和 kubelet 日志。
3. 参考节点相关诊断文档。

---

## 3. 日志分析技巧

### 3.1 当前日志 vs 上一次日志
- **当前日志**：适用于容器仍在运行的场景，观察最近输出。
- **上一次日志**：容器崩溃后重启，上一次日志包含崩溃时的错误信息，对诊断 CrashLoopBackOff 至关重要。

### 3.2 多容器 Pod
- 使用 `kubectl logs <pod> -c <container-name>` 查看特定容器日志。
- 工具返回的 `current_logs` 和 `previous_logs` 已包含所有容器，格式为 `[container-name] logs...`。

### 3.3 日志尾部行数
- 使用 `tail` 参数控制行数（默认 200），避免日志过长。

---

## 4. 事件解读

### 4.1 事件类型
- `Normal`：正常事件（如调度、拉取镜像成功、启动）
- `Warning`：异常事件（如失败、错误、重试）

### 4.2 关键事件原因
| 原因 | 含义 |
|------|------|
| `FailedScheduling` | 调度失败，通常伴随资源不足或约束不满足 |
| `FailedMount` | 挂载卷失败 |
| `FailedPull` | 拉取镜像失败 |
| `BackOff` | 容器反复崩溃，进入退避重启 |
| `Killing` | 容器被终止（如 OOMKilled、健康检查失败） |

### 4.3 关联分析
- 将事件与 Pod 状态、日志结合，例如：
  - 事件中出现 `OOMKilled` + 日志中无错误 → 内存不足
  - 事件中出现 `FailedMount` + 日志中无错误 → PVC 或挂载问题
  - 事件中出现 `BackOff` + 日志中有配置错误 → 应用启动失败

---

## 5. 资源限制检查

### 5.1 检查 requests 与 limits
- **requests**：调度依据，保证最低资源
- **limits**：容器允许使用的最大资源

### 5.2 常见问题
- **缺少 limits**：容器可能无限制消耗资源，导致节点压力。
- **requests 过高**：浪费资源，可能造成调度困难。
- **limits 过低**：容易 OOMKilled 或被 CPU 限流。

### 5.3 查看方法
通过 `pod` 子命令返回的 `containers[].resources` 字段可查看。

### 5.4 建议修复
- 若因 OOMKilled 且 limits 过低 → 提高内存 limits
- 若节点资源紧张且 Pod 未设置 requests → 建议设置合理的 requests

---

## 6. 镜像版本不一致

### 6.1 现象
`image_mismatch` 字段显示 `spec_image` 与 `running_image` 不同。

### 6.2 可能原因
- 镜像标签被覆盖（如使用 `latest` 标签且镜像已更新）
- 镜像拉取策略 `imagePullPolicy` 为 `IfNotPresent`，但本地已有旧版本

### 6.3 处理
- 若预期使用新镜像，可删除 Pod 或设置 `imagePullPolicy: Always`。
- 若不需要，可忽略或回滚 Deployment。

---

## 7. 诊断决策树（供模型参考）

```
Pod 状态
├── Pending → 检查调度条件（节点资源、污点、PVC、亲和性）
├── CrashLoopBackOff → 查看上一次日志和终止原因
│   ├── OOMKilled → 增加内存 limit
│   ├── 应用错误 → 回滚 Deployment 或修复配置
│   └── 健康检查失败 → 调整探针配置
├── ImagePullBackOff → 检查镜像名称、认证、网络
├── Running 但未就绪 → 检查 readinessProbe、容器启动状态
└── Unknown/Terminating → 检查节点状态和 kubelet
```

---

## 8. 注意事项（Gotchas）

- ❌ **只查看当前日志**：对于崩溃的 Pod，必须查看上一次日志才能获取真正错误。
- ❌ **忽略重启次数**：高重启次数通常意味着持续性问题，而非偶发。
- ❌ **直接删除 Pod**：若由 Deployment 管理，删除 Pod 会自动重建，应优先回滚或更新配置。
- ❌ **忽略 Pod 终止原因**：仅看 `phase` 不足，需结合 `state.terminated.reason`。
- ⚠️ **日志可能包含敏感信息**：输出前提醒用户。
- ⚠️ **多容器 Pod**：注意区分容器，错误可能发生在 initContainer。

