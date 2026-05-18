# 安全策略 — k8s-check-fix

## 设计原则

1. **默认只读**  
   所有诊断子命令（`sweep`, `pod`, `deploy`, `resources`, `events`）仅执行只读操作：`kubectl get`、`kubectl describe`、`kubectl logs`、`kubectl top`、`kubectl rollout status`、`kubectl rollout history`。不修改集群状态。

2. **写操作需显式用户确认**  
   `fix` 子命令配合 `--confirm` 标志是唯一能修改集群的路径。执行前必须：
   - 向用户展示完整的命令
   - 解释操作内容及风险
   - 获得用户明确同意（如“是”、“执行”、“确认”）

3. **严格写操作白名单**  
   只有以下 `kubectl` 命令允许通过 `fix --confirm` 执行：
   - `kubectl rollout undo ...` — 回滚 Deployment
   - `kubectl rollout restart ...` — 重启 Deployment 的 Pod
   - `kubectl scale ...` — 扩缩容 Deployment
   - `kubectl delete pod ...` — 删除特定 Pod（强制重启）
   - `kubectl cordon ...` — 将节点标记为不可调度
   - `kubectl uncordon ...` — 将节点标记为可调度

   任何其他命令（包括 `kubectl exec`、`kubectl apply`、`kubectl edit` 等）均被拒绝。

4. **绝不执行 `kubectl exec`**  
   `kubectl exec` 不在白名单内，无法通过本技能执行。这防止了在容器内执行任意命令的最高风险操作。

5. **无凭证泄露**  
   本技能不会在输出中包含：
   - kubeconfig 文件路径或内容
   - ServiceAccount token
   - Secret 值（从不读取 Kubernetes Secret）
   - 云厂商凭证
   - 证书数据

6. **无命令注入风险**  
   - 脚本使用 `set -euo pipefail`
   - 所有 JSON 构造使用 `jq --arg` / `jq --argjson`，无字符串拼接
   - Shell 变量均使用双引号引用
   - `fix` 命令解析为数组并在执行前校验白名单
   - 禁止使用 `eval`、反引号、`bash -c` 等对用户输入求值的构造

## RBAC 最小权限建议

本技能的最佳实践是绑定一个**只读 ClusterRole**。以下为推荐的 RBAC 配置：

### 只读角色（诊断所需）

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: k8s-check-fix-readonly
rules:
  # sweep, pod, deploy, events
  - apiGroups: [""]
    resources: ["pods", "pods/log", "events", "nodes", "componentstatuses", "services"]
    verbs: ["get", "list"]
  - apiGroups: ["apps"]
    resources: ["deployments", "replicasets"]
    verbs: ["get", "list"]
  # resources（需要 metrics-server）
  - apiGroups: ["metrics.k8s.io"]
    resources: ["nodes", "pods"]
    verbs: ["get", "list"]
```

### 写入角色（可选，仅当需要修复功能时）

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: k8s-check-fix-write
rules:
  # rollout undo / restart
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["get", "list", "patch", "update"]
  - apiGroups: ["apps"]
    resources: ["deployments/rollback"]
    verbs: ["create"]
  # scale
  - apiGroups: ["apps"]
    resources: ["deployments/scale"]
    verbs: ["get", "update", "patch"]
  # delete pod
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["delete"]
  # cordon/uncordon
  - apiGroups: [""]
    resources: ["nodes"]
    verbs: ["get", "patch"]
```

**建议**：从只读角色开始，仅当需要自动修复时才绑定写入角色。始终通过 `--confirm` 机制保留人工批准环节。

## 本技能可以访问的内容

- Pod 状态、日志（当前和上一次）、事件
- 节点状态、条件、资源指标
- Deployment 规范、滚动状态、修订历史
- ReplicaSet 详情
- 集群事件（所有类型）
- 组件状态（如果可用）
- 通过 metrics-server 获取的资源使用情况（`kubectl top`）

## 本技能无法访问的内容

- Kubernetes Secret（从不读取）
- ConfigMap 内容（从不读取）
- 容器文件系统（无 `kubectl exec` 或 `kubectl cp`）
- 准入控制器配置
- etcd 数据
- 云厂商 API

## 威胁模型

| 威胁 | 缓解措施 |
|------|----------|
| 代理执行破坏性命令 | 写操作白名单 + 用户确认门禁 |
| 通过 `kubectl exec` 执行任意命令 | `exec` 不在白名单中，被无条件拒绝 |
| 输出泄露凭证 | 输出 JSON 不包含 secret、token、kubeconfig |
| 通过 Pod/Deployment 名称注入命令 | 所有值通过 `jq --arg` 传递；Shell 变量加引号 |
| 集群权限过大 | 提供最小 RBAC 建议；默认只读 |
| 日志中包含敏感数据 | 通过 RBAC 限制 `pods/log` 访问；建议应用层日志脱敏 |

## 关于 Pod 日志的注意事项

Pod 日志可能包含敏感信息（PII、token、内部 URL）。当本技能通过 `kubectl logs` 获取日志时，这些数据会传递给 LLM 进行分析。如果你的日志包含受监管数据：

- 在 RBAC 中限制 `pods/log` 访问范围到特定命名空间。
- 使用 `--namespace` 标志限定作用域。
- 在应用层实现日志脱敏。

