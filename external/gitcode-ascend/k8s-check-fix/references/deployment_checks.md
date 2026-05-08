# Deployment 通用检查细则

本文档提供 Deployment 诊断的系统化方法，适用于滚动更新卡住、扩容失败、回滚需求、Pod 版本不一致等场景。  
**使用建议**：当用户报告 Deployment 相关问题时，或通过 `sweep` 发现滚动更新异常时，可参考以下步骤进行分析。

---

## 1. 快速概览：获取 Deployment 信息

### 1.1 使用 `deploy` 子命令
```python
k8s_check_fix(subcommand="deploy", target="<deployment-name>", namespace="<namespace>")
```
返回的 JSON 包含：
- `deployment`：Deployment 详情（replica 计数、策略、容器配置等）
- `rollout_status`：滚动更新状态文本
- `rollout_history`：历史版本（Revisions）
- `replicasets`：关联的 ReplicaSet 列表（含镜像版本和 revision 标签）
- `events`：Deployment 相关事件

### 1.2 关注的关键字段
- `replicas.desired`：期望副本数
- `replicas.ready`：就绪副本数
- `replicas.available`：可用副本数
- `replicas.unavailable`：不可用副本数
- `generation`：期望的规格版本
- `observed_generation`：实际生效的版本（若小于 generation，说明未完全处理）
- `conditions`：滚动更新条件（如 `Progressing`、`Available`）

---

## 2. 根据滚动状态分类诊断

### 2.1 滚动更新卡住（ProgressDeadlineExceeded）

**典型症状**：
- `kubectl rollout status` 超时
- 事件中出现 `Failed to progress` 或 `ProgressDeadlineExceeded`
- `unavailableReplicas` > 0 持续不恢复

**常见原因**：
- 新版本 Pod 无法启动（CrashLoopBackOff、ImagePullBackOff）
- 健康检查（readinessProbe）持续失败
- Pod 中断预算（PDB）阻止了足够的 Pod 终止
- 资源不足，新 Pod 无法调度

**诊断步骤**：
1. 查看 `unavailableReplicas` 和 `conditions` 中 `Progressing` 的 `reason`。
2. 查看 `replicasets` 列表，找出最新的 ReplicaSet（`revision` 最大）：
   - 检查该 ReplicaSet 的 Pod 状态：`kubectl get pods -l <selector> --show-labels`。
   - 对问题 Pod 使用 `pod` 子命令深入排查。
3. 检查 Deployment 事件中的具体错误（如 `FailedCreate`、`FailedPod`）。
4. 检查 PDB：`kubectl get pdb` 确认是否阻止了滚动。

### 2.2 新 Pod 无法启动（CrashLoopBackOff / ImagePullBackOff）

**典型症状**：
- `replicas.ready` 低于 `replicas.desired`
- 最新 ReplicaSet 中的 Pod 状态异常

**诊断步骤**：
1. 从 `replicasets` 中找到最新的 ReplicaSet，获取其镜像版本。
2. 对异常 Pod 执行 `pod` 子命令，分析日志和事件。
3. 根据故障类型（CrashLoopBackOff、ImagePullBackOff）参考对应故障指南或 `pod_checks.md`。
4. 如镜像配置错误，可建议回滚。

### 2.3 健康检查失败导致无法就绪

**典型症状**：
- 新 Pod 处于 Running 状态但 `ready` 为 0
- `rollout_status` 显示 `waiting for rollout to finish`
- 事件中 `Readiness probe failed`

**诊断步骤**：
1. 检查最新 ReplicaSet 的 Pod 就绪状态。
2. 查看 Pod 事件中 readinessProbe 失败原因。
3. 检查容器内服务是否正常响应探测端点（如 `/health`）。
4. 建议调整 readinessProbe 配置（初始延迟、超时时间、失败阈值）或修复应用。

### 2.4 Pod 中断预算（PDB）阻止终止

**典型症状**：
- 滚动更新进度卡住，事件显示 `Eviction: Pod cannot be evicted due to PDB`
- `unavailableReplicas` 为 0，但 `desired` 未达到

**诊断步骤**：
1. 检查 PDB：`kubectl get pdb -n <namespace>`。
2. 查看 PDB 的 `status.currentHealthy` 和 `desiredHealthy`。
3. 如果 PDB 限制过严（例如 `minAvailable` 过高），滚动更新可能无法继续。
4. 建议临时调整 PDB 或增加副本数。

---

## 3. 回滚分析

### 3.1 何时建议回滚
- 新版本导致大量 Pod 崩溃
- 健康检查持续失败，无法上线
- 用户明确表示需要回到上一稳定版本

### 3.2 回滚步骤
1. **确认历史版本**：从 `rollout_history` 中查看可用的 revision。
2. **展示回滚命令**：
   ```bash
   kubectl rollout undo deployment/<deployment-name> -n <namespace> [--to-revision=<revision>]
   ```
3. **解释影响**：回滚会触发新的滚动更新，可能短暂中断服务。
4. **等待用户确认**，然后使用 `fix` 子命令执行（需 `--confirm`）。

### 3.3 回滚后验证
- 执行 `deploy` 子命令检查新状态。
- 确认 `replicas.available` 达到期望值。
- 检查事件是否正常。

---

## 4. 扩容与缩容问题

### 4.1 扩容失败
**典型症状**：
- `replicas.desired` 增加后，`ready` 副本数未增长
- 事件显示 `FailedCreate` 或调度失败

**诊断步骤**：
1. 检查节点资源是否充足（`resources` 子命令）。
2. 检查 PDB 是否阻止增加（罕见）。
3. 检查最新 ReplicaSet 的 Pod 调度事件。
4. 如有调度失败，参考 `node_checks.md` 或 `pod_checks.md`。

### 4.2 缩容失败
**典型症状**：
- 期望副本数减少，但 `ready` 副本数未减少
- 事件显示 `FailedDelete` 或终止 Pod 失败

**诊断步骤**：
1. 检查 PDB 是否阻止终止（`minAvailable` 限制）。
2. 检查 Pod 是否有 finalizers 阻止删除。
3. 检查 Pod 是否处于 Terminating 状态过久（可能卡在容器退出）。

---

## 5. 事件解读

### 5.1 Deployment 事件类型
| 事件原因 | 含义 |
|----------|------|
| `ScalingReplicaSet` | 扩容或缩容操作 |
| `FailedCreate` | 创建 Pod 失败（资源、调度、配置） |
| `FailedDelete` | 删除 Pod 失败 |
| `ProgressDeadlineExceeded` | 滚动更新超过截止时间 |
| `ReplicaSetCreateError` | 创建 ReplicaSet 失败 |
| `ReplicaSetUpdateError` | 更新 ReplicaSet 失败 |

### 5.2 关联分析
- 多个 `FailedCreate` 事件 + 节点资源不足 → 扩容或增加节点。
- `ProgressDeadlineExceeded` + Pod 处于 CrashLoopBackOff → 应用问题，需回滚。
- `FailedDelete` + PDB → PDB 限制过严。

---

## 6. ReplicaSet 对比

### 6.1 查看版本变化
- `replicasets` 列表按 `revision` 排序，最新 revision 通常最大。
- 对比不同 revision 的容器镜像：`replicasets[].containers[].image`。

### 6.2 镜像不一致
- 如果最新 revision 的镜像与用户预期不符，可能是：
  - 镜像标签更新但 Deployment 未重新部署
  - 回滚未完全生效
  - 手动修改了 ReplicaSet 导致偏离

### 6.3 清理旧 ReplicaSet
- 默认保留历史版本数量由 `spec.revisionHistoryLimit` 控制（默认 10）。
- 若磁盘空间紧张，可适当减少，但过少可能影响回滚能力。

---

## 7. 诊断决策树（供模型参考）

```
Deployment 问题
├── 滚动更新卡住
│   ├── 检查新 ReplicaSet Pod 状态
│   │   ├── CrashLoopBackOff → 查看日志，回滚或修复镜像/配置
│   │   ├── ImagePullBackOff → 检查镜像名称、认证
│   │   ├── Pending → 检查资源、节点、PVC
│   │   ├── Running 但未就绪 → 检查 readinessProbe
│   │   └── Terminating 卡住 → 检查 PDB、finalizers
│   └── 事件中无 Pod 创建 → 检查 PDB 或 API Server 错误
├── 新 Pod 运行正常但无法上线
│   └── 检查 readinessProbe 配置和应用健康
├── 扩容失败
│   └── 节点资源、调度事件、PDB
├── 缩容失败
│   └── PDB、Pod finalizers
└── 需要回滚
    └── 确认历史 revision，执行 rollout undo
```

---

## 8. 注意事项（Gotchas）

- ❌ **仅看 Deployment 状态而不看 ReplicaSet 和 Pod**：ReplicaSet 和 Pod 才是真实反映问题的载体。
- ❌ **在滚动更新卡住时直接重启 Deployment**：可能掩盖根本原因，应先诊断。
- ❌ **忽略 PDB 影响**：PDB 可能在用户不知情的情况下阻止滚动。
- ❌ **回滚后不验证**：回滚后仍需确认 Pod 状态和业务正常。
- ⚠️ **镜像标签使用 `latest`**：可能导致版本混乱，建议使用明确的版本标签。
- ⚠️ **revisionHistoryLimit 过小**：可能丢失回滚点，影响故障恢复。

---

通过以上步骤，模型可系统化地分析 Deployment 问题，结合 ReplicaSet、Pod 状态、事件和 PDB 等多维度数据，给出准确诊断和修复建议。当遇到复杂问题时，可进一步参考 Pod 或节点相关指南。
