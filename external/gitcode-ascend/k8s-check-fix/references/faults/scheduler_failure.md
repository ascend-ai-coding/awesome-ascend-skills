# 故障类型：kube-scheduler 进程异常 / 配置错误

## 典型症状
- 新创建的 Pod 持续处于 `Pending` 状态，且无任何调度事件（`kubectl describe pod` 中 Events 为空）。
- Pod 事件可能提示：`no scheduler is registered for pod`。
- 节点资源充足，但 Pod 无法被调度。
- 滚动更新、扩容等操作全部卡住，已有 Pod 运行正常。

## 诊断步骤（使用现有工具）
1. **检查 scheduler Pod 状态**  
   - `kubectl get pods -n kube-system | grep scheduler`，查看是否 Running。
2. **查看 scheduler 日志**  
   - `kubectl logs -n kube-system kube-scheduler-<xxx>`，查找错误信息（如配置解析失败、无法连接 API Server）。
3. **确认 scheduler 是否启用**  
   - 检查 scheduler 配置文件（通常位于 `/etc/kubernetes/manifests/kube-scheduler.yaml`）是否存在，内容是否有效。
4. **检查 API Server 是否正常**  
   - `kubectl get nodes` 确认 API Server 可访问。

## 修复方案（按风险等级分级）

### 方案1：重启 scheduler Pod（如果 Pod 存在但崩溃）
- **操作**：删除 scheduler Pod，由 kubelet 自动重建。
- **命令**：
  ```bash
  kubectl delete pod -n kube-system kube-scheduler-<xxx>
  ```
- **风险**：极低，重建过程中新 Pod 可能短暂无法调度，但已有 Pod 不受影响。

### 方案2：修复 scheduler 配置文件
- **适用场景**：scheduler 因配置错误启动失败（如配置了不存在的调度插件）。
- **操作**：
  1. 登录 master 节点，编辑 `/etc/kubernetes/manifests/kube-scheduler.yaml`。
  2. 修正错误配置（例如恢复默认配置）。
  3. 等待 kubelet 自动重启 scheduler（约 30 秒）。
- **风险**：低，仅影响调度，已有 Pod 不受影响。

### 方案3：重新部署 scheduler（极端情况）
- **操作**：如果 scheduler 静态 Pod 文件丢失，可以从集群备份恢复或使用 kubeadm 重新生成。
- **命令**（kubeadm 集群）：
  ```bash
  kubeadm init phase control-plane scheduler --config kubeadm-config.yaml
  ```
- **风险**：中，需确保配置文件正确。

## 修复执行流程
1. **向用户展示诊断结论**  
   - 说明 scheduler 进程异常或配置错误，导致 Pod 无法调度。
2. **根据情况提出修复方案**  
   - 如果 scheduler Pod 处于 CrashLoopBackOff → 查看日志，定位配置错误，推荐方案2。  
   - 如果 scheduler Pod 不存在 → 可能需要重建，推荐方案3。  
   - 如果 scheduler Pod 存在但未调度 → 可能是静态 Pod 文件问题。
3. **等待用户确认后执行**  
   - 若使用 `kubectl delete pod`，可通过 `fix` 子命令（需用户确认）执行。
   - 若需要编辑文件或执行 kubeadm 命令，提示用户手动操作。
4. **验证修复效果**  
   - 创建测试 Pod，确认调度正常：`kubectl run test --image=nginx --restart=Never --dry-run=client -o yaml | kubectl apply -f -`，然后检查状态。

## 关联故障
- 如果 API Server 不可用导致 scheduler 无法注册 → 先解决 API Server 问题。
- 如果调度失败事件显示资源不足或节点亲和性不匹配 → 参考 [pending_no_resources.md](pending_no_resources.md)

## 注意事项（Gotchas）
- ❌ 错误：直接重启 master 节点或 kubelet 而不先检查 scheduler 日志。  
  ✅ 正确：先通过日志定位具体原因，再修复。
- ❌ 错误：在修复 scheduler 之前，手动删除大量 Pending Pod，以为会重新调度。  
  ✅ 正确：修复 scheduler 后，这些 Pod 会自动尝试调度，无需删除。
- ⚠️ 如果 scheduler 配置了自定义调度器名称，确保 Pod 的 `spec.schedulerName` 匹配。
