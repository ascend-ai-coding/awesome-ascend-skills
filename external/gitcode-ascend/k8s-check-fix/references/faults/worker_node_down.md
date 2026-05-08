# 故障类型：Worker 节点宕机

## 典型症状
- `kubectl get nodes` 显示该节点状态为 `NotReady`。
- 节点上的 Pod 状态变为 `Unknown` 或 `Terminating`。
- Deployment/StatefulSet 控制器会在其他正常节点重建 Pod（如果副本数足够）。
- 业务可能出现短暂中断（如果该节点运行了唯一的 Pod 实例）。

## 诊断步骤（使用现有工具）
1. **确认节点状态**  
   - `kubectl get nodes` 查看节点是否为 `NotReady`。
2. **检查节点上的 Pod 状态**  
   - `kubectl get pods -o wide | grep <node-name>`，查看 Pod 状态。
3. **检查节点事件**  
   - `kubectl describe node <node-name>`，查看最近事件（如 `NodeNotReady`、`NodeRebooted`）。
4. **排查节点实际运行情况**（需登录节点或通过云控制台）  
   - 确认节点是否可 ping 通，SSH 是否可连接。
   - 检查 kubelet 服务状态：`systemctl status kubelet`。
   - 检查节点资源（CPU、内存、磁盘）是否耗尽。

## 修复方案（按风险等级分级）

### 方案1：恢复节点操作系统（如宕机原因可修复）
- **操作**：重启节点，或修复硬件/网络故障。
- **命令**（手动执行）：
  ```bash
  # 如果节点可 SSH 连接
  systemctl restart kubelet
  # 或者
  reboot
  ```
- **风险**：低，节点恢复后 kubelet 会自动注册并启动 Pod。

### 方案2：从集群中删除节点（如果节点永久损坏）
- **操作**：先驱逐节点上的 Pod，再删除节点。
- **命令**（手动执行，需 `kubectl` 可用）：
  ```bash
  # 驱逐节点上所有 Pod（会触发重建）
  kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data
  # 删除节点
  kubectl delete node <node-name>
  ```
- **风险**：中，驱逐过程可能导致 Pod 短暂中断；删除节点后，需要重新加入集群。

### 方案3：修复 kubelet 配置（如证书问题）
- 如果节点 `NotReady` 是由于 kubelet 证书过期或配置错误，参考 [kubelet 证书过期故障指南](kubelet_cert_expired.md)。

## 修复执行流程
1. **向用户展示诊断结论**  
   - 说明节点宕机，当前该节点上的 Pod 已无法访问，部分工作负载已自动迁移。
2. **提出修复建议**  
   - 如果是临时故障 → 推荐方案1，恢复节点。  
   - 如果是永久性硬件故障 → 推荐方案2，从集群中移除节点。
3. **等待用户确认**  
   - 如果执行 `drain` 和 `delete node`，需用户明确同意（因为会触发 Pod 迁移，可能影响业务）。
4. **验证修复效果**  
   - 节点恢复后，确认节点状态变为 `Ready`，Pod 重新运行。

## 关联故障
- 如果节点恢复后仍显示 `NotReady`，可能涉及 kubelet 证书过期 → [kubelet 证书过期故障指南](kubelet_cert_expired.md)
- 如果节点资源不足导致 Pod 被驱逐 → [pending_no_resources.md](pending_no_resources.md)

## 注意事项（Gotchas）
- ❌ 错误：节点宕机后立即删除节点，不先执行 `drain`，导致 Pod 未正常终止。  
  ✅ 正确：先执行 `drain`（如果 API 可用），确保 Pod 在其他节点重建。
- ❌ 错误：在单副本应用所在节点宕机时，手动创建 Pod 试图恢复，但忘记删除原 Pod，导致 IP 冲突。  
  ✅ 正确：等待控制器自动重建，或手动删除原 Pod 的 finalizer。
- ⚠️ `kubectl drain` 默认不驱逐 DaemonSet Pod，需加 `--ignore-daemonsets`。如果节点上有重要的本地存储数据，需额外处理。
