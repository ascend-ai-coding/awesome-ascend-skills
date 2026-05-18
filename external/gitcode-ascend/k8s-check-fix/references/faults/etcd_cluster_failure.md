# 故障类型：etcd 集群多数节点宕机 / 数据损坏

## 典型症状
- `kubectl` 命令无响应，所有资源的增删改查操作全部失败。
- API Server 日志持续报错 `etcd cluster is unavailable` 或 `etcdserver: request timed out`。
- 已运行的业务 Pod 不受影响，但无法调度新 Pod、无法执行滚动更新。
- `kubectl get componentstatuses`（如果可用）显示 etcd-0 等组件状态为 `Unhealthy`。

## 诊断步骤（使用现有工具）
1. **检查控制面组件状态**  
   - 若 `kubectl` 可用，执行 `kubectl get componentstatuses`（较新集群可能已移除）。
   - 如果无法访问 API，需直接登录到 master 节点查看 etcd 进程状态。
2. **查看 API Server 日志**（如果仍有访问）  
   - `kubectl logs -n kube-system <apiserver-pod>` 可能会包含 etcd 连接错误。
3. **确认 etcd 集群多数节点状态**  
   - 登录到每个 etcd 节点，检查 etcd 进程是否运行：`systemctl status etcd` 或 `ps aux | grep etcd`。
   - 检查 etcd 数据目录是否存在且可读：`ls -l /var/lib/etcd`。
4. **检查 etcd 集群成员**  
   - 在任一健康 etcd 节点上执行：`ETCDCTL_API=3 etcdctl member list`（需有证书）。

## 修复方案（按风险等级分级）

### 方案1：恢复多数 etcd 节点（如果仅进程停止）
- **操作**：启动已停止的 etcd 进程。
- **命令**（在对应节点上手动执行）：
  ```bash
  systemctl start etcd
  ```
- **风险**：低，仅恢复进程，数据完整。

### 方案2：从快照恢复数据（如果数据损坏）
- **操作**：使用最近的 etcd 快照恢复数据。
- **前提**：需要有快照文件（如 `snapshot.db`）。
- **命令**（手动执行，需在 etcd 节点上）：
  ```bash
  # 停止 etcd
  systemctl stop etcd
  # 备份损坏的数据目录
  mv /var/lib/etcd /var/lib/etcd.bak
  # 从快照恢复
  ETCDCTL_API=3 etcdctl snapshot restore snapshot.db --data-dir /var/lib/etcd
  # 启动 etcd
  systemctl start etcd
  ```
- **风险**：中，会丢失快照之后的增量数据；需确保快照来源可信。

### 方案3：重建 etcd 集群（如果多数节点永久损坏）
- **操作**：从单个健康节点重建 etcd 集群。
- **复杂操作**：通常需要按照 etcd 官方文档重建成员，或使用集群备份恢复。
- **风险**：高，可能导致数据丢失或集群完全不可用，需谨慎。

## 修复执行流程
1. **向用户展示诊断结论**  
   - 必须要跟用户明确说明 “etcd 集群多数节点不可用或数据损坏，导致 API Server 无法访问”。
2. **根据具体情况提出修复方案**  
   - 如果是进程停止 → 推荐方案1  
   - 如果是数据损坏且有快照 → 推荐方案2  
   - 如果多数节点永久损坏 → 需重建集群
3. **提示用户手动执行**  
   - 由于修复操作涉及底层系统命令，且需要登录到 master 节点，无法通过 `kubectl` 自动完成。  
   - 请用户根据提供的命令手动执行，或指导用户操作。
4. **验证修复效果**  
   - 执行 `kubectl get nodes` 确认 API 恢复。
   - 检查 etcd 健康状态：`etcdctl endpoint health`。

## 关联故障
- 如果 API Server 无法启动但 etcd 正常，可能涉及 API Server 证书问题 → [kube-apiserver 证书过期故障指南](apiserver_cert_expired.md)

## 注意事项（Gotchas）
- ❌ 错误：在 etcd 集群多数节点宕机时直接重启 API Server 或 kubelet。  
  ✅ 正确：先恢复 etcd 集群，再检查 API Server。
- ❌ 错误：在没有快照的情况下删除数据目录。  
  ✅ 正确：始终先备份损坏的数据目录，再尝试恢复。
- ⚠️ 使用 etcd 快照恢复时，确保数据目录权限正确（通常为 `etcd:etcd`），否则 etcd 无法启动。
