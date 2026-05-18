# 故障类型：kubelet 证书过期

## 典型症状
- 节点状态变为 `NotReady`，`kubectl get nodes` 显示该节点不可用。
- kubelet 日志持续报错：`x509: certificate has expired` 或 `failed to rotate certificates`。
- API Server 拒绝该节点的连接请求，节点无法上报状态。
- 已运行的 Pod 可能仍在该节点上运行，但无法与 API Server 通信（如 `kubectl logs` 失败）。

## 诊断步骤（使用现有工具）
1. **检查节点状态**  
   - `kubectl get nodes` 查看节点是否为 `NotReady`。
2. **查看 kubelet 日志**  
   - 登录节点执行：`journalctl -u kubelet | grep -i certificate`。
3. **检查证书有效期**  
   - 在节点上执行：
     ```bash
     openssl x509 -in /var/lib/kubelet/pki/kubelet-client-current.pem -noout -text | grep "Not After"
     ```
     查看过期时间。
4. **确认是否配置了自动轮转**  
   - 检查 `/var/lib/kubelet/config.yaml` 中 `rotateCertificates` 是否为 `true`。

## 修复方案（按风险等级分级）

### 方案1：自动轮转证书（如果已配置）
- **操作**：重启 kubelet，触发证书轮转。
- **命令**（手动执行）：
  ```bash
  systemctl restart kubelet
  ```
- **风险**：低，kubelet 会向 API Server 请求新证书（需 API Server 可访问）。

### 方案2：手动重新签发证书
- **适用场景**：自动轮转失败或未配置。
- **操作**：
  1. 备份旧证书：`mv /var/lib/kubelet/pki /var/lib/kubelet/pki.bak`
  2. 重启 kubelet，它会向 API Server 申请新证书（需 API Server 可用，且 kubelet 有权限生成证书签名请求）。
- **风险**：低，但需要 API Server 正常运行。

### 方案3：使用 kubeadm 更新节点证书（kubeadm 集群）
- **操作**：在 master 节点上执行：
  ```bash
  kubeadm certs renew node-<node-name>
  ```
  然后将新证书分发到节点。
- **风险**：中，需了解 kubeadm 证书管理。

## 修复执行流程
1. **向用户展示诊断结论**  
   - 说明 kubelet 证书已过期，导致节点无法注册。
2. **提出修复方案**  
   - 如果 API Server 正常，推荐方案1或方案2。  
   - 如果集群为 kubeadm 且 API Server 正常，推荐方案3。
3. **提示用户手动执行**  
   - 由于涉及节点文件系统和服务重启，需用户登录节点或通过远程执行。
4. **验证修复效果**  
   - 等待 kubelet 重启后，检查节点状态：`kubectl get nodes`。
   - 检查新证书有效期：`openssl x509 -in /var/lib/kubelet/pki/kubelet-client-current.pem -noout -text | grep "Not After"`。

## 关联故障
- 如果 API Server 证书同时过期 → 先修复 API Server 证书。
- 如果节点因证书过期而无法加入集群，但 kubelet 日志显示其他错误 → 参考 [worker_node_down.md](worker_node_down.md)

## 注意事项（Gotchas）
- ❌ 错误：在 kubelet 证书过期后，手动删除节点并重新加入，而不清理旧证书。  
  ✅ 正确：先更新证书，节点会自动恢复。
- ❌ 错误：重启 kubelet 前未检查 API Server 是否正常，导致轮转失败。  
  ✅ 正确：先确保 API Server 可用。
- ⚠️ 如果使用云厂商托管 Kubernetes，通常由云平台自动管理证书，节点恢复可能需要联系云支持。
