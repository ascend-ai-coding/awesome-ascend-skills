# 故障类型：kube-apiserver 证书过期

## 典型症状
- `kubectl` 访问集群报错：`x509: certificate has expired or is not yet valid`。
- 控制面组件（scheduler、controller-manager、kubelet）均无法连接 apiserver。
- 已运行 Pod 不受影响，但集群所有管控能力失效（无法创建/删除资源，无法查看部分资源）。
- API Server 日志中可能包含 `failed to verify certificate` 或 `certificate has expired`。

## 诊断步骤（使用现有工具）
1. **检查证书有效期**（如果仍有访问权限）  
   - `kubectl get --raw /api/v1/` 可能返回证书错误。
   - 直接查看 API Server 证书文件：`openssl x509 -in /etc/kubernetes/pki/apiserver.crt -noout -text`。
2. **确认是否所有控制面组件都无法连接**  
   - 查看 scheduler 或 controller-manager 日志（如果集群有这些 Pod）：
     ```bash
     kubectl logs -n kube-system kube-scheduler-<xxx>
     ```
     可能看到 `x509: certificate has expired`。
3. **检查节点 kubelet 是否正常**（部分节点可能仍在运行，但无法上报状态）  
   - `kubectl get nodes` 可能显示节点 `NotReady` 或连接错误。

## 修复方案（按风险等级分级）

### 方案1：使用 kubeadm 自动续期证书
- **适用场景**：集群使用 kubeadm 部署，且证书未超过 kubeadm 的自动续期范围。
- **操作**：在 master 节点上执行：
  ```bash
  kubeadm certs renew all
  ```
- **风险**：低，kubeadm 会重新签发所有证书，并更新 kubeconfig 文件。

### 方案2：手动重新签发证书
- **适用场景**：非 kubeadm 部署，或 kubeadm 自动续期失败。
- **操作**：
  1. 备份原有证书目录 `/etc/kubernetes/pki/`。
  2. 使用集群原有的 CA 证书重新签发 API Server 证书（需提供证书签名请求）。
  3. 将新证书覆盖到原位置。
  4. 重启 API Server、scheduler、controller-manager 等服务。
- **风险**：中，操作复杂，需熟悉证书签发流程。

### 方案3：替换整个集群证书（极端情况）
- **操作**：使用备份的 CA 证书或重新生成集群证书，并重启所有组件。
- **风险**：高，可能导致所有客户端需要更新 kubeconfig。

## 修复执行流程
1. **向用户展示诊断结论**  
   - 明确 API Server 证书已过期，导致控制面失效。
2. **根据集群部署方式提出建议**  
   - 如果是 kubeadm 集群，推荐方案1，并给出命令。  
   - 如果是手动部署，推荐方案2，并提示用户按集群文档操作。
3. **提示用户手动执行**（因为涉及系统文件操作）  
   - 要求用户登录 master 节点，执行命令（如 `kubeadm certs renew all`），然后重启控制面组件（或重启 kubelet 触发自动重启）。
4. **验证修复效果**  
   - 重新尝试 `kubectl get nodes`，确认 API 恢复正常。
   - 检查证书新有效期：`kubeadm certs check-expiration`（kubeadm 集群）。

## 关联故障
- 如果只有 kubelet 证书过期 → [kubelet 证书过期故障指南](kubelet_cert_expired.md)
- 如果 etcd 证书也过期，可能需要同时处理。

## 注意事项（Gotchas）
- ❌ 错误：直接重启 API Server 而不更新证书。  
  ✅ 正确：先更新证书，再重启组件。
- ❌ 错误：在 kubeadm 集群中使用手动签发方式覆盖 kubeadm 管理的证书，可能导致下次 `kubeadm renew` 失败。  
  ✅ 正确：优先使用 kubeadm 统一管理。
- ⚠️ 证书更新后，需要重启所有使用该证书的组件（API Server、scheduler、controller-manager、kubelet 等），否则旧证书仍会被缓存。
