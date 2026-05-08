# 故障类型：CNI 插件故障

## 典型症状
- 新创建的 Pod 持续处于 `ContainerCreating` 状态。
- `kubectl describe pod` 事件提示：`failed to set up sandbox container network` 或 `network plugin is not ready`。
- 跨节点 Pod 无法互相 ping 通，Service 访问异常。
- 已有 Pod 可能不受影响（如果网络插件已配置好），但新 Pod 无法获取 IP。

## 诊断步骤（使用现有工具）
1. **检查 CNI 插件 Pod 状态**  
   - 根据使用的 CNI 类型（Calico、Flannel、Weave 等），检查其 DaemonSet 是否正常运行：
     ```bash
     kubectl get pods -n kube-system | grep -E 'calico|flannel|weave'
     ```
2. **查看 CNI 插件日志**  
   - 获取 CNI Pod 日志，查找错误：`kubectl logs -n kube-system <cni-pod>`
3. **检查节点上的 CNI 配置**  
   - 登录节点，检查 `/etc/cni/net.d/` 目录下是否有配置文件。
   - 检查 CNI 二进制文件是否存在于 `/opt/cni/bin/`。
4. **确认节点网络状态**  
   - 查看节点上 kubelet 日志：`journalctl -u kubelet | grep -i cni`。

## 修复方案（按风险等级分级）

### 方案1：重启 CNI 插件 DaemonSet
- **操作**：删除 CNI 插件 Pod，让其自动重建。
- **命令**：
  ```bash
  kubectl delete pod -n kube-system <cni-pod>  # 删除所有 CNI Pod（或通过 rollout restart）
  ```
- **风险**：低，重建过程中网络可能短暂中断，但通常 Pod 会很快恢复。

### 方案2：重新部署 CNI 插件
- **适用场景**：CNI 配置文件丢失或损坏。
- **操作**：根据集群使用的 CNI 类型，重新应用其 DaemonSet YAML。
- **命令**（示例）：
  ```bash
  kubectl apply -f https://docs.projectcalico.org/manifests/calico.yaml
  ```
- **风险**：中，需确保 YAML 与集群版本兼容；可能短暂影响网络。

### 方案3：修复节点上的 CNI 配置
- **操作**：手动恢复 CNI 配置文件或二进制文件。
- **命令**（手动执行，需登录节点）：
  ```bash
  # 恢复备份的配置
  cp /etc/cni/net.d/10-<cni>.conf.bak /etc/cni/net.d/10-<cni>.conf
  # 重启 kubelet
  systemctl restart kubelet
  ```
- **风险**：低，仅影响该节点。

## 修复执行流程
1. **向用户展示诊断结论**  
   - 说明 CNI 插件故障导致 Pod 无法创建网络。
2. **根据问题提出修复方案**  
   - 如果 CNI Pod 处于 CrashLoopBackOff → 查看日志，修复配置或镜像问题（可能需重新部署）。  
   - 如果 CNI Pod 正常但节点上配置文件缺失 → 推荐方案3。  
   - 如果整个 CNI 插件未部署 → 推荐方案2。
3. **等待用户确认后执行**  
   - 若使用 `kubectl delete pod`，可通过 `fix` 子命令执行。  
   - 若需要重新部署或修改文件，提示用户手动操作。
4. **验证修复效果**  
   - 创建测试 Pod，检查网络连通性：`kubectl run test --image=nginx --restart=Never --rm -it -- sh`，尝试 ping 其他 Pod IP。

## 关联故障
- 如果 Pod 网络正常但 Service 访问异常 → 可能是 kube-proxy 问题（需单独排查）。
- 如果节点因网络插件故障而 NotReady → 参考 [worker_node_down.md](worker_node_down.md)

## 注意事项（Gotchas）
- ❌ 错误：在 CNI 插件故障时，尝试通过 `kubectl exec` 进入 Pod 调试（可能无法进入）。  
  ✅ 正确：优先检查 CNI 插件自身状态。
- ❌ 错误：误删 CNI 配置文件后未重启 kubelet，导致配置不生效。  
  ✅ 正确：修改配置后必须重启 kubelet。
- ⚠️ 不同 CNI 插件的恢复方式可能不同，如 Calico 需要额外检查 BGP 配置，Flannel 需要确认 overlay 网络配置等。在日志中查找具体错误并针对性解决。
