# 网络通用检查细则

本文档提供 Kubernetes 网络问题的系统化诊断方法，适用于 Pod 无法通信、Service 访问异常、跨节点不通、DNS 解析失败等场景。  
**使用建议**：当用户报告网络相关问题时，或通过 Pod 状态发现网络错误（如 `failed to set up sandbox container network`）时，可参考以下步骤进行分析。

---

## 1. 网络栈概览

Kubernetes 网络涉及多个组件：
- **CNI 插件**（Calico、Flannel、Weave 等）：负责 Pod IP 分配、网络连通性
- **kube-proxy**：负责 Service 的 IP 负载均衡（iptables / IPVS 模式）
- **CoreDNS**：集群内部 DNS 解析
- **网络策略（NetworkPolicy）**：控制 Pod 之间或 Pod 与外部的访问规则

诊断时按层次排查：Pod 网络 → Service 网络 → DNS → 策略。

---

## 2. 快速概览：检查网络基础

### 2.1 使用现有工具
- **节点状态**：`kubectl get nodes`，检查是否有 `NetworkUnavailable` 条件。
- **CNI Pod 状态**：`kubectl get pods -n kube-system | grep -E 'calico|flannel|weave|cilium'`。
- **kube-proxy 状态**：`kubectl get pods -n kube-system | grep kube-proxy`。
- **CoreDNS 状态**：`kubectl get pods -n kube-system | grep coredns`。
- **Service 列表**：`kubectl get svc`。
- **Endpoint 列表**：`kubectl get endpoints`。

### 2.2 关注的关键指标
- CNI Pod 是否 Running，重启次数是否过高。
- kube-proxy 是否 Running，日志有无错误。
- CoreDNS 是否 Running，是否有 `READY` 列显示就绪数。
- Service 的 Endpoint 是否指向正确的 Pod IP。

---

## 3. Pod 网络问题

### 3.1 症状
- Pod 一直 `ContainerCreating`，事件提示 `failed to set up sandbox container network`。
- Pod 可以启动，但无法 ping 通其他 Pod 或 Service IP。
- 跨节点 Pod 通信失败，同节点正常。

### 3.2 诊断步骤
1. **检查 CNI 插件状态**：
   - 查看 CNI DaemonSet 的 Pod 是否 Running：`kubectl get pods -n kube-system -l k8s-app=calico-node`（或其他选择器）。
   - 查看 CNI Pod 日志：`kubectl logs -n kube-system <cni-pod>`，关注错误（如 BGP 邻居断开、IPAM 错误）。
2. **检查节点 CNI 配置**：
   - 登录节点，检查 `/etc/cni/net.d/` 目录是否存在配置文件。
   - 检查 `/opt/cni/bin/` 是否存在 CNI 二进制文件。
   - 查看 kubelet 日志中 CNI 相关错误：`journalctl -u kubelet | grep -i cni`。
3. **检查网络连通性**：
   - 从问题 Pod 内部 ping 另一个 Pod IP（需有 `ping` 工具，或使用 `kubectl exec` 进入调试容器）。
   - 若同节点 Pod 可通，跨节点不通 → 检查 CNI 的跨节点通信机制（如 Calico 的 BGP、Flannel 的 VXLAN）。
4. **检查节点防火墙**：确保节点之间允许 CNI 所需端口（如 4789 VXLAN、179 BGP、8472 Flannel 等）。

### 3.3 常见 CNI 问题
| 问题 | 可能原因 | 处理 |
|------|----------|------|
| CNI Pod CrashLoopBackOff | 镜像版本不兼容、配置错误 | 重新部署或回滚 CNI 版本 |
| 跨节点不通 | BGP 邻居未建立、VXLAN 隧道未创建 | 检查 CNI 日志，确认路由表 |
| 节点 NotReady 且 NetworkUnavailable | CNI 配置缺失或失败 | 检查 CNI DaemonSet 和节点配置 |

详细 CNI 故障处理可参考 [CNI 插件故障指南](faults/cni_failure.md)。

---

## 4. Service 访问问题

### 4.1 症状
- 通过 ClusterIP 无法访问服务。
- Service 的 Endpoints 显示 `<none>` 或 IP 列表不对。
- 访问 Service 的 NodePort 不通。

### 4.2 诊断步骤
1. **检查 Service 定义**：
   - `kubectl describe svc <service-name>`，查看 `Selector`、`Ports`、`Endpoints`。
2. **检查 Endpoints**：
   - `kubectl get endpoints <service-name>`，确认是否有正确的 Pod IP。
   - 如果 Endpoints 为空，检查 Pod 是否满足 selector 标签，且处于 `Running` 状态。
3. **检查 kube-proxy**：
   - 查看 kube-proxy 日志：`kubectl logs -n kube-system <kube-proxy-pod>`，关注错误（如无法连接 API Server、iptables 更新失败）。
4. **测试 Service 连通性**：
   - 从集群内任意 Pod 尝试访问 Service IP：`curl http://<service-ip>:<port>` 或 `wget -O- http://<service-ip>:<port>`。
   - 如果失败，检查 iptables 规则：在节点上执行 `iptables-save | grep <service-ip>`（iptables 模式）或 `ipvsadm -L -n`（IPVS 模式）。

### 4.3 常见 Service 问题
| 问题 | 可能原因 | 处理 |
|------|----------|------|
| Endpoints 为空 | Pod 标签不匹配、Pod 未就绪 | 修正 Pod 标签或等待就绪 |
| 访问 Service 超时 | kube-proxy 未更新规则、后端 Pod 不健康 | 重启 kube-proxy，检查后端 Pod 状态 |
| NodePort 不通 | 节点防火墙未放行端口、kube-proxy 未监听 | 检查防火墙规则，确认 kube-proxy 配置 |

---

## 5. DNS 解析问题

### 5.1 症状
- 容器内无法解析 Service 域名（如 `my-service.default.svc.cluster.local`）。
- `nslookup` 或 `dig` 超时。

### 5.2 诊断步骤
1. **检查 CoreDNS Pod 状态**：
   - `kubectl get pods -n kube-system -l k8s-app=kube-dns`（通常为 coredns）。
   - 查看 CoreDNS 日志：`kubectl logs -n kube-system <coredns-pod>`。
2. **检查 CoreDNS Service**：
   - `kubectl get svc -n kube-system kube-dns`，确认 ClusterIP 和端口（53）。
3. **测试 DNS 解析**：
   - 从测试 Pod 执行：`nslookup kubernetes.default.svc.cluster.local`。
   - 如果失败，检查 Pod 的 `/etc/resolv.conf` 是否配置了正确的 DNS 服务器（应为 CoreDNS Service IP）。
4. **检查网络策略**：确保 CoreDNS 的 Service 没有被 NetworkPolicy 阻断。

### 5.3 常见 DNS 问题
| 问题 | 可能原因 | 处理 |
|------|----------|------|
| CoreDNS Pod CrashLoopBackOff | 配置错误、无法访问 API Server | 检查 CoreDNS 配置，重启 Pod |
| 解析超时 | CoreDNS 负载过高、网络延迟 | 增加 CoreDNS 副本，检查节点资源 |
| 外部域名无法解析 | CoreDNS 上游 DNS 配置错误 | 检查 Corefile 中的转发配置 |

---

## 6. NetworkPolicy 影响

### 6.1 症状
- 本来可以通信的 Pod 突然无法访问。
- Service 访问被拒绝，但 Endpoints 正常。

### 6.2 诊断步骤
1. **检查是否启用了 NetworkPolicy**：
   - `kubectl get networkpolicies --all-namespaces`，查看是否有策略。
2. **分析策略规则**：
   - `kubectl describe networkpolicy <policy-name>`，查看 `spec.ingress` 和 `spec.egress`。
   - 确认策略是否允许目标流量（如允许特定命名空间或标签的 Pod）。
3. **临时禁用策略测试**：
   - 如有权限，可以删除或修改策略，观察问题是否消失（仅用于测试，需用户确认）。

### 6.3 常见 NetworkPolicy 问题
| 问题 | 可能原因 | 处理 |
|------|----------|------|
| 默认拒绝所有 | 未显式允许，导致所有流量被阻断 | 创建允许策略或修改现有策略 |
| 标签选择器错误 | 策略选择的 Pod 不符合预期 | 修正选择器 |
| 端口不匹配 | 策略未开放所需端口 | 调整策略端口范围 |

---

## 7. 跨节点通信问题

### 7.1 症状
- 同节点 Pod 可通信，跨节点不通。
- Pod 能 ping 通 Node IP，但无法 ping 通其他节点的 Pod IP。

### 7.2 诊断步骤
1. **检查 CNI 跨节点机制**：
   - 对于 Calico：检查 BGP 邻居状态（`calicoctl node status` 或从日志）。
   - 对于 Flannel：检查 VXLAN 隧道接口（`ip link show flannel.1`）和路由表。
2. **检查节点间网络**：
   - 登录节点，ping 另一个节点的 Node IP，确认基础网络连通。
   - 检查节点防火墙是否放行 CNI 所需端口（如 VXLAN 4789、BGP 179）。
3. **查看路由表**：
   - 在节点上执行 `ip route`，确认 Pod 网段的路由指向正确的下一跳（如 Calico 会为每个节点配置一条路由）。

### 7.3 常见跨节点问题
| 问题 | 可能原因 | 处理 |
|------|----------|------|
| BGP 邻居未建立 | 节点防火墙阻止、Calico 配置错误 | 检查端口、校准配置 |
| VXLAN 隧道不通 | 节点间 UDP 4789 被阻 | 开放端口或改用 host-gw 模式（如网络支持） |
| 路由表缺失 | CNI 未正确配置路由 | 重启 CNI Pod，检查 CNI 日志 |

---

## 8. 事件解读

### 8.1 Pod 网络相关事件
| 事件原因 | 含义 |
|----------|------|
| `FailedCreatePodSandBox` | 创建网络沙箱失败（CNI 问题） |
| `NetworkNotReady` | 节点网络未就绪（CNI 问题） |
| `Failed to set up sandbox container network` | CNI 调用失败 |

### 8.2 Service 相关事件
| 事件原因 | 含义 |
|----------|------|
| `Failed to create endpoint` | Service 创建 Endpoint 失败 |
| `Failed to update endpoint` | Endpoint 更新失败 |

### 8.3 关联分析
- `FailedCreatePodSandBox` + CNI Pod 异常 → CNI 问题。
- 大量 `Failed to update endpoint` + kube-proxy 日志错误 → kube-proxy 问题。
- 没有事件但 Service 不通 → 可能是 kube-proxy 规则未更新或防火墙。

---

## 9. 诊断决策树（供模型参考）

```
网络问题
├── Pod 无法创建（ContainerCreating）
│   ├── 事件提示 CNI 错误 → 检查 CNI 插件状态和配置
│   └── 节点 NetworkUnavailable → 检查 CNI DaemonSet 和节点日志
├── Pod 已运行但无法通信
│   ├── 同节点 Pod 可通，跨节点不通 → 检查 CNI 跨节点机制、节点间防火墙
│   └── 所有 Pod 不通 → 检查节点网络基础、kubelet 配置
├── Service 访问失败
│   ├── Endpoints 为空 → 检查 Pod 标签和就绪状态
│   ├── Endpoints 正确但访问不通 → 检查 kube-proxy、iptables/IPVS、后端 Pod 健康
│   └── NodePort 不通 → 检查节点防火墙、kube-proxy 配置
├── DNS 解析失败
│   ├── CoreDNS Pod 异常 → 检查 CoreDNS 日志和配置
│   └── Pod resolv.conf 配置错误 → 检查 DNS 策略
└── 策略导致阻断
    └── 检查 NetworkPolicy 规则，临时禁用测试
```

---

## 10. 注意事项（Gotchas）

- ❌ **直接重启 kubelet 或节点**：可能暂时缓解但掩盖根本问题，应先诊断 CNI 和 kube-proxy。
- ❌ **忽略防火墙规则**：云厂商的安全组或节点防火墙可能阻断 CNI 或 kube-proxy 所需端口。
- ❌ **假设所有 CNI 行为一致**：不同 CNI 诊断命令不同，需根据实际使用插件查阅文档。
- ❌ **在 NetworkPolicy 开启时默认允许所有**：策略可能已拒绝流量，需显式允许。
- ⚠️ **kube-proxy 模式差异**：iptables 模式和 IPVS 模式诊断命令不同，需确认当前模式。
- ⚠️ **CoreDNS 副本数不足**：高负载时可能出现解析延迟，建议至少 2 个副本并配置反亲和性。

---

通过以上步骤，模型可系统化地分析网络问题，从 Pod 网络、Service、DNS、策略到跨节点通信，层层递进。当遇到特定组件问题时，可进一步参考相关故障指南（如 CNI 故障、CoreDNS 配置等）。
