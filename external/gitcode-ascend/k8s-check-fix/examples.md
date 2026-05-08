# 使用示例

本文档展示 `k8s-check-fix` 技能的典型使用场景，涵盖集群控制面故障、Pod 异常、滚动更新卡住、节点问题、网络故障及远程执行模式。每个示例包含用户提问、AI 诊断过程、技能调用及修复确认的简化交互流程。

---

## 1. 集群控制面故障（etcd 多数节点宕机）

**用户**  
> 我的生产集群 `kubectl` 无响应，现有 Pod 还在，但无法部署新应用，请帮忙排查。

**AI 诊断**  
- 调用 `sweep` 无响应 → 判断控制面故障。  
- 读取 `guides/faults/etcd_cluster_failure.md`，建议登录控制平面节点检查 etcd 进程。  
- 发现两个 etcd 进程已停止，集群仅剩一个节点。

**AI 输出**  
```
🏥 集群诊断报告

**根本原因**：etcd 集群多数节点宕机（3 节点中 2 个 etcd 进程停止），导致 API Server 无法访问。

**恢复建议**：
1. 启动已停止的 etcd 进程：
   
   systemctl start etcd
 
2. 若数据损坏，需从快照恢复。

**风险**：启动进程后集群即可恢复，无数据丢失风险。
```

**用户确认后**，手动执行命令，集群恢复。

---

## 2. Pod 频繁重启（CrashLoopBackOff / OOMKilled）

**用户**  
> `api` Pod 一直在崩溃，日志里出现 OOM 错误，怎么解决？

**AI 诊断**  
- 调用 `pod` 子命令获取详细信息：
  ```
  k8s_check_fix(subcommand="pod", target="api-7f8d4-x2k9p", namespace="prod")
  ```
- 返回 JSON 显示：`last_state.terminated.reason = "OOMKilled"`，`previous_logs` 中有内存分配失败信息。

**AI 输出**  
```
## 🏥 Pod 详细检查：api-7f8d4-x2k9p

**诊断**：容器因 OOMKilled 退出，当前内存限制 128Mi，日志显示应用需要 256Mi。

**修复建议**：增加内存限制至 256Mi：

kubectl set resources deployment/api -c api --limits=memory=256Mi -n prod


是否执行？
```

**用户确认后**，AI 调用 `fix` 子命令执行修复，Pod 恢复正常。

---

## 3. Deployment 滚动更新卡住

**用户**  
> 更新镜像后，Deployment `my-app` 滚动更新卡住，新 Pod 起不来，旧 Pod 还在。

**AI 诊断**  
- 调用 `deploy` 子命令：
  ```
  k8s_check_fix(subcommand="deploy", target="my-app", namespace="prod")
  ```
- 输出显示：`unavailableReplicas: 1`，事件 `FailedCreate` 提示 `exceeded quota`。

**AI 输出**  
```
📦 Deployment 分析：my-app

**滚动状态**：期望 3 副本，就绪 2，不可用 1，超时。

**事件**：新 Pod 创建失败，原因：ResourceQuota 不足。

**建议**：
- 调整 ResourceQuota 增加资源，或
- 回滚到上一版本：
 
  kubectl rollout undo deployment/my-app -n prod
  

请选择操作：A. 调整配额（需手动） B. 回滚
```

**用户选择 B**，AI 执行回滚，滚动恢复。

---

## 4. 节点 NotReady（kubelet 证书过期）

**用户**  
> 一个 worker 节点状态为 `NotReady`，其他节点正常，请帮助排查。

**AI 诊断**  
- 调用 `sweep` 发现 `node-3` 为 `NotReady`。  
- 读取 `guides/node_checks.md`，建议登录节点查看 kubelet 日志。  
- 用户提供节点访问权限，AI 执行：
  ```bash
  ssh node-3 "journalctl -u kubelet --tail=50 | grep certificate"
  ```
  输出：`x509: certificate has expired`

**AI 输出**  
```
🏥 节点诊断报告：node-3

**根本原因**：kubelet 证书过期，节点无法注册。

**恢复建议**：

systemctl restart kubelet

（若自动轮转失败，需手动备份证书目录后重启）

是否执行？
```

**用户确认后**，手动重启 kubelet，节点恢复 `Ready`。

---

## 5. CNI 网络故障

**用户**  
> 新 Pod 一直 `ContainerCreating`，跨节点 Pod 无法 ping 通。

**AI 诊断**  
- 调用 `sweep` 发现问题 Pod 状态为 `ContainerCreating`，事件提示 `failed to set up sandbox container network`。  
- 读取 `guides/network_checks.md` 和 `guides/faults/cni_failure.md`。  
- 检查 CNI 插件 Pod（Calico）状态：`CrashLoopBackOff`，日志显示 `Failed to create BGP client`。

**AI 输出**  
```
🌐 网络诊断报告

**根本原因**：Calico CNI 插件故障，BGP 无法建立，导致 Pod 网络无法配置。

**恢复建议**：重启 Calico DaemonSet

kubectl rollout restart daemonset/calico-node -n kube-system


是否执行？
```

**用户确认后**，AI 调用 `fix` 子命令执行重启，网络恢复正常。

---

## 6. 远程执行模式

**用户**  
> 我的 kubectl 安装在跳板机 `bastion.example.com` 上，无法将 kubeconfig 暴露给本地。能用这个技能远程诊断吗？

**AI 回答**  
可以，使用 `remote_host` 参数即可。例如：

```
k8s_check_fix(subcommand="sweep", remote_host="user@bastion.example.com", remote_key="~/.ssh/bastion.pem", context="prod")
```

技能会通过 SSH 在远程主机执行 `kubectl`，并将 JSON 结果返回，无需传输 kubeconfig。所有诊断和修复命令同样遵循安全规则，写操作仍需用户确认。

---

通过以上示例，你可以快速了解如何在各类故障场景中使用 `k8s-check-fix` 技能，实现从诊断到安全修复的闭环。
