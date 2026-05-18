# 诊断指南索引

本目录包含 Kubernetes 集群诊断与修复的专项故障指南和通用检查细则，采用渐进式披露设计，供 AI 模型在执行诊断时按需查阅。

---

## 专项故障指南（`faults/` 目录）

这些文件针对已知的控制平面或节点故障，提供从症状到修复的完整路径。

| 文件 | 适用场景 | 核心症状 |
|------|----------|----------|
| `etcd_cluster_failure.md` | etcd 集群多数节点宕机或数据损坏 | kubectl 命令无响应；API Server 日志报 etcd 不可用；无法调度新 Pod |
| `apiserver_cert_expired.md` | kube-apiserver 证书过期 | kubectl 报 x509 证书过期；所有控制面组件无法连接 API Server |
| `scheduler_failure.md` | kube-scheduler 进程异常或配置错误 | 新 Pod 持续 Pending，无调度事件；事件提示 no scheduler registered |
| `worker_node_down.md` | Worker 节点宕机 | 节点状态 NotReady；节点上 Pod 变为 Unknown/Terminating |
| `kubelet_cert_expired.md` | kubelet 证书过期 | 节点 NotReady；kubelet 日志报 x509 证书过期 |
| `cni_failure.md` | CNI 插件故障 | Pod 处于 ContainerCreating；事件提示 failed to set up sandbox container network；跨节点通信中断 |

---

## 通用检查细则

这些文件提供系统化的诊断方法，适用于未明确故障类型或需要深入排查的场景。

| 文件 | 用途 | 适用场景 |
|------|------|----------|
| `pod_checks.md` | Pod 通用诊断流程 | Pod 状态异常（CrashLoopBackOff、Pending、ImagePullBackOff 等） |
| `node_checks.md` | 节点通用诊断流程 | 节点 NotReady、资源压力（MemoryPressure、DiskPressure） |
| `deployment_checks.md` | Deployment 通用诊断流程 | 滚动更新卡住、扩缩容失败、回滚需求 |
| `network_checks.md` | 网络通用诊断流程 | Pod 网络不通、Service 访问异常、DNS 解析失败 |
| `security_notes.md` | 模型安全行为准则 | 所有诊断和修复操作执行前参考，确保安全合规 |

---

## 使用建议

1. **先执行 `sweep`** 获取整体健康概览。
2. **根据主要症状**，在“专项故障指南”中查找匹配的故障类型，快速定位问题。
3. **若问题不明确或涉及多个组件**，参考“通用检查细则”进行系统性排查。
4. **任何修复操作前**，务必查阅 `security_notes.md` 确认安全流程。
5. **遇到已知错误模式**，可对照 `gotchas.md` 避免常见陷阱。

通过按需加载这些指南，AI 助手可高效完成从诊断到修复的完整闭环。