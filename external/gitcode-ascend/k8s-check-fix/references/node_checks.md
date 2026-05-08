# 节点通用检查细则

本文档提供节点诊断的系统化方法，适用于节点状态异常（如 `NotReady`）、资源压力、调度问题等场景。  
**使用建议**：当通过 `sweep` 发现节点异常，或用户报告节点相关问题时，可参考以下步骤进行分析。

---

## 1. 快速概览：获取节点基本信息

### 1.1 使用现有工具
- **节点状态**：`kubectl get nodes` 查看节点状态和角色。
- **节点详情**：`kubectl describe node <node-name>` 获取条件、资源、事件。
- **节点资源**：`kubectl top node`（需 metrics-server）查看实时 CPU/内存使用。

### 1.2 关注的关键字段
- `Status`：`Ready`、`NotReady`、`Unknown` 等。
- `Conditions`：`Ready`、`MemoryPressure`、`DiskPressure`、`PIDPressure`、`NetworkUnavailable`。
- `Allocatable`：节点可分配资源（CPU、内存、Pod 数）。
- `Taints`：节点污点，可能阻止 Pod 调度。
- `Events`：节点相关事件（如 `NodeNotReady`、`NodeRebooted`）。

---

## 2. 根据节点状态分类诊断

### 2.1 NotReady
**常见原因**：
- kubelet 未运行或崩溃
- 网络分区（节点与 API Server 通信失败）
- 节点资源耗尽（磁盘满、内存不足）
- kubelet 证书过期
- 节点操作系统故障

**诊断步骤**：
1. **登录节点**（如可访问）：
   - 检查 kubelet 状态：`systemctl status kubelet` 或 `journalctl -u kubelet --tail=100`。
   - 查看 kubelet 日志中的错误（如证书过期、连接 API Server 失败）。
2. **检查节点条件**：
   - `kubectl describe node <node>` 查看 `Conditions` 中的 `Ready` 状态详细信息。
   - 若 `MemoryPressure` 或 `DiskPressure` 为 True，说明资源紧张。
3. **检查网络连通性**：
   - 从节点尝试访问 API Server（使用 `curl -k https://<apiserver>:6443/healthz`）。
   - 检查防火墙、路由、CNI 插件状态。
4. **检查 kubelet 证书**：
   - 在节点上执行 `openssl x509 -in /var/lib/kubelet/pki/kubelet-client-current.pem -noout -text | grep "Not After"` 确认是否过期。

### 2.2 资源压力（MemoryPressure / DiskPressure / PIDPressure）
**常见原因**：
- 内存不足：Pod 使用内存超过节点容量
- 磁盘不足：容器日志、镜像、本地卷占用过多
- PID 压力：节点上运行的进程数超过限制

**诊断步骤**：
1. **查看资源使用**：
   - `kubectl top nodes` 查看 CPU/内存使用百分比。
   - 登录节点使用 `free -m`、`df -h`、`ps aux` 等命令确认。
2. **识别资源消耗大户**：
   - 使用 `kubectl top pods --all-namespaces` 查看 Pod 资源使用。
   - 检查是否有 Pod 未设置 limits 导致无限增长。
3. **检查 Pod 驱逐情况**：
   - `kubectl get events --field-selector reason=Evicted --all-namespaces`。
4. **清理资源**：
   - 删除不需要的容器镜像：`docker system prune -a`（或 containerd 对应命令）。
   - 删除已终止的 Pod：`kubectl delete pod --field-selector status.phase=Succeeded --all-namespaces`。
   - 调整 Pod 资源限制或增加节点容量。

### 2.3 节点被封锁（SchedulingDisabled / Cordoned）
**常见原因**：
- 运维人员手动执行 `kubectl cordon` 进行维护
- 节点出现故障自动封锁

**诊断步骤**：
1. 检查节点是否被封锁：`kubectl describe node <node>` 查看 `Unschedulable` 是否为 `true`。
2. 查看事件中是否有 `Node cordoned`。
3. 如需恢复，执行 `kubectl uncordon <node>`（需用户确认）。

### 2.4 节点频繁重启或宕机
**常见原因**：
- 硬件故障（磁盘、内存、电源）
- 操作系统内核崩溃
- kubelet 崩溃循环

**诊断步骤**：
1. 登录节点查看系统日志：`journalctl -xe --since "1 hour ago"`。
2. 检查节点 uptime：`uptime`。
3. 查看节点事件：`kubectl describe node <node>` 中 Events 可能有 `NodeRebooted`。
4. 若为云环境，检查云控制台硬件状态。

---

## 3. 节点条件详解

| 条件 | 含义 | 影响 |
|------|------|------|
| `Ready` | 节点健康，kubelet 可上报状态 | 为 True 时节点可接受 Pod |
| `MemoryPressure` | 节点内存不足 | 新的 Pod 可能无法调度，已有 Pod 可能被驱逐 |
| `DiskPressure` | 节点磁盘空间不足 | 新的 Pod 可能无法调度，已有 Pod 可能被驱逐 |
| `PIDPressure` | 节点进程数过多 | 新的 Pod 可能无法调度 |
| `NetworkUnavailable` | 节点网络配置不正确 | 通常由 CNI 插件设置，为 True 时 Pod 网络不可用 |

### 处理建议
- **MemoryPressure**：增加节点内存、减少 Pod 内存使用、调整 Pod 资源限制。
- **DiskPressure**：清理容器镜像、删除无用文件、扩展磁盘容量。
- **PIDPressure**：调整 `--max-pods` kubelet 参数，或减少 Pod 数量。
- **NetworkUnavailable**：检查 CNI 插件状态（参考 CNI 故障指南）。

---

## 4. 节点事件解读

### 常见事件

| 事件原因 | 含义 |
|----------|------|
| `NodeNotReady` | 节点变为 NotReady |
| `NodeReady` | 节点恢复 Ready |
| `NodeRebooted` | 节点重启（通常伴随 NotReady） |
| `NodeHasDiskPressure` | 磁盘压力触发 |
| `NodeHasMemoryPressure` | 内存压力触发 |
| `NodeHasPIDPressure` | PID 压力触发 |
| `NodeSchedulable` | 节点从不可调度变为可调度 |
| `NodeNotSchedulable` | 节点被封锁 |

### 关联分析
- 多个 `NodeNotReady` 事件 + 节点资源压力 → 可能是资源耗尽导致 kubelet 无响应。
- 单个节点频繁 `NodeNotReady` + `NodeRebooted` → 可能是硬件问题。
- 所有节点同时 `NodeNotReady` → 可能是 API Server 或网络故障。

---

## 5. kubelet 日志关键错误

### 常见错误及含义

| 日志内容 | 含义 |
|----------|------|
| `x509: certificate has expired` | kubelet 证书过期 |
| `failed to get node info from API server` | API Server 不可达 |
| `eviction manager: eviction thresholds met` | 资源压力触发驱逐 |
| `failed to start cni` | CNI 插件配置错误 |
| `node not found` | 节点在 API Server 中不存在或已被删除 |
| `failed to run Kubelet: failed to create kubelet` | kubelet 启动失败（通常配置错误） |

### 处理建议
- 证书过期 → 参考 [kubelet 证书过期故障指南](faults/kubelet_cert_expired.md)
- API Server 不可达 → 检查网络和 API Server 状态
- CNI 问题 → 参考 [CNI 插件故障指南](faults/cni_failure.md)
- 资源驱逐 → 参考资源压力部分

---

## 6. 污点与容忍度

### 6.1 常见污点
| 污点 | 含义 | 影响 |
|------|------|------|
| `node.kubernetes.io/not-ready` | 节点 NotReady | 节点默认添加，阻止 Pod 调度 |
| `node.kubernetes.io/unreachable` | 节点不可达 | 节点默认添加，阻止 Pod 调度 |
| `node.kubernetes.io/out-of-disk` | 磁盘压力 | 节点添加，阻止新 Pod 调度 |
| `node.kubernetes.io/memory-pressure` | 内存压力 | 节点添加，阻止新 Pod 调度 |
| `node.kubernetes.io/disk-pressure` | 磁盘压力 | 节点添加，阻止新 Pod 调度 |
| `node.kubernetes.io/pid-pressure` | PID 压力 | 节点添加，阻止新 Pod 调度 |
| `node.kubernetes.io/network-unavailable` | 网络不可用 | CNI 插件添加，阻止 Pod 网络 |

### 6.2 检查方法
- 查看节点污点：`kubectl describe node <node>` 中 `Taints` 部分。
- 查看 Pod 容忍度：`kubectl get pod <pod> -o yaml | grep tolerations -A 10`。

### 6.3 处理建议
- 如果节点有污点但需要调度 Pod，可添加容忍度（需谨慎）。
- 污点通常由节点控制器自动添加，修复底层问题后会自动移除。

---

## 7. 节点资源管理

### 7.1 可分配资源
- 节点总资源减去系统预留和 kubelet 预留，才是 Pod 可用的。
- 查看 `Allocatable` 字段：`kubectl describe node <node>`。

### 7.2 资源耗尽预防
- 设置合理的 Pod 资源 requests/limits。
- 使用资源配额（ResourceQuota）限制命名空间总量。
- 监控节点资源使用趋势，及时扩容。

---

## 8. 诊断决策树（供模型参考）

```
节点状态
├── NotReady
│   ├── 检查 kubelet 是否运行
│   ├── 检查网络连通性
│   ├── 检查证书是否过期
│   └── 检查节点条件（资源压力）
├── 资源压力
│   ├── MemoryPressure → 检查 Pod 内存使用，增加 limit 或节点内存
│   ├── DiskPressure → 清理镜像和日志，扩展磁盘
│   └── PIDPressure → 减少 Pod 数量，调整 kubelet 参数
├── 被封锁（Unschedulable）
│   └── 确认是否为维护，若需恢复则 uncordon
├── 频繁重启
│   └── 检查系统日志、硬件、云控制台
└── 网络不可用
    └── 检查 CNI 插件状态（参考 CNI 故障指南）
```

---

## 9. 注意事项（Gotchas）

- ❌ **仅看节点状态而不查看条件和事件**：`NotReady` 背后可能有多种原因，需综合条件分析。
- ❌ **在资源压力时直接驱逐 Pod**：应先诊断根本原因，如内存泄漏则应调整应用。
- ❌ **忽略污点对调度的影响**：Pod 处于 Pending 时，污点是常见原因。
- ❌ **手动删除节点后不清理**：删除节点前应执行 `kubectl drain` 确保 Pod 优雅迁移。
- ⚠️ **云托管集群**：节点可能由云平台管理，部分操作需通过云控制台进行。
- ⚠️ **日志敏感性**：kubelet 日志可能包含敏感信息，输出前提醒用户。

