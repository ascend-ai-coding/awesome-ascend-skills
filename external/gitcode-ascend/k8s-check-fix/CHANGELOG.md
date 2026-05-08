# 更新日志

本文件记录 `k8s-check-fix` 技能的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
并遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.0.0] — 2026-03-25

### 新增
- **首次发布** `k8s-check-fix` 技能，提供 Kubernetes 集群诊断与安全修复能力。
- **六个诊断子命令**：
  - `sweep` – 全集群健康检查（节点、问题 Pod、告警事件、组件状态）。
  - `pod` – Pod 深入排查（描述、日志、上一次日志、事件、镜像版本差异）。
  - `deploy` – Deployment 分析（滚动状态、历史版本、ReplicaSet、事件）。
  - `resources` – 资源压力检测（节点使用率、Top Pod、缺少限制的 Pod）。
  - `events` – 近期事件（汇总统计、Top 原因）。
  - `fix` – 安全修复（需 `--confirm`）严格白名单（`rollout undo`、`rollout restart`、`scale`、`delete pod`、`cordon`、`uncordon`）。
- **默认只读** – 所有诊断命令仅执行只读操作，不修改集群状态。
- **用户确认修复** – 写操作必须展示完整命令并等待用户明确同意。
- **远程执行支持** – 通过 SSH 在跳板机上运行 `kubectl`（`--remote-host`、`--remote-key`、`--remote-user`）。
- **多集群支持** – 使用 `--context` 切换集群上下文。
- **结构化 JSON 输出** – 所有命令输出均为 JSON，便于 AI 解析。
- **RBAC 错误检测** – 自动识别权限不足并给出友好提示。
- **环境预检** – 检查 `kubectl`、`jq` 和集群连通性（包括远程 SSH）。
- **渐进式披露结构**：
  - `guides/faults/` – 控制平面和节点故障的详细恢复指南：
    - etcd 集群故障
    - API Server 证书过期
    - kube-scheduler 故障
    - Worker 节点宕机
    - kubelet 证书过期
    - CNI 插件故障
  - `guides/pod_checks.md` – Pod 诊断系统化流程。
  - `guides/node_checks.md` – 节点健康分析。
  - `guides/deployment_checks.md` – Deployment 问题排查。
  - `guides/network_checks.md` – 网络问题诊断。
  - `guides/security_notes.md` – AI 模型安全行为准则。
- **配置文件支持** – `config.json` 持久化用户偏好（默认上下文、命名空间、只读模式）。
- **错题本** – `gotchas.md` 记录常见诊断错误，帮助模型避免陷阱。
- **输出模板** – 确保报告格式一致：
  - 通用诊断报告模板（sweep/resources/events）
  - Pod 详细检查模板
  - Deployment 分析模板
  - 修复计划确认模板
  - 输出风格指南
- **完善文档**：
  - `SKILL.md` – AI 助手使用指南。
  - `README.md` – 用户概览。
  - `SECURITY.md` – 安全策略、威胁模型、RBAC 建议。
  - `CHANGELOG.md` – 本文件。
- **模块化脚本架构**：
  - 主入口 `scripts/k8s-check-fix.sh`（参数解析与路由）。
  - 共享库 `scripts/lib/`（公共函数、k8s 封装、远程执行、预检）。
  - 子命令实现 `scripts/subcommands/`（sweep、pod、deploy、resources、events、fix）。

### 安全
- **严格写操作白名单** – 仅允许六种安全 kubectl 命令。
- **禁用 `kubectl exec`** – 完全阻止。
- **防注入** – 所有参数通过 `jq --arg` 传递，变量加双引号。
- **无凭证泄露** – 输出中不含 kubeconfig 路径、token、Secret 内容。
- **用户确认门禁** – 每次写操作均需用户明确同意，且完整展示命令。
- **会话级只读模式** – 可通过配置开启，禁止所有写操作。

### 依赖
- `kubectl`（本地或远程）
- `jq`（本地）
- SSH 客户端（仅远程执行模式）

