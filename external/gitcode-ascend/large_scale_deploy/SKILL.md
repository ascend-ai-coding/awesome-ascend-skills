---
name: external-gitcode-ascend-large_scale_deploy
description: 自动化大规模集群安装部署工具，用于 ascend-deployer 组件批量部署。当用户需要跨集群部署组件或执行批量安装操作时调用。
original-name: large_scale_deploy
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# 大规模集群部署

## 功能概述

用于 ascend-deployer 的大规模集群自动化安装部署工具。帮助用户在集群环境中跨多个节点部署各种 Ascend 组件。

## ascend-deployer 仓库

### 仓库地址

```
https://gitcode.com/Ascend/ascend-deployer
```

### 获取方式

```bash
git clone https://gitcode.com/Ascend/ascend-deployer.git
```

## 前置条件

使用此 skill 前，请确保满足以下要求：

1. **ascend-deployer 仓库**：工作空间中必须有 ascend-deployer 仓库
2. **Linux 环境**：当前环境必须是 Linux 系统
3. **清单配置**：用户必须已配置 `large_scale_inventory.ini` 文件
4. **网络访问**：能够 SSH 访问集群中所有目标节点

## 支持的组件

以下组件可以安装：

| 组件 | 说明 |
|------|------|
| ascend-device-plugin | Kubernetes 的 Ascend 设备插件 |
| ascend-docker-runtime | Ascend Docker 运行时 |
| ascend-operator | Kubernetes 的 Ascend 操作器 |
| clusterd | Ascend Clusterd 服务 |
| copy_pkgs | 包复制工具 |
| deepseek_cntr | DeepSeek 容器 |
| deepseek_pd | DeepSeek 并行分布式 |
| docker_images | Docker 镜像管理 |
| driver | NPU 驱动 |
| fault-diag | 故障诊断工具 |
| firmware | NPU 固件 |
| kernels | 内核包 |
| mindie_image | MindIE 镜像 |
| mindio | MindIO 库 |
| mindspore | MindSpore 框架 |
| nnae | 神经网络加速引擎 |
| nnrt | 神经网络运行时 |
| noded | Ascend Noded 服务 |
| npu | NPU 相关包 |
| npu-exporter | NPU 指标导出器 |
| python | Python 环境 |
| pytorch | PyTorch 框架 |
| resilience-controller | 弹性控制器 |
| sys_pkg | 系统包 |
| tensorflow | TensorFlow 框架 |
| toolbox | Ascend 工具箱 |
| toolkit | Ascend 工具包 |
| volcano | Volcano 调度器 |

## 使用方法

脚本位置：`ascend-deployer/large_scale_install.sh`

### 单组件安装

```bash
cd ascend-deployer
bash large_scale_install.sh --install=<组件名>
```

示例 - 安装固件：
```bash
bash large_scale_install.sh --install=firmware
```

### 多组件安装

```bash
bash large_scale_install.sh --install=<组件1>,<组件2>,<组件3>
```

示例 - 安装固件和内核：
```bash
bash large_scale_install.sh --install=firmware,kernels
```

示例 - 安装驱动、固件和工具包：
```bash
bash large_scale_install.sh --install=driver,firmware,toolkit
```

## 工作流程

### 步骤 1：验证前置条件

1. 检查 ascend-deployer 仓库是否存在
2. 验证是否为 Linux 环境
3. 确认 `large_scale_inventory.ini` 已配置

### 步骤 2：验证清单配置

确保 `large_scale_inventory.ini` 包含：
- 所有目标节点的 IP 地址
- SSH 连接信息
- 节点分组和角色

### 步骤 3：选择组件

根据以下因素帮助用户选择合适的组件：
- 部署目的（训练、推理、开发）
- 硬件配置
- 软件需求

### 步骤 4：执行安装

进入 ascend-deployer 目录，使用选定的组件运行安装命令：
```bash
cd ascend-deployer
bash large_scale_install.sh --install=<选定的组件>
```

### 步骤 5：验证安装

安装完成后，验证：
- 所有组件已正确安装
- 服务在目标节点上运行
- 安装日志中没有错误

## 常见安装场景

以下命令需在 `ascend-deployer` 目录下执行：

### 场景 1：基础 NPU 环境搭建

在新集群上搭建基础 NPU 环境：
```bash
bash large_scale_install.sh --install=driver,firmware,toolkit,toolbox
```

### 场景 2：深度学习训练环境

用于深度学习训练工作负载：
```bash
bash large_scale_install.sh --install=driver,firmware,toolkit,pytorch,nnae
```

### 场景 3：Kubernetes 部署

用于基于 Kubernetes 的部署：
```bash
bash large_scale_install.sh --install=ascend-device-plugin,ascend-docker-runtime,ascend-operator,volcano
```

### 场景 4：全栈安装

安装完整的 Ascend 软件栈：
```bash
bash large_scale_install.sh --install=driver,firmware,toolkit,toolbox,nnae,nnrt,pytorch,tensorflow,mindspore
```

### 场景 5：监控设置

用于集群监控：
```bash
bash large_scale_install.sh --install=npu-exporter,fault-diag
```

## 清单文件格式

`large_scale_inventory.ini` 文件应遵循以下格式：

```ini
[master]


[worker]
1.1.1.1-1.1.1.9 ansible_ssh_user="root" ansible_ssh_pass="test1234" step_len=3 test_expr="master-{ip}-{int(index)+1}-y"


[deploy_node]


[npu_node]


[large_scale]
SUB_GROUP_MAX_SIZE=5


[all:vars]
```

### 清单文件字段说明

| 字段 | 说明 |
|------|------|
| [master] | 主节点分组 |
| [worker] | 工作节点分组，支持 IP 范围表示法（如 1.1.1.1-1.1.1.9） |
| [deploy_node] | 部署节点分组 |
| [npu_node] | NPU 节点分组 |
| [large_scale] | 大规模部署配置 |
| SUB_GROUP_MAX_SIZE | 子组最大节点数 |
| ansible_ssh_user | SSH 用户名 |
| ansible_ssh_pass | SSH 密码 |
| step_len | 步长 |
| test_expr | 测试表达式模板 |

## 故障排查

### 常见问题

1. **SSH 连接失败**
   - 验证 SSH 密钥是否正确配置
   - 检查到目标节点的网络连通性
   - 确保 ansible_user 具有适当的权限

2. **组件安装失败**
   - 检查目标节点上的可用磁盘空间
   - 验证仓库源是否可访问
   - 查看安装日志以获取具体错误

3. **服务无法启动**
   - 检查服务状态：`systemctl status <服务名>`
   - 查看服务日志：`journalctl -u <服务名>`
   - 验证配置文件是否正确

## 注意事项

1. **安装顺序**：某些组件存在依赖关系。建议按以下顺序安装：
   - sys_pkg（系统包）
   - driver（NPU 驱动）
   - firmware（NPU 固件）
   - toolkit, toolbox（工具）
   - kernels（内核包）
   - nnae, nnrt（运行时）
   - 框架（pytorch, tensorflow, mindspore）

2. **并行安装**：可以在单个命令中安装多个组件以提高效率

3. **回滚**：如果安装失败，请查看 ascend-deployer 文档中的回滚说明

4. **日志**：安装日志通常存储在 `/var/log/ascend-deployer/`

## 相关文件

| 文件 | 说明 |
|------|------|
| large_scale_install.sh | 主安装脚本 |
| large_scale_inventory.ini | 集群节点清单配置 |
| ansible.cfg | Ansible 配置文件 |
| playbooks/ | Ansible playbook 目录 |
