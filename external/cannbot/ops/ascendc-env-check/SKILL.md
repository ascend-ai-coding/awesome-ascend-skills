---
name: external-cannbot-ops-ascendc-env-check
description: Ascend C 算子开发环境检查技能。用于：(1) 通过 npu-smi 查询 NPU 设备信息（设备列表、状态、资源使用），(2) 检查
  CANN 环境配置（CANN Toolkit、Ops、自定义算子包），(3) 验证开发依赖是否完整。触发关键词：环境检查、NPU设备、npu-smi、CANN安装、设备查询、资源监控、检查CANN环境变量。
original-name: ascendc-env-check
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-19'
synced-commit: 943f3bfc36e24068e065ca7ace72fbff86f4a09c
license: UNKNOWN
---

# Ascend C 环境检查

快速检查开发环境配置和 NPU 设备状态。

## 工作流程

```
环境检查
    │
    ├─ NPU 设备检查
    │   └─ npu-smi list / scripts/npu_info.sh
    │
    └─ CANN 环境检查
        └─ scripts/check_env.sh
```

## NPU 设备检查

### 快速命令

```bash
# 查看设备详细信息（包含设备列表）
npu-smi info

# 监控设备资源
npu-smi top
```

### 脚本工具

```bash
# 综合 NPU 信息（推荐）
bash scripts/npu_info.sh
```

详细命令参数见 [npu_commands.md](references/npu_commands.md)，npu-smi 不可用时的回退方案见 [asys_commands.md](references/asys_commands.md)

## CANN 环境检查

```bash
# 完整环境检查（推荐）
bash scripts/check_env.sh
```

### 检查项

| 检查项 | 说明 | 必需 |
|--------|------|------|
| ASCEND_HOME_PATH | CANN Toolkit 路径 | 是 |
| ASCEND_OPP_PATH | CANN Ops 路径 | 运行时必需 |
| 自定义算子包 | op_api 库 | 运行自定义算子必需 |
| CANN 工具 | msprof/cannsim | 可选 |

> ⚠️ **注意**：官方环境变量为 `ASCEND_HOME_PATH`，不是 `ASCEND_HOME`。部分旧文档或示例代码可能使用 `ASCEND_HOME`，这是错误用法。

详细环境配置见 [env_config_guide.md](references/env_config_guide.md)

## 诊断脚本

| 脚本 | 用途 |
|------|------|
| `scripts/npu_info.sh` | NPU 设备信息综合查询 |
| `scripts/check_env.sh` | CANN 环境配置检查 |

## 常见问题

- **NPU 不可见**：检查 `npu-smi list` 是否能识别设备
- **算子运行失败**：优先运行 `check_env.sh` 检查环境配置
- **设备被占用**：使用 `npu-smi info` 查看设备状态

详细排查见 [troubleshooting.md](references/troubleshooting.md)