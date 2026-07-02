---
name: external-cannbot-ops-ascendc-env-check
description: Ascend C 算子开发环境检查技能。用于：(1) 通过 npu-smi 查询 NPU 设备信息（设备列表、状态、资源使用），(2) 检查
  CANN 环境配置（CANN Toolkit、Ops、自定义算子包），(3) 验证开发依赖是否完整，(4) 运行时检测当前设备 NPU 架构。触发关键词：环境检查、NPU设备、npu-smi、CANN安装、设备查询、资源监控、检查CANN环境变量、NPU架构、npu
  arch。
original-name: ascendc-env-check
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-02'
synced-commit: 342679f37ec1b052a8b64d6c9fd59f67e0c89073
license: UNKNOWN
---

# Ascend C 环境检查

快速检查开发环境配置和 NPU 设备状态。

## 工作流程

```
环境检查
    │
    ├─ NPU 设备检查
    │   └─ npu-smi info -m / scripts/npu_info.sh
    │
    ├─ CANN 环境检查
    │   └─ scripts/check_env.sh
    │
    └─ NPU 架构检测
        └─ scripts/get_npu_arch.py
```

## NPU 设备检查

### 快速命令

```bash
# 查看设备详细信息（包含设备列表）
npu-smi info

# 监控设备资源
npu-smi info -t usages -i <device_id>
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
| CANN 版本 | 检测版本号及运行时依赖基线 | 建议确认 |
| ASCEND_OPP_PATH | CANN Ops 路径 | 运行时必需 |
| 自定义算子包 | op_api 库 | 运行自定义算子必需 |
| CANN 工具 | msprof/cannsim | 可选 |
| Simulator状态 | 检查必要的模拟器状态 | KirinX90、Kirin9030 等 Kirin 平台开发必需 |

> ⚠️ **注意**：官方环境变量为 `ASCEND_HOME_PATH`，不是 `ASCEND_HOME`。部分旧文档或示例代码可能使用 `ASCEND_HOME`，这是错误用法。

详细环境配置见 [env_config_guide.md](references/env_config_guide.md)，版本配套关系见其中「CANN 版本兼容性」章节

## NPU 架构检测

通过 `libascend_hal.so` 查询当前设备 NPU 架构：

```bash
# 输出 dav-{NpuArch} 格式（如 dav-3510）
python3 scripts/get_npu_arch.py

# 仅输出裸数值（如 3510）
python3 scripts/get_npu_arch.py --raw
```

**原理**：调用 `halGetChipInfo` 获取芯片型号 → 读取 `platform_config/{SocVersion}.ini` 的 `NpuArch` 字段 → 输出结果。

**依赖**：Ascend driver 和 CANN toolkit。

## 诊断脚本

| 脚本 | 用途 |
|------|------|
| `scripts/npu_info.sh` | NPU 设备信息综合查询 |
| `scripts/check_env.sh` | CANN 环境配置检查 |
| `scripts/get_npu_arch.py` | 运行时检测当前设备 NPU 架构 |

也可直接调用 `_npu_info.py` Python 脚本获取结构化数据，支持 `--json`（完整 JSON 输出）、`--list`（设备 ID 列表）、`--health`（健康状态）等参数。

## Kirin 平台开发

Kirin 系列芯片（KirinX90、Kirin9030等以 Kirin 开头的平台）是端侧 AI 处理器，当前主要支持使用模拟器 Simulator 的开发方式，需要注意环境检查结果中的 Simulator 支持情况，**如不支持则环境检查结论是不支持 Kirin 平台开发，需要强调并告知用户。**

当前 Kirin 开发使用的 CANN 版本和服务器有差异，Kirin 系列芯片开发，需要安装对应的 mobile-station 版本的 CANN 才有 Kirin 的 Simulator。

详细内容（mobile-station CANN 的安装方法、常见问题）见 [kirin_platform_guide.md](references/kirin_platform_guide.md)

## 常见问题

- **NPU 不可见**：先执行 `npu-smi info -m` 检查设备映射表，再排查驱动是否安装正确
- **算子运行失败**：**优先**运行 `check_env.sh` 检查环境配置是否完整，并检查关键环境变量（`ASCEND_HOME_PATH`、`ASCEND_OPP_PATH`）是否已正确设置
- **确认是否有进程占用 NPU**：使用 `npu-smi info -t usages -i <device_id>` 查看运行中的进程；注意空闲设备仍会有少量 HBM 被驱动占用（正常现象），不应误判为设备被占用

详细排查见 [troubleshooting.md](references/troubleshooting.md)