---
name: external-cannbot-tools-msnpureport-toolkit
description: msnpureport 工具使用技能。用于：(1) 导出 Device 侧系统类日志和维测信息（单次导出、常驻进程连续导出），(2) 查询
  Device 维测配置（日志级别、Coremask、加速器复位、singlecommit），(3) 设置 Device 维测配置（日志级别、TaskSchedule
  自动复位、AI Core singlecommit、屏蔽 AI Core/Vector Core、icache bit 翻转校验范围），(4) 配置 msnpureport
  自身维测日志级别与展示位置，(5) 解读导出目录结构并定位日志文件。触发关键词：msnpureport、Device侧日志、导出Device日志、黑匣子、bbox、hisi_logs、stackcore、slog、连续导出日志、Device日志级别、Coremask、屏蔽AI
  Core、屏蔽Vector Core、accelerator_recover、singlecommit、icachecheck、msnpureport_auto_export.sh、vmcore、fault_event。
original-name: msnpureport-toolkit
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# msnpureport 工具

导出 Device 侧系统类日志与维测信息，并查询/设置 Device 侧维测配置。工具随昇腾驱动包部署，路径为 `{Driver安装目录}/driver/tools/msnpureport`，驱动部署完成后可在任意目录执行。

## 适用范围与前置条件

| 项 | 要求 |
|----|------|
| 产品形态 | 仅适用于 Ascend EP 形态；昇腾虚拟化实例场景不支持 |
| 用户权限 | 导出类与所有 `config --set` 类命令**仅支持 root 用户**执行；查询类命令普通用户可执行（驱动需以 `--install-for-all` 安装） |
| 执行目录 | 不要在加锁目录下执行（`lsattr` 显示 `i` 属性即加锁）。导出目录须对普通用户无访问权限，否则存在日志被删除或系统信息泄露风险 |
| 容器场景 | 需加 `--docker` 参数，并配置 `export PATH=/usr/local/Ascend/driver/tools:$PATH`。非特权容器仅支持导出 Host 侧驱动内核日志；连续导出不支持 |
| Device ID | 命令中的 `-d/--device` 均为**逻辑 ID**，非 `npu-smi info` 显示的物理 ID |

环境自检：`bash scripts/preflight.sh`

## 场景路由

| 场景 | 命令 | 详细参考 |
|------|------|---------|
| 复现后一次性收集日志 | `msnpureport report` | [export-one-time.md](references/export-one-time.md) |
| 任务运行前布防，持续收集 | `msnpureport report --permanent` | [export-permanent.md](references/export-permanent.md) |
| 查询当前 Device 维测配置 | `msnpureport config --get` | [config-query.md](references/config-query.md) |
| 调整 Device 日志级别 | `msnpureport config --set --log ...` | [config-set-log.md](references/config-set-log.md) |
| AI Core 故障定位（复位/串行/屏蔽核/icache） | `msnpureport config --set ...` | [config-set-aicore.md](references/config-set-aicore.md) |
| 调整工具自身日志级别与打屏 | `msnpureport report --print/--log_level` | [tool-log-config.md](references/tool-log-config.md) |
| 脚本方式连续导出（旧方式） | `msnpureport_auto_export.sh` | [auto-export-script.md](references/auto-export-script.md) |
| 看不懂导出目录 | — | [output-layout.md](references/output-layout.md) |

产品支持差异见 [product-support.md](references/product-support.md)，常见问题见 [troubleshooting.md](references/troubleshooting.md)。

## 核心命令速查

```bash
# 子命令与帮助
msnpureport help                 # 总帮助
msnpureport config --help        # config 子命令帮助
msnpureport report --help        # report 子命令帮助
msnpureport version              # 版本信息

# 单次导出（先 cd 到已创建的存放目录，如 /var/log/npu/report）
msnpureport report               # 默认集合：系统类日志 + 调用栈 + 黑匣子 + 事件调度 + Host内核日志 + Device OS日志
msnpureport report -d 0          # 指定 Device 逻辑 ID
msnpureport report -a            # 追加黑匣子设备事件信息
msnpureport report -f            # 再追加黑匣子存储空间历史维测信息（最全）
msnpureport report -t 2          # 只导黑匣子相关（含寄存器信息，不支持多进程并发）
msnpureport report -t 3          # 只导 coredump 调用栈

# 连续导出（root，任务启动前执行，Ctrl+C 或 kill -15 <pid> 结束）
msnpureport report --permanent
msnpureport report --permanent -d 0 -o mypath

# 查询配置
msnpureport config --get         # 全量配置（日志级别 + Coremask + 复位 + singlecommit + icache）
msnpureport -r                   # 仅查日志级别（旧方式）

# 设置 Device 日志级别
msnpureport config --set --log -g info          # 全局级
msnpureport config --set --log -m CCE:debug     # 模块级
msnpureport config --set --log -e enable        # Event 日志开关
```

## 使用要点

**导出前准备**：先创建专用目录（如 `/var/log/npu/report`）并 `cd` 进去，导出内容落在**当前目录**下以时间戳命名的文件夹中。同一路径不要并发导出，否则时间戳目录名冲突；单次导出多进程建议不超过 4 个。

**导出不全时改用连续导出**：任务运行时间过长导致 Device 侧日志被老化、或 Device 异常导致无法导出，都会造成日志缺失。这种情况在任务启动**之前**用 `--permanent` 常驻导出，可拿到异常前的完整日志。

**`config --set` 类命令有副作用**，执行前必须确认：

- 昇腾 AI 应用进程运行期间不建议执行，可能导致应用运行异常，建议进程退出后再执行。
- `--accelerator_recover 0`、`--aic_switch 0`、`--aiv_switch 0`、`--singlecommit 1` 均为**定位用临时配置**，会影响性能或业务可用性，定位完成后必须及时恢复；其中 `--accelerator_recover 0` 需**重启运行环境**才能恢复 AI Core 业务。
- 屏蔽核后若剩余核数小于算子所需核数，涉及核间同步的算子会执行失败。

**Device ID 语义按产品区分**：Ascend 950PR/950DT 的导出目录中 Device ID 为逻辑 ID；Atlas A3/A2/训练/推理系列及 Atlas 200I/500 A2 为物理 ID。SMP 工作模式下多个 Device 共用一个 OS，指定组内任一 Device ID 会导出该 OS 下所有 Device 日志，且同一 OS 内不支持多个导出进程并发。

**关联工具**：`stackcore/`、`ub_info/` 下的文件需用 asys 工具解析；黑匣子导出卡住或过慢时，用 `npu-smi set -t reset -i <id> -c <chip_id>` 复位芯片后重新导出。

## 信息来源

本技能内容来自 CANN 社区仓 [cann/driver](https://gitcode.com/cann/driver) 的 `docs/zh/msnpureport` 官方文档。命令输出示例因版本而异，请以环境实际输出为准。
