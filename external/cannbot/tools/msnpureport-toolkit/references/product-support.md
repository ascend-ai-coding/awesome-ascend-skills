# 产品支持情况

## 总体约束

- msnpureport 仅适用于 **Ascend EP 形态**。
- **昇腾虚拟化实例场景不支持**使用该工具。

## 功能支持矩阵

| 功能 | Ascend 950PR/950DT | Atlas A3 训练/推理系列 | Atlas A2 训练/推理系列 | Atlas 训练系列 | Atlas 推理系列 | Atlas 200I/500 A2 推理 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 单次导出日志和文件 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 连续导出日志和文件 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 查询 Device 维测配置 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 设置 Device 日志级别 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 设置 TaskSchedule 自动复位加速器 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 设置 AI Core singlecommit | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 屏蔽指定 AI Core 任务执行 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 屏蔽指定 Vector Core 任务执行 | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ |
| 设置 icache bit 翻转校验范围 | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| 配置工具维测日志展示位置 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 配置工具维测日志级别 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 查询版本/帮助信息 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## 导出类型（`-t`）支持差异

| 类型 | Ascend 950PR/950DT | Atlas A3 训练/推理系列 | Atlas A2 训练/推理系列 | Atlas 推理系列 | Atlas 200I/500 A2 推理 |
|------|:---:|:---:|:---:|:---:|:---:|
| `-t 4` vmcore 文件 | ✓ | ✓ | ✓ | 不生成 | 不生成 |
| `-t 6` Unified Bus 维测信息 | ✓ | ✗ | ✗ | ✗ | ✗ |
| `-t 7` AO 区日志信息 | ✓ | ✗ | ✗ | ✗ | ✗ |
| `-t 8` AO 计数 | ✓ | ✗ | ✗ | ✗ | ✗ |
| `-t 9` 串口录音 | ✓ | ✗ | ✗ | ✗ | ✗ |

> `-t 4` 的 vmcore 文件在 Device OS 心跳丢失时与黑匣子日志同时生成；Atlas 训练系列产品的 vmcore 支持情况未在源文档中列出，请以实际环境为准。

## 架构相关差异

**AI Core 分离架构**：将 AI Core 拆成矩阵计算（AI Cube，AIC）和向量计算（AI Vector，AIV）两个独立的核。分离架构产品中，`--aic_switch` 的 AI Core 特指 **AIC 核**。典型分离架构产品：

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

**`-t 2` 寄存器信息**：Atlas A2 训练/推理系列产品、Atlas A3 训练/推理系列产品在 `-t 2` 下还支持导出当前硬件寄存器信息，导出后建议复位芯片。

**Device ID 语义**：Ascend 950PR/950DT 导出目录中的 Device ID 为**逻辑 ID**；其余产品（Atlas A3/A2 训练推理系列、Atlas 训练系列、Atlas 推理系列、Atlas 200I/500 A2 推理产品）为**物理 ID**。注意命令行 `-d` 参数在所有产品上均为逻辑 ID。

> 本矩阵依据 CANN 社区仓 [cann/driver](https://gitcode.com/cann/driver) `docs/zh/msnpureport` 各章节「产品支持情况」小节整理。产品支持情况随版本演进，实际执行前建议用 `msnpureport <subcommand> --help` 确认当前版本支持的参数。
