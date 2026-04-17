---
name: mindspeed-llm-train-profiler
description: 指导并自动化完成昇腾 NPU 上 MindSpeed-LLM 训练的 Profiling 数据采集。支持配置并运行带 Profiling 的模型训练，包括 CPU 采集、内存采集、不同采集级别（level0/level1/level2）和自定义 step 范围。生成的 Profiling 数据可用 MindStudio Insight 进行性能分析。当用户需要在模型训练中采集 Profiling 数据、进行训练性能分析、或执行 性能数据采集/Profiling采集 时触发。触发关键词：profiling、性能分析、性能数据采集、Profiling采集、训练框架profiling、MindSpeed-LLM profiling。
---

# MindSpeed-LLM 训练 Profiling 数据采集（昇腾 NPU）

- 使用用户的语言回复。
- 以下内容不要翻译：命令、文件路径、环境变量、包名、错误信息。

## 第 0 步：确认前置条件

回复用户：

> 开始 Profiling 采集前，请提供以下信息：
>
> **必要：**
> - 训练脚本路径（`.sh`）— 已配置好模型、数据、权重的可运行脚本
>
> **可选（有默认值）：**
> - 采集级别：level0 / **level1**（推荐）/ level2
> - 采集步骤：默认 step 2 ~ 4（不含 step 4，实际采集 step 2 和 3）
> - 采集内容：**CPU**（默认开启）、内存、堆栈、tensor 形状
> - NPU 卡数：默认单卡
>
> 没有训练脚本则无法采集。

**用户未提供任何信息时：** 停止。回复："Profiling 需要可运行的训练脚本，请先准备好再进行采集。"

**用户提供了训练脚本后：** 展示最终配置表，等待用户确认后再执行。

> | 配置项 | 值 |
> |--------|-----|
> | 模型 | （从脚本中读取） |
> | 训练脚本 | （路径） |
> | NPU 卡数 | （从脚本中读取 NPUS_PER_NODE） |
> | 采集级别 | level1（推荐） |
> | 采集步骤 | step 2 ~ 4（不含 4） |
> | 采集卡号 | rank 0 |
> | CPU 采集 | 开启 |
> | 内存采集 | 关闭 |
> | 堆栈采集 | 关闭 |
> | Tensor 形状采集 | 关闭 |
>
> 确认后开始。如需修改请说明。

**禁止下载模型、转换权重或创建训练脚本。**

### 用户输入与变量映射

| 用户说 | 变量 | 默认值 |
|--------|------|--------|
| 训练脚本（.sh） | `TRAINING_SCRIPT` | *（必填）* |
| "CPU" / "采集CPU" | `PROFILE_WITH_CPU=true` | `true` |
| "内存" / "memory" | `PROFILE_WITH_MEMORY=true` | `false` |
| "Level0/1/2" | `PROFILE_LEVEL` | `level1` |
| "step N 到 M" | `PROFILE_STEP_START=N PROFILE_STEP_END=M` | `2, 4` |
| "堆栈" / "stack" | `PROFILE_WITH_STACK=true` | `false` |
| "shape" / "维度" | `PROFILE_RECORD_SHAPES=true` | `false` |
| "所有卡" / "all ranks" | `PROFILE_RANKS=-1` | `0`（仅 rank 0） |

说明：NPU 卡数从训练脚本中读取（`NPUS_PER_NODE` 或 `--nproc_per_node`）。Profiling 默认只采集 rank 0。仅在用户明确要求时才设置 `PROFILE_RANKS=-1` 采集所有卡。

## 第 1 步：检查环境

运行 `npu-smi info` 确认 NPU 可用。如不可用，**停止**："当前环境未检测到 NPU，请在有 Ascend NPU 的环境中运行。"

如需加载 CANN 环境：`source /usr/local/Ascend/ascend-toolkit/set_env.sh`

## 第 2 步：验证脚本并执行 Profiling

**禁止修改用户的原始训练脚本。** 基于原始脚本创建新脚本。

1. 读取原始脚本，检查其引用的关键路径是否存在（权重目录、数据文件、tokenizer 目录）。
2. 如有路径不存在，**停止并报告：**
   > 训练脚本中引用的路径不存在：
   > - `<缺失路径>` — 未找到
   >
   > 请检查脚本中的路径配置，确保训练环境就绪后再进行 Profiling 采集。
3. 创建**新的** Profiling 脚本（使用带时间戳的文件名，如 `run_profiling_$(date +%Y%m%d%H%M%S).sh`，避免覆盖已有文件）。复制原始脚本内容，在 `torchrun` 命令中添加以下 Profiling 参数（在 `| tee` 之前；如无 `| tee` 则追加到 `torchrun` 命令末尾；如无法识别命令格式则展示参数并请用户指定插入位置）：

```
--profile --profile-step-start 2 --profile-step-end 4 \
--profile-ranks 0 --profile-level level1 --profile-with-cpu \
--profile-save-path ./profiling_output
```

4. 如果原始脚本已包含 `--profile` 参数，标注已有配置并询问用户是保留还是覆盖。
5. 根据用户的 Profiling 配置调整参数值。运行前创建输出目录。
6. 运行**新脚本**。原始脚本必须保持不变。
7. **如训练失败（非零退出码），向用户报告错误：**
   > 训练脚本执行失败（退出码: N）。Profiling 采集需要训练正常运行。
   >
   > 请修复训练脚本后重试。

## 第 3 步：验证并报告结果

训练完成后，验证 Profiling 输出：

1. 检查 Profiling 输出目录是否存在，是否包含 `*_ascend_pt/` 子目录。
2. 至少应包含 `PROF_*/device_*/data/`（NPU 数据）。
3. 向用户报告：输出位置、使用的配置、目录结构。
4. 建议使用 MindStudio Insight 进行可视化分析。

如输出缺失或为空，检查：`PROFILE_STEP_START` >= 1、训练是否执行到了采集步骤、CANN 环境是否已加载。

## 故障排查

| 症状 | 解决方法 |
|------|----------|
| `NPU out of memory` | 减小训练脚本中的 batch size |
| Profiling 目录为空 | `PROFILE_STEP_START` 必须 >= 1；训练必须执行到该步骤 |
| `Address already in use` | 修改训练脚本中的 `MASTER_PORT` |
| 权重形状不匹配 | 权重的 TP/PP 与训练配置不一致，通知用户后停止采集 |

## 禁止事项

- 禁止修改用户的原始训练脚本，必须创建新脚本。
- 禁止安装框架、下载模型或转换权重。
- 禁止设置 `--profile-step-start 0` — 必须 >= 1。
- 禁止在大规模训练中采集所有步骤 — 2-3 步即可。
- 没有训练脚本禁止继续执行。
- 除非用户明确要求，禁止设置 `--profile-ranks -1`。

## 参考文档

详见 [reference/mindspeed-profiling-args.md](reference/mindspeed-profiling-args.md) 获取完整参数表（Mcore CLI 参数 + FSDP2 YAML 配置）。
