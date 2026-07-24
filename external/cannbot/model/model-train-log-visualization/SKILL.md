---
name: external-cannbot-model-model-train-log-visualization
description: 用于 NPU 大模型训练的日志可视化。当用户提到训练日志作图、loss/grad_norm 曲线、两份训练日志对比、误差曲线，或需要从 torchtitan
  风格训练日志按 step 提取并可视化性能指标（含 memory/tps/tflops/mfu/elapsed_time_per_step/indexer_loss）时，优先使用本技能；即使用户只说“画训练日志曲线”“对比两份训练日志”也应触发。
original-name: model-train-log-visualization
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# model-train-log-visualization 技能

用于从训练 stdout 日志中解析指标并绘制可视化曲线。

## 适用范围与约定

- 绘图脚本按 **torchtitan / torchtitan-npu 风格的 stdout 训练日志格式**解析指标（`loss`、`grad_norm`、`memory`、`tps`、`tflops`、`mfu`、`elapsed_time_per_step` 等）。
- `indexer_loss` 为部分模型（如 DeepSeek 系列）特有的可选指标，日志中不存在时自动跳过并告警。
- 若使用其他框架且日志字段格式不同，需相应调整脚本的解析正则。

## 适用场景

- 用户希望从日志文件绘制 `loss`、`grad_norm` 曲线。
- 用户希望在同一张图中对比两份日志（正常 vs 异常）。
- 用户希望追加 `memory`、`tps`、`tflops`、`mfu`、`elapsed_time_per_step`、`indexer_loss` 曲线。
- 用户希望在双日志对比中查看 `loss` 的绝对误差与相对误差曲线。

## 所需输入

- 主日志路径（必需）
- 对比日志路径（可选）
- 可选指标列表（可空）
- 输出图片路径（可选）

## 执行流程

> 脚本路径相对本技能目录。若以源码方式调用，请补全到 `model/model-train-log-visualization/scripts/plot_training_logs.py`。

### Step 1：与用户交互确认输入

按顺序询问：

1. 主日志路径（必填）
2. 是否需要第二份日志做对比（可选）
3. 是否追加可选指标（可选：`memory`(等价 `memory_gib`)、`memory_pct`、`tps`、`tflops`、`mfu`(等价 `mfu_pct`)、`elapsed_time_per_step`、`indexer`(等价 `indexer_loss`)）
4. 输出路径（可选，不填则自动命名）

### Step 2：调用绘图脚本

单日志示例：

```bash
python scripts/plot_training_logs.py \
  --log-a /path/to/train.log \
  --metrics memory,tps,indexer \
  --output /tmp/train_single.png \
  --no-show
```

> 说明：`memory` 会映射到 `memory_gib`。

双日志示例：

```bash
python scripts/plot_training_logs.py \
  --log-a /path/to/baseline.log \
  --log-b /path/to/problem.log \
  --metrics memory,tps,indexer \
  --baseline b \
  --output /tmp/train_compare.png \
  --no-show
```

> 说明：双日志没有共同 step 时，脚本会报错退出，避免生成误导性对比图。

### Step 3：返回结果

输出必须包含：

- 主图路径
- 解析到的 step 范围和关键摘要
- 对齐告警（例如双日志 step 不一致）
- 指标缺失告警（若某些可选指标不存在）

## 输出约束

- 所有曲线横轴统一为 `step`。
- 无论单日志还是双日志，`loss` 与 `grad_norm` 都必须绘制。
- 双日志模式下必须额外绘制：
  - `loss abs error`
  - `loss rel error`
  - `grad_norm abs error`
  - `grad_norm rel error`
