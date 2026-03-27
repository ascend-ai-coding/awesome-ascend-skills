---
name: external-mindstudio-mindstudio_profiler_data_check
description: 当用户提供 MindStudio profiler 采集的性能数据（框架 profiler、msprof 命令行）时，对数据完整性、采集状态及关键配置进行校验，确保后续分析工具能正常运行。
keywords:
- profiler
- 性能分析
- 数据检查
original-name: mindstudio_profiler_data_check
synced-from: https://github.com/kali20gakki/mindstudio-skills
synced-date: '2026-03-26'
synced-commit: 266c7821de7b51b683d4605960d0d86f7d631e03
license: UNKNOWN
---


# MindStudio Profiler 数据校验

## 技能目标

在深入性能分析前，对 profiler 数据进行「体检」，确保：
- 正确识别数据类型（框架 profiler / msprof）
- 采集过程正常结束（流程正常 Stop）
- 数据已解析，关键交付件存在
- 识别 profiler 配置，预判后续分析可行性

## 数据类型特征

| 类型 | 标识特征 | 采集方式 |
|------|----------|----------|
| **框架 profiler** | `[*]_ascend_pt`（PyTorch）或 `[*]_ascend_ms`（MindSpore） | 代码内嵌，进程级 |
| **msprof** | 含 `PROF_{}` 目录 | 命令行，进程级 |

框架 profiler 含 torch_npu/MindSpore 框架 API、CANN API、device 算子、硬件指标；msprof 含 CANN API、device 算子、硬件指标。

## 依赖与资源

- **MCP 工具**：`msprof-mcp__get_profiler_config(profiler_path)` — 解析 profiler_info.json（仅框架 profiler）
- **离线解析**：
  - **框架 profiler**：`scripts/offline_parse_pytorch.py <profiler_path>`（PyTorch），`scripts/offline_parse_mindspore.py <profiler_path>`（MindSpore）
  - **msprof**：`msprof --export=on --output=<output_path>`（output 指向 `PROF_{}` 所在路径或导出目录）

## 分析步骤

- **多卡原则**：多卡数据默认只抽查一张卡（如 Rank0）；仅当用户明确要求「每张卡都检查」「逐卡校验」时，再逐卡检查。
- **类型绑定原则（必须严格遵守）**：数据类型的判定依据是**用户给定的顶层路径**。判定后，**全程只执行该类型对应的步骤**，不得混用。
  - 若顶层路径以 `[*]_ascend_pt` 或 `[*]_ascend_ms` 结尾 → **仅用框架 profiler 规则**。
  - 若顶层路径为 `PROF_[*]` 目录 → **仅用 msprof 规则**。

### 1. 识别数据类型

- `[*]_ascend_pt` → 框架 profiler（PyTorch）；多子目录→多卡，单目录→单卡
- `[*]_ascend_ms` → 框架 profiler（MindSpore）；多子目录→多卡，单目录→单卡
- `PROF_[*]` → msprof 命令行；多子目录 → 多卡，单目录 → 单卡

### 2. 校验采集状态 (Stop Check)

- **框架 profiler**：检查 `profiler_info.json` 或 `profiler_info_{Rank_ID}.json`；缺失则提示「采集未正常 Stop」，终止，建议检查 `profiler.stop()`
- **msprof**：检查 `PROF_{}/device_{}/` 下是否存在 `end_info.{}`（`{}` 为 device 编号占位）；缺失则提示「采集未正常结束」，终止

### 3. 解析配置（仅框架 profiler）

- **框架 profiler**：调用 `get_profiler_config` 读取 profiler_info.json，关注 `profiler_level`、`with_stack`、`with_modules`、`record_shapes`、`profile_memory`。
- **msprof**：无需检查配置，跳过本步骤。

### 4. 校验解析状态 (Parse Check)

- **框架 profiler**：检查是否存在 `ASCEND_PROFILER_OUTPUT` 或对应解析输出目录；若缺失则停止分析，询问用户是否协助解析，同意则执行 `offline_parse_pytorch.py`（PyTorch）或 `offline_parse_mindspore.py`（MindSpore）。
- **msprof**：检查 `PROF_{}` 下是否存在 `mindstudio_profiler_output` 目录（即 `msprof --export=on` 的导出结果）；若缺失则停止分析，询问是否协助执行 `msprof --export=on --output=<path>`。
- 解析完成（或已存在解析结果）后，继续第 5 步。

### 5. 检查关键交付件

- **框架 profiler**：按 `export_type` 检查——Text 模式需 `trace_view.json`、`kernel_details.csv`；DB 模式需 `[*]_profiler_[*].db`。缺失则警告影响 Timeline/算子分析。
- **msprof**：检查 `PROF_{}` 下是否含 `msprof_[*].db`，或 `mindstudio_profiler_output` 下是否含 `msprof_{timestamp}.json`、`op_summary.csv`。都缺失则警告。

## 输出策略

本技能为**前置校验**，**不必每次都输出完整报告**：

- **用户明确要求**「检查数据」「看校验结果」时，输出**简洁结构化报告**，按以下格式（每项一行，无则略过）：
  ```
  【类型】框架 profiler (PyTorch/MindSpore) 或 msprof 命令行采集 | 单卡/多卡
  【状态】valid / invalid / unparsed
  【配置】level、with_stack、profile_memory、采集步数等关键项（仅框架 profiler）
  【缺失】trace_view.json、kernel_details.csv 等（如有）
  【建议】需解析 / 检查 profiler.stop / 无
  【下一步】推荐的分析技能或操作
  ```
- **前置调用**（用户在做 Timeline / 算子分析等）：
  - `invalid`（未 Stop）→ 必须告知并终止
  - `unparsed`（未解析）→ 告知需先解析并询问是否协助
  - 校验通过 → 静默进入后续分析

## 约束

- `invalid`（未正常 Stop）必须终止流程
- `unparsed`（未解析）必须优先引导解析，否则后续工具无法运行
