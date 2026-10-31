---
name: external-cannbot-community-model-model-recommend-precision-comparison
description: 模型精度对比（工作流条目 4）。所有优化手段（shape 策略/编译选项/算子改写/融合/并发/自定义算子）落地后的精度验证门禁与问题分层定位：标杆建立（用户
  reference / onnxruntime）、verify.py 判定、dump_data 逐节点对比。触发：精度验证、精度对比、输出异常、NaN/Inf、shape
  不符、误差超限、逐节点对比、atol/rtol 判定、标杆生成。
original-name: model-recommend-precision-comparison
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# 模型精度对比

## 流水位置（强制）

**先于一切优化执行**，三重角色：
1. **前置基线门禁**：在一切优化前建立 golden 标杆（用户 `reference/` 优先 → CPU onnxruntime 生成 → 报错）、判定标准与逐节点对比框架
2. **优化验证门禁**：此后每个优化项（shape 策略 / 编译选项 / onnx 改写 / 融合 / 并发参数 / 自定义算子）落地后必须过本门禁
3. **「改动校验双门禁」的精度侧执行者**：所有对模型执行的改动，精度门禁通过是性能测量的前置条件（精度不过直接回退，不进性能测量）；精度门禁单独通过**不足以落地**——还须过性能门禁（见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 全局约束与 [performance-analysis](../model-recommend-performance-analysis/SKILL.md)）

## Prerequisites

- 基线输出（`models/M<id>/output/output_i.bin` + `output_shapes.json`，`THREAD_NUM=1` 串行采集）
- 优化后输出
- 优化阶段记录（改了什么）

## 目标

所有优化手段落地后的精度验证与问题定位，保证「性能优化不损精度」。**所有精度判定必须基于与标杆的数值对比，不可凭直觉；每次只改一个变量。**

## 输入 / 输出契约

**输入**：基线输出、优化后输出、优化阶段记录。

**输出**：精度验证结论（通过/不通过 + 指标）；不通过时输出定位报告（节点 → 参数）+ 修复验证。

## 判定标准（verify.py 口径）

- 整型与 bool：**精确相等**
- 浮点：`allclose`，默认 `rtol=atol=1e-3`（`verify.py --rtol/--atol` 可配）
- **逐输出阈值优先**：若存在 `models/M<id>/config/precision_gate.json`（或 `--gate <file>`），按输出名命中 `outputs.<name>` 的 `rtol/atol`，未命中回退 `default`，再回退 `--rtol/--atol`。业务上允许更宽阈值的输出（如检测框坐标 atol=0.5）必须在 gate 文件里写明 `note`（依据 + 谁确认的），否则视为降低门禁标准。
  ```json
  {
    "default": {"rtol": 1e-3, "atol": 1e-3},
    "outputs": {
      "det_box": {"rtol": 1e-3, "atol": 0.5, "note": "坐标允许半像素；用户 2026-10-xx 确认"}
    }
  }
  ```
  `verify_report.json` 会同时记录 `precision_gate` 路径与每个输出的 `rtol/atol`，门禁结论按输出逐条引用；缺该文件时 `verify.py` 会打印 WARN，说明所有输出按全局阈值判定。
- 报告：`output/verify_report.json`（简述）+ `output/verify_table.log`（误差区间统计 + 绝对/相对误差前 30 明细，忽略 `abs_err<=1e-6` 可忽略项）
- 自定义算子按 dtype 的 atol/rtol 标准验收（见 [kernel-generation](../model-recommend-kernel-generation/SKILL.md)）

## 工作流步骤（分层排查）

### Step 1: 收集信息

精度症状、优化阶段、精度模式（`FORCE_16` / `ENABLE_HF32` / `ge.exec.precision_mode` / `op_precision_mode`）、标杆来源。

### Step 2: 快速诊断

症状分类表匹配 → 快速验证清单 → 发现问题直接修复。

精度症状速查表（推荐模型版）：

| 症状 | 可能原因 |
|------|---------|
| 输出全 NaN/Inf | 数值溢出；`FORCE_16` / `allow_hf32` 引入的下溢；大 embedding 表 fp16 越界 |
| 输出 shape 不符 | 输出动态维解析错误（`output_overrides` 缺失）；声明 shape 与 GE 返回不一致（看 `[GE OUTPUT][WARN]`，以 `output_shapes.json` 为准） |
| 整型输出错误 | 索引/查表逻辑被等价改写破坏（bool cast / domain fold / 算子替换假设不成立） |
| 浮点误差超限 | 精度模式（fp16/hf32 路径）、GE 融合算子数值差、`op_precision_mode` 配置 |
| 部分输入路输出恒定/缺失 | onnx 改写引入语义偏差（如 `Min(bool,bool)` 改写假设不成立、ScatterND 索引语义差异） |
| 自定义算子输出错 | AscendC kernel 实现错误或 GE 接入（infer shape/dtype/tiling）错误 |

### Step 3: 分节点定位（dump_data）

`bash run.sh <id> --dump_data`：开启 GE 数据 dump（`models/M<id>/.cache/dump_data`，每类保留最新 3 份）→ 与 CPU 参考逐节点对比 → 定位首个偏差节点。

### Step 4: 精细定位（按需）

- 改写 onnx 暴露中间输出（将目标节点输出加入 `graph.output`，重新 prepare/run）逐步缩小
- onnxruntime 可用时可用其逐节点输出做参考；不可用时用 numpy 按 onnx 语义重算参考

### Step 5: 修复与验证

按根因修复（改写假设修正 / 精度模式回退 / 算子参数修正）→ 重新全量精度验证。

### Step 6: 输出调试报告

问题概要 / 根因分析 / 修复措施 / 修复验证 / 遗留风险。

## 昇腾 NPU 专属要点

- **标杆来源优先级**：用户 `reference/output_i.npy`（次 `.bin`）→ onnxruntime 生成（**本环境未安装**，需 `pip install`，华为云镜像）→ 报错。**bf16 与自定义算子模型 onnxruntime 不可用，必须用户提供标杆**。
- **精度验证仅串行单并发**（`THREAD_NUM=1`，`--verify` 在并发下会报错）；并发参数变更不影响数值（同图同输入），但 shape/batch 变更会改变输出 shape 与数值口径，须重新对齐标杆。
- **精度模式开关一次只动一个**：`FORCE_16` / `ENABLE_HF32` / `precision_mode` / `op_precision_mode` 的收益与精度同时记录（联动 [ge-graph-optimization](../model-recommend-ge-graph-optimization/SKILL.md)）。
- 输出动态维以运行时 `output_shapes.json` 为准；`verify.py` 据此解释 `output_i.bin`，静态声明不一致不中断但要看 `[GE OUTPUT][WARN]`。
- 调试中间 tensor 与日志必须保留现场（tmp/），用于后续分析。

## 产物目录约束

- **文档格式约束（GFM 删除线）**：Markdown 文档禁止裸单波浪线 `~`——同段落/同表格单元内成对出现会被 GFM 渲染为删除线；数值区间用全宽 `～`（`-15%～-30%`）或 `-`/`至`，约数用 `≈`；交付前 grep 确认文档无裸 `~`（详见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 全局约束）
- **文档类交付件** → `docs/{model_name}/precision-comparison/`：精度验证结论、定位报告、调试报告
- **数据类交付件** → `data/{model_name}/precision-comparison/`：精度对比结构化数据（逐输出 pass/fail、误差指标，供回归环节程序化引用）
- **中间产物** → `tmp/{model_name}/precision-comparison/`：dump 的中间 tensor、逐节点对比原始数据、改写试验 onnx
- 全局约定与判定标准见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md)；违反目录约定的产物视为未交付

## 验收标准

- 端到端逐输出与标杆一致（verify 全 pass + 关键指标）
- 逐节点对比时首个偏差节点可复现
- 修复后全量回归通过
- 标杆来源与生成口径记录在案（可复现）
- **每笔模型执行改动均有精度门禁结论并已写入 perf_ledger（`precision.verify_pass`）**；未过精度的改动已回退且未进入性能测量

## 依据

【框架】`gen_golden.py`（标杆生成与来源优先级）、`verify.py`（判定口径与报告）、`--dump_data`（GE 数据 dump）、`--verify` 串行约束。
【实测】本环境 onnxruntime 未安装（安装或用户标杆二选一）。
【技能库经验】分层排查、症状速查表、对比驱动原则（不可凭直觉、每次只改一个变量）。
