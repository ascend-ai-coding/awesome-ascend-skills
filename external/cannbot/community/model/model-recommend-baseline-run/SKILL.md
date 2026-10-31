---
name: external-cannbot-community-model-model-recommend-baseline-run
description: 单卡基线运行（工作流条目 2）。onnx 路径经 prepare.sh 准备、compile.sh 编译、run.sh 运行（pytorch/HF
  路径用其自身脚本 + torch_npu），采集首基线（单并发时延 avg/TP99/TP999、QPS、HBM 峰值、profiling、精度验证），作为功能验证与一切优化的
  first baseline。触发：基线运行、首基线、模型跑通、基线采集、新 CANN/框架版本首跑、性能基线、精度基线。
original-name: model-recommend-baseline-run
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# 单卡基线运行（GE 图模式首基线）

## Prerequisites

- 已通过 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 环境自检（onnx 路径：NPU 可用 + GE_SESSION_DEMO 编译通过；pytorch/HF 路径：NPU 可用 + 脚本可运行，不涉及 GE_SESSION_DEMO）
- [structure-analysis](../model-recommend-structure-analysis/SKILL.md) 已产出结构档案（确认单卡可执行、无致命兼容性问题）
- **所有不明确项已与用户确认**（见 Step 1 决策清单，禁止默认拍板）

## 目标

模型在单卡以 GE 图模式（`GeSession` + `RunGraphAsync`）跑通，采集**串行单并发**基线：时延（avg/TP99/TP999）、QPS、HBM 峰值、profiling、精度验证，作为功能验证与后续优化的 first baseline。推荐模型单卡可执行，**无需裁剪**（原 LLM 工作流的裁剪层基线不适用）。

**基线的双重身份**：本基线是此后所有「改动校验双门禁」的同口径对照基准（见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 全局约束）——每笔模型执行改动落地前的精度门禁（串行 verify）与性能门禁（同口径前后对比）均以本基线或上一已验收配置为参照，故基线的口径记录（卡/batch/THREAD_NUM/CORE_NUM/CircleCount/统计策略/标杆来源）必须完整，保证任意改动可对照复测。

## 输入 / 输出契约

**输入**：`models/M<id>/` 工作区（onnx 必需）；目标卡（空闲确认后）；用户确认的决策清单。

> **pytorch / HuggingFace 路径为实验性**：本 skill 目前只有 onnx / GE 侧实现，pytorch 侧需用户提供等价脚本（结构导出 / 基线脚本 / 并发压测 / `torch_npu.profiler` 采集）。

**输出**：

- `data/{model_name}/baseline/baseline_metadata.json`（基线数据：环境快照、决策记录、计时、HBM 峰值、精度结论、prof_archive 指针、**输入/标杆指纹 `fingerprints`**）
- `docs/{model_name}/baseline-run/baseline_report.md`（基线报告）
- `prof/{model_name}/baseline/`（profiling 归档）
- `tmp/{model_name}/baseline-run/`（脚本、HBM 采样数据、调试日志）

## 工作流步骤

### Step 1: 决策清单（必须与用户逐项确认，禁止默认）

| 决策项 | 选项要点 |
|--------|---------|
| 输入来源 | 真实输入（`input_data/*.npy`）/ `--gen-input` 随机构造（`input_data/` 须为空，固定种子可复现；整型值范围有限，索引类输入需注意） |
| shape 策略 | `all_static`（默认推荐）/ `all_dynamic` / `custom`（详见 [ge-graph-optimization](../model-recommend-ge-graph-optimization/SKILL.md)） |
| batch | `shape_config.batch_size`（`all_static` 下数据沿第 0 维自动扩展/裁剪） |
| 标杆来源 | 用户 `reference/output_i.npy`（bf16/自定义算子模型**必须**）/ onnxruntime 生成（本环境未安装，需先 `pip install`） |
| 采集内容 | 时延 + HBM 峰值 + profiling + 精度（推荐全采）/ 仅时延 |
| 目标卡 | `GE_DEVICE_ID` / `START_DEVICE`，经 Step 2 空闲确认后向用户报备 |

### Step 2: 环境与设备确认

1. **设备空闲确认**：`npu-smi info` 查进程与 HBM 占用；被占用则换卡或等待（`sweep.py` 内置占用检测逻辑可复用）。
2. **实测锚定（首次执行时）**：小规模 matmul + HBM copy 带宽，建立本机算力/带宽基准，记入 `data/env-baseline/`（供 [performance-analysis](../model-recommend-performance-analysis/SKILL.md) bound 判定复用）。
3. **框架可用性**：`bash compile.sh` 编译通过（首次或源码变更后）。

### Step 3: 模型准备与校验

1. `bash prepare.sh <id> [--gen-input] [--input-shape name=d0,d1,...]`：模型分析 → 配置生成 → 输入规范化；外部权重同目录校验（框架自带，缺失即报错）。
   - 输入存在**除第 0 维之外的动态维**时，`--gen-input` 必须同时给 `--input-shape`（如 `--input-shape serving_default_x=1,3,640,640`），否则 `prepare_input.py` 直接报错而不是静默生成 `[1,3,1,1]` 这类退化输入；也可用 `GEN_INPUT_SHAPE="x=1,3,640,640;y=1,512"` 传入。
2. `python3 tools/inspect_config.py <id>`：配置交叉校验，error 必须清零；告警逐条确认（存在告警时退出码 2）。
3. 输出含动态维时：补 `shape_config.output_overrides` 或确认保留 `-1` 运行时解析（`--adopt-output-shapes` 回填探测值）。

### Step 4: 冒烟测试（先行必做）

`bash run.sh <id> --circle 10`：验证建图、编译、执行、输出落盘全链路无 dtype/shape 错误；失败转 [runtime-debug](../model-recommend-runtime-debug/SKILL.md)。

**可行性探测（batch 与动态 shape，必做）**：在正式基线之前，按条目 1 的 `feasibility.json` 逐项探测一次，把「能不能用」提前到第一轮之前，而不是等到循环里才发现：

```bash
bash run.sh <id> --batch 2 --dry-run      # 先看会改什么（不落盘）
bash run.sh <id> --batch 2 --circle 10    # 10 次 warmup+执行，失败即判定不支持
bash run.sh <id> --restore                # 无论成败都恢复 shape_config/io_config/input_data
```

- `--batch` 会改写 `shape_config` / `io_config` 并重生成 `input_data`，脚本会先备份到 `models/M<id>/.cache/pre_batch_*`，用 `--restore` 回退；
- 判定为不支持时，在 `feasibility.json` 写 `batch_support: no` / `dynamic_support: no` 与失败证据（错误码/日志行），并**禁止**后续条目 3 的 batch 候选与 sweep 的 batch 阶段；
- 关闭全部 pass、换 fp32 复测仍失败，才可判定为模型/执行路径限制（而非本项目的优化改动引起）。

### Step 5: 正式基线（串行单并发）

- `bash run.sh <id>`（`THREAD_NUM=1`）：框架自动 warmup 一次触发编译（不计时），随后压测循环
- `[GE PROGRESS]` 行提取：`avg session run time` / `TP99` / `TP999` / `Qps`；`CircleCount` 足够大（如 1000+）保证打点稳定，`--debug-freq` 控制打印频率
- HBM 峰值：`npu-smi info` 后台线程采样（如 50ms 间隔）；GE 模式下无 torch 侧内存 API 可依赖
- 记录编译产物状态（`kernel_meta` 复用与否、冷启动编译耗时）

### Step 6: 精度验证（仅串行单并发）

- `bash run.sh <id> --verify`：推理后调用 `tools/verify.py`
- 标杆解析顺序：用户 `reference/output_i.npy`（次 `.bin`）→ onnxruntime 生成 → 报错
- 判定：整型/bool 精确相等；浮点 `allclose` 默认 `rtol=atol=1e-3`；报告 `output/verify_report.json` + `verify_table.log`
- 不通过转 [precision-comparison](../model-recommend-precision-comparison/SKILL.md)

### Step 7: profiling 采集与归档

- `bash run.sh <id> --prof ge`（或 `msprof`）：产物落 `models/M<id>/.cache/profiling/PROF_*`
- **归档到 `prof/{model_name}/baseline/`**：完整采集目录（含 `mindstudio_profiler_output` 标准汇总）+ 提取的 kernel 汇总（kernel_name/count/total_us/avg_us/total_pct 降序）
- 采集前确认卡空闲；如遇 op_summary/trace 解析失败，检查 `libsqlite3.so` 软链（历史环境已知问题）

### Step 8: 产物落盘与报告

基线 JSON（含环境快照、决策记录、计时、HBM 峰值、精度结论、prof 指针）+ 报告（环境/决策/冒烟/基线结果/精度/已知问题/流转结论）。首次写入 `data/{model_name}/optimization_log/perf_ledger.json`（初始基线记录，格式见主工作流文档全局强制输出要求）。

## 昇腾 NPU 专属要点

- **运行约束（框架强制）**：`DEVICE_NUM=1`；并发（`THREAD_NUM>1`）仅压测（不 dump/不验证/不落盘输出）；精度验证仅串行单并发——基线一律串行采集，并发性能归 [multi-thread-optimization](../model-recommend-multi-thread-optimization/SKILL.md)。
- 单次请求时延即图执行一次的时延（含 GE 内部 H2D，输入以 `kOnHost` 放置）；`MaxQPS` 可限速。
- onnxruntime 缺失时 golden 不可自动生成：bf16 与自定义算子模型**必须**用户提供标杆；安装用华为云镜像。
- 换 shape 策略/配置后首次运行含编译耗时，计时以 warmup 之后为准（框架已内置）。
- 运行必须经 `run.sh`/`test.sh`（工作目录 `models/M<id>/.cache/run`、缓存治理依赖脚本）；直接执行二进制会散落 GE 产物。

## 产物目录约束

- **文档格式约束（GFM 删除线）**：Markdown 文档禁止裸单波浪线 `~`——同段落/同表格单元内成对出现会被 GFM 渲染为删除线；数值区间用全宽 `～`（`-15%～-30%`）或 `-`/`至`，约数用 `≈`；交付前 grep 确认文档无裸 `~`（详见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 全局约束）
- **数据类交付件** → `data/{model_name}/baseline/`：`baseline_metadata.json`（长期保留，供回归与优化环节程序化引用）

**基线指纹（同口径的物质基础）**：`baseline_metadata.json` 必须含 `fingerprints` 段。

```bash
# 记录 + 并入基线（--merge 会同时写 config/provenance.json，之后可直接无参比对）
python3 tools/check_provenance.py <M<id>> --merge data/{model_name}/baseline/baseline_metadata.json

# 每次「优化前后」结论产出前比对（无参读 config/provenance.json；也可 --baseline 指定基线文件）
python3 tools/check_provenance.py <M<id>>
```

指纹分两域，**不要混为一谈**：

| 域 | 字段 | 变化时 |
| --- | --- | --- |
| frozen（非实验变量） | `input_sha256`、`golden_sha256` | 变了就是换了题：`exit 1`，只能重建基线，不允许声明 |
| declared（预期实验变量） | `model_sha256`、`shape_config_sha256`、`io_config_sha256` | 换模型 / 改 shape 策略 / 改 io 配置**本就是优化动作**：先用 `--declare <字段>` 声明，工具输出 `declared_change: ... baseline -> now` 记录，黏进本轮报告即可 |

要点：
- 未建立基线指纹时工具以 `exit 2` 退出（不再"WARN 后静默通过"）；
- `--declare` 只接受 declared 三字段，声明 frozen 字段会直接报参数错误；
- `--merge` 与无参比对是同一份记录（merge 会同步写 `config/provenance.json`），不存在"写了基线但复查读不到"的情况。
- **profiling 归档** → `prof/{model_name}/baseline/`：完整 PROF_* 目录 + kernel 汇总
- **文档类交付件** → `docs/{model_name}/baseline-run/`：基线报告
- **中间产物** → `tmp/{model_name}/baseline-run/`：HBM 采样数据、冒烟/调试日志、归档脚本
- 全局约定与判定标准见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md)；违反目录约定的产物视为未交付

## 验收标准

- 冒烟先行通过；正式基线无 dtype/shape 错误
- 串行基线数据落盘（含 warmup 口径说明、CircleCount/统计策略记录）
- 精度 verify 有结论（通过/不通过 + 指标；不通过有定位流转）
- profiling 已归档（kernel 汇总产出，kernel 事件数 > 0）
- HBM 峰值有采样数据
- 报告含流转结论（→ [concurrency-strategy-analysis](../model-recommend-concurrency-strategy-analysis/SKILL.md) / [precision-comparison](../model-recommend-precision-comparison/SKILL.md) / [performance-analysis](../model-recommend-performance-analysis/SKILL.md)）
- perf_ledger.json 已写入初始基线记录

## 依据

【框架】`prepare.sh`（一键准备 + 权重校验）、`compile.sh`、`run.sh`（warmup 内置、`--verify`、`--prof`）、`verify.py`（标杆与判定口径）、运行约束（单设备/并发仅压测/精度仅串行）。
【实测】npu-smi info（设备/HBM）；本机 onnxruntime 缺失、CANN 9.1.0。
【技能库经验】基线采集范式（冒烟先行、同口径计时、profiling 归档）。
