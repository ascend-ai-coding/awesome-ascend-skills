---
name: external-cannbot-community-model-model-recommend-ascend-env-baseline
description: 昇腾 NPU（Ascend 910B 系列，单卡）推荐模型推理环境基线检查与 GE_SESSION_DEMO 框架自检，是所有推荐模型分析与优化
  skill 的公共前置入口，含总体流水路由表与全局产物目录约束（docs/ 正式交付件、tmp/ 中间产物）。触发：环境检查、NPU 设备查询、CANN 版本确认、GE
  Session 环境检查、GE_SESSION_DEMO 编译、工作流路由、产物目录约定。
original-name: model-recommend-ascend-env-baseline
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# 昇腾 NPU 环境基线与工作流路由（推荐模型 · GE 图模式）

所有推荐模型推理分析与优化 skill 的公共前置。任一下游 skill 执行前必须先完成本 skill 的环境自检。

## Step 1: 环境自检（必做）

逐项执行并记录结果（环境参考值为 2026-10 实测，机器/版本变更后须重测）：

| 检查项 | 命令 | 环境参考值 |
|--------|------|-----------|
| NPU 设备 | `npu-smi info` | Ascend910B 系列（npu-smi Name 显示 Ascend910，PCI 0xD803，aarch64），容器内**单卡可见**，Health OK |
| HBM 容量 | `npu-smi info` | 每芯片 65536 MB |
| 设备空闲 | `npu-smi info` | AICore/Memory 占用为 0；有占用先确认进程归属 |
| CANN | `echo $ASCEND_HOME_PATH` 或 `source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh` | **9.1.0**（aarch64-linux） |
| GE v2 组件 | `ls $ASCEND_HOME_PATH/$(uname -m)-linux/lib64/libge_runner_v2.so` | 存在（`GeSession` + `RunGraphAsync` + `aclgrphParseONNX`） |
| 算子工具链 | `which atc msopgen msprof` | 均可用（标准 AscendC 开发与 profiling） |
| Python 侧 | `python3 -c "import onnx; print(onnx.__version__)"` | onnx 1.22.0；**onnxruntime 未安装**（golden 需安装或用户提供标杆） |
| 执行框架（**仅 onnx 路径**） | `cd GE_SESSION_DEMO && bash compile.sh` | 编译通过，产出 `./GEsession_test_acl`；缺失时从插件 `framework/GE_SESSION_DEMO/` 复制；pytorch/HF 路径记 `not_applicable` |

已知环境约束（2026-10 实测记录）：

- **cannbot-dsl 不可用**：当前环境不支持，所有算子开发一律走标准 AscendC（msopgen 自定义算子工程）
- **torchair / npugraph_ex 不可用**：torch_npu 2.9.0.post6 未编译 torchair；torch_npu 仅作辅助工具（golden/小工具），**不是执行路径**——图执行只有 GE Session v2 一条路
- **onnxruntime（CPU）未安装**：`gen_golden.py` 需先安装（`pip install onnxruntime`，按环境可用源）或用户直接提供 `reference/` 标杆；bf16 与自定义算子模型 onnxruntime 不可用，必须用户提供标杆
- **单卡执行**：`DEVICE_NUM` 恒为 1；卡选择经 `GE_DEVICE_ID` / `START_DEVICE`（`TOTAL_DIE` 取模）
- **并发约束**：`THREAD_NUM>1` 仅用于性能压测（不做 dump/精度验证/输出落盘）；精度验证仅串行单并发
- `compile.sh` 依赖系统头文件 `nlohmann/json.hpp`（缺失时 `dnf install -y nlohmann-json-devel`）
- **执行框架来源（仅 onnx 路径）**：`GE_SESSION_DEMO` 随插件内置在 `framework/GE_SESSION_DEMO/`（project 级 `init.sh` 自动复制到工程根；global 级与统一安装器需手工复制），目录布局与维护约定见 `framework/README.md`，框架路径与编译产物校验和信息须随环境基线归档；本工作流不引用任何外部链接。框架当前**不含** `src/graph_fusion_pass.{h,cc}`，图融合 pass 需按 GE graph pass API 自行新增
- **依赖全量核对**：外部依赖 25 项清单与缺失处置见 `workflows/references/external-dependencies.md`（在插件仓 recommend-infer-optimize 内），条目 0 逐项记录版本号与结果
- 运行必须经 `test.sh` / `run.sh` 入口（工作目录与缓存治理依赖脚本设置）；直接执行二进制会导致 GE 产物散落根目录

异常处理：

- NPU 不可用 → 停止所有下游 skill，报告环境问题
- `libge_runner_v2.so` 缺失 → CANN 安装不完整，先修复
- `compile.sh` 失败 → 检查 nlohmann-json-devel 与 CANN 路径（`CANN_HOME`）

## Step 2: 硬件与框架基线

### 2.1 硬件规格获取原则

- AIC/AIV 分离形态：GE 限核选项 `CORE_NUM="AIC|AIV"`（GE_SESSION_DEMO 默认 `24|48`）；**实际核心数与 Buffer 规格（UB/L1/L0 等）运行时经 `PlatformAscendC` 接口获取，禁止硬编码**
- **算力与带宽基准以实测锚定**（不引用纸面值）：本机小规模 matmul TFLOPS + HBM copy 带宽测试，结果记入 `data/env-baseline/`，供 [performance-analysis](../model-recommend-performance-analysis/SKILL.md) bound 判定与 [fusion-planning-estimation](../model-recommend-fusion-planning-estimation/SKILL.md) Roofline 使用

### 2.2 执行框架基线（GE_SESSION_DEMO，仅 onnx 路径）

pytorch / HuggingFace 输入跳过本步：其执行与采集基座是模型自身脚本 + `torch_npu`（`torch_npu.profiler`），不需要本框架。

- GE Session **v2**（`GEInitializeV2` / `GeSession` / `RunGraphAsync`），单设备一个 Session，单 Graph 同一 `graph_id` 重复 `AddGraph`
- onnx 统一由 GE parser（`aclgrphParseONNX`）解析；C++ 运行层配置驱动、不含模型分支
- 输入张量以 `gert::kOnHost` 放置，H2D 由 GE 内部完成；执行前 warmup 一次触发编译不计时
- 支持 dtype：`float32` / `float16` / `bfloat16` / `int32` / `int64` / `bool`（fp16/bf16 位模式无损传递）
- 模型专用环境经 `models/M<id>/M<id>_env.sh` 注入，`test.sh` 每次运行自动 source

## Step 3: 工作流路由

总体流水（依赖顺序 ≠ skill 编号）：

```
(0)环境自检 [ascend-env-baseline，本 skill]
  → (1)模型结构分析 structure-analysis → (2)单卡基线 baseline-run → (3)并发策略 concurrency-strategy-analysis
  → (4)精度门禁 precision-comparison（先于一切优化）
  ╔═══════════════════════════════════════════════════╗
  ║ (5)性能分析 performance-analysis（三表 + bound）  ║
  ║   └ 四项专项 profiling-deep-analysis              ║
  ║    （算子执行/多流掩盖/H2D-D2H/host 瓶颈四专项）  ║
  ║ (6)shape 策略 ge-graph-optimization               ║
  ║ (7)算子与图优化 fusion-operator-adaptation        ║
  ║ (8)多线程 multi-thread-optimization               ║
  ║ (9)访存 memory-access-optimize                    ║
  ║ (10)融合规划 fusion-planning-estimation           ║
  ║ (11)kernel 生成 kernel-generation                 ║
  ║ (12)自定义算子 custom-op-development              ║
  ║ (13)运行时诊断 runtime-debug（贯穿）              ║
  ║ (14)回归验收 regression-baseline-mgmt             ║
  ║     → 下一步优化动作表 → 下一轮循环               ║
  ╚═══════════════════════════════════════════════════╝
```

**循环终止条件**：达到指定性能目标（连续 2 次复测达标）/ 连续 3 轮无收益（改善 ≤ 噪声带）/ 安全阀（无动作、轮次预算默认 10）。状态：`data/{model_name}/optimization_log/loop_state.json`。

| 条目 | Skill | 进入时机 |
|------|-------|---------|
| 1 | [structure-analysis](../model-recommend-structure-analysis/SKILL.md) | 新模型接入 / 可执行性评估 |
| 2 | [baseline-run](../model-recommend-baseline-run/SKILL.md) | 模型分析后；CANN/框架版本变更后重采 |
| 3 | [concurrency-strategy-analysis](../model-recommend-concurrency-strategy-analysis/SKILL.md) | 基线就绪后；场景/batch/TP99 上限变更后重做 |
| 4 | [precision-comparison](../model-recommend-precision-comparison/SKILL.md) | **先于一切优化**：建立 golden 标杆与判定标准；此后每个优化项落地后复检 |
| 5 | [performance-analysis](../model-recommend-performance-analysis/SKILL.md) | 基线采集、优化验证、瓶颈定位 |
| 5（专项） | [profiling-deep-analysis](../model-recommend-profiling-deep-analysis/SKILL.md) | 条目 5 的四项专项；**循环每轮固定动作**（算子执行/多流掩盖/H2D-D2H/host 瓶颈） |
| 6 | [ge-graph-optimization](../model-recommend-ge-graph-optimization/SKILL.md) | 基线跑通后（shape 策略与编译选项调优） |
| 7 | [fusion-operator-adaptation](../model-recommend-fusion-operator-adaptation/SKILL.md) | 热点明确后，不新增算子实现的手段收益优先 |
| 8 | [multi-thread-optimization](../model-recommend-multi-thread-optimization/SKILL.md) | 并发策略推荐产出后 / overlap 机会表产出后 |
| 9 | [memory-access-optimize](../model-recommend-memory-access-optimize/SKILL.md) | memory-bound 热点确认后 |
| 10 | [fusion-planning-estimation](../model-recommend-fusion-planning-estimation/SKILL.md) | profiling 热点 + 未适配项清单就绪后 |
| 11 | [kernel-generation](../model-recommend-kernel-generation/SKILL.md) | 融合规划产出 Top-N 开发项后 |
| 12 | [custom-op-development](../model-recommend-custom-op-development/SKILL.md) | 图含 GE 无法解析/无原生实现的算子（功能补齐） |
| 13 | [runtime-debug](../model-recommend-runtime-debug/SKILL.md) | 任一环节报错/卡死 |
| 14 | [regression-baseline-mgmt](../model-recommend-regression-baseline-mgmt/SKILL.md) | 每个优化项收口；全流程收口时产出最终优化报告 + **下一步优化动作表**（下一轮执行入口） |
| 循环 | [optimization-loop](../model-recommend-optimization-loop/SKILL.md) | 条目 0-4 就绪后编排 5→14 迭代；终止条件满足后收口 |

已移除条目（推荐模型不适用，不得引用）：KVCache 优化（推荐模型无 KVCache）、量化策略分析（不做量化；`FORCE_16`/`ENABLE_HF32` 等精度模式开关属 [ge-graph-optimization](../model-recommend-ge-graph-optimization/SKILL.md) 范畴且须过精度门禁）。

## 全局约束：产物目录约定（所有 skill 强制）

| 目录 | 用途 | 保留策略 |
|------|------|---------|
| `docs/{model_name}/{skill}/` | **文档类正式交付件**：结构档案、图集（drawio）、方案文档、分析/验收报告、台账、经验沉淀 | 长期保留，可评审、可追溯 |
| `data/{model_name}/...` | **数据类正式交付件**：`baseline/`（baseline_metadata.json 及各阶段基线）、性能/精度对比数据等供后续环节程序化引用的数据 | 长期保留 |
| `prof/{model_name}/...` | **profiling 归档**：完整 `PROF_*` 采集目录（含 mindstudio_profiler_output 标准汇总 csv）+ 提取的 kernel 汇总 | 长期保留 |
| `ops/{op_name}/` | **生成的算子**：AscendC 工程源码、golden 参考实现、精度测试、benchmark、集成产物 | 长期保留 |
| `tmp/{model_name}/{skill}/` | **所有中间产物**：原始 trace、dump 数据、编译产物、调试日志、估算草稿、profiler 原始工作目录 | 可随时清理，不作为交付依据 |

- `{model_name}` 统一取 GE_SESSION_DEMO 工作区编号 `M<id>`
- 判定标准：**人读的文档 → docs/；机器可读/被后续环节程序化引用的数据 → data/；本工作流生成的算子工程 → ops/（按算子名组织，与模型无关）；可再生或仅用于当次调试的内容 → tmp/**
- GE_SESSION_DEMO **框架自管理目录不挪动**：`models/M<id>/{output,report,.cache,sweep}` 由框架写入；profiling 正式归档须从 `models/M<id>/.cache/profiling` 同步拷贝到 `prof/{model_name}/`
- `tmp/` 是唯一的中间产物目录，任何 skill 不得在其他目录散落中间产物
- 与模型无关的交付件可省略 `{model_name}/` 层（如 `docs/env-baseline/`）
- 本 skill 产物：环境自检记录 → `docs/env-baseline/`；环境快照与实测锚定数据 → `data/env-baseline/`（算力/带宽锚定统一落 `anchor.json`：TFLOPS、copy 带宽、测量条件）；临时探测脚本输出 → `tmp/env-baseline/`

### 全局文档格式约束：禁止裸波浪线（所有输出文档的 skill 强制）

**问题**：GFM（GitHub Flavored Markdown）会把**同段落/同表格单元内成对的单波浪线**渲染为删除线。典型踩坑：区间写法 `-15~-30%`——一个单元格里出现两处 `~` 后，`~` 之间的文字整段被划掉，且不影响语义仅污染渲染，极易漏检。

**规则**：

- Markdown 交付文档中**禁止裸单波浪线 `~`**（含表格单元格、列表、正文）
- 数值区间：用全宽 `～`（如 `-15%～-30%`）或 `-` / `至`
- 约数：用 `≈`（如 `≈64GB`）
- 需要写字面 `~` 时必须放行内代码 `` `~` `` 中（代码 span 不触发删除线）
- **交付前自检**：`grep -n "~" <doc>.md` 应无输出（行内代码内除外）
- 单行仅一个 `~` 技术上不会配对渲染，但为防后续编辑引入第二个 `~`，一律按本规则替换

各 skill 的「产物目录约束」节已内联此约束要点；本节为唯一真相源。

### 全局约束：改动校验双门禁（所有涉及模型执行改动的 skill 强制）

**任何对模型执行的改动，落地前必须同时通过「精度门禁」与「性能门禁」，缺一不可；任一不通过即回退。**

- **适用改动**：onnx 模型改动（等价改写/算子替换/自定义算子接入/domain 折叠/bool cast）、配置改动（shape 策略/batch/输出 overrides）、GE 编译选项（FORCE_16/ENABLE_HF32/precision_mode/op_precision_mode/融合开关/CORE_NUM/GRAPH_MAX_PARALLEL/THREAD_NUM）、算子实现变更（kernel/tiling/部署路径）、运行时修复中涉及上述任何一项的改动。分析类只读动作（条目 1/3/5/10 的分析本身）不触发，结论转化为改动时即触发。
- **精度门禁**：串行单并发（`THREAD_NUM=1`）`run.sh <id> --verify` 全部输出 pass（整型/bool 精确相等；浮点 `allclose` 默认 `rtol=atol=1e-3`）；精度相关选项须额外记录误差指标；并发/限核参数虽理论上不影响数值，仍须做一次串行 verify 确认。精度不过 → 直接回退，不测性能。
- **性能门禁**：同口径对比（同卡/同 batch/同 THREAD_NUM/CORE_NUM/同 CircleCount 与统计策略）；目标场景主指标（单并发 avg/TP99；吞吐场景 QPS/TP99）**不得劣化超过噪声带**（噪声带以重复测量方差确定，典型 ±2%～±5%）；改善小于噪声带不得宣称提升；允许局部指标劣化换主指标显著改善，但必须显式声明、量化、经用户确认。算子级改动必须补**整图端到端对比**（算子级提升不等于端到端提升）。
- **落地流程**：改动前确认同口径对照基准 → 实施（可开关/可回退）→ 精度门禁 →（过）→ 性能门禁 →（过）→ 落地并写 `data/{model_name}/optimization_log/perf_ledger.json`（before/after/delta/precision/`reverted=false`）；任一不过 → 回退并记录原因。
- **不可回退的改动禁止实施**（不可逆模型覆盖、框架侧修改——框架需求记录移交，不擅自改）。
- 判定细则：精度见 [precision-comparison](../model-recommend-precision-comparison/SKILL.md)，性能测量与噪声带见 [performance-analysis](../model-recommend-performance-analysis/SKILL.md)，门禁台账与回归见 [regression-baseline-mgmt](../model-recommend-regression-baseline-mgmt/SKILL.md)。

## 信息渠道

| 渠道 | 入口 | 用途 |
|------|------|------|
| 执行框架 GE_SESSION_DEMO（内置） | `<插件目录>/framework/GE_SESSION_DEMO/`（project 级自动复制到工程根，其余安装方式手工复制）；手册 `GE_SESSION_DEMO/README.md`（运行环境第 10 章、能力表第 11 章）、`doc/README_EN.md`、`framework/推荐推理GE方案部署指南.md` | 执行框架唯一真源 |
| 框架布局与维护 | `<插件目录>/framework/README.md`、`LICENSE`、`NOTICE` | 目录布局、使用方式、维护边界 |
| CANN 自带文档与头文件 | `$ASCEND_HOME_PATH` 下的 include 与文档（离线可查） | GE/ACL/AscendC API、算子清单、错误码 |
| 插件内经验沉淀 | `skills/*/SKILL.md`、`workflows/references/*.md` | 判定标准与适配要点 |
| 外部依赖清单 | `workflows/references/external-dependencies.md`（在插件仓 recommend-infer-optimize 内） | 25 项依赖、获取方式、缺失处置矩阵 |
| 环境实测 | npu-smi / atc / msprof / 试运行 | 最终仲裁依据 |

## 依据

【实测】npu-smi info（Ascend910B 系列、单卡、HBM 65536MB）；`$ASCEND_HOME_PATH`（CANN 9.1.0）；`libge_runner_v2.so` 存在性；`which atc msopgen msprof`；`pip list`（cannbot-dsl 缺失）；import 警告（torch_npu 未编译 torchair）；`bash compile.sh` 通过。
【框架】GE_SESSION_DEMO `README.md`（运行约束：单设备、并发仅压测、精度仅串行；已验证能力表 §11）。
【框架】CANN 9.1.0 自带文档与头文件（随 Toolkit 安装，离线可查）。
