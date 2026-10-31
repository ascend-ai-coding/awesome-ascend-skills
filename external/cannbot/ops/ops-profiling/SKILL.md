---
name: external-cannbot-ops-ops-profiling
description: NPU 性能采集与分析，支持 msprof application 标准可比较计时（每例 10 组 warmup→target 取中位数）、msprof
  深度指标采集与算子级瓶颈定位、kernel-level 对比测试（--compare/--quick/--batch）和交互式可视化。核心工具：collect_perf_application.py（标准可比较计时）、msprof_profile_run.sh（标准采集
  / --compare 对比测试 / --quick 快速对比 / --batch 批量并行）、msprof_perf_summary.py（瓶颈分析、对比报告）与
  compute_efficiency.py（MBU/MFU/瓶颈类型/理论上限）。加速比对比用 msprof_profile_run.sh 的 compare/quick
  模式，输出 performance.json 和 markdown 对比报告即完成，无需可视化；quick/compare 均支持 --keep-prof 保留默认会清理的原始
  PROF 目录。另含 msopprof-visualization 子技能，用于用户明确要求的交互式 HTML 性能报告：支持含 collection_manifest.json
  的采集目录以 visualize 模式重渲染，也支持算子工程经 run_pipeline.py 采集并生成报告；不直接接收 msprof_profile_run.sh
  的 PROF_GROUP_* 归档，不用于加速比对比场景。当用户在算子开发过程中提到"上板性能"、"算子性能测试"、"硬件性能验证"、"NPU性能采集"、"NPU
  profiling"、"性能对比"、"加速比"、"性能可视化"、"性能报告"等场景时触发。
original-name: ops-profiling
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# 上板性能采集与调优

在真实 NPU 上采集算子性能数据，系统化解读指标文件，判定性能是否达标，定位瓶颈类型，并给出可操作的优化建议。

本技能保留 `msprof`、`msprof op` 两条独立路径。先按"外部选择优先"确定工具，只有 `msprof` 路径使用 10 组相邻预热与中位数的新计时协议；`msprof op` 保持自身的采集和归档口径。

## 高频问答速查（先查这里——本节答案即为权威结论，已与脚本实际行为对齐，无需再读源码验证）

| 问题 | 答案 |
|---|---|
| compare 模式下用例文件加载优先级？ | ① `<op>_perf_cases.jsonl`（标准命名）→ ② 任意 `*.jsonl` → ③ 任意 `*.json`（非 `.bak`）→ ④ `model.py` 的 `get_input_groups()`。同时存在时**选①**（详见下文「用例加载优先级」） |
| 怎么做标准可比较计时？ | `collect_perf_application.py`：一次 `msprof application` 覆盖完整 case 集，每例固定 10 组 `warmup → target`，取 target 中位数；大 case 集按会话上限自动拆分（详见「使用方式 1」与 [application_runner_contract.md](references/application_runner_contract.md)） |
| 怎么生成交互式 HTML 性能报告？ | 使用 [msopprof-visualization](msopprof-visualization/SKILL.md)：已有含 `collection_manifest.json` 的采集目录用 `run_pipeline.py --mode visualize` 重渲染；有算子工程用其 `run_pipeline.py` 采集并生成报告。`msprof_profile_run.sh` 的 `PROF_GROUP_*` 归档不能直接用于此渲染链路。 |
| 怎么做加速比对比？ | `msprof_profile_run.sh --compare`（全量）/ `--quick`（1 轮快速），输出 `performance.json` + markdown 报告即完成，**不**走可视化子技能 |
| quick / compare 如何保留原始 PROF？ | 推荐 `--keep-prof` 选项（quick 与 compare 模式均支持透传），作用是保留 msprof 原始 PROF 目录（默认在采集后清理），供后续深度分析或可视化工具导入；**不需要手工备份 PROF 目录，也不需要切换标准模式重新采集**。保留已有数据与补采缺失指标是不同需求，不能把重采作为保留目录的前提，也不能承诺标准模式可提供未经确认的逐指令流水数据。 |
| MC² / 多 rank（fork）算子用什么采集？ | **必须 `msprof`**，禁止 `msprof op`（详见 msprof-guide.md「MC² 多 rank 算子采集」） |
| 采集结果如何判定瓶颈？ | 用 `msprof_perf_summary.py` 解析 `PROF_GROUP_*`，输出瓶颈类型与各 pipe 占比，判定规则见 `references/msprof-guide.md`；需要 MBU/MFU/理论上限量化时用 `compute_efficiency.py`（[cost-model-contract.md](references/cost-model-contract.md)） |

### 阅读纪律（防上下文爆炸，必须遵守）

- `scripts/msprof_perf_summary.py`（2400+ 行）与 `scripts/msprof_profile_run.sh` 是**可执行工具，不是文档**。需要确认某个行为时，只允许 `grep -n "<函数名/关键字>" scripts/msprof_perf_summary.py` 定位后**读命中处前后 ~30 行**；**禁止整文件通读**。
- `references/` 下各文档按文末「参考资源」表的"何时查阅"**按需加载**；回答本文件已写明的事实性问题时，不需要打开任何 reference 或 script。
- 本 SKILL.md 覆盖全部使用决策；若与脚本实际行为发现不一致，以脚本行为为准并在本文件勘误。

---

## 工具选择

选择依据是**外部输入与探测结果**，不得凭固定偏好选路。外部未指定时，**先探测工具可用性**（如 `which msopprof msprof`；必要时核对 `$ASCEND_HOME_PATH` / `$ASCEND_OPP_PATH` 的 CANN 安装完整性），再按探测结果分支：

1. 外部输入显式指定 `msprof` 或 `msprof op`：原样执行。调用方工作流或其他 skill 给出的 profiler 选择属于外部输入，本 skill 不解析其业务约束，也不覆盖其选择。
2. **只有 msopprof 可用**（`msprof` 不在）：走 `msprof op`。
3. **只有 msprof 可用**（`msopprof` 不在）：走 `msprof`。
4. **两者都可用**：向用户确认，或遵循项目既有约定；未确认前不得默认选定。
5. **两者都不可用**：停止采集并报告 **CANN / ASCEND_HOME 环境问题**（CANN 未安装或 `$ASCEND_HOME_PATH` / `$ASCEND_OPP_PATH` 未配置），不猜测选路。

补充规则：
- 已选定 `msprof` 后，环境不稳定导致 application 数据无法通过身份、频率、launch 映射或波动门禁：回落到 `msprof op`。必须记录回落原因，不得混合两条路径的数据做前后对比。
- 环境稳定性必须由实际探测与采集产物判断，不能仅凭工具是否安装推断。

**MC² / 多 rank（fork）算子是硬约束**：算子通过 `fork()` 创建多个子进程绑定不同 NPU 卡、子进程间经 SHMEM UDMA / BarrierAll / CrossCoreFlag 协同通信（如 alltoall_matmul、allgather_matmul、matmul_reducescatter）时，无论外部是否指定，都**必须使用 `msprof`、禁止 `msprof op`**（`msprof op` 对 fork 程序的采集行为未定义）。加载 [`references/msprof-guide.md`](references/msprof-guide.md) 的「MC² 多 rank 算子采集」章节，同时参考 `ascendc-perf-optimize` skill 的 `references/comm-compute/index.md`「性能采集方法」章节。

| 工具 | 流程文档 | 用途 |
|------|----------|------|
| **`collect_perf_application.py`** | [`references/application_runner_contract.md`](references/application_runner_contract.md) | `msprof application` 标准可比较计时：每例固定 10 组 `warmup → target`，取 target 中位数；内置 CANN 版本适配（`--aic-mode=task-based` 版本相关、不完整安装预检、单会话 relevant-launch 上限（默认 540）自动拆分与空导出重试） |
| **`msprof_profile_run.sh`** | [`references/msprof-guide.md`](references/msprof-guide.md) | `msprof` 深度指标采集、兼容对比测试和批量并行 |
| **`msprof_perf_summary.py`** | [`references/msprof-guide.md`](references/msprof-guide.md) | 统一解析入口：瓶颈分析、对比报告生成、批量汇总 |
| **`perf_summary.py`** | [`references/msprof-op-guide.md`](references/msprof-op-guide.md) | msprof op 归档（需 msopprof） |
| **`compute_efficiency.py`** | [`references/cost-model-contract.md`](references/cost-model-contract.md) | 实测耗时 → **MBU/MFU/瓶颈类型/理论上限 + 优化结论**；成本模型(FLOP/字节)可注册或 CLI 直传 |

---

## 使用方式

采集 CANNBotDSL 算子的性能摸底数据时，读取 [DSL 公开入口测量](references/cannbot-dsl.md)，按该方法区分编译、公开入口与设备耗时，并保留完整设备任务证据。

### 1. `msprof` 标准可比较计时

一次 `msprof application` 覆盖完整 case 集。每个 case 固定执行 10 组相邻的 `warmup → target`，只统计每组 target 的完整 kernel sequence 总耗时，最终取 10 个 target 总耗时的中位数。不得改为均值、丢弃前若干次或追加样本择优。

warmup 固定按以下优先级选择：

1. 默认使用现成 ops：普通目标使用 `ArgMax`，目标属于 ArgMax 家族时使用 `Max`。
2. 只有 ops probe 失败时，才构建并使用内置 `OpsProfilingWarmup` 包。
3. ops probe 和内置包构建都失败时，采集失败。

先把内置包源码落地到目标工程，作为无 ops 环境时的构建后备；此步骤不默认编包：

```bash
python3 scripts/install_compiled_warmup.py --project-dir /path/to/operator-project
```

工程内持久 runner 有现成模板（[`references/application_runner_contract.md`](references/application_runner_contract.md)「现成 runner 模板」）：**有 torch/框架接口**用 `assets/app_runner/torch/torch_app_runner.py`；**只有 aclnn 接口**（算子已注册并编出 vendor 包、无框架绑定）用 `assets/app_runner/aclnn/` 纯 C++ 模板——warmup 直接走 aclnnArgMax/aclnnMax，不依赖 torch。两类模板只改标记的 ADAPTER 段，命令行、execution manifest 与 warmup 协议完全一致。

再按 [`references/application_runner_contract.md`](references/application_runner_contract.md) 实现工程内持久 runner，执行：

```bash
python3 scripts/collect_perf_application.py \
  --cases ./cases.csv \
  --runner-file ./run_cases.py \
  --identity-manifest ./perf-identity.json \
  --target-op Exp \
  --ops-warmup-probe-cmd 'python3 ./run_cases.py --probe-warmup-op {warmup_op}' \
  --warmup-build-cmd "bash ./ops_profiling_warmup_ext/build.sh --soc=ascend950" \
  --compiled-warmup-binary ./ops_profiling_warmup_ext/ops_profiling_warmup/_C.so \
  --application-cmd 'python3 ./run_cases.py --cases ./cases.csv --case-ids {case_ids} --samples {samples} --execution-manifest {execution_manifest} --warmup-mode {warmup_mode} --warmup-op {warmup_op}' \
  --aic-metrics PipeUtilization \
  --out-dir ./perf/baseline
```

采集和复用门禁见 [`references/msprof-guide.md`](references/msprof-guide.md) 与 runner 契约。

### 2. `msprof` 深度瓶颈分析（标准采集模式）

对单个可执行文件采集 7 组 aic-metrics + sample-based：

```bash
bash scripts/msprof_profile_run.sh --warm-up=3 --output=./msprof_output -- \
    ./matmul_tutorial_mxfp4_pingpong 8448 4096 4096

# 解析结果
python3 scripts/msprof_perf_summary.py ./msprof_output/PROF_GROUP_* <ops_dir>
```

该入口用于补充诊断。其多指标结果不能替代第 1 节的标准 `total_us`，也不能混入 baseline/candidate 的可比较计时。

若源码含循环内跨核同步或重复目标写入，按 [`references/msprof-guide.md`](references/msprof-guide.md#核间同步与负载偏斜的证据) 联合查看逐核负载、同步频次和任务依赖；把同步开销与负载不均衡作为待验证假设，不用聚合 busy 比例直接换算某个屏障的耗时。

### 3. 对比测试模式（kernel-level 加速比）

对算子目录下的 `model.py` vs `model_new_ascendc.py` 做对比测试：

```bash
bash scripts/msprof_profile_run.sh --compare --output-dir=./output/GELU --warm-up=3 --device=0
```

输出：`performance.json` + `performance.log` + `perf_report.md`

### 4. 快速模式（1 轮采集，只获取 kernel 时间）

对算子目录做快速对比测试，只跑 1 轮 msprof，不采集 7 个 aic-metrics：

```bash
bash scripts/msprof_profile_run.sh --quick --output-dir=./output/GELU --warm-up=3 --device=0
```

输出：`performance.json` + `performance.log` + `perf_report.md`

### 5. 批量并行模式（多 NPU）

扫描目录下所有算子子目录，多 NPU 并行执行对比测试：

```bash
bash scripts/msprof_profile_run.sh --batch --base-dir=./output_performance --max-jobs=7 --device-start=1
```

输出：各子目录 `performance.json` + `batch_performance.log` + `batch_report.md` + `batch_summary.json`

---

## 输入目录结构（对比模式）

**传统模式（端到端自动开发）：**
```
{output_dir}/
├── model.py              # 参考 PyTorch 实现
├── model_new_ascendc.py  # AscendC 实现
├── <op_name>.json        # 测试用例（JSON Lines）
└── kernel/               # AscendC kernel + whl 包
```

**JSONL 模式（ascend-kernel 工程结构）：**
```
csrc/ops/<op>/test/
├── <op>_perf_cases.jsonl   # JSONL 用例
├── model.py                # 参考实现
├── model_new_ascendc.py    # AscendC 实现
└── kernel/                 # kernel 工程
```

### 用例加载优先级

1. **优先**：`<op>_perf_cases.jsonl`（标准 JSONL 命名）
2. **回退**：任意 `*.jsonl` 文件
3. **回退**：任意 `*.json` 文件（非 `.bak`，JSON Lines 格式）
4. **回退**：从 `model.py` 的 `get_input_groups()` 加载

---

## 输出格式（对比模式）

Markdown 报告包含：
- **对比表**：`Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比`
- **全量汇总**：用例数、平均加速比、自定义/标杆更优条数
- **按数据类型汇总**：分 dtype 的统计
- **简短分析**：整体趋势结论
- **深度瓶颈分析入口**：提供 msprof 深度分析命令

额外输出：
- `performance.json` — 结构化数据（含 geomean/mean/median/min/max 加速比）
- `performance.log` — 屏显日志

---

## 深度分析：msprof / msprof op

当性能不达标或需要根因分析时，使用深度分析流程。

**arch22（A2/A3）支持详细性能分析**，可按 [CANN 官方算子性能分析文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/devaids/opdev/optool/atlasopdev_16_0097.html) 解读采集结果。该链接对应 CANN 8.0.0 alpha001；实际参数、指标与功能以目标 CANN/工具版本为准。arch35 的 RegBase/VF/RVEC 指标不用于推断 arch22 瓶颈，缺失字段不按零值处理。

工具路由只使用上方「工具选择」一节，不得在深度分析阶段重新选路。选择 `msprof` 时加载 [`references/msprof-guide.md`](references/msprof-guide.md)；选择 `msprof op` 时加载 [`references/msprof-op-guide.md`](references/msprof-op-guide.md)。

### 可视化报告（上板数据 → HTML）

当用户要求把上板采集数据可视化为交互式 HTML 报告（Details、Roofline、Timeline、Cache 热力图、Source 等页面）时，加载子技能 [`msopprof-visualization/SKILL.md`](msopprof-visualization/SKILL.md)：

- 有算子工程可上板 replay → 用其 `run_pipeline.py` 做最小非重复采集并生成报告；
- 已有含 `collection_manifest.json` 的采集目录 → 只用 visualize 模式重渲染，不得重复采集。

以下场景**不加载**可视化子技能：

- 加速比对比（compare / quick / batch）：输出 `performance.json` + markdown 即完成；
- MC² / fork 多进程算子：子技能基于 `msprof op`，该场景对 `msprof op` 禁用；
- 纯瓶颈分析或 CSV 指标解读且用户未提及可视化。

注意两套链路输出不互通：本技能的 `PROF_GROUP_*` 归档不能直接喂给可视化渲染，可视化子技能的采集目录结构由其自身 `collect.py` 产生。

### 参考资源

| 文件 | 内容 | 何时查阅 |
|------|------|---------|
| [`references/msprof-guide.md`](references/msprof-guide.md) | `msprof`：标准可比较计时、深度采集 / 归档 / **主 Bound 判定** / **MC² 多 rank 采集**（文末章节） | 选用 `msprof` 时，或 MC² / fork 多进程算子 |
| [`references/msprof-op-guide.md`](references/msprof-op-guide.md) | `msprof op`：构建 / 采集 / 归档 / 判定 / 瓶颈 / 回归 | 外部选择或环境回落到 `msprof op` 时（不适用于 MC²） |
| [`references/application_runner_contract.md`](references/application_runner_contract.md) | application runner、manifest、launch 映射与中位数协议 | `msprof` 可比较计时 |
| [`references/csv_fields_reference.md`](references/csv_fields_reference.md) | CSV 字段定义与阈值 | 理解指标含义时 |
| [`references/optimization_quickref.md`](references/optimization_quickref.md) | 瓶颈类型与优化方法 | 定位瓶颈后 |
| [`references/cost-model-contract.md`](references/cost-model-contract.md) | 成本模型契约、MBU/MFU/天花板计算、峰值口径 | 需要量化 bound / 天花板 / 前置剪枝时 |
| [`references/compiled_warmup.md`](references/compiled_warmup.md) | 内置 warmup 包的构建与使用 | ops probe 失败需要构建后备 warmup 时 |

---

## 错误处理

- 外部显式指定的工具不可用：直接报告缺失，不擅自换工具。
- `msprof` 路径持续无法通过环境或数据门禁：保存失败证据，记录原因后回落到 `msprof op`。
- application launch 映射不完整、乱序或身份不一致：整份 collection 作废，不手工补值。

---

## 适用场景总览

| 场景 | 推荐命令 | 说明 |
|------|----------|------|
| 标准可比较性能采集 | `collect_perf_application.py` | `msprof application`，每例 10 组 `warmup → target`，取 target 中位数；大 case 集自动按会话上限拆分 |
| **MC² 多 rank 算子采集** | `msprof --aic-mode=task-based` | **必须用 msprof，禁止 msprof op**，详见 msprof-guide.md「MC² 多 rank 算子采集」章节 |
| 算子开发完成后的性能验收 | `msprof_profile_run.sh --compare` | 快速对比自定义算子 vs 标杆 |
| 性能问题定位 | `msprof_profile_run.sh` + `msprof_perf_summary.py` | 深度瓶颈分析 |
| 优化效果验证 | `msprof_profile_run.sh` + `msprof_perf_summary.py` | 对比优化前后的归档数据 |
| 算子生成阶段快速测试 | `msprof_profile_run.sh --quick` | 快速对比测试（1 轮采集，只获取时间） |
| Agent team 测试阶段 | `--quick` / `--compare` + 标准采集 | 先快速对比，不达标再深度分析 |
| 端到端自动开发 Phase 5 | `msprof_profile_run.sh --quick` | 集成到 tilelang2ascendc-ops-generator 流程（只获取加速比） |
| 进化优化前基线测试 | `msprof_profile_run.sh --quick` | 快速获取基线加速比 |
| 批量性能测试 | `msprof_profile_run.sh --batch` | 多 NPU 并行批量测试 |
| 上板数据可视化报告 | `msopprof-visualization` 子技能 | 交互式 HTML：Details/Roofline/Timeline/Cache/Source 等页面 |
