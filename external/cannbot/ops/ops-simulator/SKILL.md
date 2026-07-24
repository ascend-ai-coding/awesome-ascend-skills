---
name: external-cannbot-ops-ops-simulator
description: NPU 仿真器技能。提供 CANN Simulator 的使用指导，包括精度仿真、性能仿真、流水线分析。当需要在无 NPU 硬件环境下验证算子功能、分析性能瓶颈、定位精度问题时使用。也用于分析已有的
  cannsim 性能报告（summary.json）并给出优化建议。
original-name: ops-simulator
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# NPU 仿真器使用指南

## 概述

CANN Simulator 是一款面向算子开发场景的 SoC 级芯片仿真工具，通过 `cannsim` 命令行工具提供以下能力：

- **精度仿真**：输出 bit 级精度结果，协助算子精度验证
- **性能仿真**：输出指令流水图，协助定位性能瓶颈


### cannsim 主命令

`cannsim` 是性能仿真分析的命令行入口，提供两个子命令：

| 子命令 | 功能 | 说明 |
|--------|------|------|
| `cannsim record` | 执行仿真 | 在仿真环境中运行用户程序，记录仿真数据 |
| `cannsim report` | 生成报告 | 基于仿真结果生成性能分析报告和流水线图 |

**使用方式**：`cannsim <子命令> [选项]`

## 适用场景

| 场景 | 说明 |
|------|------|
| 无 NPU 硬件环境 | 在没有真实 NPU 硬件的情况下进行算子开发 |
| 精度验证 | 需要 bit 级精度验证的场景 |
| 性能调优 | 需要分析指令流水、定位性能瓶颈 |
| 资源受限 | 芯片资源紧缺时的替代验证方案 |

### 使用约束

| 约束项 | 说明 |
|--------|------|
| 芯片限制 | 仅支持 Ascend 950 芯片架构 |
| 单卡场景 | 仅支持单卡，代码中只能设置为 0 卡 |
| 算子类型 | 仅支持 AI Core 计算类算子（不支持 MC2/HCCL） |
| 架构限制 | 支持 X86，ARM |

## 使用步骤

### 1. 执行仿真

```bash
# 精度仿真 + 性能仿真（生成报告）
cannsim record ./ascendc_kernels_bbit -s Ascend950 --gen-report

# 指定输出目录
cannsim record ./ascendc_kernels_bbit -s Ascend950 --gen-report -o ./sim_output

# 传递算子自定义参数
cannsim record ./ascendc_kernels_bbit -s Ascend950 --gen-report -u "--shape 1024,1024 --dtype float16"
```

### 2. 生成性能报告

```bash
# 从仿真结果生成流水线报告（默认当前目录）
cannsim report -e ./cannsim_Ascend950_*

# 指定输出目录
cannsim report -e ./cannsim_Ascend950_* -o ./report_output

# 指定查看的 Core ID
cannsim report -e ./cannsim_Ascend950_* -n 0         # 查看单个 core
cannsim report -e ./cannsim_Ascend950_* -n 0-2       # 查看 core 范围
cannsim report -e ./cannsim_Ascend950_* -n 0-2,5,12-14  # 混合格式
cannsim report -e ./cannsim_Ascend950_* -n all       # 查看所有 core
```

### 3. 查看输出文件

```
cannsim_output/
├── cannsim.log               # 仿真执行日志
└── report/
    ├── trace_core0.json      # 指令流水图文件
    └── ...
```

### 4. 性能瓶颈定位（Trace 空泡分析）

cannsim 生成的 `trace_core*.json` 是 Chrome Trace Format，包含每个 core 各 pipeline（SCALAR/VECTOR/MTE2/CUBE 等）的指令级周期事件。按 `references/pipeline-bubble-analysis.md` 的判定标准自主分析：

> 提示：trace 分析成本高。建议先用 `summary.json` 做快速诊断（见下方"性能分析"章节），仅在需要确认根因时下钻到 trace。

**分析目标**：定位主 pipeline（VECTOR/CUBE）idle 空泡的根因。

**核心方法**：
1. **空泡分类**：按 7 大类 / ~30 子类型标准判定等待原因（DATA_STALL、SCALAR_OVERHEAD、STRUCTURAL 等）
2. **因果归因**：检查"主 pipeline idle 时谁在 busy"，追踪具体阻塞源
3. **周期性模式检测**：判断空泡是系统性（每个 tile 重复）还是偶发
4. **重叠度判定**：验证双缓冲是否生效（MTE2 与 VECTOR/CUBE 时间重叠比例）

## 命令参考

### cannsim record - 执行仿真

在 AscendOps 仿真环境中运行用户程序，记录仿真数据。

**基本语法**：`cannsim record <user_app> -s <SOC_VERSION> [选项]`

| 参数 | 简写 | 必填 | 说明 |
|------|------|------|------|
| `user_app` | - | 是 | 用户编译后的可执行程序路径 |
| `--soc-version` | `-s` | 是 | 目标芯片版本（如 Ascend950） |
| `--gen-report` | `-g` | 否 | 仿真结束后生成性能报告 |
| `--output` | `-o` | 否 | 仿真结果输出目录 |
| `--user-option` | `-u` | 否 | 传递给用户程序的自定义参数 |

**示例**：
```bash
cannsim record ./ascendc_kernels_bbit -s Ascend950 --gen-report
cannsim record ./ascendc_kernels_bbit -s Ascend950 --gen-report -o ./sim_output
cannsim record ./ascendc_kernels_bbit -s Ascend950 --gen-report -u "--shape 1024,1024"
```

### cannsim report - 生成性能报告

基于仿真结果生成可视化的性能分析报告和指令流水线图。

**基本语法**：`cannsim report -e <EXPORT_FOLDER> [选项]`

| 参数 | 简写 | 必填 | 说明 |
|------|------|------|------|
| `--export` | `-e` | 是 | 仿真结果文件夹路径（cannsim record 的输出） |
| `--output` | `-o` | 否 | 流水线图输出目录 |
| `--core-id` | `-n` | 否 | 指定 Core ID（支持 0、0-2、0-2,5、all 等格式） |

**示例**：
```bash
cannsim report -e ./cannsim_Ascend950_*
cannsim report -e ./cannsim_Ascend950_* -n 0-2
cannsim report -e ./cannsim_Ascend950_* -n all -o ./report_output
```

> 详细命令参数、输出目录结构、返回值说明见 [references/simulator-advanced.md](references/simulator-advanced.md)

## 常见问题

| 问题 | 解决方法 |
|------|---------|
| 仿真失败 | 确保代码中只设置 0 卡，仅使用 AI Core 计算算子 |
| 性能报告未生成 | 确保使用 `--gen-report` 参数 |
| 找不到仿真结果 | 使用 `-o` 指定输出目录，或检查当前目录下的 `cannsim_*` 文件夹 |

## 参考资料

| 文件 | 内容 | 何时查阅 |
|------|------|---------|
| `references/simulator-advanced.md` | 仿真进阶命令参考 | 需要高级参数或批量仿真时 |
| `references/troubleshooting.md` | 问题排查指南 | 仿真失败或报告未生成时 |
| `references/pipeline-bubble-analysis.md` | 指令流水空泡分类、因果归因、周期性模式检测 | 生成 trace 后，定位性能瓶颈根因时 |
| `scripts/trace_bubble_analyzer.py` | 自动化空泡分析脚本（兼容 msprof/cannsim 双格式） | 批量分析多核 trace 或获取预分析报告时 |
| `references/performance-metrics-reference.md` | summary.json 字段、阈值、Analysis Priority | 分析 summary.json 时 |
| `references/performance-issues-general.md` | 多核负载不均衡 + Kernel 利用率不足 | 分析 summary.json 时 |
| `references/performance-issues-aic.md` | AIC: CUBE/MTE2/MTE1/FIXPIPE/SCALAR + L0C→UB | 分析 summary.json 时 |
| `references/performance-issues-aiv.md` | AIV: VECTOR/MTE2/MTE3/SCALAR + SIMT | 分析 summary.json 时 |
| `references/performance-issues-template.md` | 新增 issue 条目的规范 | 维护 |

---

## 性能分析（Performance Analysis）

分析已有 `summary.json`（无需重跑仿真），定位瓶颈并给出优化方向。

> **summary.json vs trace — 如何选择**：从这里开始。`summary.json` 成本很低（几 KB 的聚合指标），下方快速诊断流程能解决大部分问题。只有当 summary 已指明瓶颈类型但未定位具体原因时（例如 overlap 偏低但源码中已开启双缓冲，或需要查看具体哪条指令在何时阻塞流水线），才需要下钻到 trace 级空泡分析（上方 §4，`references/pipeline-bubble-analysis.md`）。先用 summary，再用 trace 确认因果链。

### 分析工作流

1. **生成报告** — 执行 `cannsim record --gen-report` 或 `cannsim report`
2. **读取 `summary.json`** — 位于 `sim_output/cannsim_*/report/summary.json`
3. **检查 `top_level_diagnosis`** — 读取 `dominant_pipeline` 和 `imbalance_ratio` 获得初步判断
4. **识别瓶颈类型** — 按下方快速诊断表定位，然后查阅对应的 reference
5. **推荐优化方案** — 从对应 issue reference 中提取具体修复动作
6. **迭代验证** — 重新运行仿真，对比指标变化

### summary.json 结构

`summary.json` 包含 8 个段落；其中 6 个始终存在，另外 2 个（`cache`、`bandwidth`）可能缺失。完整的字段定义、阈值和标准的"分析优先级"排序见 [performance-metrics-reference.md](references/performance-metrics-reference.md)（唯一可信源）。

### 快速诊断（Quick Diagnosis）

**Step 0 — 核数检查**：检查 `kernel_info.ai_core_active`。如果在多核芯片上 `== 1`，先确认确实是单核运行：用 `blockDim` 或 `per_core` 数组长度核对——若 `blockDim > 1` 但仅采到一个核的数据，则属于 profiler 采样假象（kernel 实为多核，`imbalance_ratio` 仍为 1.0、`per_core` 缺失，真实不均衡仍可能存在），此时不要给出"开满核"的结论，应按多核不均衡走结构化检查。确认确为单核后 — 让所有核都参与计算是最优先的修复，优先级高于下方所有瓶颈类型规则。此时 `imbalance_ratio` 会是 `1.0`（只追踪了一个核），**不应**被理解为"负载均衡"；所有接近零的 overlap 都是单核运行的假象，不是双缓冲问题。请跳转到 [通用问题 §1.1](references/performance-issues-general.md)，在 kernel 变为多核运行之前不要继续执行下方瓶颈类型表。

**Step 1 — 多核负载均衡**：检查 `top_level_diagnosis.imbalance_ratio`
- `> 1.3` → 需要做负载均衡 tiling
- `> 2.0` → 严重不均衡，应在处理其他问题前优先解决此问题

**Step 1.5 — Kernel 利用率合理性检查**：如果 `dominant_pipeline_util < 0.50` **且** 所有 `pipeline_overlap.*` 接近零 **且** 计算流水空闲 — 对于 **Cube** 型 kernel `AIC_CUBE.mean < 0.10`，对于 **纯 Vector** 型 kernel `AIVx_SIMD.mean` **与** `AIVx_SIMT.mean` 均 `< 0.05` — 说明 kernel 工作量不足以支撑分析（可能是 `blockDim` 对 shape 而言过大，或 shape 本身太小）。`imbalance_ratio` **不**是必要条件（均衡但 shape 过小同样适用）。请跳转到 [通用问题 §2 Kernel 利用率不足](references/performance-issues-general.md)；不要继续执行 Step 2 — 瓶颈类型规则会误判。

**Step 2 — 识别瓶颈类型**，基于 `top_level_diagnosis.dominant_pipeline`（下表源自 [performance-metrics-reference.md](references/performance-metrics-reference.md) §2 利用率解读 / §3 — 该文件是阈值的唯一可信源）：

| 主导流水线 | 瓶颈类型 | 下一步检查的关键指标 | 主要优化动作 |
|------------|----------|---------------------|-------------|
| `AIC_CUBE` (mean > 0.80) | CUBE 瓶颈 | `pipeline_overlap.AIC_MTE2_vs_AIC_CUBE`、`AIC_MTE1_vs_AIC_CUBE` | 核间 Cube/Vector 交错；对 L0A/L0B 和 L1 开启双缓冲 |
| `AIVx_SIMD` (mean > 0.50) | VECTOR 瓶颈 | `aiv_vector_instructions.ub_traffic_ratio` | 若 `ub_traffic_ratio ≥ 1.0`，使用 VF RegAPI |
| `AIVx_SIMT`（最高） | SIMT 瓶颈 | `SIMT_ExecIPC`、`SIMT_BranchIPC` | 已是 SIMT 模式；若分支发散（`SIMT_BranchIPC` 偏低），降低分支发散。参见 [AIV §5](references/performance-issues-aiv.md) |
| `AIC_MTE2` / `AIVx_MTE2` | MTE2 瓶颈 | `bandwidth.*.avg_transaction_gbps` | 开启双缓冲；若 DMA 粒度较小，使用 UB 批量搬运 |
| `AIC_MTE1` | MTE1 瓶颈 | `pipeline_overlap.AIC_MTE1_vs_AIC_CUBE` | 对 L0A/L0B 开启双缓冲 |
| `AIC_FIXP` | FIXPIPE 瓶颈 | `pipeline_overlap.AIC_FIXP_vs_AIC_CUBE` | 若 < 0.30：CUBE 因排空 L0C 而阻塞 → 增大 N 轴 tile 尺寸 |
| `AIC_SCALAR` / `AIVx_SCALAR` | SCALAR 瓶颈 | `scalar_instructions.*.load_store_ratio`、`pipeline_overlap.AIC_SCALAR_vs_AIC_CUBE` | 若 ratio ≥ 0.30 则存在溢出；若 SCALAR_vs_CUBE overlap 偏低（< 0.20）则为反压（按 CUBE Bound 处理）|

**Step 3 — 带宽/搬运交叉验证**（独立于 `dominant_pipeline`）：
- 若 `bandwidth` 存在：扫描是否存在*冗余搬运* — 最常见的是 `AIC_FIXPIPE_L0C_TO_OUT` + `AIVx_MTE2_OUT_TO_UB` 搬运相同数据。此特征说明 matmul tile 经由 GM 绕了一圈，可通过 L0C→UB 直连 Fixpipe 消除。
- 若 `bandwidth` 缺失：使用备用信号 — 在一个不应需要 AIV 侧 GM 输入的 kernel 中 `pipe_utilization.AIVx_MTE2.mean > 0`，加上 `pipeline_overlap.AIC_CUBE_vs_AIVx_VEC = 0`。根因相同，修复方法相同。

参见 [performance-issues-aic.md §5](references/performance-issues-aic.md)。

详细诊断树和修复方法：
- [performance-issues-general.md](references/performance-issues-general.md) — 多核负载不均衡
- [performance-issues-aic.md](references/performance-issues-aic.md) — CUBE / MTE2 / MTE1 / FIXPIPE 瓶颈（AIC）、Cube→Vector L0C→UB 直连、SCALAR 瓶颈（AIC）
- [performance-issues-aiv.md](references/performance-issues-aiv.md) — VECTOR 瓶颈、MTE2/MTE3 瓶颈（AIV）、流水线重叠
