---
name: external-cannbot-community-model-model-recommend-core-limit-stress
description: 限核压测与性能摸高（自定义优化交付件开发完成后的最后一步）。在单卡昇腾 NPU + GE_SESSION_DEMO 上，先做 HF32 前置门禁（MatMul
  主导时先开启并验证 avg 收益），再固定设备与 batch、做 CPU 绑核、CORE_NUM=AIC|AIV 找最低时延，最后增大 THREAD_NUM 在
  TP99 约束下找最大 QPS。只改运行参数，不改模型、demo 推理/运行逻辑。触发：限核压测、性能摸高、核心数寻优、并发压测、QPS 最大化、TP99 约束、CPU
  绑核、AIC/AIV 限核、HF32。
original-name: model-recommend-core-limit-stress
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-11-28'
synced-commit: 7a09cdfac4b2869dca6b6fd4531a0a5e539d6096
license: UNKNOWN
---

# 限核压测与性能摸高

## 何时用 / 边界

- 用于**自定义优化交付件（插件/Pass/自定义算子）完成并验证之后**的最后一步：在单卡上把部署性能压到上限。
- 只支持 **onnx + GE_SESSION_DEMO**；pytorch/HF 后续再扩展。
- 不做：模型迁移/报错修复、插件/Pass/算子开发、精度问题根因、精度模式调整、batch 调优。

**只改运行参数**，不改模型图、demo 推理/运行逻辑、精度模式与融合/编译选项；`test.sh` 的 profiling 采集/导出修复不算越界。HF32 是 `runtime_config` 优化开关，不是精度模式，按“Step 1 前置门禁”处理。

## 前置固定项（开始前确认）

| 项 | 要求 |
| --- | --- |
| 设备 | 单卡；`--device` / `START_DEVICE` 开始前固定；多卡不作为调优项 |
| 设备占用 | 每轮扫描前 `npu-smi info` 确认目标 die 无其他进程；有占用则记录 PID/进程名，等待释放或隔离 |
| batch | 开始前固定；**本 skill 不扫描 batch** |
| 模型与输入 | 与优化定稿一致，不改 |
| 精度模式 | 与优化定稿一致，不改；**常规限核不做精度门禁**。HF32 状态由 Step 1 决定，需单独 `--verify` |
| HF32 输入 | analysis 建议中的 `matmul_ratio` 与 priority；无建议时由用户提供或按 Step 1 计算 |
| 测量口径 | `CircleCount ≥ 5000`、`repeat = 3`、`DebugFreq = 1000`、`PrintOutput=False` |
| TP99 阈值 | **先问用户**；未提供则按后文"拐点规则"判定，不得静默跳过 |

## 测试对象（两个主限制 + 绑核/HF32 前置）

| 测试项 | 作用 | 阶段目标 |
| --- | --- | --- |
| **CORE_NUM = AIC\|AIV** | 限核 | 找**最低时延** |
| **THREAD_NUM** | 并发 | 找**最大 QPS**（TP99 约束） |
| CPU 绑核 | 亲和性 | 找**最低时延**（A/B） |

- `GRAPH_MAX_PARALLEL` 不是独立测试项：只确认 `THREAD_NUM` 能否大于该值，记录观察即可。
- HF32 是前置门禁，不属于主限制扫描；按“Step 1”执行，结果决定后续调优状态。
- 上述固定项均不参与扫描。
- 本 skill 只产出**推荐档**最终配置（TP99 约束下 QPS 最优），不另出保守/激进档。

## 指标与口径

- **HF32 门禁**：主看端到端 `avg`；`avg_gain_pct >5%` 且 `>噪声带` 才保留。
- **时延阶段**（CPU 绑核、CORE_NUM）：主看 `avg`，`TP99` 为约束/并列判定；记录 `TP999`。
- **摸高阶段**（THREAD_NUM）：主看 `QPS`，`TP99` 为约束；记录 `avg/TP99/TP999`。
- 复测：同配置 3 次取中位数；可剔除首样本与最大偏差，但必须保留原始单次结果与剔除记录；记录变异系数 `cv`；改善未超过噪声带不判优。

## 压测流程

```
Step 0 固定与确认：设备、batch、TP99 阈值、测量口径、npu-smi 无占用、读取 matmul_ratio
Step 1 HF32 前置门禁：按 >20% / ≤20% 决定是否开启；先 --verify，再做 off/on A/B；avg 收益 >5% 且 >噪声带则保留，否则关闭
Step 2 基线：在最终 HF32 状态（开/关）下采集 avg/TP99/TP999/QPS
Step 3 CPU 绑核：不绑 vs 候选亲和性，找最低时延
Step 4 限核分析：读 AIC/AIV 算子占比与（可选）kernel 时长，预测最优限核
Step 5 限核细化：先测预测点 → 黄金分割/二分 → 邻点校验，找最低时延
Step 6 确认"最低时延标准"：固定最佳绑核 + 最佳 CORE_NUM，记录
Step 7 THREAD_NUM 摸高：在最佳绑核 + CORE_NUM 下增大并发，找最大 QPS
Step 8 定稿：最终配置（推荐档）复测 3 次，写入 best_ledger.json（独立使用时回填交付书）
```

> Step 1 决定 HF32 开/关，Step 2–8 全部在该状态下进行；不得先调优后切换 HF32。

> Step 3–5 找**最低时延**，Step 6 固定该标准；之后才进入"增大 THREAD_NUM 摸最大 QPS"。以上命令均在 `demo/` 下执行。

## Step 1：HF32 前置门禁

```text
matmul_ratio = Σ(MatMul / BatchMatMul / BMM 及其变体 kernel time) / kernel_sum
avg_gain_pct = (avg_off - avg_on) / avg_off × 100%
```

- `matmul_ratio` 从 profiling 的 `op_statistic` / `op_summary` 汇总；无 profiling 时由用户提供，不臆造。

| 条件 | 动作 |
| --- | --- |
| `matmul_ratio > 20%` | 必须评估：开启 `ENABLE_HF32=01`，先 `--verify`，再做 off/on A/B |
| `matmul_ratio ≤ 20%` | 可选尝试：同样 `--verify` + A/B，记录数据 |

执行顺序：

1. 先测当前配置 `bash run.sh <id> --verify`，确认 golden 可用。
2. 开启 `ENABLE_HF32=01`（仅 MatMul；含 Conv 可评估 `11`），重跑 `--verify`。
3. PASS 后，在**同一设备/batch/初始 threads/cores（未限核）** 下做 HF32 off/on A/B，3 次中位数，测 `avg/TP99/TP999/QPS`。
4. 判定：
   - `avg_gain_pct > 5%` 且 `> 噪声带` → 保留 HF32，后续 Step 2–8 全部在开启状态进行；
   - 否则 → 关闭 HF32，记录已收集的 HF32 性能数据，后续 Step 2–8 在关闭状态进行。
5. 精度失败、NaN/Inf、误差超容差或无 golden → 直接关闭，不进入 A/B 结论。
6. 结果写入 `stress_rounds_<ts>.md`（工作流模式下交付书结论由 `model-recommend-regression-baseline-mgmt` 汇总；独立使用时另回填交付书）；**不改变 `best_ledger.json` 三轴结构**。

> HF32 是 `runtime_config` 优化项，不是精度模式；未通过 `--verify` 不得作为最终配置。

## CORE_NUM 测试策略（分析优先，不枚举）

1. **分析预测（先算再测）**
   - 读结构档案的 AIC/AIV 算子计数与占比（AIC：MatMul/Conv 等；AIV：elementwise/reduce/查表等）。
   - 若有 profiling，读 AIC/AIV 时长分布或 Task Type（AI_CORE/AI_VECTOR_CORE）时长比。
   - 判定主导侧：AIC 主导 → 优先保 AIC；AIV/查表主导 → 优先保 AIV；不确定时取本机上限的中值（`24|48` 仅为示例，必须从环境读取）。
   - 预测候选：主导侧固定在本机上限（保瓶颈资源），**只下调非主导侧**寻找最低时延。
   - 上限必须来自环境（`npu-smi` / `PlatformAscendC`），禁止硬编码。
2. **先测预测点**：直接跑该候选（`THREAD_NUM=1`、最佳绑核），记录 avg/TP99。
3. **黄金分割/二分**：对调优维度做黄金分割；若实测单调可用二分加速。
4. **停止条件**：不设固定评估次数；**连续 3 次评估的 avg/TP99 变化均小于噪声带即停止**（收敛或到达本机上限/下限）；10% 仅作边际 QPS 的辅助判断。
5. **邻点校验**：对最优点两侧各测 1-2 个相邻值，防非单峰遗漏。
6. **第二轮**：主导侧利用率 >80% 或邻点不单调时，对主导侧做一次局部细化（±4，步长 1）。
7. **判定**：avg 最小；avg 相同取 TP99 更小。只记录最优一次。
8. **可选执行器**：`tools/sweep.py` 的 Phase1（固定 AIC 全量、黄金分割搜 AIV、`--core-budget`）可直接复用。

## THREAD_NUM 测试策略

1. 固定最佳 CPU 亲和性与最佳 CORE_NUM。
2. 从 `THREAD_NUM=1` 开始递增（建议 1,2,3,4,5,6,7,8，可按需扩展）。
   - 线程收益上限 ≈ **单请求 kernel 并行度 × 核数上界**；超过后 QPS 平台化、TP99 恶化。
   - **争抢信号**：QPS 增长放缓但 TP99 陡增 → 线程过多或限核失衡，回调线程/核数后复测。
   - **饱和判定（Little 定律/排队视角）**：并发增加后单实例 avg/TP99 线性上升 → 资源已饱和；QPS 不再增长但时延上升 → host/同步/串行点未消除，回到分析定位。
3. 若用户给了 TP99 阈值：只在 `TP99 ≤ 阈值` 的配置中取 **QPS 最大**。
4. 若未给阈值：按**拐点规则**停止。
   - 以 `THREAD_NUM=1` 的 QPS 作为线性基准；
   - 逐点比较 QPS 边际增量；当增量明显低于线性基准（默认容差 10%）记为"非线性点"；
   - 出现非线性后**继续再测**；若连续 2 个点仍未恢复线性增长则停止；
   - 在所有已测点中取 **QPS 最大**。
   - 例（线程:QPS）：1:400、2:800、3:1200、4:1600（线性）、5:1800（非线性）、6:1900、7:1950 → 测到 7 后停止，取 QPS 最大点。
5. 确认 `THREAD_NUM` 能否大于 `GRAPH_MAX_PARALLEL` 及其影响，记录观察；不作为独立测试项。
6. 只记录最优一次。

## CPU 绑核策略

1. 先读环境拓扑：`lscpu`、`numactl -H`、`npu-smi info`（NUMA 与 CPU 核分布），**不硬编码核号**。
2. 优先选择与 NPU 同 NUMA 节点的 CPU 核；避免与其他繁忙进程争抢。
3. A/B 对比：不绑 vs 候选亲和性（`taskset -c <cpulist>` 或 `numactl --cpunodebind/--membind <node>`）。
4. 在 `THREAD_NUM=1` 下比较 `avg/TP99`，取最低时延的亲和性；只记录最优一次。
5. 绑核通过命令前缀包装 `run.sh`/`test.sh` 实现。

## 记录与回填

- **最优记录（结构化，供工作流）**：`models/M<id>/sweep/best_ledger.json`
  - 每个测试项只保留最优一条：参数 + avg/TP99/TP999/QPS + 可复现命令。
  - 同一测试项不重复记录多轮结果。
- **逐轮记录**：`models/M<id>/sweep/stress_rounds_<ts>.md`
  - 记录每一轮的"参数 + 性能 + 噪声带 + 停止原因 + npu-smi 快照"，用于追溯。
  - HF32 前置门禁记录：`matmul_ratio`、`ENABLE_HF32` 值、`--verify` 结果、A/B 指标、`avg_gain_pct`、最终决策（开/关）。
- **交付书**：
  - 被工作流调用时：**不写** `model_delivery_manual.md`，由 `model-recommend-regression-baseline-mgmt` 生成；本 skill 只提供 `best_ledger.json` 与 `stress_rounds`。
  - 独立使用时：可将 `best_ledger.json` 的最优项回填 `models/M<id>/model_delivery_manual.md`。
- **复现命令**必须包含：`MODEL_ID / THREAD_NUM / CORE_NUM / START_DEVICE / CircleCount / DebugFreq` 与绑核前缀。
- 交付书建议表格（工作流合并时使用）：

| 测试项 | 参数 | avg(ms) | TP99(ms) | TP999(ms) | QPS | 复现命令 |
| --- | --- | --- | --- | --- | --- | --- |
| CPU 绑核 | `<cpulist/numa>` |  |  |  |  |  |
| CORE_NUM | `AIC\|AIV` |  |  |  |  |  |
| THREAD_NUM | `N` |  |  |  |  |  |
| HF32 前置门禁 | `ENABLE_HF32=01`（如保留） |  |  |  |  |  |
| 最终配置 | 绑核 + AIC\|AIV + N |  |  |  |  |  |

## 工具使用

- 主路径：用 `run.sh <id> --cores/--threads/--circle`（及 `taskset`/`numactl` 包装）逐点压测。
- 可选：`tools/sweep.py` 作辅助——Phase1 为"固定 AIC 全量、黄金分割搜 AIV、`--core-budget`"；它**不含 CPU 绑核与拐点停止**，由 skill 侧补齐；**不得修改工具代码**。
- 不改 `run.sh` / `test.sh` / `src/` / `tools/` 的运行逻辑；profiling 采集/导出修复除外。

## 验收标准

- CPU 绑核、CORE_NUM、THREAD_NUM 三项各自有最优记录（`best_ledger.json` 各一条；工作流合并到交付书）。
- 限核选择有**分析依据**（AIC/AIV 占比或 kernel 时长），不是全枚举。
- 最终仅交付**推荐档**配置；若使用 `tools/sweep.py`，记录其调用参数。
- 最终配置复测 3 次稳定（`cv` 在合理范围），满足 TP99 约束。
- HF32 前置门禁有结论：`matmul_ratio`、`--verify`、A/B、`avg_gain_pct`、开/关决策；后续调优与最终配置一致。
- 逐轮记录完整、可追溯；复现命令可直接复跑。
- 未修改模型、demo 推理/运行逻辑、精度模式；HF32 已验证精度并记录实际值。

## 风险与回退

- 限核过低可能 NaN/timeout：回退到上一组核数并记录。
- 并发增大可能导致 TP99 拖尾：按 TP99 约束或拐点规则停止。
- 噪声导致误判：必须 3 次复测，改善小于噪声带不判优。
- 设备被占用：先确认空闲，避免污染结果。
- HF32 可能降低精度；未通过 `--verify` 或超出容差必须回退；不得在未确认精度时作为最终配置。
- 所有改动均为运行参数，回退只需恢复上一组参数，无文件回退。

## 与 GE 自定义交付件 skill 的关系

- 作为自定义优化交付件完成后的最后一步使用；实现逻辑不依赖 GE 自定义交付件 skill。
- 测试验收只使用 GE_SESSION_DEMO 的 CLI（`run.sh` / `test.sh` 的参数与环境变量），不接入其它测试框架。
