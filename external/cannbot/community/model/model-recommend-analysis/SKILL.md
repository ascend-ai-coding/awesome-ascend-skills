---
name: external-cannbot-community-model-model-recommend-analysis
description: 推荐模型 onnx 推理在适配完成后的"分析 + 优化建议"（单次）。优化信息只来自模型图（onnx 原图 + dump_graph 各阶段）与
  Profiling（step_trace/op_statistic/op_summary/api_statistic/task_time/fusion_result
  等解析结果）。先做数据质量门禁与总账分流，再按 host/算子/gap/图结构逐层定位，输出 Pass/Plugin/自定义算子建议，并给出证据、收益上界、成本与暂缓项；MatMul
  主导时可给出 HF32 运行配置建议。触发：适配后分析、Profiling 分析、瓶颈定位、优化点定位、图结构验证、优化建议、HF32、GE Session 优化。
original-name: model-recommend-analysis
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-11-28'
synced-commit: 7a09cdfac4b2869dca6b6fd4531a0a5e539d6096
license: UNKNOWN
---

# 适配后分析与优化点定位

必读：数据质量 → 总账 → Host → 算子 → gap → 图结构 → 收益上界
按需：实验统计、深化分析、跨模型（加载 `references/deep.md`）；HF32（MatMul 主导时）
输出前：读取 `assets/report_contract.md`

## 何时用 / 边界

- 前置：模型已适配跑通、功能验证通过、已有可复现基线（由工作流提供）。
- 只支持 **onnx + GE_SESSION_DEMO + 单卡**。
- 只读分析：不采集、不解析、不实施、不验证、不做循环/收口/压测。
- 深化分析（Roofline/消融/动态 shape/并发/跨模型）由 `references/deep.md` 按需承担。
- 融合规划与收益排序（候选清单/bound/双轨分流）见 `references/planning.md`。
- 不修改 onnx 原图、不做 graph-rewrite；优化只通过 Pass/Plugin/自定义算子/runtime_config。
- 不在优化期间尝试精度模式。
- 缺必需数据时停止并上报，不臆测；信息不足时输出"无法判断/需要补充数据"。

## 输入（由工作流提供）

| 类型 | 内容 |
| --- | --- |
| 必需 | 模型、配置、指标（avg/TP99/QPS）、约束、已实施项 |
| 模型图 | onnx 原图（只读）+ `dump_graph` 各阶段（parse/optimize/fusion/build），附采集时间与 sha256 |
| Profiling | `models/M<id>/.cache/profiling/PROF_*/mindstudio_profiler_output/` 下的 `step_trace_*.csv/json`、`op_statistic_*.csv`、`op_summary_*.csv`、`api_statistic_*.csv`、`task_time_*.csv`、`fusion_op_*.csv`、`msprof_*.json`；`fusion_result.json` 来自 `.cache/run`；附采集时间与 sha256 |
| 运行配置 | Pass `.so` 实际状态、config/env、die/threads/cores/dtype |
| 可选 | 上一轮 MD/JSON、人工确认的语义/业务信息、消融/并发/动态 shape 实验结果、Roofline 参数（FLOPs/访存量/峰值） |

- 工作流保证输入已采集并随附 sha256；本 skill 不重复采集，但须按 §1 门禁核验数据可用性与配置实际状态。
- 缺"必需"项时：输出缺项清单、停止分析。

## 口径

| 术语 | 定义 |
| --- | --- |
| `wall` | 稳态单次迭代墙钟，取 2/4–3/4 区间均值 |
| `kernel_sum` | 单次迭代所有 device kernel 的 `Total Time` 之和 |
| `non_kernel` | `wall - kernel_sum` |
| `gap` | 相邻 task 间隔 `start_i - end_{i-1}` |
| `wait` | 算子的 `Task Wait Time` |
| `effect/match` | 融合 Pass 的实际生效次数/匹配次数 |
| 关键路径 | 决定迭代墙钟的串行执行链；非关键路径的优化不产生端到端收益 |
| 收益上界 | 可消除的 kernel/launch/gap 时间之和，需扣除重叠与新增开销 |
| `cv` | 变异系数（标准差/均值），衡量同配置多次运行的稳定性 |
| 置信度 | 结论可信程度：高（稳定复现+图/effect+精度）/中（趋势一致但样本少或噪声大）/低（单次或仅推断） |

## 分析流程

### 1 数据质量门禁

先确认数据可用，再下结论：

- 排除首迭代（编译/冷启动）；首迭代超过后续平均值 5 倍时单独标注，不计入平均。
- 取稳态区间 2/4–3/4，不使用最后一个进度窗口。
- **统一统计口径**：每场景 3 次取中位数；允许剔除首样本与最大偏差样本，但必须保留原始单次结果与剔除记录，并在结论中说明；不得反复挑选数据。
- 同配置中段结果差异 >3% 时，先排查 host 抖动、共享 die、频率/温度、其他进程；未排除前不下结论。
- 识别 200ms 以上长尾或 TP999 异常点，检查是否为真实抖动或测量污染。
- 确认 Profiling 来自目标 die；memcpy/H2D 异常大时确认是否被采集放大。
- 确认 Pass `.so` 实际存在、config/env 一致；切换 Pass/core/dtype/shape 后清理 `kernel_meta` 与 `fusion_result`。
- 质量不达标：记录原因，停止出结论；先补采或复测。

### 2 总账分流

```text
kernel_sum = Σ(op_statistic.Total Time)
non_kernel = wall - kernel_sum
non_kernel_ratio = non_kernel / wall
```

| 现象 | 结论 | 下一步 |
| --- | --- | --- |
| `non_kernel_ratio < 20%` | 偏 kernel/compute-bound | 分析 Top kernel、Cube/MTE2 利用率、shape 与 layout |
| 20%–35% | 混合型 | 同时分析 Top kernel 与 host API |
| `> 35%` | host/launch/sync/空闲占比较高 | 优先分析 launch、sync、memcpy、gap，不先微调单 kernel |
| `kernel_sum` 低但 `wall` 高 | 依赖、空闲或同步 | 检查 task timeline、stream 数量、event wait |
| 单 kernel 小但 `gap` 大 | host/launch 或同步 | 评估融合、减少 kernel 数、多 stream |

> 阈值为经验参考，需结合模型与采集条件确认。

### 3 Host 分解

| 观察 | 异常信号 | 可能方向 |
| --- | --- | --- |
| `Node@launch`/`launch` | 次数多、单次 5–10us、总时长接近 gap | 减少 kernel 数、融合、多 stream |
| `aclrtSynchronizeStream*` | 次数多、单次 100us 以上、总时长超过 5% | 检查动态 tiling、同步点、多 stream |
| `aclrtMemcpy`/`MemcpyAsync` | 次数多、单次大、与迭代强相关 | 输入常驻、批处理、减少拷贝；注意采集放大 |
| `aclnn*GetWorkspaceSize` | 每个 op 一次、总时长高 | 图模式、算子融合、workspace 复用 |
| `ModelExecute` | 明显大于 `kernel_sum` | host/同步/调度占比高 |

- host API 可能与 device 执行重叠，不能直接把 host 总时长当作可优化收益；用时间线确认关键路径。

### 4 算子级分析

`op_statistic`：

| 观察 | 解释 |
| --- | --- |
| `Ratio > 15%` 且 `Avg` 大 | 可能为 compute/memory 主瓶颈 |
| `Count` 大、`Avg` 仅数微秒 | 更可能是 launch/调度/小算子问题 |
| `Max` 显著大于 `Avg` | 存在长尾、异常 shape 或同步 |
| `Core Type=AI_CPU` | dtype/placement 或无法下沉 |
| `Core Type=AI_CORE/MIX_AIC` | Cube 相关；结合 Cube/MAC 利用率 |
| `Core Type=AI_VECTOR_CORE` | Vector 相关；结合 scalar/MTE2 |

`op_summary`：

| 字段 | 读法 |
| --- | --- |
| `Task Duration` | 大：compute/memory 候选；小：不是主因 |
| `Task Wait Time` | 大且前一个 kernel 小：host/sync/依赖；大且前一个 kernel 大：正常流水 |
| `Block Num` | 远小于可用核：小 shape/并行不足；接近或超过：可能过订阅 |
| `Input/Output Shapes` | 小 shape 的 Cube 利用率低；动态 shape 可能触发 tiling/sync |
| `Input/Output Data Types` | 出现 f64/int64/AI_CPU：dtype 或 placement 问题 |
| `OP State` | 动态标记可能对应动态 tiling 或同步 |

> 利用率阈值为经验参考（如 MAC/Cube 低、scalar/AI_CPU 高），需结合 shape 与时间线确认。

### 5 gap 与 wait 聚合

```text
gap_before[op] = Σ(start(op_i) - end(prev_i))
gap_after[op]  = Σ(start(next_i) - end(op_i))
wait[op]       = Σ(Task Wait)
```

- 按总 gap 排序，再按 count 排序。
- 大 gap 且 kernel 小：host/launch 或同步；大 gap 且 kernel 大：compute 或正常流水。
- gap 集中于少数 op：依赖或同步点；分散于大量小 op：launch 或小算子数量问题。

### 6 图结构验证

Profiling 判断"是否值得做"（耗时是否显著），图结构判断"是否能做"（是否具备融合/合并/重排条件），**两者需同时满足**。

1. 通过 `--dump_graph --circle 1` 获取 Build、RunCustomPass_AfterInferShape 或 AfterOriginGraphOptimize 图。
2. 识别结构：同输入/同 shape 的 MatMul/BMM、Cast 链、连续小算子、重复子图、布局往返、多消费者中间张量。
3. 对每个候选检查：语义是否等价或可重排；中间张量是否单消费者；能否由已有标准算子表达；是否改变输出 dtype/接口/精度。
4. 用 `fusion_result` 或 Pass 日志确认 `effect/match`，不仅凭源码推断。

重复子图注意：n-gram 统计耗时会因 shape/Reshape/Concat 链放大，必须回 `op_summary` 核对真实 kernel 时间；"重复次数多"不等于"每个节点都耗时"。

### 7 收益上界与门禁

```text
candidate_kernel_saving = Σ(候选 op kernel time) × 命中比例
launch_saving          = 可删除 kernel 数 × 单次 launch 成本 × 关键路径占比
gap_saving             = 可消除 gap（需时间线证明不属于正常流水）
upper_bound            = candidate_kernel_saving + launch_saving + gap_saving
                         - epilogue_cost - precision_cost
```

- 按 `effect/match`、shape、dtype、消费者条件折算命中率，不假设 100%。
- 避免重复计数：kernel 时间与其 gap/launch 可能重叠，以时间线确认关键路径。
- 融合可能使单 kernel 变慢，需评估 epilogue、寄存器、UB、tiling。
- 单 Pass 收益低于 1% 且在噪声范围 → 不保留。
- 自定义算子周期与收益不匹配 → 不开发。
- 修改 onnx、重新切分、修改输出 dtype 属产品决策，需签核。

### 8 深化分析（按需）

当总账/算子/图结构定位不足以判断收益时，加载 `references/deep.md`：

| 触发 | 深化内容 |
| --- | --- |
| 需要判断优化天花板 | Roofline、compute/memory bound |
| 需要确认 op 是否在关键路径 | 消融、反事实 |
| 动态 shape/tiling 是候选 | shape 分布、tiling/缓存、典型/最坏口径 |
| 并发/吞吐是候选 | 并发扫描、Little 定律、饱和判断 |
| dtype/格式是候选 | 精度、dtype、ND/NZ、opImplMode |
| 内存/搬运是候选 | MTE2、workspace、拷贝、常驻 |
| 重复子图/成本归属不清 | 图规范化、单实例成本归属 |
| 用户要求跨模型/版本 | 结构比例、单位成本、effect 对比 |

- 子 skill 只补充证据与方法，不单独输出建议；结论回主 analysis 汇总。
- 加载后若瓶颈分类变化，主 analysis 重新归类并更新 ledger。

### 9 优先级、置信度与不确定性

```text
优先级 ≈ (收益上界 × 命中概率 × 关键路径占比) / (开发周期 × 风险 × 维护成本)
```

- 优先：大 gap/host 空泡、低风险、可验证、影响多个实例的结构问题。
- 次优：单算子 compute/format 优化、少量改图的融合。
- 最后：自定义算子、改 dtype/接口、重新切分、框架级改造。
- 不优先：噪声内微优化、非关键路径、无法验证的收益。
- 置信度：高 = 多次稳定复现 + 图/effect + 输出/精度验证；中 = 趋势一致但样本少或噪声大；低 = 单次/仅推断/缺图或精度证据，不得作为实施依据。
- 区分"实测/推断/假设"；缺失数据写"无法判断"并列出补充项；收益给区间，风险给触发条件；不用"肯定/必然"描述未验证效果。

### 10 结论

- 归入一类或混合：host / sync / dtype / compute / memory / fusion / idle。
- 每条建议必须同时具备：证据、图结构支持、收益上界、成本风险、验证方式、置信度。
- 不满足者写入暂缓或越界，不输出为优化项。
- 给出验证计划与回退条件；区分"实测值/推断值/未验证假设"。

可优化类别：

| 类别 | 证据 | 方向 |
| --- | --- | --- |
| Host launch-bound | 小 kernel 多、launch 次数多、gap 分散 | 减少 kernel 数、融合、图优化、多 stream |
| Sync/tiling-bound | sync 多、动态 shape、wait 集中 | 静态 shape、预编译 tiling、减少同步点 |
| AI_CPU dtype | `Core Type=AI_CPU`、f64/int64、Cast/Split 链 | 调整 dtype、前移转换、切分或自定义 |
| Compute-bound | 单 kernel 大、cube/mac 高 | kernel、算法、数据格式或库调优 |
| Memory-bound | MTE2 高、memcpy 多、layout 转换多 | 布局融合、减少中间张量、权重格式、常驻 |
| Small-op fusion | 同源链、相同 shape/输入、连续逐元素 | 融合、合并；需标准算子或自定义算子 |
| Idle/dependency | 无显著 host 开销但 gap 大 | 检查依赖、event、stream 与异步执行 |
| Runtime config | `matmul_ratio >20%` | HF32：`ENABLE_HF32=01`，见 §11 |

### 11 运行配置优化：HF32

```text
matmul_ratio = Σ(MatMul / BatchMatMul / BMM 及其变体的 kernel time) / kernel_sum
```

- `matmul_ratio > 20%`：正式优化项，`priority=P1`，**必须在限核 Step 1 前置评估**。
- `matmul_ratio ≤ 20%`：正式优化项，`priority=P2`，可选尝试；若 avg 收益 >5% 也保留。
- 收益判定：`avg_gain_pct = (avg_off - avg_on) / avg_off × 100%`；需 `>5%` 且 `>噪声带`。
- 精度判定：`--verify` PASS；建议记录 `max_abs_err`、`max_rel_err`、必要时 `cosine`。
- 精度失败、收益 ≤5% 或未超过噪声带 → 关闭 HF32，继续正常限核/并发。
- 无 golden 或精度不可接受 → 不开启/不保留 HF32。
- HF32 归类为 `runtime_config` 优化项，不是精度模式；`owner=model-recommend-core-limit-stress`，`needs_human_eval=true`。
- 由主 analysis 提出建议，`core-limit-stress` 在限核最前面执行 A/B；不进入常规优化循环。
- 默认只建议 `ENABLE_HF32=01`（仅 MatMul）；含 Conv 时可评估 `11`；`1/True` 为全部，需更强精度验证。
- 收益按 MatMul/BMM kernel 时间 × 命中率折算，不假设固定倍数。

正式建议示例：

```json
{
  "id": "OPT-HF32",
  "source": "profiling",
  "action_type": "runtime_config",
  "needs_human_eval": true,
  "priority": "P1",
  "owner": "model-recommend-core-limit-stress",
  "status": "suggested",
  "target": {"kind": "op", "ref": "MatMul/BMM", "pattern": ""},
  "change": {"action": "enable", "value": "ENABLE_HF32=01"},
  "evidence": [
    {"source": "op_statistic", "metric": "matmul_ratio", "value": null, "detail": "MatMul/BMM kernel time / kernel_sum"}
  ],
  "graph_support": {"matched": false, "detail": "运行配置优化，不依赖图结构"},
  "expected_gain": {
    "metric": "avg_latency",
    "min": null,
    "max": null,
    "unit": "%",
    "basis": "heuristic",
    "confidence": "medium",
    "upper_bound_note": "按 MatMul kernel 时间 × 命中率折算；仅当 avg 收益 >5% 且 >噪声带时保留"
  },
  "roi": {"cycle": "低", "risk": "中", "maintenance": "低", "verdict": "go", "note": "需精度验证"},
  "precision_risk": "needs-verification",
  "verification": {
    "method": "先 --verify，再做 HF32 off/on A/B；失败或收益不足则关闭",
    "command": "分别以 ENABLE_HF32=00 / 01 运行 --verify 与 A/B 压测"
  },
  "why": "MatMul/BMM 占 kernel 时间 >20%",
  "note": "主 analysis 提出；core-limit Step 1 执行"
}
```

## 输出与记录

固定两份文件，各保留一份、持续更新，不按轮次新建：

- `models/M<id>/analysis/model_analysis_report.md`：人读。
- `models/M<id>/analysis/optimization_suggestions.json`：机读。

输出前读取本 skill 的 `assets/report_contract.md`，按其中的 MD 结构、JSON 契约、枚举与复用规则生成；主文件不再重复整段模板。

关键约束：

- `schema_version` 当前为 `1.4`；`action_type` 含 `runtime_config`，`owner` 含 `model-recommend-core-limit-stress`。
- 复用旧文件：保留已有 `status` 与历史，只追加新建议、更新轮次/账本/终止结论。
- 同一问题不重复生成建议；已 `done/rejected` 的不再提出，除非证据变化并在 `evidence` 注明。

## 终止与越界

- **终止**：约束内已无证据支撑的优化建议 → `termination.reached=true` 并说明原因；工作流据此收口或进入 `model-recommend-core-limit-stress`。
- **暂缓**（写入 `deferred`，不得作为优化项输出）：
  - 收益处于测量噪声范围，多次中位数仍不稳定；
  - 优化目标不在关键路径，与 device 执行重叠；
  - 无图结构模式，只能改库/框架/产品接口；
  - 需要修改 ONNX、重新切分或修改输出 dtype，但无负责人或精度签核；
  - 自定义算子周期显著大于收益上界；
  - 依赖当前 CANN 版本的不稳定项（编译超时、segfault、NZ 长尾等），无法白名单规避；
  - 精度或语义无法证明，缺 golden 且差异无法解释。
- **越界**：超出当前约束或职责（改图/改模型、训练专属、端侧/LLM 专属、量化等）→ 写入 `out_of_scope`，仅人工参考。

## 自检 / 禁止

自检问题（任一项不满足即停止出结论）：

1. 是否有完整 Profiling（step_trace/op_summary/api_statistic）与图 dump？
2. 是否确认 Pass `.so`、config、model env、dtype 的实际状态？
3. 是否排除首迭代、离群点与 host 抖动，并记录统计口径？
4. 是否计算 wall/kernel_sum/non_kernel 与关键 gap？
5. 是否检查 Top op 的 Core Type、dtype、shape、wait？
6. 是否验证候选 pattern 的消费者/结构条件？
7. 是否确认 Pass `effect/match`，而不是只看源码？
8. 是否估计收益上界并扣除重叠/epilogue/精度成本？
9. 是否标注精度、接口、版本、越界风险与置信度？
10. 是否给出验证计划与回退条件？
11. 如建议 HF32，是否按 `matmul_ratio >20%` 判定，并说明 `--verify`、avg 收益 >5% 与关闭条件？
12. 是否需要加载 `references/deep.md`？深化结论是否已回传并更新瓶颈分类？
13. 输出前是否读取 `assets/report_contract.md`，并保持 `schema_version=1.4` 与枚举一致？

禁止：

- 数据质量未通过或共享设备抖动未排除时下结论。
- 无 profiling/图证据给建议；融合类缺任一证据。
- 收益上界重复计数；未完成 ROI 就推荐自定义算子。
- 修改 onnx、做 graph-rewrite、擅自建议自定义算子；编造参数/接口。
- 把最后一个进度窗口、首迭代、单次最优/最差值当作结论；混合不同 die/threads/cores/Pass 配置对比。
- 把 host 总时长直接当作可优化收益。
- 信息不足时不给"无法判断/需要补充数据"，直接给优化项。
- MD 冗长或 JSON 不可解析；两份文件按轮次新建。
