# `msprof` 采集与分析

> 使用各 CANN 自带的 `msprof`：多组 `--aic-metrics` + sample-based `aicore.db`，得到可分析的 `op_summary_*`、`per_core_cycles.csv` 与 `summary.txt`。

---

## 适用场景

- 目标环境以 `msprof` 为采集手段，或团队约定采用本工具链。

---

## Step 1：构建算子（如果有指定的调用方式，这一步可跳过）

**直调算子**：

```bash
cd ops/{operator_name} && mkdir -p build && cd build && cmake .. && make -j
```

**aclnn 算子**：

```bash
bash build.sh --pkg --soc=ascend910b --ops={operator_name} --vendor_name=custom -j16
./build_out/*.run --install-path=$CANN
bash build.sh --run_example {operator_name} eager cust --vendor_name=custom
```

---

## Step 2：采集

原生 `msprof` 每次运行只支持一个 `--aic-metrics` 组，且 `op_summary_*.csv` 是 per-op 聚合值而非逐核。流程：

1. 在正式采集前 warm-up N 次（规避 DVFS）
2. 按顺序分别采集 7 组 `--aic-metrics`：`PipeUtilization`、`ArithmeticUtilization`、`Memory`、`MemoryL0`、`MemoryUB`、`L2Cache`、`ResourceConflictRatio`
3. 额外跑一次 `--aic-mode=sample-based`，从 `device_0/sqlite/aicore.db`（`AICoreOriginalData.task_cyc`）拿到**逐核 cycle**
4. 以 aicore_time / max_cycles 反推主频，把逐核 cycle 折算成 per-core 时长

一键脚本：

```bash
# 位于 {skill_path}/scripts/msprof_profile_run.sh
bash {skill_path}/scripts/msprof_profile_run.sh \
     --warm-up=3 \
     --output=./msprof_output \
     -- ./demo arg1 arg2 ...
```

**关键点**：msprof 的 `op_summary.csv` **没有** `Current Freq/Rated Freq` 和逐核 `time(us)` 字段；逐核分析必须依赖 `PROF_Sample` 下的 `aicore.db`。采集落盘目录树见文末 **数据目录结构 → 临时输出**（根路径为 `--output` 下的 `PROF_GROUP_*`）。

---

## Step 3：归档 + 统计摘要

```bash
GROUP_DIR=$(ls -td <output_dir>/PROF_GROUP_* | head -1)

python3 {skill_path}/scripts/msprof_perf_summary.py $GROUP_DIR ops/{operator_name}
```

脚本会：

1. 在 `ops/{operator_name}/docs/perf/round_NNN/` 创建归档目录（轮次自动递增）
2. 按目标 `Op Name` 在 7 份 `op_summary.csv` 里合并列，复制为 `op_summary_<Metric>.csv`
3. 读 `PROF_Sample/.../aicore.db` 的 `task_cyc`，根据 `aicore_time(us)` 反推主频，换算每核耗时
4. 输出 `summary.txt`，在全局统计之外附加「逐核负载均衡」段，例如：

   ```
   --- 逐核负载均衡 (sample-based aicore.db) ---
     有效核数: 32  | 主频推算: 1.651 GHz (0.6058 ns/cycle)
     min=185.93us  avg=193.13us  max=199.27us
     (max-min)/max = 6.69%  ->  达标 (<10%)
     Top-3 慢核: Core3=199.27us, ...
     Top-3 快核: Core21=185.93us, ...
     [提示] 前半段 core 均值 198.17us vs 后半段 188.10us，差距 5.99%
            疑似两簇 (NUMA / L2 slice) 负载偏斜，建议尝试 block swat / 尾轮均衡策略。
   ```

5. 同步把 `per_core_cycles.csv` 归档到同目录，方便二次分析

归档完成后即结束本 Step 的职责。下游从 `summary.txt`（含逐核负载均衡段）、`op_summary_*.csv`、`per_core_cycles.csv` 读取指标后，按 **下文「主 Bound 判定」** 推导**主 bound 档位**；算子族级细化与报告专有条目见 **`/ascendc-performance-optimization`**。

---

## 主 Bound 判定（msprof 归档）

本节适用于 **`msprof` 经 Step 2～3 得到的归档目录**（`round_NNN/`）。判据与 `/ops-simulator` 流水图路径**同一套优先级表**，便于对接 `/ascendc-performance-optimization`。

### 输入与输出

| 项目 | 说明 |
|------|------|
| **输入** | 从归档中的 `op_summary_*.csv`、`summary.txt` 等抽取的**单核侧**各流水线 **busy 占 case（或 task）总时长** 的百分比；口径须与 `/ops-simulator` 流水分析中的「占 case」可比 |
| **不适用** | msprof 聚合指标**无**可对照流水图的气泡时间；报告中气泡列标注「不适用」，**不得**填写气泡数值 |
| **输出** | **主 bound 档位**：`MTE2 BOUND` / `CUBE BOUND` / `VEC BOUND` / `FIXP BOUND` / `MTE3 BOUND` / `SCALAR BOUND` / **无 bound** |

各流水线 busy 占比的**抽取与列映射**以本文 Step 3 产物及 [`csv_fields_reference.md`](csv_fields_reference.md) 为准；若归档 CSV 表头与文档示例不一致，**以实际表头为准**。

### 判定规则（与 `/ops-simulator` 一致）

在已得到各流水线 busy 占比后，**从上到下**匹配**第一条成立**：

| 优先级 | Bound 类型 | 判断规则 |
|--------|-----------|----------|
| 1 | MTE2 BOUND | MTE2 busy 占 case > 80%，或（MTE2 在 8 条中占比最大且 > 70%） |
| 2 | CUBE BOUND | CUBE busy 占 case > 80%，或（CUBE 在 8 条中占比最大且 > 70%） |
| 3 | VEC BOUND | PUSHQ busy 占 case > 80% |
| 4 | FIXP BOUND | FIXP busy 占 case > 80% |
| 5 | MTE3 BOUND | MTE3 busy 占 case > 80% |
| 6 | SCALAR BOUND | SCALAR busy 占 case > 80%，或 SCALARLDST busy 占 case > 80% |
| — | **无 bound** | 以上条件均不满足 |

「占比最大」比较对象为上述单元对应的统计。

---

## 数据目录结构

### 临时输出（`msprof_profile_run.sh` 的 `--output` 目录下）

```
<output_dir>/PROF_GROUP_<YYYYMMDD_HHMMSS>/
├── PROF_PipeUtilization/PROF_*/mindstudio_profiler_output/op_summary_*.csv
├── PROF_ArithmeticUtilization/...
├── PROF_Memory/...
├── PROF_MemoryL0/...
├── PROF_MemoryUB/...
├── PROF_L2Cache/...
├── PROF_ResourceConflictRatio/...
└── PROF_Sample/PROF_*/device_0/sqlite/aicore.db   ← 逐核 task_cyc
```

### 持久归档

```
ops/{算子名}/docs/perf/
├── round_001/
│   ├── op_summary_PipeUtilization.csv
│   ├── op_summary_ArithmeticUtilization.csv
│   ├── op_summary_Memory.csv
│   ├── op_summary_MemoryL0.csv
│   ├── op_summary_MemoryUB.csv
│   ├── op_summary_L2Cache.csv
│   ├── op_summary_ResourceConflictRatio.csv
│   ├── op_statistic_<Metric>.csv
│   ├── task_time_<Metric>.csv
│   ├── per_core_cycles.csv
│   └── summary.txt
└── ...
```

字段含义详见 [`csv_fields_reference.md`](csv_fields_reference.md)（按 `op_summary_*` 列名对齐同类指标）。

---

## 注意事项

1. **必须 warm-up**：脚本默认 `--warm-up=3` 可调，避免 DVFS 影响首次运行
2. **无频率字段**：`op_summary` 没有 `Current Freq/Rated Freq`；脚本通过 `aicore_time / max_cycles` 反推主频
3. **MTE2/MTE3 带宽共享**：同时读写 GM 时总带宽共享，评估搬运段负载时宜按 MTE2、MTE3 合并字节量与平台带宽对照
4. **小数据量场景**：数据量很小时头开销占比会很高，这不一定是算子问题
5. **多核同地址访问**：多核同时读同一 512B 地址范围会被串行化，导致 MTE2 耗时异常

---

## 相关资源

| 文件 | 内容 |
|------|------|
| [`csv_fields_reference.md`](csv_fields_reference.md) | 字段定义和阈值（按 `op_summary_*` 列名对齐），供下游分析角色参考 |
| `scripts/msprof_profile_run.sh` | 一键采集脚本（Step 2 调用） |
| `scripts/msprof_perf_summary.py` | 归档 + 摘要 + 逐核负载均衡（Step 3 调用） |
