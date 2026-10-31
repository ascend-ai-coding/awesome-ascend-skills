# `msprof` 采集与分析

> `msprof` 路径分为两类：标准可比较计时使用单次 application、每例 10 组相邻 `warmup → target` 并取 target 中位数；多指标与 sample-based 采集只用于补充瓶颈诊断。

> **CANN 版本注意事项（实测 9.0.0 build 20260422 vs 9.1.0 build 20260610，Ascend950PR）**：
> ① `--aic-mode=task-based` 在 9.0.0 是 `op_summary` 导出的必要参数，9.1.0 可省（带参无害）——`collect_perf_application.py` 固定传参；
> ② 单会话 relevant op 记录数过大时 9.0.0 会**静默导出空 `op_summary` 且 msprof exit 0**（实测 540 条稳定、720 条抖动、840 条全败）——采集器默认按 540 上限自动拆会话并对空导出自动重试一次；
> ③ 不完整安装（ops 包缺失，如裁剪版 9.1.0 仅剩 `libopapi_oam.so`）没有系统 aclnn warmup 算子，framework_ops warmup 不可用——采集器预检 `cann_environment` 并在双回退失败时给出明确原因；
> ④ 双 CANN 共存机器上 shell 实际解析的 toolkit 可能与预期不同（同算子 9.1.0 实测快 1.3~1.5x，跨版本数据不可直接对比），采集前确认 `$ASCEND_HOME_PATH`。
> 详见 [`application_runner_contract.md`](application_runner_contract.md) 的"CANN 版本差异与不完整安装"一节。

---

## 适用场景与分流

先按 `SKILL.md` 的统一路由确定工具。本文件仅适用于外部选择或按「工具选择」规则选中 `msprof` 的场景。使用 `msprof` 采集时，首先判断算子是否为 MC² 多 rank 算子：

| 判定条件 | 采集流程 |
|---------|---------|
| 算子通过 `fork()` 创建多个子进程，各绑定不同 NPU 卡，子进程间通过 SHMEM UDMA / BarrierAll / CrossCoreFlag 协同通信（如 alltoall_matmul、allgather_matmul、matmul_reducescatter） | → **[MC² 多 rank 采集](#mc-多-rank-算子采集)**（本文末尾章节），**禁止 `msprof op`** |
| 需要 baseline/candidate 可比较的 `total_us` | → [Step 2A：标准可比较计时](#step-2a标准可比较计时) |
| 已有标准计时，需要进一步定位流水、内存或逐核问题 | → [Step 2B：补充深度诊断](#step-2b补充深度诊断) |

> MC² 多 rank 算子的采集命令、数据结构、稳态取值方法与单卡有本质区别（跨 rank 同步、木桶效应），`msprof op` 不适用于此类算子。如果不确定算子是否为 MC²，先检查源码是否调用 `fork()` 和 `aclshmemx_*` API。

arch22（A2/A3）的详细分析见 [CANN 8.0.0 alpha001 官方文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/devaids/opdev/optool/atlasopdev_16_0097.html)。先按目标工具版本确认可采指标，再分析流水利用率、搬运、冲突及负载；不要求 arch35 专有的 RegBase/VF/RVEC 字段。补充诊断不改变标准计时协议。

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

## Step 2A：标准可比较计时

标准计时由 `collect_perf_application.py` 执行，一次 `msprof application` 覆盖完整 case 集：

```text
case 1: (warmup → synchronize → target → synchronize) × 10
case 2: (warmup → synchronize → target → synchronize) × 10
...
每组只汇总 target 的完整 kernel sequence
每 case 对 10 个 target 总耗时取 median → total_us
```

固定门禁：

- 采样数必须是 10，不允许改成其他次数；
- 每个 target 前紧邻一次 warmup，warmup 时间不计入 `total_us`；
- target 发射多个 kernel 时先求和，再在 10 个和值上取中位数；
- execution manifest 声明精确 kernel sequence，相关 launch 多发、少发、乱序均失败；
- 一次 application 覆盖完整 case 集，runner 只初始化一次；
- baseline/candidate 的源码、二进制、case、设备、CANN、频率、runner、warmup、指标与协议身份必须一致。

默认先探测并使用现成 Max/ArgMax；只有 ops 环境不可用时才构建工程内 warmup。先落地其源码作为后备，此步骤不默认编包：

```bash
python3 {skill_path}/scripts/install_compiled_warmup.py \
  --project-dir /path/to/operator-project
```

再按 [`application_runner_contract.md`](application_runner_contract.md) 实现 runner 并采集：

```bash
python3 {skill_path}/scripts/collect_perf_application.py \
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

`collection_manifest.json.status=complete`、launch 映射相等且 `performance.csv.sample_count=10` 才能作为可比较数据。`spread_over_median > 10%` 说明环境或 conditioning 不稳定：先排查并重采；若 `msprof application` 路径持续无法通过门禁且外部未强制指定工具，记录原因后回落到 `msprof op`，不得混用两条路径的结果。

## Step 2B：补充深度诊断

原生 `msprof` 每次运行只支持一个 `--aic-metrics` 组，且 `op_summary_*.csv` 是 per-op 聚合值而非逐核。以下流程用于诊断，不作为标准 `total_us`：

1. 在诊断采集前 warm-up N 次（规避 DVFS）
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

`--compare`、`--quick`、`--batch` 是上游兼容入口；需要验收或优化收益比较时，仍以 Step 2A 产物为准。`--quick` / `--compare` 均支持 `--keep-prof` 透传：加上后保留 msprof 原始 PROF 目录（默认在采集后清理），供后续深度分析或可视化工具导入；不需要手工备份 PROF 目录，也不需要切换标准模式重新采集。

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

## 核间同步与负载偏斜的证据

当源码含循环内跨核同步、生产者—消费者依赖或多核写入同一目标时，结合以下证据提出可检验的瓶颈假设：

- 从源码和 tiling 数清每个 case 的循环次数、每轮同步点及参与核，画出写入、发布、等待、消费的依赖顺序；同时记录负载随输入索引分布变化的情况。
- 从 `summary.txt` 与 `per_core_cycles.csv` 看有效核数、最快/最慢核和负载分布；有可用时间线时对照等待区间。慢核决定尾部完成时间，快核空闲既可能来自任务不均，也可能来自同步等待。
- 聚合的流水 busy 比例和逐核总 cycle **不能**单独给出一次屏障、一次原子操作或某阶段的毫秒收益。没有阶段级证据时，只给出次数与可能的关键路径，不把推算当实测。
- 对热点目标集中和分散的代表性输入分别记录数据分布与 case 身份。后续只用语义有效、身份和计时协议一致的隔离候选 A/B 测量收益；深度指标仍只用于解释，不替代标准 `total_us`。

此节只提供诊断证据；具体改变同步粒度或归约路径时，还须核对数值顺序、可见性、workspace 和目标平台能力。

---

## 主 Bound 判定（msprof 归档）

本节适用于 **`msprof` 经 Step 2～3 得到的归档目录**（`round_NNN/`）。判据与 `/ops-simulator` 流水图路径**同一套优先级表**，便于对接 `/ascendc-performance-optimization`。

> MC² 多 rank 算子同样使用本节判定规则，但需额外注意 MTE2 污染问题（见 [MC² 注意事项](#mc-注意事项)）。

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

## MC² 多 rank 算子采集

MC² 通算融合算子通过 fork 多个子进程实现多 rank 协同，每个 rank 绑定一张 NPU 卡。这类算子的性能采集与单卡算子**有本质区别**——通信开销需要跨卡交互才能真实体现，多 rank 间的 BarrierAll/CrossCoreFlag 同步使整体性能受制于最慢的 rank（木桶效应）。

> **`msprof op`（msopprof）不适用于 MC² 多 rank 算子**——它设计用于单进程单设备，对 fork 程序的行为未定义，采集到的数据不可靠。

### 采集命令

```bash
# MC² 多 rank 算子专用采集命令
msprof --output={prof_dir} \
       --ai-core=on \
       --aic-mode=task-based \
       --aic-metrics=PipeUtilization \
       --task-time=on \
       --ascendcl=on \
       {code_dir}/build/{exe} {args...}
```

**关键参数说明**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `--aic-mode=task-based` | 固定 | task-based 模式确保每个 kernel launch 产生独立记录，而非聚合 |
| `--aic-metrics=PipeUtilization` | 推荐 | 采集各流水线占比，用于瓶颈分析 |
| `--task-time=on` | 固定 | 记录每次 kernel 的 Task Duration |
| `--ascendcl=on` | 固定 | 采集 AscendCL 层信息 |

### 多设备数据结构

msprof 包裹 fork 程序时，会自动为每个活跃设备分别产出 profiling 数据：

```
{prof_dir}/
├── PROF_{timestamp}_*/           # device 0
│   └── mindstudio_profiler_output/
│       └── op_summary_*.csv      # ← Task Duration(us) 在这里
├── PROF_{timestamp}_*/           # device 1
│   └── mindstudio_profiler_output/
│       └── op_summary_*.csv
├── PROF_{timestamp}_*/           # device 2
│   └── ...
└── PROF_{timestamp}_*/           # device 3
    └── ...
```

每个 `op_summary_*.csv` 中每行对应一次 kernel launch。如果程序内部循环 10 次，则每个设备有 10 行。

### 稳态取值

MC² 算子 perf 模式下循环执行至少 10 次算子调用：

1. **前 5 次为 warm-up**，丢弃
2. **取后 5 次 Task Duration 的均值**作为该 rank 的稳态性能

### 跨 rank 取最大值

多 rank 协同算子存在**木桶效应**——所有 rank 通过 BarrierAll / CrossCoreFlag 同步，整体性能由**最慢的 rank** 决定.

> 此方法适用于 MC² 算子的**所有**性能测试场景：基线 profiling、隔离测试、方案对比测试。

### MC² 注意事项

1. **必须使用 `msprof`、禁止 `msprof op`**：`msprof op` 对 fork 程序的采集行为未定义，数据不可靠
2. **必须 warm-up**：前 5 次受 DVFS 和 L2 cache 预热影响，不可作为性能指标
3. **必须取跨 rank max**：木桶效应决定整体性能由最慢 rank 决定，取平均会高估性能
4. **隔离测试取 max 的正确性**：注释通信后某些 rank 的 Task Duration 可能极短，这是因为该 rank 无需等待通信同步。取 max 代表包含了完整同步等待的真实通信时间

---

## 相关资源

| 文件 | 内容 |
|------|------|
| [`csv_fields_reference.md`](csv_fields_reference.md) | 字段定义和阈值（按 `op_summary_*` 列名对齐），供下游分析角色参考 |
| [`application_runner_contract.md`](application_runner_contract.md) | 标准 application runner、execution manifest 与 launch 映射契约 |
| [`compiled_warmup.md`](compiled_warmup.md) | 工程内 warmup 的安装、构建与调用 |
| `scripts/collect_perf_application.py` | 10 组相邻预热、target 中位数和可比较门禁 |
| `scripts/msprof_profile_run.sh` | 补充深度指标采集（Step 2B 调用） |
| `scripts/msprof_perf_summary.py` | 深度诊断归档 + 摘要 + 逐核负载均衡（Step 3 调用） |
