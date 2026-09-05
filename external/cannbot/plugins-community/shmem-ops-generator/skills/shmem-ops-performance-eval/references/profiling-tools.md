# SHMEM 性能采集工具与方法

本文说明 SHMEM 算子性能采集工具、内置打点口径、msprof 解析口径和辅助 profiling 方法。

## 1. 采集工具

### 1.1 msprof（新增采集口径）

MindStudio Profiler 是 Phase 6 新增的官方采集口径。必须采集 kernel 级 `task_time`，并与 SHMEM 原有 `[PERF]` 指标并列报告：

```bash
msprof --output="${OUT}" \
  --task-time=l2 --runtime-api=on --type=text \
  bash "scripts/perf.sh" "${DEVICE_LIST}" "${BASE_COUNT}" "${DTYPE}" "${ITERS}"
```

baseline 使用 HCCL/aclnn 时加 `--hccl=on`：

```bash
msprof --output="${OUT}" \
  --task-time=l2 --runtime-api=on --hccl=on --type=text \
  bash "baseline/scripts/run_baseline.sh" "${DEVICE_LIST}" "${BASE_COUNT}" "${DTYPE}" "${ITERS}"
```

产出位置：

- `PROF_*/device_*/summary/task_time_*.csv` 或 `PROF_*/mindstudio_profiler_output/task_time_*.csv`
- 主要字段：`kernel_name`、`kernel_type`、`task_time(us)`、`task_start(us)`、`task_stop(us)`、`Device_id`、`Stream_id`、`Task_id`

### 1.2 msprof op（可选诊断）

`msprof op` 用于单 kernel 指标、roofline、memory/pipeline/source 等深度诊断。它不替代 Phase 6 的端到端对比；标准 msprof 口径使用 `msprof --task-time=l2`。

### 1.3 inline event / `[PERF]`（保留原有口径）

SHMEM 算子生成的 `--perf` 模式必须保留原有前后打点和 `[PERF]` 输出。`e2e_us` 只来自该口径；`kernel_us` 后续会与 msprof 口径并列：

```text
[PERF] source=inline_event e2e_us=<val> kernel_us=<val> algo_bandwidth_GBps=<val> e2e_bus_bandwidth_GBps=<val> kernel_bus_bandwidth_GBps=<val> bandwidth_utilization_pct=<val> payload_bytes=<val> bus_factor=<val> peak_bandwidth_GBps=<val>
```

### 1.4 SHMEM Cycle Profiling

`SHMEMI_PROF_START/END` + `aclshmemx_get_prof(nullptr, true)` 用于 kernel 内部 phase 定位。它可以解释瓶颈，并与 `[PERF]` / msprof 结果共同写入报告。

### 1.5 mssanitizer

内存访问检查工具，不直接用于性能采集，但可用于排除越界导致的性能异常。

```bash
mssanitizer --log-level=error ${EXEC_BIN} ${ARGS}
```

## 2. msprof task_time 解析口径

解析步骤必须可复现，并在报告中记录 profile 目录、命令、warmup、measure、过滤规则。

1. 找到所有 `task_time_*.csv`。
2. 按 `kernel_name` / `kernel_type` 过滤目标算子；丢掉 `memset` / `memcpy` 等噪声。
3. 每个 PE 按调用顺序跳过前 `WARMUP` 轮，只保留正式测量轮。
4. **先算每张卡自己的稳态时延**：对该卡上保留的目标 kernel 的 `task_time` 取平均（或 design 指定用中位数）。通算融合算子则用该卡上多轮次相关kernel时延之和取平均。
5. **`kernel_us_msprof` = 各卡稳态时延里最大的那一个**（最慢卡）。多卡要等最慢的做完，**禁止**把各卡时延简单平均后当最终结果。
6. 诊断可另报（不能替代主字段）：每张卡各自的时延、最慢卡是哪一张；可选看同一轮里「所有卡最早开始到最晚结束」的时间包络。
7. 跨测量轮可报告平均 / 中位数 / 最小 / 最大；写入 `kernel_us_msprof`、做达标对比时，默认是「每卡先平均，再取最慢卡」。

主时延列：

| 场景 | 主时延字段 | 怎么算 |
| --- | --- | --- |
| 多 PE（默认） | `kernel_us_msprof` | 各卡稳态时延取最大值（最慢卡） |
| 单 PE | `kernel_us_msprof` | 就是这一张卡的稳态时延 |
| 通算融合 | `kernel_us_msprof` | 每卡算融合跨度（或 compute+comm），再取最慢卡；另附 compute/comm 分段 |
| baseline | 与 SHMEM 相同 | 同样取最慢卡 |

## 3. 性能指标计算

指标定义和公式见 [timing-and-metrics-standard.md](timing-and-metrics-standard.md) §4。注意：

- SHMEM 必须输出 inline_event 完整指标：`e2e_us`、`kernel_us`、带宽、payload/bus_factor/peak_bandwidth 等。
- SHMEM msprof 解析只补充 kernel 口径：`kernel_us_msprof`、`kernel_bus_bandwidth_GBps_msprof`、`bandwidth_utilization_pct_msprof`。
- baseline 只要求 msprof kernel 口径。
- 与 baseline 的主对比使用 SHMEM msprof kernel 口径 vs baseline msprof；inline_event 指标并列保留，不删除、不改名。
- `logical_payload_bytes` 和 `bus_factor` 见 timing-and-metrics-standard.md §4.1/§4.3。
- `peak_bandwidth_GBps` 和 `bandwidth_utilization_percent` 见 timing-and-metrics-standard.md §4.4。

## 4. 采集规范

### 4.1 Warmup

- runner 必须支持 `WARMUP` 环境变量或等价参数。
- msprof 解析必须跳过 warmup kernel，不得把 warmup 和 measure 混入统计。
- 推荐 `WARMUP=10`、`MEASURE=40`；资源紧张或长耗时算子可降到 `WARMUP=3`、`MEASURE=10`，必须在报告标注。

### 4.2 多卡统计

多 PE 时整次算子要等**最慢的那张卡**做完。因此：

- **`kernel_us_msprof` MUST = 最慢卡的 task 执行时间** \(\max_r T_r\)（§2），**禁止**对各 PE `task_time` 简单平均。
- 可额外报告 per-rank \(T_r\)、slowest-rank 对应的 rank id，用于定位**长尾**（多数卡快、个别卡明显更慢，拖高整次结束时间）。
- collective envelope（跨卡 start/stop 包络）仅作可选诊断，**不替代** `kernel_us_msprof` 的最慢卡口径。

### 4.3 结果记录格式

```yaml
case_id: L_fp16_8pe
source: inline_event
log: data/perf/shmem_inline_L_fp16_20260716_210000.log
e2e_us: 847.03
kernel_us: 545.97
kernel_bus_bandwidth_GBps: 92.10
---
case_id: L_fp16_8pe
source: msprof_task_time
profile_dir: data/perf/msprof_shmem_L_fp16_20260716_210000
latency_source: task_time slowest_rank mean
warmup: 10
measure: 40
kernel_us_msprof: 562.64
compute_us: 254.82
comm_us: 562.64
logical_payload_bytes: 67108864
bus_factor: 0.75
kernel_bus_bandwidth_GBps_msprof: 58.94
---
case_id: L_fp16_8pe
source: baseline_msprof_task_time
baseline_profile_dir: data/perf/msprof_baseline_L_fp16_20260716_205500
e2e_us: 534.50
kernel_us: 534.50
ratio: 62.6%
```
