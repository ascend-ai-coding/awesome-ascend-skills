# 性能采集与对比（inline 打点 + msprof 命令参考）

> **采集口径**：SHMEM MUST 保留原有 `[PERF]` 前后打点指标，`e2e_us` 仍然只来自该口径；同时额外采集 `msprof --task-time=l2` 的 `task_time_*.csv`，用于补充 `kernel_us_msprof`。baseline 只要求 msprof 口径。
> **Hard Gate**：baseline 与 SHMEM MUST 分阶段、分 profile 目录采集；NEVER 同一条命令混跑。对比时用离线解析脚本读取 SHMEM `[PERF]` 日志、SHMEM msprof 目录和 baseline msprof 目录。

环境见 [custom-ops-entrypoints.md §0](../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md)。

---

## 1. 分阶段采集（推荐）

### 阶段 A — baseline msprof

```bash
OP=<op_name>
DEVICE_LIST=0,1,2,3,4,5,6,7
BASE_COUNT=8388608
DTYPE=float16
WARMUP=10
MEASURE=40
ITERS=${MEASURE}
TAG="$(date +%Y%m%d_%H%M%S)"
OUT="${SHMEM_REPO}/custom-ops/${OP}/data/perf/msprof_baseline_${BASE_COUNT}_${DTYPE}_${TAG}"

cd "${SHMEM_REPO}/custom-ops/${OP}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source "${SHMEM_REPO}/install/set_env.sh"

WARMUP="${WARMUP}" \
msprof --output="${OUT}" \
  --task-time=l2 --runtime-api=on --hccl=on --type=text \
  bash "baseline/scripts/run_baseline.sh" \
    "${DEVICE_LIST}" "${BASE_COUNT}" "${DTYPE}" "${ITERS}"
```

baseline runner 必须把 warmup 和 measure 分开：`WARMUP` 轮先运行并同步，随后只运行 `ITERS` 轮正式测量。msprof 目录中仍会包含 warmup kernel；解析时 MUST 丢弃前 `WARMUP` 轮。

### 阶段 B — SHMEM msprof

间隔 >=30s，**新开 shell**。该阶段同时保存 `[PERF] source=inline_event` 日志和 msprof profile：

```bash
OP=<op_name>
DEVICE_LIST=0,1,2,3,4,5,6,7
BASE_COUNT=8388608
DTYPE=float16
WARMUP=10
MEASURE=40
ITERS=${MEASURE}
TAG="$(date +%Y%m%d_%H%M%S)"
OUT="${SHMEM_REPO}/custom-ops/${OP}/data/perf/msprof_shmem_${BASE_COUNT}_${DTYPE}_${TAG}"
LOG="${SHMEM_REPO}/custom-ops/${OP}/data/perf/shmem_inline_${BASE_COUNT}_${DTYPE}_${TAG}.log"

cd "${SHMEM_REPO}/custom-ops/${OP}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source "${SHMEM_REPO}/install/set_env.sh"

WARMUP="${WARMUP}" \
msprof --output="${OUT}" \
  --task-time=l2 --runtime-api=on --type=text \
  bash "scripts/perf.sh" \
    "${DEVICE_LIST}" "${BASE_COUNT}" "${DTYPE}" "${ITERS}" \
  2>&1 | tee "${LOG}"
```

HCCL/aclnn baseline 采集保留 `--hccl=on`；纯 SHMEM 采集默认不加 `--hccl=on`，除非算子实现本身会调用 HCCL/aclnn。

### 阶段 C — 离线解析与对比

```bash
OP=<op_name>
BASE_COUNT=8388608
DTYPE=float16

BASE_PROF="${SHMEM_REPO}/custom-ops/${OP}/data/perf/msprof_baseline_${BASE_COUNT}_${DTYPE}_<tag>"
SHMEM_PROF="${SHMEM_REPO}/custom-ops/${OP}/data/perf/msprof_shmem_${BASE_COUNT}_${DTYPE}_<tag>"
SHMEM_LOG="${SHMEM_REPO}/custom-ops/${OP}/data/perf/shmem_inline_${BASE_COUNT}_${DTYPE}_<tag>.log"

python3 "${SHMEM_REPO}/custom-ops/scripts/lib/parse_msprof_perf.py" \
  --baseline-prof "${BASE_PROF}" \
  --shmem-prof "${SHMEM_PROF}" \
  --shmem-inline-log "${SHMEM_LOG}" \
  --warmup "${WARMUP}" \
  --measure "${MEASURE}" \
  --op-kind "<transport|collective|fused_compute_comm>" \
  --payload-bytes "<logical_payload_bytes>" \
  --bus-factor "<bus_factor>" \
  --peak-bandwidth-gbps "<peak_bandwidth_GBps>"
```

若项目尚未有通用 `parse_msprof_perf.py`，本阶段必须生成算子本地解析脚本（例如 `scripts/parse_msprof_<op>.py`），逻辑必须同时解析 SHMEM `[PERF]` 日志和 msprof profile，并遵循 [profiling-tools.md](profiling-tools.md) §2。

> **S 档采集**：Round 0 和最终轮需额外采集 S 档。将 `BASE_COUNT` 替换为 S 档规模（按 [testcase-scale-standard.md](../../shmem-ops-testcase-gen/references/testcase-scale-standard.md)），其他流程与 L 档相同。中间优化轮次仅用 L 档。

---

## 2. 产物布局

```text
custom-ops/
├── scripts/lib/parse_msprof_perf.py        # 可选通用解析器
└── <op_name>/
    ├── baseline/scripts/run_baseline.sh    # baseline runner，供 msprof 包裹
    ├── scripts/perf.sh                     # SHMEM runner，供 msprof 包裹
    ├── scripts/parse_msprof_<op>.py        # 无通用解析器时生成
    └── data/perf/
        ├── shmem_inline_<case>_<tag>.log
        ├── msprof_baseline_<case>_<tag>/
        └── msprof_shmem_<case>_<tag>/
```

---

## 3. 达标口径

- SHMEM 结果必须包含：`e2e_us`（来自 `[PERF] source=inline_event`）、`kernel_us_inline`（来自 `[PERF]`）和 `kernel_us_msprof`（来自 `task_time_*.csv`）。
- baseline 结果只要求 msprof `kernel_us` / `kernel_bus_bandwidth_GBps`。
- 达标主对比：SHMEM `kernel_us_msprof` / `kernel_bus_bandwidth_GBps_msprof` vs baseline msprof；SHMEM inline_event 并列保留，用于历史口径和差异分析。

达标线与判定规则见 [timing-and-metrics-standard.md](timing-and-metrics-standard.md)。

---

## 4. Docker

每阶段独立 `docker exec`（见 [docker-exec-contract.md](../../shmem-ops-dev/references/docker-exec-contract.md)）。msprof 输出目录必须挂载到容器外可读取路径。
