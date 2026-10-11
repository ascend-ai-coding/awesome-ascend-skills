# SHMEM 算子时间打点与指标计算标准

## 1. SHMEM 与 HCCL 的执行模型

### 1.1 HCCL 执行模型

```
用户数据在 Device GM
        │
        ▼
 ┌─────────────────┐
 │  HcclCollective  │   ← 用户调用点
 │  (内部完成全部   │
 │   staging/路由/  │   ← 所有内部开销对用户不可见
 │   通信/规约)     │
 └────────┬────────┘
          │
          ▼
   结果在 Device GM
```

HCCL 是黑盒 API。用户只提供 input/output device buffer，HCCL 内部完成所有 staging、路由和通信。最终计时必须通过 msprof 的 kernel/task timeline 获取：

```
H2D copy (不计时)  →  warmup  →  HcclCollective × measure  →  msprof task_time
```

### 1.2 SHMEM 执行模型

远端 PE 能访问的只有本 PE 的对称内存（`aclshmem_malloc` 分配）。用户的 `input` 在普通 Device GM（`aclrtMalloc`），需要搬到对称内存后才能被远端 PE 读取。

**搬运方式取决于通信引擎对地址类型的支持**：

| 引擎 | put_nbi src 能否是用户 GM | get_nbi dst 能否是用户 GM | 放置方式 |
| --- | --- | --- | --- |
| MTE | 能 | 能 | **做法 A**：kernel 内直接用 `put_nbi(symm, user_gm, ...)` 搬运 |
| SDMA | 能 | 能 | 同上 |
| UDMA | 能 | 能 | 同上 |
| RDMA | 不能 | 不能 | **做法 B**：Host 侧 `aclrtMemcpy(user_gm → symm)`，再进 kernel |

**做法 A（kernel 内放置）— MTE / SDMA / UDMA**：

```
input(GM) ──[kernel launch]──▶ kernel 内 put_nbi(GM→对称) ──▶ 远端 get/计算 ──▶ output(GM)
```

所有数据搬运都在 kernel 内部完成。e2e_us 起点 = kernel launch。**做法 A 下 e2e_us≈kernel_us 是正常的**，不是问题——数据搬运在 kernel 内自然覆盖。代码中 **MUST** 在注释中说明引擎选择（如 `// MTE put_nbi src=local GM, no Host-side memcpy needed`）。

**做法 B（kernel 外放置）— RDMA**：

```
input(GM) ──[aclrtMemcpy D2D: GM→对称内存]──▶ 对称内存
                                                  │
                                             [barrier]
                                                  │
              ──[kernel launch]──▶ kernel 内通信/计算 ──▶ output(GM)
```

e2e_us 起点 = aclrtMemcpy 前（kernel 外的搬运 **MUST** 计入 e2e 循环）。e2e_us **MUST** > kernel_us，差值等于 kernel 外的搬运时间。

**约束**：
- MTE/SDMA/UDMA 下 **SHOULD** 选用做法 A——直接在 kernel 内搬运，避免无意义的 kernel 外循环
- RDMA 下 **MUST** 选用做法 B——双方数据都必须在对称内存上
- 设计 DSL `schedule` 中 **MUST** 标注使用的引擎和对应的放置方式（A 或 B）
- **禁止**为制造 e2e_us > kernel_us 的假象而在做法 A 路径上增加无意义的 kernel 外搬运

### 1.3 公平对比原则

HCCL API 调用时间包含了内部的 staging 操作。SHMEM 的 e2e 计时覆盖从用户 input(GM) 到结果 output(GM) 的全部搬运。要实现公平对比，两侧的计时边界 **MUST** 对齐：

| 边界 | HCCL | SHMEM |
| --- | --- | --- |
| 计时起点 | 数据在 Device GM | 数据在 Device GM（用户 input 尚未被任何搬运操作触碰） |
| 计时终点 | 结果在 Device GM | 结果在 Device GM |

## 2. 双指标方案

每个 SHMEM 算子 **MUST** 报告两个延迟指标，字段名保持既有 `e2e_us` / `kernel_us` 不变。Phase 6 需要输出两套来源：

- `source=inline_event`：来自原有前后打点 `[PERF]`，提供 `e2e_us` 和 `kernel_us`
- `source=msprof_task_time`：来自 `msprof --task-time=l2` 的 `task_time_*.csv`，只提供 kernel 口径时间（记为 `kernel_us_msprof`）

| 指标 | 含义 | 起点 | 终点 | 用途 |
| --- | --- | --- | --- | --- |
| `e2e_us` | 端到端延迟——覆盖从用户 input(GM) 到结果 output(GM) 的全部数据搬运 | 仅 inline_event：原有 e2e 打点起点 | inline_event：stream sync 后 | 与历史 `[PERF]` 口径保持一致 |
| `kernel_us` | 原有打点 kernel 执行时间 | inline_event：kernel launch 前 | inline_event：stream sync 后 | 内部优化定位 |
| `kernel_us_msprof` | msprof：各卡目标 task 时延里取**最慢那张卡** | msprof：目标 task `task_start(us)` | msprof：目标 task `task_stop(us)` | 与 baseline msprof 公平对比 |


- 与 HCCL/aclnn baseline 对比时，主对比使用 SHMEM `kernel_us_msprof` vs baseline msprof `kernel_us`
- 内部优化分析时同时保留 SHMEM `source=inline_event` 与 `source=msprof_task_time`，观察二者差异
- **做法 A（MTE/SDMA/UDMA）下 e2e_us≈kernel_us 是正常的**——数据搬运在 kernel 内部完成，无需额外 staging 步骤
- **做法 B（RDMA 或显式 Host 侧搬运）下 e2e_us MUST > kernel_us**——差值即 kernel 外的搬运时间

## 3. 各算子 phase 分解与双采集口径

> **采集规则**：本节定义哪些操作属于 e2e/kernel 口径。SHMEM 必须保留原有 `[PERF]` 前后打点结果，其中 `e2e_us` 仍然用原方法；msprof task_time 只新增 kernel 口径结果。baseline 只要求 msprof kernel 口径结果。
>
> 以下示例使用 **做法 B**（Host 侧 `aclrtMemcpy` 把用户 GM 搬到对称内存）。若使用 MTE/SDMA/UDMA，数据放置可在 kernel 内部完成（做法 A），无需 Phase 1 的 Host 侧搬运——msprof 中 e2e 和 kernel 可能共享同一个 task/span。

### 3.1 all-to-all (transport, fp16) — 做法 B 示例

```text
Phase 1 [staging]   aclrtMemcpyAsync(stagingSym, inputDev, D2D, stream)
Phase 2 [barrier]   aclshmem_barrier_all()
Phase 3 [kernel]    LaunchAlltoallHalf(...)
Phase 4 [sync]      aclrtSynchronizeStream

e2e_latency    = Phase 1 + Phase 2 + Phase 3 + Phase 4
kernel_latency = Phase 3 + Phase 4
```

推荐 runner 结构（既输出 `[PERF]`，又供 msprof 包裹）：

```cpp
// warmup: 不进入最终统计，解析 task_time 时按 WARMUP 跳过
for (int i = 0; i < warmup; ++i) {
    aclrtMemcpyAsync(stagingSym, totalBytes, inputDev, totalBytes,
                     ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    aclshmem_barrier_all();
    LaunchAlltoallHalf(blockDim, stream, ...);
    aclrtSynchronizeStream(stream);
}

// measure: 保留原有前后打点并输出 [PERF]；外层可用 msprof 包裹
for (int i = 0; i < iters; ++i) {
    aclrtMemcpyAsync(stagingSym, totalBytes, inputDev, totalBytes,
                     ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    aclshmem_barrier_all();
    LaunchAlltoallHalf(blockDim, stream, ...);
    aclrtSynchronizeStream(stream);
}
```

measure 循环中每轮重新 staging + barrier 是为了模拟真实调用模式。staging 与 kernel 的 overlap 属于后续优化。

### 3.2 reduce-scatter (collective reduce, float32)

```text
Phase 1 [staging]   aclrtMemcpyAsync(stagingSym, inputDev, D2D, stream)
Phase 2 [barrier]   aclshmem_barrier_all()
Phase 3 [kernel]    LaunchReduceScatterFloat(...)
Phase 4 [sync]      aclrtSynchronizeStream

e2e_latency    = Phase 1 + Phase 2 + Phase 3 + Phase 4
kernel_latency = Phase 3 + Phase 4
```

### 3.3 dispatch-combine (transport, fp16 payload + int32 route)

staging 包含多个 buffer：

```text
Phase 1a [staging payload]    aclrtMemcpyAsync(payloadSym, inputDev, D2D, stream)
Phase 1b [staging route_dst]  aclrtMemcpyAsync(routeDstSym, routeDstDev, D2D, stream)
Phase 1c [staging route_slot] aclrtMemcpyAsync(routeSlotSym, routeSlotDev, D2D, stream)
Phase 2  [barrier]            aclshmem_barrier_all()
Phase 3a [route prefetch]     LaunchDispatchRoutePrefetch(...)  （仅 dispatch）
Phase 3b [kernel]             LaunchDispatchHalf / LaunchCombineHalf
Phase 4  [sync]               aclrtSynchronizeStream

e2e_latency    = Phase 1a + 1b + 1c + Phase 2 + Phase 3a + 3b + Phase 4
kernel_latency = Phase 3a + 3b + Phase 4
```

### 3.4 HCCL baseline

```
Phase 1 [H2D]        aclrtMemcpy(inputDev, hostInput, H2D)  ← 不计时
Phase 2 [collective]  HcclCollective(inputDev, outputDev)    ← 计时
Phase 3 [sync]        aclrtSynchronizeStream                 ← 计时

hccl_latency = Phase 2 + Phase 3
```

对比时：`e2e_latency_us` 对标 `hccl_latency`。

## 4. 指标计算

### 4.1 logical_payload_bytes

统一按单 PE 语义数据量计算，**MUST** 在输出中标注口径。

| 算子 | logical_payload_bytes | 口径 |
| --- | --- | --- |
| all-to-all | `n_pes × shard_elems × sizeof(dtype)` | 单 PE 输入（= 单 PE 输出） |
| reduce-scatter | `n_pes × shard_elems × sizeof(dtype)` | 单 PE 输入（全量） |
| dispatch | `local_tokens × hidden × sizeof(half)` | 单 PE 发送量 |
| combine | `local_tokens × hidden × sizeof(half)` | 单 PE 拉回量 |

### 4.2 algo_bandwidth

```text
algo_bandwidth_GBps = logical_payload_bytes / latency_s / 1e9
```

- 与 Baseline **带宽达标、Round 间对比**：使用 `kernel_bus_bandwidth_GBps`；**NEVER** 用 e2e 带宽作达标主指标
- 与 Baseline **时延参考**：使用 `e2e_latency_us`
- 内部分析：使用 `kernel_latency_us` 推算 kernel 口径 algo/bus 带宽
- 报告中标注使用的是哪个 latency
- 不乘 2，统一按 input size 计算（遵循 NCCL `algBw` 惯例）

### 4.3 bus_factor 和 bus_bandwidth

> **本表是 bus_factor 的唯一参照源。** code-gen、code-review、perf-eval 等 skill 中的 bus_factor 取值 **MUST** 以本表为准，**禁止**在其他文件中重复定义或使用 deviating 值。

```text
kernel_bus_bandwidth_GBps = algo_bandwidth_GBps(kernel) × bus_factor
```

| 算子 | bus_factor | 推导 |
| --- | --- | --- |
| AllReduce | `2 * (n_pes - 1) / n_pes` | ReduceScatter + AllGather 两阶段，每阶段 `(n-1)/n` |
| ReduceScatter | `(n_pes - 1) / n_pes` | 每步搬运 1/n 数据，共 (n-1) 步 |
| AllGather | `(n_pes - 1) / n_pes` | 每步搬运 1/n 数据，共 (n-1) 步 |
| AllToAll / Shuffle | `(n_pes - 1) / n_pes` | 每个 PE 发送 (n-1)/n 的数据到远端 |
| Broadcast | `1` | 单源广播，发送方 bandwidth 不放大 |
| P2P | `1` | 点对点，无集合通信放大因子 |
| dispatch | `(n_pes - 1) / n_pes`（近似） | 均匀路由假设；路由不均匀时按实际远端搬运量推导 |
| combine | `(n_pes - 1) / n_pes`（近似） | 同 dispatch |

dispatch-combine 的 bus_factor 取决于路由分布。路由不均匀时，按 `实际远端搬运 bytes / logical_payload_bytes` 计算，并在报告中说明。

### 4.4 bandwidth_utilization

```text
bandwidth_utilization_percent = kernel_bus_bandwidth_GBps / peak_bandwidth_GBps × 100
```

`peak_bandwidth_GBps` **MUST** 记录来源（SoC、链路类型、PE 数、拓扑）。按以下决策表取值（参考 [hardware-architecture.md](../../shmem-ops-design/references/hardware-architecture.md)）：

| 通信模式 | peak_bandwidth_GBps | 适用算子 |
| --- | --- | --- |
| P2P（单条 HCCS 链路单向） | **28 GB/s** | P2P put/get、非对称点对点搬运 |
| 集合通信（8PE full-mesh 聚合） | **196 GB/s**（= 7 × 28） | AllReduce、ReduceScatter、AllGather、AllToAll 等对称集合通信 |
| dispatch / combine（稀疏路由） | **28 GB/s**（P2P 口径，远端搬运量不确定） | dispatch、combine 等路由不确定算子 |
| 通算融合（通信+计算） | 按通信阶段选 P2P 或集合通信值 | fused_compute_comm 算子的通信阶段 |

> utilization >100% 时在报告中注释（可能原因：bus_factor 高估、链路实际带宽超出标称值）。

## 5. 性能输出格式

### 5.1 SHMEM inline_event 输出（保留原有 `[PERF]`）

```text
[PERF] source=inline_event e2e_us=<val> kernel_us=<val> algo_bandwidth_GBps=<val> e2e_bus_bandwidth_GBps=<val> kernel_bus_bandwidth_GBps=<val> bandwidth_utilization_pct=<val> payload_bytes=<val> bus_factor=<val> peak_bandwidth_GBps=<val>
```

### 5.2 SHMEM msprof 解析输出

```text
case=<id> source=msprof_task_time profile_dir=<path> latency_source=<task_time|fused_span> kernel_us_msprof=<val> kernel_bus_bandwidth_GBps_msprof=<val> bandwidth_utilization_pct_msprof=<val> payload_bytes=<val> bus_factor=<val> peak_bandwidth_GBps=<val>
```

**注意**：`e2e_us` 不从 msprof 计算；msprof 只补充 kernel 口径。`bus_factor` 在输出中保留以供公式验证，但不作为性能结果表的主列输出（详见 SKILL.md 和 §4.3）。

| 字段 | 说明 |
| --- | --- |
| `e2e_us` | 仅 inline_event：原有 e2e 打点 |
| `kernel_us` | inline_event：原有 kernel 打点 |
| `kernel_us_msprof` | msprof：各卡稳态时延取最慢卡（见 [profiling-tools.md §2](profiling-tools.md)） |
| `algo_bandwidth_GBps` | `logical_payload_bytes / e2e_us / 1e9`（e2e 口径算法带宽） |
| `e2e_bus_bandwidth_GBps` | `algo_bandwidth_GBps × bus_factor`；e2e 口径总线带宽，**仅作参考** |
| `kernel_bus_bandwidth_GBps` | `logical_payload_bytes / kernel_us / 1e9 × bus_factor`；**达标对比 MUST 用此字段** |
| `bandwidth_utilization_pct` | `kernel_bus_bandwidth_GBps / peak_bandwidth_GBps × 100` |
| `payload_bytes` | 单 PE 语义数据量 |
| `bus_factor` | 数值及来源 |
| `peak_bandwidth_GBps` | 硬件峰值带宽（按 SoC/拓扑取值） |

### 5.3 baseline msprof 解析输出

```text
case=<id> source=baseline_msprof_task_time baseline_api=<HcclAlltoAllV|aclnn...> profile_dir=<path> latency_source=<task_time|fused_span> kernel_us=<val> kernel_bus_bandwidth_GBps=<val> payload_bytes=<val> bus_factor=<val>
```

### 5.4 性能报告对比表

对比表格式和字段定义见 [SKILL.md §性能对比表](../SKILL.md)、[perf-chat-output-spec.md §2](perf-chat-output-spec.md) 和 [performance-report.md 模板](../templates/performance-report.md)。此处仅列出核心结构：

- 列名（带宽达标）：`metric | SHMEM inline_event | SHMEM msprof | Baseline msprof | delta | 达标`；时延参考列可并列 `e2e_us`、`kernel_us_inline`、`kernel_us_msprof`
- `Baseline` 列适用 hccl / aclnn / stitched，不限 HCCL
- **达标判断 MUST 使用 SHMEM `kernel_bus_bandwidth_GBps_msprof`** 与 baseline msprof `kernel_bus_bandwidth_GBps` 对比；有 baseline 时默认 ≥ baseline **80%**，无 baseline 时通信算子带宽利用率 ≥ 20%。inline_event 并列展示，不作为替代 baseline 的口径
- `e2e_us` / `e2e_bus_bandwidth_GBps` 仅作参考，**NEVER** 作为 Round 对比或达标主指标（见 [performance-eval/SKILL.md §达标规则](../SKILL.md)）
- `kernel_us` 列用于内部优化分析

## 6. 计算示例

### 6.1 all-to-all (fp16, 8 PE, shard_elems = 8388608)

```text
logical_payload_bytes = 8 × 8388608 × 2 = 134,217,728 (128 MiB per PE)
bus_factor = (8 - 1) / 8 = 0.875

# 假设测量值
e2e_us = 1200.0
kernel_us = 1050.0

algo_GBps (e2e)    = 134217728 / (1200.0 × 1e-6) / 1e9 = 111.85
algo_GBps (kernel) = 134217728 / (1050.0 × 1e-6) / 1e9 = 127.83
bus_GBps (e2e)     = 111.85 × 0.875 = 97.87
bus_GBps (kernel)  = 127.83 × 0.875 = 111.85

# HCCL baseline: hccl_us = 1100.0
algo_GBps (hccl)   = 134217728 / (1100.0 × 1e-6) / 1e9 = 122.02
bus_GBps (hccl)    = 122.02 × 0.875 = 106.77

# 达标判断 (kernel_bus_bandwidth_GBps vs hccl baseline)
bus_GBps (kernel, SHMEM) = 127.83 × 0.875 = 111.85
bus_GBps (hccl)          = 106.77
ratio: 111.85 / 106.77 = 104.8% >= 80% -> PASS

# e2e 带宽仅参考，不作达标主指标
# bus_GBps (e2e): 97.87 / 106.77 = 91.7%  ← 不可单独用于 PASS/FAIL
```

### 6.2 reduce-scatter (float32, 8 PE, shard_elems = 8388608)

```text
logical_payload_bytes = 8 × 8388608 × 4 = 268,435,456 (256 MiB per PE input)
bus_factor = (8 - 1) / 8 = 0.875

algo_GBps = logical_payload_bytes / latency_s / 1e9
bus_GBps  = algo_GBps × 0.875
```

### 6.3 dispatch (fp16, 8 PE, local_tokens=1024, hidden=4096)

```text
logical_payload_bytes = 1024 × 4096 × 2 = 8,388,608 (8 MiB per PE)
bus_factor ≈ (8 - 1) / 8 = 0.875  (均匀路由假设)

algo_GBps = logical_payload_bytes / latency_s / 1e9
bus_GBps  = algo_GBps × bus_factor
```

### 6.4 combine (fp16, 8 PE, local_tokens=1024, hidden=4096)

```text
logical_payload_bytes = 1024 × 4096 × 2 = 8,388,608 (8 MiB per PE)
bus_factor ≈ 0.875 (同 dispatch)
```

## 7. Device 侧关键片段打点

> **完整操作指南**：[device-profiling-guide.md](device-profiling-guide.md) —— 涵盖宏实现原理、kernel 侧打点规范、Host 侧 `aclshmemx_get_prof` 操作、数据结构、cycle2us 换算、从 broadcast/mte_perftest 等 benchmark 提取的完整工作示例。

性能采集 **MUST** 同时覆盖端到端指标和关键 Device 片段指标。优化判断 **NEVER** 只依赖总耗时；**MUST** 能回答"时间主要花在 copy、remote put/get、signal/wait、barrier、local compute 还是 finalize"。

参考 SHMEM examples 的 profiling 模式，在 kernel 中按 phase 插入 `SHMEMI_PROF_START(frame_id)` / `SHMEMI_PROF_END(frame_id)`，Host 侧在 stream 同步后调用 `aclshmemx_get_prof(nullptr, true)` 导出每个 PE、block、frame 的 cycles、count 和平均耗时。

基本要求：

- kernel 包含 `utils/prof/shmemi_prof.h`
- 每个关键执行片段分配稳定的 `frame_id`，并在 README 或性能报告中给出 frame map
- frame map 至少覆盖当前算子的主要耗时来源：
  - `copy_in` / `copy_out`：GM、UB、symmetric buffer 搬运
  - `remote_put_get`：MTE/SDMA/RDMA put/get 或 collective transport
  - `signal_or_barrier_wait`：signal wait、barrier、quiet、handle wait
  - `local_compute`：matmul、reduce、layout transform、packing/unpacking 等计算
  - `finalize`：写回、metadata、tail/final reduce 等收尾
- 采集结果 **MUST** 记录每个 frame 的 `cycles`、`count`、`avg_us`、`max_core_us`、占端到端时间比例，以及长尾 block/PE
- 优化优先从占比最大的 frame 或长尾最明显的 frame 入手
- profiling 编译开关、环境变量和日志路径 **MUST** 可复现；默认性能路径 **NEVER** 永久打开 printf、DumpTensor 或重型 debug 日志

示例打点形态：

```cpp
#include "utils/prof/shmemi_prof.h"

constexpr int32_t kProfRemotePut = 0;
constexpr int32_t kProfSignalWait = 1;
constexpr int32_t kProfLocalCompute = 2;

SHMEMI_PROF_START(kProfRemotePut);
aclshmemx_mte_put_nbi(dst, src, ub, bytes, elems, peer, EVENT_ID0);
SHMEMI_PROF_END(kProfRemotePut);

SHMEMI_PROF_START(kProfSignalWait);
aclshmem_signal_wait_until(signal, ACLSHMEM_CMP_EQ, expected);
SHMEMI_PROF_END(kProfSignalWait);
```

Host 侧形态参考：

```cpp
ACL_CHECK(aclrtSynchronizeStream(stream));
aclshmemx_get_prof(nullptr, true);
```

## 8. 计算指标

通信算子中计算阶段的算力使用率（如适用）**MAY** 计算有效算力和算力使用率，并与硬件峰值对比。纯搬运算子标注 N/A。

公式：

```text
compute_latency_s = compute_frame_us / 1e6
effective_flops = op_count_flops / compute_latency_s
compute_utilization_percent = effective_flops / peak_flops_for_dtype_and_soc * 100
```

字段定义：

- `op_count_flops`：按算子语义统计的有效 FLOPs，**MUST** 给出公式。例如 GEMM 通常为 `2 * M * N * K`
- `compute_frame_us`：优先来自 Device profiling 的 compute frame；没有分段打点时可用端到端时间，但 **MUST** 标注该值包含通信/等待开销
- `peak_flops_for_dtype_and_soc`：硬件峰值算力，**MUST** 记录 SoC、dtype、数据格式、是否使用 cube/vector 单元以及来源
- 输出通信阶段时间占比，**NEVER** 只给整体时间

## 9. 采集方法选择

按场景选择：

| 场景 | 推荐方法 |
| --- | --- |
| API/example 级通信 | msprof task_time；SHMEM cycle profiling 仅作内部片段诊断 |
| kernel 内片段定位 | `SHMEMI_PROF_START/END` + `aclshmemx_get_prof` |
| 算子级端到端 | msprof `--task-time=l2` |
| Python/PyTorch 集成 | torch_npu profiler + 输出文件 correctness check |
| 跨机/RDMA | 同时记录 control barrier、handle wait、网络路径和 PE 拓扑 |

采集 **MUST** 记录命令、workdir、环境变量、case、重复次数和日志路径。

## HCCL baseline 展开模式

采集 HCCL 时间前按 [hccl-expansion-mode.md](hccl-expansion-mode.md) 优先 `AIV`、回退 `AI_CPU`；模式差异主要影响小消息启动时延，不改变带宽公式。