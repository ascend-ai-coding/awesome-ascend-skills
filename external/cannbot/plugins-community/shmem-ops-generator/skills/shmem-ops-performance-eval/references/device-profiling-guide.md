# Device 侧关键片段打点操作指南

提供 SHMEMI_PROF 宏对（Device cycle 打点）及 `aclshmemx_get_prof`（Host 侧导出，推荐）的完整操作流程。

> **API 状态**：`aclshmemx_show_prof()` 为零参数 deprecated API；**MUST** 使用 `aclshmemx_get_prof(nullptr, true)` 替代（等价行为，两参数公开接口）。

---

## 1. 概述

SHMEM 内置一套 Device 侧 cycle 级打点机制：

- **Device 侧**：`SHMEMI_PROF_START(frame_id)` / `SHMEMI_PROF_END(frame_id)` 宏对，包裹需要计时的代码段
- **Host 侧**：`aclshmemx_get_prof(nullptr, true)` 将采集到的 cycles / count / avg_us 打印到标准输出

**适用场景**：定位 kernel 内某段代码的 cycle 开销，分析通信/计算/同步各段占比，对比优化前后特定段的 cycle 变化。

**与 msprof 的区别**：msprof 提供 host 侧调度的宏观 timeline（task_time、launch latency 等），SHMEMI_PROF 提供 kernel 内部逐 block 逐 frame 的 cycle 级细分。

---

## 2. 前置条件

### 2.1 头文件

kernel 代码中必须包含：

```cpp
#include "utils/prof/shmemi_prof.h"
```

该头文件位于 SHMEM SDK 仓库的 `src/device/utils/prof/shmemi_prof.h`，用户代码通过编译时 `-I` 路径引用。

> **边界**：仅用于 `SHMEMI_PROF_*` 性能宏；宏内部 `aclshmemi_get_state` 为 SDK 实现细节，算子 **禁止** 直接调用 `aclshmemi_*`。默认性能路径 **禁止** 常驻 include。详见 [internal-api-boundary.md §4](../../shmem-ops-code-gen/references/internal-api-boundary.md)。

### 2.2 环境变量：指定采集 PE

**MUST** 在启动前设置环境变量 `SHMEM_CYCLE_PROF_PE`，指定在哪个 PE 上采集 profiling 数据：

```bash
export SHMEM_CYCLE_PROF_PE=0   # 在 PE 0 上采集
```

不设置该环境变量时，`prof_util_init` 会将 `profs->pe_id` 设为 -1 并跳过所有采集（`device_state->profs` 保持 nullptr，kernel 内所有 `SHMEMI_PROF_START/END` 宏不执行采集逻辑）。

**采集仅发生在指定 PE**，其他 PE 不受影响。

### 2.3 数据结构

```cpp
#define ACLSHMEM_CYCLE_PROF_MAX_BLOCK  64    // 最大采集 block 数
#define ACLSHMEM_CYCLE_PROF_FRAME_CNT  1024  // 最大 frame 数

typedef struct {
    int64_t ccount[ACLSHMEM_CYCLE_PROF_FRAME_CNT];   // frame 被调用的次数
    int64_t cycles[ACLSHMEM_CYCLE_PROF_FRAME_CNT];    // frame 累计 cycle 数
} aclshmem_prof_block_t;

typedef struct {
    int32_t pe_id;
    aclshmem_prof_block_t block_prof[ACLSHMEM_CYCLE_PROF_MAX_BLOCK];
} aclshmem_prof_pe_t;
```

`device_state->profs` 指向的 `aclshmem_prof_pe_t` 在 `prof_util_init` 中通过 `aclrtMalloc` 分配在 Device 端，kernel 内 `SHMEMI_PROF_START/END` 宏通过 `__gm__` 指针直接读写。

---

## 3. 宏实现原理

`shmemi_prof.h` 中的宏实现：

```cpp
#define SHMEMI_PROF_START(pf_id)                                                         \
    if ((pf_id) < ACLSHMEM_CYCLE_PROF_FRAME_CNT && AscendC::GetBlockIdx() < ACLSHMEM_CYCLE_PROF_MAX_BLOCK) { \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();        \
        if (device_state->profs != nullptr && device_state->profs->pe_id == shmem_my_pe()) { \
            pipe_barrier(PIPE_ALL);                                                       \
            auto cycle = AscendC::GetSystemCycle();                                       \
            device_state->profs->block_prof[AscendC::GetBlockIdx()].cycles[pf_id] -= cycle; \
        }                                                                                 \
    }

#define SHMEMI_PROF_END(pf_id)                                                           \
    if ((pf_id) < ACLSHMEM_CYCLE_PROF_FRAME_CNT && AscendC::GetBlockIdx() < ACLSHMEM_CYCLE_PROF_MAX_BLOCK) { \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();        \
        if (device_state->profs != nullptr && device_state->profs->pe_id == shmem_my_pe()) { \
            pipe_barrier(PIPE_ALL);                                                       \
            auto cycle = AscendC::GetSystemCycle();                                       \
            device_state->profs->block_prof[AscendC::GetBlockIdx()].cycles[pf_id] += cycle; \
            device_state->profs->block_prof[AscendC::GetBlockIdx()].ccount[pf_id] += 1;   \
        }                                                                                 \
    }
```

**关键行为**：

| 行为 | 说明 |
| --- | --- |
| `START` 减去当前 cycle | 使得后续 `END` 的 `+= cycle` 自然得到差值 |
| `END` 累加 count | 同一 frame 在一轮 kernel 调用中可能被多次执行（如循环内），`count` 记录执行次数 |
| `pipe_barrier(PIPE_ALL)` | 每次打点前后插入全流水 barrier 以保证 cycle 读数准确，这会引入额外同步开销 |
| 仅在指定 PE 采集 | `device_state->profs->pe_id == shmem_my_pe()` 门控 |
| block 数量限制 | `GetBlockIdx() < ACLSHMEM_CYCLE_PROF_MAX_BLOCK`，即只采集前 64 个 block |

**`pipe_barrier(PIPE_ALL)` 的影响**：该 barrier 会引入额外同步开销，影响绝对 cycle 数值，但由于 START 和 END 都执行 barrier，差值在一定程度上抵消。**不影响相同条件下的对比结论**，但不可与未插桩版本直接比较绝对值。

---

## 4. Kernel 侧打点规范

### 4.1 包含头文件与分配 frame_id

```cpp
#include "utils/prof/shmemi_prof.h"

// 使用 constexpr 定义稳定的 frame_id
constexpr int32_t kProfCopyIn       = 0;   // copy_in: GM/UB/symmetric buffer 搬运
constexpr int32_t kProfRemotePutGet = 1;   // remote_put_get: MTE/SDMA/RDMA put/get 通信
constexpr int32_t kProfSignalWait   = 2;   // signal_or_barrier_wait: 同步等待
constexpr int32_t kProfLocalCompute = 3;   // local_compute: matmul/reduce/layout 变换
constexpr int32_t kProfFinalize     = 4;   // finalize: 写回/metadata/收尾
```

### 4.2 5 类 frame 覆盖要求

每个算子的 kernel MUST 覆盖以下 5 类执行阶段（见 `timing-and-metrics-standard.md` §7）：

| # | frame 类别 | 说明 | 典型打点位置 |
| --- | --- | --- | --- |
| 1 | `copy_in` / `copy_out` | GM、UB、symmetric buffer 之间的搬运 | DataCopyIn / DataCopyOut 前后 |
| 2 | `remote_put_get` | MTE/SDMA/RDMA 的跨 PE put/get 或 collective transport | MTE put/get 调用循环前后 |
| 3 | `signal_or_barrier_wait` | signal wait、barrier、quiet、handle wait 等同步 | signal_wait_until / barrier 之前 |
| 4 | `local_compute` | matmul、reduce、layout transform、packing/unpacking | 计算核心循环前后 |
| 5 | `finalize` | 写回 output、metadata 写入、tail/final reduce | 最后写回/收尾代码段前后 |

**不要求每个算子都同时使用全部 5 个 frame_id**，但要求：报告中缺失某 frame 时 **MUST** 说明该阶段不存在的原因（例如纯通信算子无 `local_compute` 阶段）。

### 4.3 打点示例

以下展示通信算子典型模式（信号-搬运分离）。

#### 4.3.1 信号-搬运分离模式

root PE 将数据搬运到对称堆并通知 receiver，非 root PE 等待 signal 后从对称堆读取：

```cpp
const int my_pe = aclshmem_my_pe();

// frame 0: root 的 copy_in —— 将 input 搬运到 symmetric buffer
if (my_pe == root_pe) {
    SHMEMI_PROF_START(0);
    int64_t done = 0;
    while (done < local_len) {
        uint32_t copy_elems = std::min((uint32_t)(local_len - done), ub_elems);
        aclshmemx_mte_put_nbi(gva_data + start + done, input + start + done,
                              tmp_buf, ub_size, copy_elems, root_pe, EVENT_ID0);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        aclshmemx_signal_op(gva_sync + flag_offset, magic + chunk_seq,
                            ACLSHMEM_SIGNAL_SET, root_pe);
        done += copy_elems;
    }
    SHMEMI_PROF_END(0);
}

// frame 2: read + finalize（外层包裹）
// frame 1: signal_wait（内层嵌套，仅包裹同步等待）
SHMEMI_PROF_START(2);
int64_t done = 0;
while (done < local_len) {
    SHMEMI_PROF_START(1);
    aclshmem_signal_wait_until(source_signal, ACLSHMEM_CMP_GE, magic + chunk_seq);
    SHMEMI_PROF_END(1);
    uint32_t copy_elems = std::min((uint32_t)(local_len - done), ub_elems);
    aclshmemx_mte_get_nbi(output + start + done, gva_data + start + done,
                          tmp_buf, ub_size, copy_elems, root_pe, EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    done += copy_elems;
}
SHMEMI_PROF_END(2);
```

frame 2 包含 frame 1 的嵌套打点，用于分离外层总耗时和内层关键子阶段（总 read-finalize 时间 vs 纯 signal_wait 时间）。

#### 4.3.2 分段计算模式

通信与计算分阶段执行，每个阶段独立打点：

```cpp
// frame 0: remote put —— 向其他 PE 发送数据
SHMEMI_PROF_START(0);
for (int peer = 0; peer < n_pes; peer++) {
    if (peer == my_pe) continue;
    aclshmemx_mte_put_nbi(...);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}
SHMEMI_PROF_END(0);

// frame 1: remote get —— 从其他 PE 收集数据
SHMEMI_PROF_START(1);
for (int peer = 0; peer < n_pes; peer++) {
    aclshmem_signal_wait_until(source_signal, ACLSHMEM_CMP_GE, ...);
    aclshmemx_mte_get_nbi(...);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}
SHMEMI_PROF_END(1);

// frame 2: local compute —— 本地计算
SHMEMI_PROF_START(2);
// reduce / matmul / layout transform ...
SHMEMI_PROF_END(2);
```

### 4.4 打点注意事项

1. **打点位置**：放在要计时的代码段**两端**，不要放在循环/条件之外
2. **循环内打点**：同一 frame_id 在循环内多次执行，END 会累加 cycles 和 count，最终 avg_us = cycles / count
3. **流水线考虑**：`pipe_barrier(PIPE_ALL)` 在 `SetFlag` / `WaitFlag` 密集的代码段前后打点时，可能影响流水线效率；建议将打点放在较大的逻辑块边界而非每条指令
4. **避免永久开启**：非性能采集路径的编译 **NEVER** 包含 SHMEMI_PROF 宏（即不定义 prof 编译开关），避免静态开销

---

## 5. Host 侧采集与导出

### 5.1 初始化流程

SHMEM 在 `aclshmemx_init_attr` → `prof_util_init` 中自动完成：
1. 读取环境变量 `SHMEM_CYCLE_PROF_PE`
2. 若当前 PE 匹配，在 Device 端分配 `aclshmem_prof_pe_t`，初始化并赋值给 `global_state->profs`
3. 若不匹配，`profs` 保持 nullptr

### 5.2 导出打印

kernel 执行完成后，在 Host 侧 stream 同步后调用：

```cpp
aclshmemx_get_prof(nullptr, true);
```

输出固定格式表格：

```
============================================================
BlockID    FrameID    Cycles         Count          AvgTime(us)
------------------------------------------------------------
0          0          12345678       4              61.728
0          1          8765432        4              43.827
0          2          23456789       4              117.284
============================================================
```

**各列含义**：

| 列 | 含义 |
| --- | --- |
| BlockID | 采集的 block 编号（0 ~ min(block_dim, 64) - 1） |
| FrameID | 用户在 kernel 中指定的 `frame_id` |
| Cycles | 该 block 上该 frame 的累计 cycle 数 |
| Count | 该 block 上该 frame 被执行的次数（通常等于测量轮数 × 每次调用中的循环次数） |
| AvgTime(us) | 平均单次耗时 = Cycles / Count / cycle2us |

**cycle 到 us 的换算由 `prof_data_print` 内部完成**，根据 `aclrtGetSocName()` 自动选择：
- Ascend910B 系列：cycle2us = 50
- Ascend950 系列：cycle2us = 1000

### 5.3 典型 Host 侧调用位置

```cpp
// 1. 执行多轮 kernel；保留原有 inline_event 打点，同时外层可用 msprof 包裹
for (int i = 0; i < warmup_iters; i++) {
    LaunchOperator(block_dim, stream, ...);
    aclrtSynchronizeStream(stream);
}
auto t0 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < measure_iters; i++) {
    LaunchOperator(block_dim, stream, ...);
    aclrtSynchronizeStream(stream);
}
auto t1 = std::chrono::high_resolution_clock::now();
double kernel_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / measure_iters;

// 2. 打印原有 [PERF] 指标（完整字段集见 timing-and-metrics-standard.md §[PERF] 行；
//    此处仅示意 inline_event source 标记 + kernel_us，实际交付 MUST 输出全部字段：
//    e2e_us/kernel_us/algo_bandwidth_GBps/e2e_bus_bandwidth_GBps/kernel_bus_bandwidth_GBps/
//    bandwidth_utilization_pct/payload_bytes/bus_factor/peak_bandwidth_GBps）
//    msprof 解析结果在离线脚本中以 source=msprof_task_time 输出
std::cout << "[PERF] source=inline_event"
          << " kernel_us=" << kernel_us
          << " payload_bytes=" << payload_bytes << std::endl;

// 3. 打印 Device frame 数据
aclshmemx_get_prof(nullptr, true);
```

---

## 6. 将采集数据填入性能报告

### 6.1 填充 Device Frame Table

根据 `aclshmemx_get_prof(nullptr, true)` 的打印输出，将 `AvgTime(us)` 填入 `performance_report.md §3.4`：

```
| frame_id | phase | avg_us | max_core_us | count | percent_of_e2e | bottleneck_note |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | copy_in | 61.73 | 68.21 | 4 | 22.1% | - |
| 1 | signal_wait | 43.83 | 45.14 | 4 | 15.7% | peer PE 间无串行 |
| 2 | remote_get+finalize | 117.28 | 120.05 | 4 | 42.0% | 主要耗时来源 |
```

**各字段来源**：

| 字段 | 来源 |
| --- | --- |
| `avg_us` | 打印输出中各 block 的 `AvgTime(us)` 取平均值 |
| `max_core_us` | 打印输出中各 block 的 `AvgTime(us)` 取最大值 |
| `count` | 打印输出中的 `Count`（各 block 一致时取任一值，不一致时取最大值） |
| `percent_of_e2e` | `frame_avg_us / e2e_us × 100%` |
| `phase` 与 `bottleneck_note` | 由开发者根据 frame_id 对应的代码段人工填写 |

### 6.2 基于 frame 数据的瓶颈分析

从 frame table 出发的瓶颈分析角度：

1. 占比最高的 frame → 主要优化方向
2. `max_core_us / avg_us` 比值 → 长尾效应（> 1.5x 需关注）
3. `signal_wait` 占比 → 同步模式是否高效
4. 通信 frame 的 algo_bw / peak_bw → 带宽利用率

---

## 7. 限制与注意事项

| 限制 | 详情 |
| --- | --- |
| 单 PE 采集 | 仅 `SHMEM_CYCLE_PROF_PE` 指定的 PE 参与采集，其它 PE 无数据 |
| block 上限 | 只采集 `block_idx < ACLSHMEM_CYCLE_PROF_MAX_BLOCK`（64）的 block |
| frame 上限 | `frame_id < ACLSHMEM_CYCLE_PROF_FRAME_CNT`（1024） |
| barrier 开销 | 每次 START/END 包含 `pipe_barrier(PIPE_ALL)`，影响绝对值但不影响对比 |
| 编译依赖 | prof 功能在 SHMEM 初始化时自动判断，无需单独编译选项 |

**如果环境不支持 prof 采集**（如 SHMEM_CYCLE_PROF_PE 未设置、构建环境无 prof 支持），**MUST** 在性能报告 §3.4 说明缺失原因；瓶颈分析仍可基于 e2e/kernel event 时间结合代码路径反推完成。

---

## 8. 相关参考

- `timing-and-metrics-standard.md` §7 —— 打点覆盖要求与规范
- `profiling-tools.md` §1.2 —— msprof 与 SHMEM cycle profiling 对比
- SHMEM SDK 宏源码：`src/device/utils/prof/shmemi_prof.h`
- SHMEM SDK 数据结构：`include/host_device/shmem_common_types.h`
- SHMEM SDK Host 导出：`src/host/utils/prof/prof_util.cpp`
