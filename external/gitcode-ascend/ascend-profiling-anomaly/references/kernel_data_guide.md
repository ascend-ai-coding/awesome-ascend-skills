# Kernel Data Guide: Raw Data → Step → Structure → Block/Side → Op

## Table of Contents
1. [Raw Data Sources & Column Schemas](#1-raw-data-sources--column-schemas)
2. [Level 1: Step — Kernel Assignment & Metrics](#2-level-1-step)
3. [Level 2: Structure (Layer) — Segmentation & Metrics](#3-level-2-structure-layer)
4. [Level 3: Block / Side — Classification & Dual Clocks](#4-level-3-block--side)
5. [Level 4: Op — Individual Operator Analysis](#5-level-4-op)
6. [Multi-Stream Handling](#6-multi-stream-handling)
7. [Communication Dedup Rules](#7-communication-dedup-rules)
8. [Trace View JSON Structure](#8-trace-view-json-structure)
9. [Common Pitfalls](#9-common-pitfalls)

---

## 1. Raw Data Sources & Column Schemas

### 1.1 kernel_details.csv (PRIMARY device source)

This is the most granular device-side data. Each row is one kernel invocation on the NPU.

| Column | Type | Unit | Description |
|---|---|---|---|
| `Name` | string | — | Kernel or task name, e.g. `MatMul`, `Add`, `HcomAllReduce` |
| `Task Type` | string | — | Execution unit: `AI_CORE`, `AI_CPU`, `HCCL`, `MIX_AIC`, `MIX_AIV`, `FFTS_PLUS`, `DVPP` |
| `Accelerator Core` | string | — | Core type used: `AiCore`, `AiCpu`, `AiVector`, `MixAic`, etc. |
| `Start Time(us)` | float | μs | Absolute start timestamp on device clock |
| `Duration(us)` | float | μs | Kernel execution time (device busy) |
| `Wait Time(us)` | float | μs | Time kernel spent waiting before execution started |
| `Block Dim` | integer | — | Parallelism dimension |
| `Input Shapes` | string | — | e.g. `"[2048,4096];[4096,4096]"` — may be empty if `record_shapes=false` |
| `Output Shapes` | string | — | e.g. `"[2048,4096]"` |
| `Input Data Types` | string | — | e.g. `"FLOAT16;FLOAT16"` |
| `Output Data Types` | string | — | e.g. `"FLOAT16"` |
| `Stream ID` | integer | — | Device stream this kernel ran on |
| `Task ID` | integer | — | Task identifier |
| `Cycle Counter` | integer | — | Hardware cycle count (if PMU enabled) |

**Key relationships:**
- `total_cost_us = Duration(us) + Wait Time(us)` — what you see in timeline view
- A kernel's device-busy interval is `[Start Time(us), Start Time(us) + Duration(us)]`
- The wait happens BEFORE the start: the kernel queued at `Start Time(us) - Wait Time(us)` but started executing at `Start Time(us)`

### 1.2 op_summary_*.csv (FALLBACK device source)

Aggregated per-operator statistics. Use when `kernel_details.csv` is unavailable or for summary statistics.

| Column | Type | Unit | Description |
|---|---|---|---|
| `Op Name` | string | — | Operator name (may differ slightly from kernel name) |
| `Task Type` | string | — | Same enum as kernel_details |
| `Count` | integer | — | Number of invocations |
| `Total Duration(us)` | float | μs | Sum of all kernel durations |
| `Min Duration(us)` | float | μs | Minimum single kernel duration |
| `Max Duration(us)` | float | μs | Maximum single kernel duration |
| `Avg Duration(us)` | float | μs | Average kernel duration |
| `Total Wait Time(us)` | float | μs | Sum of all wait times |
| `Input Shapes` | string | — | Representative shape |

**Limitation:** `op_summary` does not have per-invocation timestamps, so you cannot build device intervals from it. Use only for aggregated statistics or when `kernel_details.csv` is missing.

### 1.3 trace_view.json (HOST + DEVICE events)

Chrome Trace Format. Contains both host and device events in a single timeline.

| Field | Description |
|---|---|
| `ph` | Phase: `X` (complete event), `B`/`E` (begin/end), `i` (instant) |
| `ts` | Timestamp in μs |
| `dur` | Duration in μs (for `X` events) |
| `name` | Event name |
| `cat` | Category: `cpu_op`, `python_function`, `user_annotation`, `kernel`, `communication`, etc. |
| `pid` | Process ID (host PID or device ID) |
| `tid` | Thread ID (host thread or stream ID) |
| `args` | Dictionary of extra fields: `Call Stack`, `Input Shapes`, etc. |

**Key categories and what they tell you:**

| Category | Side | What it shows |
|---|---|---|
| `cpu_op` | host | PyTorch CPU operators (aten::*, torch_npu::*) |
| `python_function` | host | Python function calls (if with_stack=true) |
| `user_annotation` | host | Step markers (`ProfilerStep#N`), custom annotations |
| `kernel` | device | Same events as kernel_details.csv but in trace format |
| `communication` | device | HCCL collective operations |
| `AscendCL` | host | ACL runtime API calls (aclrtMalloc, aclrtMemcpy, etc.) |

### 1.4 communication.json

Authoritative source for communication timing. Overrides `kernel_details.csv` for total comm volume.

| Field | Description |
|---|---|
| `collective_name` | e.g. `HcomAllReduce`, `HcomAllGather` |
| `start_timestamp_us` | Start time |
| `duration_us` | Communication duration |
| `message_size_bytes` | Data volume |
| `transport_type` | `RDMA`, `SDMA`, etc. |
| `group_name` | Communication group |

### 1.5 operator_details.csv (HOST operator source)

| Column | Type | Description |
|---|---|---|
| `Op Name` | string | Host operator name |
| `Input Shapes` | string | Input tensor shapes |
| `Call Stack` | string | Python call stack (if with_stack=true) |
| `Execution Time(us)` | float | Host-side execution time |
| `Device Duration(us)` | float | Corresponding device execution time |

---

## 2. Level 1: Step

### 2.1 Step boundary detection

Steps are identified (in priority order) by:

1. `user_annotation` events named `ProfilerStep#N` in `trace_view.json`
2. `Iteration#N` or `Step#N` annotations
3. Custom user markers (model-specific)
4. If no markers exist, treat the entire capture span as a single pseudo-step

Each step defines a **service window**:
- `W_i = [S_i, S_{i+1})` where `S_i` is step `i` start, `S_{i+1}` is next step start
- For the last step: `W_i = [S_i, E_i]` where `E_i` is the terminal end of the last kernel in that step

### 2.2 Assigning kernels to steps

For each kernel in `kernel_details.csv`:

```
step_assignment: kernel belongs to step i if:
    step_start_us <= kernel.Start_Time_us < step_end_us
```

Clip kernel intervals to the step window:
```
effective_start = max(step_start_us, kernel.Start_Time_us)
effective_end = min(step_end_us, kernel.Start_Time_us + kernel.Duration_us)
```

This handles boundary kernels that straddle two steps (rare but possible with async dispatch).

### 2.3 Per-step kernel inventory

For each step, build a table of all assigned kernels:

| Per-step field | Computation |
|---|---|
| `total_kernel_count` | Count of all kernels in the step |
| `ai_core_count` | Count where `Task Type = AI_CORE` |
| `ai_cpu_count` | Count where `Task Type = AI_CPU` |
| `hccl_count` | Count where `Task Type = HCCL` |
| `kernel_duration_sum_ms` | Sum of all `Duration(us)` / 1000 |
| `kernel_wait_sum_ms` | Sum of all `Wait Time(us)` / 1000 |
| `distinct_streams` | Set of unique `Stream ID` values |
| `stream_count` | Number of distinct streams used |

### 2.4 Per-step bubble metrics

From the merged device intervals (see `scripts/reference_host_gap_branch.py`):

| Metric | Definition |
|---|---|
| `service_ms` | Step window duration |
| `device_busy_union_ms` | Total merged device-busy time |
| `underfeed_ms` | `service_ms - device_busy_union_ms` |
| `underfeed_ratio` | `underfeed_ms / service_ms` |
| `prelaunch_gap_ms` | Gap from step start to first device activity |
| `tail_gap_ms` | Gap from last device activity to step end |
| `internal_bubble_total_ms` | Sum of gaps between merged segments |
| `largest_internal_bubble_ms` | Single largest internal gap |
| `bubble_count` | Number of internal gaps |

### 2.5 Step grouping

Steps with similar kernel composition form a **step group** (identified by `step_group_id`). Grouping is based on:
- Same total kernel count (within tolerance)
- Same sequence of operator names (template match)
- Same dominant task type distribution

The **dominant group** is the group contributing the most total time. Anomaly analysis focuses on the dominant group.

---

## 3. Level 2: Structure (Layer)

### 3.1 What is a structure?

Within a step, kernels form repeating **structures** corresponding to model layers. For example, a transformer model's step might contain:

```
[Embedding kernels] [Block0 kernels] [Block1 kernels] ... [BlockN kernels] [Head kernels]
```

Each transformer block is one structure. The blocks typically have near-identical kernel sequences.

### 3.2 Segmentation methods

Structures are identified by (in priority order):

1. **User annotations**: Layer-level markers in trace_view.json (e.g., `TransformerBlock#0`, `Attention`, `MLP`)
2. **Name-pattern repetition**: Find the longest repeating subsequence of kernel names within the step. Each repetition is one structure.
3. **Time-gap segmentation**: Large gaps in the device timeline can indicate structure boundaries, especially when combined with name-pattern changes.

### 3.3 Per-structure kernel data

For each structure within a step:

| Field | Description |
|---|---|
| `structure_id` | Identifier (e.g., `block_0`, `block_1`) |
| `structure_type` | Template type (e.g., `transformer_block`, `embedding`, `head`) |
| `kernel_count` | Number of kernels in this structure |
| `start_us` | Start time of first kernel |
| `end_us` | End time of last kernel |
| `wall_ms` | `(end_us - start_us) / 1000` |
| `device_busy_union_ms` | Merged device-busy time within this structure span |
| `kernel_sum_ms` | Sum of kernel durations |
| `total_cost_ms` | Sum of (duration + wait) for all kernels |
| `ai_core_pct` | Fraction of kernel_sum from AI_CORE tasks |
| `ai_cpu_pct` | Fraction of kernel_sum from AI_CPU tasks |
| `hccl_pct` | Fraction of kernel_sum from HCCL tasks |
| `structure_share` | This structure's wall time / step service time |

### 3.4 Structure-level bubble analysis

For each structure, also compute:
- Internal bubbles within the structure span
- Bubble between this structure and the previous one (inter-structure gap)
- Whether the structure has its own prelaunch-like gap at the start

This tells you whether bubbles concentrate **inside** structures (kernel dispatch issues within a layer) or **between** structures (host launch lag between layers).

### 3.5 Identifying dominant structures

Rank structures by:
1. `wall_ms` — which structure takes the most wall time
2. `total_cost_ms` — which structure has the highest total cost
3. Bubble contribution — which structure contains or borders the largest bubbles

If rankings disagree, report all perspectives.

---

## 4. Level 3: Block / Side

### 4.1 Block vs Side classification

Within each structure:

- **Block (main compute)**: The primary chain of kernels that forms the forward or backward pass. Typically AI_CORE MatMul, Conv, Attention kernels on the main compute stream.
- **Side (auxiliary)**: Everything else — element-wise small ops, communication (HCCL AllReduce), AI_CPU ops, synchronization events, memory operations.

Classification heuristics:
1. Kernels on the main compute stream with `Task Type = AI_CORE` and `Duration > threshold` → Block
2. Kernels with `Task Type = HCCL` → Side (communication)
3. Kernels with `Task Type = AI_CPU` → Side (AICPU)
4. Small kernels (`Duration < 10us`) on any stream → Side (small ops)
5. Everything else → classify by stream: main stream = Block, other streams = Side

### 4.2 Four timing perspectives (MANDATORY)

For BOTH block and side, always compute all four:

| Perspective | Definition | What it reveals |
|---|---|---|
| `wall_ms` | End of last kernel - Start of first kernel | Real elapsed time including gaps |
| `busy_union_ms` | Merged device-busy time | Actual device utilization |
| `kernel_sum_ms` | Arithmetic sum of all kernel durations | Total compute ignoring overlap |
| `total_cost_ms` | Sum of `duration_us + wait_us` for all kernels | Full cost including wait |

**Why all four matter:**
- `kernel_sum >> wall`: High stream parallelism (overlap between streams)
- `wall >> busy_union`: Large gaps/bubbles within this block/side
- `total_cost >> kernel_sum`: Heavy wait time — check for wait-anchor false hotspots
- `busy_union ≈ wall ≈ kernel_sum`: Clean, sequential execution

### 4.3 Per-block/side kernel table

For each block and side, maintain a kernel inventory:

| Field | Description |
|---|---|
| `kernel_count` | Total kernels in this block/side |
| `dominant_task_type` | Most common Task Type |
| `dominant_stream` | Primary stream ID |
| `top_kernels_by_duration` | Top 5 kernels ranked by Duration(us) |
| `top_kernels_by_total_cost` | Top 5 kernels ranked by total_cost |
| `wait_heavy_kernels` | Kernels where wait_ratio > 0.8 |

### 4.4 Block/side bubble analysis

For each block:
- Internal bubble metrics (same as step level but scoped to the block's kernel set)
- Any gap between block end and side start (or vice versa)
- Whether the side's communication overlaps with the block's compute (pipeline overlap)

If the block has high `wall_ms` but also high internal bubbles, the bottleneck may not be compute but rather host dispatch within the block.

---

## 5. Level 4: Op

### 5.1 Op ↔ Kernel relationship

One **op** (operator) may produce one or more **kernels**. For example:
- `aten::linear` might produce a `MatMul` kernel + a `BiasAdd` kernel
- A fused op might produce a single kernel
- An HCCL collective might produce multiple communication kernels

The linkage between ops and kernels:
- In `trace_view.json`, `cpu_op` events (host-side) have `ts`/`dur` that bracket the device kernels they launched
- In `operator_details.csv`, `Op Name` may correspond to multiple rows in `kernel_details.csv` via naming conventions

### 5.2 Op-level metrics

For each unique op (grouped by name or name+shape):

| Metric | Definition |
|---|---|
| `call_count` | Number of invocations |
| `duration_us_avg` | Average kernel duration per invocation |
| `duration_us_total` | Total kernel duration across all invocations |
| `wait_us_avg` | Average wait time per invocation |
| `total_cost_us_avg` | Average (duration + wait) per invocation |
| `wait_ratio` | `wait_us_avg / total_cost_us_avg` |
| `task_type` | AI_CORE / AI_CPU / HCCL |
| `input_shapes` | Representative input shapes |

### 5.3 Op ranking — always present TWO rankings

1. **By total_cost**: Includes wait time. High-ranking ops here may be wait-anchor false hotspots.
2. **By kernel_duration**: Pure compute cost. This is where real kernel inefficiency shows.

When these two rankings disagree significantly for the same op, it signals wait-anchor risk.

### 5.4 Wait-anchor detection

```
For each op:
    wait_ratio = wait_us_avg / (duration_us_avg + wait_us_avg)
    total_cost_rank = rank by total_cost_us_avg (1 = most expensive)

    if wait_ratio > 0.95 and duration_us_avg < 10.0 and total_cost_rank <= 10:
        → WAIT_ANCHOR_FALSE_HOTSPOT
```

Report these prominently. They are a common source of mis-diagnosis.

### 5.5 AICPU classification

For ops with `Task Type = AI_CPU`:

```
masked_ratio = time that this AI_CPU kernel overlaps with AI_CORE activity / AI_CPU kernel duration

if masked_ratio >= 0.9:  → AICPU_MASKED_BUT_UNDESIRABLE
if 0.2 <= masked_ratio < 0.9:  → AICPU_PARTIALLY_EXPOSED
if masked_ratio < 0.2:  → AICPU_EXPOSED_NOT_ALLOWED
```

If `AICPU_EXPOSED_NOT_ALLOWED` ops overlap with step bubbles, escalate anomaly severity.

---

## 6. Multi-Stream Handling

Ascend NPU can use multiple device streams (compute stream, communication stream, copy stream, etc.).

### 6.1 Key rules

- **Interval merge**: Collect intervals from ALL streams into one set before merging. The merged busy union reflects total device utilization across streams.
- **Overlap**: Two kernels on different streams can execute simultaneously. This is why `kernel_sum > wall` is possible and normal.
- **Stream ID**: Available in `kernel_details.csv` column `Stream ID` and in `trace_view.json` as `tid` for device events.

### 6.2 Per-stream breakdown

When reporting, optionally break down per stream:

| Stream | Typical content |
|---|---|
| Main compute stream | AI_CORE forward/backward kernels |
| Communication stream | HCCL AllReduce, AllGather, etc. |
| Copy/DMA stream | HostToDevice, DeviceToHost memcpy |

This helps identify whether bubbles are on the compute stream (host dispatch issue) or between streams (synchronization issue).

---

## 7. Communication Dedup Rules

Communication kernels can appear in both `kernel_details.csv` and `communication.json`. To avoid double-counting:

1. If `communication.json` exists, use it as the authoritative source for total communication time
2. Identify HCCL kernels in `kernel_details.csv` by `Task Type = HCCL` or name matching `Hcom*`
3. When building device intervals, HCCL kernels are still included (they represent device-busy time on the comm stream)
4. For total-cost accounting, reconcile: use `communication.json` totals, not the sum of HCCL kernel durations from `kernel_details.csv`

---

## 8. Trace View JSON Structure

```json
{
  "traceEvents": [
    {
      "ph": "X",
      "ts": 1000000.0,
      "dur": 500.0,
      "name": "aten::linear",
      "cat": "cpu_op",
      "pid": 12345,
      "tid": 1,
      "args": {
        "Input Shapes": "[[2048, 4096], [4096, 4096]]",
        "Call Stack": "train.py:42 -> model.py:128 -> ..."
      }
    },
    {
      "ph": "X",
      "ts": 1000200.0,
      "dur": 300.0,
      "name": "MatMul",
      "cat": "kernel",
      "pid": 0,
      "tid": 0,
      "args": {
        "Stream ID": 0,
        "Task Type": "AI_CORE"
      }
    }
  ]
}
```

**Matching host ops to device kernels**: A `cpu_op` event at `ts=T, dur=D` encloses the device kernels it launched. Device kernels starting within `[T, T+D]` are candidates. This is approximate — async dispatch means the device kernel may start after the host op completes.

---

## 9. Common Pitfalls

### 9.1 Confusing wait time with execution time
`Wait Time(us)` is time the kernel spent **queued** before starting. It is NOT part of device-busy time. The device was idle during the wait. This is why wait-anchor false hotspots exist — ops with huge wait but tiny duration appear expensive in total_cost but consume almost no device compute.

### 9.2 Using kernel_sum as wall time
`kernel_sum` can exceed `wall_ms` when streams overlap. It can also be less than `wall_ms` when bubbles exist. Always compute and report both.

### 9.3 Treating op_summary as timestamped data
`op_summary_*.csv` only has aggregate stats (count, avg, min, max). It does NOT have per-invocation timestamps. You cannot build interval sets or detect bubbles from op_summary alone.

### 9.4 Ignoring stream multiplicity
If you merge intervals from only one stream, you miss overlap from other streams, making busy_union look lower than reality. Always merge across all streams.

### 9.5 Misattributing inter-structure gaps to intra-structure problems
A large gap between transformer block N and block N+1 is an inter-structure gap (likely host dispatch lag). A large gap within block N is an intra-structure gap (possibly AICPU exposure or sync). The distinction matters for root-cause attribution.

### 9.6 Dropping boundary kernels
Kernels that straddle step boundaries should be clipped, not dropped. A kernel starting 10μs before step end with 100μs duration contributes 10μs to this step and 90μs to the next.
