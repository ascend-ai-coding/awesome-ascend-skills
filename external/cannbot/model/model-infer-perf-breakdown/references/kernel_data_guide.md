# NPU性能数据说明

## kernel_details.csv

This is the most granular device-side data. Each row is one kernel invocation on the NPU.

---

## 字段列表

### 基础标识

| Column | Type | Unit | Description |
|---|---|---|---|
| `Step Id` | integer | — | Training/inference step identifier |
| `Device_id` | integer | — | NPU device identifier |
| `Model ID` | integer | — | Model identifier |
| `Task ID` | integer | — | Task identifier |
| `Stream ID` | integer | — | Device stream this kernel ran on |
| `Name` | string | — | Kernel or task name, e.g. `MatMul`, `Add`, `HcomAllReduce` |
| `Type` | string | — | Kernel type name |

### 执行状态

| Column | Type | Unit | Description |
|---|---|---|---|
| `OP State` | string | — | Operation execution state |
| `Accelerator Core` | string | — | Core type used: `AiCore`, `AiCpu`, `AiVector`, `MixAic`, etc. |

### 时间信息

| Column | Type | Unit | Description |
|---|---|---|---|
| `Start Time(us)` | float | μs | Absolute start timestamp on device clock |
| `Duration(us)` | float | μs | Kernel execution time (device busy) |
| `Wait Time(us)` | float | μs | Time kernel spent waiting before execution started |

### 并行参数

| Column | Type | Unit | Description |
|---|---|---|---|
| `Block Dim` / `Block Num` | integer | — | Parallelism dimension |
| `Mix Block Dim` / `Mix Block Num` | integer | — | Mixed parallelism dimension |
| `HF32 Eligible` | string | — | HF32 eligibility flag |

### 张量信息（可选）

以下字段取决于 `record_shapes` 配置，可能为空：

| Column | Type | Unit | Description |
|---|---|---|---|
| `Input Shapes` | string | — | e.g. `"[2048,4096];[4096,4096]"` |
| `Output Shapes` | string | — | e.g. `"[2048,4096]"` |
| `Input Data Types` | string | — | e.g. `"FLOAT16;FLOAT16"` |
| `Output Data Types` | string | — | e.g. `"FLOAT16"` |
| `Input Formats` | string | — | e.g. `"ND;ND"` |
| `Output Formats` | string | — | e.g. `"ND"` |

### 上下文

| Column | Type | Unit | Description |
|---|---|---|---|
| `Context ID` | integer | — | Context identifier |

### AI Core 指标

| Column | Type | Unit | Description |
|---|---|---|---|
| `aicore_time(us)` | float | μs | AI Core execution time |
| `aic_total_cycles` | integer | — | AI Core total cycles |

**AI Core 详细指标（可选，取决于 profiling 配置）**：

| Column | Type | Unit | Description |
|---|---|---|---|
| `aic_mac_time(us)` | float | μs | MAC unit time |
| `aic_mac_ratio` | float | — | MAC time ratio (0-1) |
| `aic_scalar_time(us)` | float | μs | Scalar unit time |
| `aic_scalar_ratio` | float | — | Scalar time ratio (0-1) |
| `aic_mte1_time(us)` | float | μs | MTE1 (memory transfer engine 1) time |
| `aic_mte1_ratio` | float | — | MTE1 time ratio (0-1) |
| `aic_mte2_time(us)` | float | μs | MTE2 time |
| `aic_mte2_ratio` | float | — | MTE2 time ratio (0-1) |
| `aic_fixpipe_time(us)` | float | μs | FixPipe unit time |
| `aic_fixpipe_ratio` | float | — | FixPipe time ratio (0-1) |
| `aic_icache_miss_rate` | float | — | AI Core ICache miss rate |
| `cube_utilization(%)` | float | % | Cube utilization percentage |

**AI Core FLOPs**：

| Column | Type | Unit | Description |
|---|---|---|---|
| `aic_mac_fp16_ratio` | float | — | FP16 MAC ratio |
| `aic_mac_int8_ratio` | float | — | INT8 MAC ratio |
| `aic_cube_fops` | integer | — | Cube FLOPs count |

### AI Vector 指标

| Column | Type | Unit | Description |
|---|---|---|---|
| `aiv_time(us)` | float | μs | AI Vector execution time |
| `aiv_total_cycles` | integer | — | AI Vector total cycles |

**AI Vector 详细指标（可选，取决于 profiling 配置）**：

| Column | Type | Unit | Description |
|---|---|---|---|
| `aiv_vec_time(us)` | float | μs | Vector unit time |
| `aiv_vec_ratio` | float | — | Vector time ratio (0-1) |
| `aiv_scalar_time(us)` | float | μs | AI Vector scalar unit time |
| `aiv_scalar_ratio` | float | — | AI Vector scalar time ratio (0-1) |
| `aiv_mte2_time(us)` | float | μs | AI Vector MTE2 time |
| `aiv_mte2_ratio` | float | — | AI Vector MTE2 time ratio (0-1) |
| `aiv_mte3_time(us)` | float | μs | AI Vector MTE3 time |
| `aiv_mte3_ratio` | float | — | AI Vector MTE3 time ratio (0-1) |
| `aiv_icache_miss_rate` | float | — | AI Vector ICache miss rate |

**AI Vector FLOPs**：

| Column | Type | Unit | Description |
|---|---|---|---|
| `aiv_vec_fp32_ratio` | float | — | FP32 vector ratio |
| `aiv_vec_fp16_ratio` | float | — | FP16 vector ratio |
| `aiv_vec_int32_ratio` | float | — | INT32 vector ratio |
| `aiv_vec_misc_ratio` | float | — | Misc vector ratio |
| `aiv_vector_fops` | integer | — | Vector FLOPs count |

---

## Task Type 取值

| Task Type | Description |
|-----------|-------------|
| `AI_CORE` | AI Core kernel |
| `AI_CPU` | AI CPU kernel |
| `HCCL` | Collective communication |
| `MIX_AIC` | Mixed AI Core |
| `MIX_AIV` | Mixed AI Vector |
| `FFTS_PLUS` | FFTS Plus |
| `DVPP` | Digital Vision Pre-Processing |
