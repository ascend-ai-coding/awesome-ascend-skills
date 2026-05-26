---
name: external-cannbot-ops-aiss-tiling-solver
description: 使用 AISS-TilingSolver 工具自动求解 Ascend C 算子（MatMul / Vector）的最优 Tiling 参数，包括下载安装、构造
  JSON 输入、运行求解、结果解读与故障排查。触发：当用户使用 TilingSolver 工具求解时。
original-name: aiss-tiling-solver
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-26'
synced-commit: ac5bbd2b4cf427d011874e11f8d1e8b1bef66eda
license: UNKNOWN
---

# AISS-TilingSolver

指导用户使用 `AISS-TilingSolver` CLI 工具，为 Ascend C 算子自动求解最优 tiling 参数。

**核心原则：引导用户使用 TilingSolver CLI 工具。不要自己手动计算 tiling、不要给通用的 GPU tiling 建议。唯一的任务就是帮用户正确调用工具并解读工具的输出。**

## 获取工具

```bash
# aarch64
curl -LO https://gitcode.com/HIT1920/TilingSolver/releases/download/latest/tiling_solver_aarch64
curl -LO https://gitcode.com/HIT1920/TilingSolver/releases/download/latest/platform_info_aarch64
mv tiling_solver_aarch64 tiling_solver
mv platform_info_aarch64 platform_info

# x86_64
curl -LO https://gitcode.com/HIT1920/TilingSolver/releases/download/latest/tiling_solver_x86_64
curl -LO https://gitcode.com/HIT1920/TilingSolver/releases/download/latest/platform_info_x86_64
mv tiling_solver_x86_64 tiling_solver
mv platform_info_x86_64 platform_info

chmod +x tiling_solver platform_info
```

- `tiling_solver` — tiling 参数求解器（aarch64 / x86_64）
- `platform_info` — 硬件平台信息采集工具（aarch64 / x86_64）
- `platform_info` — 硬件平台信息采集工具

环境要求：aarch64 或 x86_64 架构，glibc ≥ 2.38。

## 完整使用流程

### 第一步：采集硬件平台参数

```bash
./platform_info
```

输出包含核心数、各级缓存容量、带宽等信息。

### 第二步：构造输入 JSON

将第一步采集到的硬件参数填入 JSON，作为自定义平台参数。同时填入算子形状和数据类型。

### 第三步：运行求解

```bash
./tiling_solver input.json
```

输出为 JSON，包含最优 tiling 参数。

## 算子类型

### MatMul（矩阵乘法）

```json
{
  "type": "matmul",
  "M": 1024,
  "N": 640,
  "K": 256,
  "a_dtype": "float16",
  "b_dtype": "float16",
  "c_dtype": "float32",
  "has_bias": false,
  "aic_core_num": 20,
  "aiv_core_num": 40,
  "L0A_SIZE": 65536,
  "L0B_SIZE": 65536,
  "L0C_SIZE": 131072,
  "L1_SIZE": 524032,
  "UB_SIZE": 196352,
  "L2_SIZE": 176160768,
  "HBM_SIZE": 34359738368,
  "BT_SIZE": 1024,
  "FP_SIZE": 2048,
  "L0A_ALIGN": 512,
  "L0B_ALIGN": 512,
  "L0C_ALIGN": 64,
  "UB_ALIGN": 32,
  "FRACTAL_DIM": 16,
  "DATABLOCK_SIZE": 32,
  "L2_BW": 110,
  "HBM_BW": 32,
  "L1_TO_L0_BW": 64,
  "FIXPIPE_BW": 64,
  "GM_TRANSFER_GRANULARITY": 512,
  "MIN_TRANSFER_FOR_PEAK_BW": 16384,
  "CUBE_OPS_PER_CYCLE": 4096,
  "optimize": true,
  "timeout_ms": 10000
}
```

必填字段：`type`, `M`, `N`, `K`。其他有默认值。硬件参数从 `platform_info` 获取后填入。

输出字段：`base_m`, `base_n`, `base_k`（内层循环切块），`single_core_m`, `single_core_n`（每个核的 M/N 范围），`step_ka`, `step_kb`（K 方向 L1 缓冲步长），`depth_a1`, `depth_b1`（L1 缓冲深度），`db_l0a`, `db_l0b`, `db_l0c`（L0 双缓冲，1 或 2），`iterate_order`（0=M 优先，1=N 优先），`solve_time_ms`, `status`。

### Vector（逐元素算子，如 Add / LeakyReLU）

```json
{
  "type": "vector",
  "total_length": 16384,
  "dtype": "float16",
  "num_inputs": 2,
  "num_outputs": 1,
  "temp_buf_dtypes": ["float32"],
  "UB_SIZE": 196352,
  "aiv_core_num": 40,
  "optimize": true
}
```

必填字段：`type`, `total_length`。其他有默认值。同样建议从 `platform_info` 获取硬件参数。

输出字段：`block_dim`（AI Core 核数），`tile_num`（每核 tile 数），`buf_num`（缓冲数），`tile_length`（每 tile 元素数），`solve_time_ms`, `status`。

## 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `optimize` | bool | `true` | 是否用 cost model 搜索最优解 |
| `timeout_ms` | int | `10000` | Z3 求解超时时间（毫秒） |

支持的数据类型：`float16`, `float32`, `bfloat16`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `bool`。也支持别名 `fp16`, `fp32`, `bf16`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `half`, `float`, `double`。

## 结果解读

- `status: "ok"` — 求解成功，参数可直接应用于算子实现
- `status: "unsat"` — 约束不可满足，当前 shape/dtype 在此硬件上无可行解
- `status: "timeout"` — Z3 求解超时，可尝试增大 `timeout_ms` 或关闭 `optimize`
- `status: "error"` — 输入参数非法或内部异常

## 故障排查

- **"cannot execute binary file"** — 下载了错误架构的二进制，确认用 `uname -m` 查看平台（aarch64 / x86_64），下载对应版本
- **"GLIBC_2.38 not found"** — glibc 版本过旧，升级系统或使用容器
- **返回 unsat** — 尝试缩小维度确认是否有可行解，再逐步放大
- **返回 timeout** — 增大 `timeout_ms`（如 30000），或设 `"optimize": false` 获取可行解

## 协助用户

1. 未提供算子类型和维度时，主动询问
2. 建议用户先运行 `platform_info` 获取硬件参数
3. 将 platform_info 输出整合到 tiling JSON 中
4. 构造完整 JSON 文件并运行 `tiling_solver`
5. 解读输出，解释每个字段对算子性能的影响
6. 报错时根据 status 诊断并给出修复建议
