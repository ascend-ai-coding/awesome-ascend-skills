# MemoryL0 字段说明

`MemoryL0.csv` 描述 AIC 的 L0A、L0B、L0C 数据通路带宽，同时保留 AIC/AIV 公共时长字段。

本文使用 `BW(count, bytes_per_transfer, D)` 表示：

```text
count * bytes_per_transfer * 1,000,000 / 1024^3 / D
```

其中，`D` 是公共规则选出的时长：混合 Kernel 且 Task Log 时长有效时使用 `D_op`，否则使用 AIC 行时长。

## 字段说明

| 字段 | 含义 | 适用 Core | 单位 | 依赖 | 计算规则 | `NA` 条件 | 分析说明 |
|---|---|---|---|---|---|---|---|
| `block_id` | PMU 聚合记录所属的 Block 标识 | 通用 | 无 | PMU 键中的 `blockId` | 直接输出 `blockId` | 正常 PMU 行不为 `NA` | 与 `sub_block_id` 共同定位数据行 |
| `sub_block_id` | Core 类型和 Sub-block 的组合标识 | 通用 | 无 | `coreType`、`subBlockId` | AIC 为 `cubeN`，AIV 为 `vectorN` | 正常 PMU 行不为 `NA` | `N` 是 `subBlockId`，不是 `coreId` |
| `aic_time(us)` | AIC 行总执行时长 | AIC | us | AIC `C`、`F_AIC` | `C / F_AIC` | 非 AIC 行或频率无效 | 该字段始终是 AIC 行周期时长，不替换为 `D_op` |
| `aic_total_cycles` | AIC 行总周期数 | AIC | cycle | AIC `C` | 直接输出 `C` | 非 AIC 行 | 用于核对 AIC 行采样周期 |
| `aic_l0a_read_bw(GB/s)` | L0A 到 Cube 方向的读取带宽 | AIC | GB/s | L0A 到 Cube 传输次数、`D` | `BW(L0A 到 Cube 传输次数, 64, D)` | 非 AIC 行、L0A 到 Cube 传输次数缺失或 `D` 无效 | 计数按每次 64 字节换算 |
| `aic_l0a_write_bw(GB/s)` | L1 到 L0A 方向的写入带宽 | AIC | GB/s | L1 到 L0A 传输次数、`D` | `BW(L1 到 L0A 传输次数, 256, D)` | 非 AIC 行、L1 到 L0A 传输次数缺失或 `D` 无效 | 计数按每次 256 字节换算 |
| `aic_l0b_read_bw(GB/s)` | L0B 到 Cube 方向的读取带宽 | AIC | GB/s | L0B 到 Cube 传输次数、`D` | `BW(L0B 到 Cube 传输次数, 256, D)` | 非 AIC 行、L0B 到 Cube 传输次数缺失或 `D` 无效 | 计数按每次 256 字节换算 |
| `aic_l0b_write_bw(GB/s)` | L1 到 L0B 方向的写入带宽 | AIC | GB/s | L1 到 L0B 传输次数、`D` | `BW(L1 到 L0B 传输次数, 256, D)` | 非 AIC 行、L1 到 L0B 传输次数缺失或 `D` 无效 | 计数按每次 256 字节换算 |
| `aic_l0c_read_bw_cube(GB/s)` | L0C 到 Cube 方向的读取带宽 | AIC | GB/s | L0C 到 Cube 传输次数、`D` | `BW(L0C 到 Cube 传输次数, 1024, D)` | 非 AIC 行、L0C 到 Cube 传输次数缺失或 `D` 无效 | 计数按每次 1024 字节换算 |
| `aic_l0c_write_bw_cube(GB/s)` | Cube 到 L0C 方向的写入带宽 | AIC | GB/s | Cube 到 L0C 传输次数、`D` | `BW(Cube 到 L0C 传输次数, 1024, D)` | 非 AIC 行、Cube 到 L0C 传输次数缺失或 `D` 无效 | 计数按每次 1024 字节换算 |
| `aiv_time(us)` | AIV 行总执行时长 | AIV | us | AIV `C`、`F_AIV` | `C / F_AIV` | 非 AIV 行或频率无效 | AIV 行可能存在，但本 Section 没有 AIV L0 指标 |
| `aiv_total_cycles` | AIV 行总周期数 | AIV | cycle | AIV `C` | 直接输出 `C` | 非 AIV 行 | 用于核对 AIV 行采样周期 |

## 使用要点

- 六个 L0 带宽字段只适用于 AIC；AIV 行中的这些字段为 `NA`。
- 混合 Kernel 中，带宽分母可能是 Task Log 得到的 `D_op`，不能默认使用同一行的 `aic_time(us)` 反算。
