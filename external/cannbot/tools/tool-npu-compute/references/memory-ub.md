# MemoryUB 字段说明

`MemoryUB.csv` 描述 AIV 的 UB 与 Vector、GM 相关数据通路带宽，同时保留 AIC/AIV 公共时长字段。

本文使用以下定义：

```text
GM 到 UB 相关计数 = max(0, 主存读取次数 - 第一类修正次数 - 第二类修正次数)
BW(count, bytes_per_transfer, D) = count * bytes_per_transfer * 1,000,000 / 1024^3 / D
```

`D` 是公共规则选出的时长：混合 Kernel 且 Task Log 时长有效时使用 `D_op`，否则使用 AIV 行时长。

## 字段说明

| 字段 | 含义 | 适用 Core | 单位 | 依赖 | 计算规则 | `NA` 条件 | 分析说明 |
|---|---|---|---|---|---|---|---|
| `block_id` | PMU 聚合记录所属的 Block 标识 | 通用 | 无 | PMU 键中的 `blockId` | 直接输出 `blockId` | 正常 PMU 行不为 `NA` | 与 `sub_block_id` 共同定位数据行 |
| `sub_block_id` | Core 类型和 Sub-block 的组合标识 | 通用 | 无 | `coreType`、`subBlockId` | AIC 为 `cubeN`，AIV 为 `vectorN` | 正常 PMU 行不为 `NA` | `N` 是 `subBlockId`，不是 `coreId` |
| `aic_time(us)` | AIC 行总执行时长 | AIC | us | AIC `C`、`F_AIC` | `C / F_AIC` | 非 AIC 行或频率无效 | AIC 行可能存在，但本 Section 没有 AIC UB 指标 |
| `aic_total_cycles` | AIC 行总周期数 | AIC | cycle | AIC `C` | 直接输出 `C` | 非 AIC 行 | 用于核对 AIC 行采样周期 |
| `aiv_time(us)` | AIV 行总执行时长 | AIV | us | AIV `C`、`F_AIV` | `C / F_AIV` | 非 AIV 行或频率无效 | 该字段始终是 AIV 行周期时长，不替换为 `D_op` |
| `aiv_total_cycles` | AIV 行总周期数 | AIV | cycle | AIV `C` | 直接输出 `C` | 非 AIV 行 | 用于核对 AIV 行采样周期 |
| `aiv_ub_read_bw_vector(GB/s)` | UB 到 Vector 方向的数据读取带宽 | AIV | GB/s | UB 到 Vector 传输次数、`D` | `BW(UB 到 Vector 传输次数, 256, D)` | 非 AIV 行、UB 到 Vector 传输次数缺失或 `D` 无效 | 计数按每次 256 字节换算 |
| `aiv_ub_write_bw_vector(GB/s)` | Vector 到 UB 方向的数据写入带宽 | AIV | GB/s | Vector 到 UB 传输次数、`D` | `BW(Vector 到 UB 传输次数, 256, D)` | 非 AIV 行、Vector 到 UB 传输次数缺失或 `D` 无效 | 计数按每次 256 字节换算 |
| `aiv_ub_read_bw_gm(GB/s)` | UB 与 GM 读取方向的预留带宽字段 | AIV | GB/s | 当前实现未提供所需 DBI 数据 | 固定输出 `NA` | 所有数据行均为 `NA` | 不能据此判断该方向的实际带宽 |
| `aiv_ub_write_bw_gm(GB/s)` | 当前实现按 GM 主存读计数扣除指定子项后计算的 UB/GM 带宽 | AIV | GB/s | AIV 主存读取次数、两类修正次数、`D` | `BW(GM 到 UB 相关计数, 128, D)` | 非 AIV 行、任一计算所需计数缺失或 `D` 无效 | 按当前代码公式解释，不根据字段名反推相反的数据方向 |

## 使用要点

- 四个 UB 带宽字段只适用于 AIV；AIC 行中的这些字段为 `NA`。
- 混合 Kernel 中，带宽分母可能是 Task Log 得到的 `D_op`，不能默认使用同一行的 `aiv_time(us)` 反算。
- `aiv_ub_read_bw_gm(GB/s)` 当前固定不可用；Section 还采集其他计数，但当前四个公开 UB 带宽公式不使用它们。
