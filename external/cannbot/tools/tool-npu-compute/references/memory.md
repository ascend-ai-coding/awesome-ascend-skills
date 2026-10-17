# Memory 字段说明

`Memory.csv` 描述 AIC/AIV 主存访问、AIC L1、AIV UB、MTE 指令与活跃比例，以及部分数据通路的数据量和带宽利用率。

本文使用以下定义：

```text
GM 到 UB 相关计数 = max(0, 主存读取次数 - 第一类修正次数 - 第二类修正次数)
BW(count, bytes_per_transfer, D) = count * bytes_per_transfer * 1,000,000 / 1024^3 / D
DataKB(count, bytes_per_transfer) = count * bytes_per_transfer / 1024
Usage(bw, maximum) = 100 * min(bw, maximum) / maximum
```

对 AIC 指标，`D` 在 `D_op` 有效时为 `D_op`，否则为 AIC 行时长；对 AIV 指标，回退时使用 AIV 行时长。当前实现按 SoC 名称选择单 Core 最大带宽：

| 通路 | SoC 名称包含 `959` 或 `95A` | 其他名称 |
|---|---:|---:|
| GM 到 L1 | 177.57 GB/s | 162.77 GB/s |
| L0C 到 L1 | 412.15 GB/s | 377.80 GB/s |
| L0C 到 GM | 142.05 GB/s | 130.21 GB/s |
| GM 到 UB | 177.96 GB/s | 163.13 GB/s |

## 字段说明

| 字段 | 含义 | 适用 Core | 单位 | 依赖 | 计算规则 | `NA` 条件 | 分析说明 |
|---|---|---|---|---|---|---|---|
| `block_id` | PMU 聚合记录所属的 Block 标识 | 通用 | 无 | PMU 键中的 `blockId` | 直接输出 `blockId` | 正常 PMU 行不为 `NA` | 与 `sub_block_id` 共同定位数据行 |
| `sub_block_id` | Core 类型和 Sub-block 的组合标识 | 通用 | 无 | `coreType`、`subBlockId` | AIC 为 `cubeN`，AIV 为 `vectorN` | 正常 PMU 行不为 `NA` | `N` 是 `subBlockId`，不是 `coreId` |
| `aic_time(us)` | AIC 行总执行时长 | AIC | us | AIC `C`、`F_AIC` | `C / F_AIC` | 非 AIC 行或频率无效 | 公共时长字段始终使用行周期，不替换为 `D_op` |
| `aic_total_cycles` | AIC 行总周期数 | AIC | cycle | AIC `C` | 直接输出 `C` | 非 AIC 行 | 是 AIC MTE 比值字段的分母 |
| `aic_l1_read_bw(GB/s)` | AIC 从 L1 读取数据的平均带宽 | AIC | GB/s | AIC L1 读取次数、AIC `D` | `BW(AIC L1 读取次数, 256, D)` | 非 AIC 行、L1 读取次数缺失或 `D` 无效 | 计数按每次 256 字节换算 |
| `aic_l1_write_bw(GB/s)` | AIC 向 L1 写入数据的平均带宽 | AIC | GB/s | AIC L1 写入次数、AIC `D` | `BW(AIC L1 写入次数, 128, D)` | 非 AIC 行、L1 写入次数缺失或 `D` 无效 | 计数按每次 128 字节换算 |
| `aic_main_mem_read_bw(GB/s)` | AIC 主存读取平均带宽 | AIC | GB/s | AIC 主存读取次数、AIC `D` | `BW(AIC 主存读取次数, 128, D)` | 非 AIC 行、主存读取次数缺失或 `D` 无效 | 表示所选时长内的平均带宽 |
| `aic_main_mem_write_bw(GB/s)` | AIC 主存写入平均带宽 | AIC | GB/s | AIC 主存写入次数、AIC `D` | `BW(AIC 主存写入次数, 128, D)` | 非 AIC 行、主存写入次数缺失或 `D` 无效 | 表示所选时长内的平均带宽 |
| `aic_mte1_instructions` | AIC MTE1 指令计数 | AIC | 次 | AIC MTE1 指令执行次数 | 直接输出 AIC MTE1 指令执行次数 | 非 AIC 行或 MTE1 指令执行次数缺失 | 指令次数不等于活跃周期数 |
| `aic_mte1_ratio` | AIC MTE1 活跃周期占总周期的比值 | AIC | 比值 | AIC MTE1 活跃周期、AIC `C` | `AIC MTE1 活跃周期 / C` | 非 AIC 行、MTE1 活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aic_mte2_instructions` | AIC MTE2 指令计数 | AIC | 次 | AIC MTE2 指令执行次数 | 直接输出 AIC MTE2 指令执行次数 | 非 AIC 行或 MTE2 指令执行次数缺失 | 指令次数不等于活跃周期数 |
| `aic_mte2_ratio` | AIC MTE2 活跃周期占总周期的比值 | AIC | 比值 | AIC MTE2 活跃周期、AIC `C` | `AIC MTE2 活跃周期 / C` | 非 AIC 行、MTE2 活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aic_mte3_instructions` | AIC MTE3 指令计数 | AIC | 次 | AIC MTE3 指令执行次数 | 直接输出 AIC MTE3 指令执行次数 | 非 AIC 行或 MTE3 指令执行次数缺失 | 指令次数不等于活跃周期数 |
| `aic_mte3_ratio` | AIC MTE3 活跃周期占总周期的比值 | AIC | 比值 | AIC MTE3 活跃周期、AIC `C` | `AIC MTE3 活跃周期 / C` | 非 AIC 行、MTE3 活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aiv_time(us)` | AIV 行总执行时长 | AIV | us | AIV `C`、`F_AIV` | `C / F_AIV` | 非 AIV 行或频率无效 | 公共时长字段始终使用行周期，不替换为 `D_op` |
| `aiv_total_cycles` | AIV 行总周期数 | AIV | cycle | AIV `C` | 直接输出 `C` | 非 AIV 行 | 是 AIV MTE 比值字段的分母 |
| `aiv_ub_to_gm_bw(GB/s)` | AIV UB 到 GM 方向的预留带宽字段 | AIV | GB/s | 当前实现未提供所需 DBI 数据 | 固定输出 `NA` | 所有数据行均为 `NA` | 不能据此判断 UB 到 GM 的实际带宽 |
| `aiv_gm_to_ub_bw(GB/s)` | AIV GM 到 UB 相关平均带宽 | AIV | GB/s | AIV 主存读取次数、两类修正次数、AIV `D` | `BW(GM 到 UB 相关计数, 128, D)` | 非 AIV 行、任一计算所需计数缺失或 `D` 无效 | 修正后的计数小于零时按零计算 |
| `aiv_main_mem_read_bw(GB/s)` | AIV 主存读取平均带宽 | AIV | GB/s | AIV 主存读取次数、AIV `D` | `BW(AIV 主存读取次数, 128, D)` | 非 AIV 行、主存读取次数缺失或 `D` 无效 | 表示所选时长内的平均带宽 |
| `aiv_main_mem_write_bw(GB/s)` | AIV 主存写入平均带宽 | AIV | GB/s | AIV 主存写入次数、AIV `D` | `BW(AIV 主存写入次数, 128, D)` | 非 AIV 行、主存写入次数缺失或 `D` 无效 | 表示所选时长内的平均带宽 |
| `aiv_mte2_instructions` | AIV MTE2 指令计数 | AIV | 次 | AIV MTE2 指令执行次数 | 直接输出 AIV MTE2 指令执行次数 | 非 AIV 行或 MTE2 指令执行次数缺失 | 指令次数不等于活跃周期数 |
| `aiv_mte2_ratio` | AIV MTE2 活跃周期占总周期的比值 | AIV | 比值 | AIV MTE2 活跃周期、AIV `C` | `AIV MTE2 活跃周期 / C` | 非 AIV 行、MTE2 活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aiv_mte3_instructions` | AIV MTE3 指令计数 | AIV | 次 | AIV MTE3 指令执行次数 | 直接输出 AIV MTE3 指令执行次数 | 非 AIV 行或 MTE3 指令执行次数缺失 | 指令次数不等于活跃周期数 |
| `aiv_mte3_ratio` | AIV MTE3 活跃周期占总周期的比值 | AIV | 比值 | AIV MTE3 活跃周期、AIV `C` | `AIV MTE3 活跃周期 / C` | 非 AIV 行、MTE3 活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `read_main_memory_datas(KB)` | 当前 Core 行的主存读取数据量 | AIC、AIV | KB | 当前 Core 的主存读取次数 | `DataKB(主存读取次数, 128)` | 主存读取次数缺失 | 每行按本行 Core 类型取值，不合并 AIC/AIV |
| `write_main_memory_datas(KB)` | 当前 Core 行的主存写入数据量 | AIC、AIV | KB | 当前 Core 的主存写入次数 | `DataKB(主存写入次数, 128)` | 主存写入次数缺失 | 每行按本行 Core 类型取值，不合并 AIC/AIV |
| `GM_to_L1_datas(KB)` | 当前实现用于表示 GM 到 L1 的数据量 | AIC | KB | AIC 主存读取次数 | `DataKB(AIC 主存读取次数, 128)` | 非 AIC 行或主存读取次数缺失 | 当前代码复用 AIC 主存读取次数计算该字段 |
| `GM_to_L1_bw_usage_rate(%)` | GM 到 L1 平均带宽占配置最大带宽的比例 | AIC | % | AIC 主存读取次数、AIC `D`、GM 到 L1 最大带宽 | `Usage(BW(AIC 主存读取次数, 128, D), maximum)` | 非 AIC 行、主存读取次数缺失、`D` 无效或最大带宽无效 | 结果上限为 100% |
| `L0C_to_L1_datas(KB)` | L0C 到 L1 的数据量 | AIC | KB | AIC L0C 到 L1 传输次数 | `DataKB(AIC L0C 到 L1 传输次数, 128)` | 非 AIC 行或 L0C 到 L1 传输次数缺失 | 计数按每次 128 字节换算 |
| `L0C_to_L1_bw_usage_rate(%)` | L0C 到 L1 平均带宽占配置最大带宽的比例 | AIC | % | AIC L0C 到 L1 传输次数、AIC `D`、L0C 到 L1 最大带宽 | `Usage(BW(AIC L0C 到 L1 传输次数, 128, D), maximum)` | 非 AIC 行、L0C 到 L1 传输次数缺失、`D` 无效或最大带宽无效 | 结果上限为 100% |
| `L0C_to_GM_datas(KB)` | L0C 到 GM 的数据量 | AIC | KB | AIC 主存写入次数 | `DataKB(AIC 主存写入次数, 128)` | 非 AIC 行或主存写入次数缺失 | 当前代码复用 AIC 主存写入次数计算该字段 |
| `L0C_to_GM_bw_usage_rate(%)` | L0C 到 GM 平均带宽占配置最大带宽的比例 | AIC | % | AIC 主存写入次数、AIC `D`、L0C 到 GM 最大带宽 | `Usage(BW(AIC 主存写入次数, 128, D), maximum)` | 非 AIC 行、主存写入次数缺失、`D` 无效或最大带宽无效 | 结果上限为 100% |
| `GM_to_UB_datas(KB)` | AIV GM 到 UB 相关数据量 | AIV | KB | AIV 主存读取次数、两类修正次数 | `DataKB(GM 到 UB 相关计数, 128)` | 非 AIV 行或任一计算所需计数缺失 | 修正后的计数小于零时按零计算 |
| `GM_to_UB_bw_usage_rate(%)` | GM 到 UB 相关平均带宽占配置最大带宽的比例 | AIV | % | GM 到 UB 相关计数、AIV `D`、GM 到 UB 最大带宽 | `Usage(BW(GM 到 UB 相关计数, 128, D), maximum)` | 非 AIV 行、任一计算所需计数缺失、`D` 无效或最大带宽无效 | 结果上限为 100% |
| `UB_to_GM_datas(KB)` | AIV UB 到 GM 方向的预留数据量字段 | AIV | KB | 当前实现未提供所需 DBI 数据 | 固定输出 `NA` | 所有数据行均为 `NA` | 不能据此判断 UB 到 GM 的实际数据量 |
| `UB_to_GM_bw_usage_rate(%)` | AIV UB 到 GM 方向的预留带宽利用率字段 | AIV | % | 当前实现未提供所需 DBI 数据 | 固定输出 `NA` | 所有数据行均为 `NA` | 不能据此判断 UB 到 GM 的实际带宽利用率 |

## 使用要点

- `aic_time(us)` 和 `aiv_time(us)` 始终表示对应 PMU 行的周期时长；混合 Kernel 的带宽分母可能使用 `D_op`，两者不能直接等同。
- 数据量按单条 Core 行输出，不代表整个 Kernel 在所有 Core 上的总数据量。
- 带宽利用率使用代码内按 SoC 名称选择的单 Core 最大带宽，并将超过最大值的观测带宽限制为最大值，因此结果不超过 100%。
- 固定为 `NA` 的字段不能按零带宽或零数据量解释。
