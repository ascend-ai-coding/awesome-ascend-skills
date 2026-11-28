# PipeUtilization 字段说明

`PipeUtilization.csv` 描述 AIC/AIV 各执行流水的活跃时长、活跃周期占比、指令缓存未命中率及部分数据通路在活跃时间内的带宽。

本文沿用公共符号 `C`、`F_AIC`、`F_AIV` 和二进制 GB/s。`ActiveBW(bytes, active_cycles, F)` 表示：

```text
bytes * 1,000,000 / 1024^3 / (active_cycles / F)
```

复合计数定义如下：

```text
L1 写入修正计数 = max(0, max(0, L1 写入次数 - L0C 到 L1 传输次数) - floor(主存读取次数 / 2))
Fixpipe 相关传输计数 = 四类 Fixpipe 相关数据传输次数之和
GM 到 UB 相关计数 = max(0, 主存读取次数 - 第一类修正次数 - 第二类修正次数)
```

## 字段说明

| 字段 | 含义 | 适用 Core | 单位 | 依赖 | 计算规则 | `NA` 条件 | 分析说明 |
|---|---|---|---|---|---|---|---|
| `block_id` | PMU 聚合记录所属的 Block 标识 | 通用 | 无 | PMU 键中的 `blockId` | 直接输出 `blockId` | 正常 PMU 行不为 `NA` | 与 `sub_block_id` 共同定位数据行 |
| `sub_block_id` | Core 类型和 Sub-block 的组合标识 | 通用 | 无 | `coreType`、`subBlockId` | AIC 为 `cubeN`，AIV 为 `vectorN` | 正常 PMU 行不为 `NA` | `N` 是 `subBlockId`，不是 `coreId` |
| `aic_time(us)` | AIC 行总执行时长 | AIC | us | AIC `C`、`F_AIC` | `C / F_AIC` | 非 AIC 行或频率无效 | 用于观察本行 AIC 总周期对应的时间 |
| `aic_total_cycles` | AIC 行总周期数 | AIC | cycle | AIC `C` | 直接输出 `C` | 非 AIC 行 | 是多个 AIC `ratio` 字段的分母 |
| `aic_cube_time(us)` | Cube 流水活跃时长 | AIC | us | Cube 活跃周期、`F_AIC` | `Cube 活跃周期 / F_AIC` | 非 AIC 行、Cube 活跃周期缺失或频率无效 | 表示 Cube 活跃周期折算的时间 |
| `aic_cube_ratio` | Cube 活跃周期占 AIC 总周期的比值 | AIC | 比值 | Cube 活跃周期、AIC `C` | `Cube 活跃周期 / C` | 非 AIC 行、Cube 活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aic_scalar_time(us)` | AIC Scalar 流水活跃时长 | AIC | us | Scalar 活跃周期、`F_AIC` | `Scalar 活跃周期 / F_AIC` | 非 AIC 行、Scalar 活跃周期缺失或频率无效 | 表示 Scalar 活跃周期折算的时间 |
| `aic_scalar_ratio` | AIC Scalar 活跃周期占总周期的比值 | AIC | 比值 | Scalar 活跃周期、AIC `C` | `Scalar 活跃周期 / C` | 非 AIC 行、Scalar 活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aic_mte1_time(us)` | AIC MTE1 流水活跃时长 | AIC | us | MTE1 活跃周期、`F_AIC` | `MTE1 活跃周期 / F_AIC` | 非 AIC 行、MTE1 活跃周期缺失或频率无效 | 表示 MTE1 活跃周期折算的时间 |
| `aic_mte1_ratio` | AIC MTE1 活跃周期占总周期的比值 | AIC | 比值 | MTE1 活跃周期、AIC `C` | `MTE1 活跃周期 / C` | 非 AIC 行、MTE1 活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aic_mte1_active_bw(GB/s)` | AIC MTE1 活跃期间的 L1 读取带宽 | AIC | GB/s | AIC L1 读取次数、MTE1 活跃周期、`F_AIC` | `ActiveBW(AIC L1 读取次数 * 256, MTE1 活跃周期, F_AIC)` | 非 AIC 行、任一计算所需计数缺失、频率无效或活跃周期非正数 | AIC L1 读取次数由其他 Section 提供，单独采集时通常不可用 |
| `aic_mte2_time(us)` | AIC MTE2 流水活跃时长 | AIC | us | MTE2 活跃周期、`F_AIC` | `MTE2 活跃周期 / F_AIC` | 非 AIC 行、MTE2 活跃周期缺失或频率无效 | 表示 MTE2 活跃周期折算的时间 |
| `aic_mte2_ratio` | AIC MTE2 活跃周期占总周期的比值 | AIC | 比值 | MTE2 活跃周期、AIC `C` | `MTE2 活跃周期 / C` | 非 AIC 行、MTE2 活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aic_mte2_active_bw(GB/s)` | AIC MTE2 活跃期间的主存读取带宽 | AIC | GB/s | AIC 主存读取次数、MTE2 活跃周期、`F_AIC` | `ActiveBW(AIC 主存读取次数 * 128, MTE2 活跃周期, F_AIC)` | 非 AIC 行、任一计算所需计数缺失、频率无效或活跃周期非正数 | AIC 主存读取次数由其他 Section 提供，单独采集时通常不可用 |
| `aic_mte3_time(us)` | AIC MTE3 流水活跃时长 | AIC | us | MTE3 活跃周期、`F_AIC` | `MTE3 活跃周期 / F_AIC` | 非 AIC 行、MTE3 活跃周期缺失或频率无效 | 表示 MTE3 活跃周期折算的时间 |
| `aic_mte3_ratio` | AIC MTE3 活跃周期占总周期的比值 | AIC | 比值 | MTE3 活跃周期、AIC `C` | `MTE3 活跃周期 / C` | 非 AIC 行、MTE3 活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aic_mte3_active_bw(GB/s)` | AIC MTE3 活跃期间的 L1 写入相关带宽 | AIC | GB/s | AIC L1 写入次数、L0C 到 L1 传输次数、主存读取次数、MTE3 活跃周期、`F_AIC` | `ActiveBW(L1 写入修正计数 * 256, MTE3 活跃周期, F_AIC)` | 非 AIC 行、任一计算所需计数缺失、频率无效或活跃周期非正数 | 除 MTE3 活跃周期外的计算所需计数由其他 Section 提供，单独采集时通常不可用 |
| `aic_fixpipe_time(us)` | AIC Fixpipe 流水活跃时长 | AIC | us | Fixpipe 活跃周期、`F_AIC` | `Fixpipe 活跃周期 / F_AIC` | 非 AIC 行、Fixpipe 活跃周期缺失或频率无效 | 表示 Fixpipe 活跃周期折算的时间 |
| `aic_fixpipe_ratio` | AIC Fixpipe 活跃周期占总周期的比值 | AIC | 比值 | Fixpipe 活跃周期、AIC `C` | `Fixpipe 活跃周期 / C` | 非 AIC 行、Fixpipe 活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aic_fixpipe_active_bw(GB/s)` | AIC Fixpipe 活跃期间的相关数据带宽 | AIC | GB/s | 四类 AIC Fixpipe 相关数据传输次数、Fixpipe 活跃周期、`F_AIC` | `ActiveBW(Fixpipe 相关传输计数 * 128, Fixpipe 活跃周期, F_AIC)` | 非 AIC 行、任一计算所需计数缺失、频率无效或活跃周期非正数 | 数据传输次数由其他 Section 提供，单独采集时通常不可用 |
| `aic_icache_miss_rate` | AIC 指令缓存未命中计数与访问计数的比值 | AIC | 比值 | ICache 未命中次数、ICache 访问次数 | `ICache 未命中次数 / ICache 访问次数` | 非 AIC 行、任一计算所需计数缺失或 ICache 访问次数为零 | 原始比值，不乘以 100 |
| `aiv_time(us)` | AIV 行总执行时长 | AIV | us | AIV `C`、`F_AIV` | `C / F_AIV` | 非 AIV 行或频率无效 | 用于观察本行 AIV 总周期对应的时间 |
| `aiv_total_cycles` | AIV 行总周期数 | AIV | cycle | AIV `C` | 直接输出 `C` | 非 AIV 行 | 是多个 AIV `ratio` 字段的分母 |
| `aiv_vec_time(us)` | Vector 流水活跃时长 | AIV | us | Vector 活跃周期、`F_AIV` | `Vector 活跃周期 / F_AIV` | 非 AIV 行、Vector 活跃周期缺失或频率无效 | 表示 Vector 活跃周期折算的时间 |
| `aiv_vec_ratio` | Vector 活跃周期占 AIV 总周期的比值 | AIV | 比值 | Vector 活跃周期、AIV `C` | `Vector 活跃周期 / C` | 非 AIV 行、Vector 活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aiv_scalar_time(us)` | AIV Scalar 流水活跃时长 | AIV | us | Scalar 活跃周期、`F_AIV` | `Scalar 活跃周期 / F_AIV` | 非 AIV 行、Scalar 活跃周期缺失或频率无效 | 表示 Scalar 活跃周期折算的时间 |
| `aiv_scalar_ratio` | AIV Scalar 活跃周期占总周期的比值 | AIV | 比值 | Scalar 活跃周期、AIV `C` | `Scalar 活跃周期 / C` | 非 AIV 行、Scalar 活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aiv_mte2_time(us)` | AIV MTE2 流水活跃时长 | AIV | us | MTE2 活跃周期、`F_AIV` | `MTE2 活跃周期 / F_AIV` | 非 AIV 行、MTE2 活跃周期缺失或频率无效 | 表示 MTE2 活跃周期折算的时间 |
| `aiv_mte2_ratio` | AIV MTE2 活跃周期占总周期的比值 | AIV | 比值 | MTE2 活跃周期、AIV `C` | `MTE2 活跃周期 / C` | 非 AIV 行、MTE2 活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aiv_mte2_active_bw(GB/s)` | AIV MTE2 活跃期间的 GM 到 UB 相关带宽 | AIV | GB/s | AIV 主存读取次数、两类修正次数、MTE2 活跃周期、`F_AIV` | `ActiveBW(GM 到 UB 相关计数 * 128, MTE2 活跃周期, F_AIV)` | 非 AIV 行、任一计算所需计数缺失、频率无效或活跃周期非正数 | 除 MTE2 活跃周期外的计算所需计数由其他 Section 提供，单独采集时通常不可用 |
| `aiv_mte3_time(us)` | AIV MTE3 流水活跃时长 | AIV | us | MTE3 活跃周期、`F_AIV` | `MTE3 活跃周期 / F_AIV` | 非 AIV 行、MTE3 活跃周期缺失或频率无效 | 表示 MTE3 活跃周期折算的时间 |
| `aiv_mte3_ratio` | AIV MTE3 活跃周期占总周期的比值 | AIV | 比值 | MTE3 活跃周期、AIV `C` | `MTE3 活跃周期 / C` | 非 AIV 行、MTE3 活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aiv_mte3_active_bw(GB/s)` | AIV MTE3 活跃带宽字段 | AIV | GB/s | 当前实现未提供所需 DBI 数据 | 固定输出 `NA` | 所有数据行均为 `NA` | 不能据此判断 AIV MTE3 实际带宽 |
| `aiv_icache_miss_rate` | AIV 指令缓存未命中计数与访问计数的比值 | AIV | 比值 | ICache 未命中次数、ICache 访问次数 | `ICache 未命中次数 / ICache 访问次数` | 非 AIV 行、任一计算所需计数缺失或 ICache 访问次数为零 | 原始比值，不乘以 100 |

## 使用要点

- `time(us)` 和 `ratio` 描述流水活跃周期的不同表示，前者依赖频率，后者依赖总周期。
- 活跃带宽只使用流水活跃时长作为分母，不等同于以整个 Kernel 时长为分母的平均带宽。
- 本 Section 未采集但公式依赖的计数，只有在同次采集的其他 Section 提供后才可能计算；缺少时输出 `NA`。
