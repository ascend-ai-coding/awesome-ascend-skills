# ResourceConflictRatio 字段说明

ResourceConflictRatio.csv 描述 AIC/AIV 执行资源的等待和冲突情况。比例字段是原始比值，不乘以 100。

AIC 指标使用 AIC 总周期作为分母，AIV 指标使用 AIV 总周期作为分母。字段不适用、所需计数缺失或分母为零时输出 `NA`；计数为零且分母有效时输出数值 `0`。

## 字段说明

| 字段 | 含义 | 适用 Core | 单位 | 依赖 | 计算规则 | NA 条件 | 分析说明 |
|---|---|---|---|---|---|---|---|
| `block_id` | PMU 聚合记录所属的 Block 标识 | 通用 | 无 | PMU 聚合键中的 `blockId` | 直接输出 `blockId` | 正常 PMU 行不为 `NA` | 与 `sub_block_id` 共同定位数据行 |
| `sub_block_id` | Core 类型和 Sub-block 的组合标识 | 通用 | 无 | `coreType`、`subBlockId` | AIC 为 `cubeN`，AIV 为 `vectorN` | 正常 PMU 行不为 `NA` | `N` 是 `subBlockId`，不是 `coreId` |
| `aic_time(us)` | AIC 行总执行时长 | AIC | us | AIC `C`、`F_AIC` | `C / F_AIC` | 非 AIC 行或频率无效 | 用于观察 AIC 总周期对应的时间 |
| `aic_total_cycles` | AIC 行总周期数 | AIC | cycle | AIC `C` | 直接输出 `C` | 非 AIC 行 | 是 AIC 等待比例字段的分母 |
| `aic_cube_wait_ratio` | AIC Cube 等待周期占 AIC 总周期的比值 | AIC | 比值 | Cube 等待周期、AIC `C` | `Cube 等待周期 / C` | 非 AIC 行、Cube 等待周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aic_mte1_wait_ratio` | AIC MTE1 等待周期占 AIC 总周期的比值 | AIC | 比值 | MTE1 等待周期、AIC `C` | `MTE1 等待周期 / C` | 非 AIC 行、MTE1 等待周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aic_mte2_wait_ratio` | AIC MTE2 等待周期占 AIC 总周期的比值 | AIC | 比值 | MTE2 等待周期、AIC `C` | `MTE2 等待周期 / C` | 非 AIC 行、MTE2 等待周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aic_mte3_wait_ratio` | AIC MTE3 等待周期占 AIC 总周期的比值 | AIC | 比值 | MTE3 等待周期、AIC `C` | `MTE3 等待周期 / C` | 非 AIC 行、MTE3 等待周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aiv_time(us)` | AIV 行总执行时长 | AIV | us | AIV `C`、`F_AIV` | `C / F_AIV` | 非 AIV 行或频率无效 | 用于观察 AIV 总周期对应的时间 |
| `aiv_total_cycles` | AIV 行总周期数 | AIV | cycle | AIV `C` | 直接输出 `C` | 非 AIV 行 | 是 AIV 冲突和等待比例字段的分母 |
| `aiv_vec_stu_cflt_ratio` | AIV Vector STU 冲突周期占 AIV 总周期的比值 | AIV | 比值 | Vector STU 冲突周期、AIV `C` | `Vector STU 冲突周期 / C` | 非 AIV 行、STU 冲突周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aiv_vec_ldu_cflt_ratio` | AIV Vector LDU 冲突周期占 AIV 总周期的比值 | AIV | 比值 | Vector LDU 冲突周期、AIV `C` | `Vector LDU 冲突周期 / C` | 非 AIV 行、LDU 冲突周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aiv_vec_sfu_cflt_ratio` | AIV Vector SFU 冲突周期占 AIV 总周期的比值 | AIV | 比值 | 四类 AIV Vector SFU 冲突周期、AIV `C` | `四类 AIV Vector SFU 冲突周期之和 / C` | 非 AIV 行、任一 SFU 冲突周期缺失或 `C` 为零 | 四类 SFU 冲突周期全部存在时才计算；原始比值，不乘以 100 |
| `aiv_vec_wait_ratio` | AIV Vector 等待周期占 AIV 总周期的比值 | AIV | 比值 | Vector 等待周期、AIV `C` | `Vector 等待周期 / C` | 非 AIV 行、Vector 等待周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aiv_mte2_wait_ratio` | AIV MTE2 等待周期占 AIV 总周期的比值 | AIV | 比值 | MTE2 等待周期、AIV `C` | `MTE2 等待周期 / C` | 非 AIV 行、MTE2 等待周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aiv_mte3_wait_ratio` | AIV MTE3 等待周期占 AIV 总周期的比值 | AIV | 比值 | MTE3 等待周期、AIV `C` | `MTE3 等待周期 / C` | 非 AIV 行、MTE3 等待周期缺失或 `C` 为零 | 原始比值，不乘以 100 |

## 分析要点

- AIC 等待比例和 AIV 冲突、等待比例只能在对应 Core 行中解释。
- `aiv_vec_sfu_cflt_ratio` 汇总多个 SFU 冲突计数，任一组成计数不可用时整体为 `NA`。
- 比例为 `0` 表示计数有效且未观测到对应等待或冲突，`NA` 表示无法计算或不适用。
