# ArithmeticUtilization 字段说明

ArithmeticUtilization.csv 描述 AIC Cube 和 AIV Vector 的算术资源利用情况，包括活跃周期比例和指令计数。

比例字段是原始比值，不乘以 100。AIC 指标使用 AIC 总周期作为分母，AIV 指标使用 AIV 总周期作为分母；分母为零、计数缺失或字段不适用时输出 NA。

## 字段说明

| 字段 | 含义 | 适用 Core | 单位 | 依赖 | 计算规则 | NA 条件 | 分析说明 |
|---|---|---|---|---|---|---|---|
| `block_id` | PMU 聚合记录所属的 Block 标识 | 通用 | 无 | PMU 聚合键中的 `blockId` | 直接输出 `blockId` | 正常 PMU 行不为 `NA` | 与 `sub_block_id` 共同定位数据行 |
| `sub_block_id` | Core 类型和 Sub-block 的组合标识 | 通用 | 无 | `coreType`、`subBlockId` | AIC 为 `cubeN`，AIV 为 `vectorN` | 正常 PMU 行不为 `NA` | `N` 是 `subBlockId`，不是 `coreId` |
| `aic_time(us)` | AIC 行总执行时长 | AIC | us | AIC `C`、`F_AIC` | `C / F_AIC` | 非 AIC 行或频率无效 | 用于观察 AIC 总周期对应的时间 |
| `aic_total_cycles` | AIC 行总周期数 | AIC | cycle | AIC `C` | 直接输出 `C` | 非 AIC 行 | 是 AIC 利用率字段的分母 |
| `aic_cube_ratio` | AIC Cube 活跃周期占 AIC 总周期的比值 | AIC | 比值 | Cube 活跃周期、AIC `C` | `Cube 活跃周期 / C` | 非 AIC 行、Cube 活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aic_cube_fp_ratio` | AIC Cube 浮点指令活跃周期占 AIC 总周期的比值 | AIC | 比值 | Cube 浮点指令活跃周期、AIC `C` | `Cube 浮点指令活跃周期 / C` | 非 AIC 行、浮点指令活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aic_cube_int_ratio` | AIC Cube 整数指令活跃周期占 AIC 总周期的比值 | AIC | 比值 | Cube 整数指令活跃周期、AIC `C` | `Cube 整数指令活跃周期 / C` | 非 AIC 行、整数指令活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aic_cube_total_instr_number` | AIC Cube 执行的总指令数 | AIC | 条 | Cube 总指令计数 | 直接输出总指令计数 | 非 AIC 行或计数缺失 | 用于观察 Cube 总体指令量 |
| `aic_cube_fp_instr_number` | AIC Cube 执行的浮点指令数 | AIC | 条 | Cube 浮点指令计数 | 直接输出浮点指令计数 | 非 AIC 行或计数缺失 | 用于观察 Cube 浮点指令量 |
| `aic_cube_int_instr_number` | AIC Cube 执行的整数指令数 | AIC | 条 | Cube 整数指令计数 | 直接输出整数指令计数 | 非 AIC 行或计数缺失 | 用于观察 Cube 整数指令量 |
| `aiv_time(us)` | AIV 行总执行时长 | AIV | us | AIV `C`、`F_AIV` | `C / F_AIV` | 非 AIV 行或频率无效 | 用于观察 AIV 总周期对应的时间 |
| `aiv_total_cycles` | AIV 行总周期数 | AIV | cycle | AIV `C` | 直接输出 `C` | 非 AIV 行 | 是 AIV 利用率字段的分母 |
| `aiv_vec_ratio` | AIV Vector 活跃周期占 AIV 总周期的比值 | AIV | 比值 | Vector 活跃周期、AIV `C` | `Vector 活跃周期 / C` | 非 AIV 行、Vector 活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aiv_vec_vf_ratio` | AIV Vector VF 指令活跃周期占 AIV 总周期的比值 | AIV | 比值 | AIV VF 指令活跃周期、AIV `C` | `AIV VF 指令活跃周期 / C` | 非 AIV 行、VF 指令活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aiv_vec_sfu_ratio` | AIV Vector SFU 指令活跃周期占 AIV 总周期的比值 | AIV | 比值 | AIV SFU 指令活跃周期、AIV `C` | `AIV SFU 指令活跃周期 / C` | 非 AIV 行、SFU 指令活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |
| `aiv_vec_simt_vf_ratio` | AIV SIMT VF 指令活跃周期占 AIV 总周期的比值 | AIV | 比值 | AIV SIMT VF 指令活跃周期、AIV `C` | `AIV SIMT VF 指令活跃周期 / C` | 非 AIV 行、SIMT VF 指令活跃周期缺失或 `C` 为零 | 原始比值，不乘以 100 |

## 分析要点

- AIC 和 AIV 字段分别适用于对应 Core；另一种 Core 行的专用字段通常为 `NA`。
- 指令数量和活跃周期比例反映不同维度，不能用指令数替代周期比例。
- 只有字段依赖的计数和总周期同时有效时，比例才具有可比性。
