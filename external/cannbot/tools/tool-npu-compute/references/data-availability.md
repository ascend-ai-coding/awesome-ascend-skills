# 数据可用性说明

本文用于判断 Section CSV 中的 `NA` 来源。`NA` 表示该单元格不可用或不适用，不是数值零；统计平均值、总和或比例时应排除 `NA`，不能先替换为零。

## `NA` 来源

| 类型 | 当前实现条件 | 判断方法 |
|---|---|---|
| Core 类型不适用 | 当前 PMU 行不包含字段要求的 AIC 或 AIV 数据 | 根据 `sub_block_id` 的 `cubeN` 或 `vectorN` 与字段前缀判断 |
| 计算所需计数缺失 | 公式依赖的任一 PMU 计数不在当前行中 | 对照字段参考文档中的“依赖”和本次选择的 Section |
| 分母无效 | 比值分母为零 | 计算所需计数齐全但总周期或指定计数为零时出现 |
| 频率或时长无效 | Core 频率不是正数、行总周期为零导致带宽时长非正数，或所选操作时长无效 | 检查公共时长字段、频率环境和字段使用的时长规则 |
| 硬件最大带宽不可用 | 带宽利用率公式未取得有效最大带宽 | 当前代码未匹配到相应通路的正数上限时出现 |
| 当前实现未提供数据 | 公开字段已保留，但生成该字段所需的 DBI 数据尚未提供 | 字段在适用 Core 行中也固定为 `NA` |

非有限的计算结果和负数计数器值同样输出为 `NA`。

计数为零与数据不可用不同。计算所需计数存在、分母有效且分子为零时，比值输出数值 `0`；只有字段不适用或计算条件不完整时才输出 `NA`。`ResourceConflictRatio` 中的等待和冲突比例尤其需要按此规则区分。

## Core 类型适用性

- `cubeN` 行提供 AIC 数据，AIV 专用字段通常为 `NA`。
- `vectorN` 行提供 AIV 数据，AIC 专用字段通常为 `NA`。
- `block_id` 和 `sub_block_id` 是行标识，不受 Core 指标可用性影响。
- `Memory` 中 `read_main_memory_datas(KB)` 和 `write_main_memory_datas(KB)` 使用当前行 Core 的主存访问计数，因此 AIC 和 AIV 行都可能有值。
- `ArithmeticUtilization` 的 AIC Cube 字段只适用于 `cubeN` 行，AIV Vector 字段只适用于 `vectorN` 行。
- `ResourceConflictRatio` 的 AIC 等待字段只适用于 `cubeN` 行，AIV 冲突和等待字段只适用于 `vectorN` 行。
- `aiv_vec_sfu_cflt_ratio` 依赖全部 SFU 冲突组成计数；任一计数缺失时该字段为 `NA`。

## PipeUtilization 的跨 Section 计数依赖

`PipeUtilization` 单独选择时，采集结果足以计算流水活跃时间、流水比值和指令缓存未命中率，但以下活跃带宽还依赖其他 Section 提供的计数：

| 字段 | 其他 Section 提供的计数依赖 |
|---|---|
| `aic_mte1_active_bw(GB/s)` | AIC L1 读取次数 |
| `aic_mte2_active_bw(GB/s)` | AIC 主存读取次数 |
| `aic_mte3_active_bw(GB/s)` | AIC L1 写入次数、L0C 到 L1 传输次数和主存读取次数 |
| `aic_fixpipe_active_bw(GB/s)` | 四类 AIC Fixpipe 相关数据传输次数 |
| `aiv_mte2_active_bw(GB/s)` | AIV 主存读取次数和两类修正次数 |

同一次采集同时选择 `PipeUtilization` 和 `Memory` 时，合并后的采集计数包含上述依赖，这些字段在 Core 类型、频率和活跃周期等其他条件满足时可以计算。只选择 `PipeUtilization` 时，上述字段通常因计算所需计数缺失为 `NA`。

## 当前固定为 `NA` 的字段

以下字段由当前实现明确写为 `NA`，不是负载观测到零流量：

| Section | 字段 | 当前原因 |
|---|---|---|
| `PipeUtilization` | `aiv_mte3_active_bw(GB/s)` | 未提供所需 DBI 数据 |
| `Memory` | `aiv_ub_to_gm_bw(GB/s)` | 未提供所需 DBI 数据 |
| `Memory` | `UB_to_GM_datas(KB)` | 未提供所需 DBI 数据 |
| `Memory` | `UB_to_GM_bw_usage_rate(%)` | 未提供所需 DBI 数据 |
| `MemoryUB` | `aiv_ub_read_bw_gm(GB/s)` | 未提供所需 DBI 数据 |

## 分析原则

1. 先根据行标识判断 Core 类型，再判断字段是否适用。
2. 对适用字段，按公式依赖依次检查计算所需计数、分母、频率或时长和硬件上限。
3. 区分“固定不可用”和“本次采集条件未满足”；两者都不能解释为零。
4. 比较多次采集时，只比较两侧都有有效值的同名字段，并说明被排除的 `NA` 数量。
