# L2Cache 字段说明

`L2Cache.csv` 分别描述 AIC 和 AIV 的 L2 读写请求在近端、远端的命中、未命中和 Victim 计数，并计算读写命中率。

对任一 Core 类型，读写命中率定义为：

```text
read_hit_rate = 100 * (close_read_hit_count + far_read_hit_count)
                / sum(all_close_and_far_read_counts)
write_hit_rate = 100 * (close_write_hit_count + far_write_hit_count)
                 / sum(all_close_and_far_write_counts)
```

计算命中率时要求对应六类计数全部存在；六类计数的总和为零时，命中率按 `0` 输出。

## 字段说明

| 字段 | 含义 | 适用 Core | 单位 | 依赖 | 计算规则 | `NA` 条件 | 分析说明 |
|---|---|---|---|---|---|---|---|
| `block_id` | PMU 聚合记录所属的 Block 标识 | 通用 | 无 | PMU 键中的 `blockId` | 直接输出 `blockId` | 正常 PMU 行不为 `NA` | 与 `sub_block_id` 共同定位数据行 |
| `sub_block_id` | Core 类型和 Sub-block 的组合标识 | 通用 | 无 | `coreType`、`subBlockId` | AIC 为 `cubeN`，AIV 为 `vectorN` | 正常 PMU 行不为 `NA` | `N` 是 `subBlockId`，不是 `coreId` |
| `aic_time(us)` | AIC 行总执行时长 | AIC | us | AIC `C`、`F_AIC` | `C / F_AIC` | 非 AIC 行或频率无效 | 用于观察本行 AIC 周期对应的时间 |
| `aic_total_cycles` | AIC 行总周期数 | AIC | cycle | AIC `C` | 直接输出 `C` | 非 AIC 行 | L2 计数字段不以该值为分母 |
| `aic_read_close_hit` | AIC 近端读命中计数 | AIC | 次 | AIC 近端读命中次数 | 直接输出 AIC 近端读命中次数 | 非 AIC 行或近端读命中次数缺失 | 与其他五个 AIC 读计数组合分析 |
| `aic_read_close_miss` | AIC 近端读未命中计数 | AIC | 次 | AIC 近端读未命中次数 | 直接输出 AIC 近端读未命中次数 | 非 AIC 行或近端读未命中次数缺失 | 与其他五个 AIC 读计数组合分析 |
| `aic_read_close_victim` | AIC 近端读 Victim 计数 | AIC | 次 | AIC 近端读 Victim 次数 | 直接输出 AIC 近端读 Victim 次数 | 非 AIC 行或近端读 Victim 次数缺失 | 计入读请求总数，不计入命中数 |
| `aic_read_far_hit` | AIC 远端读命中计数 | AIC | 次 | AIC 远端读命中次数 | 直接输出 AIC 远端读命中次数 | 非 AIC 行或远端读命中次数缺失 | 与近端命中共同计入读命中数 |
| `aic_read_far_miss` | AIC 远端读未命中计数 | AIC | 次 | AIC 远端读未命中次数 | 直接输出 AIC 远端读未命中次数 | 非 AIC 行或远端读未命中次数缺失 | 与其他五个 AIC 读计数组合分析 |
| `aic_read_far_victim` | AIC 远端读 Victim 计数 | AIC | 次 | AIC 远端读 Victim 次数 | 直接输出 AIC 远端读 Victim 次数 | 非 AIC 行或远端读 Victim 次数缺失 | 计入读请求总数，不计入命中数 |
| `aic_read_hit_rate(%)` | AIC L2 读命中率 | AIC | % | AIC 近端和远端的读命中、未命中及 Victim 次数 | `100 * (近端读命中次数 + 远端读命中次数) / 六类读计数总和` | 非 AIC 行或任一计算所需计数缺失 | 六类读计数总和为零时输出 0 |
| `aic_write_close_hit` | AIC 近端写命中计数 | AIC | 次 | AIC 近端写命中次数 | 直接输出 AIC 近端写命中次数 | 非 AIC 行或近端写命中次数缺失 | 与其他五个 AIC 写计数组合分析 |
| `aic_write_close_miss` | AIC 近端写未命中计数 | AIC | 次 | AIC 近端写未命中次数 | 直接输出 AIC 近端写未命中次数 | 非 AIC 行或近端写未命中次数缺失 | 与其他五个 AIC 写计数组合分析 |
| `aic_write_close_victim` | AIC 近端写 Victim 计数 | AIC | 次 | AIC 近端写 Victim 次数 | 直接输出 AIC 近端写 Victim 次数 | 非 AIC 行或近端写 Victim 次数缺失 | 计入写请求总数，不计入命中数 |
| `aic_write_far_hit` | AIC 远端写命中计数 | AIC | 次 | AIC 远端写命中次数 | 直接输出 AIC 远端写命中次数 | 非 AIC 行或远端写命中次数缺失 | 与近端命中共同计入写命中数 |
| `aic_write_far_miss` | AIC 远端写未命中计数 | AIC | 次 | AIC 远端写未命中次数 | 直接输出 AIC 远端写未命中次数 | 非 AIC 行或远端写未命中次数缺失 | 与其他五个 AIC 写计数组合分析 |
| `aic_write_far_victim` | AIC 远端写 Victim 计数 | AIC | 次 | AIC 远端写 Victim 次数 | 直接输出 AIC 远端写 Victim 次数 | 非 AIC 行或远端写 Victim 次数缺失 | 计入写请求总数，不计入命中数 |
| `aic_write_hit_rate(%)` | AIC L2 写命中率 | AIC | % | AIC 近端和远端的写命中、未命中及 Victim 次数 | `100 * (近端写命中次数 + 远端写命中次数) / 六类写计数总和` | 非 AIC 行或任一计算所需计数缺失 | 六类写计数总和为零时输出 0 |
| `aiv_time(us)` | AIV 行总执行时长 | AIV | us | AIV `C`、`F_AIV` | `C / F_AIV` | 非 AIV 行或频率无效 | 用于观察本行 AIV 周期对应的时间 |
| `aiv_total_cycles` | AIV 行总周期数 | AIV | cycle | AIV `C` | 直接输出 `C` | 非 AIV 行 | L2 计数字段不以该值为分母 |
| `aiv_read_close_hit` | AIV 近端读命中计数 | AIV | 次 | AIV 近端读命中次数 | 直接输出 AIV 近端读命中次数 | 非 AIV 行或近端读命中次数缺失 | 与其他五个 AIV 读计数组合分析 |
| `aiv_read_close_miss` | AIV 近端读未命中计数 | AIV | 次 | AIV 近端读未命中次数 | 直接输出 AIV 近端读未命中次数 | 非 AIV 行或近端读未命中次数缺失 | 与其他五个 AIV 读计数组合分析 |
| `aiv_read_close_victim` | AIV 近端读 Victim 计数 | AIV | 次 | AIV 近端读 Victim 次数 | 直接输出 AIV 近端读 Victim 次数 | 非 AIV 行或近端读 Victim 次数缺失 | 计入读请求总数，不计入命中数 |
| `aiv_read_far_hit` | AIV 远端读命中计数 | AIV | 次 | AIV 远端读命中次数 | 直接输出 AIV 远端读命中次数 | 非 AIV 行或远端读命中次数缺失 | 与近端命中共同计入读命中数 |
| `aiv_read_far_miss` | AIV 远端读未命中计数 | AIV | 次 | AIV 远端读未命中次数 | 直接输出 AIV 远端读未命中次数 | 非 AIV 行或远端读未命中次数缺失 | 与其他五个 AIV 读计数组合分析 |
| `aiv_read_far_victim` | AIV 远端读 Victim 计数 | AIV | 次 | AIV 远端读 Victim 次数 | 直接输出 AIV 远端读 Victim 次数 | 非 AIV 行或远端读 Victim 次数缺失 | 计入读请求总数，不计入命中数 |
| `aiv_read_hit_rate(%)` | AIV L2 读命中率 | AIV | % | AIV 近端和远端的读命中、未命中及 Victim 次数 | `100 * (近端读命中次数 + 远端读命中次数) / 六类读计数总和` | 非 AIV 行或任一计算所需计数缺失 | 六类读计数总和为零时输出 0 |
| `aiv_write_close_hit` | AIV 近端写命中计数 | AIV | 次 | AIV 近端写命中次数 | 直接输出 AIV 近端写命中次数 | 非 AIV 行或近端写命中次数缺失 | 与其他五个 AIV 写计数组合分析 |
| `aiv_write_close_miss` | AIV 近端写未命中计数 | AIV | 次 | AIV 近端写未命中次数 | 直接输出 AIV 近端写未命中次数 | 非 AIV 行或近端写未命中次数缺失 | 与其他五个 AIV 写计数组合分析 |
| `aiv_write_close_victim` | AIV 近端写 Victim 计数 | AIV | 次 | AIV 近端写 Victim 次数 | 直接输出 AIV 近端写 Victim 次数 | 非 AIV 行或近端写 Victim 次数缺失 | 计入写请求总数，不计入命中数 |
| `aiv_write_far_hit` | AIV 远端写命中计数 | AIV | 次 | AIV 远端写命中次数 | 直接输出 AIV 远端写命中次数 | 非 AIV 行或远端写命中次数缺失 | 与近端命中共同计入写命中数 |
| `aiv_write_far_miss` | AIV 远端写未命中计数 | AIV | 次 | AIV 远端写未命中次数 | 直接输出 AIV 远端写未命中次数 | 非 AIV 行或远端写未命中次数缺失 | 与其他五个 AIV 写计数组合分析 |
| `aiv_write_far_victim` | AIV 远端写 Victim 计数 | AIV | 次 | AIV 远端写 Victim 次数 | 直接输出 AIV 远端写 Victim 次数 | 非 AIV 行或远端写 Victim 次数缺失 | 计入写请求总数，不计入命中数 |
| `aiv_write_hit_rate(%)` | AIV L2 写命中率 | AIV | % | AIV 近端和远端的写命中、未命中及 Victim 次数 | `100 * (近端写命中次数 + 远端写命中次数) / 六类写计数总和` | 非 AIV 行或任一计算所需计数缺失 | 六类写计数总和为零时输出 0 |

## 使用要点

- 命中率的分母包含命中、未命中和 Victim 六类计数，不能只使用命中与未命中两类计数重新计算。
- AIC 与 AIV 使用相同类型的 L2 计数和公式，但必须在各自数据行内计算，不能跨 Core 类型合并。
- 当六类计数总和为零时，代码明确输出 `0`；该值与计算所需计数缺失时输出的 `NA` 含义不同。
