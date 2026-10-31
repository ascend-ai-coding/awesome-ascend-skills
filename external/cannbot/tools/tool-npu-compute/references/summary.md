# summary.jsonl 字段说明

`summary.jsonl` 按行保存 JSON 对象，用于查看所选 PMU Section 的汇总指标以及当前 Kernel 的概览信息。使用[summary 检查脚本](../scripts/inspect_summary.py)检查文件结构和数值类型。

## 记录组成

- 每个所选 PMU Section 对应一条汇总记录，`category` 是 Section 名称。
- Section 记录的指标字段与同名 CSV 一致，但不包含 `block_id` 和 `sub_block_id`。
- 最后一条记录固定为 `OpInfoSummary`。
- 同一个 `category` 在一个文件中只出现一次。

Section 记录中的每个指标分别对有效的 Task 级数值计算算术平均值。某个指标没有有效数值时为 `null`；不同指标的有效样本数可能不同，因此不能用汇总值还原各 Block 或 Sub-block 的明细。各指标的含义、单位和不可用条件以对应 Section 字段说明为准。

## OpInfoSummary 字段

| 字段 | 含义 | 单位 | 类型 | 可为 `null` 的条件 | 分析说明 |
|---|---|---|---|---|---|
| `category` | 记录类别 | 无 | string | 不允许为 `null` | 固定为 `OpInfoSummary` |
| `Op Name` | 当前 Kernel 的可读名称 | 无 | string/null | 未获得 Kernel 名称元数据 | 用于标识当前汇总对应的 Kernel |
| `Op Type` | 由 Task 级 PMU 记录中的 Core 类型判定的算子类型 | 无 | string/null | 没有可识别的 AIC 或 AIV Task 级 PMU 记录 | `cube` 表示仅 AIC，`vector` 表示仅 AIV，`mix` 表示同时包含 AIC 和 AIV |
| `Task Duration(us)` | 当前 Kernel 对应 Task 的执行时长 | us | number/null | 缺少有效 Task Log 时间区间 | 用于观察 Kernel 的整体执行时长 |
| `Block Dim` | Kernel 启动时配置的 Block 数量 | 个 | integer/null | 未获得 Block Dim 元数据 | 用于理解 Kernel 的启动规模 |
| `Mix Block Dim` | 混合 Kernel 的 Block 数量扩展字段 | 个 | integer/null | 当前结果未提供该值 | 当前输出为 `null`，不用于推断混合 Kernel 的实际 Block 数量 |
| `Device Id` | 当前 Kernel 所在 Device 的逻辑编号 | 无 | integer/null | 未获得 Device 元数据 | 用于区分多 Device 场景中的采集对象 |
| `Pid` | 执行当前 Kernel 的目标进程 ID | 无 | integer | 不允许为 `null` | 用于关联目标进程 |
| `Current Freq` | 计算指标时使用的当前 Core 频率 | MHz | number/null | 频率无效，或混合 Kernel 的 AIC 与 AIV 当前频率不一致 | 用于理解周期到时间的换算条件 |
| `Rated Freq` | 硬件信息中记录的额定 Core 频率 | MHz | number/null | 额定频率缺失，或混合 Kernel 的 AIC 与 AIV 额定频率不一致 | 用于比较当前频率与额定频率 |
| `aicore_parallel_utilization` | 各 Block 的 AIC Cube 与 AIV Vector 利用率之和的平均值 | 比值 | number/null | 未选择 `PipeUtilization`，或缺少可用的流水线利用率数据 | 反映参与统计的 Block 的平均计算利用程度 |
| `aicore_parallel_balance` | 根据各 Block 执行时长离散程度计算的并行均衡度 | 比值 | number/null | 缺少有效 Block 执行时长或平均时长为零 | 值越接近 1 表示参与统计的 Block 时长越均衡 |
| `aicore_gm_bw_theoretical(GB/s)` | 计算 GM 带宽使用率时采用的理论带宽 | GB/s | integer/null | 未选择 `Memory` | 当前值为 1600 GB/s |
| `aicore_gm_read_bw(GB/s)` | `Memory` 汇总中的 AIC 与 AIV 主存读取带宽之和 | GB/s | number/null | 未选择 `Memory`，或 AIC 与 AIV 主存读取带宽均不可用 | 反映当前 Kernel 的汇总 GM 读取带宽 |
| `aicore_gm_write_bw(GB/s)` | `Memory` 汇总中的 AIC 与 AIV 主存写入带宽之和 | GB/s | number/null | 未选择 `Memory`，或 AIC 与 AIV 主存写入带宽均不可用 | 反映当前 Kernel 的汇总 GM 写入带宽 |
| `aicore_gm_bw_usage_rate(%)` | GM 读写带宽之和占理论带宽的百分比 | % | number/null | GM 读取或写入带宽不可用 | 按 `(读取带宽 + 写入带宽) / 1600 × 100` 计算 |

## 分析方式

1. 先运行检查脚本，确认记录顺序、字段和类型符合目录定义。
2. 使用 Section 汇总值观察 Kernel 整体特征，使用同名 CSV 分析 Block 和 Sub-block 明细。
3. `null` 表示当前字段没有可用数值，不按零参与比较或计算。
4. 比较多次采集时，先确认 `Op Name`、`Op Type`、`Block Dim`、Device 和频率条件可比。

```bash
python3 scripts/inspect_summary.py <summary.jsonl> --format markdown
```
