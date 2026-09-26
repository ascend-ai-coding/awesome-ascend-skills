# 指标通用规则

本文说明七种 `npu-compute` PMU Section CSV 共同使用的数据语义。各 Section 特有的计数含义和公式在对应的 Section 参考文档中说明。

## 数据模型

每个输出行来自一条 PMU 聚合记录，其键包含 `blockId`、`subBlockId`、`coreType` 和 `coreId`。CSV 使用以下字段标识数据行：

| 字段 | 当前含义 |
|---|---|
| `block_id` | PMU 聚合键中的 `blockId`。 |
| `sub_block_id` | AIC 行为 `cube` 加 `subBlockId`，AIV 行为 `vector` 加 `subBlockId`。 |

`coreId` 是内部 PMU 键的组成部分，但不会拼接到 `sub_block_id`。一行通常只提供本行 Core 类型对应的指标，因此另一种 Core 类型的字段通常为 `NA`。

公式使用以下符号：

| 符号 | 含义 |
|---|---|
| `C` | 当前 Core 类型的 `totalCycles`。 |
| `F_AIC`、`F_AIV` | 对应 Core 的有效频率，单位为 MHz。AIC/AIV 专用频率为正数时使用专用频率，否则使用公共配置频率。 |
| `D_row` | 当前 Core 数据行的时长，单位为微秒，计算公式为 `C / F`。 |
| `D_op` | 为 AIC/AIV 混合 Kernel 选择的有效 Task Log 操作时长。 |

## 公共字段

七种 PMU Section CSV 都包含相同的行标识和时长概念，但字段的准确顺序由各 Section 的表头定义。

| 字段 | 公式或表示方式 |
|---|---|
| `aic_time(us)` | AIC `totalCycles / F_AIC`；本行没有 AIC 数据或频率无效时为 `NA`。 |
| `aic_total_cycles` | AIC `totalCycles`，按整数格式输出；本行没有 AIC 数据时为 `NA`。 |
| `aiv_time(us)` | AIV `totalCycles / F_AIV`；本行没有 AIV 数据或频率无效时为 `NA`。 |
| `aiv_total_cycles` | AIV `totalCycles`，按整数格式输出；本行没有 AIV 数据时为 `NA`。 |

1 MHz 等于每微秒一个周期，因此周期数除以 MHz 后的单位为微秒。

## 公共计算规则

### 比值和百分比

原始比值的计算公式为：

```text
ratio = numerator / denominator
```

分子或分母所需计数缺失，或者分母为零时，结果不可用。名称含 `ratio` 的字段不会乘以 100。

百分比字段会乘以 100：

```text
percentage = 100 * numerator / denominator
```

部分带宽利用率字段会先使用配置的最大带宽限制观测带宽，再计算百分比：

```text
usage(%) = 100 * min(observed_bandwidth, maximum_bandwidth) / maximum_bandwidth
```

### 时长

流水活跃时长的计算公式为：

```text
active_time_us = active_cycles / F
```

活跃周期计数缺失或频率不是正数时，结果不可用。

### 数据量

数据量使用二进制 KB：

```text
data_KB = transfer_count * bytes_per_transfer / 1024
```

传输计数的含义和每次传输对应的字节数由具体字段定义。

### 带宽

带宽使用二进制 GB，时长单位为微秒：

```text
bandwidth_GB_s = bytes * 1,000,000 / 1024^3 / duration_us
```

普通的逐行带宽在已选择 `D_op` 时使用 `D_op`，否则使用 `D_row`。活跃带宽使用相关流水活跃周期除以 Core 频率得到的时长。

## 时长选择

当选定的 PMU 数据集合同时包含至少一行 AIC 数据和一行 AIV 数据时，当前实现将其识别为混合 Kernel，并按以下规则从 Task Log 计算 `D_op`：

1. 按 Replay ID 对 Task Log 分组。
2. 每组需要一条开始记录（`funcType == 0`）和一条结束记录（`funcType == 1`），且两条记录的 Task 和 Stream 标识相同。
3. 丢弃记录不完整、开始或结束记录重复、标识不一致以及结束计数不大于开始计数的数据组。
4. 使用 A5 Task Scheduler 频率将有效的 System Counter 差值转换为微秒。
5. 取所有有效 Replay 时长的中位数；有效时长数量为偶数时，取中间两个值的平均数。

`D_op` 用于计算 `Memory`、`MemoryL0` 和 `MemoryUB` 数据行。混合 Kernel 没有有效 Task Log 时长，或者 Kernel 不是混合类型时，这些计算使用对应 Core 的数据行时长。`ArithmeticUtilization`、`PipeUtilization`、`ResourceConflictRatio` 和 `L2Cache` 使用各自的逐行计算，不使用 `D_op`。

## 数据可用性和输出格式

本行不包含指标对应的 Core 类型、计算所需计数缺失、分母为零、频率或时长无效，或者所需硬件上限不可用时，指标输出为 `NA`。应结合字段定义和数据行判断具体原因，不得将 `NA` 替换为零。

有限的计算结果保留小数点后六位；计数器按整数格式输出。非有限计算结果和负数计数器值输出为 `NA`。

## 比较规则

- 比较相同 Section 中名称完全一致的字段。
- 根据 `block_id` 和 `sub_block_id` 匹配数据行，不能只按行号匹配。
- 比较数值前，先检查字段适用的 Core 和 `NA` 分布。
- 跨采集比较前，确认相关硬件环境和采集配置具有可比性。
- 分开陈述观测差异和瓶颈判断。除非指标定义本身提供限制，否则瓶颈判断需要阈值或基线。
