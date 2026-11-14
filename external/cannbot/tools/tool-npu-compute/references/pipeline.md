# Pipeline 时间线说明

`PipeTrace.json` 是 `Pipeline` Section 的时间线结果。文件中的每个事件表示一条流水线从忙碌开始到忙碌结束的完整区间。

## 顶层结构

| 字段 | JSON 类型 | 含义 |
|---|---|---|
| `displayTimeUnit` | string | 时间线查看器的展示单位，当前值为 `ns`。该值不改变 `ts` 和 `dur` 的数据单位。 |
| `profilingType` | string | 结果类型标识，当前值为 `op`。 |
| `schemaVersion` | integer | PipeTrace 数据结构版本，当前值为 `1`。 |
| `traceEvents` | array | 流水线事件数组，可以为空。 |

## 事件字段

| 字段 | JSON 类型 | 含义 |
|---|---|---|
| `cname` | string | 流水线对应的时间线颜色名称。 |
| `dur` | number | 流水线忙碌区间的持续时间，单位为微秒。 |
| `name` | string | 流水线名称。 |
| `ph` | string | 完整区间事件标识，当前值为 `X`。 |
| `pid` | string | 事件所属的来源和 Core 轨道标识。 |
| `tid` | string | 事件所属的流水线轨道，值与 `name` 一致。 |
| `ts` | number | 流水线忙碌区间的开始时间，单位为微秒。 |

事件覆盖区间为 `[ts, ts + dur]`。`ts` 和 `dur` 均应为有限非负数。事件在 JSON 数组中的位置不用于表示时间先后，分析时应使用 `ts` 排序。

## 流水线

| 流水线 | `cname` | 含义 |
|---|---|---|
| `SCALAR` | `startup` | 标量（Scalar）流水线的忙碌区间。 |
| `VECTOR` | `rail_idle` | Vector 流水线的忙碌区间。 |
| `CUBE` | `rail_response` | Cube 流水线的忙碌区间。 |
| `MTE1` | `thread_state_iowait` | MTE1 流水线的忙碌区间。 |
| `MTE2` | `yellow` | MTE2 流水线的忙碌区间。 |
| `MTE3` | `rail_animation` | MTE3 流水线的忙碌区间。 |
| `FIXP` | `thread_state_unknown` | Fixpipe 流水线的忙碌区间。 |

未出现某类流水线事件只表示该文件中没有对应的完整忙碌区间，不能据此判断该流水线不受支持。

## 轨道标识

`pid` 有两种形式：

```text
group<group_id>.<core>
process<process_ordinal>.result<result_sequence>.device<device_id>.replay<replay_id>.group<group_id>.<core>
```

只有一个来源片段时使用第一种形式。合并多个来源片段时使用第二种形式，前缀用于区分进程、结果、Device 和 Replay 来源。

`core` 的取值为：

| 取值 | 含义 |
|---|---|
| `cubecore` | AIC 轨道。 |
| `veccore0` | 第一条 AIV 轨道。 |
| `veccore1` | 第二条 AIV 轨道。 |

`group_id` 是 BIU 通道的 Group 标识。解析过程中使用的 Block 标识不会写入 `PipeTrace.json`，因此不能从 `pid` 推断 Block。

## 采样边界

Pipeline 流水图基于采样数据生成。`group_id` 的取值范围为 `0` 到 `5`，一份结果最多包含 6 个采样 Group；每个 Group 中实际出现的 Core 轨道可能是 `cubecore`、`veccore0` 或 `veccore1`。

Group 和 Core 轨道只表示 `PipeTrace.json` 中实际采样并写入事件的数据，与目标程序启用的 Core 数量没有直接对应关系。即使目标程序启用了全部 Core，结果中也最多展示 6 个采样 Group；不能根据 Group 数、Core 轨道数或缺失的轨道推断目标程序实际启用的 Core 数量。

## 摘要计算

- 事件数是 `traceEvents` 的元素数量。
- 轨道由 `pid` 和 `tid` 共同标识；轨道数是不同 `(pid, tid)` 组合的数量。
- 时间线开始时间是全部事件 `ts` 的最小值。
- 时间线结束时间是全部 `ts + dur` 的最大值。
- 时间跨度是结束时间减去开始时间。
- 流水线事件数按 `name` 统计。
- 流水线累计忙碌时间按 `name` 对 `dur` 求和。
- 采样 Group 是从事件 `pid` 中提取并去重、升序排列的 `group_id`。
- 各 Group 的 Core 轨道是该 Group 实际出现事件的 `core` 去重结果。

不同轨道或不同来源的事件可以在时间上重叠，累计忙碌时间不等于 Kernel 执行时长。合并结果中的来源前缀用于区分数据来源，不表示各 Replay 之间存在连续的全局时间轴。

## 数据边界

- 只有同时找到开始和结束的忙碌区间才会写入事件；缺少任一端点的区间会被跳过。
- `traceEvents` 为空是合法结构，表示没有写入完整区间，不表示所有流水线都处于空闲状态。
- 时间线只描述文件中已写入的完整区间，不能单独证明流水线利用率、性能瓶颈或不同轨道之间的因果关系。
