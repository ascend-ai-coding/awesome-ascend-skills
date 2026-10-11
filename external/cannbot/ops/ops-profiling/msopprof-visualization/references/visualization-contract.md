# 可视化模块合同

## 通用规则

1. 所有页面使用同一个 `report_payload.json`；
2. 数据解析与页面渲染分离；
3. `0` 是有效数据，`NA` 表示缺失；
4. 不得通过视觉插值补齐不存在的数据；
5. 页面宽度自适应，不依赖固定大屏宽度；
6. 报告必须离线可打开，不依赖远程资源。

## Details

包含：

- 基本信息和各 Block duration；
- Core/Subcore Occupancy；
- Compute pipeline、breakdown、wait/conflict 和 counters；
- Memory workload topology、带宽、请求数和原生表；
- profiler advice。

Occupancy 的异常值颜色表示统计偏离，不表示性能结论。必须保留原始 core/subcore 映射。

## Memory topology

执行绑定优先级：

```text
table_per_block[i] <-> core_memory_map[i]
```

`core_no` 是物理核元数据，可能在 dispatch 中重复，不能作为 Block ID 选择键。

每条边必须包含方向、数值、Block 和 dispatch 绑定。相反方向是两条独立边。边的 label 必须绑定到固定 `edge_id`，不得使用“最近连线”推断。

## Roofline

必须保留：

- 所有有效 chart group；
- 所有实测 point；
- 带宽、算力上限、ridge point；
- 算术强度和实测性能；
- pointer crosshair、point hover 和 curve hover。

不得过滤未知但合法的图表名称。

## Timeline

必须保留原始事件开始时间、持续时间、PID/TID/lane 和 args。交互包含：

- 缩放、适配、重置；
- 范围选择；
- Slice Detail；
- Slice List；
- event count 与 payload 一致。

## Cache

必须展示真实 Hit/Miss 计数或比率、Block 和 cache family。没有 cache event cell 时显示明确空状态，不能复制同一 Block 数据填满热力图。

## Source

必须建立：

- file；
- line；
- instruction；
- address/opcode/pipe；
- line-to-instruction 与 instruction-to-line 关系；
- 可用时的 stall/GPR 状态。

Source 页面不得只显示头文件，也不得只提取二进制 printable strings。

## On-Chip Memory

仅使用 `memory_info.json`，展示 allocation、地址、lifetime、bank/group 和执行顺序。缺失时输出诊断页。

## 不可用页面

诊断页至少包含：

- 模块名称；
- 不可用原因；
- 当前证据；
- 需要补充的 artifact 或采集动作。
