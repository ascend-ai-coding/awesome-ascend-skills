# kernel_details 字段查法

多流分析和调试都要从 profiling 的 `kernel_details.csv` 精准取少量字段——填 `resource_hint`、判争核、算 overlap、解释拖尾。原则是只取必要的几条 op、几个字段，不把整份明细搬进上下文。

## 什么时候下探

确认某个算子的具体细节时，例如：

- `Stream ID` / `Task ID` 是否符合方案设计；
- shape 是否引入额外 `TransData` / `BroadcastTo` / `MemSet`；
- `Block Dim` / `Mix Block Dim` 是否导致主副流争核；
- `aic_*_ratio` / `aiv_*_ratio` 是否显示 cube / vector 长尾；
- 等待或拖尾是否来自某个具体算子。

## 字段集：设计 vs post-mortem 是两套

**跨流候选的"设计"和"落地复盘"用不同字段集，不要混用。** 仅看设计字段集会漏掉 `Start Time`，从而判断不了 overlap 是真是假——两段时间线的 `Start Time` 重叠程度才是 overlap 是否成立的唯一证据。

| 排查目标 | 推荐字段 |
| --- | --- |
| 跨流候选**设计**（dim 阈值、Stream 配对、填 resource_hint） | `Stream ID,Task ID,Name,Input Shapes,Output Shapes,Duration(us),Wait Time(us)` |
| 跨流候选 **post-mortem**（验证 overlap 是否真成立，必查） | `Stream ID,Name,Start Time(us),Duration(us),Wait Time(us),Input Shapes` —— 用 `Start Time` + `Duration` 重建甘特图，按 [`timeline-overlap-check.md`](timeline-overlap-check.md) 算 `overlap_pct` |
| 多流后 cube 是否打满 | `Name,Duration(us),aicore_time(us),aic_mac_ratio,aic_mte2_ratio,cube_utilization(%)` |
| Vector 长尾 / 搬运瓶颈 | `Name,Duration(us),aiv_time(us),aiv_vec_ratio,aiv_mte2_ratio,aiv_mte3_ratio` |
| 控核场景下 Block Dim | `Name,Duration(us),Block Dim,Mix Block Dim,Accelerator Core` |

## 怎么取

直接从 `kernel_details.csv` 读需要的几行几列。例如用 pandas：

```python
import pandas as pd
df = pd.read_csv("<kernel_details.csv>")
print(list(df.columns))                       # 先看有哪些列，避免猜列名

cols = ["Stream ID", "Name", "Start Time(us)", "Duration(us)"]
print(df.loc[df["Name"].str.contains("xxx", na=False), cols])   # 按算子名/索引取需要的几行
print(df.iloc[<row_index_list>][cols])                          # 或按行号取
```

只取必要的几条 op、几个字段，别把整张表灌进上下文。

## 关键约束

- 不要把整份 `kernel_details.csv` 或 per-op 明细灌进上下文，只取需要的几行几列。
- 先 `print(df.columns)` 确认列名，不要猜。
- 判主 / 副是否争 cube 用 `Block Dim` 加和 vs 设备核数，不要用"算子类型是 MatMul / Vector"二元推断（MatMul 也能 blk=1 占很少核）。
- overlap 只能在**同一次采集内**用 `Start Time` 计算；不同采集的 `Start Time` 基准不同、不可比。
