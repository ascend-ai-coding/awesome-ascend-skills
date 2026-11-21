---
type: "PyPTO Performance Optimization Card"
title: "发布粒度按 (核数, 每项工作量, 组数) 扫描"
description: "发布粒度同时决定消费者追赶速度与握手密度；同一粒度在不同核数、每项工作量和组数下可能反向。把粒度做成可扫描参数，先小范围扫 1/2/4，再把最优值固化为 host 规则。"
status: "stable"
tags: ["pypto-pro", "pipeline", "publish-granularity", "tuning"]
item_id: "pipe-10"
bound_hint: "PIPELINE"
applicability: "多阶段或 C/V kernel 用固定发布粒度，粒度是运行时标量或编译期常量；消费者按完成计数追赶生产者；不同核数/每项工作量/组数下，握手成本与追赶收益的平衡会翻转"
target_api_gate: "仅限 Ascend 950PR 或 950DT；发布粒度须能在不重编译的情况下扫描（运行时 scalar）或按 TilingKey 分档；发布计数与 cross_core/sync_all 事件须已核验"
---

# 技术卡片 pipe-10：发布粒度按 (核数, 每项工作量, 组数) 扫描

- **适用 bound**：PIPELINE
- **一句话**：把发布粒度当作随核数、每项工作量和组数变化的调参维度，先小范围扫描，再固化 host 规则；不要用“越大越好”或“越细越好”的单一规则。

## 何时用（诊断特征）

- 存在 `publish_items`（等价于槽位容量 `slot_cap`）这类粒度参数，决定一次跨核通知覆盖多少 item。
- 消费者按“已完成 item 数 / 发布粒度”追赶生产者：粒度越细，启动越早；每个发布单元的握手/屏障越多。
- 同一粒度在不同核数、每项工作量或组数下表现不同，甚至方向相反。
- 当前 host 规则只按单一维度（如输出宽度）决定粒度，未区分核数与组数。

## 何时不适用

- 粒度由消费者块形状固定（如必须等于完整 m-tile），不能独立调整。
- 握手/屏障成本远小于计算，粒度变化落在噪声内；保持默认即可。
- 粒度会影响数值正确性（如部分和必须按固定块对齐）时，只能在满足对齐的候选里扫。
- 消费者已经能按组流水，细粒度的边际收益很小，优先检查其他 bound。

## 原理

发布粒度同时决定两件事：

1. **消费者追赶速度**：完成计数的步长；越细，消费者越早开始，生产者尾段越短。
2. **握手密度**：每个发布单元的 `set/wait_cross_core`、`sync_all` 等固定开销；越细，开销越高。

两者的平衡随配置变化：核数越多，单槽工作量越少，细粒度的追赶收益更容易盖过握手；核数越少，单槽工作越多，粗粒度更稳。每项工作量（如输出宽度、归约深度）改变单个 item 的计算量与消费者启动的收益；组数决定消费者是否已经能按组流水，多组时细粒度的边际收益下降。因此最优粒度是三维函数，不存在单一全局规则。

## 怎么改（before / after）

以下为**结构示意**，只展示粒度参数与 host 规则；`env_override`/`rule_by_width`/`coarse_for_small_cores` 等为占位符，实际发布点、同步事件与设备配置须按当前算子核验，片段不可直接编译。

**before（单一规则，只按一个维度）：**

```结构示意
# 只按输出宽度决定，忽略核数与组数
publish_items = rule_by_width(width)
```

**after（先扫描，再按 (核数, 每项工作量, 组数) 固化）：**

```结构示意
# 扫描阶段：运行时 scalar，不重编译
publish_items = env_override("PUBLISH_ITEMS", rule_by_width(width))

# 固化阶段：把每个 (cores, work_per_item, groups) 的最优值写成 host 规则
if cores <= small_core_threshold:
    publish_items = coarse_for_small_cores(work_per_item, groups)
elif groups > 1 and work_per_item >= large_work_threshold:
    publish_items = mid_granularity(work_per_item, groups)
else:
    publish_items = fine_granularity(work_per_item)
```

**扫描方法：**

- 固定代码、输入、采样协议，只改 `publish_items`；粒度不参与代码生成时用运行时 scalar 扫描，参与时按 TilingKey 分档并固定其余编译条件。
- 先扫 1/2/4 三档定位方向（必要时补 8）；最优在端点时再测相邻档。
- 每个候选按固定协议采样（本项目默认 3 次预热 + 20 次采样），比较 20 次采样的均值；同时记录 min/median 与波动做稳定性核对，不要用单次最优值代替均值。
- 记录 `Task Duration`、跨核事件/屏障计数、消费者启动时刻、最差核完成时刻。
- 只保留可复现的最优；固化后在每个交付点复测，不跨核数边界外推。

## 性能与验证指标

- 指标：`Task Duration`、跨核事件/屏障计数、消费者启动延迟、最差核完成时刻。
- 正确性：每个候选都跑完整输出校验；粒度只改发布顺序时数值结果应不变；默认档作为负对照。
- 测量：在目标形态固定输入与采样协议，逐候选扫描发布粒度并 A/B 复测；同一粒度在不同核数与组数下方向可能相反。

## 技术限制与风险

- 粒度若不参与代码生成，优先用运行时 scalar 扫描；进入 tiling key 会增加编译变体，固化后再决定是否折叠。
- 细粒度会增加 `pl.system.set/wait_cross_core`/`sync_all` 次数；屏障成本高的平台上可能整体回退。
- 消费者就绪条件不同：若消费者只认完整块，仅靠细粒度收益有限，需要配合 `pipe-08` 的子块部分和。
- 规则固化后仍要在每个交付点复测；核数/形状边界（如 24/28/32 核）不要线性外推。
- 与相邻卡片：`pipe-08` 改消费者就绪粒度和部分和保留；本卡只调发布粒度，不改变“整块就绪”的条件。`sync-01` 选择同步原语；本卡在既定原语下调粒度。`pipe-02`/`pipe-03` 重排或交错阶段，不改变发布粒度；`pipe-09` 移动跨迭代装载，本卡调发布粒度，二者可独立使用。

## 参考资料

- 相关卡片：`pipe-08`（子块部分和）、`sync-01`（同步原语选型）。
- 验证案例：`ops/pro/kimi_k3/mega_recurrent_kda_replayssm`（`DESIGN.md` §5.2、`KEY_M3_K` 取 1/2/4/8）。该算子的实测差异为：16/24 AIC 用 2 比 1 快 5–6.5%，N=12/24 用 1 因握手过密慢约 5%/11%，8 核 N=6 多组细 slot 慢约 7%；属于该算子实例，不作通用最优值。
