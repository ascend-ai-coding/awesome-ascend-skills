# Investigation 合同审计方法

本文只审计前述阶段已经记录的 Blaze 源码事实。它不发现新候选、不补搜源码、不推荐方案、不匹配场景，也不产生项目路线或可执行性结论。

## 1. 原子状态与边界

```text
coverage = out_of_scope | indexed | deep
candidate_result = found | not_found | unknown
subject_result = found | not_found | unknown
evidence_basis = {source_observed, documented, example_assembled}
evidence_status = source_observed | documented | example_assembled |
                  not_applicable | conflict | unknown | unsupported
assembly_status = complete | partial
object_readiness = ready | partial | blocked | unknown
```

- `coverage` 只表达调查深度；`deep` 可与 `partial`、`unknown` 或 `blocked` 共存。
- `evidence_basis` 是来源集合，`device_verified` 不属于 Step 2。
- `not_applicable` 必须由 `applies_when=false` 和来源支持；不能用作跳过未调查事实的标记。
- `unsupported` 只用于 Blaze 源码或官方约束明确拒绝准确对象；`not_found`、缺失目录、未调查和 `unknown` 均不是 `unsupported`。
- `object_readiness` 只能描述一个候选或证据对象的本地闭合度，不能提升为需求级支持判断。

对象本地聚合顺序：明确 conflict 或明确拒绝影响必需事实时为 `blocked`；身份、适用性、specialization、binding 或 Routing 无法判断时为 `unknown`；对象身份明确但结构或必需事实未闭合时为 `partial`；其余才为 `ready`。

## 2. Blaze 组装方案审计

每个 candidate evaluation 的记录必须能机械核验：

1. 绑定单一 concrete candidate、真实入口和明确 partition；
2. 有 concrete witness，且 `coverage=deep`、`assembly_status=complete`，或准确说明为何为 partial；
3. Kernel/Policy/BlockMmad/BlockScheduler/可选 Epilogue/TilingData/Params 成员来自同一真实调用链；
4. 适用的硬约束、TilingData/Params、Tensor API Routing、物理数据、device ABI、资源/workspace 和输出生命周期各有事实、明确限制或 unknown；
5. 任何 conflict、unknown、明确拒绝都绑定 requirement IDs 和 source refs；
6. 不使用跨 partition witness、Asset、历史项目或猜测模板参数补齐缺口。

完整 candidate 是 Step 3 的输入材料，不是“官方完整支持”结论。不同 candidate 能否覆盖同一需求、能否组合或应选哪一个，只能由 Step 3 根据精确接口和约束判断。

## 3. 依赖与物理数据审计

每个已声明 source question 必须得到下列三种之一：

1. 直接 Blaze 源码事实及其 source refs；
2. 来源绑定的已观察限制或明确拒绝；
3. `unknown`，包括已搜索边界、缺少的关系和受影响 requirement IDs。

逐项检查：

```text
TilingData -> Params -> consumer
Scheduler field -> semantics/unit/legal domain
API call -> definition/specialization/Routing
logical tensor -> physical representation -> address units
Kernel entry -> GM ABI/grid/workspace/output lifecycle
extra I/O/output dataflow -> producer/consumer/lifecycle facts
```

没有现成 host Tiling 只需如实记录；若 device 侧字段语义和合法域已闭合，它不是报告阻塞项。未知 Params/ABI/物理地址语义必须保留为 unknown，不能以示例观察值替代。

## 4. 报告完整性审计

关闭报告前确认：

- 需求投影、demand partitions、hard constraints 和 source questions 可追溯到用户需求；
- 每个 candidate/subject 的 ID、适用范围、状态和来源唯一且可回溯；
- 已读取根、未读范围、候选扩展和停止位置已记录；
- 明确限制与未闭合事实没有混用；
- `out_of_scope` 只表示未调查，不表示不支持；
- 不存在场景 ID、场景匹配、bundle、组合 option、官方支持状态、交接状态或项目路线结论；
- 若有 `supplement_scope`，其 `attempt` 仅为 1，问题是语义化且不含场景信息。

若报告中仍有影响 Step 3 判断的 unknown，保持其事实状态。报告仍可冻结；Step 3 决定是否需要该事实并发起唯一补充，不能由本审计方法把 unknown 改写成“不支持”。

## 5. Step 3 消费规则

Step 3 可以从报告读取：需求投影、候选组装方案、局部结构状态、依赖与物理/ABI事实、已观察限制、明确拒绝、未闭合事实、证据账本和读取边界。

Step 3 不能把 `indexed`、`out_of_scope`、`not_found` 或 `unknown` 当作完整方案证据；也不能要求 Step 2 预先提供项目路线、候选组合、场景信息或支持结论。若现有事实不足以判断精确需求，Step 3 只能写入一次语义化 `supplement_scope` 并返回 Step 2。
