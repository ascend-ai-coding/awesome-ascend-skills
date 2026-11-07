+# 排序、TopK 与变长输出

本 archetype 是路由层，覆盖 moe_gating_top_k_softmax、nms、top_k、unique。它不是完整实现手册；实施前必须继续读任务合同、DESIGN/PLAN 模板、golden-generation、review-checklist，以及 source-map 中对应参考实现的路线审计结论。

## 路线选择

| 场景 | 执行单元 | 说明 |
|---|---|---|
| 小 k、局部 TopK | AIV | selection/insertion/heap 策略 |
| 全量排序或大 k | AIC/AIV 或分段排序 | 先分段，再跨核 merge |
| 去重 + inverse | AIV | hash/sort + compaction |
| NMS | AIV | score 排序 + IoU 抑制 + 变长输出 |

## 关键设计点

1. comparator 必须覆盖输入 dtype：有符号 int8/int32/int64、无符号 uint8、fp16/bf16/fp32，NaN/Inf 规则显式冻结。
2. tie-break 是合同的一部分：相同值按原始索引升序或降序，不得依赖 atomic 或核调度顺序。
3. 任意 dim 的 TopK 先做维度合并/分解，dim 可为负；输出 value dtype 与输入一致，idx 固定 int64。
4. 变长输出模式：预分配上限，kernel 写 count，host 只做 view/narrow；核心计算不得拆到第二个计算 kernel。
5. unique 冻结输出顺序（first occurrence 或 sorted）、return_inverse、ND flatten 语义和空输入。
6. NMS 冻结 box 坐标语义、IoU 除零、分数排序、阈值开闭区间和保留索引顺序。

## 验证顺序

1. k=1、k=dim_size、重复值、全等值、负值、NaN/Inf。
2. 任意 rank、dim=0/中间维/负 dim、跨核分段与 merge。
3. 变长输出覆盖空结果、全保留、重复输入和 count 正确性。
4. NMS 用重叠/相离/包含/除零 box 固化判定。

## 事实源

- ops-nn：index/unique、index/top_k_top_p_sample
- ops-transformer：moe/moe_gating_top_k_softmax、mc2/engram_fetch_* 的 sort/unique 组件
- cann-samples：Samples/2_Performance/moe_init_routing_story

L3 映射：moe_gating_top_k_softmax、nms、top_k、unique。
