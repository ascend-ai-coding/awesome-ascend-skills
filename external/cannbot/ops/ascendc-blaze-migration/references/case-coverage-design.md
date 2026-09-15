# 用例覆盖设计

## 目录

1. 目标与边界
2. 特征轴与约束提取
3. 属性组合推导
4. Shape 行为分类
5. 随机 Shape 生成与冻结
6. 对齐与合法非对齐
7. 属性与 Shape 交叉
8. 覆盖义务与压缩
9. Design 交付表
10. G1/G3 门禁

## 1. 目标与边界

本参考文档定义 G1 如何形成验证义务，以及 G3 如何把义务实例化为覆盖迁移算子关键合同和实现行为的具体用例。G1 不生成具体 case；G3 的正式功能性能用例总数必须为 10~30，不规定跨算子固定 Shape 数值或组合模板。

实际算子的特征、合法值、行为断点和覆盖义务在 G1 写入 `migration-design.md`；具体 Shape、随机结果、用例和覆盖结论在 G3 追加到同一文档。不能用本参考文档的示例代替目标算子的调查。

覆盖对象至少包括：

- 对外接口属性、合法和非法组合；
- dtype、format、transpose、packing 和可选 Tensor；
- M/N/K 等 Shape 行为和边界；
- 对齐、合法非对齐、tile tail 和 group tail；
- TilingKey、Kernel、fallback 和拒绝路径；
- Buffer、layout、数据搬运、同步和生命周期；
- 迁移范围矩阵中的 `contract_id`、`behavior_id` 和差异分类。

## 2. 特征轴与约束提取

G1 从以下事实源提取特征和验证义务：

1. 目标算子的对外接口文档；
2. Host Tiling、checker、binary config、TPL_SEL 和 TilingKey；
3. 活动 Kernel 及其依赖闭包；
4. 现有测试、runner 和输入构造；
5. 固定 ops-tensor checkout 的 API、组件和契约测试。

`migration-design.md` 的 G1 章节必须为每个特征记录：

| 字段 | 内容 |
|---|---|
| `feature_id` | 稳定标识 |
| `feature_name` | dtype、format、transpose、packing 等 |
| `legal_values` | 合法值或合法区间 |
| `illegal_values` | 必须拒绝的值或组合 |
| `dependencies` | 与其他属性的依赖关系 |
| `impact` | 路由、layout、数学、资源或接口校验影响 |
| `evidence` | 文档、代码、编译或运行证据 |

不能从已有测试名称反推完整支持域，也不能因为某个组合没有现成测试就静默排除。

## 3. 属性组合推导

G1 按以下顺序得到属性义务集合：

```text
候选属性组合
→ 接口约束过滤
→ Host Tiling/路由约束过滤
→ 活动实现约束过滤
→ 合法组合、非法组合和排除理由分别记录
```

以下维度只要出现在目标合同或活动路径中，就必须纳入分析：

- dtype；
- format；
- transpose；
- weight 表示和 packing；
- quant/antiquant 模式；
- scale、offset、bias 和其他可选 Tensor；
- groupSize；
- 路由、fallback 和错误处理；
- 影响 Buffer、layout、同步或数学过程的属性。

关键离散属性组合原则上形成完整义务。只有能够证明两个维度在当前活动路径中语义独立，才允许等价类压缩；压缩理由必须记录在 G1 覆盖义务表中。

非法组合必须形成独立拒绝义务，G3 再实例化为辅助非法用例；只有对外合同明确规定该组合不可构造且已有更高层拒绝证据时才允许排除，并记录拒绝阶段和证据。

## 4. Shape 行为分类

不规定跨算子通用的数值挡位。小、中、大只是算子专属的行为分类，边界必须从当前算子的事实中推导。

候选行为断点包括：

- 最小和最大合法 Shape；
- tile 边界及 `tile-1 / tile / tile+1`；
- groupSize 边界和 group tail；
- block/core 分配或多核切换；
- Buffer 数量、ping-pong 或并发度切换；
- workspace 或 Tiling 参数切换；
- TilingKey、Kernel 或 fallback 切换；
- 设备资源压力或大规模调度路径。

迁移工程可使用小、中、大名称，但必须在 design 中给出算子专属定义。例如：

| 类别 | 行为依据 |
|---|---|
| 小规模 | 最小合法工作量、单 tile、单轮或最少资源路径 |
| 中规模 | 多 tile、稳定 Buffer 复用和常规调度 |
| 大规模 | 多轮、多核、资源压力或大 workspace 路径 |

如果算子没有清晰的三类行为，应记录实际分类和原因，不得机械套用。

## 5. 随机 Shape 生成与冻结

随机只在 G3 用于生成具体数值，不用于决定 G1 覆盖结构。

G1 先冻结：

- 属性组合；
- Shape 行为类别；
- 对齐或合法非对齐目标；
- tile、group、Buffer、路由和行为覆盖义务。

G3 再在合法约束内随机生成 M/N/K 或其他 Shape 参数。每条用例必须记录：

- `size_class`；
- `shape_constraints`；
- `shape_seed`；
- `generator_version`；
- `resample_count`；
- 最终 Shape；
- 实际 tile、tail、group 和路由；
- 输入资产 SHA256。

随机候选不满足合同或目标行为时允许重抽，但必须记录原因和次数。不得为了绕过失败、缩短执行或改善性能而重新选择 Shape。

G3.3 后不得重新随机。G4 和 G5 必须读取同一冻结 Shape、seed 和输入文件。

## 6. 对齐与合法非对齐

分别分析 M、N、K 或算子实际相关维度的：

- tile 对齐；
- groupSize 对齐；
- packing 对齐；
- scale/offset Shape 对齐；
- stride、offset 和搬运对齐；
- 接口合同规定的必要对齐。

“非对齐”必须是合法非对齐：可以打破内部 tile 对齐，但不能破坏接口、packing、groupSize、scale/offset Shape 或地址解释合同。

每条 Shape 用例记录：

```text
dimension
alignment_basis
aligned_or_legal_nonaligned
tile_tail
group_tail
packing_state
legality_evidence
```

## 7. 属性与 Shape 交叉

不要求所有属性组合与所有 Shape 组合形成完整笛卡尔积。以下情况必须交叉覆盖：

- 属性改变 TilingKey、Kernel 或 fallback；
- 属性改变 layout、地址、packing 或 stride；
- 属性改变 Buffer、同步或生命周期；
- 属性改变数学、累加、舍入或转换；
- 属性与 M/N/K 规模存在源代码分支；
- 属性与 tail、group 或多核行为存在耦合。

只有明确证明实现行为等价时，才能共享 Shape 行为类别或具体 Shape。证明必须说明相同的路由、layout、资源、同步和数学语义。

## 8. 覆盖义务与压缩

G1 的 `migration-design.md` 必须维护验证义务表：

| obligation_id | 属性条件 | Shape 类别 | 对齐条件 | 路由 | behavior_id | 状态 | 压缩依据 |
|---|---|---|---|---|---|---|---|

G3 必须另建 obligation-to-case 映射，将每条覆盖义务绑定至少一个具体 `case_id`，或者绑定可审计的合同排除证据。`uncovered_contract_obligations` 不能由人工直接填写为空，必须由两张表共同推导。

覆盖统计至少包括：

- 合法属性组合总数、已覆盖数和缺失数；
- 每个属性值出现次数；
- 每个 Shape 行为类别覆盖数；
- 对齐和合法非对齐覆盖数；
- tile/group/tail 覆盖数；
- 路由、fallback 和拒绝路径覆盖数；
- `contract_id`、`behavior_id` 覆盖数；
- 未覆盖义务和压缩义务。

正式功能性能用例数量必须为 10~30，但数量合规不能替代覆盖统计。非法拒绝等辅助用例不计入该数量，除非接口合同明确要求其性能属性。

## 9. Design 交付表

G3 必须在 `migration-design.md` 追加以下逐 case 表格：

1. 冻结用例总表：记录 case、属性、Shape、场景、覆盖关系、路由、输出、基线比较规则、逐字节规则和性能属性；
2. 输入与执行明细表：记录参数顺序、最终 Shape、stride/offset/packing、seed、输入资产、预期路由、执行与重置规则；
3. 特征约束表：记录特征轴、合法/非法值、依赖、影响和证据；
4. 覆盖义务表：记录 obligation 与 case 的绑定、状态和压缩依据；
5. 覆盖统计表：记录属性、Shape 行为、对齐、路由和行为覆盖结果。

每个正式用例同时具有功能和性能验证要求；“性能”不能是脱离功能合同的独立空壳 case。外部 `validation/cases/`、runner 注册表和执行清单不能替代 design 中的逐 case 设计表。

## 10. G1/G3 门禁

G1 必须完成：

- 特征轴和约束恢复；
- 合法/非法属性组合推导；
- Shape 行为类别和关键断点识别；
- 对齐/合法非对齐覆盖设计；
- 随机生成约束和冻结字段设计；
- 覆盖义务、压缩依据和 runner 合同计划；
- 不生成具体 Shape、seed、输入、case 或 runner。

G1 关闭前必须确认：

```text
算子合法域已恢复
→ 关键属性组合已覆盖或有排除证据
→ Shape 行为类别和断点已覆盖
→ 对齐/合法非对齐义务闭合
→ contract_id/behavior_id/路由覆盖闭合
→ 每条义务可以在 G3 被具体化和判定
```

G3 必须确认：

```text
正式功能性能用例数量在 10~30
→ G1 每条义务绑定具体 case 或审计排除项
→ 最终 Shape、seed 和输入已冻结
→ 每个正式用例同时定义功能和性能要求
→ design、case、runner、G4/G5 执行清单集合一致
```

存在未解释义务缺口、无依据压缩、不可重现随机、缺失最终 Shape、按 Blaze 实现删改用例或集合不一致时，G3 不得关闭。
