# Dispatch / Combine 接口契约

## 公开接口

以下路径位于 [cann/ops-transformer](https://gitcode.com/cann/ops-transformer) 仓库，使用时选择与目标环境配套的分支或标签。参数、返回值、支持范围和缓冲区布局直接阅读上游文档及实现，本页保留阅读入口和核对要点。若目录迁移，按表中的接口名在仓内检索。

### 推理分发与合并

| 阅读主题 | ops-transformer 阅读入口 | 核对要点 |
| --- | --- | --- |
| 低延迟 Dispatch / Combine | `ops-transformer/mc2/common/torch_extension/moe_distribute_buffer.py` | 阅读 `MoeDistributeBuffer.low_latency_dispatch`、`low_latency_combine`，核对返回值及其消费关系，尤其是专家行序、有效行统计、量化尺度、回传索引、收发计数和展开的路由权重。 |
| 推理量化与合并权重 | `ops-transformer/mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV4.md`、`ops-transformer/mc2/moe_distribute_combine_v2/docs/aclnnMoeDistributeCombineV4.md` | 按实际调用的接口版本阅读产品支持、参数和约束，确认 `quantMode`、输出数据类型、尺度布局，以及 `topk_weights`、`expand_scales` 的用途和可省略条件。 |
| 专家部署映射与剪枝 | `ops-transformer/mc2/moe_update_expert/docs/aclnnMoeUpdateExpert.md` | 阅读 `MoeUpdateExpert` 的逻辑专家映射、部署表、输出编号和剪枝规则，核对 mask、权重与部署编号如何按 token 和 top-k 槽位对应。 |
| 分段 Dispatch | `ops-transformer/mc2/moe_distribute_dispatch_setup/docs/aclnnMoeDistributeDispatchSetup.md`、`ops-transformer/mc2/moe_distribute_dispatch_teardown/docs/aclnnMoeDistributeDispatchTeardown.md` | 核对通信模式、调用顺序、`commCmdInfoOut` 的配对关系，以及哪些输出交给后续专家计算和 Combine。 |
| 分段 Combine | `ops-transformer/mc2/moe_distribute_combine_setup/docs/aclnnMoeDistributeCombineSetup.md`、`ops-transformer/mc2/moe_distribute_combine_teardown/docs/aclnnMoeDistributeCombineTeardown.md` | 核对回传辅助信息、索引和通信命令的来源，确认结果何时可读、缓冲区何时可复用。 |

专家输出应与同轮 Dispatch 的路由信息保持对应。若专家计算重排过行，回传前恢复顺序或同步维护映射。量化尺度跟随对应的特征行及分组，路由权重跟随源 token 的 top-k 槽位，同一次权重不重复相乘。

### 训练分发与反向

阅读 `ops-transformer/mc2/common/docs/torchapi_ElasticBuffer.md`，并结合 `ops-transformer/mc2/common/torch_extension/elastic_buffer.py` 中的实现核对以下内容。

| 阅读位置 | 核对要点 |
| --- | --- |
| `ElasticBuffer.dispatch` 的输入与量化约束 | 支持的数据类型、特征与尺度的传入形式、尺度分组、形状及对齐要求。 |
| Dispatch 输出与专家行组织 | 去重粒度、各专家有效行范围、路由槽位及权重的展开方式，区分传输条目数与专家计算行数。 |
| `ElasticBuffer.combine` 与 `EPHandle` | 专家输出行序、句柄配套关系、是否对 `x` 加权，以及 `topk_weights` 是否承担按源槽位回传标量或梯度的职责。不要直接套用推理接口的加权行为。 |
| cached 路由 | 缓存复用条件、句柄内容、允许重新传入或必须省略的路由参数。 |
| 异步调用与 `current_stream_wait` | 发起流、完成阶段、返回结果的取得方式，以及发送数据和缓冲区的生存期。 |

- **分发。** 同一 token 命中多个专家时，各专家仍有独立的计算贡献。即使实现对同一目的 rank 的特征传输去重，也不能因此少算一个专家或少加一份梯度。
- **反向。** 根据前向保存的路由信息对源 token 的输出梯度 `dy` 做 Dispatch，将同一梯度分发到每个命中的专家行。若合并前计算了 `wᵢ vᵢ`，专家输出 `vᵢ` 的梯度为 `wᵢ dy`，槽位权重 `wᵢ` 的梯度为特征维上的 `dot(dy, vᵢ)`。各专家分别反传后，用 Combine 将各份输入梯度送回源 rank，在同一 token 行上相加。权重梯度按原 top-k 槽位分别回传，具体接口见上方 Combine 阅读入口。若权重在激活后应用，参阅 [SwiGLU 后乘权](../reading/mega-moe-reading-guide.md#swiglu-后乘路由权重的反向)。
- **缓存与异步。** 路由缓存只在接口允许且通信组、专家部署及 token 数量、顺序和路由均未改变时按约定复用。异步结果需按接口规定的完成方式等待后再消费，不把调用返回当成通信已完成。

## 自建 MTE 示例的成对字段

下面的索引布局只用于自建示例。公开接口的回传索引应原样交给同版本 Combine。

- `expandIdx`
- `epRecvCounts`
- `epSendCounts`
- `expandX`
- `expandScales`
- `expertTokenNums`

如果这几个对象没有成对理解，后续地址规则、状态协议和回传逻辑都容易错位。

## `expandIdx` 契约

- `expandIdx` 必须固定表达来源三元组
- dispatch 侧生成时，按有效接收条目顺序写入
- combine 侧消费时，按同一顺序恢复来源 rank、来源 token 和 topK 槽位
- 默认字段顺序固定为 `[srcRankId, srcTokenIdx, srcTopKIdx]`

## `epRecvCounts` / `epSendCounts` 契约

- `epRecvCounts` 表示 dispatch 在本卡从 EP 通信域各来源 rank 实际收到的 token 数量
- `epSendCounts` 表示 combine 在本卡需要回传给 EP 通信域各目标 rank 的 token 数量
- 两者在接口语义上完全同构，都是“按 rank 维组织的 token 数量”
- dispatch 的 `epRecvCounts[srcRank]` 在 combine 阶段对应为本卡发回给 `srcRank` 的 token 数

## `A` 与有效长度

- `A` 是接口要求预留的最大容量，不是本次一定有效的 token 数
- `expandX`、`expandIdx`、`expandScales` 都按最大容量预留
- 实际有效长度由本次真实收发统计决定
- 任何实现都不应把 `A` 直接当成 dispatch 接收量或 combine 回传量

## `expandX` 契约

- `expandX` 只承载特征载荷，不包含来源三元组
- HCCL window 中传输的若是通信 token，则回搬阶段必须拆成 `expandX` 与 `expandIdx`
- 写真实输出时，`expandX[i]` 与 `expandIdx[i]` 必须一一对应

## `expertTokenNums` 契约

- `expertTokenNums` 是状态矩阵聚合后的结果，不是等待阶段的原始真值
- 若 `expertTokenNumsType = 1`，表示数量；若为 `0`，表示前缀和语义
- 多核阶段若当前核只持有局部统计，不能直接写最终 `expertTokenNums`

## 量化与权重契约

- 量化路径下，`scales`、`dynamicScales`、`expandScales` 必须和 `quantMode` 闭环
- combine 消费 `expandScales` 时，应与有效 routed item 顺序保持同步
- `expertScales` 是 combine 聚合权重，不应与 `expandScales` 混用

## 常见误解

- 把 `expandIdx` 当成附加调试信息，而不是 combine 回传的主协议字段
- 把 `epRecvCounts` 当成等待阶段的原始状态，而不是聚合结果
- 把 `A` 当成真实 token 数，导致回搬越界或错误等待
- 把 `expandX` 的载荷和三元组混在同一输出语义里