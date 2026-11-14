# Dispatch / Combine 接口契约

这个页面集中收拢 dispatch / combine 成对理解时最容易出错的中间量语义、容量定义和字段对应关系。

## 成对理解的最小集合

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