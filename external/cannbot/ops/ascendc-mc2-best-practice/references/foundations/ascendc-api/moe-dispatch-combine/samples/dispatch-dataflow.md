# MoE Dispatch 数据流

这个页面承接 dispatch 的阶段控制流、接口语义和输出闭环；不展开 window 字节布局、地址公式和 UB 大小计算。

## 最小阶段划分

| 阶段 | 输入 | 产出 | 本阶段只回答什么 |
| --- | --- | --- | --- |
| 初始化 | EP 参数、`mc2Context`、输出张量 | 上下文、数据/状态区基址、window 参数、分核边界 | 上下文和基础地址从哪里来 |
| token 发送 | `x` `expertIds` | 远端 window 中的通信 token | `dstRank/localExpert/dstSlot` 和发送地址如何求 |
| 状态写入 | 各 `dstRank x localExpert` 的发送计数 | 远端 `localExpert x sourceRank` 状态 slot，slot 内写入 `flag` 和 `tokenCount` | 何时可以发布远端 ready 状态 |
| 接收等待 | 本地状态区 | `expertRecvCounts`、`epRecvCounts`、`expertTokenNums` | 何时认为所有来源 rank 的状态都已收齐 |
| 本地回搬 | 本地 window、接收统计 | `expandX` `expandIdx`，以及按需输出统计 | 如何按接收顺序把通信 token 还原成真实输出 |

阶段附加约束：

- SHMEM 库内部同步资源在 init 阶段自动清零；用户通信窗口（含状态区）需在 host 侧 `aclrtMemset` 显式清零后再传入 kernel
- `Init()` 阶段不要手动清状态
- `epRecvCounts` 和 `expertTokenNums` 是状态矩阵读取后的聚合结果，不是等待阶段的原始真值

## 真实接口参数

### 输入参数

| 参数 | 含义 / 用意 | 逻辑维度 / shape | dtype | 关键约束 |
| --- | --- | --- | --- | --- |
| `x` | 本卡待发送的 token 特征输入 | 2D，通常为 `(Bs, H)` | `FLOAT16` / `BFLOAT16` / 其他实现指定类型 | 第 0 维与 token 维一致；第 1 维为单 token 特征长度 |
| `expertIds` | 指定每个 token 的 topK 路由目标 | 2D，通常为 `(Bs, K)` | `INT32` | `expertIds[b, k]` 给出第 `b` 个 token 的第 `k` 个路由 expert |
| `scales` | 平滑或量化路径使用的缩放参数 | 与量化/平滑路径绑定 | 路径相关 | 非量化路径可为空；量化路径必须与 `quantMode`、`expandScales` 闭环 |
| `xActiveMask` | 描述哪些 token 或路由项有效 | 1D 或 2D mask，依实现而定 | `BOOL` | 当前常见路径可不启用；启用时必须同时约束有效 routed token 数 |
| `expertScales` | 每个 token-topK 路由项对应的权重 | 通常为 `(Bs, K)` | `FLOAT` | combine 聚合权重的直接来源；dispatch 若消费，必须保持与 `expertIds` 同步 |
| `groupEp` | EP 通信域标识 | 标量属性 | - | 专家并行域 |
| `epWorldSize` | EP 域内 rank 总数 | 标量属性 | `INT64` / `INT32` | EP 域大小 |
| `epRankId` | 当前卡在 EP 域中的 rank 编号 | 标量属性 | `INT64` / `INT32` | 范围 `[0, epWorldSize)` |
| `moeExpertNum` | 全局 MoE expert 总数 | 标量属性 | `INT64` / `INT32` | 总 expert 数；与 `epWorldSize`、`sharedExpertRankNum` 一起决定路由划分 |
| `expertShardType` | 共享专家卡的部署方式 | 标量属性 | `INT64` / `INT32` | 共享专家部署方式 |
| `sharedExpertNum` | 共享专家数量 | 标量属性 | `INT64` / `INT32` | 共享专家数量 |
| `sharedExpertRankNum` | 承载共享专家的 rank 数 | 标量属性 | `INT64` / `INT32` | 共享专家使用的 rank 数 |
| `quantMode` | 通信与输出使用的量化模式 | 标量属性 | `INT64` / `INT32` | `0` 非量化；其他值需与量化输出闭环 |
| `globalBs` | EP 域视角下的全局 batch 上界 | 标量属性 | `INT64` / `INT32` | EP 域全局 batch size |
| `expertTokenNumsType` | 指定 `expertTokenNums` 输出语义 | 标量属性 | `INT64` / `INT32` | `0` 前缀和；`1` token 数量 |

### 输出参数

记号约定：

- `A` 表示本卡可能接收的最大 token 数，是接口要求预留的容量上界，不是实际收到的 token 数
- `R_actual` 表示本次 dispatch 在本卡实际收到并写入的有效 routed item 数，满足 `R_actual <= A`
- `H_expand` 表示 `expandX` 的单行存储宽度，即真正写入 `expandX` 的特征长度。对普通 dtype，`H_expand = H`；对 `fp4x2` 这类打包存储类型，`H_expand = ceil(H / 2)`

`A` 的取值约束来自接口文档：

- 共享专家场景：`A = Bs * epWorldSize * sharedExpertNum / sharedExpertRankNum`
- MoE 专家场景：
  - `globalBs = 0` 时，`A >= Bs * epWorldSize * min(localExpertNum, K)`
  - `globalBs != 0` 时，`A >= globalBs * min(localExpertNum, K)`

其中 `localExpertNum` 表示本卡专家数：

- 共享专家卡：`localExpertNum = 1`
- MoE 专家卡：`localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum)`

| 参数 | 含义 / 用意 | 逻辑维度 / shape | dtype | 关键约束 |
| --- | --- | --- | --- | --- |
| `expandX` | 保存 dispatch 后本卡接收到的展开 token 特征，供 expert 计算或 combine 输入使用 | 2D，按最大容量预留，推荐按 `(A, H_expand)` 理解 | `ExpandXOutType` | 只有前 `R_actual` 行有效；`H_expand` 不包含三元组附加信息 |
| `dynamicScales` | 量化路径下与 `expandX` 对应的动态 scale | 与量化路径绑定，按最大容量预留 | 量化类型相关 | 仅量化路径有效；有效长度与 `expandX` 已写入部分对齐 |
| `expandIdx` | 描述每个有效接收条目的来源位置，供 combine 原路返回 | 2D 逻辑上按 `(A, 3)` 预留；线性存储时长度至少为 `A * 3` | `INT32` | 每个有效条目固定三元组 `[srcEpRankId, srcTokenIndex, srcTopKIndex]`；有效条目数由实际接收量决定 |
| `expertTokenNums` | 描述本卡各 expert 实际收到的 token 数量 | 1D；按本卡 expert 维度预留 | `INT64` | `expertTokenNumsType=1` 时表示数量；`expertTokenNumsType=0` 时表示前缀和语义 |
| `epRecvCounts` | 描述本卡从 EP 通信域各来源 rank 实际收到的 token 数量 | 1D；按 EP rank 维度预留，常见为 `(epWorldSize,)` | `INT32` | 按来源 rank 维组织；combine 直接作为 `epSendCounts` 输入 |
| `expandScales` | 保存有效接收条目对应的权重或量化相关 scale | 1D；按最大容量预留，常见为 `(A)` | `FLOAT` | combine 中与有效 routed item 同步消费 |

## 输出闭环要求

- `expandX`、`expandIdx`、`expandScales` 都按最大容量预留，实际有效长度由本次 dispatch 实际接收量决定
- `expandX` 的第 1 维表示纯特征载荷宽度；附加三元组信息单独写入 `expandIdx`，不进入 `expandX`
- `expandX` 有序地记录本卡所收到的所有 token，先按专家，再按来源 rank，连续分布
- HCCL window 中传输的不是纯 `expandX` 行，而是通信 token；通信 token 由 `payload + triple` 组成，回搬阶段再拆成 `expandX` 与 `expandIdx`
- 接收侧写真实输出时，收到的 token 按接收顺序连续写入 `expandX[0..R_actual)`；对应三元组按完全相同的顺序连续写入 `expandIdx[0..R_actual)`，二者必须一一对应
- `expandIdx[i]` 必须能够唯一定位有效 `expandX` 条目的来源 `(epRankId, tokenIndex, topKIndex)`

## 通信 token 与顺序规则

dispatch 最小路径固定采用：

1. `expertId -> dstRank, localExpert`
2. 统计 `curExpertCnt`，得到 `dstSlot`
3. 组织通信 token：`Align(payload, 32B) + [srcEpRankId, srcTokenIdx, srcTopKIdx]`

## `curExpertCnt` 默认实现

默认用：`在线计数 + 向量化比较归约`

实现规则：

1. 固定全局发送序：`index = tokenIdx * K + topkIdx`
2. 分核切连续区间 `[startIndex, endIndex)`
3. 对当前 `index`，取前缀区间 `[0, index)` 的 `expertIds`
4. 在 UB 中构造常量向量 `dstExpertId`，对前缀做比较与归约
5. 得到 `curOtherExpertCnt` 后，计算

$$
curExpertCnt = index - curOtherExpertCnt
$$

6. 用 `curExpertCnt` 作为 `dstSlot` 发窗

向量化步骤：

1. `Duplicate(dstExpIdTensor_, dstExpertId, index)`
2. `Sub(subExpIdTensor_, expertIdsTensor_, dstExpIdTensor_, index)`
3. `Abs(...)`
4. `Mins(..., 1, index)`
5. `ReduceSum(...) -> curOtherExpertCnt`
6. `curExpertCnt = index - curOtherExpertCnt`

## 本页只保留的物理语义

- token 发送阶段写入的是完整通信 token，而不是纯 `expandX` 行；通信 token 由特征载荷和来源三元组组成，回搬阶段再拆成 `expandX` 与 `expandIdx`
- dispatch 数据区的逻辑组织按 `sourceRank -> localExpert -> tokenSlot` 理解；`dstRank`、`localExpert` 和 `dstSlot` 的推导属于发送语义的一部分，但字节级布局和地址公式留到后续布局设计阶段统一处理
- dispatch 状态区的逻辑组织按 `localExpert -> sourceRank -> stateSlot` 理解；等待阶段读取的是完整的 `expert x sourceRank` 状态矩阵，而不是只读聚合后的 rank 总数
- `expertRecvCounts`、`epRecvCounts`、`expertTokenNums` 都是在等待阶段从状态矩阵聚合得到，不是状态区中的原始存储对象
- 状态区在 host 侧 `aclrtMemset` 显式清零后传入 kernel，`Init()` 阶段不要额外手动清状态；状态复位发生在一轮等待成功、数据消费完成之后

## 下一跳

- 同步与可见性规则：`../api-rules/sync-and-visibility.md`
- dispatch / combine 中间量契约：`../api-rules/dispatch-combine-contracts.md`
- 分核改造：`../tiling-scheme/split-core-design.md`