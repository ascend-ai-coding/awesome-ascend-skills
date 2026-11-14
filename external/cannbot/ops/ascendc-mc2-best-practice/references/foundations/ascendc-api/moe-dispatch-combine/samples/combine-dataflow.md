# MoE Combine 数据流

这个页面承接 combine 的阶段控制流、槽位语义和聚合闭环；不展开 window 字节布局、地址公式和状态槽大小计算。

## 最小阶段划分

| 阶段 | 输入 | 产出 | 本阶段只回答什么 |
| --- | --- | --- | --- |
| 初始化 | EP 参数、`mc2Context`、输出张量 | 上下文、回传地址、聚合输出地址、分核边界 | 回传与聚合地址从哪里来 |
| 回传发送 | `expandX` `expandIdx` `epSendCounts` | 回程 window 与远端状态 | 如何按 `expandIdx` 恢复来源位置并写回 |
| 接收等待 | 本地状态区 | 可聚合的完成状态 | 何时本地 token 所需贡献已经收齐 |
| 本地聚合 | 本地 window、`expertScales` | `x` | 如何恢复原 token 并完成加权求和 |

阶段附加约束：

- 回传阶段遵循“先写数据，再写状态”
- 等待完成后再清状态，避免旧轮次状态污染下一轮
- `expandIdx` 必须能唯一定位原始 `(srcRank, srcTokenIdx, srcTopKIdx)`

## 真实接口参数

以下参数直接对齐 `aclnnMoeDistributeCombineGetWorkspaceSize`。

### 输入参数

记号约定：

- `A` 表示 dispatch 侧为本卡接收量预留的最大 token 容量
- `H_expand` 表示 `expandX` 的单行存储宽度。对普通 dtype，`H_expand = H`；对 `fp4x2` 这类打包存储类型，`H_expand = ceil(H / 2)`

| 参数 | 含义 / 用意 | 逻辑维度 / shape | dtype | 关键约束 |
| --- | --- | --- | --- | --- |
| `expandX` | dispatch 传来并经过比如 FFN 计算的 token 特征输入，是 combine 的主数据输入 | 2D，按接口容量预留，常见为 `(A, H_expand)` | `ExpandXType` | 这里只包含特征载荷，不包含三元组附加信息 |
| `expertIds` | 原始 token 的 topK expert 路由表 | 2D，通常为 `(Bs, K)` | `INT32` | 与 dispatch 输入保持一致 |
| `expandIdx` | 标识每个有效展开条目的来源 rank、token 和 topK 槽位 | 2D，按容量视角预留为 `(A, 3)`；线性存储时长度至少为 `A * 3` | `INT32` | 每个有效条目固定为 `[srcEpRankId, srcTokenIndex, srcTopKIndex]` |
| `epSendCounts` | 描述本卡需要回传给 EP 通信域各目标 rank 的 token 数量 | 1D；按 EP rank 维度组织，常见为 `(epWorldSize,)` | `INT32` | 接口语义上与 dispatch 输出的 `epRecvCounts` 完全同构 |
| `expertScales` | token-topK 维度上的聚合权重 | 通常为 `(Bs, K)` | `FLOAT` | combine 聚合权重 |
| `xActiveMask` | 有效 token 或路由项的 mask | 1D 或 2D mask，依实现而定 | `BOOL` | 常见有 token 级 mask 和 expert 级 mask 两条路径 |
| `residualX` | 可选残差输入 | 通常为 `(Bs, H)` | `ExpandXType` | 仅在 combine 带残差融合语义时使用 |
| `gamma` | 可选归一化权重 | 通常为 `(H,)` 或路径相关 | `ExpandXType` | 仅在 combine 带 RMSNorm / 归一化融合语义时使用 |
| `sharedExpertX` | shared expert 额外输出输入 | 通常为 `(Bs, H)` | `ExpandXType` | 若存在 shared expert 直加路径，会在最终聚合时累加 |
| `oriX` | 原始 token 输入 | 通常为 `(Bs, H)` | `ExpandXType` | 若存在 copy expert 或 const expert 路径，常会直接读取 |
| `constExpertAlpha1` | const expert 的第一组权重 | 通常为 `(constExpertNum, H)` | `ExpandXType` | 仅在 const expert 路径存在时使用 |
| `constExpertAlpha2` | const expert 的第二组权重 | 通常为 `(constExpertNum, H)` | `ExpandXType` | 仅在 const expert 路径存在时使用 |
| `constExpertV` | const expert 的常量向量 | 通常为 `(constExpertNum, H)` | `ExpandXType` | 仅在 const expert 路径存在时使用 |
| `performanceInfo` | 可选性能等待信息输出缓冲 | 路径相关 | `INT32` | 仅在需要性能统计输出时使用 |
| `groupList` | 预留的 group list 参数 | 路径相关 | 路径相关 | 当前版本通常不支持 |
| `expandScales` | 与 `expandX` 对齐的权重或量化 scale | 通常按 routed item 对齐 | `FLOAT` | 若启用，必须与 `expandX` 行对齐 |
| `groupEp` | EP 通信域标识 | 标量属性 | - | 专家并行域 |
| `epWorldSize` | EP 域内 rank 总数 | 标量属性 | `INT64` / `INT32` | EP 域大小 |
| `epRankId` | 当前卡在 EP 域中的 rank 编号 | 标量属性 | `INT64` / `INT32` | 当前 rank |
| `moeExpertNum` | 全局 MoE expert 总数 | 标量属性 | `INT64` / `INT32` | MoE expert 总数 |
| `expertShardType` | 共享专家卡的部署方式 | 标量属性 | `INT64` / `INT32` | 共享专家部署方式 |
| `sharedExpertNum` | 共享专家数量 | 标量属性 | `INT64` / `INT32` | 共享专家数量 |
| `sharedExpertRankNum` | 承载共享专家的 rank 数 | 标量属性 | `INT64` / `INT32` | 共享专家使用的 rank 数 |
| `globalBs` | EP 域视角下的全局 batch 上界 | 标量属性 | `INT64` / `INT32` | EP 域全局 batch size |
| `outDtype` | 输出 `x` 的数据类型控制参数 | 标量属性 | `INT64` / `INT32` | 输出 `x` 的数据类型控制 |
| `commQuantMode` | 通信量化模式 | 标量属性 | `INT64` / `INT32` | 与 dispatch 量化路径配套 |
| `groupListType` | group list 的格式控制参数 | 标量属性 | `INT64` / `INT32` | 预留参数 |

### 输出参数

| 参数 | 含义 / 用意 | 逻辑维度 / shape | dtype | 关键约束 |
| --- | --- | --- | --- | --- |
| `XOut` | combine 聚合后的最终 token 输出 | 2D，通常恢复为 `(Bs, H)` | `ExpandXType` 或量化/输出控制对应类型 | 第 `b` 行对应原始 token 维 |
| `performanceInfo` | 可选性能等待输出 | 路径相关 | `INT32` | 仅在 `isPerformance` 打开时写回；按来源 rank 做 atomic max |

## 通用控制流

只要 combine 采用“回来源槽位 + 本地等待 + 本地聚合”这类协议，主路径通常可以抽象成：

1. 初始化发送侧和聚合侧所需的基础缓冲
2. 按 `expandIdx` 把 `expandX` 回传到来源 rank 的 window 槽位，并在对应状态槽发布 ready
3. 计算或加载 token mask / expert mask，并把 special expert 相关过滤条件融合进有效项集合
4. 各核按 token 子区间轮询本地状态区，等待该 token 需要的 expert/shared expert 槽位 ready
5. 状态满足后，从本地 window 读出各 expert 结果，按 `expertScales` 聚合，并叠加 shared/copy/const expert 等额外路径；最后写回 `XOut`
6. 如果需要性能统计，再把等待信息或附加统计写回对应输出

## 核心约束

- 回传阶段的目标地址必须与 `expandIdx[i] = [srcRank, srcTokenIdx, srcTopKIdx]` 一一对应
- 若使用 window + 状态协议，通常是“先写来源 rank 的 window 数据区，再写对应状态槽”
- 等待阶段是按 token 检查本地状态区累计和，不是按整卡一次性 barrier 后再处理
- 状态满足后立即清该 token 的状态槽，然后进入本地聚合
- 量化打开时，发送端先量化写 window，消费端再反量化后参与聚合，combine 必须与 dispatch / 通信侧量化布局配套
- combine 可能同时支持 moe expert、shared expert、zero expert、copy expert、const expert 等多路径；不要把 combine 固化成单一路径乘权求和

## 容量与有效长度

- `expandX` 和 `expandIdx` 采用“最大容量预留 + 实际有效长度由统计张量确定”的使用方式
- combine 不应把 `A` 误认为本次一定收到的实际 token 数
- `epSendCounts` 记录的是按 rank 维组织的回传 token 数；阅读实现时，需要额外确认代码如何从这组计数推导本轮总发送任务数，再切到各核
- 实际本地聚合 token 集通常由 `xActiveMask`、special expert 过滤以及其他有效项统计共同决定

## 本页只保留的物理语义

- combine 若采用“回来源 token 槽位”的协议，window 数据区应按 `token -> slot` 理解，而不是按“收到多少条就线性追加多少条”理解
- 一个常见 token 区域包含 `K + sharedExpertNum` 个槽位：前 `K` 个槽位对应 moe expert，后续槽位对应 shared expert；是否存在 zero/copy/const 等额外路径，不改变 `expandIdx` 对 moe/shared 主路径槽位的定位语义
- `expandIdx = [srcRank, srcTokenIdx, srcTopKIdx]` 的作用，是把某个 `expandX` 条目恢复到来源 rank 的 token 槽位；这决定了回传写入顺序和后续等待粒度
- 若采用 ready 状态协议，状态区与数据区保持同一套 `token -> slot` 逻辑索引；等待按 token 粒度判断是否“所需槽位都已到齐”，而不是对整块状态区做统一 barrier
- ready 的最小语义只要求能区分“未就绪 / 已就绪”；更具体的字节布局、地址公式和状态槽大小放到后续布局设计阶段统一处理

## 最小路径

1. 根据 `epSendCounts` 表示的各 rank 回传计数，先确定本轮总发送任务，再切出本核负责的发送任务区间
2. 从 `expandIdx` 读取 `[srcRank, srcTokenIdx, srcTopKIdx]`，把 `expandX` 写回来源 rank 的对应 window 槽位
3. 每写完一个槽位，再把对应状态槽置 ready
4. 来源侧按 token 粒度轮询本地状态区，直到该 token 需要的 moe/shared expert 槽位就绪
5. 清该 token 状态槽，按 `expertScales` 聚合 moe expert 结果
6. 若命中特殊专家路径，再追加 zero/copy/const/shared expert 的对应处理
7. 把最终结果写回 `XOut`；若启用性能信息，再额外写 `performanceInfo`

## 常见特殊路径

- `zero expert`：该 expert 不贡献任何数值，只影响等待目标与分支选择
- `copy expert`：直接读取 `oriX`，乘以对应 `expertScale` 后累加
- `const expert`：先读取 `oriX`，再用 `constExpertAlpha1/2` 和 `constExpertV` 计算混合结果后累加
- `shared expert`：从 shared expert 对应 window 槽直接读取并累加；若 `hasSharedExpertX_` 打开，还会额外加上 `sharedExpertX`
- `expert mask / token mask`：会改变有效 token 集和等待目标数，必须参与状态目标值判断
- `performanceInfo`：不是功能输出，而是等待时间统计输出

## 常见预留能力

- 有些实现会预留 Add / RMSNorm / residual 融合辅助函数，但主路径未必真的启用
- `residualX`、`gamma` 等输入即使已经进入 kernel 签名，也不代表当前主路径已经完成“聚合 + 残差 + RMSNorm”融合
- 描述 combine 时，应把这类项标成“接口已预留 / 路径可选”，不要直接写成“当前主流程已经完成融合”

## 必答项

- `epSendCounts` 是否与 dispatch 的 `epRecvCounts` 保持同构，即“按 rank 维组织的回传 token 数量”
- 代码是如何从这组按 rank 的计数推导出总发送任务数和本核发送区间的
- `expandIdx` 是否固定为 `[srcRankId, srcTokenIdx, srcTopKIdx]`
- 当前代码是“先二次 dispatch 回来源 window，再本地等待聚合”，还是其他协议
- 每个 token 在 window / 状态区里是否固定有 `(K + sharedExpertNum)` 个槽位，还是使用其他槽位组织方式
- 聚合是否先 cast 到 float，再做按 expertScale 的累加
- zero/copy/const/shared expert 哪些路径启用，是否侵入主热路径
- `performanceInfo`、量化路径、mask 路径是否启用
- `residualX` / `gamma` 是当前主流程真实使用，还是仅为预留融合路径

## 下一跳

- dispatch / combine 中间量契约：`../api-rules/dispatch-combine-contracts.md`
- 同步与可见性规则：`../api-rules/sync-and-visibility.md`
- 分核改造：`../tiling-scheme/split-core-design.md`