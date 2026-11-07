# MoE Dispatch/Combine 规格模板

这个页面承接生成或设计阶段的规格补齐工作。进入样例工程改造前，优先先补这份规格。

## 基础信息

- 算子名：
- 目标算子：`dispatch` / `combine` / 成对生成
- 参考实现：`mc2` 中哪个算子或版本
- 目标芯片：Atlas A2 系列 / Atlas A3 系列 / Ascend 950PR/950DT / 其他（对应架构 dav-2201 / dav-2201 / dav-3510）

## 并行域

- `epWorldSize`：
- `epRankId`：

## 专家配置

- `moeExpertNum`：
- `sharedExpertNum`：
- `sharedExpertRankNum`：
- `expertShardType`：
- `topK`：

## 输入输出

- `x` shape / dtype：
- `expertIds` shape / dtype：
- `scales` / `expertScales`：
- `A` 的定义与取值：按最大接收量如何计算
- `expandX` 预留 shape / dtype：最大容量是否按 `(A, H_expand)` 预留
- `expandIdx` 预留 shape / dtype：是否按 `(A, 3)` 或等价线性长度预留，以及三元组字段顺序
- `expertTokenNums` shape / dtype / 语义：当前是否表示“每个专家收到的 token 数量”或前缀和
- `epRecvCounts` / `epSendCounts` shape / dtype / 语义：当前是否表示“EP 通信域各来源 rank 收到/回传的 token 数量”
- `dynamicScales` / `expandScales` shape / dtype：
- 需要哪些中间量输出：`expandX` / `expandIdx` / `assistInfoForCombine` / `expertTokenNums` / `epRecvCounts` / `dynamicScales` / `expandScales`

## kernel 路线

- 是否采用 `window + status`：
- shared expert 与 moe expert 是否分路径：
- 是否开启量化：
- `quantMode`：

## 设计优先级

- 优先产物：设计文档 / kernel 骨架 / host 接口 / API 文档
- 是否先忽略详细 tiling：是 / 否
- 是否需要阶段划分摘要：是 / 否

## 生成检查清单

### 规格

- 是否明确芯片、算子类型、EP/shared expert、quantMode
- 是否明确中间量需求，以及哪些真实接口参数被裁剪为占位
- 是否引用了流程说明中的阶段划分摘要或等价的阶段输入/输出约束

### kernel

- 是否明确数据区/状态区布局、目标地址计算、状态发送与等待机制
- 是否说明 shared expert、量化路径是否启用
- 是否保留 HCCL Window 基础地址函数或等价封装

### 中间量与 API

- `A` 是否按接口文档定义为“本卡可能接收的最大 token 数”
- `expandX` / `expandIdx` 是否按“最大容量预留 + 实际有效长度由统计张量确定”设计
- `expandIdx` 是否明确为三元组 tensor，以及字段顺序是否固定为 `[rankId, tokenId, topkId]`
- `epRecvCounts` / `epSendCounts`、`expertTokenNums` 语义是否明确
- 量化相关 `dynamicScales` / `expandScales` 是否闭环
- 主路径是否以 `DataCopyPad` / `SyncAll` / `GatherMask` 为核心

### 输出

- 是否先给知识和设计，再展开 tiling 或代码细节
- 是否保持真实接口名与 `mc2` 阶段结构
- 是否避免臆造接口名或中间量语义