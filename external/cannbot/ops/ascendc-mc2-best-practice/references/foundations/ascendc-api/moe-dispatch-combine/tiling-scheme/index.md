# MoE Dispatch/Combine Tiling Scheme

这个专题模块只负责物理布局与抽象切块设计，不负责样例工程复制、接口补齐或通用 API 教材。

适用场景：

- 接口语义、状态协议语义和输出闭环已经明确，接下来要把这些语义落实成 window 布局、状态槽布局和分核方案
- 需要设计发送、状态写入、等待、回搬、最终统计的阶段切分
- 需要设计 `workspaceGM` 共享、跨核 prefix sum、输出 offset 和状态段边界

不适用场景：

- 规格补齐问题：`../samples/spec-template.md`
- 样例工程对齐、接口补齐和数据流闭环问题：`../samples/index.md`
- `DataCopyPad`、`SetValue/GetValue`、状态区一致性等 API 用法问题：`../api-rules/index.md`

## 进入条件

进入本专题前的前置条件：

- 芯片、EP、shared expert、`quantMode` 等规格已经明确
- dispatch / combine 的阶段语义、状态协议语义和输出闭环已经明确
- 当前任务已经开始决定物理布局、共享 workspace 和分核方案，而不是仍处于样例工程对齐或接口补齐阶段

## 信息披露顺序

本专题的信息披露顺序如下：

1. `window-memory-layout.md`：dispatch / combine 逻辑语义对应的 window 数据区、状态区、地址规则和 UB 规模
2. `multi-core-formulas.md`：各阶段总工作量、状态槽数量和每核边界公式
3. `split-core-design.md`：发送、状态写入、等待、回搬、统计各阶段的分核实现

## 设计主线

MoE dispatch/combine 的分核默认按阶段拆分，不共用一套全局 `blockStart_/blockEnd_`。优先回答四个问题：

1. 哪些工作项可以作为分核轴，哪些参数只能视为配置或输出语义
2. 各阶段分别由哪些核参与，边界如何计算
3. 哪些单核下的本地累计值在多核下已经退化为“本核局部值”
4. 哪些结果必须落到共享 GM，再由后续阶段显式搬回 UB 使用

## 阶段切分规则

### 发送阶段

- 按 token 或 linear token 分核
- 若存在 expert mask、shared expert、topK 展开，先形成有效工作项，再做切分
- 发送阶段顺手得到的局部累计值，只能服务本核当前发送段，不能直接驱动后续状态发布

### 状态写入阶段

- 按 `expert` 或 `expert x srcRank` 状态槽分核，不复用 token 区间
- 写状态依赖的是“本卡发给某个 expert 的总量”，不是某个发送核顺手得到的局部累计值
- 状态写入前必须明确选择“重算”或“先跨核汇总后再发布”

### 等待阶段

- 先按总状态槽数计算每核负责的状态段
- 每个核只轮询自己的状态段
- 等待成功后，立即对本核状态段的 `token_count` 做局部求和，并写入共享 `workspaceGM`

### 输出回搬阶段

- 必须复用等待阶段同一组状态段边界
- 回搬前先读取共享 workspace 中各核局部总量，做跨核前缀和，得到本核全局起始 offset
- 回搬顺序固定为“状态段顺序 -> 段内 slot 顺序”

### 最终统计输出阶段

- `expertTokenNums`、`epRecvCounts` 这类最终输出放到最后收口
- 当前核若只持有局部结果，不能直接写最终输出

## 三条硬规则

- 分核设计同时满足：不重不漏、无数据竞争、顺序可验证
- host 侧只传总核数；每阶段具体使用多少核由 kernel 决定
- 任何 UB、本地 Tensor、成员数组、局部变量里的中间统计，默认先判定为“本核局部值”

## Workspace 与共享状态

- `workspaceGM` 是所有核可见的共享 GM，不是某个核的私有临时区
- 多核共享 workspace 默认按核分槽位，每个核固定读写自己的槽位
- 每核局部总量先落到 workspace，再由读取侧显式 `DataCopyPad` 读回 UB 做 prefix sum
- 不要把 `SyncAll` 当成 cache 一致性保证

## 默认实施顺序

1. 先定 window 数据区和状态区布局
2. 再定各阶段工作量公式和每核边界
3. 只改发送阶段分核并验证
4. 只改状态写入分核并验证
5. 把等待和回搬作为一组联动分核并验证
6. 再补共享计数区和跨核前缀和
7. 最后处理最终统计输出

禁止一开始同时改发送、状态、等待、回搬、统计全部阶段。

## 文件落点

- 多核改造主战场：`split-core-design.md`
- 文件改动定位：`../samples/index.md` 与 `../samples/change-routing.md`
- 接口与中间量语义：`../samples/dispatch-dataflow.md` 或 `../samples/combine-dataflow.md`

## 详细材料

- 数据区 & 状态区内存布局公式：`window-memory-layout.md`
- 各阶段分核数量公式：`multi-core-formulas.md`
- 双缓冲四区轮转协议：`double-buffer-protocol.md`
- 完整分核设计与按文件落地清单：`split-core-design.md`
- 样例工程改造路径：`../samples/index.md`
- API 与一致性注意点：`../api-rules/index.md`