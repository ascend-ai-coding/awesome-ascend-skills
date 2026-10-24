# MoE Dispatch/Combine API Rules

这个专题模块只负责 MoE dispatch/combine 场景下的 API 使用约束、共享内存访问方式和状态协议注意点，不重复通用 Ascend C API 教材。

## 适用场景

- 需要确认 `DataCopyPad`、`SetValue/GetValue`、cache 可见性、一致性操作是否可用
- 需要确认 `expandIdx`、`epRecvCounts`、`epSendCounts`、状态区等对象的语义边界
- 需要梳理 dispatch/combine 成对理解时的 API 和数据协议约束

## 不适用场景

- 规格未补齐、工程架构未确定、dispatch / combine 接口语义未闭环的内容，归入 `../samples/index.md`
- window 物理布局、状态槽分配、workspace 分槽和各阶段分核边界设计，归入 `../tiling-scheme/index.md`

## 进入条件

进入本模块前，至少应当已经明确当前实现确实会碰到共享 GM / workspace 访问、状态区发布与等待、核间可见性，或需要解释 `expandIdx`、`epRecvCounts`、`epSendCounts` 这类对象的成对契约。

通用 API 参数和基础示例请查阅 `ascendc-api-best-practices` skill；本页只保留 MoE 特有规则。

## 详细页面

- MTE 通信窗口地址获取（A3/A5 平台、原子语句、易错点）：`mte-address-access.md`
- 接口契约（中间量成对语义）：`dispatch-combine-contracts.md`
- 同步与可见性：`sync-and-visibility.md`

## 成对语义优先

- `dispatch` 和 `combine` 必须成对理解，尤其是 `expandIdx`、`epRecvCounts`、`epSendCounts`
- 设计或改代码时，优先确认这些中间量在 dispatch 侧如何生成、在 combine 侧如何消费，不要单独局部解释字段

## 共享 GM / Workspace 访问规则

- 共享 GM、`workspaceGM`、状态区、window 数据区默认走 `DataCopyPad` 或等价的 GM 可见路径
- 默认禁止用 `GlobalTensor::SetValue()` / `GetValue()` 直接访问共享 GM、共享 workspace、共享状态结果
- 极少数场景如果必须使用 `SetValue()` / `GetValue()`，必须额外补齐 `DataCacheCleanAndInvalid` 或等价一致性操作，并重新验证核间可见性
- `SyncAll` 只保证执行同步，不保证 Data Cache 与 GM 一致性

## 状态协议

- SHMEM 库内部同步资源在 init 阶段自动清零；用户通信窗口（含状态区）需在 host 侧 `aclrtMemset` 显式清零后再传入 kernel
- `Init()` 阶段不要手动清状态；状态复位只应发生在一轮消费完成之后
- 发布顺序默认是“先写 count/offset/附带信息，再发布 ready”
- 等待侧只轮询自己负责的状态段
- 状态消费完成后，只清理本核负责段，或在统一收口后清理；不要沿用单核整块 reset 习惯

## 多核共享计数规则

- 每核只写自己的共享槽位，不要多个核直接累加同一地址
- 写完共享槽位后，同步，再统一读取
- 读取其他核产出的共享结果时，先通过 `DataCopyPad` 搬回 UB，再在 UB 中做 prefix sum 或后续计算

## 读写边界

- 生成 kernel 时要显式考虑分核边界，至少明确哪些核参与发送、等待、回搬，哪些核应拿空区间直接跳过
- 没有确认芯片、EP、shared expert、`quantMode` 之前，不应扩写实现
- host 侧只传总核数；kernel 侧负责决定每个阶段实际使用多少核以及如何切分工作项

## 常见误用

- 把发送阶段顺手得到的局部累计值直接当成状态发布总量
- 在 `SyncAll` 后直接 `GetValue()` 读取其他核刚写入的共享结果
- 把本核局部统计只放在 UB、局部数组或成员变量里，却让其他核隐式依赖它
- 在状态尚未全部消费完成前提前清理共享状态

## 与其他专题模块的关系

- 样例工程闭环：`../samples/index.md`
- 多核切块设计：`../tiling-scheme/index.md`
- 通用 API 文档：查阅 `ascendc-api-best-practices` skill