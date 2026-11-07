# Sample Helper 分层与查找

这页不是通用“工具函数手册”，也不负责解释 helper 背后的 API 语义、平台结构或同步规则。它只回答一件事：在 sample 工程里，常见 helper 分成哪几层、分别应去哪个文件找。

默认参考工程：`moe_dispatch_direct_invoke_sample/`。

## 适用场景

- 已经知道自己要改 sample 工程中的某一类 helper，但还不确定它属于哪一层、在哪个文件
- 需要快速判断“这个逻辑应该放在 compat 层、地址拼装层，还是状态动作层”
- 需要沿着 sample 的 helper 组织方式，在自己的工程里决定是否保留、合并或改名

## 不在本页范围的问题

- `mc2Context` / `winContext` 的来源、底层结构和平台差异：看 `../api-rules/mte-address-access.md`
- dispatch / combine 的阶段语义和中间量闭环：看 `dispatch-dataflow.md` / `combine-dataflow.md`
- 同步点、`SyncAll` 何时使用、状态发布/消费顺序：看 `../api-rules/sync-and-visibility.md`
- window 布局、状态槽布局和分核边界：看 `../tiling-scheme/index.md`

## helper 分层地图

| 分层 | 典型职责 | 优先文件 | 常见函数 |
| --- | --- | --- | --- |
| 上下文 / compat 层 | 从 `mc2Context` 读取 rank、window 基址、平台相关字段 | `include/moe_dispatch_base_compat.h` | `GetRankId()`、`GetRankDim()`、`GetWinSize()`、`GetStatusDataSpaceGm()`、`GetBaseWindAddrByRankId()`、`GetBaseWindStateAddrByRankId()` |
| 通信上下文初始化层 | 在 kernel 侧绑定 host 传入的 `mc2Context`，缓存最常用上下文 | `kernel/mte_dispatch_comm.h` | `InitHcclContextByAddr()` |
| window 参数初始化层 | 初始化 token 大小、payload 对齐、window 内各区域偏移和状态读写 buffer | `kernel/mte_dispatch_comm.h` | `InitDispatchWindow()`、`InitBuffer()` |
| 地址拼装层 | 在基址之上叠加布局偏移，得到数据地址和状态 slot 地址 | `kernel/mte_dispatch_comm.h` | `GetDispatchDataAddr()`、`GetLocalWindowDataAddr()`、`GetDispatchStateAddr()` |
| 状态动作层 | 发布状态、等待状态、消费后复位状态 | `kernel/mte_dispatch_comm.h` | `SetRemoteStatus()`、`WaitRemoteStatus()`、`ClearLocalStatus()` |

## 使用方法

1. 先判断当前改动属于“读上下文”“拼地址”还是“做状态动作”。
2. 再定位对应文件中的已有 helper，确认其组织方式。
3. API 语义、平台差异或同步策略问题不在本页范围，分别归入对应专题页。

## 组织上的判断口径

- compat 层只负责把平台差异和 window 基址读取收敛起来，不掺入 expert、slot、token 等业务偏移
- 地址拼装层只负责“先拿基址，再叠加布局偏移”，不直接承担状态发布或等待逻辑
- 状态动作层依赖地址拼装结果，但不应重新展开平台字段解析
- 若某个 helper 同时承担上下文解析、业务地址拼装和状态发布，说明该处职责已经耦合；优先处理方式是先拆层，再扩写实现

## 相关页面

- `mc2Context` / `winContext`：`../api-rules/mte-address-access.md`
- 同步与状态协议：`../api-rules/sync-and-visibility.md`
- 文件落点：`change-routing.md`