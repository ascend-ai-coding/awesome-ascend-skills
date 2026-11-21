# get-all-gather-quant-matmul 场景设计

> 场景 ID：`get-all-gather-quant-matmul`
> 状态：`not_found`（AllGather GET 钩子未见源码记录；AllToAll GET 钩子已存在但官网无算子使用）
> 支持范围：GET 模式，AllGather + QuantMatmul，N 轴切分，UDMA

## 1. 准入条件

| 条件 | 说明 |
|------|------|
| 通信方向 | GET（计算→通信） |
| 编排形态 | AIC 先算，AIV 后拉 |
| 切分轴 | N 轴按 rank 切分（⚠️ 无官方样例，为原理推导） |
| 通信引擎 | UDMA |
| 算子类型 | AllGather + QuantMatmul |
| 官方参考 | 无（已验证事实：`block/aiv_comm/all_to_all/all_to_all_udma_get.h` 的 `AllToAllCommGetImpl` 已注册进 `CollectiveCommHelper<AllToAll, GET, ...>` 分发但无 kernel 使用方；**AllGather GET 钩子在源码中未见记录**——不得假定 `all_gather_udma_get.h` 存在，须在 Step 2 调查中以真实源码核对后记录 `not_found` 或补充证据） |

## 2. 消费输入

消费 matmul 链路事实（设计前核对结果），需求四要素中 `communication_direction=GET`、`split_axis=N`、`orchestration=AIC-first`。

## 3. GET 设计边界

### 3.1 通信方向与编排

| 维度 | 合同 |
|------|------|
| 方向 | GET = 计算→通信：AIC 先算 C 写到 Win 区，AIV 从远端 Win 区拉回本 rank 的 C 段 |
| 编排 | AIC 先 SetFlag → AIV WaitFlag |
| 回压 | AIC `CrossCoreWaitFlag<0x2, PIPE_M>(tid-bufCnt)` 环形回压（原理推导：回压 gate AIC 计算主流水，pipe 选取见 `api-crosscore-sync.md` §4；与 [`fusion.md`](../../fundamentals/fusion.md) §3.4 契约一致；bufCnt 为深度参数） |

### 3.2 数据分布

| 维度 | 合同 |
|------|------|
| 切分轴 | N 轴按 rank 切分，每 rank 持有 N/rankSize 段（输出 C 按 N 分布） |
| A 数据 | 各 rank 持有完整 A `[M, K]`（**A 不切分**：C 按 N 切分需全 M 的 A） |
| B 数据 | 各 rank 持有 B 的 N/rankSize 段 `[K, N/rankSize]`（**B 按 N 切分**，与输出 C 的 N 切分一致；见 [`../../../fundamentals/architecture.md`](../../fundamentals/architecture.md) §4 GET 行"B N-split"） |
| C 输出 | 每 rank 写出 N/rankSize 段到 Win 区，AIV 拉回完整 C |

> ⚠️ 本场景标注 not_found（原理推导，未经官方源码验证）：上述分布为按 AllGather+Matmul（GET、N-split 输出）的语义推导，实现前必须对照源码逐条验证钩子行为。

### 3.3 Win 区布局

| 维度 | 合同 |
|------|------|
| 环形复用 | Win 槽位环形复用 + 回压（bufCnt 控制槽位数） |
| 回压机制 | AIC `CrossCoreWaitFlag<0x2, PIPE_M>(tid-bufCnt)` 等待槽位可用（原理推导，非官方样例） |
| 数据/元数据 | 分离，host/kernel 偏移同源 |

### 3.4 通信轮次

| 维度 | 合同 |
|------|------|
| T 推导 | 由 host 派生，默认 `T \| mSeg` 无尾块 |
| flag 编排 | AIC SetFlag(tid) → AIV WaitFlag(tid)，T=1 单次 / T>1 逐轮配对 |

### 3.5 flag 编排

| 维度 | 合同 |
|------|------|
| 发起方 | AIC 先 SetFlag → AIV WaitFlag |
| Wait 参数 | Wait(true) 按 waitLast 早退语义：DoWait 仅在 currentTileIdx_ == totalTiles - 1 时执行一次 |
| SyncAll | 不需要（CrossCore flag 已保证时序） |
| flagId | tid（tile 索引） |

## 4. 联合合同

```
AIC: mm 计算 → 写 Win 区 → SetFlag(tid) → WaitFlag(tid-bufCnt) 回压
AIV: WaitFlag(tid) → GET 从远端拉回 → 写输出 GM
```

## 5. Step 3 前提

- 设计前核对已记录官方未覆盖项（GET 无官方参考）
- 场景语义命中 `get-all-gather-quant-matmul` 判据
- GET 钩子源码已只读验证：AllToAll GET 钩子（`block/aiv_comm/all_to_all/all_to_all_udma_get.h`）已确认存在；AllGather GET 钩子已确认**未见源码记录**（not_found），设计按钩子契约级原理推导进行，实现前须以真实源码补证

## 6. 验证合同摘要

| 维度 | 合同 |
|------|------|
| golden 语义 | 每 rank 输入（本地数据 + 远端数据来源）与输出语义明确，N 轴切分 |
| 精度标准 | 以 ST verify_result.py 为准 |
| 多卡矩阵 | R=2/4 双档 |
| GET 稳定性 | 4+rank 不稳定已知风险（见 [`fusion.md`](../../fundamentals/fusion.md) §6.2.5 PUT 优先） |

## 7. 已知限制

- 官网无 GET 算子样例，GET 内容为钩子契约级描述
- 地址公式与 self 跳过规则需从 `block/aiv_comm/` 头文件直接验证
- GET 模式 4+rank 不稳定，PUT 优先
