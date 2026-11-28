# get-all-gather-quant-matmul 场景设计

> 场景 ID：`get-all-gather-quant-matmul`
> 状态：`not_found`（AllGather GET 钩子未见源码记录；AllToAll GET 钩子已存在但官网无算子使用）
> 支持范围：GET 模式，AllGather + QuantMatmul，N 轴切分，UDMA

## 1. 准入条件

| 条件 | 说明 |
|------|------|
| 通信方向 | GET：接收方读取远端结果片段 |
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
| 方向 | 每个 rank 对其他 rank 的已就绪 C 列块发起远端读，写入本地结果的相应列区间 |
| 编排 | AIC 先 SetFlag → AIV WaitFlag |
| 回压 | AIC `CrossCoreWaitFlag<0x2, PIPE_M>(tid-bufCnt)` 环形回压（原理推导：回压 gate AIC 计算主流水，pipe 选取见 `api-crosscore-sync.md` §4；与 [`fusion.md`](../../fundamentals/fusion.md) §3.4 契约一致；bufCnt 为深度参数） |

### 3.2 数据分布

| 维度 | 合同 |
|------|------|
| 切分轴 | 局部 MatMul 的 N 列跨 rank 分片；AllGather 完成后，每 rank 得到完整 `[M, N]` 结果 |
| A 数据 | 各 rank 持有完整 A `[M, K]`（**A 不切分**：C 按 N 切分需全 M 的 A） |
| B 数据 | 各 rank 持有 B 的列块 `[K, N/rankSize]`，用于计算相同列范围的局部 C（见 [`architecture.md`](../../fundamentals/architecture.md) §4） |
| C 输出 | 每 rank 写出 N/rankSize 段到 Win 区，AIV 拉回完整 C |

> 这一数据分布是该定制场景的设计。可按列拼接各 rank 的局部乘积，与完整矩阵乘结果比对；还须在设备上验证 GET 的就绪与完成语义。

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
| Wait 参数 | 局部输出使用前以及重用远端源槽前须确认相应读取已完成；`Wait(true)` 可能提前返回，不能作为逐块完成凭据 |
| 同步范围 | CrossCore flag 通知本卡 AIC/AIV；跨 rank 的就绪与旧槽释放按通信协议确认 |
| flagId | tid（tile 索引） |

## 4. 联合合同

```
AIC: 复用时等待旧槽释放 → mm 计算并写本 rank 的 C 列块 → 发布就绪
AIV: 等待远端对应列块就绪 → GET 到本地输出对应列 → 确认读取完成，通知源 rank 可复用
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
| GET 稳定性 | 验证目标 rank 数下的逐轮就绪、完成与缓冲复用；§6.2.5 尚无 4+ rank 的稳定性结论 |

## 7. 已知限制

- 官网无 GET 算子样例，GET 内容为钩子契约级描述
- 地址公式与 self 跳过规则需从 `block/aiv_comm/` 头文件直接验证
- 4 个及以上 rank 的 GET 稳定性仍需按目标配置实测。
