# get-all-gather-quant-matmul 开发指导

> **场景 ID**：`get-all-gather-quant-matmul`（GET 模式，AllGather + QuantMatmul 融合，N 轴切分，UDMA 直调）
> **生效条件**：设计冻结后由 Step 3 编译进 PLAN（`implementation_route=apace_custom`，`selected_scenario=get-all-gather-quant-matmul`）；Step 4 **只执行 PLAN**，不得在实现期新增设计决策。
> **前置阅读**：[`../../fundamentals/communication.md`](../../fundamentals/communication.md) §3（GET 钩子契约）、[`../../fundamentals/fusion.md`](../../fundamentals/fusion.md) §2/§3.4（GET 选型与环形回压）、[`../../fundamentals/compute.md`](../../fundamentals/compute.md) §3（QuantMatmulMxKernel）。

---

## 1. 状态说明：无官方参考，契约须源码验证

| 事项 | 状态 |
|:---|:---|
| GET 钩子基础设施 | ✅ 已存在：`apace/block/aiv_comm/all_to_all/all_to_all_udma_get.h`（`AllToAllCommGetImpl`），已注册进 `CollectiveCommHelper<AllToAll, GET, ...>` 分发 |
| AllGather GET 钩子 | ❌ 未实现（communication.md 钩子支持表：AllGather/GET = —） |
| GET 使用方 kernel | ❌ 官网 kernel/ 下两个算子均为 PUT 模式，无 GET 使用方 |

**开发纪律**：本文档为原理推导级指导。任何钩子行为假设（`PostInit`/`DoCommit`/`DoWait`/`DoFinalize` 职责、barrier 顺序、self 跳过、`Wait(true)` waitLast 早退语义）必须在实现前对照 `block/aiv_comm/` 源码逐条验证，并在 PLAN 第 11 章记录验证结论；与源码不符时以源码为准并回退设计阶段。

## 2. 有序动作（Step 4 按序执行）

| # | 动作 | 关键要点 |
|:--|:---|:---|
| 1 | 适配通信对象 | 优先**复用** `AllToAllCommGetImpl`（逐 targetRank 从远端 Win 拉回）。⚠️ **待验证假设**：其"按请求方 rankId 取槽位"语义是否天然覆盖 AllGather"收齐所有 rank 数据"语义，取决于 Win 布局合同，实现前必须对照源码逐条验证（本场景 not_found）；仅当分发层要求 `CollectiveCommHelper<AllGather, GET, ...>` 时才新写 `AllGatherCommGetImpl`，且须保持与 AllToAll GET 相同的钩子契约（barrier 先 Core 后 Device、self rank 跳过） |
| 2 | 适配计算 | 复用 `QuantMatmulMxKernel`（`quant_matmul_mx_kernel.h`，CommPolicy 策略注入），AIC 按 N 轴切分算 C 并写入 Win 区（每 rank 持有 C 的 N/rankSize 列，B 可 N-split，见 architecture.md §4） |
| 3 | 实现 Win 槽位环形复用 + 回压 | Win 区按 bufferCount 槽位环形复用；**AIC 侧 `CrossCoreWaitFlag` 等待槽位可用 flag**（AIV 拉走数据后置位），禁止无回压覆写（fusion.md §3.4） |
| 4 | 完成 flag 编排闭环 | AIC 算完写 Win → `SetFlag`（C-ready）；AIV `WaitFlag` → `Commit(GET)` → 拉取完成 → `SetFlag`（V-done / 槽位释放）；尾部按 `Wait(true)` waitLast 语义处理——**最后一轮通信不被 Drain**，须由 `Finalize`/`DoFinalize` 的 CrossDevice barrier 完成 final drain，防止尾部越界覆盖 |

> ⚠️ GET `DoCommit` 内 `CrossCore()+CrossDevice()` barrier 是对端 AIC 槽位就绪的回压保证，**不可省**；barrier 顺序为先 Core 后 Device（与 PUT 相反）。

## 3. 文件级合同

| 文件 | 状态 | 说明 |
|:---|:---|:---|
| `kernel/{op_name}_tiling_data.h` | MODIFY | tiling 结构体：N 轴切分字段、Win 偏移、bufferCount、flag id、tile 字节数 |
| `kernel/{op_name}_impl.h` | MODIFY | Impl：Init/Run 编排，GET 模式 `AllToAllProcess` 中**不需要 SyncAll**（同步全由 CrossCore flag 负责，fusion.md §3.1） |
| `kernel/quant_matmul_mx_kernel.h` | REUSE | 复用官方 `QuantMatmulMxKernel`，仅注入 GET CommPolicy |
| `src/kernel_launcher.h` | MODIFY | `__global__` 入口含 `KERNEL_TYPE_MIX_AIC_1_1`；UDMA 模式带 `__gm__ CommContext*` |
| `src/main.cpp` | MODIFY | Tiling + fork 多 rank + HCCL 建链 + launch；host 前置校验 + perf 模式 L2 flush |
| `scripts/gen_data.py` / `verify_result.py` | MODIFY | golden 按 N 轴切分核对；多 rank 校验 |
| `block/` `tiling/` | 禁止修改 | 与官网仓原始文件完全一致（R4） |

## 4. 验证矩阵

| 维度 | 要求 |
|:---|:---|
| 精度 | golden 比对覆盖 N 轴各 rank 列段；dtype 全变体；**重点回归 R=2 与 R=4+**（GET 多 rank 数据可见性风险，见 §6） |
| 时序 | 大 shape / 多轮（bufferCount 环形翻转 ≥ 2 圈）下无脏读、无"假通过" |
| 同步 | flag idx 配对（AIV WaitFlag idx == AIC SetFlag idx）；末轮 final drain 经 DoFinalize 验证 |
| 性能 | 真实大 shape × R=2/4 双档，与 mc2 融合算子 / hccl 分步路径对标归档（R15）；perf 每轮实调 L2 flush kernel（R20） |
| 可靠性 | 单轮 PUT 大小按 case 复核（bring-up 期 ≤512KB 起步，非硬上限）；所有 hcomm 调用返回值 assert |

## 5. 合规映射（本场景重点 R 项）

| R 项 | 本场景落点 |
|:---|:---|
| R1/R2 | 禁 `__schedmode__(1)`/`core_ratio`；入口含 `KERNEL_TYPE_MIX_AIC_1_1` |
| R4 | `block/` `tiling/` 零修改（GET 钩子只用不改） |
| R5 | C-ready / V-done / 槽位释放 flag idx 双侧配对 |
| R6 | UDMA 模式 `__gm__ CommContext*` |
| R12 | commBuf/barrierBuf 与 TPipe 管理 buffer 物理隔离 |
| R13 | 通信对象 `totalJobs=rankSize`（后 rankSize 核各负责 1 个 targetRank 并行 GET） |
| R14 | Win 数据/元数据区分离，host/kernel 偏移同源（硬红线）；单轮 PUT 大小按 case 复核（风险提示） |
| R15/R20 | 投产级性能对标 + L2 flush 实接线 |
| R7/R8 | 禁 `AscendC::Matmul` 高阶 API、禁 `Hccl::*` |

## 6. 额外警示：GET 模式 4+ rank 不稳定性

- **PUT 优先原则**（fusion.md §6.2.5）：GET 模式在 4+ rank 存在数据可见性问题，且官网无 GET 算子样例；设计期应已论证"必须 GET"的理由，否则回退 PUT。
- 已知绕行方案（communication.md 陷阱 #11）：GET `Drain` 返回非 0 时，可用 `TeamBarrier.CrossDevice()`（跨 rank 就绪）+ `SyncAll<true>`（核间可见）后直接 `DataCopyPad` 读远端 Win 区。
- R≥4 精度回归不通过时**禁止带病交付**：先在 PLAN 第 11 章记录 `design_issue`，回退设计阶段评估改 PUT 或绕行方案。
