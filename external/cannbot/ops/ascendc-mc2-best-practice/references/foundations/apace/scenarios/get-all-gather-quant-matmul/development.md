# get-all-gather-quant-matmul 开发指导

> **场景 ID**：`get-all-gather-quant-matmul`（GET 模式，AllGather + QuantMatmul 融合，N 轴切分，UDMA 直调）
> **生效条件**：设计冻结后由 Step 3 编译进 PLAN（`implementation_route=apace_custom`，`selected_scenario=get-all-gather-quant-matmul`）；Step 4 **只执行 PLAN**，不得在实现期新增设计决策。
> **前置阅读**：[`../../fundamentals/communication.md`](../../fundamentals/communication.md) §3（GET 钩子契约）、[`../../fundamentals/fusion.md`](../../fundamentals/fusion.md) §2/§3.4（GET 选型与环形回压）、[`../../fundamentals/compute.md`](../../fundamentals/compute.md) §3（QuantMatmulMxKernel）。

---

## 1. 状态说明：无官方参考，契约须源码验证

| 事项 | 状态 |
|:---|:---|
| GET 钩子基础设施 | ✅ 已存在：`apace/block/aiv_comm/all_to_all/all_to_all_udma_get.h`（`AllToAllCommGetImpl`），已注册进 `CollectiveCommHelper<AllToAll, GET, ...>` 分发 |
| AllGather GET 钩子 | `not_found`：上述源码范围内未见 AllGather GET 分发 |
| GET 使用方 kernel | ❌ 官网 kernel/ 下两个算子均为 PUT 模式，无 GET 使用方 |

**开发纪律**：本文档为原理推导级指导。任何钩子行为假设（`PostInit`/`DoCommit`/`DoWait`/`DoFinalize` 职责、barrier 顺序、self 跳过、`Wait(true)` waitLast 早退语义）必须在实现前对照 `block/aiv_comm/` 源码逐条验证，并在 PLAN 第 11 章记录验证结论；与源码不符时以源码为准并回退设计阶段。

## 2. 有序动作（Step 4 按序执行）

| # | 动作 | 关键要点 |
|:--|:---|:---|
| 1 | 适配通信对象 | 以 PUT AllGather 核对各来源 rank 与列偏移，以 AllToAll GET 钩子核对远端读取。验证源槽寻址、到达语义及本地列的处理，再以已核实的 block 接口实现 GET AllGather 并接入通信分发。 |
| 2 | 适配计算 | 复用 `QuantMatmulMxKernel`（`quant_matmul_mx_kernel.h`，CommPolicy 策略注入），AIC 按 N 轴切分算 C 并写入 Win 区（每 rank 持有 C 的 N/rankSize 列，B 可 N-split，见 architecture.md §4） |
| 3 | 实现 Win 槽位环形复用 + 回压 | 同一源槽将被改写时，先确认所有需要这个旧列块的远端 rank 都已完成读取，再通知本地 AIC 可复用（fusion.md §3.4）。 |
| 4 | 完成 flag 编排闭环 | AIC 写 Win 后发布就绪；各 AIV 确认对应远端槽就绪，提交 GET，确认读取完成后通知源 rank。依据实际钩子核对最后一轮的完成路径，不能将提前返回的 `Wait(true)` 当成读取完成。 |

> 参考 AllToAll GET 钩子在 `DoCommit` 中先 `CrossCore()` 再 `CrossDevice()`。将该协议用于 AllGather 前，要验证它覆盖了每个源 rank 的就绪与旧槽释放。

## 3. 文件级合同

| 文件 | 状态 | 说明 |
|:---|:---|:---|
| `kernel/{op_name}_tiling_data.h` | MODIFY | tiling 结构体：N 轴切分字段、Win 偏移、bufferCount、flag id、tile 字节数 |
| `kernel/{op_name}_impl.h` | MODIFY | Impl 负责 Init/Run、AIC/AIV 通知和跨 rank 通信完成协议（fusion.md §3.1） |
| `kernel/quant_matmul_mx_kernel.h` | REUSE | 复用官方 `QuantMatmulMxKernel`，仅注入 GET CommPolicy |
| `src/kernel_launcher.h` | MODIFY | `__global__` 入口含 `KERNEL_TYPE_MIX_AIC_1_1`；UDMA 模式带 `__gm__ CommContext*` |
| `src/main.cpp` | MODIFY | Tiling + fork 多 rank + HCCL 建链 + launch；host 前置校验 + perf 模式 L2 flush |
| `scripts/gen_data.py` / `verify_result.py` | MODIFY | golden 按 N 轴切分核对；多 rank 校验 |
| `block/` `tiling/` | 禁止修改 | 与官网仓原始文件完全一致（R4） |

## 4. 验证矩阵

| 维度 | 要求 |
|:---|:---|
| 精度 | 分别核对每 rank 的局部 N 列、AllGather 后各 rank 的完整输出；覆盖支持的 dtype，在 R=2 与 R≥4 下复验。 |
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

## 6. 多 rank GET 的验证

- 4 个及以上 rank 的稳定性尚待验证。比较逐 rank、逐轮的列块，定位源槽就绪、GET 完成与复用时序；固定 GET 的需求应保留远端读的语义。
- `Drain` 返回非零时，记录 rank、通道、轮次和错误码，并沿该通道确认读取是否完成。跨 rank barrier 或本地核间同步本身不能证明一次失败的 GET 已完成。
- R≥4 精度回归不通过时**禁止带病交付**：先在 PLAN 第 11 章记录 `design_issue`，回退设计阶段评估改 PUT 或绕行方案。
