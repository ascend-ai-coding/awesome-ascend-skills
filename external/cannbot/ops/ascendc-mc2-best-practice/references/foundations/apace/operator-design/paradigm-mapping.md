# 语义范式映射与架构差异（apace 路线）

> **定位**：目标需求存在**注册形态既有实现**（上游仓 ops-transformer `mc2/` 目录下的 HCCL/CCU 注册算子）时，在 apace 路线上实现同类语义的**选型与转换指南**。回答三个问题：选哪个原型参考、哪些可借鉴、哪些必须按 apace 架构重写。
>
> **参考实现定位**：优先在当前 CANN 环境内置 apace 路径下定位（Step 1 实测登记）；注册形态原型经 `scripts/fetch_apace.sh` 拉取上游仓后在 `mc2/<op>/` 目录查阅。已有实现仅作选型与借鉴边界的示例引用，不作为内容主体。
>
> **核心原则**：参考原型的**算法逻辑**（切分策略、通算重叠编排、量化语义、尾块处理），重写**架构机制**（通信引擎、同步原语、matmul 栈、tiling 生成、host 拉起）。禁止整段复制注册形态 kernel 代码——include 体系、同步原语、通信调用在 apace 下全部不兼容。
>
> **最优先参考永远是 apace 已有算子**（同为 apace 架构，骨架可直接复用）；注册形态原型只提供算法语义与验证基准。

## 1. 选型总表

按目标算子语义选取参考（apace 路线三态：`apace_native` 直接调官方 kernel / `apace_custom` 按场景组合 / `unsupported`）：

| 目标算子语义 | apace 已有参考（首选） | 注册形态原型（算法借鉴，上游仓 `mc2/` 目录） | apace 路线 | 通信组件现状 |
|:---|:---|:---|:---|:---|
| AllGather + MX 量化 matmul（每卡独立 A/B，全量 C 输出） | `kernel/all_gather_quant_matmul/`（AllGather+PUT、M 轴输入切分、FragmentTensor 三区） | `all_gather_matmul`（fp16/bf16）、`all_gather_matmul_v2`（量化语义） | 标准 = native；变体见场景 [`put-all-gather-quant-matmul`](../scenarios/put-all-gather-quant-matmul/design.md) | AllGather+PUT 已注册 |
| AllToAll + MX 量化 matmul（K 分段累加 `y = Σ_r A_r @ B_r`） | `kernel/all_to_all_quant_matmul/`（UDMA 版 + hcomm/CCU 版） | `allto_all_matmul`（先通后算+permute）、`matmul_allto_all`（先算后通） | 标准 = native；变体见场景 [`put-all-to-all-quant-matmul`](../scenarios/put-all-to-all-quant-matmul/design.md) | AllToAll+PUT/GET 已注册 |
| ReduceScatter 语义 + 量化 matmul（先算后通，M 轴输出分片） | 无 | `matmul_reduce_scatter`（M 轴 RS 流水）、`quant_reduce_scatter`（纯量化 RS，one-shot 互读） | `apace_custom`，场景 [`compute-first-reduce-scatter`](../scenarios/compute-first-reduce-scatter/design.md) | ReduceScatter 仅枚举预留；用 AllToAll+PUT + 本地归约组合实现 |
| AllReduce + 量化 matmul | 无 | `matmul_all_reduce`（量化通信三段式）、`quant_all_reduce`（纯量化 AR one-shot） | 无场景先例——设计前先读 §4 两条技术路线 | AllReduce**无枚举值**；须先扩展（见 [`communication.md`](../fundamentals/communication.md) §7） |
| 纯量化规约（无 matmul） | 无 | `quant_all_reduce`、`quant_reduce_scatter`（AIV-only 窗口互读 + 边读边反量化累加） | 视同上，评估后登记场景 | 同上 |
| 分组/变长 token 族（grouped alltoallv 类） | 无 | `grouped_mat_mul_allto_allv` 等 | 零场景命中 → 按注册表规则暂判 `unsupported`，需求澄清后评估登记新场景 | — |

> 语义歧义警示：apace 的 `all_to_all_quant_matmul` 是 **K 分段累加**语义（B 为 `[rankSize*K, N]` 拼接权重、逐 rank L0C 累加），与注册形态 `allto_all_matmul` 的 **permute+单次 matmul** 语义不同。需求若为后者，AIC 数据流须重新设计，仅通信循环骨架可复用。

## 2. 七维架构差异表（注册形态 → apace 直调形态）

| 维度 | 注册形态原型（`mc2/<op>/`） | apace（`apace/`） | 转换动作 |
|:---|:---|:---|:---|
| **通信引擎** | AIC 发消息 → HCCL server（AICPU KFC / CCU）执行跨卡搬运 | AIV 直接 UDMA（Urma `WriteNbi/ReadNbi`）写/读对端通信窗口，`Drain` 等完成 | 通信调用全部重写为 `CollectiveComm` 四段式 |
| **通信同步** | `hccl_.Wait(handleId)` 轮询 GM finishedTurnCnt | `Wait(Drain + TeamBarrier)` + `CrossCoreSetFlag<0x2,PIPE_MTE3>(round)` ↔ `CrossCoreWaitFlag<0x2,PIPE_MTE2>(round)`（flagId=轮次号） | 同步原语全部替换；EVENT_ID 4/5/6 自由约定作废 |
| **核型分工** | `KERNEL_TYPE_MIX_AIC_1_2`（AIC 兼通信+计算，AIV 辅助 ND2NZ/bias cast） | `KERNEL_TYPE_MIX_AIC_1_1`（AIV 纯通信，AIC 纯 Blaze matmul） | AIV 职责重设计；ND2NZ 链路删除（Blaze 直读 DN） |
| **matmul 栈** | AscendC `MatmulImpl` / Catlass / mat_mul_v3、qbmm_v3 | Blaze `BlockMmad<MatmulWithScaleMx,...>` + `BlockSchedulerQuantBatchMatmulV3`；多 rank 离散源用 FragmentTensor + blaze_ext | 计算调用全部重写；MX scale 融合语义相通 |
| **tiling** | op_host 性能建模配平 + tiling key 二进制分发 | host 侧 `QuantMatmulTilingSwat::GetTilingData` + 手写 `CommTilingData`（5 字段）推导；kernel 消费扁平 POD 按值传参 | tiling 生成与下发结构重写 |
| **workspace/窗口** | op_host 统一规划（gather 区+nd2nz 区+系统区）；输出可零拷贝 | host `CommChannelBuilder` 建链（HcclGetHcclBuffer 窗口 + 独立 barrier 区）；kernel 静态 UB 布局（每 comm 实例 512B + barrier 32B） | 窗口建立与内存规划重写 |
| **host 拉起** | aclnn 两段式 API + op_graph KFC/CCU 任务生成 + tiling key 注册 | ST 直调：fork 多 rank（rankId=deviceId）→ `HcclCommInitRootInfoConfig` → `CreateDeviceContext` → `<<<usedCoreNum>>>` launch | 接入层整体不在 apace 范围（hcomm 变体是通往注册形态的桥梁，见 [`operator-anatomy.md`](operator-anatomy.md) §6.2 CCU 变体模板） |

量化语义两者共通（MX：data fp8/fp4 + scale e8m0 每 64 元素组 2 字节 sub-scale），可直接借鉴。

## 3. 可参考 vs 必须重写清单

### 3.1 可参考（算法逻辑层）

| 类别 | 内容 |
|:---|:---|
| 切分策略 | M 轴切 tile（tileCnt 主块 + tailCnt 尾块）驱动通算重叠；长/短块配平思想 |
| 流水编排 | 本地数据先算掩盖通信首延迟；tile i+1 通信与 tile i 计算重叠；per-tile 依赖粒度同步（apace 的 dependTileIdx 机制即对应物） |
| 量化通信三段式 | AllReduce = requant → A2A → 低比特域 reduceSum → AG → 反量化（通信量减半以上） |
| one-shot 窗口互读 | 各 rank 数据入本卡窗口、读方错卡序轮询互读 + 状态区软同步 + 边读边反量化累加（与 UDMA 模型最同构的原型形态） |
| 尾块/非对齐处理 | paddedTailM 对齐、尾块单独轮次 |
| 输出布局 | C 按 rank 分段；AllToAll 结果按 rank 拼接 |
| 验证基准 | golden = CPU 反量化 + matmul + 通信语义模拟；容差按量化类型分型 |

### 3.2 必须重写（架构适配层）

| 类别 | 注册形态写法 → apace 写法 |
|:---|:---|
| 通信调用 | `hccl_.AllGather/AlltoAll/ReduceScatter` → `CollectiveComm<Op, Mode, T, TeamBarrier>` 的 Init/Commit/Wait/Finalize |
| 通信等待 | `hccl_.Wait(handleId)` → CommPolicy::WaitTile → `CrossCoreWaitFlag<0x2, PIPE_MTE2>(tileIdx)`（UDMA）或 `hccl_.Wait(handle)`（hcomm 变体） |
| matmul 调用 | `MatmulCompute`/Catlass/qbmm → Blaze BlockMmad（+ blaze_ext Fragment 版） |
| B 矩阵预处理 | ND→NZ 转换 → 直接 DN 布局进 Blaze（转换链路删除） |
| tiling 生成 | op_host tilingFunc + tiling key → SWAT 引擎 + 手写 CommTilingData 推导 |
| host 拉起 | aclnn/op_graph → ST main.cpp 直调（fork + HCCL + `<<<>>>`） |
| 多 arch 分发 | arch 三套 + tiling key → 单一 dav-3510，dtype 用不同入口函数区分 |

### 3.3 判断口诀

写任何一段代码前自问：
- 这段代码描述"**算什么、怎么切、何时重叠**"（算法逻辑）→ 参考原型；
- 这段代码描述"**怎么通信、怎么同步、怎么算、怎么被拉起**"（架构机制）→ 用 apace 组件重写，骨架抄 apace 已有算子。

典型错误：把 `hccl_.Wait()`、`CrossCoreSetFlag(EVENT_ID_6)`、ND2NZ 逻辑、tiling key 分发直接搬进 apace kernel——这些在 apace 中要么不存在、要么语义完全不同。

## 4. AllReduce 类算子的两条技术路线（无 apace 先例，设计前必读）

语义收敛建议（apace 适用范围 = MX 量化通算融合；非量化/伪量化变体不迁移）：

**路线 A：直接规约（WriteReduceNbi）**——扩展 CollectiveComm 新增 AllReduce op，AIV 用 `Hcomm::WriteReduceNbi`（hcomm 原语，支持 int8/16/32、uint32、half、float、bfloat16 + `HcommUrmaReduceOp`）把部分和远程原位规约进对端输出窗口。组件改动最小（先补 `CommCollectiveOp::AllReduce` 枚举值——当前只有 AllToAll/AllGather/ReduceScatter，见 [`communication.md`](../fundamentals/communication.md) §7 扩展指南）；难点是远程规约完成语义（Drain 只保证本端发出，须 CrossDevice barrier 确保对端累加落窗）与首轮覆盖/清零语义（每 tile 独立 slot 天然免清零）。

**路线 B：三段式分解（注册形态验证过的算法骨架）**——AllReduce = AllToAll → 低比特域 ReduceSum → AllGather。复用 apace 已有 AllToAll+PUT、AllGather+PUT 组件；中段 reduceSum 需要 AIV 向量计算（`block/aiv_compute/` 当前为空占位，须先建设）。流水掩蔽时序（第 1 块 ReduceSum 与 A2A 重叠、后续块与前一块 AG 重叠、段间 SyncAll 防精度）借鉴上游仓 `matmul_all_reduce` 的 `based_a2a_rs_ag` 实现。

**建议**：先路线 A 打通功能闭环（顺带沉淀 AllReduce 通信原语），profiling 后若通信量成瓶颈再按路线 B 演进——两条路线 kernel 侧骨架一致（AIC 产 tile → AIV 逐 tile 通信），演进成本低。

## 5. 原型研读操作指引

1. **先读 README.md**（原型算子目录：语义公式、shape/dtype 约束、支持矩阵），再按需读 op_kernel 主实现与 op_host/op_tiling。
2. 提炼**三要素**记录到 DESIGN.md：
   - **语义**：公式、输入输出 shape、dtype、约束（整除/对齐/范围）
   - **算法骨架**：切分轴、通算重叠时序、量化/反量化时机
   - **验证基准**：golden 生成方式、容差（可直接继承到 ST gen_data/verify_result）
3. 对照 §3 清单逐项判定借鉴/重写边界，并在 DESIGN.md §0.3 登记判定结果。
4. 检查原型是否跨算子复用组件——同类思想在 apace 中应沉淀为 `block/` 共享组件而非算子内私货。
5. 语义参考 ≠ 实现参考：注册形态算子均为 HCCL/CCU 注册实现、未用 apace——只借鉴算法与 golden，代码照 §2 差异表全部重写。
