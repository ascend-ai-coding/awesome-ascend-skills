# apace 算子开发方法论（从已有实现归纳的通用流程与决策点）

> **定位**：从 apace 已有算子实现（AllToAll 型 / AllGather 型 / compute-first 型）归纳的**通用开发方法论**。适用于任何 apace 通算融合算子，不绑定具体算子。事实细节（API 签名/常量/锚点）见 fundamentals 各文档，本文只回答"按什么顺序做、在哪些点做什么决策、每个决策的判据是什么"。
>
> 已有实现仅作决策点的取值示例（一行一例），不作为内容主体；参考实现定位方式：优先用 CANN 环境内置 apace 路径（Step 1 实测登记），需要跟踪上游时用 `scripts/fetch_apace.sh` 拉取。

## 1. 五阶段流程

```
阶段 1 语义分析 ──→ 阶段 2 路线决策 ──→ 阶段 3 设计合同 ──→ 阶段 4 实现组装 ──→ 阶段 5 验证
（做什么）        （怎么走）          （写什么合同）       （按合同拼装）      （怎么算过）
```

| 阶段 | 核心问题 | 产出 | 门禁（不满足回退） |
|:---|:---|:---|:---|
| 1 语义分析 | 数学定义 / 输入输出分布 / dtype / 约束 | REQUIREMENTS.md（含通信路径与底座维度） | 需求 8 维拷问全过（`requirement-analysis/grill-protocol.md`） |
| 2 路线决策 | 官方 kernel 可否直接用 / 场景命中 / 组件可组合性 | 路线结论（native / custom / unsupported）+ selected_scenario | capability-declaration 命中 supported；场景语义唯一命中或显式 unsupported |
| 3 设计合同 | D1-D6 决策点逐项定值 | DESIGN.md + PLAN.md | 设计三拷问（范式映射/同步合同/验证合同）逐项有锚点 |
| 4 实现组装 | 按合同在 kernel/<op>/ 下组装 | 代码 + ST 工程 | R1-R21 红线自检全过、编译通过 |
| 5 验证 | 精度 / 性能 / 边界 | 精度 PASS + perf 归档 + REVIEW.md | golden 先行、T=1 基线先行、R15 投产门槛 |

每阶段失败回退点：阶段 5 发现精度问题 → 回阶段 3 检查 D4/D5 决策（多数精度问题源于同步/累加合同错，非代码 bug）；阶段 4 发现组件能力缺口 → 回阶段 2 重判路线（禁止绕过 apace 接口层自建）。

## 2. 六个关键决策点（D1-D6）

设计合同（DESIGN.md）必须对以下六点逐项给出取值与依据。每个决策点附三类已有实现的取值示例（仅示意判据的用法）。

### D1 数据流方向：通信在前 vs 计算在前

**判据**：算子语义中通信的输入是**原始输入**还是**计算的产物**。原始输入 → 通信在前（AIV 搬数、AIC 消费）；计算产物 → 计算在前（AIC 生产、AIV 消费）。方向决定 flag 通道方向与谁等待谁。

| 取值 | flag 方向 | 生产者 | 示例 |
|:---|:---|:---|:---|
| 通信在前 | AIV Set → AIC Wait | AIV（通信） | AllToAll/AllGather 型：先聚合 A 再 matmul |
| 计算在前 | AIC Set → AIV Wait | AIC（计算） | ReduceScatter 语义型：先 matmul 出部分和再聚合 |

### D2 切分轴与通信轮次 T

**判据**：切分轴由**数据分布语义**决定（每卡持有什么、要交换什么），与通信原语无关——同一 AllToAll PUT 可服务输入 K 切分，也可服务输出 M 分布。T（commTurn = tileCnt + tailCnt）是"同步开销 vs 流水粒度"的权衡：T 大 → 流水细、同步多；T=1 → 串行基线。

- 硬约束：flagId 取轮次号时轮次索引 < FLAG_ID_MAX(16) 且避开 SyncAll 保留区（`SyncAll<true>` 占 14 → 实际安全上界 13）；waitedMask 为 uint32 → 通信 tile 总数 ≤ 32。
- 尾块处理三选一：T 整除无尾块（最简）/ 单尾块 ≤ 头块直传 / padding 对齐 + realFragmentSize 限读。
- 两阶段策略：精度期 T=1 串行基线 → 性能期扫描 tileCnt。

### D3 数据通路：连续 GM vs FragmentTensor

**判据**：通信后（或计算前）的多 rank 数据在 GM 上**连续**还是**离散**。连续 → 直接 GM 地址 + Slice；离散（各 rank 窗口段拼装）→ FragmentTensor 按 assembleAxis 虚拟拼装。窗口布局（rank-major、data/scale 分段 winOffset）必须在 tiling 头文件注释画明。

| 取值 | 适用 | 示例 |
|:---|:---|:---|
| 连续 GM | AllToAll 窗口（rank-major 拼接即目标布局） | A2A 型：Win 区即 `[rank][chunk]` |
| FragmentTensor | 聚合后逻辑形状跨 rank 离散分布 | AG 型：HEAD/MAIN/TAIL 三区；compute-first：本卡 GM 各 rank 段打包 |

### D4 同步合同

**判据**：按"谁通知谁、通知几次、在哪等"三问填写。硬规则：

- CrossCore flag：Set/Wait 的 idx 与次数**严格配对**（含计数式固定 flagId），flagId ∈ [0, FLAG_ID_MAX)；
- TeamBarrier totalJobs 按**分核映射**取值：通信在前（前 R 核映射）= rankSize，CrossDevice 由 `Wait<BARRIER_DEVICE>` 内建；计算在前严格分离（后 R 核映射）= 1 + 指定核显式 CrossDevice——两种映射不得混用；
- `SyncAll<true>` 一律放分核守卫**外**（全 AIV 同序同次数）；多余核不参与通信但必须参与 SyncAll；
- 通信完成语义：PUT 型 Wait = Drain + 跨卡 fence（保证对端写入本卡窗口可见）。

### D5 累加合同

**判据**：部分和在哪里累加、谁触发写出。选项按精度与 L0C 容量权衡：

| 形态 | 语义 | 适用 |
|:---|:---|:---|
| REMOTE-only（splitKNum=R） | 全部 rank 在同一 L0C 累加，计满单次 fixpipe | 融合基线；无 AtomicAdd |
| LOCAL + AtomicAdd（splitKNum=R-1） | 本地先行写 C，远端原子累加 | 本地计算掩盖通信首延迟 |
| DEFERRED_SYNC（splitKNum=R） | per-tile 本地算驻留 L0C → wait → 远端累加 → 单次 fixpipe | 精度优先/免原子 |
| staging + 增量归约 | mm 写 staging 即通信源，AIV 侧逐轮归约 | 计算在前（部分和需要跨卡规约） |

L0C 容量校验：`baseM × baseN × 4B ≤ L0C`（dbL0c=2 时减半）。

### D6 入口合同

**判据**：dtype 组合数决定入口变体数；host 运行期 dispatch 选择入口（禁硬编码单入口）。固定要素：`__gm__ CommContext*` 首参、tilingData 按值、每个入口含 `KERNEL_TYPE_MIX_AIC_1_1`、禁 `__schedmode__(1)`。tiling 结构 = `QuantMatmulTilingData` + `CommTilingData`（data/scale 按需）+ 模式开关，pack(8)+alignas(8)。

## 3. 跨阶段纪律（从实现中反复验证的规则）

1. **golden 先行**：先冻结每卡输入/输出分布与切分轴，再写 gen_data；golden 错则一切精度验证失效。
2. **T=1 基线先行**：串行基线全绿才能开多 tile——串行路径会掩盖窗口布局与 flag 配对问题。
3. **事实锚点制**：设计文档中每条接口事实必须带 `文件:行号` 锚点；读不到 ≠ 不存在，禁止虚构接口。
4. **共享层零修改**：`block/` `tiling/` 只读复用（CMake 直引）；发现组件缺口 → 回路线决策评估扩展（走共享层评审），不在算子目录私造副本。
5. **镜像同步**：红线 R 系列在 checklist（规范源）/SKILL/场景文档多处镜像，改一处必须全查。
6. **组件缺口显式登记**：需求超出已有通信组件（如 AllReduce 无枚举值）时，按 `communication.md` §7 扩展指南三步法评估，不默认 unsupported 也不默认可绕过。

## 4. plugin 7 步流程映射

ops-direct-invoke plugin 执行 7 步流程时，与本方法论的对应（详细门禁见 [`workflow_integration.md`](../workflow_integration.md)）：

| plugin 步 | 对应阶段 | apace 场景要点 |
|:---|:---|:---|
| 1 环境检查 | 阶段 1 前置 | dav-3510 / 多卡 / HCCL / CANN 内置 apace 路径实测登记 |
| 2/2.5 设计+串讲 | 阶段 1-3 | DESIGN.md 按 D1-D6 逐项定值；串讲覆盖六个决策点 |
| 3-5 开发/审查/修复 | 阶段 4 | R1-R21 红线逐项检查；实现层问题项目内修复 |
| 6 验收 | 阶段 5 | 精度（golden+容差分型）+ 性能（对标归档）双门禁 |
| 7 汇报 | — | 失败案例回流 review-checklist |

## 5. 已有实现索引（示例引用）

定位方式：CANN 内置 apace（Step 1 登记路径）下 `kernel/` 目录；或 `fetch_apace.sh` 拉取上游仓 `mc2/common/op_kernel/apace/`。

| 实现类型 | 参考算子 | D1-D6 典型取值 | 详见 |
|:---|:---|:---|:---|
| AllToAll 型（通信在前，K 切分） | `all_to_all_quant_matmul` | D1 通信在前 / D2 K 轴 / D3 连续 / D4 AIV→AIC / D5 三模式可选 / D6 4 变体 | [`scenarios/put-all-to-all-quant-matmul/`](../scenarios/put-all-to-all-quant-matmul/design.md) |
| AllGather 型（通信在前，M 聚合） | `all_gather_quant_matmul` | D1 通信在前 / D2 M 轴 / D3 FragmentTensor / D4 预触发 flag0 / D5 REMOTE-only / D6 单入口 | [`scenarios/put-all-gather-quant-matmul/`](../scenarios/put-all-gather-quant-matmul/design.md) |
| compute-first 型（计算在前，输出 M 分布） | 自研范式（ReduceScatter 语义） | D1 计算在前 / D2 输出 M 轴 / D3 FragmentTensor / D4 AIC→AIV 双 flag / D5 staging+增量归约 | [`scenarios/compute-first-reduce-scatter/`](../scenarios/compute-first-reduce-scatter/design.md) |

新算子设计时：先在 §5 找到 D1-D6 取值最接近的实现作骨架参考，再按场景文档/组件手册替换差异项——**参考实现提供的是决策点的取值先例，不是可复制的代码**。
