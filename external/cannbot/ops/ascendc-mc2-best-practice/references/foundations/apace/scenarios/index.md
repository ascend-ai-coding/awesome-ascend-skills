# apace 定制扩展场景注册表

> **定位**：本表是 apace 路线的**定制扩展场景唯一注册表**。Step 3（Design）在路线决策时，**仅当**设计前核对已记录官方 kernel 未覆盖的需求项后才查询本表；官方 `apace/kernel` 可直接调用或参考复用时（`apace_native` 路线）**不读取**本表。
>
> **查阅规则**：官方未覆盖项的**需求语义**命中某场景的语义判据时，**默认查阅**该场景的 design.md / development.md 作为设计基线，并将 `selected_scenario` 记录到 DESIGN.md。准入条件只含**需求侧语义判据**（数学语义、切分语义、通信方向），不含实现侧决策（通信实现方式、编排形态由场景文档指导决策，不作为准入前提）。语义零命中 → 判 `unsupported`；多命中 → 视为注册表缺陷或需求描述不清，停止并等待澄清，禁止自行择优。
>
> ⚠️ **已有生产实现的场景不受官方覆盖性判定反向阻断**：本仓/用户工程中已存在某场景的生产实现时，阅读该场景文档不需要"官方未覆盖"前提——场景文档同时是该类算子的改造/复用手册。

## 场景注册表

| 场景 ID | 支持范围 | 准入条件 | 设计指导 | 开发指导 | 状态 |
|:---|:---|:---|:---|:---|:---|
| `put-all-to-all-quant-matmul` | PUT 模式 AllToAll+QuantMatmul 的**变体扩展**（官方 kernel 未覆盖的需求：新 dtype 组合、bias/scale 融合、不同切分/编排），K 轴切分，UDMA 直调 | 通信原语为 AllToAll；localMatmul 为 QuantMatmul（含 scale）；PUT 方向；核间数据沿 K 轴切分；UDMA；**且存在官方 kernel 未覆盖的 native gap**（标准 PUT AllToAll+QuantMatmul 需求走 `apace_native`，不读本表） | [design.md](put-all-to-all-quant-matmul/design.md) | [development.md](put-all-to-all-quant-matmul/development.md) | 已实现 |
| `put-all-gather-quant-matmul` | PUT 模式 AllGather+QuantMatmul 的**变体扩展**（新 dtype、bias 融合、rankSize>8、编排改造），M 轴输入切分，UDMA 直调 | 通信原语为 AllGather；localMatmul 为 QuantMatmul；PUT 方向；核间数据沿 **M 轴聚合**（每卡持本卡 M 行 × 全部 K 列，每卡输出全量 C）；UDMA；**且存在官方 kernel 未覆盖的 native gap**（标准需求走 `apace_native`） | [design.md](put-all-gather-quant-matmul/design.md) | [development.md](put-all-gather-quant-matmul/development.md) | 已实现 |
| `get-all-gather-quant-matmul` | GET 模式，AllGather + QuantMatmul 融合，N 轴切分，UDMA 直调 | 通信原语为 AllGather；localMatmul 为 QuantMatmul；GET 方向；核间数据沿 N 轴切分；UDMA | [design.md](get-all-gather-quant-matmul/design.md) | [development.md](get-all-gather-quant-matmul/development.md) | not_found |
| `compute-first-reduce-scatter` | compute-first 模式，ReduceScatter 语义 + QuantMatmul 融合，输出沿 M 轴切分，UDMA 直调 | **逻辑语义为 ReduceScatter**（每 rank 输出 M 轴分片、跨 rank 求和）；localMatmul 为 QuantMatmul；compute-first（先算后通信）；UDMA。（通信实现方式——AllToAll PUT + 本地增量归约——由场景文档指导决策，非准入前提） | [design.md](compute-first-reduce-scatter/design.md) | [development.md](compute-first-reduce-scatter/development.md) | 已实现 |

### 状态说明

| 状态 | 含义 |
|:---|:---|
| 已实现 | 有官方参考或生产实现，design.md / development.md 已就绪，可直接指导开发 |
| planned | 场景已规划，指导文档编写中，暂不可用于路线决策 |
| not_found | 代码仓中尚无使用该场景的 kernel；`get-all-gather-quant-matmul` 当前仅 AllToAll GET hook（`block/aiv_comm/all_to_all/all_to_all_udma_get.h`）有源码记录，**AllGather GET hook 未见源码记录**，设计指导为原理推导，须经源码验证后方可采用 |

### 参考依据

- `put-all-to-all-quant-matmul`：官方参考算子 `all_to_all_quant_matmul`（K 轴切分、UDMA 直调范式来源）；本场景定位为该范式的**变体扩展**指导，非标准 PUT 需求的重复
- `put-all-gather-quant-matmul`：官方参考算子 `all_gather_quant_matmul`（AllGather+**PUT**、M 轴输入切分、FragmentTensor 三区范式来源）；本场景定位为该范式的**变体扩展**指导。注意与 `get-all-gather-quant-matmul` 的区别：官方 AG 是 PUT 模式（AIV 逐轮 PUT 到对端窗口），GET+AllGather 无任何官方实现
- `get-all-gather-quant-matmul`：官方暂无 GET 算子样例；GET 切分策略为原理推导（见 [`../workflow_integration.md`](../workflow_integration.md) Step 2 §切分策略）
- `compute-first-reduce-scatter`：已有生产实现（先本地计算、再 AllToAll PUT + 本地增量归约实现 ReduceScatter 语义），关键编排特征为 **compute-first + localLast 双 flag + 归约多核分治**（设计/开发合同见对应场景文档）。⚠️ 官方快照基准（`mc2/common/op_kernel/apace/`）中 ReduceScatter 仅为枚举占位（`collective_comm_api.h` 的 `CommCollectiveOp::ReduceScatter`，无 block 实现与 kernel 使用方）——本场景合同为自研编排，不与官方实现混淆

## 唯一匹配约束

1. 场景准入条件（语义判据）必须**互斥**：任意需求不得同时命中两个及以上场景的语义判据。
2. **语义零命中**（官方未覆盖项无场景可覆盖）→ 路线判 `unsupported`，阻塞 DESIGN.md。
3. **多命中** → 视为注册表缺陷或需求描述不清，判 `unsupported`，停止并等待澄清，禁止自行择优。
4. 命中后默认查阅命中场景的指导文档作为设计基线；跨场景拼接设计元素时必须在 DESIGN.md 中显式论证。

## 场景扩展

新增定制扩展场景时：

1. 在本目录下创建 `<scenario-id>/` 子目录（kebab-case，与表中场景 ID 一致）
2. 子目录内提供 `design.md`（设计指导）与 `development.md`（开发指导）
3. 在本表中登记一行，初始状态为 `planned`，文档就绪后改为 `已实现`
4. 登记前必须核对与已有场景的准入条件互斥，不满足唯一匹配约束的注册不予合入
