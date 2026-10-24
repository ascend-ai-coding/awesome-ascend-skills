# FlashAttention 类算子 — 实现层参考(跨核同步 / Fixpipe / 平台参数)

> 本文档承接 [`design/`](./design/_governance.md) 的**设计决策**(WHAT,节点 G→N12),提供进入实现时的**具体实现参考**(HOW-to-implement):跨核同步原语的模板参数、PIPE 映射、平台 modeId 取值、Fixpipe 参数结构体。
> **定位**:`design/` 各节点只回答"该决策什么";本文件提供"选定后如何落地"。设计阶段无需加载本文件,进入实现阶段再查。
> **来源纪律**:本文所有 API 签名、modeId 语义、flagId 规则、Fixpipe 参数均取自 **asc-devkit 公开文档**(路径见各节末尾)。实现时应以目标 CANN 版本的实际头文件与文档为准。

---

## §1 平台 modeId 取值(先确定目标平台)

跨核同步用 `CrossCoreSetFlag<modeId, pipe>` / `CrossCoreWaitFlag<modeId, pipe>`(ISASI 公开 API)。`modeId` 决定同步范围:

| modeId | 语义 | 适用 |
|--------|------|------|
| 0 | AI Core 核间同步(所有 AIC 之间 / 所有 AIV 之间)| 纯 Cube / 纯 Vector 全核同步 |
| 1 | AI Core 内部,两个 AIV 之间 | AIV↔AIV |
| 2 | AI Core 内部,AIC 与**所有** AIV 之间(广播式:AIC set 后两个 AIV 都放行;两个 AIV 都 set 后 AIC 才放行)| 混合场景全 AIV 参与 |
| 4 | AI Core 内部,AIC 与**单个** AIV 之间(AIC set 后 AIV0 或 AIV1 放行;AIV0 或 AIV1 set 后 AIC 放行)| 混合场景单 AIV 独立握手 |

> ⚠️ **平台支持差异**(asc-devkit 公开文档明确):
> - **Ascend 950PR / 950DT**:支持 modeId 0 / 1 / 2 / 4
> - **上一代产品(如 A2/A3)**:仅支持 modeId 0 / 1 / 2(**不支持 4**)
>
> FA 类是 AIC+AIV 混合 kernel,核内 AIC↔AIV 握手在 950PR 上用 **modeId=2 或 4**;若目标是 950PR 且希望 AIC 与单个 AIV 独立握手,用 **modeId=4**。**进入实现前必须先确认目标平台支持的 modeId 集合**,照抄不支持的 modeId 会编译/运行失败。

**modeId=4 下的双 AIV flagId 映射**(asc-devkit 公开文档 `flagId取值范围说明`,950PR/950DT):

```
AIV0 set flagId 0-10  ↔  AIC wait flagId 0-10
AIV1 set flagId 0-10  ↔  AIC wait flagId 16-26   (AIV1 相对 AIV0 偏移 16)
AIC  set flagId 0-10  ↔  AIV0 wait flagId 0-10
AIC  set flagId 16-26 ↔  AIV1 wait flagId 0-10
```

即 modeId=4 下 AIC 与两个 AIV 分别握手,AIC 侧针对 AIV1 的 flagId 需加 **16 偏移**。这与 modeId=2 的"广播式"(AIC 一次 set 通知所有 AIV)语义不同。

> **来源**:`asc-devkit/docs/api/SIMD-API/基础API/同步控制/核间同步/CrossCoreSetFlag(ISASI).md`(modeId 语义、modeId 支持取值、flagId 取值范围)。

---

## §2 flagId 冲突与 pipe 约束(asc-devkit 公开文档)

> ⚠️ **flagId 冲突风险**(公开文档 `CrossCoreSetFlag(ISASI).md` 明确):
> - **Matmul 高阶 API 内部使用 CrossCoreSetFlag**——不建议同时使用 CrossCoreSetFlag 与 Matmul 高阶 API,否则 flagId 冲突。Matmul 内部占用 flagId 范围 `[0, 2N-1]`(N=Matmul 对象数,最多 4 个 → 占用 `[0,7]`)。**此占用仅在实例化 Matmul 高阶对象(`matmul::Matmul<...>` / `REGIST_MATMUL_OBJ`)时成立;直接调用裸 `Mmad(...)` Cube 指令不占用任何 flagId**(裸指令无内部握手)。故 `[0,7]` 是否需规避,取决于本 kernel 用的是高阶 Matmul 对象还是裸 Mmad 指令。
> - **SyncAll 硬件同步接口内部也使用 CrossCoreSetFlag**——占用 flagId `[11,14]`,**同样仅在实际调用 `SyncAll` 时成立**;未调用则该区间可用。
>
> 规避原则:先确认本 kernel 实际使用了哪些内部占用 flagId 的 API(高阶 Matmul 对象 / SyncAll),仅规避**实际被占用**的区间——不要因"FA 用 Mmad"就无条件规避 `[0,7]`(裸 Mmad 不占)。若拿不准,规避两区间是安全的保守做法,但应知其可能过度保守。

**flagId 取值范围**:
- modeId 0/1/2:`0-15`
- modeId 4(950PR/950DT):见 §1 的 AIV0/AIV1 映射(0-10 与 16-26)

> ⚠️ **不同生命周期事件必须用不同 flagId**。每个 flagId 对应一个**计数器**(set 加一 / wait 减一,见 §3 与 `CrossCoreSetFlag(ISASI).md`),不是电平。若把「跨 task 边界」事件与「loop 内 stage(如 V1 P-ready)」事件复用同一 flagId,边界 wait 会被**残留的 loop-stage token 提前满足** → task 间 workspace 提前复用 → **非确定性覆写**(高竞争 multi-task-per-core 场景才暴露,单 task 测不出)。规则:跨 task 边界 flag 与 loop 内 stage flag 分配**互不相同**的 flagId,且各自都避开 §2 上文**实际被占用**的区间(高阶 Matmul 对象 → [0,7];调用 SyncAll → [11,14];裸 Mmad 指令两区间均不占,可自由使用)。

**pipe 模板参数约束**(公开文档 `pipe支持的流水类型说明`):
- **modeId 0/1/2**:支持 `PIPE_V / PIPE_M / PIPE_MTE1 / PIPE_MTE2 / PIPE_MTE3 / PIPE_FIX`;**不支持 `PIPE_ALL` / `PIPE_S`**。
- **modeId 4(950PR/950DT)**:在上述基础上**额外支持 `PIPE_S`**;仅不支持 `PIPE_ALL`。
- 950PR/950DT 上 modeId 与 pipe 模板参数**生效**,`CrossCoreWaitFlag` 阻塞**指定流水**的后续指令;上一代产品上二者不生效,阻塞**全部流水**。

> **来源**:`asc-devkit/docs/api/SIMD-API/基础API/同步控制/核间同步/CrossCoreSetFlag(ISASI).md`、`CrossCoreWaitFlag(ISASI).md`。

---

## §3 跨核同步实现骨架(承接 design/execution.md §3.2 的架构模式选择)

> [`design/execution.md §3.2`](./design/execution.md) 已完成两个正交决策:**架构模式**(单/双 AIV)+ **平台 modeId**(见 §1)。本节用 asc-devkit 公开原始 API `CrossCoreSetFlag/WaitFlag` 给出两种架构模式的实现骨架。

**公开 API 签名**:
```cpp
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId);

template <uint8_t modeId = 0, pipe_t pipe = PIPE_S>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId);
```

> ⚠️ **必须显式指定模板参数**。`CrossCoreWaitFlag` 的默认 `modeId=0` 会等待 modeId-0 的 flagId 空间;若配对的 `CrossCoreSetFlag` 用了 `modeId=4`,两套 flagId 计数器**互相独立**,WaitFlag 永远等不到 → **死锁**。Set 与 Wait 两侧的 `modeId` 必须一致。

### §3.1 模式 B:双 AIV 全参与(充分利用 Vector 算力)

数据流:AIC 通过 Fixpipe(见 §4)将 Cube 输出分发到两个 AIV 的 UB,双 AIV 全程参与计算。modeId=4 下 AIC 需分别与 AIV0、AIV1 握手。

```cpp
// 平台: 950PR → CC_MODE = 4; A2/A3 → CC_MODE = 2
constexpr uint8_t CC_MODE = 4;
constexpr uint16_t AIV1_FLAG_OFFSET = 16;   // modeId=4 下 AIC↔AIV1 的 flagId 偏移

// --- AIC 侧(生产方):算完 Cube → 通知两个 AIV ---
// ... Fixpipe: L0C → UB ...
AscendC::CrossCoreSetFlag<CC_MODE, PIPE_FIX>(flagId);                    // 通知 AIV0
AscendC::CrossCoreSetFlag<CC_MODE, PIPE_FIX>(flagId + AIV1_FLAG_OFFSET); // 通知 AIV1

// --- AIV 侧(消费方):等 AIC → 读 → 回通知 ---
AscendC::CrossCoreWaitFlag<CC_MODE, PIPE_V>(flagId);   // 各 AIV 用自己的 0-10 flagId
// ... softmax / accumulate ...
AscendC::CrossCoreSetFlag<CC_MODE, PIPE_MTE3>(flagId); // 回通知 AIC

// --- AIC 等两个 AIV 完成 ---
AscendC::CrossCoreWaitFlag<CC_MODE, PIPE_MTE2>(flagId);                    // 等 AIV0
AscendC::CrossCoreWaitFlag<CC_MODE, PIPE_MTE2>(flagId + AIV1_FLAG_OFFSET); // 等 AIV1
```

> 注:上例常量 `CC_MODE` / `AIV1_FLAG_OFFSET` 为算子自定义命名,值(4 / 16)来自 §1 公开文档规则。是否封装成 Buffer 管理类由实现自行决定;asc-devkit 公开 API 层面即上述 `CrossCoreSetFlag/WaitFlag`。

### §3.2 模式 A(简化替代):单 AIV + AIV1 early-return

数据流:AIC 只与 AIV0 握手,AIV1 在 Process() 入口 early-return。无需 AIV1 偏移。

```cpp
constexpr uint8_t CC_MODE = 4;  // 950PR

// AIC → AIV0
AscendC::CrossCoreSetFlag<CC_MODE, PIPE_FIX>(flagId);
// AIV0 等待
AscendC::CrossCoreWaitFlag<CC_MODE, PIPE_MTE2>(flagId);
// AIV0 → AIC
AscendC::CrossCoreSetFlag<CC_MODE, PIPE_MTE3>(flagId);
// AIC 等待(仅一次)
AscendC::CrossCoreWaitFlag<CC_MODE, PIPE_MTE2>(flagId);
```

- **AIV1 行为**:必须在 Process() 入口 `if (GetSubBlockIdx() != 0) return;` early-return,否则其 WaitFlag 等不到信号 → 挂死。
- **代价**:浪费一半 Vector 算力。

### §3.3 PIPE 映射约定(FA 4-stage)

| 方向 | 场景 | SetFlag PIPE | WaitFlag PIPE |
|------|------|-------------|---------------|
| AIC → AIV | Cube 完成(L0C→UB Fixpipe)通知 Vector | `PIPE_FIX` | `PIPE_V` 或 `PIPE_MTE2` |
| AIV → AIC | Vector 完成(P 写 GM/L1)通知 Cube | `PIPE_MTE3` | `PIPE_MTE2` |

pipe 必须落在 §2 允许集合内(不含 `PIPE_ALL` / `PIPE_S`)。

### §3.4 架构模式决策建议

| 场景 | 推荐架构 | 平台 modeId | 理由 |
|------|---------|-----------|------|
| FA/GQA(A5/950PR),追求吞吐 | 模式 B | 4 | 双 AIV 充分利用 Vector 算力 |
| FA/GQA(A2/A3)| 模式 B | 2 | 上一代仅支持 modeId 0/1/2 |
| 快速原型验证 | 模式 A | 按平台 | 代码更简单,但浪费一半 Vector 算力 |

---

## §4 Fixpipe 输出参数(承接 design/execution.md §3.5)

Fixpipe(L0C→UB/GM)的参数结构体**因平台而异**(asc-devkit 公开文档 `L0C到UB数据搬运（Fixpipe）.md`):

| 平台 | 参数结构体 |
|------|-----------|
| Ascend 950PR / 950DT | `FixpipeParamsArch3510<config.format>` |
| Atlas 200I/500 A2 推理产品 | `FixpipeParamsM300` |

**Fixpipe 随路能力**(950PR/950DT,公开文档):搬运过程中支持随路格式转换(F32→F16 / F32→BF16 为纯 Cast,非量化)、随路量化(scalar/tensor)、随路 ReLU、随路通道合并。

**FA 用法要点**:
- C1 的 S 矩阵、C2 的 PV 矩阵均需保持 **fp32** 输出到 UB(V1/V2 在 UB 上做 fp32 计算),**不使用随路 Cast**。
- 若需把 Cube 输出分发到双 AIV,依平台参数结构体的相应字段配置(具体字段名以目标 CANN 版本 `FixpipeParamsArch3510` 文档为准)。

> **来源**:`asc-devkit/docs/api/SIMD-API/基础API/矩阵计算（ISASI）/矩阵计算的搬出/L0C到UB数据搬运（Fixpipe）.md`;参数结构体字段见 `Fixpipe搬运参数（FixpipeParamsArch3510、FixpipeParamsM300）结构体说明`。

---

## §5 清单 3(cross-core sync 时序)—— 跨核同步实现自检清单

> 本清单即通用自检 **清单 3(cross-core sync 时序)**,由 [`design/execution.md §3.2`](./design/execution.md) 的跨核同步决策引出。各变体 Self-Check 中"继承 清单 3"即指本节。

**通用**:
- [ ] AIV → AIC 的通知在数据写完之后
- [ ] AIC → AIV 的等待在数据读取之前
- [ ] 同核内 pipe 同步用 `SetFlag/WaitFlag`,跨核用 `CrossCoreSetFlag/WaitFlag`,不混用
- [ ] GM 自读自写段有明确的同步机制
- [ ] 跨阶段依赖用对应 `HardEvent`,不用 `PipeBarrier` 替代
- [ ] PipeBarrier 冗余率 < 30%(见 design/execution.md §3.4)

**模板参数(最易踩坑,GQA 死锁根因)**:
- [ ] `CrossCoreSetFlag<modeId, pipe>` 与配对的 `CrossCoreWaitFlag<modeId, pipe>` **modeId 一致**(不能一侧显式 4、另一侧用默认 0)
- [ ] `CrossCoreWaitFlag` 显式指定了 `modeId` 与 `pipe` 模板参数(默认 modeId=0 会等错 flagId 空间)
- [ ] pipe 未使用 `PIPE_ALL` / `PIPE_S`
- [ ] 目标平台确实支持所选 modeId(950PR 支持 0/1/2/4;A2/A3 仅 0/1/2)

**flagId 管理**:
- [ ] 用户自定义 flagId 避开 Matmul 高阶 API 占用区间(`[0, 2N-1]`,最多 `[0,7]`)与 SyncAll(`[11,14]`)
- [ ] 同一 flagId 计数器设置次数在允许范围内
- [ ] modeId=4(950PR)下 AIC↔AIV1 的 flagId 已加 16 偏移

**架构模式**:
- [ ] 模式 B:AIC 对 AIV0/AIV1 分别 Set/Wait(modeId=4 下两次,AIV1 加 16 偏移);AIV1 **不** early-return
- [ ] 模式 A:AIV1 在 Process() 入口 early-return;AIC 只 Set/Wait 一次

---

## §6 参考资源

| 主题 | 入口 |
|---|---|
| 设计决策(WHAT)| [`design/execution.md`](./design/execution.md) §3 |
| CrossCoreSetFlag/WaitFlag 公开文档 | `asc-devkit/docs/api/SIMD-API/基础API/同步控制/核间同步/` |
| Fixpipe(L0C→UB)公开文档 | `asc-devkit/docs/api/SIMD-API/基础API/矩阵计算（ISASI）/矩阵计算的搬出/` |
| 平台差异 / 架构基础 | `/npu-arch` |
| Ascend C API 用法(Mmad / LoadData / Fixpipe / CrossCore)| `/ascendc-api-best-practices` |
| 通用编码规范 / 入口属性 | `ops-direct-invoke` plugin `workflows/development-guide.md` |
