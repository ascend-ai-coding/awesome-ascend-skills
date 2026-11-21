# A2/A3(`arch22`)FlashAttention 实现手册 — 跨核排布最小 bring-up

> **适用范围**:Ascend 910B3(A2/A3 代际,`arch22` / `DAV_2201`);非 causal MHA;fp16 / bf16。
> **不适用**:Ascend 950(A5,`arch35`);causal / GQA / MLA / paged KV / 量化 / 多卡;跨代际、跨变体不继承本文取值。
> **性质**:单实现实证手册,不承担精度或性能的验收口径。
> **术语**:S / P / PV、`PRELOAD_N`(领先拍数)、节拍等以 [`patterns.md`](../patterns.md) 与 [`fundamentals.md §0`](../foundation/fundamentals.md) 为准;符号名一律视为角色名。
> **最后核对**:2026-11-10。

---

## §1 最小目标

一条用例、一个 KV 分块、一个 task:

```
AIC: 读 Q/K → S = Q·Kᵀ → Fixpipe 到跨核握手段 → set S_READY
AIV: wait S_READY → softmax(S) → P 写入握手区 → set S_FREE, set P_READY
AIC: wait P_READY → PV = P·V(L0C)→ Fixpipe 到握手区 → set PV_READY
AIV: wait PV_READY → 按行除以行和 → 写 attentionOut → set PV_FREE
```

通过标准:输出过精度门;在外部超时内正常结束;无 `AICORE_EXCEPTION`;确定性检查通过。

这一章之所以单独存在:cube 只解决两次矩阵乘,softmax 在 AIV,两侧必须通过 GM + 跨核 flag 形成流水。没有这一步,正确性可能已经通过,但性能上不去,或者首次 launch 直接挂死。

**先单 case、单 KV 分块、`__mix__(1, 1)`、单槽位跑通,再谈 `__mix__(1, 2)` 与多槽。**

---

## §2 flag 语义(本机已验证)

AIC↔AIV 的数据流全部用 **modeId 2**(广播式:AIC set 后所有 AIV 放行;所有 AIV set 后 AIC 才放行):

```cpp
// 生产者(AIC)通知消费者(AIV):数据已就绪
AscendC::CrossCoreSetFlag<2, PIPE_FIX>(producerNotifyConsumerId);   // S / PV 经 Fixpipe
AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(producerNotifyConsumerId);  // P 经 MTE3
AscendC::CrossCoreWaitFlag<2>(producerNotifyConsumerId);

// 消费者(AIV)通知生产者(AIC):槽已空
AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(consumerNotifyProducerId);
AscendC::CrossCoreWaitFlag<2>(consumerNotifyProducerId);
```

**三条流的收发方向(不要按"消费者永远是 AIV"记)**

| 流 | 生产者 → 消费者 | 数据就绪谁 set | 槽已空谁 set | 空槽信用谁预置 |
|---|---|---|---|---|
| S 握手段 | AIC → AIV | AIC(经 Fixpipe) | AIV(经 MTE2) | **AIV**(消费者) |
| P 握手段 | AIV → AIC | AIV(经 MTE3) | AIC(经 MTE2 / Fixpipe) | **AIC**(消费者) |
| PV 握手段 | AIC → AIV | AIC(经 Fixpipe) | AIV(经 MTE2) | **AIV**(消费者) |

**modeId 2 的信用语义是"广播 + 全到齐"**:AIC 侧 set 一次即放行**所有** AIV 子核;反过来**所有 AIV 都 set 之后,AIC 才被放行一次**。由此直接得出:AIV→AIC 方向上少一个子核 set,AIC 就永远等不到。

规则:

- **一条数据流一个 flagId**;S / P / PV 各一对(数据就绪 + 槽空闲),不要复用同一 flagId。
- **先预置空闲信用**:消费者在启动时对每个槽位发一次"槽空闲",生产者第一次等待才不会挂。注意**消费者不永远是 AIV**——P 流的消费者是 AIC,预置必须由 AIC 做。
- **产出经过哪条 pipe 就用哪条**:Fixpipe 产出的 S / PV 用 `PIPE_FIX`,MTE3 产出的 P 用 `PIPE_MTE3`。Set 侧的 pipe 会编进指令(实测反汇编:`PIPE_FIX` → `SET_CROSS_CORE.F`、`PIPE_MTE3` → `.MTE3`、`PIPE_S` → `.S`),**写错不会被忽略**,有两种结局:① AIC 上不存在的 pipe(如 `PIPE_S`) ⇒ 非法指令,`aclrtSynchronizeStream` 返回 `507015`、首例即挂;② 合法但挂错数据路径(数据由 Fixpipe 写出却 set 在 `PIPE_MTE2`) ⇒ **不报错、不挂死,静默读脏数据**,且间歇(两次运行的失败集合不同),只有精度门能抓到。
- **Wait 侧不适用上面这条**:本代 `CrossCoreWaitFlag<modeId, pipe>` 的 `modeId` / `pipe` 被编译掉(`(void)modeId; wait_flag_dev(flagId)`),任何形态都阻塞整条指令流 ⇒ 不存在"用错 pipe 就等不到"这种结局;等不到只可能来自计数、flagId 或 modeId 不配对。
- **不要混用不同 modeId**:本实现的 AIC↔AIV 队列只用 modeId 2。modeId 4 是 950PR 的单 AIV 独立握手,上一代架构不支持。
- flagId 必须在 AIC / AIV 两侧同源(host tiling 或编译期常量),不要一侧用 0、另一侧用 1。

---

## §3 子核信用(最容易挂死的地方)

- 先用 `__mix__(1, 1)`:只有一个 AIV 子核,set / wait 的信用是 1:1,最简单。
- 切到 `__mix__(1, 2)` 之后:两个 AIV 子核都会执行 AIV 段。若只有主核干活、副核提前返回,信用可能按两个子核计数,AIC 会等一个永远不来的 set ⇒ 挂死。
- **两个子核都必须参与 set / wait**:modeId 2 下 AIV→AIC 方向要求全部子核到齐才算一次信用。
- **"只让主核收发"不是可选项**:它只在 modeId 4 的单 AIV 独立握手下成立,而上一代架构不支持 modeId 4 ⇒ 这么做**必然挂死**(表现为首轮或第 N 拍之后再无进展)。
- **先用 `__mix__(1, 1)` 验证,再上 `__mix__(1, 2)`**;每加一对 flag 单独验一次,不要一次上三对。
- **子核号与核号是两件事**:子核号用 `GetSubBlockIdx()`(0/1);核号用 `GetBlockIdx() / GetSubBlockNum()`(AIV 侧 `GetBlockIdx()` 取 0 … 2N-1、`GetSubBlockNum()` = 2,AIC 侧 `GetBlockIdx()` 取 0 … N-1、`GetSubBlockNum()` = 1)。行索引用 `rowStart = 子核号 × 每子核行数`;`GetBlockIdx()` 既不能当行号也不能当子核号。

---

## §4 L0B 复用:C1 的 K 与 C2 的 V

- L0B 只有一块时,C1(K)与 C2(V)复用同一块;**C1 的 `Mmad` 读完之前不能覆盖**。
- 在装载 V 之前:

```cpp
SetWaitFlag<HardEvent::M_MTE1>();
PipeBarrier<PIPE_M>();
// then LoadData(bL0, vL1, ...);
```

- 不要只做 `M_MTE1` 的 set 而不等:`Mmad` 是异步的,必须等 M 管排空。
- 若 C1 / C2 使用不同的 L0B 块,可以不等;但不要为省一块而引入静默竞态。
- **每个复用 L0B 的拍都要等一次**:只在第一拍做,后面仍会概率性出错。
- 失败指纹:概率性 `AICORE_EXCEPTION`(搬运通道报 L0B 读写冲突)+ 输出整块 0 + 单 case 隔离跑过、全量跑挂。**先查 L0B 复用,不要先怀疑 workspace / DDR 可见性。**

---

## §5 单 case + 超时协议

1. 只跑一条小 case(单 task、单 KV 分块),加**外部超时**兜底。
2. 挂死时按顺序砍:
   - `__mix__(1, 1)` + 单 KV 分块 + 单槽位;
   - 只保留 S 的一对 flag(AIC→AIV→AIC),P / PV 暂不跨核;
   - 只让 AIC 写 S,AIV 读 S 后写一个标记到 GM,AIC 读该标记;确认 flag 通路本身能通。
3. 每次只加一条流(S → P → PV),不要一次上三对。
4. 单 case 通过后,再放到全量精度集上跑;**不要反过来**。

### 挂死排查表

| 现象 | 先查 |
|---|---|
| 第一次 launch 不返回 | 空闲信用是否预置;flagId 两侧是否同源;modeId 是否一致 |
| AIC 等"槽空闲"不来 | 消费者是否在启动时按槽位数发了空闲信用;槽位数与生产者预期是否一致 |
| AIV 等"数据就绪"不来 | 空闲信用是否预置;flagId 两侧是否同源;modeId 是否一致;Fixpipe 是否真的执行 |
| 报 `507015` / `Illegal instruction` | Set 侧是否挂了 AIC 不存在的 pipe(典型 `PIPE_S`) |
| 不报错但数值间歇性错 | Set 的 pipe 是否覆盖真实写出路径(典型:Fixpipe 写出却 set 在 `PIPE_MTE2`) |
| `1,1` 通过、`1,2` 挂 | 副核是否提前返回;信用是否按 2 计;先退回 `1,1` |
| 单分块通过、多分块挂 | 槽复用 / WAR;每槽的 data 与 free 是否成对;L0B 是否等 `M_MTE1` |
| 数值对但性能差 | 每拍全排水 / 单缓冲;先把 set / wait 拆成生产侧与消费侧,再加双缓冲 |

---

## §6 从 1 个分块到完整流水

1. **`1,1` + 单分块 + 单槽位**:S / P / PV 三对 flag 逐个加,每加一对跑一次。
2. **`1,1` + 多分块 + 两个槽位**:`head` / `tail` 指针、空闲信用、WAR;保持确定性检查通过。
3. **`1,2`**:两个 AIV 子核都参与;确认信用计数;再跑全量精度集。
4. **性能结构**:S / P 走 GM 环、PV 常驻 L0C、set / wait 拆分、双缓冲;每一步都用同一套测量口径确认收益,但**不要在架构未完整时用低分当基线**。

---

## §7 通过清单

- [ ] 单 case、单分块、单槽位:数值正确、`__mix__(1, 1)` 正常结束
- [ ] 三对 flag 逐个加,每次都保持确定性
- [ ] `1,2` 下两个子核都参与,信用配平
- [ ] 长循环(多分块 × 长 KV)不挂死、不漂移
- [ ] 挂死排查表逐行能对上现象
