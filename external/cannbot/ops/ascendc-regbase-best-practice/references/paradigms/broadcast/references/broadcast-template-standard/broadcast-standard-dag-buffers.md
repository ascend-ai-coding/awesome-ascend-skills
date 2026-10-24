# broadcast 范式 standard DAG 分析和 Buffer 划分

> UB buffer 的划分方案，包括 DAG 图分析（主要是VF融合）、存活节点分析、double buffer 决策, 需要产出 VF融合后的DAG 图和 buffer 分类表。

## 1 DAG 分析和 Buffer 划分方案

### 1.1 存活节点分析

Broadcast 场景 buffer 划分采用 **L/P 双层分析模型**：

**逻辑存活节点（L）**：
- **参考案例**: `example/examples-adam-design/adam_apply_one_assign_design.md` §1
- 计数域：计算图 tensor（每个独立数据实体计 1）
- 由计算图拓扑决定，与物理媒介无关
- **L 对给定计算图是固定的**，不随硬件/API/实现改变

**物理存活节点（P）**：
- **参考案例**: `example/examples-adam-design/adam_apply_one_assign_design.md` §2
- 计数域：UB buffer。寄存器不在计数域——UB→Reg→UB 是过路中转，中间值留在寄存器时不占 P
- **P 不由计算图唯一确定**——同一计算图，不同 API 策略/融合决策 → 不同 P。寻找在给定约束下可达的**最小峰值**

**公平基线（L 和 P 共用）**：
1. 二元及以上操作，不假设 dst 与 src 复用——每个 op 的 src 和 dst 各占独立节点
2. 单步内不假设 in-place
3. 跨步的"用完释放"允许——一个节点的所有消费者执行完后从存活列表移除

**L 与 P 的关系**：P ≥ L 恒成立（MemBase 下每个 op 的 src 和 dst 各占独立 UB buffer）。VF 寄存器链融合可打破此规律：中间值全程在寄存器，不落 UB。L 仍包含（按 tensor 计数），P 不包含（寄存器不计入 P）。此时 **P < L 合法**。

**物理基线差异来源表**：

| 差异来源 | 公平？ | 说明 |
|---------|--------|------|
| 跨步 buffer 复用 | ✓ | 生命周期不重叠可共用物理 buffer，物理侧允许 |
| MulAddDst 硬件融合 | ✓ | 硬件保证 dst 就地覆写，计为 API 能力 |
| NDDMA 随路 broadcast | ✓ | DataCopy 阶段完成，不另占 buffer |
| 软件 StoreAlign 覆写 | ✗ | 逻辑侧/物理基线均不假设 in-place |
| VF 寄存器链融合 | ✓ | 中间值全程在寄存器 |

**持有法则**——逐步标注方法：

> 持有法则是通用分析工具，完整定义（读/写/原地读写规则、三态含义、从持有推导 RAW/WAR 同步点）见 [sync-and-consistency.md](../../../common/sync-and-consistency.md) §2.2.2。此处按 Broadcast 场景复述关键点。

每个执行动作标注执行前/中/后三态，验证峰值 ≤ P。

| 动作 | 流水线 | 对 UB 的行为 |
|------|--------|-------------|
| CopyIn | MTE2 | **写** buffer — 从 GM 搬入新数据 |
| Compute | V | **读** src buffer + **写** dst buffer |
| CopyOut | MTE3 | **读** buffer — 搬出到 GM |

- **写**：写入后 buffer 是否保留在持有中，取决于新数据有无未来消费者
- **读**：读本身不改变持有，但读完后该 buffer 少了一个未来消费者。如果是最后一个消费者，执行后 buffer 从持有消失
- **原地读写**（如 MulAddDst、Muls）：buffer 被读的同时也被写，按写处理

三态定义：

| 态 | 含义 |
|----|------|
| 执行前持有 | 上一动作执行后留下的持有列表 |
| 执行中持有 | 执行前 + 当前动作新占用的 buffer（dst 写入 + src 正在被读尚未释放）。P 压力最大时刻 |
| 执行后持有 | 执行中 − 当前动作后无未来消费者的 buffer |

前一动作的执行后 = 下一动作的执行前。第一步执行前为 `[]`。

示例（三元输入，CopyIn→Mul→Mul→CopyOut）：

```cpp
// S1: in0 → square（第一步，执行前为空）
// 执行前: 持有=[]
// 执行中: 持有=[UB0]                   ← MTE2 正在写 UB0
// 执行后: 持有=[UB0]                   ← in0 后面 S2 还会读, 有未来消费者, 保留
CopyOneInput(coord, IN0, UB0);
// 执行前: 持有=[UB0]
// 执行中: 持有=[UB0, UB2]              ← V 读 UB0, 写 UB2
// 执行后: 持有=[UB0, UB2]              ← in0 后续 S2 会读, square 后续 S4 会读, 都有未来消费者
AscendC::Mul(buf_[UB2], buf_[UB0], buf_[UB0], count);

// S2: in0·mul1_x → mul_1
// 执行前: 持有=[UB0, UB2]
// 执行中: 持有=[UB0, UB1, UB2, UB3]    ← V 读 UB0,UB1, 写 UB3
// 执行后: 持有=[UB2, UB3]              ← UB0(in0)和UB1(mul1_x)无未来消费者, 消失; square/mul_1 有, 保留
AscendC::Mul(buf_[UB3], buf_[UB0], buf_[UB1], count);

// CopyOut: 读 UB，对 UB 是最后一次读
// 执行前: 持有=[UB2, UB3, UB4]
// 执行中: 持有=[UB2, UB3, UB4]         ← MTE3 正在读 UB4
// 执行后: 持有=[UB2, UB3]              ← UB4 被 CopyOut 读后无未来消费者, 消失
CopyOutOne(coord, OUT0, UB4);
```

**验证**：任意执行中持有数 ≤ P；执行后持有与释放时机一致。

**Cast 对 P 的影响**：
- Cast src 和 dst 是两块独立 UB buffer（不能 in-place，FP16 2B vs FP32 4B，硬件不支持重叠读写）
- 并发放宽：若基线 P ≤ 6，可扩大 P（**最多 +2**），为 Cast 分配更多 buffer，减少多输入争抢同一 temp
- 按需取，用不到 2 时不强拉到上限

**RegBase 对 P 的影响**——三种降节点场景：
1. **常量/标量在寄存器构造**：标量不占 UB
2. **链式中间值留在寄存器**：Reg::Mul→Reg::Sub→Reg::Add 连续计算，中间结果全程在寄存器链上
3. **寄存器内 broadcast/replicate**：从 UB LoadAlign 一个元素到 Reg，寄存器内 replicate，省掉 broadcast 源在 UB 的独立驻留

不改变的情况：单次二元操作 src0/s1 本来就在 UB 中，RegBase 仍需 UB→LoadAlign→Reg compute→StoreAlign→UB，UB 需求与 MemBase 相同。

**RegBase vs MemBase 对比**：

| 维度 | MemBase (基础 API) | RegBase (Reg矢量计算) |
|------|------|------|
| 数据源 | `LocalTensor<T>` (UB) | `RegTensor<T>` (VF 寄存器) |
| 中间结果 | 每次计算写回 UB | 留在寄存器，不碰 UB |
| 处理粒度 | 一次处理完整 LocalTensor | 每次 VL 长度，需切分+循环 |
| 搬入搬出 | DataCopy (GM↔UB) | LoadAlign/StoreAlign (UB↔Reg) |
| 函数标记 | `__aicore__` | `__simd_vf__` / `__simd_callee__` |

内存层级：

```
GM →(DataCopy)→ UB →(LoadAlign)→ VF Reg →(Compute)→ VF Reg →(StoreAlign)→ UB →(DataCopy)→ GM
```

**RegTensor 不计入 UB 预算**：RegTensor 在寄存器，UB 只算 TBuf + Workspace。

**perBuffer ≥ 48KB 为可接受性能**的下限。

### 1.2 DAG VF融合分析

Broadcast 场景的计算流以 element-wise + broadcast 为核心。以二元算子 `out = f(x, y)` 为例，VF 融合后的 DAG 流图：

```
[段 1: CopyInBrc (NDDMA 随路 broadcast)]
  DataCopy<T, NDDMA_DIMS, cfg>(buf0, gmX[off], nddmaParams)   # x 搬入 UB0（broadcast 轴 stride=0）
  DataCopy<T, NDDMA_DIMS, cfg>(buf1, gmY[off], nddmaParams)   # y 搬入 UB1（broadcast 轴 stride=0）

[段 2: VF Compute (RegBase 寄存器链)]
  LoadAlign(srcReg0, buf0, aReg)                       # UB0 → Reg
  LoadAlign(srcReg1, buf1, aReg)                       # UB1 → Reg
  asc_vf_call<MyVF<T>>(buf2, buf0, buf1, count, ...)   # 计算结果 → UB2

[段 3: CopyOut (DataCopyPad)]
  DataCopyPad(gmOut[off], buf2, extParams)             # UB2 → GM
```

> **关键**：NDDMA 随路 broadcast 在 CopyIn 阶段直接完成广播，不另占 UB buffer，降低物理存活节点。
> RegBase 将中间结果留在 VF 寄存器链传递，不落 UB，进一步降低 P 值。

**VF 融合穷举方法论**：

列出计算图中**所有**连续 Vector 操作对，逐对判据。禁止只检查"看起来像"的那一对。方法：从计算图第一条 Vector 操作开始，相邻两两配对，每一对都走三条硬规则判据：

1. **链长 ≤ 7**——超过 7 条 Vector 操作必须拆为多次 VF 调用
2. **Cast 不参与 VF**——Cast 前后必须断链（Cast 不走 Vector 寄存器链路）
3. **RegBase 指令必须存在**——对应操作在 RegBase API 白名单中能找到

**多消费者不打断链**。即使某个中间结果被 VF 外的其他步骤引用，VF 链仍可继续——VF 内部把这个值 `StoreAlign` 到 UB 即可，寄存器链路不受影响。代价是多写一次 UB 会多占一个 P slot，需要纳入物理 trace 做峰值验证。

**连续可融对合并为单条 VF 链**：从第一条 Vector op 开始，只要 RegBase 指令存在、非 Cast、累积未超 7，链就延续。不因分叉而断链。

**存活节点压力——融合 vs 拆分取舍**：

多消费者融合虽然合法，但会增加 P 压力：VF 需要同时持有所有输入 buffer + 所有输出 buffer（包括为外部消费者多写的）。如果合并后峰值超过目标 P，则需要**主动拆分** VF 来降压。

判据流程：
1. 先推满最长链（≤7）
2. 做物理 trace，检查峰值是否 ≤ 目标 P
3. 若超 P → 在分叉输出最多的位置拆分 VF，释放部分 buffer 后再续链
4. 重复直到峰值 ≤ P

**融合改变存活节点大小**：融合前每个中间结果各占 1 个 UB buffer。融合后中间结果在寄存器内不进 UB——存活节点 = 所有输入操作数 + 最终输出 + 其他还活着的 buffer。

```
A+B+C+D = E

全融 A+B+C+D： 5 个节点（A, B, C, D, E 同时活着）
融 A+B+C：     4 个节点（A, B, C, TMP1 活着）→ 再 TMP1+D = E
只融 A+B：     3 个节点（A, B, TMP1）→ 再 TMP1+C = TMP2 → TMP2+D = E
```

**存活节点扩大限制**：融合后的物理存活节点数不得超过融合前的 P + 3。超过 → 只部分融合，或调整调度先释放不再需要的 buffer。

**硬件融合指令**（一条指令完成两步，省 1 buffer）：`MulAddDst, FusedMulAdd, AddRelu, SubRelu, FusedMulAddRelu, MulCast`。

**寄存器链组合**（两条或多条 RegBase 指令串在 VF 循环内完成）：`Compare + Select`, `Mul + Add`, `Ln + Mul`, `Exp + Sub` 等。

### 1.3 额外buffer要求

- Cast 场景：FP16/BF16 输入时必须额外分配一个 Cast 中转 buffer（src/dst 不可 in-place）
- 多输出算子：若 >1 个输出，每个输出需独立 CopyOutOne 调用，输出 buffer 在 CopyOut 前不可被覆写
- Buffer 复用：同一物理 buffer 可多次复用，只要持有数 ≤ P 约束。Process 代码中 buffer 角色变更须标注 `// UBx 释放` / `// UBx 重新分配为{新角色}`

### 1.4 Double Buffer 决策

- **不开启 DB**：Broadcast 算子使用 TBuf 单缓冲，Copy 和 Compute 串行，由 Mutex 临界区串行化保序（见 [broadcast-standard-kernel-template.md](broadcast-standard-kernel-template.md) §7）
- TBuf 无自动同步，所有同步手动插入。单缓冲是 TBuf 相比 TQue 的核心优势——每个 buffer 就是 UB 上一块固定大小的空间，先做 CopyIn、再做中间结果、最后做 CopyOut，动态复用
- TBuf vs TQue 对比：

| | TBuf | TQue |
|----|------|------|
| 缓冲 | 单缓冲 | 双/多缓冲 |
| 角色锁定 | 无 | EnQue/DeQue 锁定读写角色 |
| 同步 | 手动 Mutex Lock/Unlock（或事件对） | 自动（TQue 内部管理） |
| 生命周期 | 程序员控制 | EnQue 提交, DeQue 消费后 FreeTensor |
| 适用场景 | 精控 buffer 复用（P 约束紧） | 流水线简单、不需精控 |

- Adam 为何用 TBuf：P=5 约束下需要精确复用 5 个 buffer。TQue 自带双缓冲，5 个 TQue 会变成 10 个 buffer，`perBufBytes` 减半，每轮处理数据量砍半。TBuf 单缓冲让 `perBufBytes = UB/5`（对齐 32B 后）最大化 tile 大小

**TBuf 关键约束**：
- **无自动同步**：TBuf 只是一块裸 UB，MTE2/V/MTE3 三流水线访问它必须手动同步（Mutex 临界区串行化，见 kernel-template §7）
- **单缓冲覆盖风险**：同一个 TBuf 多次 Get，旧数据会被新操作覆盖。必须通过持有法则确保覆盖不发生冲突
- **配对数固定**：通过 InitBuffer 分配的 TBuf 数量在 Init 阶段确定，不能运行时增减

**TBuf 声明与初始化**：

```cpp
// 成员变量
TBuf<TPosition::VECCALC> buf_[5];

// Init() 中分配
for (int i = 0; i < 5; i++)
    pipe_.InitBuffer(buf_[i], td_->perBufBytes);
```

`perBufBytes` 和 `Get<T>(count)` 中的 `count * sizeof(T)` 都必须 **blocksize 对齐**（硬件 `ONE_BLK_SIZE = 32`，经 `Ops::Base::GetUbBlockSize` 获取，不要写死）。`perBufBytes = floor(UB / P / ubBlockSize) * ubBlockSize`。

使用方式：
```cpp
// 方式 A: Get<T>() 无参，取全量 buffer
LocalTensor<T> buf = buf_[buffer].Get<T>();
// 方式 B: Get<T>(count) 指定元素个数，count * sizeof(T) 须 32B 对齐
LocalTensor<T> buf = buf_[buffer].Get<T>(count);
// 方式 C: GetWithOffset(size, offset) 取 buffer 子区域
LocalTensor<T> buf = buf_[buffer].GetWithOffset<T>(size, bufOffset);
```

## 2 分析结果

## 2.1 DAG 图分析结果

Broadcast 场景数据流（以二元算子为例）：

```
GM(x/y) ──NDDMA──→ UB(B0/B1) ──LoadAlign──→ Reg(vregX/vregY)
                                                  │
                                           VF Compute
                                           (asc_vf_call)
                                                  │
                                           StoreAlign
                                                  │
UB(B2) ←─Store─── Reg(vregOut) ←─────────────────┘
  │
  └──DataCopyPad──→ GM(out)
```

多输出算子增加并行 CopyOut 分支。

## 2.2 Buffer 划分产出

| 类别 | dtype | DB | 份数 | 用途 |
|------|-------|----|------|------|
| buf_[0..P-1] | D_T | 关 | P × 1 | P 个 TBuf 槽位，按 perBufBytes 均分 UB |

**perBufBytes** = `(UB / P) & ~(ubBlockSize - 1)`（P 个 buffer 均分 UB；ubBlockSize 经 `Ops::Base::GetUbBlockSize` 获取，当前平台 32B 对齐）

**perBufElems** = `perBufBytes / sizeof(float32)`，**永远按 FP32 算**，不随输入 dtype 变化。原因：FP16 CopyIn 后立即 Cast 到 FP32 data buffer——真正参与计算、填满 buffer 的数据是 FP32。即使 FP32 输入不经过 Cast，为了 TilingData 统一（perBufBytes 唯一），也按同一公式。

**关键约束**：P 值来自 §1.2 P trace 结论表，`perBufBytes` 必须满足 `perBufElems ≥ effective_shape[split.ubSplitIdx]` 的内轴需求。

## 3 Buffer 划分产出校验

**对范式开发者的要求**：
- 1. 给一个复杂输入的算子样例，以实际划分结果作为 golden，需要"人"保证正确性。
- 2. golden 如果有多个，需要全部列出
- 3. 删除 golden, agent读取本文档可以产出同样的buffer划分，如果产出不了，需要不断改进范式质量，直到成功
- 4. 样例随本章节上库

**golden 样例（Adam: 5 输入 3 输出, fp32, P=5）**：

Adam 计算流：`square=IN0*IN0 → mul_1=square*MUL1 → mul_0=MUL*IN2+mul_1(MulAddDst) → mul_3=square*mul_0*MUL3 → mul_4=mul_3*IN1(MulAddDst) → sqrt=sqrt(mul_4) → add_2=sqrt+ADD2 → truediv=1/add_2 → mul_4=truediv*IN4 → output2=IN3-mul_4`

P trace 峰值出现在 S7b(Add) 和 S9b(Mul) / S10b(Sub)，持有=[UB0, UB1, UB2, UB3, UB4]，**P=5**。

| 类别 | dtype | DB | 份数 | 用途 |
|------|-------|----|------|------|
| buf_[0..4] | float | 关 | 5 × 1 | 5 个 TBuf 槽位，动态复用（CopyIn/中间结果/CopyOut 角色轮转） |
| perBufBytes | — | — | — | (UB / 5) & ~(32-1) = (262144 / 5) & ~31 = 52416 bytes（ubBlockSize=32） |
| perBufElems | — | — | — | 52416 / 4 = 13104（按 FP32 算） |
