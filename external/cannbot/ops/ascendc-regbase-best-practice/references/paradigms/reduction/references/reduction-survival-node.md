# 存活节点 (Survival Node)

> 算子 Tiling 设计的核心概念。决定 UB buffer 划分，直接影响 tile 大小和性能。

---

## 公共概念

**定义**：在计算或执行的某一时刻，必须同时驻留的独立数据实体（tensor 或 buffer）的数量。对一个算子，取其最优调度下的**峰值**（不是 allocated 总数）作为存活节点数。

**大小 = 峰值**。逐一追踪每步的执行中持有数，取全流程最大值。

**公平基线（L 和 P 共用）**：
1. 二元及以上操作，不假设 dst 与 src 复用——每个 op 的 src 和 dst 各占独立节点
2. 单步内不假设 in-place
3. 跨步的"用完释放"允许——一个节点的所有消费者执行完后从存活列表移除

**追踪方法（L 和 P 共用）**：
1. 画出完整计算图（含 reduce）
2. 识别分叉结构（同一数据被多处引用）
3. 按最优调度逐步追踪每步的**执行中**存活数
4. 取全流程最大值 = 存活节点数
5. **互斥分支流（Tuple Reduce）**：按代码结构画一条流——公共步骤出现一次，过程独有步骤用 if/else 分支并列；分支互斥，一处 if/else 的峰值取各分支 max（禁止相加、不按分支单独记账），其余按标准存活区间判定。规则对 L、P 均适用；"不按分支单独记账"只作用于 P，L 仍按公平基线逐 op 计节点。

---

## 逻辑存活节点 (L)

**特有计数域**：计算图 tensor（每个独立数据实体计 1；跨阶段驻留的 tensor 归属其分配阶段，不计入其他阶段——如 reduceResult 驻 cacheBuf，计入 reduce 阶段）

**公平基线即为完整规则**——L 没有额外的物理复用、没有硬件能力，纯粹按计算图拓扑 + 最优调度决定。

**L 对给定计算图是固定的**，不随硬件/API/实现改变。

---

## 物理存活节点 (P)

**特有计数域**：UB buffer。寄存器不在计数域——UB→Reg→UB 是过路中转，中间值留在寄存器时不占 P。buffer 计入其分配阶段的 P，跨阶段复用不占复用阶段的 P（如 reduceResult 驻 cacheBuf，不占 P_post）。独占分配的 buffer（生命周期内不允许与其他 tensor 复用）视为全程驻留，计入 P。

**物理基线 = 逻辑基线 + 以下差异**（只能来自 API/硬件能力）：

| 差异来源 | 公平？ | 说明 |
|---------|--------|------|
| 跨步 buffer 复用 | ✓ | 生命周期不重叠可共用物理 buffer，物理侧允许 |
| MulAddDst 硬件融合 | ✓ | 硬件保证 dst 就地覆写，计为 API 能力 |
| 软件 StoreAlign 覆写 | ✗ | 逻辑侧/物理基线均不假设 in-place |
| VF 寄存器链融合 | ✓ | 中间值全程在寄存器（见下文） |

**P 不由计算图唯一确定**——同一计算图，不同 API 策略/融合决策 → 不同 P。寻找在给定约束下可达的**最小峰值**。

### P ≥ L 与例外

MemBase 下 P ≥ L 恒成立——每个 op 的 src 和 dst 各占独立 UB buffer，无硬件能力消掉中间节点。

VF 寄存器链可打破此规律：中间值全程在寄存器，不落 UB。L 仍包含（按 tensor 计数），P 不包含（寄存器不计入 P）。此时 **P < L 合法**。

> 例：reduction preReducePhase 中 `Cast(fp16→fp32) + Square + Add` 全链 VF 融合后，Cast 和 Square 的中间值在寄存器，L=4（输入 + 3 个中间 tensor），P=2（输入 buffer + 输出 buffer）。设计文档必须说明哪些中间值被 VF 消掉。

### Cast 对 P 的影响

reduction 范式中 Cast 使用 `Reg::Cast` 寄存器指令（详见 [common/cast-rules.md](../../common/cast-rules.md) §4.3），在 VF 链内完成类型转换。

- **Cast 参与 VF 融合时**：Cast 在寄存器内完成，src 和 dst 不需要同时占两个 UB buffer。Cast 不额外增加 P。
- **Cast 不参与 VF 融合时**（独立 Cast 操作）：Cast 的 src 和 dst 是两块独立 UB buffer，纳入物理 trace 取峰值。

### RegBase 对 P 的影响

RegBase 允许数据在寄存器中构造和驻留，**可以降低 P**：

- **常量/标量在寄存器构造**：标量不占 UB
- **链式中间值留在寄存器**：Reg::Cast→Reg::Mul→Reg::Add 连续计算，中间结果全程在寄存器链上

**不改变的情况**：单次二元操作 src0/s1 本来就在 UB 中，RegBase 仍需 UB→LoadAlign→Reg compute→StoreAlign→UB，UB 需求与 MemBase 相同。
