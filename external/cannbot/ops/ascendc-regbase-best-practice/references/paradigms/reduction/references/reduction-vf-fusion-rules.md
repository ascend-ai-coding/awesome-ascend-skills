# VF 融合分析

> **适用范围**: reduction 范式通用方法论。§1.1 存活节点分析（L + P）和 §1.2 VF 融合分析时使用。

## 1. 穷举所有相邻操作对

**列出计算图中所有连续 Vector 操作对，逐对判据。禁止只检查"看起来像"的那一对。**

方法：从计算图第一条 Vector 操作开始，相邻两两配对。每一对都走三条硬规则判据。全部检查完才进 §1.3 决策。

### 1.1 连续可融对 → 合并为单条 VF 链

逐对判据回答"哪些相邻操作可以融"，但**不会告诉你它们应该合成一个 VF 还是多个**。判据结束后，要再看一步：

**判据只有三条硬规则：**
1. **链长 ≤ 7**——超过 7 条 Vector 操作必须拆为多次 VF 调用
2. **Cast 可参与 VF**——reduction 范式中 Cast 使用 `Reg::Cast` 寄存器指令（见 [common/cast-rules.md](../../common/cast-rules.md) §4.3），可串入 VF 寄存器链路，不作为断点
3. **RegBase 指令必须存在**——对应操作在 [regbase_api_whitelist.md](../../../api/regbase_api_whitelist.md) 白名单中能找到

**多消费者不打断链。** 即使某个中间结果被 VF 外的其他步骤引用，VF 链仍可继续——VF 内部把这个值 `StoreAlign` 到 UB 即可，寄存器链路不受影响。代价是多写一次 UB 会多占一个 P slot，需要纳入物理 trace 做峰值验证。

```
VF 内:  ...→vdiv→raw_x(reg)→StoreAlign→UB (供VF外读)
                                │
                                └→Round→round_x(reg)→...  链继续！
```

所以判断方法为：从第一条 Vector op 开始，只要 RegBase 指令在白名单中存在、非超长链（≤7），链就延续。**不因分叉而断链。**

### 1.2 能不能融取决于 RegBase API 能力

判断一个 RegBase API 是否可用、是否 VF 安全，唯一入口是白名单：[regbase_api_whitelist.md](../../../api/regbase_api_whitelist.md)

对照计算图中相邻操作——它们的操作用 RegBase 指令能否串成寄存器链路？能 → 融。不能（指令不在白名单中）→ 不融。

寄存器链组合（两条或多条 RegBase 指令串在 VF 循环内完成）：`Compare + Select`, `Mul + Add`, `Ln + Mul`, `Exp + Sub`, `Cast(fp16→fp32) + Square`, `Cast(fp32→fp16) + 任意计算` 等。

### 1.3 存活节点压力——融合 vs 拆分取舍

多消费者融合虽然合法，但会增加 P 压力：VF 需要同时持有所有输入 buffer + 所有输出 buffer（包括为外部消费者多写的）。如果合并后峰值超过目标 P，则需要**主动拆分** VF 来降压。

**判据流程**：
1. 先推满最长链（≤7）
2. 做物理 trace，检查峰值是否 ≤ 目标 P
3. 若超 P → 在分叉输出最多的位置拆分 VF，释放部分 buffer 后再续链
4. 重复直到峰值 ≤ P

**总结**：多消费者融合="可以但不是必须"。选择融还是拆，由 P 约束决定。

## 2. 融合改变存活节点大小

融合前每个中间结果各占 1 个 UB buffer。融合后中间结果在寄存器内不进 UB——存活节点 = 所有输入操作数 + 最终输出 + 其他还活着的 buffer。

```
A+B+C+D = E

全融 A+B+C+D： 5 个节点（A, B, C, D, E 同时活着）
融 A+B+C：     4 个节点（A, B, C, TMP1 活着）→ 再 TMP1+D = E
只融 A+B：     3 个节点（A, B, TMP1）→ 再 TMP1+C = TMP2 → TMP2+D = E
```

## 3. Cast 参与 VF 融合

reduction 范式中 Cast 使用 `Reg::Cast` 寄存器指令（详见 [common/cast-rules.md](../../common/cast-rules.md) §4.3），在寄存器内完成类型转换，**可与前后 Vector 操作融合**。

Cast 在 VF 链中的位置：
- **扩位 Cast（b16→fp32）**：通常在 VF 链头部，`LoadAlign<D_T, DIST_UNPACK_B16>` → `Reg::Cast<float, D_T>` → 后续计算
- **缩位 Cast（fp32→b16）**：通常在 VF 链尾部，`Reg::Cast<D_T, float>` → `StoreAlign<D_T, DIST_PACK_B32>`

```
// 扩位 Cast + Square 融合为 1 条 VF 链
LoadAlign<half, DIST_UNPACK_B16>(b16Reg, srcUb);
Cast<float, half, kCastTraitB16ToF32>(f32Reg, b16Reg, mask);
Mul<float>(f32Reg, f32Reg, f32Reg, mask);       // Square
// ... 继续后续计算
StoreAlign<float>(dstUb, f32Reg, mask);           // 输出 fp32
```

**CastTrait 配置**：扩位用 `kCastTraitB16ToF32`（`CAST_NONE`），缩位用 `kCastTraitF32ToB16`（`CAST_RINT`），详见 [common/cast-rules.md](../../common/cast-rules.md) §4.1。

## 4. 存活节点扩大限制

融合使更多输入 buffer 同时活着——融合后的物理存活节点数不得超过融合前的 P + 3。超过 → 只部分融合，或回 §1 调整调度先释放不再需要的 buffer。

### 案例

**reduction preReducePhase: Cast→Square→Add→Mul 全链融合**

计算图 `Cast(fp16→fp32) → Square → Add → Mul`，Add/Mul 的第二操作数为寄存器标量（不占 UB）。所有中间值均为单消费者，无分叉。`Reg::Cast`、`Reg::Mul` 指令均存在，链长=4 ≤ 7。

全链合并为 **1 个 VF**。融合后 fp32 中间值在寄存器，全链只入 UB 一次（fp16 输入）、出 UB 一次（fp32 输出到 tmpBuf）。P = 2（输入 buffer + 输出 buffer）。

### 反例：A+B+C+D 在 P=3 下不能全融

全融需要 5 个节点（A,B,C,D,E 同时存活）≤ 3+3=6。看起来可行？但若当前调度下 A,B 已经和其他步骤共享 buffer，实际节点更紧张。**关键不是公式，是结合实际持有 trace 看**——在融合发生的那一步，加上所有输入 operands 之后的持有数是否超 P。超了就不能全融。
