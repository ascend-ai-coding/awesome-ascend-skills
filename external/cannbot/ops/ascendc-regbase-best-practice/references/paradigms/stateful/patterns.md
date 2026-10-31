# Stateful 类算子场景路由

> 本文档用于**场景判定**和**策略选择**。确定场景后，按链接进入对应详细文档。
>
> **Stateful 范式是修饰范式**，与主范式共同声明。主范式（broadcast / elewise / reduction 等）
> 负责计算逻辑、Tiling 切分、Kernel 骨架；Stateful 只叠加 in-place 输入的数据流约束。

---

## 范式定义

**Stateful 算子**：算子的某些输入被持久化——同一个 tensor **既当输入又当输出**，
GM 地址重叠（assign 语义）。典型如：

- Adam 优化器的 `var` / `m` / `v` / `max`（输入即输出，in-place 更新）
- Assign / Parameter 更新类算子

**核心设计问题**：GM 地址重叠带来的 CopyIn / CopyOut 顺序与 buffer 约束，
而非计算逻辑本身。计算逻辑由主范式承载。

---

## 场景判定流程

```
给定: spec.yaml 的 inputs + outputs

Step 1 — 是否存在 input 与 output 共享 GM 地址（assign 语义）？
  ├─ NO  → 非 Stateful，按主范式处理
  └─ YES → 进入 Step 2

Step 2 — in-place 输入的读取模式判定：
  ├─ 单次读取后覆写（如 Adam 的 var/m/v，一次 CopyIn 后最终 CopyOut 回写）
  │       → 场景 A: 标准约束（见下方）
  └─ 多轮迭代中跨轮复用（如 RNN hidden state 跨 kernel launch 持久化）
          → 场景 B: GM workspace 持久化（TODO：待补充）
```

⛔ **必须执行场景判定，禁止跳过或默认选择。** 判定依据是 spec.yaml 中 input/output
的 GM 地址关系，不是算子名或 category。

---

## 场景 A: 标准约束

**判据**：in-place 输入在单次 kernel launch 内完成"读取 → 计算 → 回写"，
不需要跨 launch 持久化中间状态。

### 通用规则

1. **UB 内禁止 in-place read-modify-write**
   - RegBase 硬件不支持同一 UB buffer 同一循环内 in-place read-modify-write。
   - 对单次二元操作仍需 `src0 + src1 + dst` 三块独立 UB buffer。
   - CopyIn 与 CopyOut 必须使用独立 UB buffer，禁止把 CopyOut 目标 buffer 复用作 CopyIn 源。
   - 来源：`broadcast/example/examples-adam-design/adam_apply_one_assign_design.md`
     §2.2 "RegBase 边界"、§2.6 "RegBase 无法做到同等的 in-place 覆写"。

2. **CopyOut 顺序约束**
   - in-place 输入的 CopyOut 必须晚于该 input 所有 CopyIn 完成。
   - GM 地址重叠：提前 CopyOut 覆写会污染尚未读取的输入数据。
   - 多 tile 场景下，同一 in-place 输入的 CopyIn 与 CopyOut 必须按 tile 顺序串行，
     禁止乱序回写。

3. **多核顺序约束**
   - 跨核时 in-place 输入的写回需考虑核间数据归属，禁止跨核覆写他人 tile。
   - 多核切分按主范式执行，in-place 不改变切分逻辑，只约束写回时机。

4. **TilingData 不需新增字段**
   - in-place 是 GM 地址关系，不影响 Tiling 切分逻辑。
   - 主范式的 TilingData 结构体原样使用，不新增 Stateful 专属字段。

---

## 场景 B: GM workspace 持久化

> TODO：待补充。预留场景——in-place 输入需跨 kernel launch 持久化中间状态
> （如 RNN / LSTM hidden state 跨次调用）。该场景涉及 GM workspace 布局、
> 状态字段下发、跨次调用同步等，待基于参考源码补实。

---

## 与主范式融合

Stateful 是修饰范式，与主范式共同声明：

```yaml
paradigms: [Broadcast, Stateful]   # 主范式 + 修饰范式
```

**默认主范式**：`op.paradigms` 只有 `Stateful` 一项时，按 `Elementwise` 处理。

> 该默认只是兜底。in-place 更新通常伴随 broadcast（如 Adam）或归约场景，
> 必须显式写出主范式——默认值不会替你补回主范式的 Tiling 与 Kernel 骨架。

### 修改清单

| 章节 | 修改 | 说明 |
|------|------|------|
| Kernel 入口 / CopyIn / 地址偏移 | ⚠️ 叠加顺序约束 | in-place 输入的 CopyIn/CopyOut 顺序按通用规则执行 |
| CopyOut | ⚠️ 叠加顺序约束 | in-place 输入回写须晚于其所有 CopyIn |
| Compute | ❌ 不改 | 跟主范式保持一致 |
| 跨核分核 / TilingKey / 模板类 | ❌ 不改 | in-place 不改变切分与模板策略 |
| TilingData | ❌ 不改 | 不新增 Stateful 专属字段 |
| Buffer 清单 / UB 预算 | ❌ 不改 | in-place 是 GM 层关系，UB buffer 规划跟主范式 |

### 文档书写风格

**不改的章节**:
```markdown
## 3 Compute
跟主范式保持一致。
```

**改的章节（叠加约束）**:
```markdown
## CopyOut
**主范式**: DataCopyPad 回写 GM。

**Stateful 约束**:
1. in-place 输入 {var/m/v} 的 CopyOut 须晚于其所有 CopyIn 完成
2. CopyOut 目标 UB buffer 与 CopyIn 源 buffer 物理独立
```

---

## 跨场景参考

| 主题 | 文档 |
|------|------|
| 主范式设计指南（按 op.paradigms 路由） | ascendc-regbase-best-practice skill: `references/paradigms/<主范式>/patterns.md` |
| 通用 Tiling 方法论（主范式未匹配时 fallback） | ascendc-regbase-best-practice skill: `references/paradigms/general-methodology.md` |
| in-place 硬件边界（RegBase 禁 in-place RMW） | `broadcast/example/examples-adam-design/adam_apply_one_assign_design.md` §2.2 / §2.6 |

---

## 阅读确认清单

设计生成器完成阅读和设计后，逐项确认：

- [ ] 已完成场景判定：场景 A / 场景 B / 非 Stateful
- [ ] （场景 A）已确认 in-place 输入列表（spec.yaml 中与 output 共享 GM 地址的 input）
- [ ] （场景 A）CopyOut 顺序约束已写入 DESIGN.md CopyOut 章节
- [ ] （场景 A）UB 内禁止 in-place RMW 已声明，CopyIn/CopyOut buffer 物理独立
- [ ] （场景 A）多核写回时机约束已说明
- [ ] TilingData 未新增 Stateful 专属字段
- [ ] 不改的章节已写"跟主范式保持一致"
