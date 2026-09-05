# DMA 原子操作 API 使用指南

> **适用场景**：多核/多 rank 写入同一 GM 输出地址时，通过原子加（AtomicAdd）让数据搬运自动累加，消除额外的 reduce 步骤。

---

## 1. 概述

DMA 原子操作控制**所有目的地址为 GM 的数据搬运**如何处理多核/多 rank 对同一 GM 地址的并发写入。官方原文："对后续目的地址为 GM 的数据搬运开启原子累加"——作用范围**不限于 Fixpipe（L0C→GM）**，还包括 DataCopy UB→GM（MTE3），A2/A3 上还含 L1→GM。

**核心价值**：将 reduce 操作融入 DMA 写入，实现"计算即累加"，消除额外的通信或 host 侧 reduce 开销。典型场景：多 rank 计算同一输出的不同分段部分和（split-K / ReduceScatter / AllReduce）。

---

## 2. API 签名与参数

头文件：`basic_api/kernel_operator_set_atomic_intf.h`（通过 `kernel_operator.h` 自动包含）

| API | 签名 | 说明 |
|:---|:---|:---|
| `SetAtomicAdd<T>()` | `template<typename T> void SetAtomicAdd()` | 开启原子加：写 GM 时自动累加 |
| `SetAtomicMax<T>()` | `template<typename T> void SetAtomicMax()` | 开启原子最大值（仅 950/A3/A2 支持） |
| `SetAtomicMin<T>()` | `template<typename T> void SetAtomicMin()` | 开启原子最小值（仅 950/A3/A2 支持） |
| `SetAtomicNone()` | `void SetAtomicNone()` | 关闭原子操作（已废弃，建议用 `DisableDmaAtomic`） |
| `DisableDmaAtomic()` | `void DisableDmaAtomic()` | 关闭原子操作（推荐） |

> `SetAtomicNone()` 已被 `DisableDmaAtomic()` 替代，两者功能等价（内部互相委托）。

---

## 3. 使用模式

### 3.1 决策树：何时需要原子操作

```
多核/多 rank 写入同一 GM 输出地址？
├── 是 → 需要原子操作
│   ├── 累加语义（ReduceScatter / AllReduce / split-K）→ SetAtomicAdd<T>()
│   ├── 取最大值（如某些 attention 归约）→ SetAtomicMax<T>()
│   └── 取最小值 → SetAtomicMin<T>()
└── 否 → 不需要原子操作（默认状态）
```

### 3.2 前置条件（官方约束，易遗漏）

| 前置条件 | 违反后果 |
|:---|:---|
| **GM 不会自动清零**：接口执行前不会清零 GM 数据，需开发者手动初始化 | 结果 = GM 随机残留值 + 部分和。若目标地址无"首次写入"步骤，必须先清零 |
| **浮点原子累加非确定**：多次搬运的累加顺序不固定 | 浮点结果 run-to-run 抖动。需要确定性结果时，改用单写者 + 显式 reduce，或用 PipeBarrier<PIPE_MTE3>/核间同步保证顺序 |
| 开启/关闭原子指令为 PIPE_S，与搬运指令（PIPE_FIX/PIPE_MTE3）不同流水 | 官方明确：开启原子操作的指令与后续搬运指令之间**不需要**开发者插入同步 |

### 3.3 时序模板（最小通用示例）

```cpp
// 开启：必须在首次 GM 写入之前调用
AscendC::SetAtomicAdd<float>();
// ... 需要累加的搬运指令（Fixpipe 或 DataCopy UB→GM）...
AscendC::DataCopy(cGM, cLocal, tileLen);   // UB→GM 同样被原子化
// 关闭：在所有需要累加的搬运指令发射之后调用
AscendC::DisableDmaAtomic();
```

**关键时序约束**：
1. `SetAtomicAdd` 必须在首次 GM 写入之前调用
2. `DisableDmaAtomic` 在所有需要累加的搬运指令发射之后调用
3. 作用域是核内全局的（不限于单次搬运调用），**必须显式关闭**

---

## 4. 常见错误

### 4.1 忘记关闭原子操作

```cpp
// ❌ 错误：SetAtomicAdd 后未关闭，影响同 kernel 内后续所有 GM 写入
SetAtomicAdd<bfloat16_t>();
// ... matmul ...
// 忘记 DisableDmaAtomic()！
```

**后果**：同 kernel 内后续所有目的地址为 GM 的搬运（含 UB→GM DataCopy）都会被累加，导致结果翻倍或数据损坏。

### 4.2 类型不匹配

```cpp
// ❌ 错误：T 与输出 dtype 不一致
SetAtomicAdd<float>();  // 但输出是 bfloat16_t
```

**后果**：原子操作的位宽和格式不匹配，导致数据损坏。

### 4.3 混淆 DMA 原子与 Hcomm 原子

| API | 作用层 | 场景 |
|:---|:---|:---|
| `SetAtomicAdd<T>()` | DMA 搬运层（写 GM 时随路累加） | 多核/多 rank 计算结果累加到同一 GM 地址 |
| `Hcomm::AtomicFAA` | UDMA 跨卡通信层（AIV 侧） | 跨卡 flag counter 原子递增，仅支持 int32/uint32/int64/uint64，需 channel 初始化后调用 |

两者作用层不同，不可互换。

### 4.4 对 FP8 类型使用原子操作

FP8 类型不支持原子操作。如果输出 dtype 是 FP8，需先转 BF16/FP32 再原子累加，或在 L0C 级累加后单次写出。

### 4.5 GM 未初始化直接累加

目标 GM 地址若无"首次写入"步骤（如本地首写覆盖），必须先清零，否则累加基值是随机残留数据（见 §3.2）。

---

## 5. 最佳实践

### 5.1 作用域管理

将 `SetAtomicAdd` / `DisableDmaAtomic` 严格包裹在需要累加的计算阶段内：

```cpp
if (needAccumulate) {
    SetAtomicAdd<CType>();
}
// ... 仅需要累加的搬运阶段 ...
if (needAccumulate) {
    DisableDmaAtomic();
}
```

### 5.2 支持的数据类型（A2/A3）

| dtype | SetAtomicAdd | 说明 |
|:---|:---:|:---|
| `float` | ✅ | FP32 累加 |
| `half` | ✅ | FP16 累加 |
| `bfloat16_t` | ✅ | BF16 累加 |
| `int8_t` | ✅ | INT8 累加（A2/A3；部分更早平台不支持） |
| `int16_t` | ✅ | INT16 累加 |
| `int32_t` | ✅ | 整数累加 |
| FP8 类型 | ❌ | 不支持，需先转 BF16/FP32 |

> 平台差异：310b 无 int8/bf16；`SetAtomicMax`/`SetAtomicMin` 仅 950/A3/A2 支持（310b/310p/910 等更早产品不支持）。

### 5.3 与 Matmul 流水的关系

```
Matmul 流水：
  L0A/L0B → MMAD → L0C → Fixpipe → GM
                                 ↑
                        SetAtomicAdd 在此生效（及所有 GM 写入）
```

`SetAtomicAdd` 不影响 L0C 内的累加（L0C 累加是 MMAD 行为，与 DMA 随路原子无关）。需要 L0C 级累加时不要用原子操作，直接在 L0C 累加后单次 Fixpipe 写出。

---

## 检查清单

- [ ] 多核/多 rank 写同一 GM 地址时已开启 `SetAtomicAdd<T>()`
- [ ] `T` 与输出 dtype 一致（BF16 输出 → `SetAtomicAdd<bfloat16_t>`）
- [ ] 目标 GM 地址已初始化（无首次写入步骤时先清零，见 §3.2）
- [ ] `SetAtomicAdd` 在首次 GM 写入之前调用
- [ ] `DisableDmaAtomic()` 在所有需要累加的搬运指令发射之后调用
- [ ] 浮点累加场景已评估 run-to-run 非确定性（见 §3.2）
- [ ] FP8 输出场景改用 L0C 级累加或先转 BF16
- [ ] 未混淆 `SetAtomicAdd`（DMA 搬运层）与 `Hcomm::AtomicFAA`（UDMA 通信层）

---

## 相关文档

- [api-hcomm.md](api-hcomm.md) — Hcomm AtomicFAA/AtomicCAS（UDMA 层原子操作）
- [api-buffer.md](api-buffer.md) — TBuf/TQue 选择
