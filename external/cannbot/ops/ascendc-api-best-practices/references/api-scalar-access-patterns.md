# GetValue / SetValue 替换模式索引

> **目的**：把分散在 [api-datacopy.md](api-datacopy.md)、[api-transpose.md](api-transpose.md)、[api-gathermask.md](api-gathermask.md)、[api-arithmetic.md](api-arithmetic.md)、[api-reduce.md](api-reduce.md) 等处的 GetValue / SetValue 替换知识**汇总到单一入口**，供算子编码与代码检视时查阅。
>
> **核心判定**：先分清**哪一侧**（GM 侧 vs UB 侧），再分清**用法**（data-move vs 元数据 vs 索引计算 vs 标量取样），按"必替换 / 优先替换 / 合法保留"三档处理。
>
> **适用范围**：通用判定逻辑不限代际；§3 方案 C 的 `LoadScalarForDtypeTIn` / DIST_BRC 通路为 A5（DAV_3510）RegBase 范式，跨代际使用前先查目标代际 API 支持情况。

---

## 0. 速查表

| # | 场景 | 反模式 | 正解 | 强制级别 |
|---|------|--------|------|---------|
| 1 | GM 侧批量 data-move | `for { val = xGm.GetValue(i); ... }` | `DataCopy / DataCopyPad` | **必替换** |
| 2 | 嵌套循环 GM/TBuf 单点 | `for outer { for inner { xGm.GetValue(i) } }` | `DataCopy` 批量 + 索引 | **必替换** |
| 3 | 非对齐 tail/pad | `GetValue(i)` 单点补零 | `DataCopyPad`（自动 pad） | **必替换** |
| 4 | UB 元素级 dtype cast | `GetValue→ToFloat→SetValue` | vector `Cast()` | **必替换** |
| 5 | Interleaved 数据提取 | 逐元素 SetValue/GetValue | `GatherMask`（RoPE 奇偶拆分） | 优先替换 |
| 6 | Transpose 元素重排 | 逐元素 GetValue/SetValue | `DataCopy + vector route` | 优先替换 |
| 7 | Reduce 段末取标量 | `T maxVal = broadcastLocal.GetValue(0);` | **合法保留** | 合法 |
| 8 | Broadcast 取标量 | `T coef = broadcastLocal.GetValue(0);` | **合法保留** | 合法 |
| 9 | per-channel/per-position 系数取数（GM 侧 + TBuf 侧） | `T s = scaleGm.GetValue(cCh_);`（GM）或 `T s = scaleLocal.GetValue(cIdx);`（TBuf） | **必替换**（替换路径按目标架构选择，详见 §3） | **必替换** |
| 10 | TilingData 元数据读取 | `blockDim = tiling.GetValue<...>(idx);` | **合法保留** | 合法 |
| 11 | 纯索引计算 | `offset = src.GetValue(...)` | **合法保留** | 合法 |
| 12 | TBuf 主循环单点 SetValue | `yLocal.SetValue(i, ZeroValue());` / `yStag.SetValue(i, yVal);` / `stagingUb.SetValue(dst, val);` | 整段 `DataCopy` 回写 / vector `Duplicate`/`StoreAlign` | **必替换** |
| 13 | TBuf get→modify→write 主循环 | `prev = localSum.GetValue(rowIdx); localSum.SetValue(rowIdx, prev + tileFloat.GetValue(i));` | vector `Reduce` / 段内累加 + 整段回写 | **必替换** |
| 14 | Compact gather+scatter 条件取值 | `T value = c0 < cCount ? compactLocal.GetValue(srcWBase + c0 * srcCStep) : ZeroValue();` | MTE2 一次搬到位 + `GatherMask` / Compactor 通路 | **必替换** |
| 15 | SIMT Tiling 参数预填充 | `simtTiling.SetValue(0, simtOut); simtTiling.SetValue(1, tiling_->n);` | **合法保留**（语义条件见 §2.4：初始化阶段、访问规模有界、用途为参数打包） | 合法 |
| 16 | Scatter 跨 dtype 标量桥接 | `scatterValueTensor.SetValue(0, ...); ... .ReinterpretCast<outputT>().GetValue(0);` | **合法保留**（1 元素 TBuf，固定 `SetValue(0, ...)` + `GetValue(0)`，无向量化路径） | 合法 |

> #9 扩写说明：per-channel/per-position 系数取数覆盖 GM 侧（`scaleGm.GetValue(cCh_)`）与 TBuf 侧（`scaleLocal.GetValue(cIdx)`）双侧——**数据源是 GM 还是 TBuf 与判定无关**，都应替换。替换路径选择见 §3（按目标架构与热点程度分档）。

---

## 1. ⛔️ 黑名单 API（生产代码禁用）

源自 [api-datacopy.md](api-datacopy.md) §1：

| API | 禁止原因 | 仅允许场景 |
|-----|---------|-----------|
| `GlobalTensor::SetValue(idx, val)` | 效率极低，单元素逐个写入 | **仅调试时使用** |
| `GlobalTensor::GetValue(idx)` | 效率极低，单元素逐个读取 | **仅调试时使用** |

```cpp
// ❌ 禁止：生产代码使用 SetValue/GetValue（GM 侧）
for (uint32_t i = 0; i < size; i++) {
    xGm.SetValue(i, value);    // ⛔️ 效率极低
    T val = xGm.GetValue(i);   // ⛔️ 效率极低
}

// ✅ 正确：使用 DataCopyPad 批量搬运
AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
```

**为何 GM 侧 GetValue 必禁**（三层叠加理由）：

1. **吞吐率灾难**——单元素走 PIPE_S（标量流水线），Vector 走 PIPE_V，吞吐率差**数量级**
2. **跨核缓存不一致**——真实事故（unique-507035-optional-output）：跨核 GetValue 读到各核私有 DCache 的旧值，导致 K=0、y 空
3. **跨流水同步缺陷**——GetValue 读 UB 前 MTE2 可能未搬完，读出旧值（详见 [api-pipeline.md](api-pipeline.md) 同步语义）

---

## 2. ✅ 合法保留范式（按用法分类）

> **检视/自查顺序**：先过本节白名单，再套 §0 速查表的必替换条目——避免把合法形态误判为反模式。

### 2.1 Reduce / Broadcast 段末取标量（合法保留）

源自 [api-arithmetic.md](api-arithmetic.md) 与 [api-reduce.md](api-reduce.md)：

```cpp
// ✅ 合法：reduce 段末取 1 个标量
T maxVal = broadcastLocal.GetValue(0);  // reduce 输出的 broadcast 结果
T sumVal = broadcastLocal.GetValue(0);
```

**为何合法**：

- 算法语义就是 reduce 出 1 个标量
- PIPE_S 单次取值开销可忽略（每个 chunk 仅 1 次）
- 强制改向量化反而引入不必要的 Broadcast 开销

**合法判定需同时满足语义条件**（命名仅作候选召回，不作为放行依据）：

- **数据来源**：reduce/broadcast 的输出（非中间计算结果）
- **循环位置**：不在主循环热路径（每 chunk 最多 1 次）
- **执行频率**：低频（如每 tile/每单元 1 次，非逐元素）
- **后续消费者**：标量决策（分支判断、标量乘法系数），非批量数据搬送
- **命名参考**：`broadcastLocal` / `maxLocal` / `sumLocal` / `reduceOut` 等——**仅作搜索线索**，改名不改变语义判定

### 2.2 TilingData 元数据读取（合法保留）

```cpp
// ✅ 合法：读 TilingData 元数据
uint32_t blockDim = tiling_.blockDim;        // 结构体字段直接读
uint32_t tileLen = tiling_.GetTileLen();     // 通过 getter 读
```

### 2.3 纯索引计算（合法保留）

```cpp
// ✅ 合法：纯索引推导
uint64_t offset = baseAddr + stride * idx;
```

### 2.4 SIMT Tiling 参数预填充（合法保留，速查表 #15）

```cpp
// ✅ 合法：SIMT launch 前一次性打包常量参数（库上溯源：max_pool3_d.h）
simtTiling.SetValue(0, simtOut);
simtTiling.SetValue(1, tiling_->n);
```

**合法判定需同时满足语义条件**：

- **执行阶段**：初始化/launch 前置阶段（不在主循环热路径）
- **访问规模**：有界（参数量稳定，不随输入 shape 增长；库上实测约 28 个常量参数，供参考量级，非硬性上限）
- **用途**：打包常量参数供 SIMT launch 使用（非数据搬送）
- **命名参考**：`simtTiling` / `*Tiling*` 等——**仅作搜索线索**，命名以 Tiling 结尾不是放行依据

### 2.5 Scatter 跨 dtype 标量桥接（合法保留，速查表 #16）

```cpp
// ✅ 合法：1 元素 TBuf 桥接跨 dtype（库上溯源：apply_top_k_top_p_with_sorted.h）
scatterValueTensor.SetValue(0, static_cast<inputT>(...)  );
auto outVal = scatterValueTensor.ReinterpretCast<outputT>().GetValue(0);
```

**合法判定需同时满足语义条件**：

- **数据结构**：1 元素 TBuf（`SetValue(0, ...)` + `GetValue(0)` 成对，索引恒为 0）
- **用途**：跨 dtype 标量桥接（如 fp32 计算结果转 int 位模式后取回）
- **无向量化路径**：结构论证——单个标量的 dtype 转换无法用 vector Cast 处理（cast 操作数必须 ≥1 个 vector）
- **命名参考**：`scatterValueTensor` 等——**仅作搜索线索**

---

## 3. ⚠️ 必替换范式：per-channel/per-position 系数取数（速查表 #9）

> **替换路径选择**（先判定"要不要换"，再选"怎么换"）：
>
> **Step 1：是否热点？** —— 系数访问是否每 chunk/每单元多次发生。若每 tile 只取 1 次（如逐单元取 mean/rstd），标量读取开销可忽略，**不强制替换**。
>
> **Step 2：目标架构？** —— 确认目标代际的 broadcast load 通路支持情况：
>
> | 架构 | 可用通路 | 参考真源 |
> |------|---------|---------|
> | A5（DAV_3510）RegBase | `LoadAlign` + `DIST_BRC_B32/B16`（单值广播到 256B RegTensor） | `ascendc-regbase-best-practice` skill 的 broadcast 规则文档 |
> | A2/A3（DAV_2201）或 MemBase | `Brcb`（lane0 复制到 8 lane）或 `Mul`/`Add` 的 stride-0 广播形态 | `ascendc-perf-optimize` skill 的 vec-04 技术卡片 |
>
> **不确定代际时**：先查目标架构 API 文档确认支持情况，再选方案；禁止假设所有代际都有同一通路。

**典型反模式 A（TBuf 侧，norm/pool/lp_norm_v3 等常见形态）**：

```cpp
// 当前（每 chunk 取 1 个 cIdx 系数）
for (k) {
    const float s = scaleLocal.GetValue(cIdx);  // ⛔️ 每 chunk PIPE_S 标量通路
    const float t = shiftLocal.GetValue(cIdx);
    ComputeAffine(xLocal, yLocal, len, s, t);
}
```

**典型反模式 B（GM 侧）**：

```cpp
// LoadPlaneScalars() 函数中（Init 末尾 + Process 平面切换点各调 1 次）
const float meanS  = ToFp32Scalar(runningMeanGM.GetValue(cCh_));  // ⛔️ PIPE_S
const float varS   = ToFp32Scalar(runningVarGM.GetValue(cCh_));
weightS_            = ToFp32Scalar(weightGM.GetValue(cCh_));
biasS_              = ToFp32Scalar(biasGM.GetValue(cCh_));
```

**两种反模式本质相同**：per-channel/per-position 系数按索引取 1 个元素——数据源是 GM 还是 TBuf 与判定无关，都应替换。替换路径按 Step 1/2 选择。

### 方案 C（A5 RegBase）：按需搬 + DIST_BRC broadcast

> **适用前提**：目标架构为 A5（DAV_3510）RegBase 路径；系数访问是热点（每 chunk 多次或每次取多通道）。
>
> **指令级实现真源**：`ascendc-regbase-best-practice` skill 的 broadcast 规则文档（DIST_BRC 通路定义、`LoadAlign` 用法、对齐约束）。本文档只保留判定与适用前提，不维护完整实现。

**核心思想**（两阶段）：

1. **CopyIn 阶段**：按当前 chunk 的通道范围用 DataCopyPad 批量搬系数到 UB（通道切换才搬，复用缓存）
2. **VF 阶段**：把 UB 单元素系数广播到 RegTensor 参与向量计算

**关键调用形态**（概念示意，非完整实现）：

```cpp
// 阶段 1：CopyIn——按通道范围批量搬入（库上范式：instance_norm_ar_full_reduce.h）
DataCopyPad(gammaLocal, gammaGm_[cOffset], copyParams, padParams);

// 阶段 2：VF——单元素广播到整个 RegTensor（库上范式：LoadScalarForDtypeTIn）
Reg::LoadAlign<T, Reg::LoadDist::DIST_BRC_B32>(coefReg, coefUb + idx);
```

**A2/A3 替代路径**（DIST_BRC 不可用时）：

```cpp
// Brcb：lane0 复制到 8 lane，再 stride-0 广播参与计算
Brcb(brcbTmp, temp);                          // lane0 → 8 lane
Mul(tempCast, tempCast, quantScaleTmp /*stride0*/);  // stride-0 广播
```

**何时不用方案 C**：系数访问非热点（如每 tile 只取 1 次）；UB 预算不足；目标架构不支持 DIST_BRC。

---

## 4. AR-1/2/3/4 规则要点

| 规则 | 触发模式 | 对应速查表条目 | 处理建议 |
|------|---------|--------------|---------|
| **AR-1** | 主循环 GM↔scalar GetValue/SetValue | #1, #2 | **默认必改**；确需保留时须能说明理由（如结构论证无向量化等价物） |
| **AR-2** | 嵌套 for 内 GM/TBuf GetValue 单点 | #2 | 必替换，改批量搬运 + 索引 |
| **AR-3** | tail/pad/copyLen%ALIGN≠0 用 GetValue 单点 | #3 | 必替换，改 `DataCopyPad`（自动 pad） |
| **AR-4** | UB 元素循环 `GetValue→ToFloat/ToBfloat16→SetValue` cast 漏斗 | #4 | 必替换，改 vector `Cast()` 批量转换 |

> AR-4 与"dtype 中转漏斗"检视项的关系：AR-4 是其具体形态之一；元素循环内禁用标量 cast helper（`ToFloat`/`ToBfloat16`/`ScalarCast<…>`）做 tensor 级 dtype 转换，一律改 vector `Cast()`。

---

## 5. 选择决策流程

```
出现 GetValue / SetValue
    ↓
Q1: 哪一侧？
├─ GlobalTensor (GM)         → Q2
└─ LocalTensor (UB)          → Q3
    ↓
Q2: GM 侧
├─ data-move 批量搬运        → 必替换 → DataCopy/DataCopyPad（速查表 #1）
├─ 嵌套循环内单点           → 必替换 → 批量 + 索引（速查表 #2）
├─ 非对齐 tail/pad 单点      → 必替换 → DataCopyPad 自动 pad（速查表 #3）
└─ 调试 printf               → 合法保留（仅调试）
    ↓
Q3: UB 侧
├─ data-move UB↔scalar       → 必替换 → vector ops（速查表 #4）
├─ dtype cast 漏斗（AR-4）   → 必替换 → vector Cast()（速查表 #4）
├─ reduce/broadcast 取标量   → 合法保留（速查表 #7, #8；语义判定条件见 §2.1）
├─ 常驻系数表按索引取       → 必替换（速查表 #9；替换路径按架构选择，见 §3）
├─ TBuf 主循环单点 SetValue  → 必替换（速查表 #12）
├─ get→modify→write 主循环   → 必替换（速查表 #13）
├─ compact 条件取值          → 必替换（速查表 #14）
├─ TilingData 元数据读取     → 合法保留（速查表 #10）
├─ SIMT Tiling 预填充        → 合法保留（速查表 #15，条件见 §2.4）
├─ 1 元素跨 dtype 桥接       → 合法保留（速查表 #16，条件见 §2.5）
└─ 纯索引计算                → 合法保留（速查表 #11）
```

---

## 6. 自查用检测 regex（粗筛候选，需人工复核）

> **定位**：以下 regex 是**粗筛工具**——负责把可能的反模式召回出来，**命中 ≠ 违规**。命中后必须人工核对 §2.1/2.4/2.5 的语义条件（数据来源/循环位置/执行频率/消费者）才能判定。
>
> **已知局限**：
> - `TBUF_MAIN_LOOP_SETVALUE_RE` 无对象命名过滤，TQue 的 `inQueue*.SetValue`（合法 FIFO 语义）也会命中；`.*?` 跨循环体可能误命中（如函数 A 的 for 循环 + 函数 B 的 SetValue 被误判为同循环）
> - 白名单命名仅作召回辅助，**不按前缀自动放行**——不合条件的命名（如 `scatterValueTensor` 非 1 元素桥接场景）也会命中后需人工排除

```python
import re

# 嵌套循环单元素读（覆盖速查表 #2）
NESTED_LOOP_GETVALUE_RE = re.compile(
    r"for\s*\([^)]*\)\s*\{[^{}]*?for\s*\([^)]*\)\s*\{[^{}]*?\.\s*(?:GetValue|SetValue)\s*\(",
    re.DOTALL,
)

# TBuf 主循环单点 SetValue（覆盖速查表 #12，含广义 TBuf 命名：
# Local|UB|Ub|Buf|TBuf|Stag|Staging|Tensor）
# 局限：无对象过滤（TQue 合法场景也命中）、.*? 跨循环体可能误命中
TBUF_MAIN_LOOP_SETVALUE_RE = re.compile(
    r"for\s*\([^)]+\).*?\b\w+\s*\.\s*SetValue\s*\(",
    re.DOTALL,
)

# UB 元素级 dtype cast 漏斗（覆盖速查表 #4）
UB_CAST_FUNNEL_RE = re.compile(
    r"\.\s*GetValue\s*\(.*?\)\s*\.\s*(?:To[A-Z]\w+|Cast|static_cast|reinterpret_cast)",
    re.DOTALL,
)

# per-channel 系数取数（覆盖速查表 #9，GM 侧 + TBuf 侧）
PER_CHANNEL_GETVALUE_RE = re.compile(
    r"\b(?P<buf>\w*(?:Local|UB|Ub|Buf|TBuf|Stag|Staging|Tensor)|\w*(?:Gm|GM))"
    r"\s*\.\s*GetValue\s*\(\s*(?P<idx>cCh|cCh_|cIdx|c_|c|idx)\s*\)"
)

# 合法保留语义条件（命中后跳过上述判定，详见 §2.1/2.4/2.5）
# 注：命名仅作搜索线索，不作为放行依据；每个条件需人工核对语义
LEGIT_NAMING_HINTS = (
    # #7/#8 reduce/broadcast 取标量：数据来源是 reduce 输出、低频、消费者是标量决策
    "broadcastLocal", "maxLocal", "sumLocal", "reduceOut",
    # #15 SIMT Tiling 预填充：初始化阶段、规模有界、参数打包用途
    "simtTiling",
    # #16 Scatter 跨 dtype 桥接：1 元素 TBuf、固定下标 0、无向量化路径
    "scatterValueTensor",
)
```

shell 快捷版：

```bash
# 主循环 / 嵌套循环 GetValue/SetValue 粗筛
grep -nE "(for|while).*\{[^}]*(GetValue|SetValue)\(" op_kernel/*.h op_kernel/*.cpp

# UB 侧 cast 漏斗
grep -nE "GetValue.*\n.*(ToFloat|ToBfloat16|ScalarCast)" op_kernel/*.h op_kernel/*.cpp
```

**合法保留语义条件**（命中 regex 后需人工核对，不自动放行；每个条件需同时满足）：

- Reduce/Broadcast 取标量：数据来源是 reduce/broadcast 输出、不在主循环热路径、每 chunk ≤1 次、后续消费者是标量决策（详见 §2.1）
- TQue FIFO：`inQueue*` / `outQueue*` 的 SetValue（TQue EnQue 阶段合法；若在主循环内逐元素 SetValue，仍需复核）
- GM 张量：`*Gm` / `*GM`（走 GM 通路，按 #1/#2/#9 处理）
- SIMT Tiling：初始化阶段、访问规模有界、用途为参数打包（详见 §2.4，命名仅作参考）
- Scatter 跨 dtype 桥接：1 元素 TBuf、固定下标 0、无向量化路径（详见 §2.5）
- Init/PreFunc 阶段 TBuf 预填充（不在主循环）

---

## 7. 已记录反例索引（失败模式库）

> **维护目的**：记录"违反速查表规则但曾成功落地"的案例形态，供检视与自查时对照。

### 7.1 norm 类算子双命中模式

**模式**：norm 类算子（per-channel 仿射的 layer/instance norm、rms norm、batch norm update 等）同时出现两类命中：

- **U 类**：per-channel 系数按通道/位置索引取数（GM 侧 `runningMeanGM.GetValue(cCh_)` 或 TBuf 侧 `chScale.GetValue(c)`）——命中速查表 **#9 必替换**（替换路径按目标架构选择，见 §3）
- **G 类**：paramMode==0 退化支的 4 路 GM 单点直读（`wGm.GetValue(c)` 等）——命中速查表 **#2 必替换**

**修复方向**：

- U 类：批量搬入 UB 后广播参与向量计算（A5 用 DIST_BRC，A2/A3 用 Brcb/stride-0 广播，详见 §3）
- G 类在退化支改走批量 `DataCopyPad` + UB 上向量化 rstd/scale/shift 计算

### 7.2 adaptive_avg_pool3d_abs 主循环 TBuf SetValue

**实测反例**（`adaptive_avg_pool3d_abs.h`，内部仓 cann-bench-batch 路径，行 661/737/752-755）：

```cpp
// 行 661、737：嵌套循环 TBuf 单元素读（速查表 #2 反例）
for (uint32_t hh = 0; hh < r; ++hh) {
    for (uint32_t w = 0; w < segLen; ++w) {
        const float v = xSegS.GetValue(static_cast<int32_t>(rowOff + w));
    }
}

// 行 752-755：主循环 TBuf 单点 SetValue（速查表 #12 反例）
for (uint32_t k = 0; k < tileLen; ++k) {
    ws.SetValue(static_cast<int32_t>(k < VL_F32 ? (WS_OFF_ACC0 + k)
                                                        : (WS_OFF_ACC1 + (k - VL_F32))),
                accS[k]);
    ws.SetValue(static_cast<int32_t>(k < VL_F32 ? (WS_OFF_CNTV0 + k)
                                                        : (WS_OFF_CNTV1 + (k - VL_F32))),
                static_cast<float>(eWk[k] - sWk[k]) * kdKhF);
}
```

**修复方向**（参考真源，不维护实现）：

- 嵌套循环 TBuf 单元素读 → 改 vector 通路：`ascendc-perf-optimize` skill 的 vec-04 技术卡片（标量 round-trip 向量化）
- 主循环 TBuf 单点 SetValue → 改批量回写：`ascendc-regbase-best-practice` skill 的 broadcast 规则文档（StoreAlign/Duplicate 用法）

---

## 8. 引用与依赖

| 引用源 | 内容 |
|--------|------|
| [api-datacopy.md](api-datacopy.md) §1 | DataCopy/DataCopyPad + 黑名单（#1） |
| [api-arithmetic.md](api-arithmetic.md) | Reduce/broadcast 取标量合法范式 + vector Cast 替换（#4/#7/#8） |
| [api-reduce.md](api-reduce.md) | Reduce 段末取标量合法范式 |
| [api-transpose.md](api-transpose.md) | Transpose 元素重排替代（#6） |
| [api-gathermask.md](api-gathermask.md) | Interleaved 数据提取替代（#5/#14） |
| [api-pipeline.md](api-pipeline.md) | TQue 流水同步语义（方案 C CopyIn 阶段） |
| `ascendc-perf-optimize` skill（vec-04 技术卡片） | 标量 round-trip 向量化（与本文件互补：本文件管"哪些场景该换、何时该换、换前需满足什么前提"，vec-04 管"换完怎么写"） |
| `ascendc-regbase-best-practice` skill（broadcast 规则） | DIST_BRC 通路与 `LoadAlign` 广播 load 的指令级细节（方案 C 阶段 2 的**指令真源**——本文档不维护完整实现，只引用） |
