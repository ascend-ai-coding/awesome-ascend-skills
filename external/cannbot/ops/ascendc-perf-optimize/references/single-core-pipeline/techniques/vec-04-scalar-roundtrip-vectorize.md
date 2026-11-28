# 技术卡片 vec-04：标量 round-trip 换成批量向量化（GetValue/SetValue → Brcb + 批量算）

- **适用代际**：通用方法（arch22 / arch35）；具体 API、指标和阈值按目标核验，示例或实测仅代表所标平台。 RegBase / MicroAPI / VF 部分仅适用于 arch35。
- **适用 bound**：VEC / Scalar
- **一句话**：把 per-row 的「取标量→算→写回标量」round-trip 换成「reduce→Brcb 广播→批量算→masked 一次写回」，消灭循环内所有 scalar pull。

## 何时用（诊断特征）

- SCALAR/SCALARLDST 泳道忙、出现 **V_S/S_V 同步**；VF/行循环内有 `tensor.GetValue(0)` 取标量、算完再 `SetValue` 写回的 per-row round-trip。
- 每次 `GetValue`/`SetValue` 都要 V_S/S_V 同步，把 VECTOR 流水反复排空。
- 常见于 dynamic quant / per-row scale 类：行内 reduce 出 max，再 per-row 算 `1/scale` 后逐行 `Muls`。

## 原理

把「行内 reduce → per-row 标量 `1/scale` → per-row `Muls`」换成「reduce 到 lane0 → `Brcb` 复制到 8 lane → 批量 `Div` → stride-0 广播 `Mul` → masked `Mul` 一次写回」，循环内不再有任何标量拉取，V_S/S_V 同步随之消除。

## 怎么改（before / after）

```cpp
// before（每行）：Div 出 scale → SetFlag/WaitFlag<V_S> → scale = temp.GetValue(0)
//                → SetValue(i, 1/scale) → Muls(scalar)   —— 每行一次标量 round-trip

// after（零标量）：
Brcb(brcbTmp, temp);                                   // lane0 max 广播到 8 lane
Div(quantScaleTmp, constScale, brcbTmp, 8);            // 批量 const / max
Mul(tempCast, tempCast, quantScaleTmp /*stride0*/);    // stride-0 把 8 lane 广播到 rowLen
Mul(scaleLocal[i & ~7], brcbTmp, constInvScale, mask); // masked 一次写回 1/scale 到行号槽位
```

配套：中间 cast `int32→half` 可降到 `int16→half`（quant 后值域 [-127,127] 或 [-7,7] 足够，cast 吞吐翻倍）；常量初始化（`DuplicateConst`）提到 Process 入口最前，与 MTE2 重叠。

## 性能数据

msprof `aiv_time` + SCALAR/SCALARLDST 泳道占用：消除 per-row V_S/S_V 同步后标量泳道显著变空、VECTOR 流水连续。**待实测**（预期：行数越多、原 round-trip 越密，收益越大）。

## 红线（Stop rules）

- `Brcb` / `WholeReduceMax` 等需 64 对齐，相关 buffer 按 64 对齐分配。
- 降 cast 位宽前确认值域确实落在目标类型范围内。
- 多 dtype 路径合并时，dtype 差异下沉到 `if constexpr (IsSameType<...>)`，不要在热循环里运行期判 dtype。

## 关联

- 源：外部 VF 策略 3（scalar round-trip 向量化）；案例：dynamic-quant DB symmetric merge（per-row GetValue → Brcb 批量 Div + masked scale Mul）。
- 相邻卡片：vec-07（先做低延迟归约拿到 lane0 值）、vec-05（归约后广播的寄存器内写法）。
