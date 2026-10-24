# 技术卡片 vec-09：外层 for 下沉进 VF（loop-sinking，setup 只做一次）

- **适用代际**：arch35（A5 / DAV_3510）独占；本卡依赖 RegBase / MicroAPI / VF，不用于 arch22 实现。
- **适用 bound**：VEC / Scalar
- **一句话**：把 VF 外逐行调用的标量 for 折进 VF，用 `dimA` 形参承接，固定 setup 移到循环外只做一次。

## 何时用（诊断特征）

- 调用方在 VF **外**用标量 `for` 多次调用同一 VF、每次只处理一行（`dimA=1`）。
- trace 上 VF 段被**反复启停**、相同 setup 微指令（`vbr`/`vdup`/mask 创建/常量寄存器初始化）周期性重复（属「头开销大」：Setup 阶段占比高）。

## 原理

把外层 `for` 折进 VF，用 `dimA` 形参在 VF 内承接（`for (a < dimA)`），mask / 常量寄存器等固定 setup 移到内层循环外**只做一次**，消除 N 次进出 VF 与 N 次重复 setup。

## 怎么改（before / after）

```cpp
// before：for 在 VF 外，逐行调用 dimA=1，每行重复 setup
for (a = 0; a < processNum; a++)
    VF(dst[a], src[a * R], /*dimA=*/1, R);

// after：一次调用，processNum 作为 dimA 传入；VF 内 for(a<dimA)，setup 在循环外只做一次
VF(dst, src, /*dimA=*/processNum, R);
// VF 内部：
//   <一次性 setup：mask / 常量寄存器 / DuplicateConst>
//   for (uint32_t a = 0; a < dimA; ++a) { <逐行计算，复用上面的 setup> }
```

## 性能数据

msprof `aiv_time` + Setup 阶段占比：下沉后 setup 从 N 次降到 1 次、VF 不再反复启停。**待实测**（预期：行数越多、单行越短，摊薄启动开销收益越大）。

## 红线（Stop rules）

- 仅当**跨 a 的 setup 确实可复用、各 a 之间无数据依赖**时成立。
- 若 VF 内 setup 必须随 a 变化、或 `dimA` 批量撑爆寄存器 / 撑长 VF 体（指令 cache、寄存器压力），则不宜下沉。
- 尾段单次 lane < 32 的短向量不值得起 VF（VF 启动有固定开销）。

## 关联

- 源：外部 VF 策略 9（loop-sinking）；案例：argmax-with-value（外层 for 下沉 + int64 preg split）、fused-causal-conv1d（外层 j 循环下沉）。
- 相邻卡片：vec-10（下沉后的小定长内循环全展开）、vec-12（下沉引入 int64 preg 压力时拆分）。
