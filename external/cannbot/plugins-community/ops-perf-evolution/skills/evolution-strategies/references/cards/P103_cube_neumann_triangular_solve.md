---
id: P103
bottlenecks: [scalar_compute, compute_bound]
op_families: [cv_fusion, attention, special]
complexity: L1
conflicts_with: []
synergizes_with: [P102, P87]
requires: [P102]
has_preconditions: true
has_playbook: false
quantified_gain:
  - shape: "GDR triangular solve C=64 per chunk"
    baseline_us: 98.0
    optimized_us: 3.0
    speedup: 30.0
    source: "gated_delta_rule_cv: AIV 标量 forward substitution 98µs/chunk → Cube Neumann 11 Mmad"
---

# P103: Cube Neumann 乘积式三角求逆（替代标量 forward substitution）

## 核心思想

下三角线性系统求解（forward substitution / back substitution）是 O(C²) 的**标量串行链**：每一行依赖前面所有行的结果，无法向量化，是 MIX 算子中最长的串行尾。数学上等价的替代：

```
求解 (I + A) x = b，其中 A 严格下三角（A^C = 0，C 为块维）
⇔ 求 (I - X)⁻¹，X = -A（X 严格下三角，X^C = 0，Neumann 级数有限截断）
⇔ (I - X)⁻¹ = (I + X)(I + X²)(I + X⁴)(I + X⁸)(I + X¹⁶)(I + X³²)   (C = 64)
```

整条链 = **5 次平方 Mmad + 6 次乘积 Mmad = 11 个 Mmad**，全部在 Cube 上完成，替代 C(C-1)/2 ≈ 2016 次标量迭代。fp16 中间精度误差 ~1.2e-4（在 GDR 精度预算内）。

## 代码骨架（C=64 实例，来自 gated_delta_rule_cv）

```cpp
// 输入: X16 (x = -A2) 与 F0 (I + x)，均为 fp16 GM workspace slot
// 输出: A16 (solved, 即 (I+X)^{-1} 的乘积结果) fp16 slot

// ---- 平方链: x2 = x@x, x4 = x2@x2, ..., x32（各自 fp32→fp16 全往返）----
MMf16(X16, X16, X2,  64, 64, 64);  PipeBarrier<PIPE_ALL>();
MMf16(X2,  X2,  X4,  64, 64, 64);  PipeBarrier<PIPE_ALL>();
MMf16(X4,  X4,  X8,  64, 64, 64);  PipeBarrier<PIPE_ALL>();
MMf16(X8,  X8,  X16P,64, 64, 64);  PipeBarrier<PIPE_ALL>();
MMf16(X16P,X16P,X32, 64, 64, 64);  PipeBarrier<PIPE_ALL>();

// ---- 乘积链: R 从 F0 开始，R ← R @ (I + x^{2^i}) = R + R @ x^{2^i} ----
// c1L (L0C) 整条链持有：首个 Mmad cmatrixInitVal=1（覆盖），
// 后续 cmatrixInitVal=0（L0C 内累加 R@P 到 R 上）
auto c1L = c1_.AllocTensor<float>();
MmadParams mp{}; mp.m = 64; mp.n = 64; mp.k = 64;
// i=0: R = F0（Mmad(F0, I16, init=1)），随后同一步折叠 F0@x2（init=0）
// i=1..4: R += R @ x^{2^{i+1}}
// 每步结束 Fixpipe 回 GM slot（除最后一步写 A16 外写 R16），PipeBarrier<PIPE_ALL>()
c1_.FreeTensor(c1L);
```

## 关键修改点

1. **L0C 驻留累加**：乘积链的 `R + R@P` 利用 `cmatrixInitVal=0` 在 L0C 内直接累加，避免 fp32 中间结果往返 GM——这是把 6 次乘降压到单次 L0C 驻留的关键
2. **F0@x2 折叠**：`i==0` 步在同一个 L0C 驻留窗口内连做两个 Mmad（先 `F0@I` 初始化，再 `F0@x2` 累加），省一次 Fixpipe 往返
3. **平方链必须 fp32→fp16 全往返**：每级平方 Fixpipe 回 fp16 GM slot 再加载，防止 fp16 下三角矩阵在 L0 上精度雪崩（实测 ~1.2e-4 总误差）
4. **identity 由 host 预写**：`I16` 放 GM workspace 固定 slot，host 侧 `at::eye(64)` 一次性写入，kernel 内不构造
5. **同步纪律**：Mmad→Fixpipe、Fixpipe→下次 GM 读之间必须 `PipeBarrier<PIPE_ALL>`（L0 队列事件同步缺失会触发 aicore exception）

## 适用性检测 (grep)

```bash
# 检测标量三角求解 / 递推循环（P103 适用信号）
grep -nE "for\s*\(.*i.*\+\+.*\)\s*\{?\s*for\s*\(.*j\s*=\s*0.*j\s*<\s*i" op_kernel/*.cpp
grep -nE "solve|substitution|forward_sub|linsolve" op_kernel/*.cpp

# 前提：已是 MIX/Cube 架构（否则先走 P102）
grep -nE "KERNEL_TYPE_MIX|ASCEND_IS_AIC" op_kernel/*.cpp
```

## 常见陷阱

⚠️ **fp16 精度预算**：Neumann 链是 fp16 中间表示，误差随平方层级积累（~1.2e-4 @ C=64）；精度预算紧的算子需先用 torch 仿真验证该误差可接受，再进 kernel
⚠️ **严格下三角前提**：X 必须严格下三角（对角为 0）才有 X^C=0 的有限截断；含对角的 mask 用错（inclusive vs strict）会导致结果系统性偏差且**单 chunk 测试无法暴露**
⚠️ **C 维上限**：示例为 C=64（6 级乘积）；C=128 需 7 级（x64），workspace 与 L0C 占用重估
⚠️ **队列完整性**：每个 Mmad 操作数必须 Alloc→Load→EnQue→DeQue→Mmad→Free 完整走 a2_/b2_ 队列，缺事件同步 = aicore exception（非 hang，有报错）

## 来源

- gated_delta_rule CV 融合重写（a5_ops_slim/workspace/gated_delta_rule_cv）：AIV 标量 forward substitution 98µs/chunk → Cube Neumann 链 11 Mmad，是 geomean 3.79x → 8.85x 的关键一跃
