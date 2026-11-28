---
type: "PyPTO Performance Optimization Card"
title: "提取迭代间恒等的公共计算（每组一次共享）"
description: "当 per-iteration 计算的取数范围或表达式可证明对所有迭代恒等（如裁剪窗口在半径 ≥ 维大小-1 时塌缩为全维）时，把逐次重算提升为每组一次、组内共享；同操作数同序保证位级等价。"
status: "stable"
tags: ["pypto-pro", "vec", "reduction", "cse"]
item_id: "vec-18"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/patterns/vf-outer-reduce.md#metadata-and-invariant-inputs"
      scope: "先证明复用范围内的输入与表达式不变，再选择共享计算或共享加载，避免逐迭代重复工作。"
      boundary: "参考主要讨论不变输入复用，本卡进一步共享恒等计算结果；范围恒等判据、相同累加顺序与互斥分派须另证，不能混报收益。"
bound_hint: "VEC"
applicability: "per-iteration（逐行/逐项）循环重算某个归约/累计值，且其取数范围或表达式可证明对所有迭代恒等（范围端点按索引计算、经钳位后与索引无关，或表达式经代数化简后不含迭代变量）；共享累加寄存器可跨求和/输出两段循环存活而不引发 spill"
target_api_gate: "仅限 Ascend 950PR 或 950DT；嵌套 pl.range（组→求和 / 组→输出）与累加寄存器跨循环存活已在 950PR 交付算子的裁剪窗口归约实例核验；恒等判据须 host 侧证明并经 TilingKey 分派保证"
---
# 技术卡片 vec-18：提取迭代间恒等的公共计算（每组一次共享）

- **适用 bound**：VEC / Scalar
- **一句话**：循环里每次迭代重算的那个值，如果可证明对所有迭代是**同一批操作数、同一顺序**（范围塌缩、表达式恒等），就把它提出来每组算一次、全组共享——这是跨迭代的公共计算提取；同操作数同序使共享位级等价，不引入任何重结合。

## 何时用（诊断特征）

- VF 内 per-iteration 循环每次都完整重算某个归约/累计值（典型：逐行窗口平方和、逐行前缀式的饱和统计）。
- 该计算的**取数范围按迭代索引计算、但经钳位后与索引无关**：范围端点形如 [max(0,c−k), min(D−1,c+k)]，当 k ≥ D−1 时对任意 c 恒为 [0, D−1]——裁剪/饱和类边界是这种塌缩的最常见来源。判据在 host 侧判一次即可。
- 更一般地，任何"形式上依赖迭代变量、值上恒等"的重复计算（表达式经代数化简后不含迭代变量）都属本卡模式；范围塌缩之外的实例未经核验，采用前须按同样的证明义务补证。
- 逐次重算链在热点中占可观比例：按"per-iteration 计算成本 × 迭代数" vs "一次计算 + 各迭代尾部消费"的指令数模型初判，并用 trace/生成物确认。

## 何时不适用

- 范围仅**部分重叠**（各迭代的操作数集合有交但不相同）：共享需滑动差分（acc[i+1] = acc[i] − 旧项 + 新项），操作数集合与累加顺序改变、精度语义不同，不属本卡。
- 拟共享的是**跨迭代不变的输入数据**（如逐行读取同一份 mean/gamma）——那是输入不变量外提，见 vec-06；本卡针对**计算结果**（归约/累加值）因恒等而共享。
- 共享累加寄存器跨两段循环（求和循环 → 输出循环）存活导致 spill 或容量回退。
- 组内迭代数很大时，一次计算与逐次的指令差缩小、共享值存活期变长，须先评估寄存器压力与收益。

## 原理

- 核心是**跨迭代公共计算提取**：把"每次迭代各算一遍"改为"每组算一次、组内共享"，删除 (C−1) 次重复计算链（load + widen + square + accumulate 等）及其配套的嵌套循环控制。
- **位级等价构造**：共享计算按与逐次计算相同的操作数集合、相同顺序执行（如 k=0..C−1 升序累加与逐行窗口的累加顺序一致）——同操作数、同顺序，构造上逐位一致，不涉及任何重结合或精度合同重谈。
- 收益来源不只是指令数：被删除的重复链常为串行依赖（load→widen→square→accumulate 逐步等待），共享化同时消除循环控制与依赖链深度；具体占比以当前算子的 trace/生成物归因为准。

## 怎么改（before / after）

以下为已在 Ascend 950PR 交付算子核验的**嵌入片段**（触发实例：裁剪窗口塌缩；fp16 widen 链以 even/odd 双累加器为例，具体计算链随算子替换）。前置条件"范围恒等"由 host 侧证明并经 TilingKey 分派到本路径（kernel/VF 内不做运行时判断）。

before（逐迭代重算，item→c→k 三层）：

```python
for item in pl.range(0, items):
    ibase = item * c_dim
    for c in pl.range(0, c_dim):
        j_lo = pl.max(0, c - r)                  # r>=C-1 时恒为 0（范围塌缩）
        j_hi = pl.min(c_dim - 1, c + r)          # r>=C-1 时恒为 C-1
        acc_e = vf.full(0.0, m32, dtype=pl.DT_FP32)
        acc_o = vf.full(0.0, m32, dtype=pl.DT_FP32)
        for k in pl.range(0, j_hi - j_lo + 1):   # 每次迭代完整重算同一范围
            reg16 = vf.load_align(x_tile, (ibase + j_lo + k) * hw_tile)
            ...
        ...尾部（norm/pow/y，逐行）...
```

after（每 item 一次共享，item→k 求和 + item→c 尾部）：

```python
for item in pl.range(0, items):
    ibase = item * c_dim
    acc_e = vf.full(0.0, m32, dtype=pl.DT_FP32)  # 累加寄存器跨两段循环存活
    acc_o = vf.full(0.0, m32, dtype=pl.DT_FP32)
    for k in pl.range(0, c_dim):                 # 每组一次；升序 = 逐次计算的累加顺序
        reg16 = vf.load_align(x_tile, (ibase + k) * hw_tile)
        ...
    for c in pl.range(0, c_dim):                 # C 行复用同一 acc
        creg = vf.load_align(x_tile, (ibase + c) * hw_tile)
        ...尾部（norm = bias + alpha*acc；pow = exp(-beta*ln(norm))；y = x*pow）...
```

一般化推广时，把"窗口塌缩判据 + 求和循环"替换为当前算子的"恒等判据 + 对应计算链"，证明义务（同操作数、同顺序）与寄存器跨循环存活的核验不变。

## 性能与验证指标

比较 `Task Duration(us)`、重复计算链的指令数、依赖链深度与 spill 情况。**待实测**：重复链在热点中占比越高收益预期越大；建议以隔离对照臂（仅开关本项共享、其余不变）归因收益，不与其它优化合并报告。

验证方法：golden 容差对比 + 分派边界覆盖回归（恒等判据的边界族：最小维、判据取等边界、宽参数 × 双 dtype）；新旧输出逐位 diff（同操作数同升序时应逐位一致）可作为等价性补证。

## 技术限制与风险

- 恒等判据必须由 host 侧证明并经 TilingKey 编译期分派；判据错误（范围实际不恒等时误入本路径）会得到错误结果，判据取等边界必须进测试覆盖。
- 塌缩实例中范围参数（如 r）在本路径数学中消失（范围=全维）；实现时注意共享计算不得再消费该参数。
- 位级等价仅在"同操作数、同累加顺序"成立时成立；改写为差分/滑窗形式不属于本卡。
- 共享累加寄存器跨循环存活：fp16 even/odd 双链意味着 2 个累加寄存器长存活（fp32 列块双累加器同理）；组内迭代数较大时存活期变长，采用前评估 spill 风险。
- 共享结果被组内全部迭代消费，改变任何单次迭代的取数范围都会使共享失效——本路径与逐次路径需保持互斥分派。
- 与 vec-06 的边界：外提**不变输入**（load）是 vec-06；共享**恒等计算结果**（归约/累加）是本卡。同一算子内两者可叠加，但不得互相冒充收益。

## 参考资料

- 通用方法：合并语义等价的重复工作；本卡是其"恒等判据 + 同序位级等价构造 + 寄存器跨循环存活"的具体实例。
- 相邻卡片：[vec-06](vec-06-broadcast-once-vl-merge.md)（不变量广播一次）——输入侧对应物。
