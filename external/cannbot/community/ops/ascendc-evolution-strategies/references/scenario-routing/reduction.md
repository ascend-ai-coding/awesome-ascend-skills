# Reduction 系优化工作流（含 Normalization / Softmax）

> 适用：LayerNorm / RMSNorm / GroupNorm / Softmax（独立）/ ReduceSum·Mean·Max·Min / ArgMax 等沿轴归约算子。

## Step 0 — 族判定确认

命中 [README.md](README.md) 判定表第 6 行（沿轴归约 / 减均值除方差 / Reduce API 证据）即归本族。独立 softmax 归此；夹在 QK^T 与 PV 之间的 softmax 归 Attention 系；**通道注意力（SE/SK/Triplet/CBAM：global pool→FC→scale）有 softmax/sigmoid 但无 QK^T → 归本族**（纯 scale 则归 elementwise）。

## Step 1 — 通用前置

打开 [_common-prerequisites.md](_common-prerequisites.md) 过 C1-C5。Reduction 族常见命中：**C2**（标量归约循环 → P94，本族最高频收益）、**C5**（A2 Welford / A5 max-trick）。

## Step 2 — 族内阶段（按序推进）

| 阶段 | 判定条件（满足才做） | 命中用卡 | 跳过条件 |
|---|---|---|---|
| R1 算法模式 | 归约还是两遍/逐元素实现 | A2（Welford 单遍均值方差）；softmax 数值稳定 → A5 | 已是单遍在线算法 |
| R2 小 D 合并 | hidden ≤ 640 或行数多、D 小 | P3（小 D 多行合并成 tile）——注意与 P4 的协调（合并后 tile 数 < 核数时 P4 无意义） | D 大 |
| R3 多核与尾块 | 多核负载不均 / 尾块未处理 | P4（负载均衡）+ P11（尾块 GatherMask） | 数据天然均匀 |
| R4 归约指令化 | 归约由循环/多指令拼出 | P94（ReduceSum/ReduceMax/WholeReduceSum 替代标量循环）、P55（BlockReduceMax 替代 stride DataCopy）、P68（低延迟归约） | 已是单指令归约 |
| R5 向量链 | 归约后处理是标量链 | P95（Adds/Muls 向量链）、P99（BroadCast 批量替代逐行）、P84（Vector 效率模式）、P67（Counter 模式） | 已向量化 |
| R6 UB 复用 | 中间结果反复搬运 | P81（参数常驻）、P69（UB 融合链）、P110（归约 UB 累加器复用，依赖 P109 多轴结构） | UB 无压力 |
| R7 多轴归约 | reduction 轴多于一根 | P109（多轴合序 AR/ARA 分阶段 tiling）→ 接 P110 | 单轴归约 |
| R8 索引跟踪 | ArgMax/ArgSort 类带索引归约 | A7（索引边界安全）+ P111 不适用（那是 sort 族） | 纯值归约 |

## 组合约束

- 推荐组合见 [../strategy-compatibility.md](../strategy-compatibility.md) §3：Vector 效率链（P67+P68+P69+P84）、Reduction 多轴链（P42+P109+P110）
- P3×P4 条件性约束见 §1（合并后 tile 数 < 核数时 P4 无意义）
