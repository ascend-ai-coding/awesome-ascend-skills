# Elementwise 优化工作流

> 适用：Sin / Cos / Abs / Exp / Add / Mul 等输入输出 shape 相同、逐元素独立计算的算子（含 foreach 类）。

## Step 0 — 族判定确认

排除法兜底族：输入输出 shape 相同、逐元素独立、无跨元素依赖，且 [README.md](README.md) 判定表前 8 行均不命中。

## Step 1 — 通用前置

打开 [_common-prerequisites.md](_common-prerequisites.md) 过 C1-C5。Elementwise 族常见命中：**C2**（标量逐元素循环 → P95/P97/P100，本族最高频收益）、**C4**（P1/P7 是基本盘）。

## Step 2 — 族内阶段（按序推进）

| 阶段 | 判定条件（满足才做） | 命中用卡 | 跳过条件 |
|---|---|---|---|
| E1 tile 大小 | tile 硬编码或未向量化对齐 | P112（向量化 tile 大小选择 + 内存合并 + 多核分发） | tile 已对齐且多核吃满 |
| E2 向量效率 | 标量/低效向量链 | P95（Adds/Muls 链）、P67（Counter 模式）、P84（Vector 效率模式）——P67 与 P84 重叠时以 P84 为准 | 已是高效向量链 |
| E3 子 tensor 批量 | 存在显式子块拷贝 | P97（零拷贝子 tensor 切片）、P100（GatherMask 替代 SetValue/GetValue）、P101（host 侧列拆分改 GatherMask） | 无子块操作 |
| E4 UB 复用 | 高阶 API 与计算阶段争抢 TBuf | P89（UB API 临时 buffer 分时复用）、P85（buffer 分区复用） | UB 无压力 |
| E5 融合后处理 | 逐元素链后有 broadcast/mask | P12（广播掩码）、P113（广播对齐 tiling） | 纯逐元素 |

## 组合约束

- P67×P84 条件性约束（P84 含 Counter 子模式，同时选中以 P84 为准）见 [../strategy-compatibility.md](../strategy-compatibility.md) §1
- 推荐组合：Elementwise 向量链（P67+P84+P112）见 §3
