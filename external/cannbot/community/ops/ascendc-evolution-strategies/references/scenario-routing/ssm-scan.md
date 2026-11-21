# SSM/Scan 优化工作流（GDN-KDA / GLA）

> 适用：无 QK^T、无 softmax 的线性递归 / chunk-scan 算子（Gate Delta Rule、Gated Linear Attention、Mamba 等）。
> 分支数学形态（见 README 判定表第 2 行）：
> · **GDN/KDA** — delta rule：`S ← β⊙(k⊗v − corr) + S`（含 corr 修正项）
> · **GLA** — 门控衰减：`S = α⊙S + k⊗v`（含衰减门 α，无修正项）
> 主链由 **P132-P136** 支撑（本插件沉淀，参考 chunk_gated_delta_rule 实现）；仍无对症的方向（GLA cumprod chunked scan、反向 adjoint-scan）按 X 系列沉淀。

## Step 0 — 族判定确认

命中 [README.md](README.md) 判定表第 2 行（沿序列维递归/chunk-scan，**无 QK^T、无 softmax**）即归本族。
判据需读 `model.py` 数学结构（α/β/corr 项），不靠算子名；名字含 Attention/注意力但无 QK^T 即非 FA 行。

## Step 1 — 通用前置

打开 [_common-prerequisites.md](_common-prerequisites.md) 过 C1-C5。SSM 常见命中：**C2**（标量递归循环 → P94/P95）、**C5**（数值稳定 → A2/A5）。

## Step 2 — 分支路由（按序）

**小类映射**（源自 [attention_family.md](../../../ascendc-evolution-knowledge/references/a3/algorithm_insights/attention_family.md) 分类，小类 → 分支 + 推荐策略）：

| 小类（计算模式） | 分支 | 推荐策略 |
|---|---|---|
| **KDA**（CP7，逐通道门控 + beta 误差修正） | delta-rule | P132 三阶段 → P133 块内 CV → **P135 求逆分治** → P134 任务划分 → P136 bf16 |
| **GDN**（CP7，门控 DeltaNet，同 delta rule 族） | delta-rule | 同上（对标 fla/gdn2 + OlmoHybrid） |
| **GLA**（CP7，标量衰减，每 head 一个 gamma） | gla | P132 三阶段（**cumprod 替代求逆**）→ P133 CV → P134 任务 → P136 bf16（跳过 P135） |
| **Mamba2**（CP7，SSD 状态空间 + D 跳连） | gla | 同上（标量衰减；注意 dt 预处理链 softplus→clamp→exp） |
| **Linear**（CP5+CP6，低秩/Nyström/核映射，**无 chunk 状态递推**） | 边缘 | 非标准 chunk-scan：LinearAttention → gla 分支；Nyström/Linformer（降维后仍 softmax）→ 标准 Softmax 路径 |

> 区分要点：**KDA/GDN** 逐通道向量门控 + 求逆（P135）；**GLA/Mamba2** 标量衰减 + cumprod（无求逆，跳过 P135）；**Linear** 无状态递推，归边缘。

## Step 3 — chunk-scan 共享基础链（以 P132-P136 为主线，按序推进）

| 阶段 | 目标 | 策略（按优先级） | 跳过条件 |
|---|---|---|---|
| S0 三阶段分解 | 逐 token 串行 → 块内预计算/块间状态扫描/输出投影 | **P132**（chunk-scan 三阶段分解，C=64） | 已是三阶段 |
| S1 块内 CV 流水 | Stage1 多级 matmul+向量预处理 AIC/AIV 重叠 | **P133**（块内预计算 CV 流水，6-flag 握手） | 无 SyncAll / 已握手 |
| S2 任务划分 | 核数吃满（group 内并行 + group 间串行） | **P134**（chunk-group 任务划分） | 核已打满 |
| S3 求逆分治 | (I-A)^{-1} 求逆并行化（**仅 GDN-KDA**） | **P135**（对角前代 + 非对角 cube MM） | GLA 分支跳过（用 cumprod） |
| S4 数值/UB | 中间量 bf16 + 状态常驻 | **P136**（bf16 量化 + state UB 常驻 + head grouping） | UB 无压力 |

> 近邻参考（非对症，勿当主卡）：P109/P110（reduction 多轴）、X15（递归投影外提）、X16（H-slice 多核切分，注意门控列禁忌）。

## 正交 trait（基础链之上叠加）

| trait | 判定 | 追加 |
|---|---|---|
| phase=bwd | 反向传播算子（KdaBwd / Mamba2Bwd / ChunkKdaBwdIntra / Gdn2Bwd） | **切换阶段组**为反向链：adjoint scan（反向模式穿过递归/chunk + checkpointing 重算），**覆盖缺口**（无对症卡，需沉淀） |
| 量化 | int8/fp8 权重/输入 | 转 quantization.md 支路（P48/P49/P70），trait 不换族 |

## 覆盖缺口说明

- GLA 的 cumprod 衰减 chunked scan：无对症卡，走通用前置 + 瓶颈驱动。
- SSM 反向（adjoint scan / reverse-mode）：无对症卡，进化后按 X 系列流程沉淀（双轨 compat+OKF + INDEX 重建）。
- 命中缺口时把本轮实测手法沉淀，别机械套 P109/P110（那是 reduction 基础，非 chunk-scan 对症）。

## 组合约束

- P132 必须先于 P133/P134/P135（三阶段是前两者的前提）。
- P135 仅 GDN-KDA；GLA 跳过，误用会找不存在的求逆。
- P133 的 flag 事件号须避开 stage2/3 号段。
- P109/P110、X15/X16 与 P132-P136 正交可组合（X15 外提投影、X16 H-slice）；X16 禁忌：门控列跨步取 `{j,H+j,2H+j,3H+j}`，禁连续 `[4hStart,4hEnd)` 切。
- Refine 阶段 profiling 证据优先于本工作流顺序（strategy-index.md "按瓶颈类型查表"）。
