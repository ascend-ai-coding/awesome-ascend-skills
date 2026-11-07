# FlashAttention 类算子 — 正交维度组合规则

> 本文档定义在**基础 FA** 核心算子(含 GQA / MHA / MLA 三形态)上正交叠加的两类 trait(量化 / 稀疏)与 feature flags 之间的交互规则。
> 多个 `subfamilies/` / `specialization/` 文件同时加载时(例如"MXFP8 GQA + Causal"),必须按本文档校核冲突。

---

## §1 正交维度交互总表

```
核心算子:基础 FA(含 GQA / MHA / MLA 三形态,沿 head / hidden-dim 变化)
  │
  ├── 正交叠加 trait(各自独立 on/off,叠在基础 FA 上):
  │     量化轴:非量化 / MXFP8 / INT8 / AntiQuant
  │     稀疏轴:非稀疏 / BlockSparse / SlidingWindow / ...
  │
  └── feature flags(可叠加在任意组合上):
        PSE / RoPE / Sink / PostQuant / Prefix / ChunkedPrefill / ...
```

**量化 × 稀疏 组合成熟度**(每格 = 基础 FA + 该量化 + 该稀疏):

```
                 ┌──── 稀疏轴 ────┐
                 │ 非稀疏 │  稀疏 │
       ┌─────────┼────────┼───────┤
量化轴 │ 非量化  │   ✓    │  📋   │
       │ MXFP8   │   ✓    │   ?   │
       │ INT8    │   ✓    │   ?   │
       │ AntiQ   │   ✓    │   ?   │
       └─────────┴────────┴───────┘
```

✓ = 已验证或已有扩展契约  📋 = 占位骨架(未落地)  ? = 未探索
基础 FA 的 GQA / MHA / MLA 三形态是核心算子内部沿 head / hidden-dim 的变化,不是叠加 trait,故不进本正交矩阵。

---

## §2 交互规则

### §2.1 基础子族 × 量化

| 交互点 | 规则 | 加载来源 |
|---|---|---|
| **Mmad 模板** | 基础子族决定 fm/filter 的角色(Q/K/P/V),量化路径决定 dtype 链(fm_dtype / filter_dtype / acc_dtype)| `base_design §8` + `quantization_design §1` |
| **P-scale 回传(MXFP8 特有)** | AIV 做完 SoftmaxFlashV2 后对 P 做量化 → P 数据(fp8)+ P-scale(e8m0) 写回 L1P 段 → AIC BMM2 读取时同时加载 P 数据 + P-scale | `quantization_design §6` |
| **I5 不变量校核** | V scale 必须沿 PV Mmad 的 K-mmad = S_k 量化,**不能**沿 D | `fundamentals §3.5` + `quantization_design §10 Q1` |
| **c1v1Loop 子循环** | MXFP8 量化路径在 BMM1 阶段引入 c1v1Loop(K 维按 256 切片),基础子族决定 task 维度不变 | `quantization_design §6` |
| **V2 末块量化链** | V2 末块的 O 在写回前需要做 fp32 → 量化 dtype cast + scale 生成,该两步链的 dtype 选择取决于输出 dtype | `quantization_design §8.3` |

**冲突解决优先级**:
1. **I5 不变量优先**:任何与 I5 冲突的设计必须拒绝(例如 V scale 沿 D)
2. **量化路径的 MX Mmad(MmadMx/MxMatmul)优先**:基础子族无法改变量化 Mmad 的 API 选择
3. **c1v1Loop 仅在量化路径激活**:非量化 GQA/MLA 不需要 c1v1Loop

### §2.2 基础子族 × 稀疏

| 交互点 | 规则 | 加载来源 |
|---|---|---|
| **Causal mask** | 与 GQA 的 Bq-major mEff 行排布交互:mask 模板只需一份 + 整体平移 curG 次 | `base_design §10.2 清单 5` |
| **SlidingWindow** | 影响 task 维度构造(sparse 模式可能跳过部分 KV chunk),但 task 维度基础结构仍按基础子族 | `sparse_design §3` |
| **BlockSparse** | sparse pattern 粒度决定是否能走 Cube(block-sparse 可走,token-level 需 gather) | `sparse_design §1` |

**冲突解决优先级**:
1. **基础子族优先**:sparse pattern 不能破坏 GQA 同 kvHead 同任务不变量(I2)
2. **Causal 与 SlidingWindow 互斥**:同一算子不能同时启用两种 mask

### §2.3 量化 × 稀疏

| 交互点 | 规则 | 加载来源 |
|---|---|---|
| **Sparse mask 与 scale 轴** | sparse pattern 跳过的 KV chunk 对应的 scale 段也必须跳过,否则 scale 与数据不对齐 | `quantization_design §4` + `sparse_design §4` |
| **Workspace 段叠加** | 量化 scale staging 段 + sparse 索引段同时存在,需要分别计算容量 | `quantization_design §4` + `sparse_design §4` |

**冲突解决优先级**:
1. **I5 不变量优先**:sparse 不能破坏 scale 轴对齐
2. **量化路径的 c1v1Loop 不受 sparse 影响**:c1v1Loop 是按 K 维切片,与 sparse pattern 正交

### §2.4 Feature flags × 其他维度

| feature flag | 与基础子族 | 与量化 | 与稀疏 | 加载来源 |
|---|---|---|---|---|
| **PSE**(partial sum extension)| 无特化 | 无特化 | 无特化 | `feature_flags §1` |
| **RoPE** | 需要在 Q/K 加载后做 RoPE 旋转,影响 L1→L0 加载路径 | 与量化 dtype 无冲突 | 与 sparse 无冲突 | `feature_flags §2` |
| **Sink** | 无特化 | 无特化 | 与 Causal/SlidingWindow 组合(保留首尾 token) | `feature_flags §3` |
| **PostQuant VF** | 输出 dtype 走量化路径,与基础子族的 V2 末块量化链耦合 | **强耦合**:PostQuant 本质是输出端的量化,必须先加载 `quantization_design §1` | 无特化 | `feature_flags §4` |
| **Prefix** | 无特化 | 无特化 | 无特化 | `feature_flags §5` |
| **ChunkedPrefill** | 影响 task 维度构造(Sq 维也切),但基础结构仍按基础子族 | 无特化 | 与 Causal 组合时需要处理 chunk 边界 mask | `feature_flags §6` |

**冲突解决优先级**:
1. **PostQuant 与量化路径强耦合**:PostQuant 的输出 dtype 复用量化路径的 V2 末块量化链
2. **Sink 与稀疏 mask 交互**:Sink 保留首尾 token,与 SlidingWindow/Causal 组合时需要显式处理 mask 边界
3. **RoPE 在量化路径下**:RoPE 旋转在 fp32 累加器上执行,不受输入量化 dtype 影响

---

## §3 冲突解决优先级总表

当多个 `subfamilies/` 文件同时加载时,按以下优先级解决冲突:

| 优先级 | 规则 |
|---|---|
| **P1** | I1-I5 不变量(`fundamentals §3`):任何变体组合都不能违反 |
| **P2** | 量化路径的 I5(scale 轴对齐):不可妥协 |
| **P3** | 基础子族的 task 维度构造:不能破坏 GQA I2 |
| **P4** | feature flags 正交性:不能破坏其他维度的不变量 |
| **P5** | 编译宏笛卡尔积:按 `design/execution.md §2`(N8 编译期特化)识别需要分离的 target |

---

## §4 加载协议

**Architect 加载流程**:

1. 读 `patterns.md` §2 确定基础子族 → 加载对应 `subfamilies/` 文件
2. 若用户需求包含量化 dtype → 额外加载 `quantization_design.md` + 按 §2.1 校核
3. 若用户需求包含 sparse pattern → 额外加载 `sparse_design.md` + 按 §2.2/§2.3 校核
4. 若用户需求包含 feature flags → 额外加载 `feature_flags.md` 对应章节 + 按 §2.4 校核
5. 若存在多维度叠加 → 按 §3 优先级解决冲突

**Developer 实现流程**:

1. 按 DESIGN.md 中 Architect 选定的变体组合,读取对应 `subfamilies/` 文件的 Self-Check 清单(§10)
2. 按本文档 §2 的交互表,逐项校核实现是否满足交互规则
3. 按本文档 §3 优先级表,在精度回归中优先验证 P1-P3 不变量

---

## §5 组合示例

| 场景 | 加载文件 | 关键交互点 |
|---|---|---|
| GQA + fp16 + Causal | `base_design` | Causal mask 与 GQA Bq-major 行排布 |
| GQA + MXFP8 + Causal | `base_design` + `quantization_design` + §2.1 | P-scale 回传 + Causal mask + I5 校核 |
| GQA + MXFP8 + SlidingWindow | `base_design` + `quantization_design` + `sparse_design` + §2.1 + §2.3 | P-scale + sparse 索引 + I5 |
| MLA + fp16 | `base_design §11` | latent absorption 链 |
| GQA + PostQuant | `base_design` + `feature_flags §4` + §2.4 | V2 末块量化链 |
| GQA + RoPE + Causal | `base_design` + `feature_flags §2` | RoPE 在 L1→L0 路径,mask 在 V1 |
