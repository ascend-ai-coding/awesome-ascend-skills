# FlashAttention 类算子 Tiling 设计 — 场景路由

> **本目录采用四层架构组织 FA 类算子的 Tiling 设计方法论**:
> - **L1 路由**(本文件):选型入口、§C 设计节点注册表、加载指南、`subfamilies/` 章节模板
> - **L2 地基** — `foundation/`:通用地基(fundamentals / roofline / extension_points / composition)
> - **L3 设计流** — `design/`:节点化设计流程(_governance / shape / resources / execution),节点 G→N12
> - **L4 变体** — `subfamilies/` + `specialization/`:变体特定设计(base / quantization;稀疏 / feature flags);`implementation_ref.md` 为实现叶子
>
> **使用方式**:先读本文确定场景 → 加载 `foundation/` 地基 → 沿 `design/` 节点 G→N12 推进 → 按需加载 `subfamilies/` / `specialization/` 变体文件。设计节点名→文件位置由 §C 注册表唯一翻译。

---

## §1 文件总览

### Layer 1: 选型入口

| 文件 | 内容 |
|------|------|
| `patterns.md`(本文件)| 变体选型表、正交维度、加载指南、`subfamilies/` 标准章节模板 |

### L2 地基: 通用地基 — `foundation/`(所有 FA 变体共享)

| 文件 | 内容 |
|------|------|
| [`fundamentals.md`](./foundation/fundamentals.md)| FA 类定位、FA-2 数学模型、不变量 I1-I5、分析阶段产出、抽象名与公开 API 对照 |
| [`roofline.md`](./foundation/roofline.md)| 基本块与三通道 Roofline(Cube / HBM / 跨核片上)、mEff/AI 推导、可行域上下界、prefill/decode/causal 场景、量化 AI 修正 |
| [`extension_points.md`](./foundation/extension_points.md)| 5 扩展点框架(ConstInfo/RunInfo / Cube / Vector / Workspace / Service)、dtype 路径、变体间禁止事项 |
| [`composition.md`](./foundation/composition.md)| 正交维度交互规则(基础子族 × 量化 × 稀疏 × feature flags)|

### L3 设计流: 节点化设计流程 — `design/`(节点 G→N12,顺序即依赖链)

| 文件 | 承载节点 | 内容 |
|------|---------|------|
| [`_governance.md`](./design/_governance.md)| G | 设计阶段产出清单 + 设计即最优门禁 + 设计依赖链 + 结构决策定位 / 验收分工 |
| [`shape.md`](./design/shape.md)| N1/N2/N3/N5/N6 | 宏观形态(kernel/UB/cube/流水深度+槽语义分类)+ 并行策略 + 基本块 Roofline 入口 + 平台依赖 + 多核切分/负载均衡 |
| [`resources.md`](./design/resources.md)| N7/N9 | 内存划分(UB/L1/L0/GM)+ Host Tiling 汇总 |
| [`execution.md`](./design/execution.md)| N4/N8/N10/N11/N12 | Softmax 算法模式 + 编译期特化 + 流水编排/跨核同步 + 开发起点 + 优化策略产出 |

### L4 变体: 变体特定设计 — `subfamilies/` + `specialization/`(按需加载)

| 文件 | 适用变体 | 内容 |
|------|---------|------|
| [`subfamilies/base_design.md`](./subfamilies/base_design.md)| FA / GQA / MHA / **MLA**(默认;MLA 为 base FA 的 kvHeadNum=1 + latent 变体)| GQA 契约、mEff / curG、BSND 写回、fp16 / bf16 dtype 路径、§11 MLA 变体分支(latent absorption / int32 段)|
| [`subfamilies/quantization_design.md`](./subfamilies/quantization_design.md)| MXFP8 / INT8 / AntiQuant | I5 落地、scale 轴对齐、P-scale 回传、量化子循环 |
| [`specialization/sparse_design.md`](./specialization/sparse_design.md)| BlockSparse / SlidingWindow(📋 占位)| 稀疏 pattern 描述、KV chunk 跳过、索引预计算 —— 当前为占位骨架,落地首个稀疏 FA 算子时细化 |
| [`specialization/feature_flags.md`](./specialization/feature_flags.md)| PSE / RoPE / Sink / PostQuant / Prefix / ChunkedPrefill | 正交特性 flag 契约(可叠加在任意变体之上)|
| [`implementation_ref.md`](./implementation_ref.md)| 实现叶子(全变体)| 跨核同步代码骨架、PIPE 映射、平台 mode/offset、Fixpipe 参数 |

---

## §1.5 设计节点注册表(唯一 节点名 → 文件 翻译桥)

> 本表是**节点名 → 落地文件#锚点**的唯一权威映射。`subfamilies/` / `specialization/` 变体模板、§4 加载协议、§5 章节模板一律引用**节点名**(左二列),不直呼文件 §号。`design/` 文件日后再拆分 / 改名,只改本表一处。
> **顺序即设计依赖链**(见 [`design/_governance.md`](./design/_governance.md) 依赖链),节点必须按 G→N12 推进。

| # | 节点名 | 依赖前置 | 落地文件#锚点 |
|---|---|---|---|
| G | 设计治理(产出清单 + 设计即最优门禁 + 依赖链 + 结构决策定位)| — | [`design/_governance.md`](./design/_governance.md) |
| N0 | 选型与分析确认 | — | [`foundation/fundamentals.md`](./foundation/fundamentals.md) §4 + 本文 §2 |
| N1 | 宏观形态(kernel/UB/cube/流水深度 + 槽语义分类)| N0 | [`design/shape.md`](./design/shape.md) §1 |
| N2 | 并行策略(默认 / split-KV,含启用判据)| N1 | [`design/shape.md`](./design/shape.md) §2 |
| N3 | 基本块与 Roofline | N2 | [`design/shape.md`](./design/shape.md) §3 → [`foundation/roofline.md`](./foundation/roofline.md) |
| N4 | Softmax 算法模式(V1/V2)| N3 | [`design/execution.md`](./design/execution.md) §1 |
| N5 | 平台依赖识别 | N3 | [`design/shape.md`](./design/shape.md) §4 |
| N6 | 多核切分 + 负载均衡 | N3 | [`design/shape.md`](./design/shape.md) §5 |
| N7 | 内存划分(UB/L1/L0/GM)| N3, N6 | [`design/resources.md`](./design/resources.md) §1 |
| N8 | 编译期特化(编译宏 / constexpr / 运行时 if 三分)| N7 | [`design/execution.md`](./design/execution.md) §2 |
| N9 | Host Tiling 汇总 | N1-N8 | [`design/resources.md`](./design/resources.md) §2 |
| N10 | 流水编排与跨核同步 | N1, N4, N6, N7 | [`design/execution.md`](./design/execution.md) §3 |
| N11 | 平台基线 + 开发起点 | 全部 | [`design/execution.md`](./design/execution.md) §4 |
| N12 | 优化策略产出(DESIGN.md 模板)| 全部 | [`design/execution.md`](./design/execution.md) §5 |

---

## §2 变体选型决策表

> 选型分**两步**:先定**基础 FA 形态**(GQA / MHA / MLA,沿 head / hidden-dim 变化的三选一),再按需**正交叠加** trait(量化 / 稀疏,各自独立可选)。**GQA / MHA / MLA 不是独立子族**,而是基础 FA 的三种形态,共享同一 service 骨架、只路由到 `base_design.md` 的不同章节;**量化与稀疏都是叠加在基础 FA 之上的正交 trait**,不与基础 FA 平级。此结构与 §3 组合空间一致。

**第一步 — 基础 FA 形态(三选一,必选)**:

| 形态 | 数学差异 | 选型触发条件 | 落地章节 |
|---|---|---|---|
| **GQA**(默认)| `Hq = G × Hkv`,G ≥ 1 个 query head 共享 KV head | 默认路径;`Sq ≥ 1 + Hq % Hkv == 0` | `subfamilies/base_design.md` §1-§10 |
| **MHA**(GQA 退化)| `G = 1`(`Hq = Hkv`)| GQA 的退化特例(`gMaxPerTask=1`)| `subfamilies/base_design.md` §10.3 |
| **MLA**(latent 变体)| KV 经 latent compression,`kvHeadNum = 1` | 数学公式含 latent absorption | `subfamilies/base_design.md` §11 |

**第二步 — 正交叠加 trait(各自独立可选,叠在基础 FA 之上)**:

| 正交 trait | 触发条件 | 落地契约 |
|---|---|---|
| **量化** | `dtype ∈ {MXFP8 / INT8 / AntiQuant}` | `subfamilies/quantization_design.md` |
| **稀疏**(📋 占位)| KV 维度有 sparse pattern(非全连接)| `specialization/sparse_design.md`(占位骨架)|

> **目录位置说明**:量化 trait 位于 `subfamilies/`(因量化改变核心 Cube 计算路径,如 MX Mmad / c1v1Loop),稀疏 trait 位于 `specialization/`(因稀疏不改核心计算路径,仅影响 tile 跳过)。两者均为正交 trait,目录差异仅反映对核心计算的修改深度。

**选型规则**:
1. 数学公式与默认 FA-2 一致 + `Hq % Hkv == 0` → **GQA 形态**(`G=1` 即 MHA 退化,同骨架)
2. 数学公式含 latent absorption → **MLA 形态**(仍属基础 FA:kvHeadNum=1 + latent 维,走 `base_design.md` §11,不单开子族)
3. **量化 / 稀疏是正交 trait,不是子族**:定完基础 FA 形态后按需叠加;"量化 GQA""稀疏 MLA"等都是合法组合
4. **未验证组合先做可行性验证**:见 §3 组合矩阵中标 `?` / `📋` 的组合(如量化 × 稀疏、非量化 × 稀疏),进入设计前须先确认扩展契约可行,不可直接假定成立

> **注**:FlashDecoding(split-KV reduce)是**并行实现技术**,不是子族 / trait。可叠加在任意组合之上,详见节点 N2([`design/shape.md` §2](./design/shape.md))。

---

## §3 正交维度与组合空间

以**基础 FA** 为唯一核心算子,在其上**正交叠加**量化、稀疏两类 trait,再叠加 feature flags:

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
基础 FA 的 GQA / MHA / MLA 三形态是核心算子内部沿 head / hidden-dim 的变化,不是叠加 trait,故不进本正交矩阵;形态内部的量化组合成熟度差异(如 MLA 量化尚未落地)见 [`composition.md`](./foundation/composition.md) 与 `base_design.md §11`。

**组合规则**:详见 [`composition.md`](./foundation/composition.md)。

---

## §4 设计节点 × 变体文件 加载协议

Architect 在设计阶段按以下映射表推进(左列节点名见 §1.5 注册表;右列变体 §号指向变体文件内部,不受 `design/` 重组影响):

| 设计节点 | `foundation/` + `design/` 通用问题 | 变体文件加载条件 |
|---|---|---|
| **N0 选型与分析确认** | `foundation/fundamentals.md` §4 分析产出 | 本文 §2 选型表 → 加载对应变体文件 |
| **变体确认后** | `foundation/extension_points.md`:该变体在 5 扩展点上的特化范围 | — |
| **N1 宏观形态** | kernel 类型 / UB 模式 / Cube 分块 / 流水级数(`design/shape.md` §1)| `base_design §1` + `quant §1`(若量化)+ `sparse §1`(若稀疏) |
| **N2 并行策略** | task 级 vs split-KV reduce(`design/shape.md` §2)| `base_design §3`(MLA 变体见 `base_design §11.3`;MLA 的 split-KV 未验证) |
| **N3 基本块与 Roofline** | `foundation/roofline.md`:三通道 Roofline + mEff/Bk 可行域 | `base_design §2`(mEff / curG)+ `quant §2`(若量化, AI 修正) |
| **N4 Softmax 模式** | V1 / V2 算法路径(`design/execution.md` §1)| `base_design §8`(fp16 / bf16 差异)+ `quant §6.2`(P-scale 回传) |
| **N6 多核切分** | task 维度构造 + 负载均衡(`design/shape.md` §5)| `base_design §3`(GQA task)+ `base_design §11.3`(MLA 变体 latent task) |
| **N7 内存划分** | UB / L1 / L0 / GM(`design/resources.md` §1)| `base_design §4` + `quant §4`(scale staging)+ `sparse §4`(索引段) |
| **N8 编译期特化** | 条件性功能归类(`design/execution.md` §2)| `base_design §5`(dtype / causal)+ `feature §1-§5`(PSE / RoPE 等) |
| **N9 Host Tiling** | 输入输出字段(`design/resources.md` §2)| `base_design §7`(ConstInfo / RunInfo)+ `quant §7`(scale 字段) |
| **N10 流水编排与跨核同步** | 节拍 / 同步 / slot(`design/execution.md` §3);跨核同步实现骨架见 `implementation_ref.md` | `base_design §6` + `quant §6`(c1v1Loop) |
| **N11 开发起点** | 基线验证 + 模块化构建(`design/execution.md` §4)| — |
| **N12 优化策略产出** | 优化产出模板 + 变体优化索引(`design/execution.md` §5)| `base/quant/sparse §9`(变体特定优化;MLA 变体见 `base_design §11.9`) |

**Feature flags 加载**:若用户需求包含 PSE / RoPE / Sink 等 → 额外加载 `specialization/feature_flags.md` 对应章节 + `foundation/composition.md` feature flags 交互部分。

---

## §5 `subfamilies/` 标准章节模板

> 5 扩展点框架见 [`extension_points.md`](./foundation/extension_points.md)。`subfamilies/` / `specialization/` 文件按以下模板组织变体特化内容。

所有变体文件**必须**按以下 10 节结构组织,每节对齐 §1.5 注册表的一个设计节点(引用**节点名**,不直呼 `design/` 文件 §号):

```markdown
# [变体名] 扩展契约

## §1 宏观决策特化
(对齐 N1 宏观形态:kernel/UB/cube/pipe 决策的变体影响)

## §2 Roofline 特化
(对齐 N3 基本块与 Roofline:AI/mEff 修正)

## §3 多核特化
(对齐 N6 多核切分:task 维度构造影响)

## §4 内存特化
(对齐 N7 内存划分:buffer 扩展段)

## §5 编译宏特化
(对齐 N8 编译期特化:条件性功能补充)

## §6 流水线特化
(对齐 N10 流水编排与跨核同步:stage / slot 影响)

## §7 Host Tiling 特化
(对齐 N9 Host Tiling 汇总:字段扩展)

## §8 dtype 路径
(变体特定的 dtype 约束与禁止反模式)

## §9 优化策略特化
(对齐 N12 优化策略产出:变体特定优化入口)

## §10 Self-Check 清单
(进入审查前的必跑清单, 继承通用 + 变体特有)
```

**纪律**:
- 变体文件**不重复** `foundation/` + `design/` 已有的通用内容,只写"该变体在此基础上有什么不同"
- 若某节对该变体无影响,显式写"本变体无特化,按对应节点(见 §1.5)的通用规则"
- §10 Self-Check 必须包含"继承 `design/` 各节点通用清单 + 本变体新增清单"

---

## §6 架构约束

1. **一个算子只属一个基础子族**:禁止"主要 GQA 但某路径走 MLA latent absorption"
2. **量化 / 稀疏与基础子族正交**:可叠加,但必须在 `composition.md` 中校核交互
3. **常量语义分离**:跨核握手槽、L1 rotation 槽、V2 自读自写槽**必须用不同常量定义**,详见 [`extension_points.md` §4.2](./foundation/extension_points.md)
4. **变体文件不修改地基/设计流**:新变体只新增 `subfamilies/` / `specialization/` 文件,不动 `foundation/` + `design/` 方法论
5. **新变体落地后回写**:若第 2/3 个同类型算子出现稳定共性 → 升级为 `subfamilies/` 完整契约;若出现跨变体共性 → 升级为 `foundation/` + `design/` 方法论

---

## §7 阶段间衔接

| 阶段 | 进入位置 |
|---|---|
| 分析(N0)| [`fundamentals.md`](./foundation/fundamentals.md):数学模型 + 不变量 + 分析产出 |
| 选型(N0)| 本文件 §2:选型决策表 |
| 扩展点 | [`extension_points.md`](./foundation/extension_points.md):5 扩展点 + dtype 路径 + 变体禁止事项 |
| 设计治理(G)| [`design/_governance.md`](./design/_governance.md):产出清单 + 设计即最优门禁 + 依赖链 |
| 设计流(N1-N12)| [`design/shape.md`](./design/shape.md) / [`resources.md`](./design/resources.md) / [`execution.md`](./design/execution.md):节点化设计流程 |
| 基本块 Roofline(N3)| [`roofline.md`](./foundation/roofline.md):三通道 Roofline + mEff/Bk 选取 |
| 组合规则 | [`composition.md`](./foundation/composition.md):正交维度交互 |
| 实现层参考 | [`implementation_ref.md`](./implementation_ref.md):跨核同步 / Fixpipe / 平台参数 |
| 通用调试 | `/ascendc-precision-debug` |
| 运行时错误 | `/ascendc-runtime-debug` |
| 精度门禁 | `/ops-precision-standard` |
| 平台差异 | `/npu-arch` |
| Ascend C API | `/ascendc-api-best-practices` |
| 编码规范 / 工程配置 | `ops-direct-invoke` plugin `workflows/development-guide.md` |
