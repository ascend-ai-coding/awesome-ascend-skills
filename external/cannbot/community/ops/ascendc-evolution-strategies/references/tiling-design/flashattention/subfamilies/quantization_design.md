# FlashAttention 量化 FA 扩展契约

> 本文档定义量化 trait(MXFP8 / INT8 / AntiQuant)的扩展契约。量化与基础 FA 形态(GQA / MHA / MLA)及稀疏 trait 正交,可叠加。
> **通用设计流程** → [`design/`](../design/_governance.md)(节点 G→N12);**扩展点框架** → [`extension_points.md`](../foundation/extension_points.md);**组合规则** → [`composition.md`](../foundation/composition.md) §2.1。

---

## §1 宏观决策特化

### §1.1 Kernel 类型

量化 FA 与基础形态一致(典型 `__mix__(1, 2)`)。量化路径不改变 kernel 类型选择。

### §1.2 UB 模式

量化 FA 默认走 streaming UB 模式,与基础形态一致。

### §1.3 Cube 分块层次(量化特化)

量化路径的 Cube 分块层次有额外约束:
- **MX Mmad**:MXFP8 路径使用 `MmadMx`(ISASI)或高阶 `MxMatmul`,fp8 类型为 `fp8_e4m3fn_t`(kernel 侧 C++ 类型;高阶 Tiling 侧亦作 `float8_e4m3_t`,对应 dtype 枚举 `DT_FLOAT8_E4M3FN`),与标准 `Mmad` 不同
- **Load2DMX**:加载数据 + scale 的专用 API(参数结构体 `LoadData2DMxParams`),具体字段单位与对齐约束见 `/ascendc-api-best-practices`
- **K_BASE 约束**:K_BASE 必须与所选 LoadData API 的 yStep 单位整除

### §1.4 流水级数

量化 FA 的流水级数与基础形态一致,但 BMM1 阶段可能引入 c1v1Loop 子循环(见 §6.1),影响流水编排细节。

---

## §2 Roofline 特化

### §2.1 AI 修正(量化)

量化 dtype 下 K/V 的 HBM 流量按量化字节宽计,主 Roofline 的 `AI_HBM = 2·mEff / sizeof(quant_T)`(见 [`roofline.md` §5](../foundation/roofline.md)):
```
Bytes_HBM ≈ 2 × Sk × D × sizeof(quant_T)
```

**fp16(2B)→ MXFP8(1B)**:sizeof 减半 → **AI_HBM 翻倍(≈ +100%)**。

> ⚠️ 早期文档曾给出"AI 提升约 48%"——那是把片上握手混进 AI 分母的旧混合模型 artifact。clean HBM 口径下 dtype 减半即 AI 翻倍。详见 [`roofline.md` §1/§5](../foundation/roofline.md)。

### §2.2 跨核片上吞吐修正

量化后 S/P/PV 跨核握手段若也走量化 dtype(例如 P 在 V1 末块量化为 fp8),片上握手 movement 减少,缓解 cross-core-bound(见 [`roofline.md` §2.3](../foundation/roofline.md))。但 P-scale(e8m0)需要额外的 L1 staging,须计入片上预算。

---

## §3 多核特化

量化 FA 的任务维度构造与基础形态一致。量化路径不改变 task 维度选择。

**c1v1Loop 对 task 的影响**:c1v1Loop 是在 BMM1 阶段内部的子循环,按 K 维切片(典型 256),不改变 task 维度构造。

---

## §4 内存特化

### §4.1 Workspace scale staging 段

量化 FA 可能新增 **scale staging 段**:
- Q / K / V 的 scale 需要从 GM 加载到 L1 供 MX Mmad(MmadMx / MxMatmul)消费
- P 在 V1 末块量化后的 P-scale(e8m0)需要从 AIV 写回 L1P 段供 AIC BMM2 读取

容量公式:**按运行时 Sk 动态分配**,不静态预留 Sk_max。

### §4.2 L1 Scale 布局

Scale 在 L1 的布局取决于量化 dtype:
- MXFP8:scale 是 E8M0,块大小 32 元素,沿 K-mmad 轴
- 数据载体 vs scale 载体的 multi-head head offset 公式**必须独立推导**(公式系数可能不同)

### §4.3 BMM2 Buffer 策略

量化 FA 的 BMM2 跨核传递因数据量减半,可从双缓冲改为**单缓冲**:
```
非量化:BMM2 UB 中间段用双缓冲(数据量大,隐藏搬运延迟);落 GM 时用三缓冲
MXFP8:  BMM2 数据量减半 → UB 中间段可退回单缓冲,省一半 buffer
```
(缓冲深度是 buffer 规划决策,不绑定任何实现的封装类名)

---

## §5 编译宏特化

### §5.1 量化 dtype 编译宏

| 维度 | 是否分离 | 理由 |
|------|---------|------|
| 量化 dtype 类型(MXFP8/INT8/AntiQ)| ✅ 必须 | 影响 typedef 与 Matmul 模板参数 |
| 量化 vs 非量化 | ✅ 必须 | 影响 c1v1Loop / scale staging TBuf |

### §5.2 constexpr 级联链

量化 FA 的 constexpr 链在非量化基础上扩展:
```
tilingKey → ... → isMxFp8 → c1v1LoopEnabled → pScaleEnabled
```

---

## §6 流水线特化

### §6.1 c1v1Loop 子循环

> `c1v1Loop` / `L1P` 是本节为描述方便使用的**概念标签**(K 维子循环 / 存放 P 的 L1 段),非公开 API 名;实际命名由实现自定。

MXFP8 量化路径在 BMM1 阶段引入 **K 维子循环**(此处记为 c1v1Loop):
- 按 K 维(典型 S2 方向)切片,每段 ≤ 256 列(切片粒度以目标量化 Mmad 的 K 维限制为准)
- 子循环次数:`CeilDiv(actS2, 切片粒度)`
- 子循环内 L1 双缓冲:当前子循环计算与下一子循环数据加载重叠(用双 slot 轮转的加载 API 实现,具体接口以目标版本为准)

### §6.2 P-scale 回传机制

MXFP8 的 V1 末块对 P 做量化后:
1. AIV 将 P 数据(fp8)写回 L1P 段(原 S 段复用)
2. AIV 将 P-scale(e8m0)写回 L1P 段的 scale 区
3. AIC BMM2 读取 P 数据 + P-scale,走 MX Mmad(MmadMx / MxMatmul)

### §6.3 Slot 分类

量化 FA 的 slot 分类与基础形态一致,但 BMM2 buffer 可能从双缓冲改为单缓冲(见 §4.3)。

---

## §7 Host Tiling 特化

### §7.1 ConstInfo 字段扩展

量化 trait 特定的 ConstInfo 字段:
- 各 scale 张量(Q / K / V / P)的量化轴与 layout
- scale dtype(典型 MXFP8 用 E8M0)
- scale 相关 Tiling 派生字段

### §7.2 RunInfo 字段扩展

量化 FA 的 RunInfo 与基础形态一致。c1v1Loop 的子循环索引在 kernel 内部维护,不透传到 RunInfo。

### §7.3 TilingData 字段扩展

量化 FA 的 TilingData 在基础形态基础上扩展:
- c1v1Loop 切片大小
- P-scale staging 段尺寸

---

## §8 dtype 路径

### §8.1 MXFP8 路径

- **MX Mmad**:`MmadMx`(ISASI)/ `MxMatmul`(高阶),fp8 类型 `fp8_e4m3fn_t`(枚举 `DT_FLOAT8_E4M3FN`)
- **数据 dtype**:E4M3(fp8)
- **Scale dtype**:E8M0(块大小 32 元素)
- **累加器**:fp32

### §8.2 I5 不变量落地校核

MX 类 scale **必须沿被消费 matmul 的 K-mmad 轴量化**:

| 张量 | 被消费 matmul | K-mmad 轴 | scale 量化轴 |
|------|------------|---------|-----------|
| Q | Q·K^T | D | D |
| K | Q·K^T | D | D |
| P | P·V | S_k | S_k |
| V | P·V | S_k | **S_k**(不是 BSND innermost 的 D)|

**反直觉陷阱**:V 在 BSND 布局下 D 是 innermost,但 V scale 必须沿 S_k。详见 [`fundamentals.md` §3.5](../foundation/fundamentals.md)。

### §8.3 V2 末块量化链

V2 末块的 O 在写回前需要做 fp32 → 量化 dtype cast + scale 生成:
- **步骤 1**:ReduceMax → amax(每行最大值)
- **步骤 2**:scale = amax / quant_dtype_max
- **步骤 3**:Cast<quant_dtype, float>(O / scale)

**禁止反模式**:cast 数值公式必须保证 amax / scale 不超量化 dtype max(典型踩坑:cast 溢出 → RINT 产出 NaN)。

### §8.4 量化 dtype 禁止反模式汇总

- scale 张量内存连续轴未对齐 K-mmad → Mmad 产出 NaN/inf
- cast 数值公式溢出 → RINT 产出 NaN
- LoadData 字段单位与 dtype 不匹配(典型:照搬 fp16 路径的 CUBE_BLOCK 单位)
- 数据载体 vs scale 载体的 head offset 公式混用同一系数

---

## §9 优化策略特化

### §9.1 子循环分解

量化 Mmad 的 K 维限制需要子循环:
- 子循环粒度:典型 256(MXFP8;以目标量化 Mmad 的 K 维限制为准)
- 子循环内 L1 双缓冲:双 slot 轮转,当前子循环计算与下一子循环加载重叠

### §9.2 Buffer 缩减

量化类型元素宽度小,buffer 可缩减:
- BMM1 结果 buffer:fp16 2B → MXFP8 1B,减半
- BMM2 跨核传递:从双缓冲改为单缓冲

### §9.3 L1 mask 位开销

P 矩阵的 1-bit mask 元数据需要计入 L1 预算(量化场景下 L1 更紧张)。

### §9.4 Stride 访问

量化 tensor 可能非连续布局,stride 信息需要通过 TilingData 传递。

---

## §10 Self-Check 清单

### §10.1 继承通用清单

- [ ] 清单 1(streaming UB 容量)
- [ ] 清单 2(slot 语义)
- [ ] 清单 3(cross-core sync 时序)([`implementation_ref.md` §5](../implementation_ref.md))
- [ ] 清单 9(长 Sk workspace 槽轮转)
- [ ] 清单 9b(workspace 外层乘子 = min(totalTasks, usedCoreNum))
- [ ] 继承基础形态的所有清单(GQA:4-8;MLA:M1-M6;稀疏:S1-S5)

### §10.2 量化 FA 特有清单

- [ ] **清单 Q1(I5 落地校核)**:所有 scale 张量的内存连续轴是否对齐其消费 matmul 的 K-mmad 方向?V scale 沿 S_k 还是沿 D 已显式声明?
- [ ] **清单 Q2(数据载体 vs scale 载体公式分离)**:数据载体的 L1 head offset 公式与 scale 载体的 L1 head offset 公式是否分别独立推导?两者公式系数是否可能不同?
- [ ] **清单 Q3(V2 末块量化路径对齐链)**:V2 末块量化路径中各 API(DataCopy 类 / Reduce 类 / Cast 类)的对齐要求是否被显式校核?
- [ ] **清单 Q4(cast 数值不溢出)**:V2 末块 cast 数值公式是否能保证 amax / scale 不会超量化 dtype max?
- [ ] **清单 Q5(LoadData 字段单位匹配 dtype)**:Load2DMX 系列 API(`LoadData2DMxParams`)的 kStep / kStartPosition 等字段单位是否与所选量化 dtype 匹配?

### §10.3 蒸馏纪律

本骨架基于首个量化 FA 算子(GQA × mxfp8 推理,A5 平台)落地形成。具体阈值、API 数值参数、cast 实现细节 → 各算子自己的 DESIGN.md。等第 2、第 3 个量化 FA 落地后,如果在 5 扩展点上出现稳定共性,可考虑把骨架升级为完整契约。
